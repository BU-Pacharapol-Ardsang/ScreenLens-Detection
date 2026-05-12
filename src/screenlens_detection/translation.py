from __future__ import annotations

import json
import os
import threading
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import packaging.version

from .languages import resolve_translation_language
from .runtime import application_data_dir, application_roots


_DEFAULT_ARGOS_PACKAGES_DIR = application_data_dir() / "argos" / "packages"
os.environ.setdefault("ARGOS_PACKAGES_DIR", str(_DEFAULT_ARGOS_PACKAGES_DIR))

try:
    import ctranslate2
except ImportError:  # pragma: no cover - optional runtime dependency
    ctranslate2 = None

try:
    from argostranslate import package as argos_package
except ImportError:  # pragma: no cover - optional runtime dependency
    argos_package = None

try:
    from deep_translator import GoogleTranslator
    from deep_translator.google import (
        BeautifulSoup,
        RequestError,
        TooManyRequests,
        TranslationNotFound,
        is_empty,
        is_input_valid,
        request_failed,
        requests,
    )
except ImportError:  # pragma: no cover - optional runtime dependency
    GoogleTranslator = None
    BeautifulSoup = None
    RequestError = TooManyRequests = TranslationNotFound = None
    is_empty = is_input_valid = request_failed = requests = None


@dataclass(slots=True, frozen=True)
class TranslationResult:
    text: str = ""


@dataclass(slots=True, frozen=True)
class _TranslationCacheKey:
    text: str
    source_language_code: str
    target_language_code: str


@dataclass(slots=True, frozen=True)
class _ArgosModelMetadata:
    from_code: str
    to_code: str
    package_version: str
    path: Path


class TranslationBackend:
    name = "disabled"

    def is_available(self) -> bool:
        return False

    def describe(self) -> str:
        return "Translation unavailable"

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        return TranslationResult(text="")

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        return [
            self.translate(
                text,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            ).text
            for text in texts
        ]

    def close(self) -> None:
        return None


class NoOpTranslationBackend(TranslationBackend):
    name = "disabled"

    def __init__(self, message: str = "Translation unavailable") -> None:
        self._message = message

    def describe(self) -> str:
        return self._message


if GoogleTranslator is not None:
    class TimedGoogleTranslator(GoogleTranslator):
        def __init__(
            self,
            *,
            source: str,
            target: str,
            request_timeout_s: float,
        ) -> None:
            super().__init__(source=source, target=target)
            self.request_timeout_s = request_timeout_s

        def translate(self, text: str, **kwargs) -> str:
            timeout_s = max(float(kwargs.pop("timeout_s", self.request_timeout_s)), 0.05)
            if is_input_valid(text, max_chars=5000):
                text = text.strip()
                if self._same_source_target() or is_empty(text):
                    return text
                self._url_params["tl"] = self._target
                self._url_params["sl"] = self._source

                if self.payload_key:
                    self._url_params[self.payload_key] = text

                response = requests.get(
                    self._base_url,
                    params=self._url_params,
                    proxies=self.proxies,
                    timeout=timeout_s,
                )
                if response.status_code == 429:
                    raise TooManyRequests()

                if request_failed(status_code=response.status_code):
                    raise RequestError()

                soup = BeautifulSoup(response.text, "html.parser")

                element = soup.find(self._element_tag, self._element_query)
                response.close()

                if not element:
                    element = soup.find(self._element_tag, self._alt_element_query)
                    if not element:
                        raise TranslationNotFound(text)
                if element.get_text(strip=True) == text.strip():
                    to_translate_alpha = "".join(ch for ch in text.strip() if ch.isalnum())
                    translated_alpha = "".join(ch for ch in element.get_text(strip=True) if ch.isalnum())
                    if to_translate_alpha and translated_alpha and to_translate_alpha == translated_alpha:
                        self._url_params["tl"] = self._target
                        if "hl" not in self._url_params:
                            return text.strip()
                        del self._url_params["hl"]
                        return self.translate(text, timeout_s=timeout_s)

                return element.get_text(strip=True)


class GoogleTranslateBackend(TranslationBackend):
    name = "google-translate"

    def __init__(
        self,
        *,
        request_timeout_s: float = 2.0,
        batch_timeout_s: float = 8.0,
        max_requests_per_batch: int = 8,
        retry_cooldown_seconds: float = 6.0,
    ) -> None:
        self._cache: dict[tuple[str, str, str], str] = {}
        self._retry_after: dict[tuple[str, str, str], float] = {}
        self._translators: dict[tuple[str, str], TimedGoogleTranslator] = {}
        self._request_timeout_s = request_timeout_s
        self._batch_timeout_s = batch_timeout_s
        self._max_requests_per_batch = max_requests_per_batch
        self._retry_cooldown_seconds = retry_cooldown_seconds

    def is_available(self) -> bool:
        return GoogleTranslator is not None

    def describe(self) -> str:
        if self.is_available():
            return "Google Translate (Online, budgeted)"
        return "Install deep-translator to enable Google Translate (Online)"

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        translated = self.translate_batch(
            [text],
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )[0]
        return TranslationResult(text=translated)

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        if not texts:
            return []

        normalized_texts = [_normalize_text(text) for text in texts]
        source_language = resolve_translation_language(source_language_code)
        target_language = resolve_translation_language(target_language_code)
        if target_language == "auto":
            return [""] * len(texts)
        if source_language != "auto" and source_language == target_language:
            return normalized_texts
        if not self.is_available():
            return [""] * len(texts)

        results = [""] * len(texts)
        translator = self._get_translator(source_language, target_language)
        deadline = perf_counter() + self._batch_timeout_s
        attempted_requests = 0

        for index, normalized in enumerate(normalized_texts):
            if not normalized:
                continue

            cache_key = (normalized, source_language, target_language)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[index] = cached
                continue

            retry_after = self._retry_after.get(cache_key, 0.0)
            if retry_after > perf_counter():
                continue

            if attempted_requests >= self._max_requests_per_batch:
                continue

            remaining_budget = deadline - perf_counter()
            if remaining_budget <= 0:
                continue

            attempted_requests += 1
            timeout_s = min(self._request_timeout_s, remaining_budget)
            try:
                translated = translator.translate(normalized, timeout_s=timeout_s)
            except Exception:
                self._retry_after[cache_key] = perf_counter() + self._retry_cooldown_seconds
                continue

            normalized_translated = _normalize_text(translated)
            if not normalized_translated:
                self._retry_after[cache_key] = perf_counter() + self._retry_cooldown_seconds
                continue

            self._cache[cache_key] = normalized_translated
            self._retry_after.pop(cache_key, None)
            results[index] = normalized_translated

        return results

    def _get_translator(self, source_language: str, target_language: str) -> TimedGoogleTranslator:
        cache_key = (source_language, target_language)
        translator = self._translators.get(cache_key)
        if translator is None:
            translator = TimedGoogleTranslator(
                source=source_language,
                target=target_language,
                request_timeout_s=self._request_timeout_s,
            )
            self._translators[cache_key] = translator
        return translator


class _ArgosPairTranslator:
    def __init__(self, pkg: object, *, device: str, compute_type: str = "auto") -> None:
        self._pkg = pkg
        self._device = device
        self._compute_type = compute_type
        self._translator: object | None = None
        self._translator_lock = threading.Lock()

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        normalized_texts = [_normalize_text(text) for text in texts]
        results = [""] * len(normalized_texts)
        active_indices: list[int] = []
        tokenized_batch: list[list[str]] = []

        for index, text in enumerate(normalized_texts):
            if not text:
                continue
            active_indices.append(index)
            tokenized_batch.append(self._pkg.tokenizer.encode(text))

        if not tokenized_batch:
            return results

        translator = self._get_translator()
        translated_batch = translator.translate_batch(
            tokenized_batch,
            max_batch_size=32,
            batch_type="examples",
            beam_size=2,
            num_hypotheses=1,
            length_penalty=0.2,
            replace_unknowns=True,
            max_decoding_length=256,
        )

        for index, translated in zip(active_indices, translated_batch, strict=False):
            hypothesis = translated.hypotheses[0] if translated.hypotheses else []
            decoded = self._pkg.tokenizer.decode(hypothesis)
            target_prefix = getattr(self._pkg, "target_prefix", "")
            if target_prefix and decoded.startswith(target_prefix):
                decoded = decoded[len(target_prefix) :]
            results[index] = _normalize_text(decoded)

        return results

    def _get_translator(self) -> object:
        translator = self._translator
        if translator is not None:
            return translator

        with self._translator_lock:
            translator = self._translator
            if translator is None:
                translator = ctranslate2.Translator(
                    str(self._pkg.package_path / "model"),
                    device=self._device,
                    compute_type=self._compute_type,
                    inter_threads=1,
                    intra_threads=0,
                )
                self._translator = translator
            return translator


class ArgosTranslateBackend(TranslationBackend):
    name = "argos-translate"

    def __init__(self) -> None:
        self._pair_translators: dict[tuple[str, str], _ArgosPairTranslator] = {}
        self._install_lock = threading.Lock()
        self._device = _resolve_argos_device()
        self._compute_type = "auto"
        self._ensure_bundled_models_installed()

    def is_available(self) -> bool:
        if argos_package is None or ctranslate2 is None:
            return False
        self._ensure_bundled_models_installed()
        return any(
            pkg.type == "translate" and {pkg.from_code, pkg.to_code}.issubset({"en", "th"})
            for pkg in argos_package.get_installed_packages()
        )

    def describe(self) -> str:
        if argos_package is None or ctranslate2 is None:
            return "Install argostranslate to enable Argos Translate (Offline)"
        if self.is_available():
            return "Argos Translate (Offline)"
        return "Argos Translate (Offline; install bundled en<->th models)"

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        translated = self.translate_batch(
            [text],
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )[0]
        return TranslationResult(text=translated)

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        if not texts:
            return []

        normalized_texts = [_normalize_text(text) for text in texts]
        target_language = resolve_translation_language(target_language_code)
        if target_language == "auto":
            return [""] * len(texts)

        results = [""] * len(texts)
        grouped_indices: dict[str, list[int]] = {}
        for index, text in enumerate(normalized_texts):
            if not text:
                continue

            source_language = self._resolve_source_language(
                text,
                source_language_code=source_language_code,
                target_language_code=target_language,
            )
            if source_language is None:
                continue
            if source_language == target_language:
                results[index] = text
                continue

            grouped_indices.setdefault(source_language, []).append(index)

        for source_language, indices in grouped_indices.items():
            pair_translator = self._get_pair_translator(source_language, target_language)
            if pair_translator is None:
                continue

            translated_batch = pair_translator.translate_batch([normalized_texts[index] for index in indices])
            for index, translated_text in zip(indices, translated_batch, strict=False):
                results[index] = translated_text

        return results

    def _get_pair_translator(self, source_language_code: str, target_language_code: str) -> _ArgosPairTranslator | None:
        cache_key = (source_language_code, target_language_code)
        translator = self._pair_translators.get(cache_key)
        if translator is not None:
            return translator

        package_candidate = _find_best_installed_argos_package(source_language_code, target_language_code)
        if package_candidate is None:
            return None

        translator = _ArgosPairTranslator(
            package_candidate,
            device=self._device,
            compute_type=self._compute_type,
        )
        self._pair_translators[cache_key] = translator
        return translator

    def _resolve_source_language(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> str | None:
        resolved_source = resolve_translation_language(source_language_code)
        if resolved_source != "auto":
            return resolved_source

        thai_count = sum("\u0E00" <= character <= "\u0E7F" for character in text)
        latin_count = sum(character.isascii() and character.isalpha() for character in text)

        if target_language_code == "th":
            if latin_count == 0 and thai_count > 0:
                return "th"
            if latin_count > 0:
                return "en"
            return None

        if target_language_code == "en":
            if thai_count == 0 and latin_count > 0:
                return "en"
            if thai_count > 0:
                return "th"
            return None

        if thai_count > latin_count:
            return "th"
        if latin_count > 0:
            return "en"
        return None

    def _ensure_bundled_models_installed(self) -> None:
        if argos_package is None:
            return

        with self._install_lock:
            bundled_models = _discover_bundled_argos_models()
            if not bundled_models:
                return

            installed_packages = argos_package.get_installed_packages()
            installed_by_pair: dict[tuple[str, str], list[object]] = {}
            for installed_package in installed_packages:
                if installed_package.type != "translate":
                    continue
                installed_by_pair.setdefault(
                    (installed_package.from_code, installed_package.to_code),
                    [],
                ).append(installed_package)

            changed = False
            for model in bundled_models:
                pair = (model.from_code, model.to_code)
                installed_matches = installed_by_pair.get(pair, [])
                up_to_date = any(
                    packaging.version.parse(pkg.package_version) >= packaging.version.parse(model.package_version)
                    for pkg in installed_matches
                )
                if up_to_date:
                    continue

                for installed_package in installed_matches:
                    argos_package.uninstall(installed_package)

                argos_package.install_from_path(model.path)
                changed = True

            if changed:
                self._pair_translators.clear()


class QueuedTranslationBackend(TranslationBackend):
    name = "queued-translation"

    def __init__(
        self,
        underlying: TranslationBackend,
        *,
        max_batch_size: int = 8,
        synchronous_batch_size: int = 0,
        retry_cooldown_seconds: float = 1.5,
    ) -> None:
        self._underlying = underlying
        self._max_batch_size = max(max_batch_size, 1)
        self._synchronous_batch_size = max(synchronous_batch_size, 0)
        self._retry_cooldown_seconds = retry_cooldown_seconds
        self._cache: dict[_TranslationCacheKey, str] = {}
        self._retry_after: dict[_TranslationCacheKey, float] = {}
        self._queue: deque[_TranslationCacheKey] = deque()
        self._queued_keys: set[_TranslationCacheKey] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"{underlying.name}-translation-worker",
            daemon=True,
        )
        self._worker.start()

    def is_available(self) -> bool:
        return self._underlying.is_available()

    def describe(self) -> str:
        return self._underlying.describe()

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        translated = self.translate_batch(
            [text],
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )[0]
        return TranslationResult(text=translated)

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        if not texts:
            return []

        normalized_texts = [_normalize_text(text) for text in texts]
        source_language = resolve_translation_language(source_language_code)
        target_language = resolve_translation_language(target_language_code)
        if target_language == "auto":
            return [""] * len(texts)
        if source_language != "auto" and source_language == target_language:
            return normalized_texts
        if not self._underlying.is_available():
            return [""] * len(texts)

        results = [""] * len(texts)
        now = perf_counter()
        keys: list[_TranslationCacheKey | None] = [None] * len(texts)
        missing_indices: list[int] = []

        with self._condition:
            for index, text in enumerate(normalized_texts):
                if not text:
                    continue

                key = _TranslationCacheKey(text, source_language_code, target_language_code)
                keys[index] = key
                cached = self._cache.get(key)
                if cached is not None:
                    results[index] = cached
                    continue

                retry_after = self._retry_after.get(key, 0.0)
                if retry_after > now:
                    continue

                missing_indices.append(index)

        sync_indices = missing_indices[: self._synchronous_batch_size]
        if sync_indices:
            sync_keys = {keys[index] for index in sync_indices if keys[index] is not None}
            with self._condition:
                self._remove_queued_keys_locked(sync_keys)

            sync_texts = [normalized_texts[index] for index in sync_indices]
            try:
                translated_batch = self._underlying.translate_batch(
                    sync_texts,
                    source_language_code=source_language_code,
                    target_language_code=target_language_code,
                )
            except Exception:
                translated_batch = [""] * len(sync_indices)

            now = perf_counter()
            with self._condition:
                for index, translated_text in zip(sync_indices, translated_batch, strict=False):
                    key = keys[index]
                    if key is None:
                        continue

                    normalized = _normalize_text(translated_text)
                    self._queued_keys.discard(key)
                    if normalized:
                        self._cache[key] = normalized
                        self._retry_after.pop(key, None)
                        results[index] = normalized
                    else:
                        self._retry_after[key] = now + self._retry_cooldown_seconds

                for index in sync_indices[len(translated_batch) :]:
                    key = keys[index]
                    if key is not None:
                        self._retry_after[key] = now + self._retry_cooldown_seconds

        queued_any = False
        sync_index_set = set(sync_indices)
        now = perf_counter()

        with self._condition:
            for index in missing_indices:
                if index in sync_index_set:
                    continue

                key = keys[index]
                if key is None:
                    continue

                cached = self._cache.get(key)
                if cached is not None:
                    results[index] = cached
                    continue

                retry_after = self._retry_after.get(key, 0.0)
                if retry_after > now:
                    continue
                if key in self._queued_keys:
                    continue

                self._queue.append(key)
                self._queued_keys.add(key)
                queued_any = True

            if queued_any:
                self._condition.notify()

        return results

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()

        self._worker.join(timeout=5.0)
        self._underlying.close()

    def _remove_queued_keys_locked(self, keys: set[_TranslationCacheKey | None]) -> None:
        resolved_keys = {key for key in keys if key is not None}
        if not resolved_keys or not self._queue:
            return

        self._queue = deque(key for key in self._queue if key not in resolved_keys)
        self._queued_keys.difference_update(resolved_keys)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._queue:
                    self._condition.wait()

                if self._closed and not self._queue:
                    return

                batch = self._pop_next_batch_locked()

            texts = [key.text for key in batch]
            source_language_code = batch[0].source_language_code
            target_language_code = batch[0].target_language_code

            try:
                translated_batch = self._underlying.translate_batch(
                    texts,
                    source_language_code=source_language_code,
                    target_language_code=target_language_code,
                )
            except Exception:
                translated_batch = [""] * len(batch)

            now = perf_counter()
            with self._condition:
                for key, translated_text in zip(batch, translated_batch, strict=False):
                    self._queued_keys.discard(key)
                    normalized = _normalize_text(translated_text)
                    if normalized:
                        self._cache[key] = normalized
                        self._retry_after.pop(key, None)
                    else:
                        self._retry_after[key] = now + self._retry_cooldown_seconds

                for key in batch[len(translated_batch) :]:
                    self._queued_keys.discard(key)
                    self._retry_after[key] = now + self._retry_cooldown_seconds

    def _pop_next_batch_locked(self) -> list[_TranslationCacheKey]:
        first = self._queue.popleft()
        batch = [first]
        deferred: deque[_TranslationCacheKey] = deque()

        while self._queue:
            candidate = self._queue.popleft()
            same_route = (
                candidate.source_language_code == first.source_language_code
                and candidate.target_language_code == first.target_language_code
            )
            if same_route and len(batch) < self._max_batch_size:
                batch.append(candidate)
            else:
                deferred.append(candidate)

        self._queue = deferred
        return batch


def create_default_translation_backend(*, mode: str | None = None) -> TranslationBackend:
    preference = normalize_translation_mode(mode if mode is not None else os.getenv("SCREENLENS_TRANSLATION_MODE", "auto"))

    if preference == "disabled":
        return NoOpTranslationBackend("Translation disabled")

    if preference == "google":
        backend = GoogleTranslateBackend()
        return _maybe_queue_translation_backend(backend) if backend.is_available() else backend

    if preference == "argos":
        backend = ArgosTranslateBackend()
        return _maybe_queue_translation_backend(backend) if backend.is_available() else backend

    for backend_factory in (ArgosTranslateBackend, GoogleTranslateBackend):
        backend = backend_factory()
        if backend.is_available():
            return _maybe_queue_translation_backend(backend)

    return NoOpTranslationBackend("Translation unavailable")


def _maybe_queue_translation_backend(backend: TranslationBackend) -> TranslationBackend:
    async_setting = os.getenv("SCREENLENS_TRANSLATION_ASYNC", "1").strip().lower()
    if async_setting in {"0", "false", "no", "off", "disabled"}:
        return backend

    sync_default = 2 if backend.name == "argos-translate" else 0
    sync_batch_size = _parse_non_negative_int(
        os.getenv("SCREENLENS_TRANSLATION_SYNC_BATCH_SIZE"),
        default=sync_default,
    )
    return QueuedTranslationBackend(backend, synchronous_batch_size=sync_batch_size)


def normalize_translation_mode(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"disabled", "none", "off"}:
        return "disabled"
    if normalized in {"google", "google_online", "online"}:
        return "google"
    if normalized in {"argos", "argos_offline", "offline"}:
        return "argos"
    return "auto"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _parse_non_negative_int(value: str | None, *, default: int) -> int:
    try:
        return max(int(value), 0) if value is not None else default
    except ValueError:
        return default


def _resolve_argos_device() -> str:
    if ctranslate2 is None:
        return "cpu"

    requested = os.getenv("SCREENLENS_ARGOS_DEVICE", "cpu").strip().lower()
    if requested in {"gpu", "cuda"} and ctranslate2.get_cuda_device_count() > 0:
        return "cuda"
    return "cpu"


def _discover_bundled_argos_models() -> list[_ArgosModelMetadata]:
    bundled_models: list[_ArgosModelMetadata] = []
    seen_paths: set[Path] = set()

    for root in application_roots():
        for relative_dir in (Path("vendor") / "argos", Path("argos")):
            candidate_dir = root / relative_dir
            if not candidate_dir.is_dir():
                continue

            for model_path in candidate_dir.glob("*.argosmodel"):
                resolved = model_path.resolve()
                if resolved in seen_paths:
                    continue

                metadata = _read_argos_model_metadata(resolved)
                if metadata is None:
                    continue

                bundled_models.append(metadata)
                seen_paths.add(resolved)

    bundled_models.sort(key=lambda model: (model.from_code, model.to_code, model.package_version))
    return bundled_models


def _read_argos_model_metadata(path: Path) -> _ArgosModelMetadata | None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            metadata_name = next(
                (name for name in archive.namelist() if name.endswith("metadata.json") and not name.endswith("/")),
                None,
            )
            if metadata_name is None:
                return None

            with archive.open(metadata_name) as metadata_file:
                metadata = json.load(metadata_file)
    except (OSError, StopIteration, zipfile.BadZipFile, json.JSONDecodeError):
        return None

    from_code = str(metadata.get("from_code", "")).strip()
    to_code = str(metadata.get("to_code", "")).strip()
    package_version = str(metadata.get("package_version", "0")).strip()
    if not from_code or not to_code:
        return None

    return _ArgosModelMetadata(
        from_code=from_code,
        to_code=to_code,
        package_version=package_version,
        path=path,
    )


def _find_best_installed_argos_package(source_language_code: str, target_language_code: str) -> object | None:
    if argos_package is None:
        return None

    candidates = [
        pkg
        for pkg in argos_package.get_installed_packages()
        if pkg.type == "translate"
        and pkg.from_code == source_language_code
        and pkg.to_code == target_language_code
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda pkg: packaging.version.parse(pkg.package_version), reverse=True)
    return candidates[0]
