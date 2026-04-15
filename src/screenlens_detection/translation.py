from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .languages import resolve_translation_language

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


class NoOpTranslationBackend(TranslationBackend):
    name = "disabled"

    def describe(self) -> str:
        return "Translation unavailable"


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
        request_timeout_s: float = 0.75,
        batch_timeout_s: float = 2.0,
        max_requests_per_batch: int = 4,
        retry_cooldown_seconds: float = 8.0,
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
            return "Google Translate (budgeted)"
        return "Install deep-translator to enable translation"

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

        normalized_texts = [" ".join(text.split()) for text in texts]
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

            normalized_translated = " ".join((translated or "").split())
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


def create_default_translation_backend() -> TranslationBackend:
    backend = GoogleTranslateBackend()
    if backend.is_available():
        return backend
    return NoOpTranslationBackend()
