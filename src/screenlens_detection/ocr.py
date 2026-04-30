from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .runtime import application_roots

try:
    from easyocr import Reader as EasyOCRReader
except ImportError:  # pragma: no cover - optional runtime dependency
    EasyOCRReader = None

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    pytesseract = None


@dataclass(slots=True, frozen=True)
class OCRResult:
    text: str = ""
    confidence: float | None = None


@dataclass(slots=True, frozen=True)
class _OCRCacheKey:
    digest: bytes
    shape: tuple[int, ...]
    language: str
    psm: int


@dataclass(slots=True)
class _QueuedOCRItem:
    key: _OCRCacheKey
    image: np.ndarray


class OCRBackend:
    name = "disabled"

    def is_available(self) -> bool:
        return False

    def describe(self) -> str:
        return "OCR unavailable"

    def runtime_diagnostics(self) -> str:
        return self.describe()

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image)

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        return OCRResult()

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        return [
            self.recognize(image, language=language, psm=psm)
            for image, psm in zip(images, psms, strict=False)
        ]

    def close(self) -> None:
        return None


class NoOpOCRBackend(OCRBackend):
    name = "disabled"

    def describe(self) -> str:
        return "Detection-only mode"


class QueuedOCRBackend(OCRBackend):
    name = "queued-ocr"

    def __init__(
        self,
        underlying: OCRBackend,
        *,
        max_batch_size: int = 32,
        synchronous_batch_size: int = 4,
        max_queue_size: int = 96,
    ) -> None:
        self._underlying = underlying
        self._max_batch_size = max(max_batch_size, 1)
        self._synchronous_batch_size = max(synchronous_batch_size, 0)
        self._max_queue_size = max(max_queue_size, self._max_batch_size)
        self._cache: dict[_OCRCacheKey, OCRResult] = {}
        self._queue: deque[_QueuedOCRItem] = deque()
        self._queued_keys: set[_OCRCacheKey] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"{underlying.name}-ocr-worker",
            daemon=True,
        )
        self._worker.start()

    def is_available(self) -> bool:
        return self._underlying.is_available()

    def describe(self) -> str:
        return f"{self._underlying.describe()} (queued)"

    def runtime_diagnostics(self) -> str:
        if self._synchronous_batch_size > 0:
            return (
                f"{self._underlying.runtime_diagnostics()} | "
                f"hybrid OCR queue, sync first {self._synchronous_batch_size}"
            )
        return f"{self._underlying.runtime_diagnostics()} | async OCR queue"

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        return self._underlying.prepare_image(image)

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        return self.recognize_batch([image], language=language, psms=[psm])[0]

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        if not images:
            return []
        if not self._underlying.is_available():
            return [OCRResult() for _image in images]

        results = [OCRResult() for _image in images]
        arrays: list[np.ndarray] = []
        keys: list[_OCRCacheKey] = []
        missing_indices: list[int] = []

        for index, image in enumerate(images):
            psm = psms[index] if index < len(psms) else 7
            array = self._normalize_queued_image(image)
            arrays.append(array)
            keys.append(self._cache_key(array, language=language, psm=psm))

        with self._condition:
            for index, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    results[index] = cached
                    continue
                missing_indices.append(index)

        sync_indices = []
        if missing_indices and self._synchronous_batch_size > 0:
            if len(missing_indices) <= self._synchronous_batch_size:
                sync_indices = list(missing_indices)
            else:
                sync_indices = sorted(
                    missing_indices,
                    key=lambda index: self._sync_priority(arrays[index]),
                    reverse=True,
                )[: self._synchronous_batch_size]

        if sync_indices:
            sync_keys = {keys[index] for index in sync_indices}
            with self._condition:
                self._remove_queued_keys_locked(sync_keys)

            missing_images = [arrays[index] for index in sync_indices]
            missing_psms = [keys[index].psm for index in sync_indices]
            try:
                recognized_batch = self._underlying.recognize_batch(
                    missing_images,
                    language=language,
                    psms=missing_psms,
                )
            except Exception:
                recognized_batch = [OCRResult() for _index in sync_indices]

            with self._condition:
                for index, result in zip(sync_indices, recognized_batch, strict=False):
                    if result.text:
                        self._cache[keys[index]] = result
                    self._queued_keys.discard(keys[index])
                    results[index] = result

        queued_any = False
        sync_index_set = set(sync_indices)

        with self._condition:
            for index in missing_indices:
                if index in sync_index_set:
                    continue
                key = keys[index]
                if key in self._queued_keys:
                    continue

                self._trim_queue_locked()
                self._queue.append(_QueuedOCRItem(key=key, image=arrays[index]))
                self._queued_keys.add(key)
                queued_any = True

            if queued_any:
                self._condition.notify()

        return results

    def _trim_queue_locked(self) -> None:
        while len(self._queue) >= self._max_queue_size:
            dropped = self._queue.popleft()
            self._queued_keys.discard(dropped.key)

    def _remove_queued_keys_locked(self, keys: set[_OCRCacheKey]) -> None:
        if not keys or not self._queue:
            return

        self._queue = deque(item for item in self._queue if item.key not in keys)
        self._queued_keys.difference_update(keys)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()

        self._worker.join(timeout=5.0)
        self._underlying.close()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._queue:
                    self._condition.wait()

                if self._closed and not self._queue:
                    return

                batch = self._pop_next_batch_locked()

            language = batch[0].key.language
            images = [item.image for item in batch]
            psms = [item.key.psm for item in batch]

            try:
                recognized_batch = self._underlying.recognize_batch(images, language=language, psms=psms)
            except Exception:
                recognized_batch = [OCRResult() for _item in batch]

            with self._condition:
                for item, result in zip(batch, recognized_batch, strict=False):
                    self._queued_keys.discard(item.key)
                    if result.text:
                        self._cache[item.key] = result

                for item in batch[len(recognized_batch) :]:
                    self._queued_keys.discard(item.key)

    def _pop_next_batch_locked(self) -> list[_QueuedOCRItem]:
        first = self._queue.popleft()
        batch = [first]
        deferred: deque[_QueuedOCRItem] = deque()

        while self._queue:
            candidate = self._queue.popleft()
            if candidate.key.language == first.key.language and len(batch) < self._max_batch_size:
                batch.append(candidate)
            else:
                deferred.append(candidate)

        self._queue = deferred
        return batch

    @staticmethod
    def _normalize_queued_image(image: object) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array).copy()

    @staticmethod
    def _cache_key(image: np.ndarray, *, language: str, psm: int) -> _OCRCacheKey:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(image.dtype).encode("ascii"))
        hasher.update(np.asarray(image.shape, dtype=np.int64).tobytes())
        hasher.update(image.tobytes())
        return _OCRCacheKey(
            digest=hasher.digest(),
            shape=tuple(int(dimension) for dimension in image.shape),
            language=language,
            psm=psm,
        )

    @staticmethod
    def _sync_priority(image: np.ndarray) -> float:
        height, width = image.shape[:2]
        area = width * height
        aspect_ratio = width / max(height, 1)
        return (
            min(width / 8.0, 160.0)
            + min(area / 1000.0, 100.0)
            + min(aspect_ratio, 24.0) * 5.0
        )


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(self, *, gpu: bool | None = None, device_preference: str = "auto") -> None:
        if gpu is not None:
            device_preference = "gpu" if gpu else "cpu"

        self._device_preference = normalize_ocr_device_preference(device_preference)
        self._gpu_available = _nvidia_cuda_available()
        self._gpu = self._resolve_gpu_enabled()
        self._readers: dict[tuple[str, ...], object] = {}

    def is_available(self) -> bool:
        return EasyOCRReader is not None

    def describe(self) -> str:
        if self.is_available():
            if self._device_preference == "gpu" and not self._gpu:
                return "EasyOCR (CPU fallback; NVIDIA CUDA unavailable)"
            device = "GPU" if self._gpu else "CPU"
            return f"EasyOCR ({device})"
        return "Install easyocr (and PyTorch) to enable the upgraded OCR backend"

    def runtime_diagnostics(self) -> str:
        if not self.is_available():
            return "EasyOCR unavailable"

        requested = _ocr_device_label(self._device_preference)
        if self._gpu:
            active = "GPU (NVIDIA CUDA)"
        elif self._device_preference == "gpu":
            active = "CPU fallback"
        else:
            active = "CPU"

        details = [
            "EasyOCR",
            f"requested {requested}",
            f"active {active}",
            _torch_runtime_summary(),
        ]

        device_name = _torch_device_name()
        if device_name:
            details.append(device_name)

        return " | ".join(part for part in details if part)

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        normalized = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = normalized.shape[:2]

        scale = 1.0
        if height < 36:
            scale = max(scale, 2.0)
        if width < 160:
            scale = max(scale, 1.5)

        if scale > 1.0:
            normalized = cv2.resize(normalized, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return cv2.copyMakeBorder(
            normalized,
            10,
            10,
            14,
            14,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        if not self.is_available():
            return OCRResult()

        try:
            reader = self._get_reader(language)
            results = reader.readtext(np.asarray(image), detail=1, paragraph=False)
        except Exception:
            return OCRResult()

        return self._aggregate_results(results)

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        if not images:
            return []
        if not self.is_available():
            return [OCRResult() for _image in images]
        if len(images) == 1:
            psm = psms[0] if psms else 7
            return [self.recognize(images[0], language=language, psm=psm)]

        try:
            reader = self._get_reader(language)
            arrays = [self._ensure_batch_image(image) for image in images]
            canvas, regions = self._tile_batch_images(arrays)
            horizontal_list = [[0, width, top, bottom] for top, bottom, width in regions]
            raw_results = reader.recognize(
                canvas,
                horizontal_list=horizontal_list,
                free_list=[],
                detail=1,
                paragraph=False,
                batch_size=max(len(arrays), 1),
                reformat=False,
            )
        except Exception:
            return [OCRResult() for _image in images]

        grouped_results: list[list[tuple[object, object, object]]] = [[] for _image in images]
        fallback_index = 0
        for result in sorted(raw_results, key=self._result_sort_key):
            region_index = self._result_region_index(result, regions)
            if region_index is None:
                if fallback_index >= len(grouped_results):
                    continue
                region_index = fallback_index
                fallback_index += 1
            grouped_results[region_index].append(result)

        return [self._aggregate_results(results) for results in grouped_results]

    @staticmethod
    def _aggregate_results(results: object) -> OCRResult:
        tokens: list[str] = []
        confidences: list[float] = []

        ordered_results = sorted(results, key=EasyOCRBackend._result_sort_key)
        for _bbox, token, confidence in ordered_results:
            normalized_token = re.sub(r"\s+", " ", str(token)).strip()
            if not normalized_token:
                continue

            tokens.append(normalized_token)
            try:
                confidences.append(float(confidence) * 100.0)
            except (TypeError, ValueError):
                continue

        normalized = " ".join(tokens)
        average_confidence = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text=normalized, confidence=average_confidence)

    @staticmethod
    def _ensure_batch_image(image: object) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _tile_batch_images(images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        gap = 16
        max_width = max((image.shape[1] for image in images), default=1)
        total_height = sum(image.shape[0] for image in images) + gap * max(len(images) - 1, 0)
        canvas = np.full((max(total_height, 1), max(max_width, 1)), 255, dtype=np.uint8)

        regions: list[tuple[int, int, int]] = []
        top = 0
        for image in images:
            height, width = image.shape[:2]
            bottom = top + height
            canvas[top:bottom, :width] = image
            regions.append((top, bottom, width))
            top = bottom + gap

        return canvas, regions

    @staticmethod
    def _result_region_index(
        result: tuple[object, object, object],
        regions: list[tuple[int, int, int]],
    ) -> int | None:
        center_y = EasyOCRBackend._result_center_y(result)
        if center_y is None:
            return None

        for index, (top, bottom, _width) in enumerate(regions):
            if top <= center_y <= bottom:
                return index
        return None

    @staticmethod
    def _result_center_y(result: tuple[object, object, object]) -> float | None:
        bbox = result[0]
        if not isinstance(bbox, list) or not bbox:
            return None

        ys = [float(point[1]) for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not ys:
            return None
        return (min(ys) + max(ys)) / 2.0

    def _get_reader(self, language: str) -> object:
        lang_list = self._resolve_language_list(language)
        reader = self._readers.get(lang_list)
        if reader is None:
            reader = EasyOCRReader(list(lang_list), gpu=self._gpu, verbose=False)
            self._readers[lang_list] = reader
        return reader

    @staticmethod
    def _resolve_language_list(language: str) -> tuple[str, ...]:
        tokens = {token.strip() for token in language.split("+") if token.strip()}
        lang_list: list[str] = []

        if "tha" in tokens:
            lang_list.append("th")
        if "eng" in tokens or not lang_list:
            lang_list.append("en")
        elif "th" in lang_list and "en" not in lang_list:
            lang_list.append("en")

        return tuple(lang_list)

    @staticmethod
    def _result_sort_key(result: tuple[object, object, object]) -> tuple[float, float]:
        bbox = result[0]
        if not isinstance(bbox, list) or not bbox:
            return (0.0, 0.0)

        xs = [float(point[0]) for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
        ys = [float(point[1]) for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not xs or not ys:
            return (0.0, 0.0)
        return (min(ys), min(xs))

    def _resolve_gpu_enabled(self) -> bool:
        if self._device_preference == "gpu":
            return self._gpu_available
        if self._device_preference == "cpu":
            return False
        return self._gpu_available


class TesseractOCRBackend(OCRBackend):
    name = "tesseract"

    def __init__(self) -> None:
        self._binary = self._resolve_binary()
        self._tessdata_dir = self._resolve_tessdata_dir(self._binary)
        self._available_languages = self._resolve_available_languages(self._tessdata_dir)
        if self._binary and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = self._binary
        if self._tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self._tessdata_dir

    @staticmethod
    def _resolve_binary() -> str | None:
        env_binary = os.getenv("TESSERACT_CMD")
        if env_binary and Path(env_binary).is_file():
            return env_binary

        for candidate in _runtime_tesseract_binary_candidates():
            if candidate.is_file():
                return str(candidate)

        resolved = shutil.which("tesseract")
        if resolved:
            return resolved

        common_windows_paths = (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        )
        for candidate in common_windows_paths:
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _resolve_tessdata_dir(binary: str | None) -> str | None:
        env_prefix = os.getenv("TESSDATA_PREFIX")
        candidates: list[Path] = []

        if env_prefix:
            prefix_path = Path(env_prefix)
            candidates.extend((prefix_path, prefix_path / "tessdata"))

        if binary:
            binary_dir = Path(binary).resolve().parent
            candidates.extend((binary_dir / "tessdata", binary_dir))

        candidates.extend(_runtime_tessdata_candidates())

        candidates.extend(
            (
                Path.home() / "AppData" / "Local" / "TesseractData" / "tessdata",
                Path.home() / "AppData" / "Local" / "TesseractData",
                Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
            )
        )

        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.traineddata")):
                return str(candidate)
        return None

    @staticmethod
    def _resolve_available_languages(tessdata_dir: str | None) -> set[str]:
        if not tessdata_dir:
            return set()

        try:
            return {path.stem for path in Path(tessdata_dir).glob("*.traineddata")}
        except OSError:
            return set()

    def is_available(self) -> bool:
        return self._binary is not None and pytesseract is not None

    def describe(self) -> str:
        if self.is_available():
            return f"Tesseract OCR ({self._binary})"
        return "Install or bundle Tesseract, or set TESSERACT_CMD to enable OCR"

    def runtime_diagnostics(self) -> str:
        if self.is_available():
            return f"Tesseract OCR | active CPU process | binary {self._binary}"
        return self.describe()

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        normalized = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if cv2.countNonZero(binary) < (binary.size // 2):
            binary = cv2.bitwise_not(binary)

        return cv2.copyMakeBorder(
            binary,
            6,
            6,
            10,
            10,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        if not self.is_available():
            return OCRResult()

        language = self._resolve_requested_language(language)
        config = f"--oem 3 --psm {psm}"
        candidates = self._build_candidates(np.asarray(image))
        best = OCRResult()
        best_score = -1.0

        for candidate in candidates:
            result = self._recognize_candidate(candidate, language=language, config=config)
            score = self._score_result(result)
            if score > best_score:
                best = result
                best_score = score

        return best

    def _resolve_requested_language(self, language: str) -> str:
        requested = [token.strip() for token in language.split("+") if token.strip()]
        if not requested or not self._available_languages:
            return language

        available = [token for token in requested if token in self._available_languages]
        if available:
            return "+".join(available)

        fallback = [token for token in ("eng", "tha") if token in self._available_languages]
        if fallback:
            return "+".join(fallback)
        return language

    def _build_candidates(self, image: np.ndarray) -> list[np.ndarray]:
        normalized = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return [normalized, otsu, adaptive]

    def _recognize_candidate(self, image: np.ndarray, *, language: str, config: str) -> OCRResult:
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(image),
                lang=language,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except (OSError, pytesseract.TesseractError):
            return OCRResult()

        tokens: list[str] = []
        confidences: list[float] = []

        for token, confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
            normalized_token = re.sub(r"\s+", " ", token).strip()
            if not normalized_token:
                continue

            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = -1.0

            tokens.append(normalized_token)
            if confidence_value >= 0:
                confidences.append(confidence_value)

        normalized = " ".join(tokens)
        average_confidence = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text=normalized, confidence=average_confidence)

    @staticmethod
    def _score_result(result: OCRResult) -> float:
        if not result.text:
            return -1.0

        confidence_score = result.confidence if result.confidence is not None else 0.0
        length_bonus = min(len(result.text), 80) / 10.0
        return confidence_score + length_bonus


def create_default_ocr_backend(*, device_preference: str | None = None) -> OCRBackend:
    preference = os.getenv("SCREENLENS_OCR_BACKEND", "auto").strip().lower()
    resolved_device_preference = normalize_ocr_device_preference(
        device_preference if device_preference is not None else os.getenv("SCREENLENS_OCR_DEVICE", "auto")
    )
    if preference in {"disabled", "none", "off"}:
        return NoOpOCRBackend()

    if preference == "tesseract":
        backend = TesseractOCRBackend()
        if backend.is_available():
            return backend
        return NoOpOCRBackend()

    if preference == "easyocr":
        backend = EasyOCRBackend(device_preference=resolved_device_preference)
        if backend.is_available():
            return _maybe_queue_ocr_backend(backend)
        return NoOpOCRBackend()

    for backend_cls in (EasyOCRBackend, TesseractOCRBackend):
        if backend_cls is EasyOCRBackend:
            backend = backend_cls(device_preference=resolved_device_preference)
        else:
            backend = backend_cls()
        if backend.is_available():
            if backend_cls is EasyOCRBackend:
                return _maybe_queue_ocr_backend(backend)
            return backend
    return NoOpOCRBackend()


def _maybe_queue_ocr_backend(backend: OCRBackend) -> OCRBackend:
    async_setting = os.getenv("SCREENLENS_OCR_ASYNC", "1").strip().lower()
    if async_setting in {"0", "false", "no", "off", "disabled"}:
        return backend
    sync_batch_size = _parse_non_negative_int(os.getenv("SCREENLENS_OCR_SYNC_BATCH_SIZE"), default=2)
    return QueuedOCRBackend(backend, synchronous_batch_size=sync_batch_size)


def _parse_non_negative_int(value: str | None, *, default: int) -> int:
    try:
        return max(int(value), 0) if value is not None else default
    except ValueError:
        return default


def normalize_ocr_device_preference(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"gpu", "cuda", "nvidia"}:
        return "gpu"
    if normalized == "cpu":
        return "cpu"
    return "auto"


def _nvidia_cuda_available() -> bool:
    if torch is None:
        return False

    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _ocr_device_label(value: str) -> str:
    labels = {
        "auto": "Auto",
        "cpu": "CPU",
        "gpu": "GPU",
    }
    return labels.get(value, value.upper())


def _torch_runtime_summary() -> str:
    if torch is None:
        return "torch unavailable"

    cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_runtime:
        build = f"torch {torch.__version__} (CUDA {cuda_runtime})"
    else:
        build = f"torch {torch.__version__} (CPU-only build)"

    cuda_available = "cuda available" if _nvidia_cuda_available() else "cuda unavailable"
    return f"{build}, {cuda_available}"


def _torch_device_name() -> str:
    if not _nvidia_cuda_available():
        return ""

    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return ""


def _runtime_tesseract_binary_candidates() -> list[Path]:
    candidates: list[Path] = []
    relative_locations = (
        Path("tesseract.exe"),
        Path("tesseract") / "tesseract.exe",
        Path("Tesseract-OCR") / "tesseract.exe",
        Path("vendor") / "tesseract" / "tesseract.exe",
    )

    for root in application_roots():
        for relative_path in relative_locations:
            candidates.append(root / relative_path)

    return _unique_paths(candidates)


def _runtime_tessdata_candidates() -> list[Path]:
    candidates: list[Path] = []
    relative_locations = (
        Path("tessdata"),
        Path("tesseract") / "tessdata",
        Path("Tesseract-OCR") / "tessdata",
        Path("vendor") / "tesseract" / "tessdata",
    )

    for root in application_roots():
        for relative_path in relative_locations:
            candidates.append(root / relative_path)

    return _unique_paths(candidates)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)

    return unique
