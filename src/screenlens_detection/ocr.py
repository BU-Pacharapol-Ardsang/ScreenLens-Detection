from __future__ import annotations

import os
import re
import shutil
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


class NoOpOCRBackend(OCRBackend):
    name = "disabled"

    def describe(self) -> str:
        return "Detection-only mode"


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

        tokens: list[str] = []
        confidences: list[float] = []

        ordered_results = sorted(results, key=self._result_sort_key)
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
            return backend
        return NoOpOCRBackend()

    for backend_cls in (EasyOCRBackend, TesseractOCRBackend):
        if backend_cls is EasyOCRBackend:
            backend = backend_cls(device_preference=resolved_device_preference)
        else:
            backend = backend_cls()
        if backend.is_available():
            return backend
    return NoOpOCRBackend()


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
