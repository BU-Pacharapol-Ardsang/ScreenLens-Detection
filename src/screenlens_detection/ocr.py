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

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        return OCRResult()


class NoOpOCRBackend(OCRBackend):
    name = "disabled"

    def describe(self) -> str:
        return "Detection-only mode"


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


def create_default_ocr_backend() -> OCRBackend:
    backend = TesseractOCRBackend()
    if backend.is_available():
        return backend
    return NoOpOCRBackend()


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
