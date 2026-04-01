from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass

from PIL import Image

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
        if self._binary and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = self._binary

    @staticmethod
    def _resolve_binary() -> str | None:
        env_binary = os.getenv("TESSERACT_CMD")
        if env_binary:
            return env_binary
        return shutil.which("tesseract")

    def is_available(self) -> bool:
        return self._binary is not None and pytesseract is not None

    def describe(self) -> str:
        if self.is_available():
            return f"Tesseract OCR ({self._binary})"
        return "Install Tesseract or set TESSERACT_CMD to enable OCR"

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        if not self.is_available():
            return OCRResult()

        config = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(Image.fromarray(image), lang=language, config=config)
        normalized = re.sub(r"\s+", " ", text).strip()
        return OCRResult(text=normalized or "")


def create_default_ocr_backend() -> OCRBackend:
    backend = TesseractOCRBackend()
    if backend.is_available():
        return backend
    return NoOpOCRBackend()

