from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from numbers import Number
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class TextDetectorOption:
    code: str
    label: str


TEXT_DETECTOR_OPTIONS = (
    TextDetectorOption(code="opencv", label="Classic OpenCV (Morphology)"),
    TextDetectorOption(code="paddleocr", label="PaddleOCR DBNet (Optional)"),
    TextDetectorOption(code="easyocr", label="EasyOCR CRAFT (Optional)"),
)


def text_detector_options() -> list[TextDetectorOption]:
    return list(TEXT_DETECTOR_OPTIONS)


def normalize_text_detector_mode(value: str | None) -> str:
    available = {option.code for option in TEXT_DETECTOR_OPTIONS}
    normalized = (value or "opencv").strip().casefold()
    return normalized if normalized in available else "opencv"


class TextDetectorBackend:
    name = "text-detector"

    def is_available(self) -> bool:
        return True

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def close(self) -> None:
        return None


class UnavailableTextDetectorBackend(TextDetectorBackend):
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return []

    def describe(self) -> str:
        return f"{self.name} unavailable: {self.reason}"


class EasyOCRCraftTextDetector(TextDetectorBackend):
    name = "EasyOCR CRAFT detector"

    def __init__(self, *, language: str, device_preference: str) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("install optional dependency easyocr") from exc

        self._reader = easyocr.Reader(
            _easyocr_languages(language),
            gpu=_should_use_gpu(device_preference),
            detector=True,
            recognizer=False,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        horizontal_list, free_list = self._reader.detect(rgb)
        boxes = list(_iter_easyocr_horizontal_boxes(horizontal_list))
        boxes.extend(_boxes_from_polygons(_iter_polygons(free_list)))
        return boxes


class PaddleOCRDbTextDetector(TextDetectorBackend):
    name = "PaddleOCR DBNet detector"

    def __init__(self, *, language: str, device_preference: str) -> None:
        try:
            from paddleocr import TextDetection
        except ImportError as exc:
            raise RuntimeError("install optional dependencies paddleocr and paddlepaddle") from exc

        self._engine = self._create_engine(TextDetection, device_preference)

    @staticmethod
    def _create_engine(text_detection_class: type, device_preference: str) -> Any:
        device = _paddle_device(device_preference)
        candidates = (
            {
                "model_name": "PP-OCRv5_mobile_det",
                "device": device,
                "enable_mkldnn": False,
                "cpu_threads": 2,
            },
            {
                "model_name": "PP-OCRv5_mobile_det",
                "device": device,
                "engine": "paddle_static",
                "enable_mkldnn": False,
                "cpu_threads": 2,
            },
            {"model_name": "PP-OCRv5_mobile_det", "device": device},
            {"model_name": "PP-OCRv5_mobile_det"},
        )
        last_error: Exception | None = None
        for kwargs in candidates:
            try:
                return text_detection_class(**kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        return text_detection_class()

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        result = self._detect_with_engine(frame)
        return _boxes_from_polygons(_iter_polygons(result))

    def _detect_with_engine(self, frame: np.ndarray) -> object:
        if hasattr(self._engine, "predict"):
            return self._engine.predict(input=frame)

        raise RuntimeError("PaddleOCR text detector does not expose predict()")


def create_deep_text_detector_backend(
    mode: str,
    *,
    language: str,
    device_preference: str,
) -> TextDetectorBackend:
    normalized = normalize_text_detector_mode(mode)
    if normalized == "easyocr":
        return _create_optional_backend(
            EasyOCRCraftTextDetector,
            language=language,
            device_preference=device_preference,
        )
    if normalized == "paddleocr":
        return _create_optional_backend(
            PaddleOCRDbTextDetector,
            language=language,
            device_preference=device_preference,
        )
    return UnavailableTextDetectorBackend("Unknown deep detector", "select a deep detector backend")


def _create_optional_backend(
    backend_class: type[TextDetectorBackend],
    *,
    language: str,
    device_preference: str,
) -> TextDetectorBackend:
    try:
        return backend_class(language=language, device_preference=device_preference)
    except Exception as exc:
        return UnavailableTextDetectorBackend(backend_class.name, str(exc))


def _easyocr_languages(language: str) -> list[str]:
    normalized = language.casefold()
    if "tha" in normalized and "eng" in normalized:
        return ["th", "en"]
    if "tha" in normalized or normalized == "th":
        return ["th"]
    return ["en"]


def _paddle_device(device_preference: str) -> str:
    normalized = device_preference.casefold()
    if normalized not in {"gpu", "cuda"}:
        return "cpu"

    try:
        import paddle
    except ImportError:
        return "cpu"

    if hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda():
        return "gpu"
    return "cpu"


def _should_use_gpu(device_preference: str) -> bool:
    normalized = device_preference.casefold()
    if normalized in {"gpu", "cuda"}:
        return True
    if normalized == "auto":
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())
    return False


def _iter_easyocr_horizontal_boxes(data: object) -> list[tuple[int, int, int, int]]:
    if _is_numeric_sequence(data, length=4):
        left, right, top, bottom = [float(value) for value in data]  # type: ignore[arg-type]
        return [_rect_from_bounds(left, top, right, bottom)]

    boxes: list[tuple[int, int, int, int]] = []
    if isinstance(data, (list, tuple)):
        for item in data:
            boxes.extend(_iter_easyocr_horizontal_boxes(item))
    return boxes


def _iter_polygons(data: object) -> list[np.ndarray]:
    polygon = _as_polygon(data)
    if polygon is not None:
        return [polygon]

    polygons: list[np.ndarray] = []
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return polygons
        for item in data:
            polygons.extend(_iter_polygons(item))
        return polygons

    if isinstance(data, dict):
        for key in ("dt_polys", "dt_boxes", "boxes", "points", "poly"):
            if key in data:
                polygons.extend(_iter_polygons(data[key]))
        return polygons

    if isinstance(data, (list, tuple)):
        for item in data:
            polygons.extend(_iter_polygons(item))
        return polygons

    if hasattr(data, "__dict__"):
        polygons.extend(_iter_polygons(vars(data)))

    return polygons


def _as_polygon(data: object) -> np.ndarray | None:
    try:
        points = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    return points[:, :2]


def _boxes_from_polygons(polygons: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for polygon in polygons:
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        boxes.append(_rect_from_bounds(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    return boxes


def _rect_from_bounds(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
    x1 = max(int(floor(min(left, right))), 0)
    y1 = max(int(floor(min(top, bottom))), 0)
    x2 = max(int(ceil(max(left, right))), x1 + 1)
    y2 = max(int(ceil(max(top, bottom))), y1 + 1)
    return x1, y1, x2 - x1, y2 - y1


def _is_numeric_sequence(data: object, *, length: int) -> bool:
    return (
        isinstance(data, (list, tuple))
        and len(data) == length
        and all(isinstance(value, Number) for value in data)
    )
