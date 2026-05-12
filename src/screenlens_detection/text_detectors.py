from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil, floor
from numbers import Number
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .onnxruntime_utils import onnxruntime_cuda_available, short_runtime_error
from .runtime import application_roots


@dataclass(slots=True, frozen=True)
class TextDetectorOption:
    code: str
    label: str


TEXT_DETECTOR_OPTIONS = (
    TextDetectorOption(code="opencv", label="Classic OpenCV (Morphology)"),
    TextDetectorOption(code="rapidocr", label="RapidOCR ONNX DBNet (Optional)"),
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


class RapidOCRTextDetector(TextDetectorBackend):
    name = "RapidOCR ONNX DBNet detector"

    def __init__(self, *, language: str, device_preference: str) -> None:
        try:
            import rapidocr
            from rapidocr import EngineType, LangDet, ModelType, OCRVersion
            from rapidocr.ch_ppocr_det import TextDetector
            from rapidocr.utils.parse_parameters import ParseParams
            from rapidocr.utils.process_img import (
                apply_vertical_padding,
                map_boxes_to_original,
                resize_image_within_bounds,
            )
        except ImportError as exc:
            raise RuntimeError("install optional dependencies rapidocr and onnxruntime") from exc

        self._use_cuda = _should_use_rapidocr_cuda(device_preference)
        self._cuda_fallback_reason = ""
        params = {
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": False,
            "Global.max_side_len": 1280,
            "Global.log_level": "error",
            "Det.engine_type": _enum_value(EngineType, "ONNXRUNTIME", "onnxruntime"),
            "Det.lang_type": _rapidocr_detection_language(LangDet, language),
            "Det.model_type": _enum_value(ModelType, "MOBILE", "mobile"),
            "Det.ocr_version": _enum_value(OCRVersion, "PPOCRV4", "PP-OCRv4"),
            "Det.limit_side_len": 960,
            "Det.limit_type": "max",
            "Det.box_thresh": 0.45,
            "Det.max_candidates": 300,
            "Det.score_mode": "fast",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": self._use_cuda,
        }

        package_dir = _rapidocr_package_data_dir(rapidocr)
        _quiet_rapidocr_logging()
        try:
            cfg = self._load_detector_config(ParseParams, package_dir, params)
            self._detector = TextDetector(cfg.Det)
        except Exception as cuda_exc:
            if not self._use_cuda:
                raise
            self._cuda_fallback_reason = short_runtime_error(str(cuda_exc))
            self._use_cuda = False
            params["EngineConfig.onnxruntime.use_cuda"] = False
            try:
                cfg = self._load_detector_config(ParseParams, package_dir, params)
                self._detector = TextDetector(cfg.Det)
            except Exception as cpu_exc:
                raise RuntimeError(
                    "failed to initialize RapidOCR detector with ONNX Runtime CUDA "
                    f"({self._cuda_fallback_reason}); CPU fallback failed ({cpu_exc})"
                ) from cpu_exc
        self._max_side_len = cfg.Global.max_side_len
        self._min_side_len = cfg.Global.min_side_len
        self._width_height_ratio = cfg.Global.width_height_ratio
        self._min_height = cfg.Global.min_height
        self._resize_image_within_bounds = resize_image_within_bounds
        self._apply_vertical_padding = apply_vertical_padding
        self._map_boxes_to_original = map_boxes_to_original

    @staticmethod
    def _load_detector_config(parse_params: object, package_dir: Path, params: dict[str, object]) -> object:
        cfg = parse_params.load(package_dir / "config.yaml")
        cfg = parse_params.update_batch(cfg, params)
        if cfg.Global.model_root_dir is None:
            cfg.Global.model_root_dir = package_dir / "models"
        cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
        cfg.Det.model_root_dir = cfg.Global.model_root_dir
        return cfg

    def describe(self) -> str:
        device = "CUDA" if self._use_cuda else "CPU fallback" if self._cuda_fallback_reason else "CPU"
        return f"{self.name} ({device})"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        original_height, original_width = frame.shape[:2]
        input_frame = np.ascontiguousarray(frame)
        input_frame, ratio_height, ratio_width = self._resize_image_within_bounds(
            input_frame,
            self._min_side_len,
            self._max_side_len,
        )
        op_record = {"preprocess": {"ratio_h": ratio_height, "ratio_w": ratio_width}}
        input_frame, op_record = self._apply_vertical_padding(
            input_frame,
            op_record,
            self._width_height_ratio,
            self._min_height,
        )
        result = self._detector(input_frame)
        if getattr(result, "boxes", None) is not None:
            result.boxes = self._map_boxes_to_original(
                result.boxes.copy(),
                op_record,
                original_height,
                original_width,
            )
        return _boxes_from_rapidocr_result(result, min_score=0.35)


def create_deep_text_detector_backend(
    mode: str,
    *,
    language: str,
    device_preference: str,
) -> TextDetectorBackend:
    normalized = normalize_text_detector_mode(mode)
    if normalized == "rapidocr":
        return _create_optional_backend(
            RapidOCRTextDetector,
            language=language,
            device_preference=device_preference,
        )
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


def _rapidocr_detection_language(lang_det_enum: object, language: str) -> object:
    normalized = language.casefold()
    if "eng" in normalized and "tha" not in normalized:
        return _enum_value(lang_det_enum, "EN", "en")
    if "tha" in normalized and "eng" not in normalized:
        return _enum_value(lang_det_enum, "MULTI", "multi")
    return _enum_value(lang_det_enum, "MULTI", "multi")


def _rapidocr_package_data_dir(rapidocr_module: object) -> Path:
    candidates: list[Path] = []
    module_file = getattr(rapidocr_module, "__file__", None)
    if module_file:
        candidates.append(Path(module_file).resolve().parent)

    candidates.extend(root / "rapidocr" for root in application_roots())

    unique_candidates = _unique_paths(candidates)
    for candidate in unique_candidates:
        if (candidate / "config.yaml").is_file() and (candidate / "models").is_dir():
            return candidate

    checked = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(f"RapidOCR package data not found. Checked: {checked}")


def _should_use_rapidocr_cuda(device_preference: str) -> bool:
    return onnxruntime_cuda_available(device_preference)


def _enum_value(enum_class: object, name: str, fallback: str) -> object:
    return getattr(enum_class, name, fallback)


def _quiet_rapidocr_logging() -> None:
    rapidocr_logger = logging.getLogger("RapidOCR")
    rapidocr_logger.setLevel(logging.ERROR)
    for handler in rapidocr_logger.handlers:
        handler.setLevel(logging.ERROR)


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


def _boxes_from_rapidocr_result(result: object, *, min_score: float) -> list[tuple[int, int, int, int]]:
    if isinstance(result, tuple) and result:
        result = result[0]

    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if boxes is not None:
        polygons = _iter_polygons(boxes)
        if scores is not None:
            polygons = [
                polygon
                for polygon, score in zip(polygons, scores, strict=False)
                if _coerce_score(score) >= min_score
            ]
        return _boxes_from_polygons(polygons)

    polygons = _iter_polygons(result)
    return _boxes_from_polygons(polygons)


def _coerce_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


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
