import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from screenlens_detection.text_detectors import (
    _boxes_from_polygons,
    _boxes_from_rapidocr_result,
    _iter_easyocr_horizontal_boxes,
    RapidOCRTextDetector,
    normalize_text_detector_mode,
    text_detector_options,
)


def test_text_detector_options_include_classic_and_deep_modes() -> None:
    option_codes = {option.code for option in text_detector_options()}

    assert {"opencv", "rapidocr", "paddleocr", "easyocr"}.issubset(option_codes)


def test_text_detector_mode_defaults_to_opencv_for_unknown_values() -> None:
    assert normalize_text_detector_mode(None) == "opencv"
    assert normalize_text_detector_mode("unknown") == "opencv"
    assert normalize_text_detector_mode("RapidOCR") == "rapidocr"
    assert normalize_text_detector_mode("EASYOCR") == "easyocr"


def test_easyocr_horizontal_boxes_convert_to_rectangles() -> None:
    assert _iter_easyocr_horizontal_boxes([[[10, 110, 20, 45]]]) == [(10, 20, 100, 25)]


def test_polygons_convert_to_axis_aligned_rectangles() -> None:
    polygon = np.array([[10.2, 20.1], [110.8, 18.9], [112.1, 44.2], [9.7, 45.3]])

    assert _boxes_from_polygons([polygon]) == [(9, 18, 104, 28)]


def test_rapidocr_result_filters_low_confidence_boxes() -> None:
    result = SimpleNamespace(
        boxes=np.array(
            [
                [[10, 20], [110, 20], [110, 45], [10, 45]],
                [[1, 1], [8, 1], [8, 8], [1, 8]],
            ],
            dtype=np.float32,
        ),
        scores=[0.91, 0.12],
    )

    assert _boxes_from_rapidocr_result(result, min_score=0.35) == [(10, 20, 100, 25)]


def test_rapidocr_detector_uses_detection_only_mode(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    created: list[object] = []

    class FakeEnum:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeParseParams:
        @staticmethod
        def load(path: object) -> object:
            return SimpleNamespace(
                Global=SimpleNamespace(
                    model_root_dir=None,
                    max_side_len=2000,
                    min_side_len=30,
                    width_height_ratio=8,
                    min_height=30,
                ),
                EngineConfig={"onnxruntime": SimpleNamespace()},
                Det=SimpleNamespace(engine_type=FakeEnum("onnxruntime")),
            )

        @staticmethod
        def update_batch(cfg: object, params: dict[str, object]) -> object:
            for key, value in params.items():
                parts = key.split(".")
                target = cfg
                for part in parts[:-1]:
                    if isinstance(target, dict):
                        target = target.setdefault(part, SimpleNamespace())
                    else:
                        if not hasattr(target, part):
                            setattr(target, part, SimpleNamespace())
                        target = getattr(target, part)
                if isinstance(target, dict):
                    target[parts[-1]] = value
                else:
                    setattr(target, parts[-1], value)
            return cfg

    class FakeTextDetector:
        def __init__(self, cfg: object) -> None:
            created.append(cfg)

        def __call__(self, image: np.ndarray) -> object:
            calls.append(image.shape)
            return SimpleNamespace(
                boxes=np.array([[[10, 20], [110, 20], [110, 45], [10, 45]]], dtype=np.float32),
                scores=[0.91],
            )

    rapidocr_module = ModuleType("rapidocr")
    rapidocr_module.__file__ = __file__
    rapidocr_module.__path__ = []
    rapidocr_module.EngineType = SimpleNamespace(ONNXRUNTIME=FakeEnum("onnxruntime"))
    rapidocr_module.LangDet = SimpleNamespace(EN=FakeEnum("en"), MULTI=FakeEnum("multi"))
    rapidocr_module.ModelType = SimpleNamespace(MOBILE=FakeEnum("mobile"))
    rapidocr_module.OCRVersion = SimpleNamespace(PPOCRV4=FakeEnum("PP-OCRv4"))
    detector_module = ModuleType("rapidocr.ch_ppocr_det")
    detector_module.TextDetector = FakeTextDetector
    parse_module = ModuleType("rapidocr.utils.parse_parameters")
    parse_module.ParseParams = FakeParseParams
    process_module = ModuleType("rapidocr.utils.process_img")
    process_module.resize_image_within_bounds = lambda image, _min_side, _max_side: (image, 1.0, 1.0)
    process_module.apply_vertical_padding = lambda image, op_record, _ratio, _min_height: (image, op_record)
    process_module.map_boxes_to_original = lambda boxes, _op_record, _height, _width: boxes
    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr_module)
    monkeypatch.setitem(sys.modules, "rapidocr.ch_ppocr_det", detector_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.parse_parameters", parse_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.process_img", process_module)

    detector = RapidOCRTextDetector(language="eng", device_preference="cpu")
    boxes = detector.detect(np.zeros((80, 160, 3), dtype=np.uint8))

    assert boxes == [(10, 20, 100, 25)]
    assert created[0].lang_type.value == "en"
    assert created[0].ocr_version.value == "PP-OCRv4"
    assert calls == [(80, 160, 3)]
