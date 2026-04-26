import numpy as np

from screenlens_detection.text_detectors import (
    _boxes_from_polygons,
    _iter_easyocr_horizontal_boxes,
    normalize_text_detector_mode,
    text_detector_options,
)


def test_text_detector_options_include_classic_and_deep_modes() -> None:
    option_codes = {option.code for option in text_detector_options()}

    assert {"opencv", "paddleocr", "easyocr"}.issubset(option_codes)


def test_text_detector_mode_defaults_to_opencv_for_unknown_values() -> None:
    assert normalize_text_detector_mode(None) == "opencv"
    assert normalize_text_detector_mode("unknown") == "opencv"
    assert normalize_text_detector_mode("EASYOCR") == "easyocr"


def test_easyocr_horizontal_boxes_convert_to_rectangles() -> None:
    assert _iter_easyocr_horizontal_boxes([[[10, 110, 20, 45]]]) == [(10, 20, 100, 25)]


def test_polygons_convert_to_axis_aligned_rectangles() -> None:
    polygon = np.array([[10.2, 20.1], [110.8, 18.9], [112.1, 44.2], [9.7, 45.3]])

    assert _boxes_from_polygons([polygon]) == [(9, 18, 104, 28)]
