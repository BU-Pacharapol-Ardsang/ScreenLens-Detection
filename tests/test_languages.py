from screenlens_detection.languages import (
    detect_language_code,
    resolve_ocr_language,
    resolve_translation_language,
)
from screenlens_detection.models import DetectionBox


def test_detect_language_code_supports_auto_detection() -> None:
    assert detect_language_code("hello world") == "eng"
    assert detect_language_code("\u0e17\u0e14\u0e2a\u0e2d\u0e1a\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22") == "tha"
    assert detect_language_code("Google \u0e41\u0e1b\u0e25\u0e20\u0e32\u0e29\u0e32") == "mixed"
    assert detect_language_code("12345") == "unknown"


def test_language_resolution_for_ocr_and_translation() -> None:
    assert resolve_ocr_language("auto") == "tha+eng"
    assert resolve_ocr_language("eng") == "eng"
    assert resolve_translation_language("eng") == "en"
    assert resolve_translation_language("tha") == "th"
    assert resolve_translation_language("mixed") == "auto"


def test_detection_box_summary_includes_before_and_after_text() -> None:
    box = DetectionBox(
        x=10,
        y=20,
        w=30,
        h=40,
        text="Google Translate",
        source_language_code="eng",
        source_language_label="English",
        target_language_code="tha",
        target_language_label="Thai",
        translated_text="\u0e01\u0e39\u0e40\u0e01\u0e34\u0e25 \u0e41\u0e1b\u0e25\u0e20\u0e32\u0e29\u0e32",
    )

    summary = box.summary(1)

    assert "[1] x=10, y=20, w=30, h=40" in summary
    assert "source: English" in summary
    assert "target: Thai" in summary
    assert "before: Google Translate" in summary
    assert "after: \u0e01\u0e39\u0e40\u0e01\u0e34\u0e25 \u0e41\u0e1b\u0e25\u0e20\u0e32\u0e29\u0e32" in summary
