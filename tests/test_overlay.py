from screenlens_detection.models import DetectionBox
from screenlens_detection.overlay import overlay_font_pixel_size, overlay_text_for_box, scale_overlay_rect


def test_overlay_prefers_translated_text() -> None:
    box = DetectionBox(
        x=10,
        y=20,
        w=100,
        h=30,
        text="Original text",
        translated_text="Translated text",
    )

    assert overlay_text_for_box(box) == "Translated text"


def test_overlay_falls_back_to_source_text() -> None:
    box = DetectionBox(
        x=10,
        y=20,
        w=100,
        h=30,
        text="Original text",
    )

    assert overlay_text_for_box(box) == "Original text"


def test_overlay_rect_scales_to_overlay_size() -> None:
    box = DetectionBox(x=100, y=80, w=300, h=60)

    assert scale_overlay_rect(
        box,
        overlay_width=960,
        overlay_height=540,
        frame_width=1920,
        frame_height=1080,
    ) == (50, 40, 150, 30)


def test_overlay_font_size_tracks_detected_box_height() -> None:
    assert overlay_font_pixel_size(14) < overlay_font_pixel_size(30)
    assert overlay_font_pixel_size(30) < overlay_font_pixel_size(60)
    assert overlay_font_pixel_size(14) <= 10
