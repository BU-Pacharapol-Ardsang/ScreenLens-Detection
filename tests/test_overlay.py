import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from screenlens_detection.models import DetectionBox, FrameAnalysis, MonitorSpec
from screenlens_detection.overlay import overlay_font_pixel_size, overlay_text_for_box, scale_overlay_rect
from screenlens_detection.overlay_tracker import TrackingFrame


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def test_overlay_tracks_existing_boxes_during_blank_scroll_frame() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )

    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[],
            content_offset_x=0,
            content_offset_y=-12,
            content_motion_confidence=0.20,
        )
    )

    assert [(box.x, box.y, box.text) for box in overlay._overlay_boxes] == [(20, 28, "Translated")]


def test_overlay_clears_blank_frame_when_tracking_confidence_is_low() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )

    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[],
            content_offset_x=0,
            content_offset_y=-12,
            content_motion_confidence=0.01,
        )
    )

    assert overlay._overlay_boxes == []


def test_overlay_keeps_unmatched_tracked_boxes_when_new_frame_has_partial_ocr() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((140, 240, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 240, 140)
    overlay.set_tracking_enabled(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[
                DetectionBox(x=20, y=40, w=80, h=20, text="Original 1", translated_text="Translated 1"),
                DetectionBox(x=20, y=80, w=80, h=20, text="Original 2", translated_text="Translated 2"),
            ],
        )
    )

    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=28, w=80, h=20, text="Original 1", translated_text="Translated 1")],
            content_offset_x=0,
            content_offset_y=-12,
            content_motion_confidence=0.20,
        )
    )

    assert [(box.x, box.y, box.text, box.missing_frames) for box in overlay._overlay_boxes] == [
        (20, 28, "Translated 1", 0),
        (20, 68, "Translated 2", 1),
    ]


def test_overlay_keeps_tracked_boxes_briefly_when_content_is_still() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )

    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[],
            content_offset_x=0,
            content_offset_y=0,
            content_motion_confidence=0.20,
        )
    )

    assert [(box.x, box.y, box.text, box.missing_frames) for box in overlay._overlay_boxes] == [
        (20, 40, "Translated", 1)
    ]


def test_overlay_realtime_tracking_offset_moves_boxes_without_ocr_expiry() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )

    overlay.apply_tracking_offset(0, -8, 0.30)
    overlay.apply_tracking_offset(0, -7, 0.30)

    assert [(box.x, box.y, box.text, box.missing_frames) for box in overlay._overlay_boxes] == [
        (20, 25, "Translated", 0)
    ]


def test_overlay_realtime_tracking_prevents_double_applying_pipeline_offset() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )
    overlay.apply_tracking_offset(0, -15, 0.30)

    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[],
            content_offset_x=0,
            content_offset_y=-200,
            content_motion_confidence=0.90,
        )
    )

    assert [(box.x, box.y, box.text, box.missing_frames) for box in overlay._overlay_boxes] == [
        (20, 25, "Translated", 1)
    ]


def test_overlay_realtime_tracking_clears_after_repeated_lost_frames() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 200, 100)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[DetectionBox(x=20, y=40, w=80, h=20, text="Original", translated_text="Translated")],
        )
    )

    for _ in range(overlay._max_realtime_lost_frames):
        overlay.apply_tracking_offset(0, 0, 0.01)

    assert overlay._overlay_boxes == []


def test_overlay_realtime_tracking_tracks_each_label_locally() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 320, 180)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)

    previous = np.full((180, 320), 255, dtype=np.uint8)
    current = np.full((180, 320), 255, dtype=np.uint8)
    cv2.putText(previous, "TAB ITEM", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(previous, "CONTENT LINE", (82, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(current, "TAB ITEM", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(current, "CONTENT LINE", (82, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

    overlay.apply_tracking_frame(TrackingFrame(previous))
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=np.zeros((180, 320, 3), dtype=np.uint8),
            processed_preview=np.zeros((180, 320, 3), dtype=np.uint8),
            boxes=[
                DetectionBox(x=14, y=14, w=112, h=26, text="Tab", translated_text="Tab translated"),
                DetectionBox(x=80, y=98, w=174, h=30, text="Content", translated_text="Content translated"),
            ],
        )
    )

    overlay.apply_tracking_frame(
        TrackingFrame(current, global_offset_x=0.0, global_offset_y=-36.0, global_confidence=0.95)
    )

    positions = {box.text: (box.x, box.y) for box in overlay._overlay_boxes}
    assert positions["Tab translated"] == (14, 14)
    assert positions["Content translated"] == (80, 62)


def test_overlay_realtime_tracking_handles_downscaled_tracking_frames() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 640, 360)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)

    previous_full = np.full((360, 640), 255, dtype=np.uint8)
    current_full = np.full((360, 640), 255, dtype=np.uint8)
    cv2.putText(previous_full, "CONTENT LINE", (160, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(current_full, "CONTENT LINE", (160, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    previous = cv2.resize(previous_full, (320, 180), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_full, (320, 180), interpolation=cv2.INTER_AREA)

    overlay.apply_tracking_frame(TrackingFrame(previous, frame_scale=0.5))
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=np.zeros((360, 640, 3), dtype=np.uint8),
            processed_preview=np.zeros((360, 640, 3), dtype=np.uint8),
            boxes=[DetectionBox(x=156, y=184, w=230, h=38, text="Content", translated_text="Content translated")],
        )
    )

    overlay.apply_tracking_frame(
        TrackingFrame(current, frame_scale=0.5, global_offset_x=0.0, global_offset_y=-60.0, global_confidence=0.95)
    )

    assert [(box.x, box.y, box.text) for box in overlay._overlay_boxes] == [
        (156, 124, "Content translated")
    ]


def test_overlay_limits_visible_boxes_by_priority() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 800, 600)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    boxes = [
        DetectionBox(x=10, y=index * 18, w=90, h=16, text=f"tiny {index}")
        for index in range(40)
    ]
    boxes.append(
        DetectionBox(
            x=100,
            y=100,
            w=320,
            h=40,
            text="Important translated line",
            translated_text="บรรทัดที่สำคัญ",
        )
    )

    overlay.update_analysis(FrameAnalysis(annotated_frame=frame, processed_preview=frame, boxes=boxes))

    assert len(overlay._overlay_boxes) == overlay._max_visible_overlay_boxes
    assert any(box.text == "บรรทัดที่สำคัญ" for box in overlay._overlay_boxes)


def test_overlay_realtime_tracking_clears_on_probable_scene_change() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 240, 160)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)

    previous = np.zeros((160, 240), dtype=np.uint8)
    current = np.full((160, 240), 255, dtype=np.uint8)
    overlay.apply_tracking_frame(TrackingFrame(previous))
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=np.zeros((160, 240, 3), dtype=np.uint8),
            processed_preview=np.zeros((160, 240, 3), dtype=np.uint8),
            boxes=[DetectionBox(x=20, y=40, w=120, h=24, text="Old", translated_text="เก่า")],
        )
    )

    overlay.apply_tracking_frame(TrackingFrame(current, global_confidence=0.02))

    assert overlay._overlay_boxes == []


def test_overlay_realtime_tracking_uses_local_consensus_for_uncached_labels() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 320, 180)
    overlay._max_local_tracked_boxes = 1
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)

    previous = np.full((180, 320), 255, dtype=np.uint8)
    current = np.full((180, 320), 255, dtype=np.uint8)
    cv2.putText(previous, "LINE ONE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(previous, "LINE TWO", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(current, "LINE ONE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(current, "LINE TWO", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

    overlay.apply_tracking_frame(TrackingFrame(previous))
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=np.zeros((180, 320, 3), dtype=np.uint8),
            processed_preview=np.zeros((180, 320, 3), dtype=np.uint8),
            boxes=[
                DetectionBox(x=18, y=48, w=130, h=30, text="Line one", translated_text="One translated"),
                DetectionBox(x=18, y=103, w=130, h=30, text="Line two", translated_text="Two translated"),
            ],
        )
    )

    overlay.apply_tracking_frame(TrackingFrame(current, global_confidence=0.02))

    positions = {box.text: (box.x, box.y) for box in overlay._overlay_boxes}
    assert positions["One translated"] == (18, 23)
    assert positions["Two translated"] == (18, 78)


def test_overlay_realtime_tracking_does_not_clear_scene_change_when_local_anchor_matches() -> None:
    from screenlens_detection.overlay import TranslationOverlay

    _app()
    overlay = TranslationOverlay()
    overlay._monitor = MonitorSpec(1, "Synthetic", 0, 0, 240, 160)
    overlay.set_tracking_enabled(True)
    overlay.set_realtime_tracking_active(True)

    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (160, 240), dtype=np.uint8)
    current = rng.integers(0, 256, (160, 240), dtype=np.uint8)
    overlay.apply_tracking_frame(TrackingFrame(previous))
    overlay.update_analysis(
        FrameAnalysis(
            annotated_frame=np.zeros((160, 240, 3), dtype=np.uint8),
            processed_preview=np.zeros((160, 240, 3), dtype=np.uint8),
            boxes=[
                DetectionBox(
                    x=72,
                    y=84,
                    w=82,
                    h=24,
                    text="Moving text",
                    translated_text="Moving translated",
                )
            ],
        )
    )

    template_rect = overlay._template_rect_for_box(overlay._overlay_boxes[0], previous.shape, 1.0)
    assert template_rect is not None
    left, top, right, bottom = template_rect
    offset_y = -32
    current[top + offset_y : bottom + offset_y, left:right] = previous[top:bottom, left:right]

    overlay.apply_tracking_frame(TrackingFrame(current, global_confidence=0.02))

    assert [(box.x, box.y, box.text) for box in overlay._overlay_boxes] == [
        (72, 52, "Moving translated")
    ]
