import cv2
import numpy as np

from screenlens_detection.motion import estimate_grayscale_offset
from screenlens_detection.overlay_tracker import OverlayTrackingWorker


def test_estimate_grayscale_offset_tracks_scroll_like_translation() -> None:
    previous = np.zeros((200, 320), dtype=np.uint8)
    cv2.putText(previous, "ScreenLens", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.putText(previous, "Detection", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

    current = np.zeros_like(previous)
    current[:182, :] = previous[18:, :]

    offset_x, offset_y, confidence = estimate_grayscale_offset(previous, current)

    assert abs(offset_x) < 1.0
    assert -19.0 <= offset_y <= -17.0
    assert confidence > 0.50


def test_estimate_grayscale_offset_returns_zero_for_unrelated_frames() -> None:
    previous = np.zeros((120, 240), dtype=np.uint8)
    current = np.full((120, 240), 255, dtype=np.uint8)

    offset_x, offset_y, confidence = estimate_grayscale_offset(previous, current, min_response=0.99)

    assert (offset_x, offset_y) == (0.0, 0.0)
    assert confidence < 0.99


def test_overlay_tracking_frame_is_downscaled_for_realtime_work() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    gray, frame_scale = OverlayTrackingWorker._prepare_tracking_frame(frame)

    assert gray.shape == (360, 640)
    assert frame_scale == 640.0 / 1920.0
