import cv2
import numpy as np

from screenlens_detection.subtitle_cleaner import (
    apply_clean_patch_to_frame,
    build_subtitle_text_mask,
    clean_patch_for_box,
    normalize_subtitle_render_mode,
)


def test_normalize_subtitle_render_mode_accepts_clean_aliases() -> None:
    assert normalize_subtitle_render_mode("clean-patch") == "clean_patch"
    assert normalize_subtitle_render_mode("inpainting") == "clean_patch"
    assert normalize_subtitle_render_mode("unknown") == "bubble"


def test_subtitle_text_mask_finds_bright_glyphs_without_selecting_whole_crop() -> None:
    frame = _synthetic_subtitle_frame()
    crop = frame[18:58, 32:132]

    mask = build_subtitle_text_mask(crop, dilate_px=2)
    mask_ratio = cv2.countNonZero(mask) / mask.size

    assert mask_ratio > 0.02
    assert mask_ratio < 0.55


def test_clean_patch_reduces_old_subtitle_pixels() -> None:
    frame = _synthetic_subtitle_frame()
    rect = (32, 18, 100, 40)
    before_bright_pixels = _bright_pixels(frame, rect)

    changed = apply_clean_patch_to_frame(frame, rect, padding_px=4, mask_dilate_px=2, inpaint_radius=3)

    assert changed is True
    assert _bright_pixels(frame, rect) < before_bright_pixels * 0.55


def test_clean_patch_reduces_dense_dark_handwriting_on_light_paper() -> None:
    frame = _synthetic_dark_handwriting_frame()
    rect = (12, 8, 260, 86)
    before_dark_pixels = _dark_pixels(frame, rect)

    changed = apply_clean_patch_to_frame(frame, rect, padding_px=4, mask_dilate_px=3, inpaint_radius=3)

    assert changed is True
    assert _dark_pixels(frame, rect) < before_dark_pixels * 0.70


def test_clean_patch_skips_oversized_crops_for_realtime_budget() -> None:
    frame = _synthetic_subtitle_frame(width=400, height=240)

    patch = clean_patch_for_box(frame, (0, 0, 400, 240), max_crop_area=2_000)

    assert patch is None


def _synthetic_subtitle_frame(*, width: int = 180, height: int = 80) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        frame[y, :, 0] = 45 + (y // 4)
        frame[y, :, 1] = 62 + (y // 5)
        frame[y, :, 2] = 82 + (y // 6)
    cv2.putText(frame, "OLD", (38, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    return frame


def _synthetic_dark_handwriting_frame() -> np.ndarray:
    frame = np.full((110, 300, 3), (202, 190, 180), dtype=np.uint8)
    for y in range(10, 96, 28):
        cv2.putText(frame, "SCHOOL ASSIGNMENT", (18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (22, 22, 22), 2, cv2.LINE_AA)
    return frame


def _bright_pixels(frame: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    x, y, width, height = rect
    crop = frame[y : y + height, x : x + width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return int(np.count_nonzero(gray > 190))


def _dark_pixels(frame: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    x, y, width, height = rect
    crop = frame[y : y + height, x : x + width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return int(np.count_nonzero(gray < 80))
