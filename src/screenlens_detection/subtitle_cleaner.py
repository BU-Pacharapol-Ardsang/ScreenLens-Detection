from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class CleanPatch:
    rect: tuple[int, int, int, int]
    image: np.ndarray


def normalize_subtitle_render_mode(mode: str | None) -> str:
    normalized = (mode or "bubble").casefold().strip().replace("-", "_")
    if normalized in {"clean", "clean_patch", "inpaint", "inpainting"}:
        return "clean_patch"
    return "bubble"


def clean_patch_for_box(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    padding_px: int = 8,
    mask_dilate_px: int = 4,
    inpaint_radius: int = 3,
    max_crop_area: int = 120_000,
) -> CleanPatch | None:
    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[2] != 3:
        return None

    crop_rect = _expanded_rect(rect, source.shape, padding_px)
    left, top, width, height = crop_rect
    if width <= 1 or height <= 1:
        return None
    if width * height > max(max_crop_area, 1):
        return None

    crop = np.ascontiguousarray(source[top : top + height, left : left + width]).copy()
    mask = build_subtitle_text_mask(crop, dilate_px=mask_dilate_px)
    mask_pixels = cv2.countNonZero(mask)

    mask_ratio = mask_pixels / max(mask.size, 1)
    if 0.003 <= mask_ratio <= 0.42:
        radius = max(int(inpaint_radius), 1)
        cleaned = cv2.inpaint(crop, mask, radius, cv2.INPAINT_TELEA)
        return CleanPatch(rect=crop_rect, image=cleaned)

    return CleanPatch(rect=crop_rect, image=_soft_background_patch(crop, mask))


def apply_clean_patch_to_frame(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    padding_px: int = 8,
    mask_dilate_px: int = 4,
    inpaint_radius: int = 3,
    max_crop_area: int = 120_000,
) -> bool:
    patch = clean_patch_for_box(
        frame,
        rect,
        padding_px=padding_px,
        mask_dilate_px=mask_dilate_px,
        inpaint_radius=inpaint_radius,
        max_crop_area=max_crop_area,
    )
    if patch is None:
        return False

    left, top, width, height = patch.rect
    frame[top : top + height, left : left + width] = patch.image
    return True


def build_subtitle_text_mask(crop: np.ndarray, *, dilate_px: int = 4) -> np.ndarray:
    source = np.asarray(crop)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    bgr = np.ascontiguousarray(source)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)

    blur_kernel = _local_blur_kernel(gray.shape)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    contrast = cv2.absdiff(gray, blurred)
    contrast_mask = cv2.threshold(contrast, 18, 255, cv2.THRESH_BINARY)[1]

    edges = cv2.Canny(gray, 40, 130)
    edge_mask = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    local_delta = gray.astype(np.int16) - blurred.astype(np.int16)
    bright_text = (
        ((value >= 178) & (saturation <= 150))
        | ((value >= 138) & (saturation >= 58))
        | ((local_delta >= 18) & (value >= 125))
    )
    dark_text = (
        ((value <= 105) & (contrast >= 16))
        | ((local_delta <= -16) & (value <= 175) & (contrast >= 12))
    )
    color_or_value_mask = np.where(bright_text | dark_text, 255, 0).astype(np.uint8)

    candidate = cv2.bitwise_and(color_or_value_mask, cv2.bitwise_or(contrast_mask, edge_mask))
    mask = cv2.bitwise_or(candidate, cv2.bitwise_and(edge_mask, contrast_mask))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask = _suppress_large_components(mask)
    solid_dark_mask = np.where(
        (value <= 110) & ((local_delta <= -12) | (contrast >= 10)),
        255,
        0,
    ).astype(np.uint8)
    mask = cv2.bitwise_or(mask, _suppress_large_components(solid_dark_mask, max_area_ratio=0.72))

    dilation = max(int(dilate_px), 0)
    if dilation:
        kernel_size = (dilation * 2) + 1
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )

    return mask.astype(np.uint8)


def _soft_background_patch(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background = _estimated_background(crop, mask)
    if cv2.countNonZero(mask) <= 0:
        return background

    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0).astype(np.float32) / 255.0
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    blended = (background.astype(np.float32) * alpha) + (crop.astype(np.float32) * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _estimated_background(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    limit = min(height, width)
    if limit < 3:
        return crop.copy()

    kernel = max(min(limit // 2, 41), 9)
    if kernel > limit:
        kernel = limit
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 3:
        return crop.copy()

    median = cv2.medianBlur(crop, kernel)
    blurred = cv2.GaussianBlur(median, (kernel, kernel), 0)
    background = cv2.addWeighted(median, 0.60, blurred, 0.40, 0)

    sample_mask = mask == 0
    if int(np.count_nonzero(sample_mask)) >= 16:
        fill_color = np.median(crop[sample_mask], axis=0)
        fill = np.empty_like(crop)
        fill[:, :] = np.clip(fill_color, 0, 255).astype(np.uint8)
        return cv2.addWeighted(fill, 0.72, background, 0.28, 0)

    return background


def _expanded_rect(
    rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    padding_px: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    frame_height, frame_width = frame_shape[:2]
    pad_x = max(int(padding_px), int(round(width * 0.04)), 2)
    pad_y = max(int(padding_px), int(round(height * 0.22)), 2)
    left = max(int(x) - pad_x, 0)
    top = max(int(y) - pad_y, 0)
    right = min(int(x + width) + pad_x, frame_width)
    bottom = min(int(y + height) + pad_y, frame_height)
    return left, top, max(right - left, 0), max(bottom - top, 0)


def _local_blur_kernel(shape: tuple[int, int]) -> int:
    height, width = shape[:2]
    base = max(min(min(height, width) // 5, 31), 7)
    return base if base % 2 == 1 else base + 1


def _suppress_large_components(mask: np.ndarray, *, max_area_ratio: float = 0.35) -> np.ndarray:
    if cv2.countNonZero(mask) <= 0:
        return mask

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    area_limit = max(int(mask.size * max(min(max_area_ratio, 1.0), 0.01)), 1)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > area_limit:
            continue
        cleaned[labels == label] = 255
    return cleaned
