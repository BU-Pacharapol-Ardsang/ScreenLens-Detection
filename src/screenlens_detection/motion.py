from __future__ import annotations

import cv2
import numpy as np


def estimate_grayscale_offset(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    source_scale: float = 1.0,
    max_dimension: int = 360,
    min_response: float = 0.08,
    max_offset_ratio: float = 0.35,
) -> tuple[float, float, float]:
    if previous_gray.shape != current_gray.shape:
        return 0.0, 0.0, 0.0

    height, width = current_gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0, 0.0, 0.0

    resize_scale = min(1.0, max_dimension / max(height, width))
    previous = previous_gray
    current = current_gray
    if resize_scale < 1.0:
        target_size = (max(int(width * resize_scale), 16), max(int(height * resize_scale), 16))
        previous = cv2.resize(previous, target_size, interpolation=cv2.INTER_AREA)
        current = cv2.resize(current, target_size, interpolation=cv2.INTER_AREA)

    previous_float = previous.astype(np.float32)
    current_float = current.astype(np.float32)
    window = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
    (shift_x, shift_y), response = cv2.phaseCorrelate(previous_float, current_float, window)

    if response < min_response:
        return 0.0, 0.0, float(response)

    coordinate_scale = max(source_scale * resize_scale, 1e-6)
    offset_x = shift_x / coordinate_scale
    offset_y = shift_y / coordinate_scale

    source_width = width / max(source_scale, 1e-6)
    source_height = height / max(source_scale, 1e-6)
    if abs(offset_x) > source_width * max_offset_ratio or abs(offset_y) > source_height * max_offset_ratio:
        return 0.0, 0.0, float(response)

    return float(offset_x), float(offset_y), float(response)
