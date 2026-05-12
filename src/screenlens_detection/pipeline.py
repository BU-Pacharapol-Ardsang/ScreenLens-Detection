from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from time import perf_counter

import cv2
import numpy as np

from .languages import (
    detect_language_code,
    get_source_language_option,
    get_target_language_option,
    language_label,
)
from .models import DetectionBox, FrameAnalysis, PipelineSettings
from .motion import estimate_grayscale_offset
from .ocr import OCRBackend, OCRFrameResult, OCRResult
from .text_detectors import (
    TextDetectorBackend,
    create_deep_text_detector_backend,
    normalize_text_detector_mode,
)
from .translation import TranslationBackend


@dataclass(slots=True)
class _TrackedTextBox:
    rect: tuple[int, int, int, int]
    stable_frames: int = 1
    missing_frames: int = 0


@dataclass(slots=True, frozen=True)
class _OCRCroppedBox:
    rect: tuple[int, int, int, int]
    crop: np.ndarray
    fingerprint: np.ndarray
    psm: int


@dataclass(slots=True)
class _CachedOCRResult:
    rect: tuple[int, int, int, int]
    fingerprint: np.ndarray
    text: str
    confidence: float | None = None
    ocr_language: str = ""
    psm: int = 7
    translated_text: str = ""
    source_language_code: str = "unknown"
    source_language_label: str = "Unknown"
    target_language_code: str = "tha"
    target_language_label: str = "Thai"
    last_seen_generation: int = 0
    last_ocr_generation: int = 0
    stable_hits: int = 1


@dataclass(slots=True, frozen=True)
class _TranslationBlock:
    indices: tuple[int, ...]
    text: str


class TextDetectionPipeline:
    """Realtime screen-text pipeline using traditional CV and optional OCR."""

    def __init__(
        self,
        settings: PipelineSettings,
        ocr_backend: OCRBackend,
        translation_backend: TranslationBackend,
    ) -> None:
        self.settings = settings
        self.ocr_backend = ocr_backend
        self.translation_backend = translation_backend
        self._recent_translations: list[DetectionBox] = []
        self._recent_translation_lookup: dict[tuple[str, str, str], str] = {}
        self._recent_translation_candidates: list[tuple[DetectionBox, str]] = []
        self._recent_ocr_results: list[_CachedOCRResult] = []
        self._ocr_cache_generation = 0
        self._last_ocr_reuse_count = 0
        self._last_ocr_candidate_count = 0
        self._last_ocr_submitted_count = 0
        self._ocr_box_tracks: list[_TrackedTextBox] = []
        self._scanline_source_boxes: list[tuple[int, int, int, int]] = []
        self._scanline_frame_index = 0
        self._scanline_last_band_index: int | None = None
        self._scanline_last_band_count = 0
        self._scanline_source_shape: tuple[int, int, int] | None = None
        self._scanline_detection_shape: tuple[int, int, int] | None = None
        self._last_hover_region_status = ""
        self._previous_motion_gray: np.ndarray | None = None
        self._current_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._current_scaled_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._active_detection_scale = 1.0
        self._blank_translation_frames = 0
        self._pending_translation_frames = 0
        self._deep_text_detector: TextDetectorBackend | None = None

    def close(self) -> None:
        self.ocr_backend.close()
        self.translation_backend.close()
        if self._deep_text_detector is not None:
            self._deep_text_detector.close()

    def process(
        self,
        frame: np.ndarray,
        *,
        monitor_label: str = "",
        cursor_position: tuple[int, int] | None = None,
    ) -> FrameAnalysis:
        started = perf_counter()
        runtime_debug_enabled = self.settings.runtime_debug_enabled
        timings_ms: dict[str, float] = {}
        timing_checkpoint = started

        detection_frame, detection_scale = self._scale_frame(frame)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "scale_frame")
        self._active_detection_scale = detection_scale
        detection_gray = self._enhance_grayscale(detection_frame)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "enhance_grayscale")
        full_frame_ocr_enabled = self._full_frame_ocr_enabled()
        if full_frame_ocr_enabled:
            self._reset_scanline_state()
            self._last_hover_region_status = ""
            boxes, detection_boxes, line_mask = self._annotate_with_full_frame_ocr(
                detection_frame,
                frame.shape,
                detection_scale,
            )
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "full_frame_ocr")
        elif self._translation_region_mode() == "hover":
            self._reset_scanline_state()
            detection_boxes, line_mask, working_boxes = self._hover_detection_pass(
                detection_frame,
                detection_gray,
                frame.shape,
                detection_scale,
                cursor_position,
            )
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "hover_detection")
        elif self.settings.scanline_roi_enabled:
            detection_boxes, line_mask, working_boxes = self._scanline_detection_pass(
                detection_frame,
                detection_gray,
                frame.shape,
                detection_scale,
            )
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "scanline_detection")
        else:
            self._reset_scanline_state()
            self._last_hover_region_status = ""
            mask = self._build_text_mask(detection_frame, detection_gray)
            line_mask = self._build_line_mask(mask)
            detection_boxes = self._detect_text_boxes(detection_frame, line_mask, mask, detection_gray)
            working_boxes = self._map_boxes_to_source_frame(detection_boxes, frame.shape, detection_scale)
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "opencv_detection")
        ocr_gray = self._source_ocr_grayscale(frame, detection_gray, detection_scale)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "ocr_grayscale")
        if not full_frame_ocr_enabled:
            stable_working_boxes = self._stabilize_ocr_boxes(working_boxes)
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "ocr_box_stability")
            motion_filtered_boxes = self._filter_motion_ocr_boxes(stable_working_boxes, ocr_gray)
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "motion_filter")
            boxes = self._annotate_with_ocr(motion_filtered_boxes, ocr_gray, frame.shape, 1.0)
            if runtime_debug_enabled:
                timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "ocr_annotation")
        motion_offset_x, motion_offset_y, motion_confidence = self._estimate_frame_offset(
            ocr_gray,
            1.0,
        )
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "motion_offset")
        self._current_motion_offset = (motion_offset_x, motion_offset_y, motion_confidence)
        self._current_scaled_motion_offset = self._current_motion_offset
        boxes = self._apply_translations(boxes)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "translation")
        self._remember_ocr_translations(boxes)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "cache_update")
        if self.settings.overlay_tracking_enabled:
            content_offset_x, content_offset_y, content_motion_confidence = self._current_motion_offset
        else:
            content_offset_x, content_offset_y, content_motion_confidence = 0.0, 0.0, 0.0
        self._previous_motion_gray = ocr_gray.copy()
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "state_update")

        annotated = self._draw_annotations(frame.copy(), boxes)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "draw_annotations")
        processed_preview = self._draw_mask_preview(line_mask, detection_boxes, output_shape=frame.shape)
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "draw_mask_preview")
        source_frame = frame.copy()
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "source_frame_copy")
        status = self._status_message()
        ocr_runtime = self.ocr_backend.runtime_diagnostics()
        ocr_available = self.ocr_backend.is_available()
        if runtime_debug_enabled:
            timing_checkpoint = self._record_runtime_timing(timings_ms, timing_checkpoint, "runtime_metadata")

        elapsed = max(perf_counter() - started, 1e-6)
        if runtime_debug_enabled:
            timings_ms["total"] = elapsed * 1000.0
        return FrameAnalysis(
            annotated_frame=annotated,
            processed_preview=processed_preview,
            boxes=boxes,
            source_frame=source_frame,
            status=status,
            ocr_runtime=ocr_runtime,
            fps=1.0 / elapsed,
            ocr_available=ocr_available,
            monitor_label=monitor_label,
            content_offset_x=content_offset_x,
            content_offset_y=content_offset_y,
            content_motion_confidence=content_motion_confidence,
            runtime_timings_ms=timings_ms,
        )

    @staticmethod
    def _record_runtime_timing(timings_ms: dict[str, float], checkpoint: float, stage: str) -> float:
        now = perf_counter()
        timings_ms[stage] = (now - checkpoint) * 1000.0
        return now

    def _scale_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        scale = self._effective_detection_scale()
        if scale == 1.0:
            return frame.copy(), scale
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interpolation)
        return scaled, scale

    def _source_ocr_grayscale(
        self,
        frame: np.ndarray,
        detection_gray: np.ndarray,
        detection_scale: float,
    ) -> np.ndarray:
        if detection_scale == 1.0 and detection_gray.shape[:2] == frame.shape[:2]:
            return detection_gray
        return self._enhance_grayscale(frame)

    @staticmethod
    def _map_boxes_to_source_frame(
        boxes: list[tuple[int, int, int, int]],
        source_shape: tuple[int, int, int],
        scale: float,
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []

        source_height, source_width = source_shape[:2]
        if scale == 1.0:
            return [
                (
                    max(min(x, source_width - 1), 0),
                    max(min(y, source_height - 1), 0),
                    max(min(w, source_width - max(min(x, source_width - 1), 0)), 1),
                    max(min(h, source_height - max(min(y, source_height - 1), 0)), 1),
                )
                for x, y, w, h in boxes
            ]

        mapped: list[tuple[int, int, int, int]] = []
        pad_x = int(np.ceil(2.0 / scale)) if scale < 1.0 else 0
        pad_y = int(np.ceil(6.0 / scale)) if scale < 1.0 else 0
        for x, y, w, h in boxes:
            left = max(int(np.floor(x / scale)) - pad_x, 0)
            top = max(int(np.floor(y / scale)) - pad_y, 0)
            right = min(max(int(np.ceil((x + w) / scale)) + pad_x, left + 1), source_width)
            bottom = min(max(int(np.ceil((y + h) / scale)) + pad_y, top + 1), source_height)
            mapped.append((left, top, right - left, bottom - top))
        return mapped

    @staticmethod
    def _map_boxes_to_detection_frame(
        boxes: list[tuple[int, int, int, int]],
        detection_shape: tuple[int, int, int],
        scale: float,
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []

        detection_height, detection_width = detection_shape[:2]
        mapped: list[tuple[int, int, int, int]] = []
        for x, y, w, h in boxes:
            left = max(min(int(round(x * scale)), detection_width - 1), 0)
            top = max(min(int(round(y * scale)), detection_height - 1), 0)
            right = min(max(int(round((x + w) * scale)), left + 1), detection_width)
            bottom = min(max(int(round((y + h) * scale)), top + 1), detection_height)
            mapped.append((left, top, right - left, bottom - top))
        return mapped

    def _scanline_detection_pass(
        self,
        detection_frame: np.ndarray,
        detection_gray: np.ndarray,
        source_shape: tuple[int, int, int],
        detection_scale: float,
    ) -> tuple[list[tuple[int, int, int, int]], np.ndarray, list[tuple[int, int, int, int]]]:
        if self._scanline_source_shape != source_shape or self._scanline_detection_shape != detection_frame.shape:
            self._reset_scanline_state()
            self._scanline_source_shape = source_shape
            self._scanline_detection_shape = detection_frame.shape

        frame_height = detection_gray.shape[0]
        band_count = self._scanline_band_count(frame_height)
        band_index = self._scanline_frame_index % band_count
        self._scanline_frame_index += 1
        self._scanline_last_band_index = band_index
        self._scanline_last_band_count = band_count

        core_top, core_bottom, scan_top, scan_bottom = self._scanline_band_bounds(
            frame_height,
            band_count,
            band_index,
        )
        roi_frame = detection_frame[scan_top:scan_bottom, :]
        roi_gray = detection_gray[scan_top:scan_bottom, :]
        mask_roi = self._build_text_mask(roi_frame, roi_gray)
        line_mask_roi = self._build_line_mask(mask_roi)

        roi_boxes = self._detect_text_boxes(
            roi_frame,
            line_mask_roi,
            mask_roi,
            roi_gray,
            filter_frame_shape=detection_frame.shape,
        )
        active_detection_boxes = [
            (x, y + scan_top, w, h)
            for x, y, w, h in roi_boxes
            if self._box_center_y_in_span((x, y + scan_top, w, h), core_top, core_bottom)
        ]
        active_source_boxes = self._map_boxes_to_source_frame(
            active_detection_boxes,
            source_shape,
            detection_scale,
        )

        source_core_top = max(int(np.floor(core_top / detection_scale)), 0)
        source_core_bottom = min(int(np.ceil(core_bottom / detection_scale)), source_shape[0])
        self._scanline_source_boxes = self._merge_scanline_source_boxes(
            active_source_boxes,
            source_core_top,
            source_core_bottom,
        )

        line_mask = np.zeros_like(detection_gray)
        line_mask[scan_top:scan_bottom, :] = line_mask_roi
        preview_boxes = self._map_boxes_to_detection_frame(
            self._scanline_source_boxes,
            detection_frame.shape,
            detection_scale,
        )
        return preview_boxes, line_mask, list(self._scanline_source_boxes)

    def _scanline_band_count(self, frame_height: int) -> int:
        if frame_height <= 0:
            return 1
        return min(max(self.settings.scanline_roi_band_count, 2), frame_height)

    def _scanline_band_bounds(
        self,
        frame_height: int,
        band_count: int,
        band_index: int,
    ) -> tuple[int, int, int, int]:
        band_height = max(int(np.ceil(frame_height / max(band_count, 1))), 1)
        core_top = min(band_index * band_height, frame_height)
        core_bottom = min(max(core_top + band_height, core_top + 1), frame_height)
        overlap = max(int(round(band_height * self.settings.scanline_roi_overlap_ratio)), 6)
        scan_top = max(core_top - overlap, 0)
        scan_bottom = min(core_bottom + overlap, frame_height)
        return core_top, core_bottom, scan_top, scan_bottom

    def _merge_scanline_source_boxes(
        self,
        active_boxes: list[tuple[int, int, int, int]],
        source_core_top: int,
        source_core_bottom: int,
    ) -> list[tuple[int, int, int, int]]:
        retained_boxes = [
            box
            for box in self._scanline_source_boxes
            if not self._box_center_y_in_span(box, source_core_top, source_core_bottom)
        ]
        combined = [*active_boxes, *retained_boxes]
        combined.sort(key=self._ocr_candidate_priority, reverse=True)

        deduped: list[tuple[int, int, int, int]] = []
        for box in combined:
            if any(self._intersection_over_union(box, existing) >= 0.78 for existing in deduped):
                continue
            deduped.append(box)

        return self._limit_detected_boxes(deduped)

    @staticmethod
    def _box_center_y_in_span(
        box: tuple[int, int, int, int],
        top: int,
        bottom: int,
    ) -> bool:
        center_y = box[1] + (box[3] / 2.0)
        return top <= center_y < bottom

    def _reset_scanline_state(self) -> None:
        self._scanline_source_boxes = []
        self._scanline_frame_index = 0
        self._scanline_last_band_index = None
        self._scanline_last_band_count = 0
        self._scanline_source_shape = None
        self._scanline_detection_shape = None

    def _translation_region_mode(self) -> str:
        mode = (self.settings.translation_region_mode or "full").casefold().strip()
        if mode in {"hover", "cursor", "hover_region"}:
            return "hover"
        return "full"

    def _hover_detection_pass(
        self,
        detection_frame: np.ndarray,
        detection_gray: np.ndarray,
        source_shape: tuple[int, int, int],
        detection_scale: float,
        cursor_position: tuple[int, int] | None,
    ) -> tuple[list[tuple[int, int, int, int]], np.ndarray, list[tuple[int, int, int, int]]]:
        line_mask = np.zeros_like(detection_gray)
        if cursor_position is None or not self._cursor_inside_source_frame(cursor_position, source_shape):
            self._last_hover_region_status = "hover waiting"
            return [], line_mask, []

        cached_boxes = self._select_hover_source_boxes(
            self._hover_cached_source_boxes(cursor_position),
            cursor_position,
        )
        if cached_boxes:
            self._last_hover_region_status = "hover cache"
            preview_boxes = self._map_boxes_to_detection_frame(cached_boxes, detection_frame.shape, detection_scale)
            self._paint_hover_preview_mask(line_mask, preview_boxes)
            return preview_boxes, line_mask, cached_boxes

        source_left, source_top, source_right, source_bottom = self._hover_source_roi_bounds(
            cursor_position,
            source_shape,
        )
        detection_left = max(int(np.floor(source_left * detection_scale)), 0)
        detection_top = max(int(np.floor(source_top * detection_scale)), 0)
        detection_right = min(int(np.ceil(source_right * detection_scale)), detection_frame.shape[1])
        detection_bottom = min(int(np.ceil(source_bottom * detection_scale)), detection_frame.shape[0])
        if detection_right <= detection_left or detection_bottom <= detection_top:
            self._last_hover_region_status = "hover ROI empty"
            return [], line_mask, []

        roi_frame = detection_frame[detection_top:detection_bottom, detection_left:detection_right]
        roi_gray = detection_gray[detection_top:detection_bottom, detection_left:detection_right]
        mask_roi = self._build_text_mask(roi_frame, roi_gray)
        line_mask_roi = self._build_line_mask(mask_roi)
        roi_boxes = self._detect_text_boxes(
            roi_frame,
            line_mask_roi,
            mask_roi,
            roi_gray,
            filter_frame_shape=detection_frame.shape,
        )
        detection_boxes = [(x + detection_left, y + detection_top, w, h) for x, y, w, h in roi_boxes]
        source_boxes = self._map_boxes_to_source_frame(detection_boxes, source_shape, detection_scale)
        selected_source_boxes = self._select_hover_source_boxes(source_boxes, cursor_position)
        preview_boxes = self._map_boxes_to_detection_frame(
            selected_source_boxes,
            detection_frame.shape,
            detection_scale,
        )

        line_mask[detection_top:detection_bottom, detection_left:detection_right] = line_mask_roi
        if preview_boxes:
            self._last_hover_region_status = (
                f"hover ROI {source_right - source_left}x{source_bottom - source_top}"
            )
        else:
            self._last_hover_region_status = "hover ROI none"
        return preview_boxes, line_mask, selected_source_boxes

    def _hover_cached_source_boxes(self, cursor_position: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        if not self._recent_ocr_results:
            return []

        boxes: list[tuple[int, int, int, int]] = []
        max_age = self._max_ocr_cache_age_frames()
        for cached in self._recent_ocr_results:
            if not cached.text:
                continue
            if self._ocr_cache_generation - cached.last_seen_generation > max_age:
                continue
            for candidate_rect, _motion_adjusted in self._iter_motion_adjusted_rects(cached.rect, scaled=False):
                if self._cursor_box_distance(cursor_position, candidate_rect) <= max(
                    self.settings.hover_region_radius,
                    self._hover_box_margin(),
                ):
                    boxes.append(candidate_rect)
                    break

        return self._dedupe_rects(boxes)

    def _select_hover_source_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        cursor_position: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []

        deduped = self._dedupe_rects(boxes)
        ranked = [
            (self._cursor_box_distance(cursor_position, box), box)
            for box in deduped
        ]
        ranked.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        if not ranked or ranked[0][0] > self._hover_box_margin():
            return []

        anchor = ranked[0][1]
        if self._translation_block_mode() != "strict" or anchor[2] < 240:
            return [anchor]

        return self._hover_strict_geometry_block(deduped, anchor)

    def _hover_strict_geometry_block(
        self,
        boxes: list[tuple[int, int, int, int]],
        anchor: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        anchor_height = max(anchor[3], 1)
        anchor_center_y = anchor[1] + (anchor[3] / 2.0)
        x_tolerance = max(int(round(anchor_height * 1.45)), 36)
        vertical_span = max(self.settings.hover_region_radius, int(round(anchor_height * 5.0)))
        selected: list[tuple[int, int, int, int]] = []

        for box in boxes:
            height_ratio = box[3] / anchor_height
            if height_ratio < 0.70 or height_ratio > 1.35:
                continue
            if abs(box[0] - anchor[0]) > x_tolerance:
                continue
            if self._horizontal_overlap_ratio(box, anchor) < 0.45:
                continue
            center_y = box[1] + (box[3] / 2.0)
            if abs(center_y - anchor_center_y) > vertical_span:
                continue
            selected.append(box)

        if len(selected) <= 1:
            return [anchor]

        selected.sort(key=lambda box: (abs((box[1] + (box[3] / 2.0)) - anchor_center_y), box[1], box[0]))
        selected = selected[:6]
        selected.sort(key=lambda box: (box[1], box[0]))
        return selected

    def _hover_source_roi_bounds(
        self,
        cursor_position: tuple[int, int],
        source_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        frame_height, frame_width = source_shape[:2]
        radius = max(self.settings.hover_region_radius, 32)
        cursor_x, cursor_y = cursor_position
        left = max(cursor_x - radius, 0)
        top = max(cursor_y - radius, 0)
        right = min(cursor_x + radius, frame_width)
        bottom = min(cursor_y + radius, frame_height)
        return left, top, right, bottom

    def _hover_box_margin(self) -> int:
        return max(self.settings.hover_box_margin, 8)

    @staticmethod
    def _cursor_inside_source_frame(
        cursor_position: tuple[int, int],
        source_shape: tuple[int, int, int],
    ) -> bool:
        x, y = cursor_position
        return 0 <= x < source_shape[1] and 0 <= y < source_shape[0]

    @staticmethod
    def _cursor_box_distance(
        cursor_position: tuple[int, int],
        box: tuple[int, int, int, int],
    ) -> float:
        cursor_x, cursor_y = cursor_position
        x, y, w, h = box
        dx = max(x - cursor_x, 0, cursor_x - (x + w))
        dy = max(y - cursor_y, 0, cursor_y - (y + h))
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _dedupe_rects(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        deduped: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if box in deduped:
                continue
            deduped.append(box)
        return deduped

    @staticmethod
    def _paint_hover_preview_mask(
        line_mask: np.ndarray,
        preview_boxes: list[tuple[int, int, int, int]],
    ) -> None:
        for x, y, w, h in preview_boxes:
            cv2.rectangle(line_mask, (x, y), (x + w, y + h), 255, -1)

    def _enhance_grayscale(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=(self.settings.clahe_grid_size, self.settings.clahe_grid_size),
        )
        enhanced = clahe.apply(gray)

        kernel_size = self._ensure_odd(self.settings.gaussian_kernel_size)
        if kernel_size > 1:
            enhanced = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
        return enhanced

    def _build_text_mask(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        block_size = self._ensure_odd(self._scaled_detection_length(self.settings.threshold_block_size, minimum=11))
        threshold_c = self.settings.threshold_c

        dark_text_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            threshold_c,
        )
        light_text_mask = cv2.adaptiveThreshold(
            255 - gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            threshold_c,
        )
        polarity_mask = cv2.bitwise_or(dark_text_mask, light_text_mask)

        stroke_mask = self._build_stroke_response_mask(gray)
        local_contrast_mask = self._build_local_contrast_mask(gray, block_size)
        edge_mask = self._build_edge_response_mask(gray)
        feature_mask = cv2.bitwise_or(stroke_mask, edge_mask)
        contrast_or_edge = cv2.bitwise_or(local_contrast_mask, edge_mask)

        adaptive_strokes = cv2.bitwise_and(polarity_mask, contrast_or_edge)
        feature_strokes = cv2.bitwise_and(feature_mask, cv2.bitwise_or(polarity_mask, local_contrast_mask))
        combined = cv2.bitwise_or(adaptive_strokes, feature_strokes)
        if cv2.countNonZero(combined) < max(
            self._scaled_detection_area(180, minimum=60),
            cv2.countNonZero(polarity_mask) // 16,
        ):
            combined = cv2.bitwise_or(
                cv2.bitwise_and(polarity_mask, local_contrast_mask),
                stroke_mask,
            )

        cleaned = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        overlay_mask = self._build_overlay_text_mask(
            frame,
            gray,
            edge_mask=edge_mask,
            local_contrast_mask=local_contrast_mask,
        )
        return cv2.bitwise_or(self._suppress_large_mask_components(cleaned), overlay_mask)

    def _build_overlay_text_mask(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        *,
        edge_mask: np.ndarray,
        local_contrast_mask: np.ndarray,
    ) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[:2] != gray.shape[:2]:
            return np.zeros_like(gray)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _hue, saturation, value = cv2.split(hsv)
        bright_mask = cv2.inRange(value, 150, 255)
        low_saturation_mask = cv2.inRange(saturation, 0, 130)
        white_ui_mask = cv2.bitwise_and(bright_mask, low_saturation_mask)

        saturated_mask = cv2.bitwise_and(cv2.inRange(value, 120, 255), cv2.inRange(saturation, 80, 255))
        color_ui_mask = cv2.bitwise_and(saturated_mask, local_contrast_mask)
        overlay_mask = cv2.bitwise_or(white_ui_mask, color_ui_mask)
        overlay_mask = cv2.bitwise_and(overlay_mask, edge_mask)
        overlay_mask = cv2.morphologyEx(
            overlay_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
        )
        overlay_mask = cv2.morphologyEx(
            overlay_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        return self._suppress_large_mask_components(overlay_mask)

    def _build_stroke_response_mask(self, gray: np.ndarray) -> np.ndarray:
        horizontal_width = max(
            self._scaled_detection_length(self.settings.morphology_width, minimum=5),
            self._scaled_detection_length(9, minimum=5),
        )
        horizontal_height = max(
            self._scaled_detection_length(self.settings.morphology_height, minimum=2),
            2,
        )
        compact_size = self._scaled_detection_length(5, minimum=3)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_width, horizontal_height),
        )
        compact_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (compact_size, compact_size))
        bright_response = cv2.max(
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, horizontal_kernel),
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, compact_kernel),
        )
        dark_response = cv2.max(
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, horizontal_kernel),
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, compact_kernel),
        )
        response = cv2.max(bright_response, dark_response)
        return self._threshold_response_mask(response, minimum_threshold=8, std_multiplier=0.45)

    def _build_local_contrast_mask(self, gray: np.ndarray, block_size: int) -> np.ndarray:
        window = max(block_size, self._scaled_detection_length(15, minimum=9))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window, window))
        local_max = cv2.dilate(gray, kernel)
        local_min = cv2.erode(gray, kernel)
        contrast = cv2.absdiff(local_max, local_min)
        return self._threshold_response_mask(contrast, minimum_threshold=16, std_multiplier=0.35)

    def _build_edge_response_mask(self, gray: np.ndarray) -> np.ndarray:
        gradient = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        gradient_mask = self._threshold_response_mask(gradient, minimum_threshold=10, std_multiplier=0.50)
        edges = cv2.Canny(gray, 40, 130)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        return cv2.bitwise_or(gradient_mask, edges)

    @staticmethod
    def _threshold_response_mask(
        response: np.ndarray,
        *,
        minimum_threshold: int,
        std_multiplier: float,
    ) -> np.ndarray:
        if response.size == 0:
            return np.zeros_like(response)

        mean, stddev = cv2.meanStdDev(response)
        adaptive_threshold = int(float(mean[0][0]) + (float(stddev[0][0]) * std_multiplier))
        threshold = max(minimum_threshold, adaptive_threshold)
        _, statistical_mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
        _, otsu_mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if cv2.countNonZero(otsu_mask) <= response.size * 0.30:
            return cv2.bitwise_or(statistical_mask, otsu_mask)
        return statistical_mask

    @staticmethod
    def _suppress_large_mask_components(mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return mask

        frame_area = mask.shape[0] * mask.shape[1]
        max_component_area = max(int(frame_area * 0.025), 12_000)
        keep_values = np.zeros(component_count, dtype=np.uint8)
        for component_index in range(1, component_count):
            x, y, w, h, area = stats[component_index]
            bounding_area = max(w * h, 1)
            fill_ratio = area / bounding_area
            aspect_ratio = w / max(h, 1)
            text_sized_component = h <= 96 and aspect_ratio >= 1.1
            if area > max_component_area and fill_ratio > 0.16 and not text_sized_component:
                continue
            if h > mask.shape[0] * 0.35 and w > mask.shape[1] * 0.06:
                continue
            keep_values[component_index] = 255
        return keep_values[labels]

    def _build_line_mask(self, mask: np.ndarray) -> np.ndarray:
        dense_mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    self._scaled_detection_length(self.settings.morphology_width + 2, minimum=11),
                    self._scaled_detection_length(13, minimum=11),
                ),
                max(self._scaled_detection_length(self.settings.morphology_height, minimum=2), 2),
            ),
        )

        line_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        return self._suppress_large_line_components(line_mask)

    @staticmethod
    def _suppress_large_line_components(mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return mask

        height, width = mask.shape[:2]
        frame_area = height * width
        keep_values = np.zeros(component_count, dtype=np.uint8)
        for component_index in range(1, component_count):
            x, y, w, h, area = stats[component_index]
            aspect_ratio = w / max(h, 1)
            if area > frame_area * 0.12:
                continue
            if w > width * 0.85 and h > height * 0.08:
                continue
            if h > height * 0.30 and aspect_ratio < 4.0:
                continue
            keep_values[component_index] = 255
        return keep_values[labels]

    def _detect_text_boxes(
        self,
        scaled_frame: np.ndarray,
        line_mask: np.ndarray,
        text_mask: np.ndarray,
        enhanced_gray: np.ndarray,
        *,
        filter_frame_shape: tuple[int, int, int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        threshold_shape = filter_frame_shape if filter_frame_shape is not None else scaled_frame.shape
        mode = normalize_text_detector_mode(self.settings.text_detector_mode)
        if mode == "opencv":
            return self._extract_text_boxes(line_mask, text_mask, enhanced_gray, threshold_shape)

        detector = self._ensure_deep_text_detector(mode)
        if not detector.is_available():
            return []

        detected_boxes = detector.detect(scaled_frame)
        return self._filter_detector_boxes(detected_boxes, threshold_shape)

    def _ensure_deep_text_detector(self, mode: str) -> TextDetectorBackend:
        if self._deep_text_detector is None:
            self._deep_text_detector = create_deep_text_detector_backend(
                mode,
                language=self.settings.ocr_language,
                device_preference=self.settings.ocr_device_preference,
            )
        return self._deep_text_detector

    def _filter_detector_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        frame_height, frame_width = frame_shape[:2]
        frame_area = max(frame_height * frame_width, 1)
        max_box_height = max(
            int(frame_height * self.settings.max_box_height_ratio),
            self._scaled_detection_length(72, minimum=36),
        )
        min_contour_area = self._scaled_detection_area(self.settings.min_contour_area, minimum=24)
        min_box_width = self._scaled_detection_length(self.settings.min_box_width, minimum=8)
        min_box_height = self._scaled_detection_length(self.settings.min_box_height, minimum=4)

        candidates: list[tuple[int, int, int, int]] = []
        for x, y, w, h in boxes:
            left = max(int(x), 0)
            top = max(int(y), 0)
            right = min(max(int(x + w), left + 1), frame_width)
            bottom = min(max(int(y + h), top + 1), frame_height)
            clipped_w = right - left
            clipped_h = bottom - top
            area = clipped_w * clipped_h
            aspect_ratio = clipped_w / max(clipped_h, 1)

            if area < min_contour_area:
                continue
            if area > int(frame_area * 0.20):
                continue
            if clipped_w < min_box_width or clipped_h < min_box_height:
                continue
            if clipped_h > max_box_height:
                continue
            if not 0.6 <= aspect_ratio <= 60.0:
                continue

            candidates.append((left, top, clipped_w, clipped_h))

        return self._limit_detected_boxes(self._merge_text_boxes(candidates))

    def _extract_text_boxes(
        self,
        line_mask: np.ndarray,
        text_mask: np.ndarray,
        enhanced_gray: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
        max_box_height = max(
            int(frame_shape[0] * self.settings.max_box_height_ratio),
            self._scaled_detection_length(72, minimum=36),
        )
        frame_area = frame_shape[0] * frame_shape[1]
        min_contour_area = self._scaled_detection_area(self.settings.min_contour_area, minimum=24)
        min_box_width = self._scaled_detection_length(self.settings.min_box_width, minimum=8)
        min_box_height = self._scaled_detection_length(self.settings.min_box_height, minimum=4)
        tall_box_threshold = self._scaled_detection_length(120, minimum=40)

        candidates: list[tuple[int, int, int, int]] = []
        for component_index in range(1, component_count):
            x, y, w, h, _component_area = stats[component_index]
            area = w * h
            aspect_ratio = w / max(h, 1)

            if area < min_contour_area:
                continue
            if area > int(frame_area * 0.08):
                continue
            if h > tall_box_threshold and area > int(frame_area * 0.035):
                continue
            if w < min_box_width or h < min_box_height:
                continue
            if h > max_box_height:
                continue
            if not 1.1 <= aspect_ratio <= 45.0:
                continue

            text_roi = text_mask[y : y + h, x : x + w]
            if text_roi.size == 0:
                continue

            foreground_ratio = cv2.countNonZero(text_roi) / max(area, 1)
            if not 0.03 <= foreground_ratio <= 0.82:
                continue

            gray_roi = enhanced_gray[y : y + h, x : x + w]
            edge_density = cv2.countNonZero(cv2.Canny(gray_roi, 50, 150)) / max(area, 1)
            if edge_density < 0.025:
                continue
            if edge_density > 0.48 and foreground_ratio > 0.22:
                continue

            sub_component_count, _sub_labels, sub_stats, _sub_centroids = cv2.connectedComponentsWithStats(
                text_roi,
                connectivity=8,
            )
            text_component_count = sub_component_count - 1
            foreground_pixels = max(cv2.countNonZero(text_roi), 1)
            largest_component_area = int(np.max(sub_stats[1:, cv2.CC_STAT_AREA])) if text_component_count > 0 else 0
            largest_component_ratio = largest_component_area / foreground_pixels
            if largest_component_ratio > 0.82 and aspect_ratio > 2.0:
                continue

            active_columns = int(np.count_nonzero(np.any(text_roi > 0, axis=0)))
            active_rows = int(np.count_nonzero(np.any(text_roi > 0, axis=1)))
            column_coverage = active_columns / max(w, 1)
            row_coverage = active_rows / max(h, 1)
            if aspect_ratio > 3.0 and column_coverage < 0.18:
                continue
            if row_coverage < 0.22:
                continue

            if text_component_count < 3 and aspect_ratio > 3.5:
                continue

            candidates.append((x, y, w, h))

        return self._limit_detected_boxes(self._merge_text_boxes(candidates))

    def _limit_detected_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if len(boxes) > self.settings.max_boxes:
            boxes = sorted(boxes, key=self._ocr_candidate_priority, reverse=True)[: self.settings.max_boxes]
        boxes.sort(key=lambda box: (box[1], box[0]))
        return boxes

    def _full_frame_ocr_enabled(self) -> bool:
        return (
            self.settings.ocr_enabled
            and self.ocr_backend.is_available()
            and self.ocr_backend.supports_full_frame()
        )

    def _annotate_with_full_frame_ocr(
        self,
        detection_frame: np.ndarray,
        original_shape: tuple[int, int, int],
        scale: float,
    ) -> tuple[list[DetectionBox], list[tuple[int, int, int, int]], np.ndarray]:
        self._ocr_cache_generation += 1
        self._last_ocr_reuse_count = 0
        self._last_ocr_candidate_count = 0
        self._last_ocr_submitted_count = 0

        frame_results = self.ocr_backend.recognize_frame(
            detection_frame,
            language=self.settings.ocr_language,
        )
        filtered_results = self._filter_full_frame_ocr_results(frame_results, detection_frame.shape)
        self._last_ocr_candidate_count = len(filtered_results)
        self._last_ocr_submitted_count = len(filtered_results)

        preview_boxes: list[tuple[int, int, int, int]] = []
        detected_boxes: list[DetectionBox] = []
        for result in filtered_results:
            preview_boxes.append(result.rect)
            text = self._normalize_recognized_text(result.text)
            confidence = result.confidence
            if text and not self._is_usable_text(text, confidence):
                text = ""
                confidence = None
            if not text:
                continue
            detected_boxes.append(
                self._detection_box_from_frame_ocr_result(
                    result,
                    original_shape=original_shape,
                    scale=scale,
                    text=text,
                    confidence=confidence,
                )
            )

        line_mask = np.zeros(detection_frame.shape[:2], dtype=np.uint8)
        for x, y, w, h in preview_boxes:
            cv2.rectangle(line_mask, (x, y), (x + w, y + h), 255, -1)

        detected_boxes.sort(key=lambda box: (box.y, box.x))
        return detected_boxes, preview_boxes, line_mask

    def _filter_full_frame_ocr_results(
        self,
        results: list[OCRFrameResult],
        frame_shape: tuple[int, ...],
    ) -> list[OCRFrameResult]:
        frame_height, frame_width = frame_shape[:2]
        frame_area = max(frame_height * frame_width, 1)
        max_box_height = max(
            int(frame_height * self.settings.max_box_height_ratio),
            self._scaled_detection_length(72, minimum=36),
        )
        min_contour_area = self._scaled_detection_area(self.settings.min_contour_area, minimum=24)
        min_box_width = self._scaled_detection_length(self.settings.min_box_width, minimum=8)
        min_box_height = self._scaled_detection_length(self.settings.min_box_height, minimum=4)

        filtered: list[OCRFrameResult] = []
        for result in results:
            x, y, w, h = result.rect
            left = max(int(x), 0)
            top = max(int(y), 0)
            right = min(max(int(x + w), left + 1), frame_width)
            bottom = min(max(int(y + h), top + 1), frame_height)
            clipped_w = right - left
            clipped_h = bottom - top
            area = clipped_w * clipped_h
            aspect_ratio = clipped_w / max(clipped_h, 1)

            if area < min_contour_area:
                continue
            if area > int(frame_area * 0.20):
                continue
            if clipped_w < min_box_width or clipped_h < min_box_height:
                continue
            if clipped_h > max_box_height:
                continue
            if not 0.4 <= aspect_ratio <= 80.0:
                continue

            filtered.append(
                OCRFrameResult(
                    rect=(left, top, clipped_w, clipped_h),
                    text=result.text,
                    confidence=result.confidence,
                )
            )

        filtered = self._limit_full_frame_ocr_results(filtered)
        filtered.sort(key=lambda result: (result.rect[1], result.rect[0]))
        return filtered

    def _limit_full_frame_ocr_results(self, results: list[OCRFrameResult]) -> list[OCRFrameResult]:
        if len(results) <= self.settings.max_boxes:
            return results
        return sorted(
            results,
            key=lambda result: self._ocr_candidate_priority(result.rect),
            reverse=True,
        )[: self.settings.max_boxes]

    def _detection_box_from_frame_ocr_result(
        self,
        result: OCRFrameResult,
        *,
        original_shape: tuple[int, int, int],
        scale: float,
        text: str,
        confidence: float | None,
    ) -> DetectionBox:
        x, y, w, h = result.rect
        mapped_x = int(x / scale)
        mapped_y = int(y / scale)
        mapped_w = int(w / scale)
        mapped_h = int(h / scale)
        mapped_w = min(mapped_w, original_shape[1] - mapped_x)
        mapped_h = min(mapped_h, original_shape[0] - mapped_y)

        source_language_code, source_language_label = self._resolve_source_language(text)
        target_language = get_target_language_option(self.settings.target_language_code)
        return DetectionBox(
            x=max(mapped_x, 0),
            y=max(mapped_y, 0),
            w=max(mapped_w, 1),
            h=max(mapped_h, 1),
            text=text,
            source_language_code=source_language_code,
            source_language_label=source_language_label,
            target_language_code=target_language.code,
            target_language_label=target_language.label,
            confidence=confidence,
        )

    def _annotate_with_ocr(
        self,
        working_boxes: list[tuple[int, int, int, int]],
        enhanced_gray: np.ndarray,
        original_shape: tuple[int, int, int],
        scale: float,
    ) -> list[DetectionBox]:
        self._ocr_cache_generation += 1
        self._last_ocr_reuse_count = 0
        self._last_ocr_candidate_count = 0
        self._last_ocr_submitted_count = 0
        ocr_attempted = self.settings.ocr_enabled and self.ocr_backend.is_available()

        if not ocr_attempted:
            ocr_candidates = self._prepare_ocr_candidates(
                self._select_ocr_boxes(working_boxes),
                enhanced_gray,
            )
            self._last_ocr_candidate_count = len(ocr_candidates)
            return [
                self._detection_box_from_ocr_candidate(
                    candidate,
                    original_shape=original_shape,
                    scale=scale,
                    text="",
                    confidence=None,
                    cached_result=None,
                )
                for candidate in ocr_candidates
            ]

        ocr_candidates = self._prepare_ocr_candidates(working_boxes, enhanced_gray)
        self._last_ocr_candidate_count = len(ocr_candidates)
        cached_items: list[tuple[_OCRCroppedBox, _CachedOCRResult]] = []
        pending_candidates: list[_OCRCroppedBox] = []
        used_cache_ids: set[int] = set()

        for candidate in ocr_candidates:
            cached_result = self._find_cached_ocr_result(candidate, used_cache_ids)
            if cached_result is None:
                pending_candidates.append(candidate)
                continue

            used_cache_ids.add(id(cached_result))
            self._touch_cached_ocr_result(cached_result, candidate)
            cached_items.append((candidate, cached_result))

        selected_pending_candidates = self._select_pending_ocr_candidates(pending_candidates)
        selected_pending_candidates.sort(key=lambda candidate: (candidate.rect[1], candidate.rect[0]))
        self._last_ocr_reuse_count = len(cached_items)
        self._last_ocr_submitted_count = len(selected_pending_candidates)

        pending_results: list[OCRResult] = []
        if selected_pending_candidates:
            pending_results = self.ocr_backend.recognize_batch(
                [self.ocr_backend.prepare_image(candidate.crop) for candidate in selected_pending_candidates],
                language=self.settings.ocr_language,
                psms=[candidate.psm for candidate in selected_pending_candidates],
            )

        pending_result_by_id = {
            id(candidate): result
            for candidate, result in zip(selected_pending_candidates, pending_results, strict=False)
        }
        ocr_cache_updates: list[_CachedOCRResult] = []
        detected_boxes: list[DetectionBox] = []

        for candidate, cached_result in cached_items:
            detected_boxes.append(
                self._detection_box_from_ocr_candidate(
                    candidate,
                    original_shape=original_shape,
                    scale=scale,
                    text=cached_result.text,
                    confidence=cached_result.confidence,
                    cached_result=cached_result,
                )
            )

        for candidate in selected_pending_candidates:
            ocr_result = pending_result_by_id.get(id(candidate), OCRResult())
            text = self._normalize_recognized_text(ocr_result.text)
            confidence = ocr_result.confidence
            if text and not self._is_usable_text(text, confidence):
                text = ""
                confidence = None

            cached_result = None
            if text:
                source_language_code, source_language_label = self._resolve_source_language(text)
                target_language = get_target_language_option(self.settings.target_language_code)
                cached_result = _CachedOCRResult(
                    rect=candidate.rect,
                    fingerprint=candidate.fingerprint,
                    text=text,
                    confidence=confidence,
                    ocr_language=self.settings.ocr_language,
                    psm=candidate.psm,
                    source_language_code=source_language_code,
                    source_language_label=source_language_label,
                    target_language_code=target_language.code,
                    target_language_label=target_language.label,
                    last_seen_generation=self._ocr_cache_generation,
                    last_ocr_generation=self._ocr_cache_generation,
                )
                ocr_cache_updates.append(cached_result)

            detected_boxes.append(
                self._detection_box_from_ocr_candidate(
                    candidate,
                    original_shape=original_shape,
                    scale=scale,
                    text=text,
                    confidence=confidence,
                    cached_result=cached_result,
                )
            )

        if ocr_cache_updates:
            self._remember_ocr_results(ocr_cache_updates)
        else:
            self._prune_ocr_results()

        detected_boxes.sort(key=lambda box: (box.y, box.x))
        return detected_boxes

    def _prepare_ocr_candidates(
        self,
        boxes: list[tuple[int, int, int, int]],
        enhanced_gray: np.ndarray,
    ) -> list[_OCRCroppedBox]:
        candidates: list[_OCRCroppedBox] = []
        for x, y, w, h in boxes:
            if self._should_skip_ocr_candidate((x, y, w, h), enhanced_gray.shape):
                continue

            pad_x = max(int(w * 0.04), 4)
            pad_y = max(int(h * 0.25), 4)
            crop_x1 = max(x - pad_x, 0)
            crop_y1 = max(y - pad_y, 0)
            crop_x2 = min(x + w + pad_x, enhanced_gray.shape[1])
            crop_y2 = min(y + h + pad_y, enhanced_gray.shape[0])
            crop = enhanced_gray[crop_y1:crop_y2, crop_x1:crop_x2]
            candidates.append(
                _OCRCroppedBox(
                    rect=(x, y, w, h),
                    crop=crop,
                    fingerprint=self._ocr_crop_fingerprint(crop),
                    psm=self._resolve_psm(w, h),
                )
            )
        return candidates

    def _select_pending_ocr_candidates(self, candidates: list[_OCRCroppedBox]) -> list[_OCRCroppedBox]:
        limit = max(self.settings.max_ocr_boxes_per_frame, 1)
        if len(candidates) <= limit:
            return list(candidates)

        return sorted(
            candidates,
            key=lambda candidate: self._ocr_candidate_priority(candidate.rect),
            reverse=True,
        )[:limit]

    def _detection_box_from_ocr_candidate(
        self,
        candidate: _OCRCroppedBox,
        *,
        original_shape: tuple[int, int, int],
        scale: float,
        text: str,
        confidence: float | None,
        cached_result: _CachedOCRResult | None,
    ) -> DetectionBox:
        x, y, w, h = candidate.rect
        mapped_x = int(x / scale)
        mapped_y = int(y / scale)
        mapped_w = int(w / scale)
        mapped_h = int(h / scale)

        mapped_w = min(mapped_w, original_shape[1] - mapped_x)
        mapped_h = min(mapped_h, original_shape[0] - mapped_y)
        target_language = get_target_language_option(self.settings.target_language_code)

        if cached_result is not None and cached_result.text == text and cached_result.source_language_code != "unknown":
            source_language_code = cached_result.source_language_code
            source_language_label = cached_result.source_language_label
        else:
            source_language_code, source_language_label = self._resolve_source_language(text)

        translated_text = ""
        if (
            cached_result is not None
            and cached_result.text == text
            and cached_result.target_language_code == target_language.code
        ):
            translated_text = cached_result.translated_text

        return DetectionBox(
            x=max(mapped_x, 0),
            y=max(mapped_y, 0),
            w=max(mapped_w, 1),
            h=max(mapped_h, 1),
            text=text,
            translated_text=translated_text,
            source_language_code=source_language_code,
            source_language_label=source_language_label,
            target_language_code=target_language.code,
            target_language_label=target_language.label,
            confidence=confidence,
        )

    @staticmethod
    def _should_skip_ocr_candidate(
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> bool:
        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return False

        x, y, w, h = box
        area = w * h
        frame_area = frame_width * frame_height
        aspect_ratio = w / max(h, 1)
        near_top = y <= int(frame_height * 0.12)
        broad_strip = w >= int(frame_width * 0.55) and h <= int(frame_height * 0.12)
        toolbar_sized = 26 <= h <= int(frame_height * 0.11) and aspect_ratio >= 9.0
        if near_top and broad_strip and toolbar_sized:
            return True

        if y <= int(frame_height * 0.16) and area >= int(frame_area * 0.045) and aspect_ratio >= 10.0:
            return True

        return False

    def _find_cached_ocr_result(
        self,
        candidate: _OCRCroppedBox,
        used_cache_ids: set[int],
    ) -> _CachedOCRResult | None:
        best_match: _CachedOCRResult | None = None
        best_score = 0.0

        for cached in self._recent_ocr_results:
            if id(cached) in used_cache_ids:
                continue
            if cached.ocr_language and cached.ocr_language != self.settings.ocr_language:
                continue
            if cached.psm != candidate.psm:
                continue

            for cached_rect, motion_adjusted in self._iter_ocr_cache_rects(cached.rect):
                overlap = self._intersection_over_union(candidate.rect, cached_rect)
                minimum_overlap = 0.68 if motion_adjusted else 0.82
                if overlap < minimum_overlap:
                    continue

                difference = self._ocr_fingerprint_difference(candidate.fingerprint, cached.fingerprint)
                difference_limit = 8.5 if motion_adjusted else 5.5
                if difference > difference_limit:
                    continue

                proximity = self._rect_center_proximity(candidate.rect, cached_rect)
                size_similarity = self._rect_size_similarity(candidate.rect, cached_rect)
                fingerprint_score = max(1.0 - (difference / max(difference_limit, 1e-6)), 0.0)
                score = (
                    (overlap * 0.48)
                    + (fingerprint_score * 0.34)
                    + (proximity * 0.10)
                    + (size_similarity * 0.08)
                    + min(cached.stable_hits * 0.01, 0.08)
                )
                if motion_adjusted:
                    score += 0.04
                if score > best_score:
                    best_match = cached
                    best_score = score

        return best_match

    def _iter_ocr_cache_rects(
        self,
        rect: tuple[int, int, int, int],
    ) -> list[tuple[tuple[int, int, int, int], bool]]:
        variants = self._iter_motion_adjusted_rects(rect, scaled=False)
        if self._current_motion_offset[2] >= 0.06 or self._current_scaled_motion_offset[2] < 0.06:
            return variants

        seen = {candidate for candidate, _motion_adjusted in variants}
        for candidate, motion_adjusted in self._iter_motion_adjusted_rects(rect, scaled=True):
            if candidate in seen:
                continue
            variants.append((candidate, motion_adjusted))
            seen.add(candidate)
        return variants

    def _touch_cached_ocr_result(self, cached: _CachedOCRResult, candidate: _OCRCroppedBox) -> None:
        cached.rect = candidate.rect
        cached.fingerprint = candidate.fingerprint
        cached.last_seen_generation = self._ocr_cache_generation
        cached.stable_hits += 1

    def _remember_ocr_results(self, updates: list[_CachedOCRResult]) -> None:
        if not updates:
            return

        for update in updates:
            update.last_seen_generation = self._ocr_cache_generation
            update.last_ocr_generation = self._ocr_cache_generation

        remembered = [*updates, *self._recent_ocr_results]
        deduped: list[_CachedOCRResult] = []
        for candidate in remembered:
            if not candidate.text:
                continue
            if self._ocr_cache_generation - candidate.last_seen_generation > self._max_ocr_cache_age_frames():
                continue
            if any(
                existing.text == candidate.text
                and existing.ocr_language == candidate.ocr_language
                and existing.psm == candidate.psm
                and self._intersection_over_union(existing.rect, candidate.rect) >= 0.92
                for existing in deduped
            ):
                continue
            deduped.append(candidate)

        limit = self._max_ocr_cache_entries()
        self._recent_ocr_results = deduped[:limit]

    def _prune_ocr_results(self) -> None:
        max_age = self._max_ocr_cache_age_frames()
        self._recent_ocr_results = [
            cached
            for cached in self._recent_ocr_results
            if self._ocr_cache_generation - cached.last_seen_generation <= max_age
        ][: self._max_ocr_cache_entries()]

    def _remember_ocr_translations(self, boxes: list[DetectionBox]) -> None:
        if not boxes or not self._recent_ocr_results:
            return

        for box in boxes:
            normalized_text = self._normalize_text_for_matching(box.text)
            if not normalized_text:
                continue

            rect = (box.x, box.y, box.w, box.h)
            best_match: _CachedOCRResult | None = None
            best_score = 0.0
            for cached in self._recent_ocr_results:
                if self._normalize_text_for_matching(cached.text) != normalized_text:
                    continue

                score = self._intersection_over_union(rect, cached.rect)
                if score <= best_score:
                    continue
                best_match = cached
                best_score = score

            if best_match is None or best_score < 0.45:
                continue

            best_match.source_language_code = box.source_language_code
            best_match.source_language_label = box.source_language_label
            if box.translated_text:
                best_match.target_language_code = box.target_language_code
                best_match.target_language_label = box.target_language_label
                best_match.translated_text = box.translated_text
            best_match.confidence = box.confidence
            best_match.last_seen_generation = self._ocr_cache_generation

    def _max_ocr_cache_entries(self) -> int:
        return max(self.settings.max_boxes * 4, self.settings.max_ocr_boxes_per_frame * 8, 128)

    def _max_ocr_cache_age_frames(self) -> int:
        return 150 if self.settings.overlay_tracking_enabled else 90

    @staticmethod
    def _ocr_crop_fingerprint(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros((12, 32), dtype=np.uint8)

        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        resized = cv2.resize(gray, (32, 12), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    @staticmethod
    def _ocr_fingerprint_difference(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            return float("inf")
        return float(np.mean(cv2.absdiff(first, second)))

    def _select_ocr_boxes(
        self,
        working_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        if not self.settings.ocr_enabled or not self.ocr_backend.is_available():
            return working_boxes

        limit = max(self.settings.max_ocr_boxes_per_frame, 1)
        if len(working_boxes) <= limit:
            return working_boxes

        prioritized = sorted(working_boxes, key=self._ocr_candidate_priority, reverse=True)
        selected = prioritized[:limit]
        selected.sort(key=lambda box: (box[1], box[0]))
        return selected

    @staticmethod
    def _ocr_candidate_priority(box: tuple[int, int, int, int]) -> float:
        _x, _y, w, h = box
        area = w * h
        aspect_ratio = w / max(h, 1)

        score = (
            min(w / 8.0, 120.0)
            + min(area / 900.0, 80.0)
            + min(aspect_ratio, 16.0) * 4.0
        )
        if aspect_ratio < 1.25:
            score -= 80.0
        if aspect_ratio > 28.0:
            score -= 35.0
        if w >= 280 and 16 <= h <= 64:
            score += 80.0
        if aspect_ratio >= 8.0 and h <= 70:
            score += 40.0
        if h > 72:
            score -= min((h - 72) * 3.0, 140.0)
        if area > 60_000:
            score -= min((area - 60_000) / 700.0, 130.0)
        return score

    def _stabilize_ocr_boxes(
        self,
        working_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        if not self.settings.ocr_enabled or not self.ocr_backend.is_available():
            self._ocr_box_tracks = []
            return working_boxes

        minimum_stable_frames = max(self.settings.stable_ocr_frames, 1)
        if minimum_stable_frames <= 1:
            return working_boxes

        matched_track_indices: set[int] = set()
        next_tracks: list[_TrackedTextBox] = []

        for box in working_boxes:
            match_index = self._find_matching_track(box, matched_track_indices)
            if match_index is None:
                next_tracks.append(_TrackedTextBox(rect=box))
                continue

            matched_track_indices.add(match_index)
            track = self._ocr_box_tracks[match_index]
            next_tracks.append(
                _TrackedTextBox(
                    rect=box,
                    stable_frames=track.stable_frames + 1,
                    missing_frames=0,
                )
            )

        for index, track in enumerate(self._ocr_box_tracks):
            if index in matched_track_indices:
                continue
            if track.missing_frames >= 1:
                continue
            next_tracks.append(
                _TrackedTextBox(
                    rect=track.rect,
                    stable_frames=track.stable_frames,
                    missing_frames=track.missing_frames + 1,
                )
            )

        self._ocr_box_tracks = next_tracks
        stable_boxes = [
            track.rect
            for track in self._ocr_box_tracks
            if track.missing_frames == 0 and track.stable_frames >= minimum_stable_frames
        ]
        stable_boxes.sort(key=lambda box: (box[1], box[0]))
        return stable_boxes

    def _find_matching_track(
        self,
        box: tuple[int, int, int, int],
        used_indices: set[int],
    ) -> int | None:
        threshold = max(min(self.settings.stable_box_iou_threshold, 1.0), 0.0)
        best_index: int | None = None
        best_score = 0.0

        for index, track in enumerate(self._ocr_box_tracks):
            if index in used_indices:
                continue
            score = self._intersection_over_union(box, track.rect)
            if score >= threshold and score > best_score:
                best_index = index
                best_score = score

        return best_index

    def _filter_motion_ocr_boxes(
        self,
        working_boxes: list[tuple[int, int, int, int]],
        enhanced_gray: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        if not self.settings.motion_filter_enabled:
            return working_boxes
        if self._previous_motion_gray is None or self._previous_motion_gray.shape != enhanced_gray.shape:
            return working_boxes

        filtered: list[tuple[int, int, int, int]] = []
        for box in working_boxes:
            if not self._box_has_excessive_motion(box, enhanced_gray):
                filtered.append(box)
        return filtered

    def _box_has_excessive_motion(
        self,
        box: tuple[int, int, int, int],
        enhanced_gray: np.ndarray,
    ) -> bool:
        if self._previous_motion_gray is None:
            return False

        x, y, w, h = box
        current_roi = enhanced_gray[y : y + h, x : x + w]
        previous_roi = self._previous_motion_gray[y : y + h, x : x + w]
        if current_roi.size == 0 or previous_roi.size != current_roi.size:
            return False

        delta = cv2.absdiff(current_roi, previous_roi)
        mean_delta = float(np.mean(delta))
        _, changed_mask = cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)
        changed_ratio = cv2.countNonZero(changed_mask) / max(delta.size, 1)
        return (
            mean_delta >= self.settings.motion_mean_threshold
            and changed_ratio >= self.settings.motion_changed_ratio_threshold
        )

    def _estimate_frame_offset(self, enhanced_gray: np.ndarray, scale: float) -> tuple[float, float, float]:
        if self._previous_motion_gray is None or self._previous_motion_gray.shape != enhanced_gray.shape:
            return 0.0, 0.0, 0.0
        return estimate_grayscale_offset(
            self._previous_motion_gray,
            enhanced_gray,
            source_scale=scale,
            max_dimension=360,
            min_response=0.08,
            max_offset_ratio=0.35,
        )

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        intersection = TextDetectionPipeline._intersection_area(first, second)
        if intersection <= 0:
            return 0.0

        first_area = max(first[2] * first[3], 1)
        second_area = max(second[2] * second[3], 1)
        union = max(first_area + second_area - intersection, 1)
        return intersection / union

    def _iter_motion_adjusted_rects(
        self,
        rect: tuple[int, int, int, int],
        *,
        scaled: bool,
    ) -> list[tuple[tuple[int, int, int, int], bool]]:
        variants = [(rect, False)]
        offset_x, offset_y, confidence = (
            self._current_scaled_motion_offset if scaled else self._current_motion_offset
        )
        if confidence < 0.06:
            return variants

        rounded_x = int(round(offset_x))
        rounded_y = int(round(offset_y))
        if abs(rounded_x) < 2 and abs(rounded_y) < 2:
            return variants

        x, y, w, h = rect
        variants.append(((x + rounded_x, y + rounded_y, w, h), True))
        return variants

    @staticmethod
    def _rect_center_proximity(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_center_x = first[0] + (first[2] / 2.0)
        first_center_y = first[1] + (first[3] / 2.0)
        second_center_x = second[0] + (second[2] / 2.0)
        second_center_y = second[1] + (second[3] / 2.0)
        distance = max(abs(first_center_x - second_center_x), abs(first_center_y - second_center_y))
        tolerance = max(first[2], first[3], second[2], second[3], 1) * 5.0
        return max(1.0 - (distance / tolerance), 0.0)

    @staticmethod
    def _rect_size_similarity(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_area = max(first[2] * first[3], 1)
        second_area = max(second[2] * second[3], 1)
        return min(first_area, second_area) / max(first_area, second_area)

    def _apply_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not boxes:
            self._blank_translation_frames += 1
            if not self.settings.overlay_tracking_enabled and self._blank_translation_frames <= 2:
                return boxes
            if not self.settings.overlay_tracking_enabled:
                self._recent_translations = []
                self._recent_translation_lookup = {}
                self._recent_translation_candidates = []
                self._pending_translation_frames = 0
            return boxes

        self._blank_translation_frames = 0
        translated_boxes = self._reuse_recent_translations(boxes)
        line_skip_indices: set[int] = set()
        if self._translation_block_mode() == "strict":
            translated_boxes, line_skip_indices = self._apply_strict_block_translations(translated_boxes)

        translated_boxes = self._apply_line_translations(translated_boxes, skip_indices=line_skip_indices)
        self._remember_translations(translated_boxes)
        return translated_boxes

    def _apply_line_translations(
        self,
        boxes: list[DetectionBox],
        *,
        skip_indices: set[int] | None = None,
    ) -> list[DetectionBox]:
        skip_indices = skip_indices or set()
        translated_boxes = list(boxes)
        grouped_indices: dict[tuple[str, str], list[int]] = {}
        for index, box in enumerate(translated_boxes):
            if index in skip_indices:
                continue
            if not box.text or box.translated_text:
                continue
            grouped_indices.setdefault((box.source_language_code, box.target_language_code), []).append(index)

        for (source_language_code, target_language_code), indices in grouped_indices.items():
            prioritized_indices = self._prioritize_translation_indices(translated_boxes, indices)
            unique_text_order: list[str] = []
            text_to_indices: dict[str, list[int]] = {}

            for index in prioritized_indices:
                text = translated_boxes[index].text
                normalized_text = self._normalize_text_for_matching(text)
                if not normalized_text:
                    continue
                if normalized_text not in text_to_indices:
                    unique_text_order.append(text)
                    text_to_indices[normalized_text] = [index]
                else:
                    text_to_indices[normalized_text].append(index)

            if not unique_text_order:
                continue

            translated_batch = self.translation_backend.translate_batch(
                unique_text_order,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            for text, translated_text in zip(unique_text_order, translated_batch, strict=False):
                normalized_text = self._normalize_text_for_matching(text)
                for index in text_to_indices.get(normalized_text, []):
                    translated_boxes[index] = replace(translated_boxes[index], translated_text=translated_text)

        return translated_boxes

    def _translation_block_mode(self) -> str:
        mode = (self.settings.translation_block_mode or "line").casefold().strip()
        if mode in {"strict", "block_strict"}:
            return "strict"
        return "line"

    def _apply_strict_block_translations(
        self,
        boxes: list[DetectionBox],
    ) -> tuple[list[DetectionBox], set[int]]:
        blocks = self._strict_translation_blocks(boxes)
        if not blocks:
            return list(boxes), set()

        block_translations = self._translate_blocks(boxes, blocks)
        block_by_anchor: dict[int, DetectionBox] = {}
        block_member_indices: set[int] = set()
        for block_index, block in enumerate(blocks):
            block_member_indices.update(block.indices)
            block_by_anchor[block.indices[0]] = self._block_detection_box(
                boxes,
                block,
                translated_text=block_translations[block_index],
            )

        translated_boxes: list[DetectionBox] = []
        line_skip_indices: set[int] = set()
        for index, box in enumerate(boxes):
            block_box = block_by_anchor.get(index)
            if block_box is not None:
                line_skip_indices.add(len(translated_boxes))
                translated_boxes.append(block_box)
                continue
            if index in block_member_indices:
                continue
            translated_boxes.append(box)

        return translated_boxes, line_skip_indices

    def _translate_blocks(
        self,
        boxes: list[DetectionBox],
        blocks: list[_TranslationBlock],
    ) -> list[str]:
        translated_blocks = [""] * len(blocks)
        grouped_indices: dict[tuple[str, str], list[int]] = {}

        for block_index, block in enumerate(blocks):
            anchor = boxes[block.indices[0]]
            normalized_text = self._normalize_text_for_matching(block.text)
            cached_translation = self._recent_translation_lookup.get(
                (anchor.source_language_code, anchor.target_language_code, normalized_text),
            )
            if cached_translation:
                translated_blocks[block_index] = cached_translation
                continue
            grouped_indices.setdefault(
                (anchor.source_language_code, anchor.target_language_code),
                [],
            ).append(block_index)

        for (source_language_code, target_language_code), indices in grouped_indices.items():
            unique_text_order: list[str] = []
            text_to_indices: dict[str, list[int]] = {}
            for block_index in indices:
                text = blocks[block_index].text
                normalized_text = self._normalize_text_for_matching(text)
                if not normalized_text:
                    continue
                if normalized_text not in text_to_indices:
                    unique_text_order.append(text)
                    text_to_indices[normalized_text] = [block_index]
                else:
                    text_to_indices[normalized_text].append(block_index)

            if not unique_text_order:
                continue

            translated_batch = self.translation_backend.translate_batch(
                unique_text_order,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            for text, translated_text in zip(unique_text_order, translated_batch, strict=False):
                normalized_text = self._normalize_text_for_matching(text)
                for block_index in text_to_indices.get(normalized_text, []):
                    translated_blocks[block_index] = translated_text

        return translated_blocks

    def _strict_translation_blocks(self, boxes: list[DetectionBox]) -> list[_TranslationBlock]:
        candidates = [
            (index, box)
            for index, box in enumerate(boxes)
            if self._is_strict_block_line_candidate(box)
        ]
        if len(candidates) < 2:
            return []

        candidates.sort(key=lambda item: (item[1].y, item[1].x))
        used_indices: set[int] = set()
        blocks: list[_TranslationBlock] = []
        max_lines = 6

        for start_index, start_box in candidates:
            if start_index in used_indices:
                continue

            block: list[tuple[int, DetectionBox]] = [(start_index, start_box)]
            while len(block) < max_lines:
                successor = self._next_strict_block_successor(block, candidates, used_indices)
                if successor is None:
                    break
                block.append(successor)

            block_indices = tuple(index for index, _box in block)
            if self._is_valid_strict_block(block) and not any(index in used_indices for index in block_indices):
                used_indices.update(block_indices)
                blocks.append(
                    _TranslationBlock(
                        indices=block_indices,
                        text="\n".join(box.text.strip() for _index, box in block if box.text.strip()),
                    )
                )

        return blocks

    def _next_strict_block_successor(
        self,
        block: list[tuple[int, DetectionBox]],
        candidates: list[tuple[int, DetectionBox]],
        used_indices: set[int],
    ) -> tuple[int, DetectionBox] | None:
        possible: list[tuple[float, int, DetectionBox]] = []
        last = block[-1][1]
        for candidate_index, candidate in candidates:
            if candidate_index in used_indices or any(index == candidate_index for index, _box in block):
                continue
            if candidate.y < last.y:
                continue
            if not self._can_append_strict_block_line(block, candidate):
                continue
            vertical_gap = max(candidate.y - last.bottom, 0)
            x_delta = abs(candidate.x - block[0][1].x)
            possible.append((vertical_gap + (x_delta * 0.15), candidate_index, candidate))

        if not possible:
            return None
        possible.sort(key=lambda item: (item[0], item[2].y, item[2].x))
        return possible[0][1], possible[0][2]

    def _can_append_strict_block_line(
        self,
        block: list[tuple[int, DetectionBox]],
        candidate: DetectionBox,
    ) -> bool:
        anchor = block[0][1]
        last = block[-1][1]
        if candidate.source_language_code != anchor.source_language_code:
            return False
        if candidate.target_language_code != anchor.target_language_code:
            return False

        heights = [box.h for _index, box in block]
        median_height = float(np.median(heights))
        height_ratio = candidate.h / max(median_height, 1.0)
        if height_ratio < 0.70 or height_ratio > 1.35:
            return False

        vertical_gap = candidate.y - last.bottom
        if vertical_gap < -max(int(round(median_height * 0.25)), 2):
            return False
        if vertical_gap > max(int(round(median_height * 1.30)), 8):
            return False

        x_tolerance = max(int(round(median_height * 1.45)), 36)
        if abs(candidate.x - anchor.x) > x_tolerance:
            return False

        last_overlap = self._horizontal_overlap_ratio(
            (last.x, last.y, last.w, last.h),
            (candidate.x, candidate.y, candidate.w, candidate.h),
        )
        if last_overlap < 0.45:
            return False

        left = min(box.x for _index, box in block)
        right = max(box.right for _index, box in block)
        candidate_center_x = candidate.x + (candidate.w / 2.0)
        if candidate_center_x < left - x_tolerance or candidate_center_x > right + x_tolerance:
            return False

        return True

    def _is_valid_strict_block(self, block: list[tuple[int, DetectionBox]]) -> bool:
        if len(block) < 2:
            return False
        combined_text = " ".join(box.text.strip() for _index, box in block)
        normalized_text = self._normalize_text_for_matching(combined_text)
        if len(normalized_text) < 48:
            return False
        if len(normalized_text.split()) < 8:
            return False
        return True

    def _is_strict_block_line_candidate(self, box: DetectionBox) -> bool:
        if not box.text or box.translated_text:
            return False
        text = " ".join(box.text.split())
        normalized_text = self._normalize_text_for_matching(text)
        if len(normalized_text) < 16:
            return False
        if self._looks_like_url(text) or self._looks_like_match_url(normalized_text):
            return False

        words = normalized_text.split()
        if len(words) < 3 and len(normalized_text) < 24:
            return False
        alpha_count = sum(character.isalpha() for character in normalized_text)
        digit_count = sum(character.isdigit() for character in normalized_text)
        if alpha_count == 0 or digit_count >= alpha_count:
            return False
        if len(words) <= 2 and len(normalized_text) < 28:
            return False
        return True

    def _block_detection_box(
        self,
        boxes: list[DetectionBox],
        block: _TranslationBlock,
        *,
        translated_text: str,
    ) -> DetectionBox:
        block_boxes = [boxes[index] for index in block.indices]
        anchor = block_boxes[0]
        left = min(box.x for box in block_boxes)
        top = min(box.y for box in block_boxes)
        right = max(box.right for box in block_boxes)
        bottom = max(box.bottom for box in block_boxes)
        confidences = [box.confidence for box in block_boxes if box.confidence is not None]
        confidence = float(np.mean(confidences)) if confidences else anchor.confidence
        return replace(
            anchor,
            x=left,
            y=top,
            w=right - left,
            h=bottom - top,
            text=block.text,
            translated_text=translated_text,
            confidence=confidence,
        )

    @staticmethod
    def _horizontal_overlap_ratio(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        left = max(first[0], second[0])
        right = min(first[0] + first[2], second[0] + second[2])
        return max(right - left, 0) / max(min(first[2], second[2]), 1)

    def _reuse_recent_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not self._recent_translation_candidates:
            return list(boxes)

        reused_boxes: list[DetectionBox] = []
        for box in boxes:
            recent = self._find_recent_translation_match(box)
            if recent is None:
                reused_boxes.append(box)
                continue

            if box.text:
                reused_boxes.append(replace(box, translated_text=recent.translated_text))
                continue

            reused_boxes.append(
                replace(
                    box,
                    text=recent.text,
                    translated_text=recent.translated_text,
                    source_language_code=recent.source_language_code,
                    source_language_label=recent.source_language_label,
                    target_language_code=recent.target_language_code,
                    target_language_label=recent.target_language_label,
                    confidence=recent.confidence,
                )
            )
        return reused_boxes

    def _find_recent_translation_match(self, box: DetectionBox) -> DetectionBox | None:
        current_text = self._normalize_text_for_matching(box.text)
        if current_text:
            direct_match = self._recent_translation_lookup.get(
                (box.source_language_code, box.target_language_code, current_text),
            )
            if direct_match:
                for recent, recent_text in self._recent_translation_candidates:
                    if (
                        recent_text == current_text
                        and recent.source_language_code == box.source_language_code
                        and recent.target_language_code == box.target_language_code
                        and recent.translated_text == direct_match
                    ):
                        return recent

        current_rect = (box.x, box.y, box.w, box.h)
        current_area = max(box.w * box.h, 1)
        best_score = 0.0
        best_recent: DetectionBox | None = None
        for recent, recent_text in self._recent_translation_candidates:
            if recent.target_language_code != box.target_language_code:
                continue
            if current_text and recent.source_language_code != box.source_language_code:
                continue

            recent_rect = (recent.x, recent.y, recent.w, recent.h)
            geometry_score, overlap = self._recent_geometry_score(current_rect, recent_rect, current_area)
            if not current_text:
                if geometry_score < 0.58:
                    continue
                score = geometry_score
            else:
                similarity = SequenceMatcher(None, current_text, recent_text).ratio()
                if not self._is_stable_translation_similarity_match(
                    current_text=current_text,
                    recent_text=recent_text,
                    similarity=similarity,
                    geometry_score=geometry_score,
                    overlap=overlap,
                ):
                    continue
                if overlap < 0.35:
                    score = (similarity * 0.82) + (geometry_score * 0.18)
                else:
                    score = (overlap * 0.50) + (similarity * 0.42) + (geometry_score * 0.08)

            if score > best_score:
                best_score = score
                best_recent = recent

        return best_recent

    def _recent_geometry_score(
        self,
        current_rect: tuple[int, int, int, int],
        recent_rect: tuple[int, int, int, int],
        current_area: int,
    ) -> tuple[float, float]:
        best_score = 0.0
        best_overlap = 0.0
        for candidate_rect, motion_adjusted in self._iter_motion_adjusted_rects(recent_rect, scaled=False):
            intersection = self._intersection_area(current_rect, candidate_rect)
            overlap = intersection / current_area
            recent_area = max(candidate_rect[2] * candidate_rect[3], 1)
            mutual_overlap = max(overlap, intersection / recent_area)
            proximity = self._rect_center_proximity(current_rect, candidate_rect)
            size_similarity = self._rect_size_similarity(current_rect, candidate_rect)
            score = (mutual_overlap * 0.62) + (proximity * 0.24) + (size_similarity * 0.14)
            if motion_adjusted:
                score += 0.08
            if score > best_score:
                best_score = score
                best_overlap = overlap
        return min(best_score, 1.0), best_overlap

    def _is_stable_translation_similarity_match(
        self,
        *,
        current_text: str,
        recent_text: str,
        similarity: float,
        geometry_score: float,
        overlap: float,
    ) -> bool:
        if not self.settings.translation_similarity_stability_enabled:
            return False
        if not current_text or not recent_text:
            return False
        if self._looks_like_match_url(current_text) or self._looks_like_match_url(recent_text):
            return False
        if self._translation_numbers(current_text) != self._translation_numbers(recent_text):
            return False

        min_chars = max(self.settings.translation_similarity_min_chars, 1)
        if min(len(current_text), len(recent_text)) < min_chars:
            return False

        alpha_count = sum(character.isalpha() for character in current_text + recent_text)
        digit_count = sum(character.isdigit() for character in current_text + recent_text)
        if alpha_count == 0 or digit_count > alpha_count:
            return False

        threshold = min(max(self.settings.translation_similarity_threshold, 0.0), 1.0)
        if overlap < 0.35:
            if not self.settings.overlay_tracking_enabled and self._current_motion_offset[2] < 0.06:
                return False
            threshold = max(threshold, 0.94)
        elif geometry_score < 0.58:
            return False

        return similarity >= threshold

    def _prioritize_translation_indices(
        self,
        boxes: list[DetectionBox],
        indices: list[int],
    ) -> list[int]:
        return sorted(indices, key=lambda index: self._translation_priority(boxes[index]), reverse=True)

    def _translation_priority(self, box: DetectionBox) -> float:
        text = box.text.strip()
        if not text:
            return -1.0

        alpha_count = sum(character.isalpha() for character in text)
        digit_count = sum(character.isdigit() for character in text)
        word_count = len(text.split())
        area_bonus = min((box.w * box.h) / 5000.0, 30.0)
        score = len(text) + (word_count * 8.0) + area_bonus + (alpha_count * 0.25)

        if self._looks_like_url(text):
            score -= 120.0
        if digit_count and digit_count >= alpha_count:
            score -= 80.0
        if word_count <= 2 and len(text) < 18:
            score -= 35.0
        if len(text) <= 4:
            score -= 45.0

        return score

    def _remember_translations(self, boxes: list[DetectionBox]) -> None:
        remembered = [box for box in boxes if box.translated_text]
        if not remembered:
            self._pending_translation_frames += 1
            if (
                not self.settings.overlay_tracking_enabled
                and self._recent_translations
                and self._pending_translation_frames <= 2
            ):
                return
        else:
            self._pending_translation_frames = 0

        if self.settings.overlay_tracking_enabled:
            remembered.extend(recent for recent in self._recent_translations if recent.translated_text)

        deduped: list[DetectionBox] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for box in remembered:
            normalized_text = self._normalize_text_for_matching(box.text)
            if not normalized_text:
                continue
            key = (
                box.source_language_code,
                box.target_language_code,
                normalized_text,
                box.translated_text,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(box)

        deduped.sort(key=self._translation_priority, reverse=True)
        multiplier = 6 if self.settings.overlay_tracking_enabled else 3
        minimum = 60 if self.settings.overlay_tracking_enabled else 24
        self._recent_translations = deduped[: max(self.settings.max_ocr_boxes_per_frame * multiplier, minimum)]
        self._recent_translation_lookup = {}
        self._recent_translation_candidates = []
        for box in self._recent_translations:
            normalized_text = self._normalize_text_for_matching(box.text)
            if normalized_text and box.translated_text:
                self._recent_translation_lookup[
                    (box.source_language_code, box.target_language_code, normalized_text)
                ] = box.translated_text
                self._recent_translation_candidates.append((box, normalized_text))

    def _merge_text_boxes(
        self,
        candidates: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        merged = sorted(candidates, key=lambda box: (box[1], box[0]))
        changed = True

        while changed:
            changed = False
            next_pass: list[tuple[int, int, int, int]] = []

            while merged:
                current = merged.pop(0)
                scan_index = 0

                while scan_index < len(merged):
                    candidate = merged[scan_index]
                    if self._should_merge_boxes(current, candidate):
                        current = self._merge_box_pair(current, candidate)
                        merged.pop(scan_index)
                        changed = True
                        continue
                    scan_index += 1

                next_pass.append(current)

            merged = sorted(next_pass, key=lambda box: (box[1], box[0]))

        return self._suppress_nested_boxes(merged)

    def _should_merge_boxes(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second

        horizontal_gap = max(max(x1, x2) - min(x1 + w1, x2 + w2), 0)
        vertical_overlap = max(min(y1 + h1, y2 + h2) - max(y1, y2), 0)
        min_height = max(min(h1, h2), 1)
        height_ratio = max(h1, h2) / min_height
        same_line = vertical_overlap / min_height >= 0.55
        merge_gap = max(h1, h2) * 2.4

        return same_line and height_ratio <= 1.8 and horizontal_gap <= merge_gap

    @staticmethod
    def _merge_box_pair(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1 + w1, x2 + w2)
        bottom = max(y1 + h1, y2 + h2)
        return left, top, right - left, bottom - top

    @staticmethod
    def _suppress_nested_boxes(
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []

        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            x, y, w, h = box
            area = max(w * h, 1)
            is_nested = False

            for kept_box in kept:
                overlap_area = TextDetectionPipeline._intersection_area(box, kept_box)
                if overlap_area / area >= 0.85:
                    is_nested = True
                    break

            if not is_nested:
                kept.append(box)

        kept.sort(key=lambda item: (item[1], item[0]))
        return kept

    @staticmethod
    def _intersection_area(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[0] + first[2], second[0] + second[2])
        bottom = min(first[1] + first[3], second[1] + second[3])
        return max(right - left, 0) * max(bottom - top, 0)

    def _resolve_psm(self, width: int, height: int) -> int:
        if width / max(height, 1) >= 6.0:
            return 7
        return self.settings.ocr_psm

    def _normalize_recognized_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        normalized = self._strip_minor_script_noise(normalized)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = normalized.strip(" |:;.,_-`~[]{}<>")
        return normalized

    @staticmethod
    def _normalize_text_for_matching(text: str) -> str:
        normalized = text.casefold()
        # Keep alphanumerics, Thai, and common Japanese (Hiragana/Katakana/Kanji)
        normalized = re.sub(r"[^0-9a-z\u0E00-\u0E7F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _translation_numbers(normalized_text: str) -> tuple[str, ...]:
        return tuple(token for token in normalized_text.split() if token.isdigit())

    @staticmethod
    def _looks_like_match_url(normalized_text: str) -> bool:
        tokens = normalized_text.split()
        collapsed = "".join(tokens)
        return (
            collapsed.startswith(("http", "www"))
            or "www" in tokens
            or any(token in {"com", "org", "net", "io", "gg"} for token in tokens)
        )

    @staticmethod
    def _looks_like_url(text: str) -> bool:
        normalized = text.casefold().replace(" ", "")
        return (
            "http://" in normalized
            or "https://" in normalized
            or normalized.startswith("www.")
            or ".com/" in normalized
            or ".com" in normalized
            or ".org" in normalized
            or ".net" in normalized
        )

    @staticmethod
    def _strip_minor_script_noise(text: str) -> str:
        thai_count = sum("\u0E00" <= character <= "\u0E7F" for character in text)
        latin_count = sum(character.isascii() and character.isalpha() for character in text)

        # Tesseract often injects a couple of Thai glyphs into otherwise English lines
        # when running in mixed-language mode. Dropping only the tiny minority script
        # improves the text that reaches translation without hurting genuinely mixed lines.
        if latin_count >= 8 and 0 < thai_count <= max(2, latin_count // 8):
            text = "".join(character for character in text if not ("\u0E00" <= character <= "\u0E7F"))
            text = " ".join(text.split())

        return text

    def _is_usable_text(self, text: str, confidence: float | None) -> bool:
        if not text:
            return False

        alpha_numeric_count = sum(character.isalnum() for character in text)
        thai_count = sum("\u0E00" <= character <= "\u0E7F" for character in text)
        meaningful_count = alpha_numeric_count + thai_count
        tokens = text.split()
        word_like_tokens = [token for token in tokens if sum(character.isalpha() for character in token) >= 2]
        symbolic_tokens = [
            token
            for token in tokens
            if sum(character.isalnum() for character in token) <= max(len(token) // 2, 1)
        ]
        short_noise_tokens = [
            token
            for token in tokens
            if len(token) <= 2 and sum(character.isalnum() for character in token) <= 1
        ]

        if meaningful_count < 2 and len(text) <= 3:
            return False
        if len(text) == 1:
            return False

        punctuation_ratio = sum(not character.isalnum() and not character.isspace() for character in text) / max(
            len(text),
            1,
        )
        if punctuation_ratio > 0.55 and meaningful_count < 5:
            return False
        if len(tokens) >= 6 and len(word_like_tokens) <= 2:
            noisy_token_count = len(symbolic_tokens) + len(short_noise_tokens)
            if noisy_token_count / max(len(tokens), 1) >= 0.42:
                return False
        if confidence is not None and confidence < 25.0 and meaningful_count < 6:
            return False
        return True

    def _draw_annotations(self, frame: np.ndarray, boxes: list[DetectionBox]) -> np.ndarray:
        for index, box in enumerate(boxes, start=1):
            cv2.rectangle(frame, (box.x, box.y), (box.right, box.bottom), (48, 231, 149), 2)
            cv2.putText(
                frame,
                f"#{index}",
                (box.x, max(box.y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (48, 231, 149),
                2,
                cv2.LINE_AA,
            )
        return frame

    def _draw_mask_preview(
        self,
        mask: np.ndarray,
        working_boxes: list[tuple[int, int, int, int]],
        *,
        output_shape: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in working_boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 215, 255), 2)
        if output_shape is not None and preview.shape[:2] != output_shape[:2]:
            preview = cv2.resize(
                preview,
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return preview

    def _status_message(self) -> str:
        source_language = get_source_language_option(self.settings.source_language_code)
        target_language = get_target_language_option(self.settings.target_language_code)
        route = f"{source_language.label} -> {target_language.label}"
        detector_status = self._detector_status_message()
        translation_status = self.translation_backend.describe()
        if self.settings.ocr_enabled and self.ocr_backend.is_available():
            if self.ocr_backend.supports_full_frame():
                ocr_status = f"{self.ocr_backend.describe()} | full-frame OCR"
                if self._last_ocr_candidate_count:
                    ocr_status = f"{ocr_status} | read {self._last_ocr_submitted_count}"
            else:
                ocr_status = f"{self.ocr_backend.describe()} | {self.settings.max_ocr_boxes_per_frame} new OCR/frame"
            if self._last_ocr_candidate_count and not self.ocr_backend.supports_full_frame():
                ocr_status = (
                    f"{ocr_status} | submitted {self._last_ocr_submitted_count}, "
                    f"reused {self._last_ocr_reuse_count}/{self._last_ocr_candidate_count}"
                )
            return f"{route} | {detector_status} | {ocr_status} | {translation_status}"
        if self.settings.ocr_enabled:
            return f"{route} | {detector_status} | {self.ocr_backend.describe()} | {translation_status}"
        return f"{route} | {detector_status} | OCR disabled | {translation_status}"

    def _detector_status_message(self) -> str:
        mode = normalize_text_detector_mode(self.settings.text_detector_mode)
        scale = self._effective_detection_scale()
        scale_suffix = f" @ {scale:.2f}x" if abs(scale - 1.0) >= 0.01 else ""
        region_suffix = ""
        if self._translation_region_mode() == "hover":
            region_suffix = f" | {self._last_hover_region_status or 'hover'}"
        scanline_suffix = ""
        if self._translation_region_mode() != "hover" and self.settings.scanline_roi_enabled:
            band_count = self._scanline_last_band_count or max(self.settings.scanline_roi_band_count, 2)
            if self._scanline_last_band_index is None:
                scanline_suffix = f" | scanline {band_count} bands"
            else:
                scanline_suffix = f" | scanline {self._scanline_last_band_index + 1}/{band_count}"
        if self._full_frame_ocr_enabled():
            return f"Native full-frame OCR detector{scale_suffix}"
        if mode == "opencv":
            return f"OpenCV morphology detector{scale_suffix}{region_suffix}{scanline_suffix}"

        detector = self._ensure_deep_text_detector(mode)
        return f"{detector.describe()}{scale_suffix}{region_suffix}{scanline_suffix}"

    def _effective_detection_scale(self) -> float:
        detector_scale = min(max(self.settings.detection_scale, 0.25), 1.0)
        return max(self.settings.upscale_factor, 1.0) * detector_scale

    def _resolve_source_language(self, text: str) -> tuple[str, str]:
        if self.settings.source_language_code not in {"auto", "tha+eng"}:
            source_language = get_source_language_option(self.settings.source_language_code)
            return source_language.code, source_language.label

        detected_code = detect_language_code(text)
        if detected_code == "unknown" and self.settings.source_language_code != "auto":
            source_language = get_source_language_option(self.settings.source_language_code)
            return source_language.code, source_language.label
        return detected_code, language_label(detected_code)

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def _detection_threshold_scale(self) -> float:
        return min(max(self._active_detection_scale, 0.25), 1.0)

    def _scaled_detection_length(self, value: int, *, minimum: int) -> int:
        return max(int(round(value * self._detection_threshold_scale())), minimum)

    def _scaled_detection_area(self, value: int, *, minimum: int) -> int:
        scale = self._detection_threshold_scale()
        return max(int(round(value * scale * scale)), minimum)
