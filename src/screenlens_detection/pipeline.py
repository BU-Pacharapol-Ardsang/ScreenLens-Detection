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
from .ocr import OCRBackend, OCRResult
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
    psm: int


@dataclass(slots=True)
class _CachedOCRResult:
    rect: tuple[int, int, int, int]
    fingerprint: np.ndarray
    text: str
    confidence: float | None = None


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
        self._ocr_box_tracks: list[_TrackedTextBox] = []
        self._previous_motion_gray: np.ndarray | None = None
        self._deep_text_detector: TextDetectorBackend | None = None

    def close(self) -> None:
        self.ocr_backend.close()
        self.translation_backend.close()
        if self._deep_text_detector is not None:
            self._deep_text_detector.close()

    def process(self, frame: np.ndarray, *, monitor_label: str = "") -> FrameAnalysis:
        started = perf_counter()

        scaled_frame, scale = self._scale_frame(frame)
        enhanced_gray = self._enhance_grayscale(scaled_frame)
        mask = self._build_text_mask(enhanced_gray)
        line_mask = self._build_line_mask(mask)
        working_boxes = self._detect_text_boxes(scaled_frame, line_mask, mask, enhanced_gray)
        stable_working_boxes = self._stabilize_ocr_boxes(working_boxes)
        motion_filtered_boxes = self._filter_motion_ocr_boxes(stable_working_boxes, enhanced_gray)
        boxes = self._annotate_with_ocr(motion_filtered_boxes, enhanced_gray, frame.shape, scale)
        boxes = self._apply_translations(boxes)
        content_offset_x, content_offset_y, content_motion_confidence = self._estimate_content_offset(
            enhanced_gray,
            scale,
        )
        self._previous_motion_gray = enhanced_gray.copy()

        annotated = self._draw_annotations(frame.copy(), boxes)
        processed_preview = self._draw_mask_preview(line_mask, working_boxes)

        elapsed = max(perf_counter() - started, 1e-6)
        return FrameAnalysis(
            annotated_frame=annotated,
            processed_preview=processed_preview,
            boxes=boxes,
            source_frame=frame.copy(),
            status=self._status_message(),
            ocr_runtime=self.ocr_backend.runtime_diagnostics(),
            fps=1.0 / elapsed,
            ocr_available=self.ocr_backend.is_available(),
            monitor_label=monitor_label,
            content_offset_x=content_offset_x,
            content_offset_y=content_offset_y,
            content_motion_confidence=content_motion_confidence,
        )

    def _scale_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        scale = max(self.settings.upscale_factor, 1.0)
        if scale == 1.0:
            return frame.copy(), scale
        scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return scaled, scale

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

    def _build_text_mask(self, gray: np.ndarray) -> np.ndarray:
        block_size = self._ensure_odd(self.settings.threshold_block_size)
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

        gradient = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        _, gradient_mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gradient_mask = cv2.dilate(
            gradient_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        combined = cv2.bitwise_and(polarity_mask, gradient_mask)
        if cv2.countNonZero(combined) < max(250, cv2.countNonZero(polarity_mask) // 10):
            combined = polarity_mask

        return cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

    def _build_line_mask(self, mask: np.ndarray) -> np.ndarray:
        dense_mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(self.settings.morphology_width * 2 + 1, 17), max(self.settings.morphology_height, 3)),
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, max(self.settings.morphology_height, 3)),
        )

        line_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, vertical_kernel)
        return cv2.morphologyEx(
            line_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

    def _detect_text_boxes(
        self,
        scaled_frame: np.ndarray,
        line_mask: np.ndarray,
        text_mask: np.ndarray,
        enhanced_gray: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        mode = normalize_text_detector_mode(self.settings.text_detector_mode)
        if mode == "opencv":
            return self._extract_text_boxes(line_mask, text_mask, enhanced_gray, scaled_frame.shape)

        detector = self._ensure_deep_text_detector(mode)
        if not detector.is_available():
            return []

        detected_boxes = detector.detect(scaled_frame)
        return self._filter_detector_boxes(detected_boxes, scaled_frame.shape)

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
        max_box_height = int(frame_height * self.settings.max_box_height_ratio)

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

            if area < self.settings.min_contour_area:
                continue
            if area > int(frame_area * 0.20):
                continue
            if clipped_w < self.settings.min_box_width or clipped_h < self.settings.min_box_height:
                continue
            if clipped_h > max_box_height:
                continue
            if not 0.6 <= aspect_ratio <= 60.0:
                continue

            candidates.append((left, top, clipped_w, clipped_h))

        merged = self._merge_text_boxes(candidates)
        merged.sort(key=lambda box: (box[1], box[0]))
        return merged[: self.settings.max_boxes]

    def _extract_text_boxes(
        self,
        line_mask: np.ndarray,
        text_mask: np.ndarray,
        enhanced_gray: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
        max_box_height = int(frame_shape[0] * self.settings.max_box_height_ratio)
        frame_area = frame_shape[0] * frame_shape[1]

        candidates: list[tuple[int, int, int, int]] = []
        for component_index in range(1, component_count):
            x, y, w, h, _component_area = stats[component_index]
            area = w * h
            aspect_ratio = w / max(h, 1)

            if area < self.settings.min_contour_area:
                continue
            if area > int(frame_area * 0.18):
                continue
            if w < self.settings.min_box_width or h < self.settings.min_box_height:
                continue
            if h > max_box_height:
                continue
            if not 1.1 <= aspect_ratio <= 45.0:
                continue

            text_roi = text_mask[y : y + h, x : x + w]
            if text_roi.size == 0:
                continue

            foreground_ratio = cv2.countNonZero(text_roi) / max(area, 1)
            if not 0.03 <= foreground_ratio <= 0.60:
                continue

            gray_roi = enhanced_gray[y : y + h, x : x + w]
            edge_density = cv2.countNonZero(cv2.Canny(gray_roi, 50, 150)) / max(area, 1)
            if edge_density < 0.025:
                continue

            sub_component_count = cv2.connectedComponents(text_roi, connectivity=8)[0] - 1
            if sub_component_count < 3 and aspect_ratio > 3.5:
                continue

            candidates.append((x, y, w, h))

        merged = self._merge_text_boxes(candidates)
        merged.sort(key=lambda box: (box[1], box[0]))
        return merged[: self.settings.max_boxes]

    def _annotate_with_ocr(
        self,
        working_boxes: list[tuple[int, int, int, int]],
        enhanced_gray: np.ndarray,
        original_shape: tuple[int, int, int],
        scale: float,
    ) -> list[DetectionBox]:
        detected_boxes: list[DetectionBox] = []
        ocr_boxes = self._select_ocr_boxes(working_boxes)
        ocr_attempted = self.settings.ocr_enabled and self.ocr_backend.is_available()
        ocr_candidates: list[_OCRCroppedBox] = []

        for x, y, w, h in ocr_boxes:
            pad_x = max(int(w * 0.04), 4)
            pad_y = max(int(h * 0.25), 4)
            crop_x1 = max(x - pad_x, 0)
            crop_y1 = max(y - pad_y, 0)
            crop_x2 = min(x + w + pad_x, enhanced_gray.shape[1])
            crop_y2 = min(y + h + pad_y, enhanced_gray.shape[0])

            crop = enhanced_gray[crop_y1:crop_y2, crop_x1:crop_x2]
            ocr_candidates.append(
                _OCRCroppedBox(
                    rect=(x, y, w, h),
                    crop=crop,
                    psm=self._resolve_psm(w, h),
                )
            )

        ocr_results: list[OCRResult | None] = [None] * len(ocr_candidates)
        candidate_fingerprints: list[np.ndarray] = []
        if ocr_attempted and ocr_candidates:
            pending_indices: list[int] = []
            pending_candidates: list[_OCRCroppedBox] = []

            for index, candidate in enumerate(ocr_candidates):
                fingerprint = self._ocr_crop_fingerprint(candidate.crop)
                candidate_fingerprints.append(fingerprint)
                cached_result = self._find_cached_ocr_result(candidate.rect, fingerprint)
                if cached_result is not None:
                    ocr_results[index] = cached_result
                    continue

                pending_indices.append(index)
                pending_candidates.append(candidate)

            pending_results = []
            if pending_candidates:
                pending_results = self.ocr_backend.recognize_batch(
                    [self.ocr_backend.prepare_image(candidate.crop) for candidate in pending_candidates],
                    language=self.settings.ocr_language,
                    psms=[candidate.psm for candidate in pending_candidates],
                )

            for index, ocr_result in zip(pending_indices, pending_results, strict=False):
                ocr_results[index] = ocr_result

        elif ocr_candidates:
            candidate_fingerprints = [self._ocr_crop_fingerprint(candidate.crop) for candidate in ocr_candidates]

        ocr_cache_updates: list[_CachedOCRResult] = []
        for candidate_index, (candidate, ocr_result) in enumerate(zip(ocr_candidates, ocr_results, strict=False)):
            x, y, w, h = candidate.rect
            text = ""
            confidence = None
            if ocr_result is not None:
                text = ocr_result.text
                confidence = ocr_result.confidence

            text = self._normalize_recognized_text(text)
            if ocr_attempted and not self._is_usable_text(text, confidence):
                continue

            if ocr_attempted and text:
                if candidate_index < len(candidate_fingerprints):
                    ocr_cache_updates.append(
                        _CachedOCRResult(
                            rect=candidate.rect,
                            fingerprint=candidate_fingerprints[candidate_index],
                            text=text,
                            confidence=confidence,
                        )
                    )

            mapped_x = int(x / scale)
            mapped_y = int(y / scale)
            mapped_w = int(w / scale)
            mapped_h = int(h / scale)

            mapped_w = min(mapped_w, original_shape[1] - mapped_x)
            mapped_h = min(mapped_h, original_shape[0] - mapped_y)
            source_language_code, source_language_label = self._resolve_source_language(text)
            target_language = get_target_language_option(self.settings.target_language_code)

            detected_boxes.append(
                DetectionBox(
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
            )

        if ocr_attempted:
            self._remember_ocr_results(ocr_cache_updates)

        return detected_boxes

    def _find_cached_ocr_result(self, rect: tuple[int, int, int, int], fingerprint: np.ndarray) -> OCRResult | None:
        best_match: _CachedOCRResult | None = None
        best_overlap = 0.0

        for cached in self._recent_ocr_results:
            overlap = self._intersection_over_union(rect, cached.rect)
            if overlap < 0.82 or overlap <= best_overlap:
                continue

            difference = self._ocr_fingerprint_difference(fingerprint, cached.fingerprint)
            if difference > 5.5:
                continue

            best_match = cached
            best_overlap = overlap

        if best_match is None:
            return None
        return OCRResult(text=best_match.text, confidence=best_match.confidence)

    def _remember_ocr_results(self, updates: list[_CachedOCRResult]) -> None:
        if not updates:
            return

        remembered = [*updates, *self._recent_ocr_results]
        deduped: list[_CachedOCRResult] = []
        for candidate in remembered:
            if not candidate.text:
                continue
            if any(
                existing.text == candidate.text
                and self._intersection_over_union(existing.rect, candidate.rect) >= 0.92
                for existing in deduped
            ):
                continue
            deduped.append(candidate)

        limit = max(self.settings.max_ocr_boxes_per_frame * 4, 32)
        self._recent_ocr_results = deduped[:limit]

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
        if h > 90:
            score -= min((h - 90) * 1.4, 90.0)
        if area > 140_000:
            score -= min((area - 140_000) / 1200.0, 120.0)
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

    def _estimate_content_offset(self, enhanced_gray: np.ndarray, scale: float) -> tuple[float, float, float]:
        if not self.settings.overlay_tracking_enabled:
            return 0.0, 0.0, 0.0
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

    def _apply_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not boxes:
            if not self.settings.overlay_tracking_enabled:
                self._recent_translations = []
                self._recent_translation_lookup = {}
                self._recent_translation_candidates = []
            return boxes

        translated_boxes = self._reuse_recent_translations(boxes)
        grouped_indices: dict[tuple[str, str], list[int]] = {}
        for index, box in enumerate(translated_boxes):
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

        self._remember_translations(translated_boxes)
        return translated_boxes

    def _reuse_recent_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not self._recent_translation_candidates:
            return list(boxes)

        reused_boxes: list[DetectionBox] = []
        for box in boxes:
            translated_text = self._find_recent_translation(box)
            if translated_text:
                reused_boxes.append(replace(box, translated_text=translated_text))
                continue
            reused_boxes.append(box)
        return reused_boxes

    def _find_recent_translation(self, box: DetectionBox) -> str:
        if not box.text:
            return ""

        current_text = self._normalize_text_for_matching(box.text)
        if not current_text:
            return ""

        direct_match = self._recent_translation_lookup.get(
            (box.source_language_code, box.target_language_code, current_text),
        )
        if direct_match:
            return direct_match

        current_rect = (box.x, box.y, box.w, box.h)
        current_area = max(box.w * box.h, 1)
        best_score = 0.0
        best_translation = ""
        for recent, recent_text in self._recent_translation_candidates:
            if recent.source_language_code != box.source_language_code:
                continue
            if recent.target_language_code != box.target_language_code:
                continue

            similarity = SequenceMatcher(None, current_text, recent_text).ratio()
            overlap = self._intersection_area(current_rect, (recent.x, recent.y, recent.w, recent.h)) / current_area
            if overlap < 0.35:
                if not self.settings.overlay_tracking_enabled or similarity < 0.88:
                    continue
                score = similarity * 0.90
            else:
                score = (overlap * 0.55) + (similarity * 0.45)

            if similarity >= 0.45 and score > best_score:
                best_score = score
                best_translation = recent.translated_text

        return best_translation

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
        normalized = re.sub(r"[^0-9a-z\u0E00-\u0E7F]+", " ", normalized)
        return " ".join(normalized.split())

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
    ) -> np.ndarray:
        preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in working_boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 215, 255), 2)
        return preview

    def _status_message(self) -> str:
        source_language = get_source_language_option(self.settings.source_language_code)
        target_language = get_target_language_option(self.settings.target_language_code)
        route = f"{source_language.label} -> {target_language.label}"
        detector_status = self._detector_status_message()
        translation_status = self.translation_backend.describe()
        if self.settings.ocr_enabled and self.ocr_backend.is_available():
            ocr_status = f"{self.ocr_backend.describe()} | {self.settings.max_ocr_boxes_per_frame} boxes/frame"
            return f"{route} | {detector_status} | {ocr_status} | {translation_status}"
        if self.settings.ocr_enabled:
            return f"{route} | {detector_status} | {self.ocr_backend.describe()} | {translation_status}"
        return f"{route} | {detector_status} | OCR disabled | {translation_status}"

    def _detector_status_message(self) -> str:
        mode = normalize_text_detector_mode(self.settings.text_detector_mode)
        if mode == "opencv":
            return "OpenCV morphology detector"

        detector = self._ensure_deep_text_detector(mode)
        return detector.describe()

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
