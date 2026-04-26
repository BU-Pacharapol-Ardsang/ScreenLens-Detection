from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import replace
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
from .ocr import OCRBackend
from .translation import TranslationBackend


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

    def close(self) -> None:
        self.translation_backend.close()

    def process(self, frame: np.ndarray, *, monitor_label: str = "") -> FrameAnalysis:
        started = perf_counter()

        scaled_frame, scale = self._scale_frame(frame)
        enhanced_gray = self._enhance_grayscale(scaled_frame)
        mask = self._build_text_mask(enhanced_gray)
        line_mask = self._build_line_mask(mask)
        working_boxes = self._extract_text_boxes(line_mask, mask, enhanced_gray, scaled_frame.shape)
        boxes = self._annotate_with_ocr(working_boxes, enhanced_gray, frame.shape, scale)
        boxes = self._apply_translations(boxes)

        annotated = self._draw_annotations(frame.copy(), boxes)
        processed_preview = self._draw_mask_preview(line_mask, working_boxes)

        elapsed = max(perf_counter() - started, 1e-6)
        return FrameAnalysis(
            annotated_frame=annotated,
            processed_preview=processed_preview,
            boxes=boxes,
            status=self._status_message(),
            ocr_runtime=self.ocr_backend.runtime_diagnostics(),
            fps=1.0 / elapsed,
            ocr_available=self.ocr_backend.is_available(),
            monitor_label=monitor_label,
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

        for x, y, w, h in ocr_boxes:
            pad_x = max(int(w * 0.04), 4)
            pad_y = max(int(h * 0.25), 4)
            crop_x1 = max(x - pad_x, 0)
            crop_y1 = max(y - pad_y, 0)
            crop_x2 = min(x + w + pad_x, enhanced_gray.shape[1])
            crop_y2 = min(y + h + pad_y, enhanced_gray.shape[0])

            crop = enhanced_gray[crop_y1:crop_y2, crop_x1:crop_x2]
            text = ""
            confidence = None
            if self.settings.ocr_enabled and self.ocr_backend.is_available():
                ocr_crop = self.ocr_backend.prepare_image(crop)
                ocr_result = self.ocr_backend.recognize(
                    ocr_crop,
                    language=self.settings.ocr_language,
                    psm=self._resolve_psm(w, h),
                )
                text = ocr_result.text
                confidence = ocr_result.confidence

            text = self._normalize_recognized_text(text)
            if self.settings.ocr_enabled and not self._is_usable_text(text, confidence):
                continue

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

        return detected_boxes

    def _select_ocr_boxes(
        self,
        working_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        if not self.settings.ocr_enabled or not self.ocr_backend.is_available():
            return working_boxes

        limit = max(self.settings.max_ocr_boxes_per_frame, 1)
        if len(working_boxes) <= limit:
            return working_boxes

        prioritized = sorted(working_boxes, key=lambda box: box[2] * box[3], reverse=True)
        selected = prioritized[:limit]
        selected.sort(key=lambda box: (box[1], box[0]))
        return selected

    def _apply_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not boxes:
            self._recent_translations = []
            return boxes

        translated_boxes = self._reuse_recent_translations(boxes)
        grouped_indices: dict[tuple[str, str], list[int]] = {}
        for index, box in enumerate(translated_boxes):
            if not box.text or box.translated_text:
                continue
            grouped_indices.setdefault((box.source_language_code, box.target_language_code), []).append(index)

        for (source_language_code, target_language_code), indices in grouped_indices.items():
            prioritized_indices = self._prioritize_translation_indices(translated_boxes, indices)
            texts = [translated_boxes[index].text for index in prioritized_indices]
            translated_batch = self.translation_backend.translate_batch(
                texts,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            for index, translated_text in zip(prioritized_indices, translated_batch, strict=False):
                translated_boxes[index] = replace(translated_boxes[index], translated_text=translated_text)

        self._remember_translations(translated_boxes)
        return translated_boxes

    def _reuse_recent_translations(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        if not self._recent_translations:
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

        current_rect = (box.x, box.y, box.w, box.h)
        current_area = max(box.w * box.h, 1)
        current_text = self._normalize_text_for_matching(box.text)
        if not current_text:
            return ""

        best_score = 0.0
        best_translation = ""
        for recent in self._recent_translations:
            if not recent.translated_text:
                continue
            if recent.source_language_code != box.source_language_code:
                continue
            if recent.target_language_code != box.target_language_code:
                continue

            overlap = self._intersection_area(current_rect, (recent.x, recent.y, recent.w, recent.h)) / current_area
            if overlap < 0.35:
                continue

            recent_text = self._normalize_text_for_matching(recent.text)
            if not recent_text:
                continue

            similarity = SequenceMatcher(None, current_text, recent_text).ratio()
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
        remembered.sort(key=self._translation_priority, reverse=True)
        self._recent_translations = remembered[: max(self.settings.max_ocr_boxes_per_frame * 3, 24)]

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
        translation_status = self.translation_backend.describe()
        if self.settings.ocr_enabled and self.ocr_backend.is_available():
            ocr_status = f"{self.ocr_backend.describe()} | {self.settings.max_ocr_boxes_per_frame} boxes/frame"
            return f"{route} | {ocr_status} | {translation_status}"
        if self.settings.ocr_enabled:
            return f"{route} | {self.ocr_backend.describe()} | {translation_status}"
        return f"{route} | OCR disabled | {translation_status}"

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
