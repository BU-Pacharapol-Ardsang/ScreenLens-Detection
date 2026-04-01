from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .models import DetectionBox, FrameAnalysis, PipelineSettings
from .ocr import OCRBackend


class TextDetectionPipeline:
    """Realtime screen-text pipeline using traditional CV and optional OCR."""

    def __init__(self, settings: PipelineSettings, ocr_backend: OCRBackend) -> None:
        self.settings = settings
        self.ocr_backend = ocr_backend

    def process(self, frame: np.ndarray, *, monitor_label: str = "") -> FrameAnalysis:
        started = perf_counter()

        scaled_frame, scale = self._scale_frame(frame)
        enhanced_gray = self._enhance_grayscale(scaled_frame)
        mask = self._build_text_mask(enhanced_gray)
        working_boxes = self._extract_text_boxes(mask, scaled_frame.shape)
        boxes = self._annotate_with_ocr(working_boxes, enhanced_gray, frame.shape, scale)

        annotated = self._draw_annotations(frame.copy(), boxes)
        processed_preview = self._draw_mask_preview(mask, working_boxes)

        elapsed = max(perf_counter() - started, 1e-6)
        return FrameAnalysis(
            annotated_frame=annotated,
            processed_preview=processed_preview,
            boxes=boxes,
            status=self._status_message(),
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
        combined = cv2.bitwise_or(dark_text_mask, light_text_mask)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(self.settings.morphology_width, 3), max(self.settings.morphology_height, 3)),
        )
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        segmented = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, open_kernel)
        return segmented

    def _extract_text_boxes(
        self,
        mask: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_box_height = int(frame_shape[0] * self.settings.max_box_height_ratio)
        frame_area = frame_shape[0] * frame_shape[1]

        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / max(h, 1)

            if area < self.settings.min_contour_area:
                continue
            if area > int(frame_area * 0.65):
                continue
            if w < self.settings.min_box_width or h < self.settings.min_box_height:
                continue
            if h > max_box_height:
                continue
            if not 0.6 <= aspect_ratio <= 25.0:
                continue

            candidates.append((x, y, w, h))

        candidates.sort(key=lambda box: (box[1], box[0]))
        return candidates[: self.settings.max_boxes]

    def _annotate_with_ocr(
        self,
        working_boxes: list[tuple[int, int, int, int]],
        enhanced_gray: np.ndarray,
        original_shape: tuple[int, int, int],
        scale: float,
    ) -> list[DetectionBox]:
        detected_boxes: list[DetectionBox] = []

        for x, y, w, h in working_boxes:
            crop = enhanced_gray[y : y + h, x : x + w]
            text = ""
            if self.settings.ocr_enabled and self.ocr_backend.is_available():
                ocr_result = self.ocr_backend.recognize(
                    crop,
                    language=self.settings.ocr_language,
                    psm=self.settings.ocr_psm,
                )
                text = ocr_result.text

            mapped_x = int(x / scale)
            mapped_y = int(y / scale)
            mapped_w = int(w / scale)
            mapped_h = int(h / scale)

            mapped_w = min(mapped_w, original_shape[1] - mapped_x)
            mapped_h = min(mapped_h, original_shape[0] - mapped_y)

            detected_boxes.append(
                DetectionBox(
                    x=max(mapped_x, 0),
                    y=max(mapped_y, 0),
                    w=max(mapped_w, 1),
                    h=max(mapped_h, 1),
                    text=text,
                )
            )

        return detected_boxes

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
        if self.settings.ocr_enabled and self.ocr_backend.is_available():
            return f"OCR enabled via {self.ocr_backend.describe()}"
        if self.settings.ocr_enabled:
            return self.ocr_backend.describe()
        return "OCR disabled in app settings"

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

