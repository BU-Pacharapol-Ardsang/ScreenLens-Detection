from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class MonitorSpec:
    index: int
    label: str
    left: int
    top: int
    width: int
    height: int

    @property
    def region(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class PipelineSettings:
    capture_interval_ms: int = 40  # Run up to ~25 FPS
    upscale_factor: float = 1.0  # Full screen game/article is already large, 1.0 reduces lag massively
    detection_scale: float = 0.66  # Run detection on a smaller image, then OCR high-res source crops.
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 4
    gaussian_kernel_size: int = 3
    threshold_block_size: int = 21  # Smaller block size runs faster
    threshold_c: int = 8
    morphology_width: int = 11
    morphology_height: int = 3
    min_contour_area: int = 150
    min_box_width: int = 20
    min_box_height: int = 10
    max_box_height_ratio: float = 0.22
    max_boxes: int = 60
    text_detector_mode: str = "opencv"
    scanline_roi_enabled: bool = False
    scanline_roi_band_count: int = 6
    scanline_roi_overlap_ratio: float = 0.18
    source_language_code: str = "auto"
    target_language_code: str = "tha"
    translation_mode: str = "argos"
    translation_region_mode: str = "full"
    hover_region_radius: int = 260
    hover_box_margin: int = 96
    translation_block_mode: str = "line"
    translation_similarity_stability_enabled: bool = True
    translation_similarity_threshold: float = 0.92
    translation_similarity_min_chars: int = 16
    ocr_enabled: bool = True
    ocr_device_preference: str = "auto"
    ocr_language: str = "tha+eng+jpn"
    ocr_psm: int = 7
    max_ocr_boxes_per_frame: int = 12
    stable_ocr_frames: int = 1
    stable_box_iou_threshold: float = 0.45
    motion_filter_enabled: bool = False
    motion_mean_threshold: float = 18.0
    motion_changed_ratio_threshold: float = 0.20
    overlay_tracking_enabled: bool = False
    overlay_tracking_mode: str = "legacy"


@dataclass(slots=True, frozen=True)
class DetectionBox:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    translated_text: str = ""
    source_language_code: str = "unknown"
    source_language_label: str = "Unknown"
    target_language_code: str = "tha"
    target_language_label: str = "Thai"
    confidence: float | None = None

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def summary(self, index: int) -> str:
        before = self.text or "<region detected>"
        after = self._translated_display_text(before)
        return "\n".join(
            (
                f"[{index}] x={self.x}, y={self.y}, w={self.w}, h={self.h}",
                f"source: {self.source_language_label}",
                f"target: {self.target_language_label}",
                f"before: {before}",
                f"after: {after}",
            )
        )

    def _translated_display_text(self, before: str) -> str:
        if self.translated_text:
            return self.translated_text
        if before != "<region detected>" and self.source_language_code == self.target_language_code:
            return before
        if before == "<region detected>":
            return "<translation unavailable>"
        return "<translation pending>"


@dataclass(slots=True)
class FrameAnalysis:
    annotated_frame: object
    processed_preview: object
    boxes: list[DetectionBox] = field(default_factory=list)
    status: str = ""
    ocr_runtime: str = ""
    fps: float = 0.0
    ocr_available: bool = False
    monitor_label: str = ""
    content_offset_x: float = 0.0
    content_offset_y: float = 0.0
    content_motion_confidence: float = 0.0
    source_frame: object | None = None
    translated_preview: object | None = None

    @property
    def detected_text(self) -> list[str]:
        return [box.text for box in self.boxes if box.text]
