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
    capture_interval_ms: int = 250
    upscale_factor: float = 1.5
    clahe_clip_limit: float = 2.5
    clahe_grid_size: int = 8
    gaussian_kernel_size: int = 3
    threshold_block_size: int = 31
    threshold_c: int = 12
    morphology_width: int = 17
    morphology_height: int = 5
    min_contour_area: int = 250
    min_box_width: int = 40
    min_box_height: int = 18
    max_box_height_ratio: float = 0.35
    max_boxes: int = 15
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_psm: int = 6


@dataclass(slots=True, frozen=True)
class DetectionBox:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    confidence: float | None = None

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(slots=True)
class FrameAnalysis:
    annotated_frame: object
    processed_preview: object
    boxes: list[DetectionBox] = field(default_factory=list)
    status: str = ""
    fps: float = 0.0
    ocr_available: bool = False
    monitor_label: str = ""

    @property
    def detected_text(self) -> list[str]:
        return [box.text for box in self.boxes if box.text]

