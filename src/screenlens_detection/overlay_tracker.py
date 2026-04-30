from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .capture import ScreenCapturer
from .models import MonitorSpec
from .motion import estimate_grayscale_offset


@dataclass(slots=True, frozen=True)
class TrackingFrame:
    gray_frame: np.ndarray
    frame_scale: float = 1.0
    global_offset_x: float = 0.0
    global_offset_y: float = 0.0
    global_confidence: float = 0.0


class OverlayTrackingWorker(QThread):
    frame_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(self, monitor: MonitorSpec, *, interval_ms: int = 120) -> None:
        super().__init__()
        self.monitor = monitor
        self.interval_ms = max(interval_ms, 90)
        self._running = False

    def run(self) -> None:
        self._running = True
        capturer = ScreenCapturer()
        previous_gray: np.ndarray | None = None

        try:
            while self._running:
                loop_started = perf_counter()
                frame = capturer.grab(self.monitor.index)
                current_gray, frame_scale = self._prepare_tracking_frame(frame)

                offset_x = 0.0
                offset_y = 0.0
                confidence = 0.0
                if previous_gray is not None:
                    offset_x, offset_y, confidence = estimate_grayscale_offset(
                        previous_gray,
                        current_gray,
                        source_scale=frame_scale,
                        max_dimension=320,
                        min_response=0.10,
                        max_offset_ratio=0.30,
                    )

                self.frame_ready.emit(TrackingFrame(current_gray, frame_scale, offset_x, offset_y, confidence))

                previous_gray = current_gray
                remaining_ms = self.interval_ms - int((perf_counter() - loop_started) * 1000)
                if remaining_ms > 0:
                    self.msleep(remaining_ms)
        except Exception as exc:  # pragma: no cover - UI thread handles emitted errors
            self.worker_error.emit(str(exc))
        finally:
            capturer.close()

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    @staticmethod
    def _prepare_tracking_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        frame_scale = min(1.0, 640.0 / max(height, width))
        if frame_scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(int(width * frame_scale), 16), max(int(height * frame_scale), 16)),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.GaussianBlur(gray, (3, 3), 0), frame_scale
