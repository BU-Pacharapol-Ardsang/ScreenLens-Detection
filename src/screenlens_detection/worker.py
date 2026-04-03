from __future__ import annotations

from time import perf_counter

from PySide6.QtCore import QThread, Signal

from .capture import ScreenCapturer
from .models import MonitorSpec, PipelineSettings
from .ocr import create_default_ocr_backend
from .pipeline import TextDetectionPipeline
from .translation import create_default_translation_backend


class ProcessingWorker(QThread):
    frame_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(self, monitor: MonitorSpec, settings: PipelineSettings) -> None:
        super().__init__()
        self.monitor = monitor
        self.settings = settings
        self._running = False
        self._smoothed_fps = 0.0

    def run(self) -> None:
        self._running = True
        capturer = ScreenCapturer()
        pipeline = TextDetectionPipeline(
            self.settings,
            create_default_ocr_backend(),
            create_default_translation_backend(),
        )

        try:
            while self._running:
                loop_started = perf_counter()
                frame = capturer.grab(self.monitor.index)
                analysis = pipeline.process(frame, monitor_label=self.monitor.label)
                analysis.fps = self._calculate_runtime_fps(loop_started)
                self.frame_ready.emit(analysis)

                remaining_ms = self.settings.capture_interval_ms - int((perf_counter() - loop_started) * 1000)
                if remaining_ms > 0:
                    self.msleep(remaining_ms)
        except Exception as exc:  # pragma: no cover - UI thread handles emitted errors
            self.worker_error.emit(str(exc))
        finally:
            capturer.close()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def _calculate_runtime_fps(self, loop_started: float) -> float:
        processing_elapsed = perf_counter() - loop_started
        effective_loop_seconds = max(processing_elapsed, self.settings.capture_interval_ms / 1000.0)
        instant_fps = 1.0 / max(effective_loop_seconds, 1e-6)

        if self._smoothed_fps == 0.0:
            self._smoothed_fps = instant_fps
        else:
            self._smoothed_fps = (self._smoothed_fps * 0.8) + (instant_fps * 0.2)

        return self._smoothed_fps
