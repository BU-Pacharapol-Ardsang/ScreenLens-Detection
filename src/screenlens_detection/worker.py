from __future__ import annotations

from time import perf_counter

from PySide6.QtCore import QThread, Signal

from .capture import ScreenCapturer
from .models import MonitorSpec, PipelineSettings
from .ocr import create_default_ocr_backend
from .pipeline import TextDetectionPipeline


class ProcessingWorker(QThread):
    frame_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(self, monitor: MonitorSpec, settings: PipelineSettings) -> None:
        super().__init__()
        self.monitor = monitor
        self.settings = settings
        self._running = False

    def run(self) -> None:
        self._running = True
        capturer = ScreenCapturer()
        pipeline = TextDetectionPipeline(self.settings, create_default_ocr_backend())

        try:
            while self._running:
                loop_started = perf_counter()
                frame = capturer.grab(self.monitor.index)
                analysis = pipeline.process(frame, monitor_label=self.monitor.label)
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

