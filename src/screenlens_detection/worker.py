from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from .capture import ScreenCapturer
from .models import MonitorSpec, PipelineSettings
from .ocr import create_default_ocr_backend
from .pipeline import TextDetectionPipeline
from .translation import create_default_translation_backend


class _LatestFrameQueue:
    def __init__(self) -> None:
        self._queue: Queue[object] = Queue(maxsize=1)
        self.dropped_frames = 0

    def put(self, frame: object) -> None:
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                    self.dropped_frames += 1
                except Empty:
                    continue

    def get(self, timeout: float) -> object:
        return self._queue.get(timeout=timeout)


class ProcessingWorker(QThread):
    frame_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(self, monitor: MonitorSpec, settings: PipelineSettings) -> None:
        super().__init__()
        self.monitor = monitor
        self.settings = settings
        self._running = False
        self._smoothed_fps = 0.0
        self._stop_event: Event | None = None

    def run(self) -> None:
        self._running = True
        stop_event = Event()
        self._stop_event = stop_event
        frame_queue = _LatestFrameQueue()
        pipeline = TextDetectionPipeline(
            self.settings,
            create_default_ocr_backend(device_preference=self.settings.ocr_device_preference),
            create_default_translation_backend(mode=self.settings.translation_mode),
        )
        capture_thread = Thread(
            target=self._capture_loop,
            args=(frame_queue, stop_event),
            name="ScreenLensCapture",
            daemon=True,
        )
        capture_thread.start()

        try:
            while self._running and not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.10)
                except Empty:
                    continue

                loop_started = perf_counter()
                analysis = pipeline.process(frame, monitor_label=self.monitor.label)
                analysis.fps = self._calculate_runtime_fps(loop_started)
                self.frame_ready.emit(analysis)
        except Exception as exc:  # pragma: no cover - UI thread handles emitted errors
            self.worker_error.emit(str(exc))
        finally:
            self._running = False
            stop_event.set()
            capture_thread.join(timeout=2.0)
            pipeline.close()
            self._stop_event = None

    def _capture_loop(self, frame_queue: _LatestFrameQueue, stop_event: Event) -> None:
        capturer = ScreenCapturer()
        try:
            while self._running and not stop_event.is_set():
                loop_started = perf_counter()
                frame_queue.put(capturer.grab(self.monitor.index))

                remaining_seconds = self.settings.capture_interval_ms / 1000.0 - (perf_counter() - loop_started)
                if remaining_seconds > 0:
                    stop_event.wait(remaining_seconds)
        except Exception as exc:  # pragma: no cover - UI thread handles emitted errors
            self._running = False
            stop_event.set()
            self.worker_error.emit(str(exc))
        finally:
            capturer.close()

    def stop(self) -> None:
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
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
