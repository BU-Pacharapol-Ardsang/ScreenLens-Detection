from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from .capture import ScreenCapturer
from .cursor import monitor_relative_cursor_position
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
        self._hover_lock = Lock()
        self._hover_candidate_position: tuple[int, int] | None = None
        self._hover_candidate_started_at: float | None = None
        self._confirmed_hover_position: tuple[int, int] | None = None

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
                analysis = pipeline.process(
                    frame,
                    monitor_label=self.monitor.label,
                    cursor_position=self._hover_cursor_position(),
                )
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

    def set_translation_region_mode(self, mode: str) -> None:
        normalized = (mode or "full").casefold().strip()
        if normalized in {"hover", "cursor", "hover_region"}:
            self.settings.translation_region_mode = "hover"
            return

        self.settings.translation_region_mode = "full"
        self.reset_hover_target()

    def reset_hover_target(self) -> None:
        with self._hover_lock:
            self._reset_hover_target_locked()

    def hover_target_confirmed(self) -> bool:
        with self._hover_lock:
            return self._confirmed_hover_position is not None

    def _reset_hover_target_locked(self) -> None:
        self._hover_candidate_position = None
        self._hover_candidate_started_at = None
        self._confirmed_hover_position = None

    def _hover_cursor_position(self) -> tuple[int, int] | None:
        if (self.settings.translation_region_mode or "full").casefold().strip() != "hover":
            self.reset_hover_target()
            return None

        position = monitor_relative_cursor_position(self.monitor)
        with self._hover_lock:
            return self._confirmed_hover_position_locked(position, perf_counter())

    def _confirmed_hover_position_locked(
        self,
        position: tuple[int, int] | None,
        now: float,
    ) -> tuple[int, int] | None:
        if position is None:
            self._reset_hover_target_locked()
            return None

        tolerance = max(int(self.settings.hover_move_tolerance), 1)
        if (
            self._hover_candidate_position is None
            or self._cursor_distance(position, self._hover_candidate_position) > tolerance
        ):
            self._hover_candidate_position = position
            self._hover_candidate_started_at = now
            self._confirmed_hover_position = None
            return None

        dwell_seconds = max(self.settings.hover_dwell_ms, 0) / 1000.0
        if self._hover_candidate_started_at is None:
            self._hover_candidate_started_at = now
            return None

        if now - self._hover_candidate_started_at >= dwell_seconds:
            self._confirmed_hover_position = self._hover_candidate_position

        return self._confirmed_hover_position

    @staticmethod
    def _cursor_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)

    def _calculate_runtime_fps(self, loop_started: float) -> float:
        processing_elapsed = perf_counter() - loop_started
        effective_loop_seconds = max(processing_elapsed, self.settings.capture_interval_ms / 1000.0)
        instant_fps = 1.0 / max(effective_loop_seconds, 1e-6)

        if self._smoothed_fps == 0.0:
            self._smoothed_fps = instant_fps
        else:
            self._smoothed_fps = (self._smoothed_fps * 0.8) + (instant_fps * 0.2)

        return self._smoothed_fps
