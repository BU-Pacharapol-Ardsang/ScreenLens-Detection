from queue import Empty

import pytest

from screenlens_detection import worker as worker_module
from screenlens_detection.models import MonitorSpec, PipelineSettings
from screenlens_detection.worker import ProcessingWorker, _LatestFrameQueue


def test_latest_frame_queue_drops_stale_frame() -> None:
    frame_queue = _LatestFrameQueue()

    frame_queue.put("first")
    frame_queue.put("second")

    assert frame_queue.get(timeout=0.01) == "second"
    assert frame_queue.dropped_frames == 1

    with pytest.raises(Empty):
        frame_queue.get(timeout=0.01)


def _monitor() -> MonitorSpec:
    return MonitorSpec(index=1, label="Monitor 1", left=100, top=200, width=800, height=600)


def test_processing_worker_confirms_hover_cursor_after_dwell(monkeypatch) -> None:
    now = 10.0
    monkeypatch.setattr(worker_module, "monitor_relative_cursor_position", lambda _monitor: (40, 50))
    monkeypatch.setattr(worker_module, "perf_counter", lambda: now)
    worker = ProcessingWorker(
        _monitor(),
        PipelineSettings(translation_region_mode="hover", hover_dwell_ms=1000),
    )

    assert worker._hover_cursor_position() is None
    now = 10.5
    assert worker._hover_cursor_position() is None
    now = 11.0
    assert worker._hover_cursor_position() == (40, 50)
    assert worker.hover_target_confirmed() is True


def test_processing_worker_ignores_hover_cursor_in_full_mode(monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "monitor_relative_cursor_position", lambda _monitor: (40, 50))
    worker = ProcessingWorker(_monitor(), PipelineSettings(translation_region_mode="full"))

    assert worker._hover_cursor_position() is None


def test_processing_worker_resets_hover_confirmation_when_cursor_moves(monkeypatch) -> None:
    positions = [(40, 50), (40, 50), (80, 90), (80, 90)]
    times = [10.0, 11.1, 11.2, 12.3]

    def fake_cursor_position(_monitor: MonitorSpec) -> tuple[int, int]:
        return positions.pop(0)

    monkeypatch.setattr(worker_module, "perf_counter", lambda: times.pop(0))
    monkeypatch.setattr(worker_module, "monitor_relative_cursor_position", fake_cursor_position)
    worker = ProcessingWorker(
        _monitor(),
        PipelineSettings(translation_region_mode="hover", hover_dwell_ms=1000),
    )

    assert worker._hover_cursor_position() is None
    assert worker._hover_cursor_position() == (40, 50)
    assert worker.hover_target_confirmed() is True
    assert worker._hover_cursor_position() is None
    assert worker.hover_target_confirmed() is False
    assert worker._hover_cursor_position() == (80, 90)
