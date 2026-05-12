import json
from datetime import datetime

import numpy as np

from screenlens_detection.models import DetectionBox, FrameAnalysis, PipelineSettings
from screenlens_detection.recording import RecordingSession, recording_fps_from_settings


class FakeVideoWriter:
    instances = []

    def __init__(self, path: str, _fourcc: int, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        FakeVideoWriter.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def test_recording_session_writes_three_streams_and_jsonl_log(tmp_path, monkeypatch) -> None:
    FakeVideoWriter.instances.clear()
    monkeypatch.setattr("screenlens_detection.recording.cv2.VideoWriter", FakeVideoWriter)
    monkeypatch.setattr("screenlens_detection.recording.cv2.VideoWriter_fourcc", lambda *_args: 1)

    session = RecordingSession(
        root=tmp_path,
        started_at=datetime(2026, 4, 29, 3, 4, 5),
        fps=5.0,
    )
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    translated = frame.copy()
    translated[:, :, 1] = 255

    session.write_frame(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=np.zeros((6, 10), dtype=np.uint8),
            source_frame=frame,
            translated_preview=translated,
            boxes=[
                DetectionBox(
                    x=1,
                    y=2,
                    w=3,
                    h=4,
                    text="Hello",
                    translated_text="สวัสดี",
                    confidence=92.0,
                )
            ],
            status="running",
            ocr_runtime="test OCR",
            fps=5.0,
            ocr_available=True,
            monitor_label="Monitor 1",
            runtime_timings_ms={"total": 12.5},
        )
    )
    session.close()

    assert session.directory == tmp_path / "20260429_030405"
    assert [writer.size for writer in FakeVideoWriter.instances] == [(20, 12), (10, 6), (20, 12)]
    assert all(writer.released for writer in FakeVideoWriter.instances)
    assert all(len(writer.frames) == 1 for writer in FakeVideoWriter.instances)

    events = [json.loads(line) for line in session.log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["recording_started", "frame", "recording_stopped"]
    assert events[1]["status"] == "running"
    assert events[1]["runtime_timings_ms"] == {"total": 12.5}
    assert events[1]["boxes"][0]["translated_text"] == "สวัสดี"


def test_recording_session_falls_back_to_source_when_previews_disabled(tmp_path, monkeypatch) -> None:
    FakeVideoWriter.instances.clear()
    monkeypatch.setattr("screenlens_detection.recording.cv2.VideoWriter", FakeVideoWriter)
    monkeypatch.setattr("screenlens_detection.recording.cv2.VideoWriter_fourcc", lambda *_args: 1)

    session = RecordingSession(
        root=tmp_path,
        started_at=datetime(2026, 4, 29, 4, 5, 6),
        fps=5.0,
    )
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :, 2] = 255

    session.write_frame(
        FrameAnalysis(
            annotated_frame=None,
            processed_preview=None,
            source_frame=frame,
            translated_preview=None,
            status="running",
            fps=5.0,
            monitor_label="Monitor 1",
        )
    )
    session.close()

    assert [writer.size for writer in FakeVideoWriter.instances] == [(20, 12), (20, 12), (20, 12)]
    assert all(np.array_equal(writer.frames[0], frame) for writer in FakeVideoWriter.instances)


def test_recording_fps_from_capture_interval_is_clamped() -> None:
    assert recording_fps_from_settings(PipelineSettings(capture_interval_ms=250)) == 4.0
    assert recording_fps_from_settings(PipelineSettings(capture_interval_ms=5000)) == 1.0
    assert recording_fps_from_settings(PipelineSettings(capture_interval_ms=10)) == 60.0
