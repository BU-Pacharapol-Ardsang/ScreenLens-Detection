from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import FrameAnalysis, PipelineSettings


@dataclass(slots=True)
class _VideoStream:
    path: Path
    fps: float
    writer: cv2.VideoWriter | None = None
    size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        normalized = _normalize_frame(frame)
        height, width = normalized.shape[:2]
        if self.writer is None:
            self.size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Unable to open recording file: {self.path}")

        if self.size is not None and (width, height) != self.size:
            normalized = cv2.resize(normalized, self.size, interpolation=cv2.INTER_AREA)

        self.writer.write(normalized)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class RecordingSession:
    def __init__(
        self,
        *,
        root: Path | str = "recordings",
        started_at: datetime | None = None,
        fps: float = 4.0,
    ) -> None:
        self.started_at = started_at or datetime.now()
        self.directory = _create_recording_directory(Path(root), self.started_at)
        self._streams = {
            "annotated": _VideoStream(self.directory / "annotated_preview.mp4", fps),
            "segmentation": _VideoStream(self.directory / "segmentation_preview.mp4", fps),
            "translated": _VideoStream(self.directory / "translated_preview.mp4", fps),
        }
        self._log_path = self.directory / "session_log.jsonl"
        self._log_file = self._log_path.open("w", encoding="utf-8")
        self._frame_index = 0
        self._closed = False
        self._write_event(
            "recording_started",
            {
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "fps": fps,
                "directory": str(self.directory),
            },
        )

    @property
    def log_path(self) -> Path:
        return self._log_path

    def write_frame(self, analysis: FrameAnalysis) -> None:
        self._ensure_open()
        translated = analysis.translated_preview
        if translated is None:
            translated = analysis.source_frame
        if translated is None:
            translated = analysis.annotated_frame

        self._streams["annotated"].write(np.asarray(analysis.annotated_frame))
        self._streams["segmentation"].write(np.asarray(analysis.processed_preview))
        self._streams["translated"].write(np.asarray(translated))
        self._frame_index += 1
        self._write_event("frame", self._frame_payload(analysis))

    def close(self) -> None:
        if self._closed:
            return

        self._write_event(
            "recording_stopped",
            {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "frames": self._frame_index,
            },
        )
        for stream in self._streams.values():
            stream.close()
        self._log_file.close()
        self._closed = True

    def _frame_payload(self, analysis: FrameAnalysis) -> dict[str, Any]:
        return {
            "frame": self._frame_index,
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
            "fps": analysis.fps,
            "monitor": analysis.monitor_label,
            "status": analysis.status,
            "ocr_runtime": analysis.ocr_runtime,
            "ocr_available": analysis.ocr_available,
            "detected_boxes": len(analysis.boxes),
            "content_offset_x": analysis.content_offset_x,
            "content_offset_y": analysis.content_offset_y,
            "content_motion_confidence": analysis.content_motion_confidence,
            "boxes": [
                {
                    "x": box.x,
                    "y": box.y,
                    "w": box.w,
                    "h": box.h,
                    "text": box.text,
                    "translated_text": box.translated_text,
                    "source_language": box.source_language_label,
                    "target_language": box.target_language_label,
                    "confidence": box.confidence,
                }
                for box in analysis.boxes
            ],
        }

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        self._log_file.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
        self._log_file.flush()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Recording session is already closed.")


def recording_fps_from_settings(settings: PipelineSettings) -> float:
    interval_ms = max(settings.capture_interval_ms, 1)
    return max(1.0, min(60.0, 1000.0 / interval_ms))


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    normalized = np.asarray(frame)
    if normalized.ndim == 2:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    if normalized.ndim != 3:
        raise ValueError("Recording frame must be a grayscale, BGR, RGB, or BGRA image.")
    if normalized.shape[2] == 4:
        return cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)
    if normalized.shape[2] != 3:
        raise ValueError("Recording frame must have 1, 3, or 4 channels.")
    if not normalized.flags["C_CONTIGUOUS"]:
        normalized = np.ascontiguousarray(normalized)
    return normalized


def _create_recording_directory(root: Path, started_at: datetime) -> Path:
    base = root / started_at.strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        directory = base if suffix == 0 else root / f"{base.name}_{suffix:02d}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return directory
    raise RuntimeError(f"Unable to create a unique recording directory under {root}")
