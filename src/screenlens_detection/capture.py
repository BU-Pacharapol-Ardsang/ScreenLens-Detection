from __future__ import annotations

from mss import mss
import cv2
import numpy as np

from .models import MonitorSpec


class ScreenCapturer:
    """Thin wrapper around MSS to enumerate monitors and grab frames."""

    def __init__(self) -> None:
        self._sct = mss()

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def list_monitors(self) -> list[MonitorSpec]:
        monitors: list[MonitorSpec] = []
        for index, monitor in enumerate(self._sct.monitors[1:], start=1):
            monitors.append(
                MonitorSpec(
                    index=index,
                    label=f"Monitor {index} ({monitor['width']}x{monitor['height']})",
                    left=monitor["left"],
                    top=monitor["top"],
                    width=monitor["width"],
                    height=monitor["height"],
                )
            )
        return monitors

    def grab(self, monitor_index: int) -> np.ndarray:
        if monitor_index <= 0 or monitor_index >= len(self._sct.monitors):
            raise ValueError(f"Monitor index {monitor_index} is out of range.")

        frame = np.array(self._sct.grab(self._sct.monitors[monitor_index]), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()

