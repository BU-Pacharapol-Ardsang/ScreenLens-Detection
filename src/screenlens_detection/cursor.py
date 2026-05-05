from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys

from .models import MonitorSpec


if sys.platform == "win32":
    USER32 = ctypes.windll.user32

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

else:
    USER32 = None
    POINT = None


def screen_cursor_position() -> tuple[int, int] | None:
    if USER32 is None or POINT is None:
        return None

    point = POINT()
    if not USER32.GetCursorPos(ctypes.byref(point)):
        return None
    return int(point.x), int(point.y)


def monitor_relative_cursor_position(monitor: MonitorSpec) -> tuple[int, int] | None:
    position = screen_cursor_position()
    if position is None:
        return None

    x = position[0] - monitor.left
    y = position[1] - monitor.top
    if x < 0 or y < 0 or x >= monitor.width or y >= monitor.height:
        return None
    return x, y
