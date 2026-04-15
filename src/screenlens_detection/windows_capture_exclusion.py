from __future__ import annotations

import ctypes
import sys


WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

IS_WINDOWS = sys.platform == "win32"
USER32 = ctypes.windll.user32 if IS_WINDOWS else None


def set_window_capture_exclusion(hwnd: int, *, excluded: bool = True) -> bool:
    if not IS_WINDOWS or USER32 is None or hwnd <= 0:
        return False

    affinity = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    if USER32.SetWindowDisplayAffinity(hwnd, affinity):
        return True

    if excluded and USER32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR):
        return True

    return False
