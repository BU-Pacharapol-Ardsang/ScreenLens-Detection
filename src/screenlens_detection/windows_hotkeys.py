from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import sys


WM_HOTKEY = 0x0312


@dataclass(slots=True, frozen=True)
class HotkeySpec:
    id: int
    label: str
    modifiers: int
    virtual_key: int


if sys.platform == "win32":
    MOD_SHIFT = 0x0004
    MOD_NOREPEAT = 0x4000
    VK_F2 = 0x71
    VK_F6 = 0x75
    VK_F7 = 0x76
    USER32 = ctypes.windll.user32

    class MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("message", wintypes.UINT),
            ("wParam", wintypes.WPARAM),
            ("lParam", wintypes.LPARAM),
            ("time", wintypes.DWORD),
            ("pt", wintypes.POINT),
        ]
else:
    MOD_SHIFT = 0
    MOD_NOREPEAT = 0
    VK_F2 = 0
    VK_F6 = 0
    VK_F7 = 0
    USER32 = None
    MSG = None


HOTKEY_SPECS: tuple[HotkeySpec, ...] = (
    HotkeySpec(id=1, label="F6", modifiers=MOD_NOREPEAT, virtual_key=VK_F6),
    HotkeySpec(id=2, label="Shift+F2", modifiers=MOD_SHIFT | MOD_NOREPEAT, virtual_key=VK_F2),
    HotkeySpec(id=3, label="F7", modifiers=MOD_NOREPEAT, virtual_key=VK_F7),
)


def hotkey_labels() -> str:
    return ", ".join(spec.label for spec in HOTKEY_SPECS)


def overlay_hotkey_labels() -> str:
    return ", ".join(spec.label for spec in HOTKEY_SPECS if spec.id in {1, 2})


def hover_lock_hotkey_label() -> str:
    for spec in HOTKEY_SPECS:
        if spec.id == 3:
            return spec.label
    return "F7"


def register_window_hotkeys(hwnd: int) -> list[str]:
    if USER32 is None:
        return ["Global hotkeys are only available on Windows."]

    failed: list[str] = []
    for spec in HOTKEY_SPECS:
        if not USER32.RegisterHotKey(hwnd, spec.id, spec.modifiers, spec.virtual_key):
            failed.append(spec.label)
    return failed


def unregister_window_hotkeys(hwnd: int) -> None:
    if USER32 is None:
        return

    for spec in HOTKEY_SPECS:
        USER32.UnregisterHotKey(hwnd, spec.id)


def extract_hotkey_id(message: object) -> int | None:
    if MSG is None:
        return None

    address = _coerce_message_address(message)
    if address is None:
        return None

    msg = MSG.from_address(address)
    if msg.message != WM_HOTKEY:
        return None
    return int(msg.wParam)


def _coerce_message_address(message: object) -> int | None:
    if isinstance(message, int):
        return message

    if isinstance(message, ctypes.c_void_p):
        return int(message.value) if message.value is not None else None

    value = getattr(message, "value", None)
    if isinstance(value, int):
        return value

    try:
        return int(message)
    except (TypeError, ValueError):
        return None
