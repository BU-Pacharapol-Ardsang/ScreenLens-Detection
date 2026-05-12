import ctypes

from screenlens_detection.windows_hotkeys import (
    MSG,
    WM_HOTKEY,
    _coerce_message_address,
    extract_hotkey_id,
    overlay_hotkey_labels,
)


def test_coerce_message_address_from_c_void_p() -> None:
    pointer = ctypes.c_void_p(123456)

    assert _coerce_message_address(pointer) == 123456


def test_extract_hotkey_id_from_int_address() -> None:
    msg = MSG()
    msg.message = WM_HOTKEY
    msg.wParam = 2

    assert extract_hotkey_id(ctypes.addressof(msg)) == 2


def test_extract_hotkey_id_from_void_pointer() -> None:
    msg = MSG()
    msg.message = WM_HOTKEY
    msg.wParam = 1

    pointer = ctypes.c_void_p(ctypes.addressof(msg))
    assert extract_hotkey_id(pointer) == 1


def test_hotkey_label_uses_f6_overlay_toggle() -> None:
    assert overlay_hotkey_labels() == "F6"
