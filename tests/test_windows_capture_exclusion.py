from screenlens_detection import windows_capture_exclusion


def test_set_window_capture_exclusion_prefers_exclude_from_capture(monkeypatch) -> None:
    class FakeUser32:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def SetWindowDisplayAffinity(self, hwnd: int, affinity: int) -> int:
            self.calls.append((hwnd, affinity))
            return 1

    fake_user32 = FakeUser32()
    monkeypatch.setattr(windows_capture_exclusion, "IS_WINDOWS", True)
    monkeypatch.setattr(windows_capture_exclusion, "USER32", fake_user32)

    assert windows_capture_exclusion.set_window_capture_exclusion(321) is True
    assert fake_user32.calls == [(321, windows_capture_exclusion.WDA_EXCLUDEFROMCAPTURE)]


def test_set_window_capture_exclusion_falls_back_to_monitor_affinity(monkeypatch) -> None:
    class FakeUser32:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def SetWindowDisplayAffinity(self, hwnd: int, affinity: int) -> int:
            self.calls.append((hwnd, affinity))
            if affinity == windows_capture_exclusion.WDA_EXCLUDEFROMCAPTURE:
                return 0
            return 1

    fake_user32 = FakeUser32()
    monkeypatch.setattr(windows_capture_exclusion, "IS_WINDOWS", True)
    monkeypatch.setattr(windows_capture_exclusion, "USER32", fake_user32)

    assert windows_capture_exclusion.set_window_capture_exclusion(654) is True
    assert fake_user32.calls == [
        (654, windows_capture_exclusion.WDA_EXCLUDEFROMCAPTURE),
        (654, windows_capture_exclusion.WDA_MONITOR),
    ]


def test_set_window_capture_exclusion_ignores_invalid_handles(monkeypatch) -> None:
    monkeypatch.setattr(windows_capture_exclusion, "IS_WINDOWS", True)
    monkeypatch.setattr(windows_capture_exclusion, "USER32", object())

    assert windows_capture_exclusion.set_window_capture_exclusion(0) is False
