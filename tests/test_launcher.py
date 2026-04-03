from pathlib import Path

from screenlens_detection.launcher import _is_debugger_session, _is_running_from_venv


def test_is_running_from_venv_detects_scripts_directory() -> None:
    root = Path("C:/repo")
    venv_scripts = root / ".venv" / "Scripts"
    current = venv_scripts / "python.exe"

    assert _is_running_from_venv(current, venv_scripts) is True


def test_is_running_from_venv_rejects_system_python() -> None:
    root = Path("C:/repo")
    venv_scripts = root / ".venv" / "Scripts"
    current = Path("C:/Python313/python.exe")

    assert _is_running_from_venv(current, venv_scripts) is False


def test_is_debugger_session_uses_debugpy_env(monkeypatch) -> None:
    monkeypatch.setenv("DEBUGPY_LAUNCHER_PORT", "64825")

    assert _is_debugger_session() is True
