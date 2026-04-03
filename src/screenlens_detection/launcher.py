from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_venv_interpreter(script_path: str | Path, *, windowed: bool) -> None:
    root = Path(script_path).resolve().parent
    venv_scripts = root / ".venv" / "Scripts"
    preferred_python = venv_scripts / ("pythonw.exe" if windowed else "python.exe")
    current_executable = Path(sys.executable).resolve()

    if _is_running_from_venv(current_executable, venv_scripts):
        return

    if not preferred_python.is_file():
        _fail_launch(
            "ScreenLens-Detection requires the project .venv interpreter.\n\n"
            f"Missing: {preferred_python}\n\n"
            "Create the virtual environment and install dependencies first.",
            windowed=windowed,
        )

    if _is_debugger_session():
        _fail_launch(
            "ScreenLens-Detection was launched under a debugger with a non-.venv interpreter.\n\n"
            f"Current interpreter: {current_executable}\n"
            f"Required interpreter: {preferred_python}\n\n"
            "Select .venv\\Scripts\\python.exe as the VS Code interpreter, then run again.",
            windowed=windowed,
        )

    try:
        subprocess.Popen(
            [str(preferred_python), str(Path(script_path).resolve()), *sys.argv[1:]],
            cwd=str(root),
            env=os.environ.copy(),
        )
    except OSError as exc:
        _fail_launch(
            "ScreenLens-Detection could not relaunch itself with the project .venv interpreter.\n\n"
            f"Interpreter: {preferred_python}\n"
            f"Error: {exc}",
            windowed=windowed,
        )

    raise SystemExit(0)


def _is_running_from_venv(current_executable: Path, venv_scripts: Path) -> bool:
    try:
        return current_executable.is_relative_to(venv_scripts.resolve())
    except ValueError:
        return False


def _is_debugger_session() -> bool:
    if sys.gettrace() is not None:
        return True
    if "debugpy" in sys.modules:
        return True
    return "DEBUGPY_LAUNCHER_PORT" in os.environ


def _fail_launch(message: str, *, windowed: bool) -> None:
    if windowed and os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "ScreenLens-Detection", 0x10)
        except Exception:
            pass
    else:
        print(message, file=sys.stderr)

    raise SystemExit(1)
