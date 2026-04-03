from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screenlens_detection.launcher import ensure_venv_interpreter
from screenlens_detection.main import main


if __name__ == "__main__":
    ensure_venv_interpreter(__file__, windowed=True)
    raise SystemExit(main())
