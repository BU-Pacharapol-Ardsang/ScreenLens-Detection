from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def application_roots() -> list[Path]:
    roots: list[Path] = []

    if is_frozen():
        roots.append(Path(sys.executable).resolve().parent)

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            meipass_path = Path(meipass).resolve()
            roots.append(meipass_path)
            roots.append(meipass_path.parent)

    package_root = Path(__file__).resolve().parent
    roots.extend(
        (
            package_root.parent.parent,
            package_root.parent,
            package_root,
        )
    )
    return _unique_paths(roots)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)

    return unique
