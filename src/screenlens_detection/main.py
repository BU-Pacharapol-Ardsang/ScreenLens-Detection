from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ScreenLens-Detection")
    window = MainWindow()
    window.show()

##############################################################################
## Might Delete Later If arrow issues are resolved and didn't use this one ##
############################################################################
    # Install arrow-drawing proxy style AFTER stylesheets are applied.
    # Qt wraps the active style when a stylesheet is present, so applying the
    # proxy first can be overridden.
    from .ui.arrow_style import ArrowProxyStyle

    app._arrow_style = ArrowProxyStyle(app.style())  # type: ignore[attr-defined]
    app.setStyle(app._arrow_style)  # type: ignore[arg-type]
    return app.exec()

