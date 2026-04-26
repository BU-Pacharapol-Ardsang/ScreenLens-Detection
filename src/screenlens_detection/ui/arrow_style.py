##############################################################################
## Might Delete Later If arrow issues are resolved and didn't use this one ##
############################################################################

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QProxyStyle, QStyle


_ARROW_COLOR = "#090b0e"


class ArrowProxyStyle(QProxyStyle):
    """Ensure combo/spin arrows remain visible across platforms.

    We draw simple caret-style arrows using polygons/lines instead of relying on
    the platform style or QSS 'image:' URLs, which can be inconsistent on
    Windows.
    """

    def drawPrimitive(self, element: QStyle.PrimitiveElement, option, painter: QPainter, widget=None) -> None:  # type: ignore[override]
        if element in {
            QStyle.PrimitiveElement.PE_IndicatorSpinUp,
            QStyle.PrimitiveElement.PE_IndicatorSpinDown,
            QStyle.PrimitiveElement.PE_IndicatorArrowDown,
        }:
            self._draw_caret_arrow(element, option, painter)
            return

        super().drawPrimitive(element, option, painter, widget)

    def _draw_caret_arrow(self, element: QStyle.PrimitiveElement, option, painter: QPainter) -> None:
        rect = option.rect
        if rect.isNull():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(QColor(_ARROW_COLOR))
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        # Inset to avoid clipping.
        left = rect.left() + rect.width() * 0.30
        right = rect.left() + rect.width() * 0.70
        top = rect.top() + rect.height() * 0.36
        bottom = rect.top() + rect.height() * 0.64
        mid_x = rect.left() + rect.width() * 0.50

        if element == QStyle.PrimitiveElement.PE_IndicatorSpinUp:
            p1 = QPointF(left, bottom)
            p2 = QPointF(mid_x, top)
            p3 = QPointF(right, bottom)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
        else:
            # SpinDown and ArrowDown
            p1 = QPointF(left, top)
            p2 = QPointF(mid_x, bottom)
            p3 = QPointF(right, top)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)

        painter.restore()
