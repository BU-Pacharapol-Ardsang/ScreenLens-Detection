from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .models import DetectionBox, FrameAnalysis, MonitorSpec
from .windows_capture_exclusion import set_window_capture_exclusion


@dataclass(slots=True, frozen=True)
class OverlayBox:
    x: int
    y: int
    w: int
    h: int
    text: str


def overlay_text_for_box(box: DetectionBox) -> str:
    translated = " ".join(box.translated_text.split())
    if translated:
        return translated
    return " ".join(box.text.split())


def scale_overlay_rect(
    box: DetectionBox,
    *,
    overlay_width: int,
    overlay_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    if frame_width <= 0 or frame_height <= 0:
        return box.x, box.y, box.w, box.h

    scale_x = overlay_width / frame_width
    scale_y = overlay_height / frame_height
    return (
        max(int(box.x * scale_x), 0),
        max(int(box.y * scale_y), 0),
        max(int(box.w * scale_x), 1),
        max(int(box.h * scale_y), 1),
    )


class TranslationOverlay(QWidget):
    def __init__(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        if hasattr(Qt.WindowType, "WindowTransparentForInput"):
            flags |= Qt.WindowType.WindowTransparentForInput

        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._monitor: MonitorSpec | None = None
        self._overlay_boxes: list[OverlayBox] = []
        self._capture_exclusion_applied = False

    def show_for_monitor(self, monitor: MonitorSpec) -> None:
        self._monitor = monitor
        self._apply_monitor_geometry(monitor)
        self.show()
        self._ensure_capture_exclusion()
        self.raise_()

    def clear_analysis(self) -> None:
        self._overlay_boxes = []
        self.update()

    def update_analysis(self, analysis: FrameAnalysis) -> None:
        self._overlay_boxes = [
            OverlayBox(box.x, box.y, box.w, box.h, text)
            for box in analysis.boxes
            if (text := overlay_text_for_box(box))
        ]
        self.update()

    def paintEvent(self, _event: object) -> None:
        if not self._overlay_boxes or self._monitor is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        frame_width = max(self._monitor.width, 1)
        frame_height = max(self._monitor.height, 1)

        for box in self._overlay_boxes:
            scaled = scale_overlay_rect(
                DetectionBox(x=box.x, y=box.y, w=box.w, h=box.h, text=box.text),
                overlay_width=max(self.width(), 1),
                overlay_height=max(self.height(), 1),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            self._paint_box(painter, QRect(*scaled), box.text)

    def _paint_box(self, painter: QPainter, rect: QRect, text: str) -> None:
        accent = QColor(48, 231, 149, 220)
        background = QColor(15, 23, 42, 212)
        text_color = QColor(248, 250, 252)

        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 10, 10)

        min_bubble_width = 180
        bubble_width = min(max(rect.width(), min_bubble_width), max(self.width() - rect.left() - 12, 80))
        bubble_height = min(max(rect.height(), 44), max(self.height() - rect.top() - 12, 36))
        bubble_rect = QRect(rect.left(), rect.top(), bubble_width, bubble_height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(bubble_rect, 10, 10)

        font_size = max(min(int(bubble_rect.height() * 0.42), 28), 11)
        painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
        painter.setPen(text_color)
        painter.drawText(
            bubble_rect.adjusted(12, 8, -12, -8),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _apply_monitor_geometry(self, monitor: MonitorSpec) -> None:
        app = QApplication.instance()
        if app is not None:
            screens = app.screens()
            screen_index = monitor.index - 1
            if 0 <= screen_index < len(screens):
                self.setGeometry(screens[screen_index].geometry())
                return

        self.setGeometry(monitor.left, monitor.top, monitor.width, monitor.height)

    def _ensure_capture_exclusion(self) -> None:
        if self._capture_exclusion_applied:
            return

        self._capture_exclusion_applied = set_window_capture_exclusion(int(self.winId()))
