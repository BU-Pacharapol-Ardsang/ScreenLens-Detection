from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
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


def overlay_font_pixel_size(rect_height: int) -> int:
    if rect_height <= 0:
        return 1
    return max(min(int(rect_height * 0.62), 28), 1)


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

        bubble_rect = rect.adjusted(0, 0, -1, -1)
        radius = max(min(rect.height() // 4, 8), 2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(bubble_rect, radius, radius)

        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bubble_rect, radius, radius)

        horizontal_padding = max(min(rect.height() // 4, 12), 2)
        vertical_padding = max(min(rect.height() // 8, 6), 1)
        text_rect = bubble_rect.adjusted(
            horizontal_padding,
            vertical_padding,
            -horizontal_padding,
            -vertical_padding,
        )
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            text_rect = bubble_rect

        font = self._font_for_text(text, text_rect, overlay_font_pixel_size(rect.height()))
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    @staticmethod
    def _font_for_text(text: str, rect: QRect, max_pixel_size: int) -> QFont:
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap
        font = QFont("Segoe UI")
        font.setWeight(QFont.Weight.DemiBold)
        for pixel_size in range(max(max_pixel_size, 1), 0, -1):
            font.setPixelSize(pixel_size)
            bounds = QFontMetrics(font).boundingRect(rect, flags, text)
            if bounds.height() <= rect.height() and bounds.width() <= rect.width():
                return font
        font.setPixelSize(1)
        return font

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
