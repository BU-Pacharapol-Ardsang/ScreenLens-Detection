import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from screenlens_detection.models import DetectionBox, FrameAnalysis, MonitorSpec
from screenlens_detection.ui import main_window as main_window_module


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FakeScreenCapturer:
    def __enter__(self) -> "FakeScreenCapturer":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def list_monitors(self) -> list[MonitorSpec]:
        return [
            MonitorSpec(
                index=1,
                label="Monitor 1 (1920x1080)",
                left=0,
                top=0,
                width=1920,
                height=1080,
            )
        ]


class DummyWorker(QObject):
    frame_ready = Signal(object)
    worker_error = Signal(str)
    finished = Signal()
    instances: list["DummyWorker"] = []

    def __init__(self, monitor: MonitorSpec, settings: object) -> None:
        super().__init__()
        self.monitor = monitor
        self.settings = settings
        self.started = False
        self.stopped = False
        self.hover_reset_count = 0
        self._hover_confirmed = False
        DummyWorker.instances.append(self)

    def start(self) -> None:
        self.started = True
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        analysis = FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            boxes=[],
            status="Synthetic worker running",
            ocr_runtime="Synthetic OCR runtime",
            fps=1.25,
            ocr_available=True,
            monitor_label=self.monitor.label,
        )
        self.frame_ready.emit(analysis)

    def stop(self) -> None:
        self.stopped = True
        self.finished.emit()

    def set_translation_region_mode(self, mode: str) -> None:
        self.settings.translation_region_mode = "hover" if mode == "hover" else "full"

    def reset_hover_target(self) -> None:
        self.hover_reset_count += 1
        self._hover_confirmed = False

    def hover_target_confirmed(self) -> bool:
        return self._hover_confirmed


class DummyOverlay:
    def __init__(self) -> None:
        self.shown_for: MonitorSpec | None = None
        self.hidden = False
        self.closed = False
        self.cleared = False
        self.tracking_mode: str | None = None
        self.tracking_enabled = False
        self.realtime_tracking_active = False
        self.render_mode: str | None = None
        self.clean_patch_options: dict[str, int] = {}

    def show_for_monitor(self, monitor: MonitorSpec) -> None:
        self.shown_for = monitor
        self.hidden = False

    def hide(self) -> None:
        self.hidden = True

    def close(self) -> None:
        self.closed = True

    def clear_analysis(self) -> None:
        self.cleared = True

    def update_analysis(self, _analysis: object) -> None:
        return None

    def set_tracking_mode(self, mode: str | None) -> None:
        self.tracking_mode = mode

    def set_tracking_enabled(self, enabled: bool) -> None:
        self.tracking_enabled = enabled

    def set_render_mode(self, mode: str | None) -> None:
        self.render_mode = mode

    def set_clean_patch_options(
        self,
        *,
        padding_px: int,
        mask_dilate_px: int,
        inpaint_radius: int,
        max_crop_area: int,
    ) -> None:
        self.clean_patch_options = {
            "padding_px": padding_px,
            "mask_dilate_px": mask_dilate_px,
            "inpaint_radius": inpaint_radius,
            "max_crop_area": max_crop_area,
        }

    def set_realtime_tracking_active(self, active: bool) -> None:
        self.realtime_tracking_active = active

    def apply_tracking_frame(self, _tracking_frame: object) -> None:
        return None


class DummyOverlayTrackingWorker(QObject):
    frame_ready = Signal(object)
    worker_error = Signal(str)
    finished = Signal()

    def __init__(self, monitor: MonitorSpec) -> None:
        super().__init__()
        self.monitor = monitor
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.finished.emit()


def test_clean_patch_preview_does_not_fallback_to_bubble_when_inpaint_skips() -> None:
    _app()
    frame = np.full((90, 180, 3), 220, dtype=np.uint8)
    cv2_text_color = (35, 35, 35)

    cv2.putText(frame, "OLD", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cv2_text_color, 2, cv2.LINE_AA)
    rendered = main_window_module.MainWindow._translated_preview_frame(
        FrameAnalysis(
            annotated_frame=frame,
            processed_preview=frame,
            source_frame=frame,
            boxes=[DetectionBox(x=20, y=22, w=90, h=34, text="OLD", translated_text="ใหม่")],
        ),
        render_mode="clean_patch",
        clean_patch_max_crop_area=1,
    )

    assert np.mean(rendered[24, 22]) > 160


def test_main_window_start_stop_preserves_selected_translation_mode(monkeypatch) -> None:
    DummyWorker.instances.clear()
    monkeypatch.setattr(main_window_module, "ScreenCapturer", FakeScreenCapturer)
    monkeypatch.setattr(main_window_module, "ProcessingWorker", DummyWorker)
    monkeypatch.setattr(main_window_module, "TranslationOverlay", DummyOverlay)
    monkeypatch.setattr(main_window_module, "OverlayTrackingWorker", DummyOverlayTrackingWorker)

    app = _app()
    window = main_window_module.MainWindow()

    try:
        google_index = window.translation_mode_combo.findData("google")
        assert google_index >= 0
        easyocr_index = window.text_detector_combo.findData("easyocr")
        assert easyocr_index >= 0
        rapidocr_ocr_index = window.ocr_backend_combo.findData("rapidocr")
        assert rapidocr_ocr_index >= 0
        scanline_index = window.scanline_roi_combo.findData(True)
        assert scanline_index >= 0
        hover_region_index = window.translation_region_mode_combo.findData("hover")
        assert hover_region_index >= 0
        strict_block_index = window.translation_block_mode_combo.findData("strict")
        assert strict_block_index >= 0
        clean_patch_index = window.subtitle_render_mode_combo.findData("clean_patch")
        assert clean_patch_index >= 0

        window.translation_mode_combo.setCurrentIndex(google_index)
        window.text_detector_combo.setCurrentIndex(easyocr_index)
        window.ocr_backend_combo.setCurrentIndex(rapidocr_ocr_index)
        window.scanline_roi_combo.setCurrentIndex(scanline_index)
        window.translation_region_mode_combo.setCurrentIndex(hover_region_index)
        window.translation_block_mode_combo.setCurrentIndex(strict_block_index)
        window.subtitle_render_mode_combo.setCurrentIndex(clean_patch_index)
        window.translation_stability_checkbox.setChecked(False)
        window.interval_spin.setValue(1000)
        window.scale_spin.setValue(1.0)
        window.detection_scale_spin.setValue(0.50)
        window.area_spin.setValue(100)
        window.ocr_boxes_slider.setValue(2)
        window.overlay_tracking_checkbox.setChecked(True)
        window.runtime_debug_checkbox.setChecked(True)
        window.annotated_preview_checkbox.setChecked(False)
        window.segmentation_preview_checkbox.setChecked(False)
        window.translated_preview_checkbox.setChecked(False)
        anchor_index = window.overlay_tracking_mode_combo.findData("anchor")
        assert anchor_index >= 0
        window.overlay_tracking_mode_combo.setCurrentIndex(anchor_index)

        window._start_worker()
        app.processEvents()

        worker = DummyWorker.instances[-1]
        assert worker.started is True
        assert worker.settings.translation_mode == "google"
        assert worker.settings.text_detector_mode == "easyocr"
        assert worker.settings.ocr_backend_mode == "rapidocr"
        assert worker.settings.detection_scale == 0.50
        assert worker.settings.scanline_roi_enabled is True
        assert worker.settings.translation_region_mode == "hover"
        assert worker.settings.translation_block_mode == "strict"
        assert worker.settings.translation_similarity_stability_enabled is False
        assert worker.settings.subtitle_render_mode == "clean_patch"
        assert worker.settings.overlay_tracking_enabled is True
        assert worker.settings.overlay_tracking_mode == "anchor"
        assert worker.settings.runtime_debug_enabled is True
        assert worker.settings.annotated_preview_enabled is False
        assert worker.settings.segmentation_preview_enabled is False
        assert worker.settings.translated_preview_enabled is False
        assert window.overlay_window.render_mode == "clean_patch"
        assert window.worker is worker
        assert window.text_detector_combo.isEnabled() is False
        assert window.ocr_backend_combo.isEnabled() is False
        assert window.scanline_roi_combo.isEnabled() is False
        assert window.translation_mode_combo.isEnabled() is False
        assert window.translation_region_mode_combo.isEnabled() is False
        assert window.translation_block_mode_combo.isEnabled() is False
        assert window.subtitle_render_mode_combo.isEnabled() is False
        assert window.translation_stability_checkbox.isEnabled() is False
        assert window.overlay_tracking_checkbox.isEnabled() is False
        assert window.runtime_debug_checkbox.isEnabled() is False
        assert window.annotated_preview_checkbox.isEnabled() is True
        assert window.segmentation_preview_checkbox.isEnabled() is True
        assert window.translated_preview_checkbox.isEnabled() is True
        assert window.overlay_tracking_mode_combo.isEnabled() is False
        assert window.stop_button.isEnabled() is True
        assert window.status_label.text() == "Synthetic worker running"
        assert window.ocr_runtime_label.text() == "Synthetic OCR runtime"

        window._handle_hotkey(3)
        assert worker.settings.translation_region_mode == "hover"
        assert window.overlay_active is True
        assert "Hover target ON" in window.status_label.text()

        window._handle_hotkey(3)
        assert worker.hover_reset_count >= 2
        assert "Hover target ON" in window.status_label.text()

        window._stop_worker()
        app.processEvents()

        assert worker.stopped is True
        assert window.worker is None
        assert window.text_detector_combo.isEnabled() is True
        assert window.ocr_backend_combo.isEnabled() is True
        assert window.scanline_roi_combo.isEnabled() is True
        assert window.translation_mode_combo.isEnabled() is True
        assert window.translation_region_mode_combo.isEnabled() is True
        assert window.translation_block_mode_combo.isEnabled() is True
        assert window.subtitle_render_mode_combo.isEnabled() is True
        assert window.translation_stability_checkbox.isEnabled() is True
        assert window.stop_button.isEnabled() is False
        assert window.status_label.text() == "Stopped"
        assert window.runtime_debug_checkbox.isEnabled() is True
    finally:
        window.close()
        app.processEvents()


def test_runtime_debug_text_formats_slowest_stage() -> None:
    text = main_window_module.MainWindow._format_runtime_debug_text(
        {
            "scale_frame": 1.25,
            "opencv_detection": 8.5,
            "translation": 3.0,
            "total": 14.0,
        }
    )

    assert "total 14.0 ms" in text
    assert "slowest opencv detect 8.5 ms" in text
    assert "translation: 3.0 ms" in text
    assert main_window_module.MainWindow._format_runtime_debug_text({}) == "Off"


def test_main_window_f7_starts_hover_overlay_and_arms_hover_target(monkeypatch) -> None:
    DummyWorker.instances.clear()
    monkeypatch.setattr(main_window_module, "ScreenCapturer", FakeScreenCapturer)
    monkeypatch.setattr(main_window_module, "ProcessingWorker", DummyWorker)
    monkeypatch.setattr(main_window_module, "TranslationOverlay", DummyOverlay)
    monkeypatch.setattr(main_window_module, "OverlayTrackingWorker", DummyOverlayTrackingWorker)

    app = _app()
    window = main_window_module.MainWindow()

    try:
        assert window.translation_region_mode_combo.currentData() == "full"

        window._handle_hotkey(3)
        app.processEvents()

        worker = DummyWorker.instances[-1]
        assert worker.started is True
        assert worker.settings.translation_region_mode == "hover"
        assert window.translation_region_mode_combo.currentData() == "hover"
        assert window.overlay_active is True
        assert window._hover_target_mode_active is True
        assert window.overlay_window.shown_for == worker.monitor
        assert "Hover target ON" in window.status_label.text()
    finally:
        window.close()
        app.processEvents()
