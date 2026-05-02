import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from screenlens_detection.models import FrameAnalysis, MonitorSpec
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


def test_main_window_start_stop_preserves_selected_translation_mode(monkeypatch) -> None:
    DummyWorker.instances.clear()
    monkeypatch.setattr(main_window_module, "ScreenCapturer", FakeScreenCapturer)
    monkeypatch.setattr(main_window_module, "ProcessingWorker", DummyWorker)

    app = _app()
    window = main_window_module.MainWindow()

    try:
        google_index = window.translation_mode_combo.findData("google")
        assert google_index >= 0
        easyocr_index = window.text_detector_combo.findData("easyocr")
        assert easyocr_index >= 0
        scanline_index = window.scanline_roi_combo.findData(True)
        assert scanline_index >= 0

        window.translation_mode_combo.setCurrentIndex(google_index)
        window.text_detector_combo.setCurrentIndex(easyocr_index)
        window.scanline_roi_combo.setCurrentIndex(scanline_index)
        window.interval_spin.setValue(1000)
        window.scale_spin.setValue(1.0)
        window.detection_scale_spin.setValue(0.50)
        window.area_spin.setValue(100)
        window.ocr_boxes_slider.setValue(2)
        window.overlay_tracking_checkbox.setChecked(True)
        anchor_index = window.overlay_tracking_mode_combo.findData("anchor")
        assert anchor_index >= 0
        window.overlay_tracking_mode_combo.setCurrentIndex(anchor_index)

        window._start_worker()
        app.processEvents()

        worker = DummyWorker.instances[-1]
        assert worker.started is True
        assert worker.settings.translation_mode == "google"
        assert worker.settings.text_detector_mode == "easyocr"
        assert worker.settings.detection_scale == 0.50
        assert worker.settings.scanline_roi_enabled is True
        assert worker.settings.overlay_tracking_enabled is True
        assert worker.settings.overlay_tracking_mode == "anchor"
        assert window.worker is worker
        assert window.text_detector_combo.isEnabled() is False
        assert window.scanline_roi_combo.isEnabled() is False
        assert window.translation_mode_combo.isEnabled() is False
        assert window.overlay_tracking_checkbox.isEnabled() is False
        assert window.overlay_tracking_mode_combo.isEnabled() is False
        assert window.stop_button.isEnabled() is True
        assert window.status_label.text() == "Synthetic worker running"
        assert window.ocr_runtime_label.text() == "Synthetic OCR runtime"

        window._stop_worker()
        app.processEvents()

        assert worker.stopped is True
        assert window.worker is None
        assert window.text_detector_combo.isEnabled() is True
        assert window.scanline_roi_combo.isEnabled() is True
        assert window.translation_mode_combo.isEnabled() is True
        assert window.stop_button.isEnabled() is False
        assert window.status_label.text() == "Stopped"
    finally:
        window.close()
        app.processEvents()
