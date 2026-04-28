from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import cv2
import numpy as np

from ..capture import ScreenCapturer
from ..languages import resolve_ocr_language, source_language_options, target_language_options
from ..models import FrameAnalysis, MonitorSpec, PipelineSettings
from ..overlay import TranslationOverlay
from ..overlay_tracker import OverlayTrackingWorker
from ..text_detectors import text_detector_options
from ..windows_capture_exclusion import set_window_capture_exclusion
from ..windows_hotkeys import extract_hotkey_id, hotkey_labels, register_window_hotkeys, unregister_window_hotkeys
from ..worker import ProcessingWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ScreenLens-Detection")
        self.resize(1400, 900)

        self.monitors: list[MonitorSpec] = []
        self.worker: ProcessingWorker | None = None
        self.overlay_tracker: OverlayTrackingWorker | None = None
        self.overlay_window = TranslationOverlay()
        self.overlay_active = False
        self._overlay_started_worker = False
        self._hotkeys_registered = False
        self._capture_exclusion_applied = False
        self._base_status = "Idle"

        self.monitor_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Monitors")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.hotkey_label = QLabel(f"Global hotkeys: {hotkey_labels()} toggle live screen overlay")

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(80, 10000)
        self.interval_spin.setValue(250)
        self.interval_spin.setSuffix(" ms")

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 3.0)
        self.scale_spin.setSingleStep(0.25)
        self.scale_spin.setValue(1.5)

        self.area_spin = QSpinBox()
        self.area_spin.setRange(50, 10000)
        self.area_spin.setValue(250)

        self.ocr_boxes_slider = QSlider(Qt.Orientation.Horizontal)
        self.ocr_boxes_slider.setRange(1, 256)
        self.ocr_boxes_slider.setPageStep(4)
        self.ocr_boxes_slider.setTickInterval(1)
        self.ocr_boxes_slider.setValue(16)
        self.ocr_boxes_value_label = QLabel(str(self.ocr_boxes_slider.value()))
        self.ocr_boxes_value_label.setMinimumWidth(28)
        self.ocr_boxes_control = QWidget()
        ocr_boxes_layout = QHBoxLayout(self.ocr_boxes_control)
        ocr_boxes_layout.setContentsMargins(0, 0, 0, 0)
        ocr_boxes_layout.setSpacing(8)
        ocr_boxes_layout.addWidget(self.ocr_boxes_slider, 1)
        ocr_boxes_layout.addWidget(self.ocr_boxes_value_label)

        self.source_language_combo = QComboBox()
        self.target_language_combo = QComboBox()
        self.text_detector_combo = QComboBox()
        self.translation_mode_combo = QComboBox()
        self.ocr_checkbox = QCheckBox("Enable OCR")
        self.ocr_checkbox.setChecked(True)
        self.overlay_tracking_checkbox = QCheckBox("Track overlay while scrolling")
        self.overlay_tracking_checkbox.setChecked(False)
        self.ocr_device_combo = QComboBox()

        self.fps_label = QLabel("0.0")
        self.detected_label = QLabel("0")
        self.monitor_label = QLabel("-")
        self.status_label = QLabel("Idle")
        self.ocr_runtime_label = QLabel("Not running")
        self.ocr_runtime_label.setWordWrap(True)

        self.preview_label = self._create_image_label("Annotated preview")
        self.mask_label = self._create_image_label("Segmentation preview")
        self.text_output = QPlainTextEdit()
        self.text_output.setReadOnly(True)

        self._build_ui()
        self._populate_ocr_device_control()
        self._populate_text_detector_control()
        self._populate_translation_mode_control()
        self._populate_language_controls()
        self._connect_signals()
        self._refresh_monitors()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        controls = QGroupBox("Controls")
        controls_layout = QGridLayout(controls)
        controls_layout.addWidget(QLabel("Monitor"), 0, 0)
        controls_layout.addWidget(self.monitor_combo, 0, 1)
        controls_layout.addWidget(self.refresh_button, 0, 2)
        controls_layout.addWidget(self.start_button, 0, 3)
        controls_layout.addWidget(self.stop_button, 0, 4)
        controls_layout.addWidget(self.hotkey_label, 1, 0, 1, 5)

        settings_box = QGroupBox("Pipeline Settings")
        settings_layout = QFormLayout(settings_box)
        settings_layout.addRow("Capture interval", self.interval_spin)
        settings_layout.addRow("Upscale factor", self.scale_spin)
        settings_layout.addRow("Min contour area", self.area_spin)
        settings_layout.addRow("Text detector", self.text_detector_combo)
        settings_layout.addRow("OCR boxes/frame", self.ocr_boxes_control)
        settings_layout.addRow("Source language", self.source_language_combo)
        settings_layout.addRow("Target language", self.target_language_combo)
        settings_layout.addRow("Translation mode", self.translation_mode_combo)
        settings_layout.addRow("OCR device", self.ocr_device_combo)
        settings_layout.addRow("", self.ocr_checkbox)
        settings_layout.addRow("", self.overlay_tracking_checkbox)

        stats_box = QGroupBox("Runtime Stats")
        stats_layout = QFormLayout(stats_box)
        stats_layout.addRow("FPS", self.fps_label)
        stats_layout.addRow("Detected boxes", self.detected_label)
        stats_layout.addRow("Monitor", self.monitor_label)
        stats_layout.addRow("Status", self.status_label)
        stats_layout.addRow("OCR runtime", self.ocr_runtime_label)

        top_row = QHBoxLayout()
        top_row.addWidget(controls, 3)
        top_row.addWidget(settings_box, 2)
        top_row.addWidget(stats_box, 2)

        views = QHBoxLayout()
        views.addWidget(self.preview_label, 2)
        views.addWidget(self.mask_label, 2)

        output_box = QGroupBox("Detected Text")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(self.text_output)

        root.addLayout(top_row)
        root.addLayout(views, 4)
        root.addWidget(output_box, 2)

        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self._refresh_monitors)
        self.start_button.clicked.connect(self._start_worker)
        self.stop_button.clicked.connect(self._stop_worker)
        self.ocr_boxes_slider.valueChanged.connect(self._update_ocr_boxes_label)

    def _set_runtime_controls_locked(self, locked: bool) -> None:
        self.monitor_combo.setEnabled(not locked)
        self.refresh_button.setEnabled(not locked)
        self.interval_spin.setEnabled(not locked)
        self.scale_spin.setEnabled(not locked)
        self.area_spin.setEnabled(not locked)
        self.ocr_boxes_slider.setEnabled(not locked)
        self.source_language_combo.setEnabled(not locked)
        self.target_language_combo.setEnabled(not locked)
        self.text_detector_combo.setEnabled(not locked)
        self.translation_mode_combo.setEnabled(not locked)
        self.ocr_device_combo.setEnabled(not locked)
        self.ocr_checkbox.setEnabled(not locked)
        self.overlay_tracking_checkbox.setEnabled(not locked)

    def _populate_ocr_device_control(self) -> None:
        self.ocr_device_combo.addItem("Auto", userData="auto")
        self.ocr_device_combo.addItem("CPU", userData="cpu")
        self.ocr_device_combo.addItem("GPU (NVIDIA CUDA)", userData="gpu")
        self.ocr_device_combo.setCurrentIndex(0)

    def _populate_text_detector_control(self) -> None:
        for option in text_detector_options():
            self.text_detector_combo.addItem(option.label, userData=option.code)
        self.text_detector_combo.setCurrentIndex(0)

    def _populate_translation_mode_control(self) -> None:
        self.translation_mode_combo.addItem("Argos Translate (Offline)", userData="argos")
        self.translation_mode_combo.addItem("Google Translate (Online)", userData="google")
        self.translation_mode_combo.addItem("Disabled", userData="disabled")
        self.translation_mode_combo.setCurrentIndex(0)

    def _populate_language_controls(self) -> None:
        for option in source_language_options():
            self.source_language_combo.addItem(option.label, userData=option.code)
        for option in target_language_options():
            self.target_language_combo.addItem(option.label, userData=option.code)

        self.source_language_combo.setCurrentIndex(0)
        target_index = self.target_language_combo.findData("tha")
        if target_index >= 0:
            self.target_language_combo.setCurrentIndex(target_index)

    def _refresh_monitors(self) -> None:
        with ScreenCapturer() as capturer:
            self.monitors = capturer.list_monitors()

        self.monitor_combo.clear()
        for monitor in self.monitors:
            self.monitor_combo.addItem(monitor.label, userData=monitor)

        if not self.monitors:
            self._set_base_status("No monitors detected")
            self.ocr_runtime_label.setText("Not running")
            self.start_button.setEnabled(False)
            return

        self._set_base_status("Ready")
        self.ocr_runtime_label.setText("Not running")
        self.start_button.setEnabled(True)

    def _start_worker(self) -> None:
        if self.worker is not None:
            return

        monitor = self.monitor_combo.currentData()
        if monitor is None:
            QMessageBox.warning(self, "ScreenLens-Detection", "No monitor selected.")
            return

        settings = PipelineSettings(
            capture_interval_ms=self.interval_spin.value(),
            upscale_factor=self.scale_spin.value(),
            min_contour_area=self.area_spin.value(),
            text_detector_mode=self.text_detector_combo.currentData(),
            max_ocr_boxes_per_frame=self.ocr_boxes_slider.value(),
            source_language_code=self.source_language_combo.currentData(),
            target_language_code=self.target_language_combo.currentData(),
            translation_mode=self.translation_mode_combo.currentData(),
            ocr_enabled=self.ocr_checkbox.isChecked(),
            ocr_device_preference=self.ocr_device_combo.currentData(),
            ocr_language=resolve_ocr_language(self.source_language_combo.currentData()),
            overlay_tracking_enabled=self.overlay_tracking_checkbox.isChecked(),
        )

        self.overlay_window.set_tracking_enabled(settings.overlay_tracking_enabled)
        self.worker = ProcessingWorker(monitor=monitor, settings=settings)
        self.worker.frame_ready.connect(self._handle_frame)
        self.worker.worker_error.connect(self._handle_error)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_base_status("Starting OCR/translation...")
        self.ocr_runtime_label.setText("Starting...")
        self._set_runtime_controls_locked(True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()
        if self.overlay_active and settings.overlay_tracking_enabled:
            self._start_overlay_tracker(monitor)

    def _stop_worker(self) -> None:
        self._overlay_started_worker = False
        if self.overlay_active:
            self._hide_overlay()
        if self.worker is None:
            return
        self.worker.stop()

    def _on_worker_finished(self) -> None:
        self._stop_overlay_tracker()
        self.worker = None
        self._set_runtime_controls_locked(False)
        self.start_button.setEnabled(bool(self.monitors))
        self.stop_button.setEnabled(False)
        self._overlay_started_worker = False
        if self._base_status != "Error":
            self._set_base_status("Stopped")
        self.ocr_runtime_label.setText("Stopped")

    def _handle_error(self, message: str) -> None:
        self._set_base_status("Error")
        self.ocr_runtime_label.setText("Error")
        QMessageBox.critical(self, "ScreenLens-Detection", message)
        self._stop_worker()

    def _handle_frame(self, analysis: FrameAnalysis) -> None:
        self.preview_label.setPixmap(self._frame_to_pixmap(analysis.annotated_frame, self.preview_label))
        self.mask_label.setPixmap(self._frame_to_pixmap(analysis.processed_preview, self.mask_label))

        if analysis.fps < 1.0:
            self.fps_label.setText(f"{analysis.fps:.2f}")
        else:
            self.fps_label.setText(f"{analysis.fps:.1f}")
        self.detected_label.setText(str(len(analysis.boxes)))
        self.monitor_label.setText(analysis.monitor_label or "-")
        self._set_base_status(analysis.status)
        self.ocr_runtime_label.setText(analysis.ocr_runtime or "Unavailable")

        if self.overlay_active:
            self.overlay_window.update_analysis(analysis)

        if analysis.boxes:
            lines = []
            for index, box in enumerate(analysis.boxes, start=1):
                lines.append(box.summary(index))
            self.text_output.setPlainText("\n\n".join(lines))
        else:
            self.text_output.setPlainText("No text regions detected in the current frame.")

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        self._ensure_capture_exclusion()
        self._ensure_hotkeys_registered()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_worker()
        self._unregister_hotkeys()
        self.overlay_window.close()
        super().closeEvent(event)

    def nativeEvent(self, event_type: object, message: int) -> tuple[bool, int]:
        if sys.platform == "win32":
            hotkey_id = extract_hotkey_id(message)
            if hotkey_id is not None:
                self._handle_hotkey(hotkey_id)
                return True, 0

        return super().nativeEvent(event_type, message)

    def _handle_hotkey(self, hotkey_id: int) -> None:
        if hotkey_id in {1, 2}:
            self._toggle_overlay_mode()

    def _toggle_overlay_mode(self) -> None:
        if self.overlay_active:
            self._disable_overlay_mode()
            return
        self._enable_overlay_mode()

    def _enable_overlay_mode(self) -> None:
        monitor = self.monitor_combo.currentData()
        if monitor is None:
            QMessageBox.warning(self, "ScreenLens-Detection", "No monitor selected.")
            return

        self.overlay_window.show_for_monitor(monitor)
        self.overlay_window.set_tracking_enabled(self.overlay_tracking_checkbox.isChecked())
        self.overlay_window.clear_analysis()
        self.overlay_active = True

        if self.worker is None:
            self._overlay_started_worker = True
            self._start_worker()
            return

        if self.overlay_tracking_checkbox.isChecked():
            self._start_overlay_tracker(monitor)
        self._overlay_started_worker = False
        self._refresh_status_label()

    def _disable_overlay_mode(self) -> None:
        owned_worker = self._overlay_started_worker
        self._overlay_started_worker = False
        self._hide_overlay()

        if owned_worker and self.worker is not None:
            self.worker.stop()

    def _hide_overlay(self) -> None:
        self._stop_overlay_tracker()
        self.overlay_active = False
        self.overlay_window.clear_analysis()
        self.overlay_window.hide()
        self._refresh_status_label()

    def _start_overlay_tracker(self, monitor: MonitorSpec) -> None:
        if self.overlay_tracker is not None:
            return

        self.overlay_window.set_realtime_tracking_active(True)
        self.overlay_tracker = OverlayTrackingWorker(monitor)
        self.overlay_tracker.frame_ready.connect(self._handle_overlay_tracking_frame)
        self.overlay_tracker.worker_error.connect(self._handle_overlay_tracker_error)
        self.overlay_tracker.finished.connect(self._on_overlay_tracker_finished)
        self.overlay_tracker.start()
        self._refresh_status_label()

    def _stop_overlay_tracker(self) -> None:
        tracker = self.overlay_tracker
        self.overlay_tracker = None
        self.overlay_window.set_realtime_tracking_active(False)
        if tracker is not None:
            tracker.stop()
        self._refresh_status_label()

    def _on_overlay_tracker_finished(self) -> None:
        self.overlay_tracker = None
        self.overlay_window.set_realtime_tracking_active(False)
        self._refresh_status_label()

    def _handle_overlay_tracking_frame(self, tracking_frame: object) -> None:
        if self.overlay_active:
            self.overlay_window.apply_tracking_frame(tracking_frame)

    def _handle_overlay_tracker_error(self, message: str) -> None:
        self._stop_overlay_tracker()
        self._set_base_status(f"{self._base_status} | Overlay tracking unavailable: {message}")

    def _set_base_status(self, text: str) -> None:
        self._base_status = text
        self._refresh_status_label()

    def _refresh_status_label(self) -> None:
        status = self._base_status
        if self.overlay_active:
            status = f"{status} | Overlay ON"
        if self.overlay_tracker is not None:
            status = f"{status} | Tracking ON"
        self.status_label.setText(status)

    def _ensure_hotkeys_registered(self) -> None:
        if self._hotkeys_registered or sys.platform != "win32":
            return

        failures = register_window_hotkeys(int(self.winId()))
        self._hotkeys_registered = True
        if failures:
            self.hotkey_label.setText(
                f"Some global hotkeys unavailable: {', '.join(failures)}"
            )
        else:
            self.hotkey_label.setText(
                f"Global hotkeys active: {hotkey_labels()} toggle live screen overlay"
            )

    def _ensure_capture_exclusion(self) -> None:
        if self._capture_exclusion_applied:
            return

        self._capture_exclusion_applied = set_window_capture_exclusion(int(self.winId()))

    def _unregister_hotkeys(self) -> None:
        if not self._hotkeys_registered or sys.platform != "win32":
            return

        unregister_window_hotkeys(int(self.winId()))
        self._hotkeys_registered = False

    def _update_ocr_boxes_label(self, value: int) -> None:
        self.ocr_boxes_value_label.setText(str(value))

    @staticmethod
    def _create_image_label(placeholder: str) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(520, 320)
        label.setStyleSheet(
            "QLabel { background: #111827; color: #d1d5db; border: 1px solid #374151; }"
        )
        return label

    @staticmethod
    def _frame_to_pixmap(frame: np.ndarray, target: QLabel) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        return pixmap.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
