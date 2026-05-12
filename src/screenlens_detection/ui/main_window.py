from __future__ import annotations

import sys

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap
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
from ..ocr import ocr_backend_options
from ..overlay import TranslationOverlay, overlay_font_pixel_size, overlay_text_for_box
from ..overlay_tracker import OverlayTrackingWorker
from ..recording import RecordingSession, recording_fps_from_settings
from ..subtitle_cleaner import clean_patch_for_box, normalize_subtitle_render_mode
from ..text_detectors import text_detector_options
from ..windows_hotkeys import (
    extract_hotkey_id,
    hover_lock_hotkey_label,
    overlay_hotkey_labels,
    register_window_hotkeys,
    unregister_window_hotkeys,
)
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
        self._hover_target_mode_active = False
        self._hotkeys_registered = False
        self._base_status = "Idle"
        defaults = PipelineSettings()

        self.monitor_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Monitors")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.record_button = QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        self.hotkey_label = QLabel(self._hotkey_help_text(prefix="Global hotkeys"))

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 10000)
        self.interval_spin.setValue(defaults.capture_interval_ms)
        self.interval_spin.setSuffix(" ms")

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 3.0)
        self.scale_spin.setSingleStep(0.25)
        self.scale_spin.setValue(defaults.upscale_factor)

        self.detection_scale_spin = QDoubleSpinBox()
        self.detection_scale_spin.setRange(0.40, 1.00)
        self.detection_scale_spin.setDecimals(2)
        self.detection_scale_spin.setSingleStep(0.05)
        self.detection_scale_spin.setValue(defaults.detection_scale)

        self.area_spin = QSpinBox()
        self.area_spin.setRange(50, 10000)
        self.area_spin.setValue(defaults.min_contour_area)

        self.ocr_boxes_slider = QSlider(Qt.Orientation.Horizontal)
        self.ocr_boxes_slider.setRange(1, 256)
        self.ocr_boxes_slider.setPageStep(4)
        self.ocr_boxes_slider.setTickInterval(1)
        self.ocr_boxes_slider.setValue(defaults.max_ocr_boxes_per_frame)
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
        self.scanline_roi_combo = QComboBox()
        self.translation_mode_combo = QComboBox()
        self.translation_region_mode_combo = QComboBox()
        self.translation_block_mode_combo = QComboBox()
        self.subtitle_render_mode_combo = QComboBox()
        self.ocr_backend_combo = QComboBox()
        self.translation_stability_checkbox = QCheckBox("Text similarity stability")
        self.translation_stability_checkbox.setChecked(defaults.translation_similarity_stability_enabled)
        self.ocr_checkbox = QCheckBox("Enable OCR")
        self.ocr_checkbox.setChecked(True)
        self.overlay_tracking_checkbox = QCheckBox("Track overlay while scrolling")
        self.overlay_tracking_checkbox.setChecked(False)
        self.runtime_debug_checkbox = QCheckBox("Runtime debug timings")
        self.runtime_debug_checkbox.setChecked(defaults.runtime_debug_enabled)
        self.overlay_tracking_mode_combo = QComboBox()
        self.ocr_device_combo = QComboBox()

        self.fps_label = QLabel("0.0")
        self.detected_label = QLabel("0")
        self.monitor_label = QLabel("-")
        self.status_label = QLabel("Idle")
        self.recording_label = QLabel("Off")
        self.ocr_runtime_label = QLabel("Not running")
        self.ocr_runtime_label.setWordWrap(True)
        self.runtime_debug_label = QLabel("Off")
        self.runtime_debug_label.setWordWrap(True)
        self.runtime_debug_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.runtime_debug_label.setStyleSheet("QLabel { font-family: Consolas, monospace; }")

        self.preview_label = self._create_image_label("Annotated preview")
        self.mask_label = self._create_image_label("Segmentation preview")
        self.translated_preview_label = self._create_image_label("Translated preview")
        self.text_output = QPlainTextEdit()
        self.text_output.setReadOnly(True)
        self._recording_session: RecordingSession | None = None

        self._build_ui()
        self._populate_ocr_backend_control()
        self._populate_ocr_device_control()
        self._populate_text_detector_control()
        self._populate_scanline_roi_control()
        self._populate_translation_mode_control()
        self._populate_translation_region_mode_control()
        self._populate_translation_block_mode_control()
        self._populate_subtitle_render_mode_control()
        self._populate_overlay_tracking_mode_control()
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
        controls_layout.addWidget(self.record_button, 0, 5)
        controls_layout.addWidget(self.hotkey_label, 1, 0, 1, 6)

        settings_box = QGroupBox("Pipeline Settings")
        settings_layout = QFormLayout(settings_box)
        settings_layout.addRow("Capture interval", self.interval_spin)
        settings_layout.addRow("Upscale factor", self.scale_spin)
        settings_layout.addRow("Detector scale", self.detection_scale_spin)
        settings_layout.addRow("Min contour area", self.area_spin)
        settings_layout.addRow("Text detector", self.text_detector_combo)
        settings_layout.addRow("Scan mode", self.scanline_roi_combo)
        settings_layout.addRow("New OCR/frame", self.ocr_boxes_control)
        settings_layout.addRow("Source language", self.source_language_combo)
        settings_layout.addRow("Target language", self.target_language_combo)
        settings_layout.addRow("Translation mode", self.translation_mode_combo)
        settings_layout.addRow("Translation region", self.translation_region_mode_combo)
        settings_layout.addRow("Translation grouping", self.translation_block_mode_combo)
        settings_layout.addRow("Subtitle style", self.subtitle_render_mode_combo)
        settings_layout.addRow("", self.translation_stability_checkbox)
        settings_layout.addRow("OCR backend", self.ocr_backend_combo)
        settings_layout.addRow("OCR device", self.ocr_device_combo)
        settings_layout.addRow("", self.ocr_checkbox)
        settings_layout.addRow("", self.overlay_tracking_checkbox)
        settings_layout.addRow("", self.runtime_debug_checkbox)
        settings_layout.addRow("Overlay tracking", self.overlay_tracking_mode_combo)

        stats_box = QGroupBox("Runtime Stats")
        stats_layout = QFormLayout(stats_box)
        stats_layout.addRow("FPS", self.fps_label)
        stats_layout.addRow("Active boxes", self.detected_label)
        stats_layout.addRow("Monitor", self.monitor_label)
        stats_layout.addRow("Status", self.status_label)
        stats_layout.addRow("Recording", self.recording_label)
        stats_layout.addRow("OCR runtime", self.ocr_runtime_label)
        stats_layout.addRow("Pipeline debug", self.runtime_debug_label)

        top_row = QHBoxLayout()
        top_row.addWidget(controls, 3)
        top_row.addWidget(settings_box, 2)
        top_row.addWidget(stats_box, 2)

        views = QHBoxLayout()
        views.addWidget(self.preview_label, 1)
        views.addWidget(self.mask_label, 1)
        views.addWidget(self.translated_preview_label, 1)

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
        self.record_button.clicked.connect(self._toggle_recording)
        self.ocr_boxes_slider.valueChanged.connect(self._update_ocr_boxes_label)

    def _set_runtime_controls_locked(self, locked: bool) -> None:
        self.monitor_combo.setEnabled(not locked)
        self.refresh_button.setEnabled(not locked)
        self.interval_spin.setEnabled(not locked)
        self.scale_spin.setEnabled(not locked)
        self.detection_scale_spin.setEnabled(not locked)
        self.area_spin.setEnabled(not locked)
        self.ocr_boxes_slider.setEnabled(not locked)
        self.source_language_combo.setEnabled(not locked)
        self.target_language_combo.setEnabled(not locked)
        self.text_detector_combo.setEnabled(not locked)
        self.scanline_roi_combo.setEnabled(not locked)
        self.translation_mode_combo.setEnabled(not locked)
        self.translation_region_mode_combo.setEnabled(not locked)
        self.translation_block_mode_combo.setEnabled(not locked)
        self.subtitle_render_mode_combo.setEnabled(not locked)
        self.ocr_backend_combo.setEnabled(not locked)
        self.translation_stability_checkbox.setEnabled(not locked)
        self.ocr_device_combo.setEnabled(not locked)
        self.ocr_checkbox.setEnabled(not locked)
        self.overlay_tracking_checkbox.setEnabled(not locked)
        self.runtime_debug_checkbox.setEnabled(not locked)
        self.overlay_tracking_mode_combo.setEnabled(not locked)

    def _populate_ocr_backend_control(self) -> None:
        for option in ocr_backend_options():
            self.ocr_backend_combo.addItem(option.label, userData=option.code)
        self.ocr_backend_combo.setCurrentIndex(0)

    def _populate_ocr_device_control(self) -> None:
        self.ocr_device_combo.addItem("Auto", userData="auto")
        self.ocr_device_combo.addItem("CPU", userData="cpu")
        self.ocr_device_combo.addItem("GPU (NVIDIA CUDA)", userData="gpu")
        self.ocr_device_combo.setCurrentIndex(0)

    def _populate_text_detector_control(self) -> None:
        for option in text_detector_options():
            self.text_detector_combo.addItem(option.label, userData=option.code)
        self.text_detector_combo.setCurrentIndex(0)

    def _populate_scanline_roi_control(self) -> None:
        self.scanline_roi_combo.addItem("Full frame", userData=False)
        self.scanline_roi_combo.addItem("Sliding bands (video/game)", userData=True)
        self.scanline_roi_combo.setCurrentIndex(0)

    def _populate_translation_mode_control(self) -> None:
        self.translation_mode_combo.addItem("Argos Translate (Offline)", userData="argos")
        self.translation_mode_combo.addItem("Google Translate (Online)", userData="google")
        self.translation_mode_combo.addItem("Disabled", userData="disabled")
        self.translation_mode_combo.setCurrentIndex(0)

    def _populate_translation_region_mode_control(self) -> None:
        self.translation_region_mode_combo.addItem("Full screen", userData="full")
        self.translation_region_mode_combo.addItem("Hover cursor region", userData="hover")
        self.translation_region_mode_combo.setCurrentIndex(0)

    def _populate_translation_block_mode_control(self) -> None:
        self.translation_block_mode_combo.addItem("Line mode", userData="line")
        self.translation_block_mode_combo.addItem("Block mode: Strict", userData="strict")
        self.translation_block_mode_combo.setCurrentIndex(0)

    def _populate_subtitle_render_mode_control(self) -> None:
        self.subtitle_render_mode_combo.addItem("Bubble overlay", userData="bubble")
        self.subtitle_render_mode_combo.addItem("Clean patch (experimental)", userData="clean_patch")
        self.subtitle_render_mode_combo.setCurrentIndex(0)

    def _populate_overlay_tracking_mode_control(self) -> None:
        self.overlay_tracking_mode_combo.addItem("Legacy motion", userData="legacy")
        self.overlay_tracking_mode_combo.addItem("Visual anchor lock", userData="anchor")
        self.overlay_tracking_mode_combo.setCurrentIndex(0)

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
            self.runtime_debug_label.setText("Off")
            self.start_button.setEnabled(False)
            return

        self._set_base_status("Ready")
        self.ocr_runtime_label.setText("Not running")
        self.runtime_debug_label.setText("Off")
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
            detection_scale=self.detection_scale_spin.value(),
            min_contour_area=self.area_spin.value(),
            text_detector_mode=self.text_detector_combo.currentData(),
            scanline_roi_enabled=bool(self.scanline_roi_combo.currentData()),
            max_ocr_boxes_per_frame=self.ocr_boxes_slider.value(),
            source_language_code=self.source_language_combo.currentData(),
            target_language_code=self.target_language_combo.currentData(),
            translation_mode=self.translation_mode_combo.currentData(),
            translation_region_mode=self.translation_region_mode_combo.currentData(),
            translation_block_mode=self.translation_block_mode_combo.currentData(),
            translation_similarity_stability_enabled=self.translation_stability_checkbox.isChecked(),
            subtitle_render_mode=self.subtitle_render_mode_combo.currentData(),
            ocr_enabled=self.ocr_checkbox.isChecked(),
            ocr_backend_mode=self.ocr_backend_combo.currentData(),
            ocr_device_preference=self.ocr_device_combo.currentData(),
            ocr_language=resolve_ocr_language(self.source_language_combo.currentData()),
            overlay_tracking_enabled=self.overlay_tracking_checkbox.isChecked(),
            overlay_tracking_mode=self.overlay_tracking_mode_combo.currentData(),
            runtime_debug_enabled=self.runtime_debug_checkbox.isChecked(),
        )

        self._apply_overlay_render_options(settings)
        self.overlay_window.set_tracking_mode(settings.overlay_tracking_mode)
        self.overlay_window.set_tracking_enabled(settings.overlay_tracking_enabled)
        self.worker = ProcessingWorker(monitor=monitor, settings=settings)
        self.worker.frame_ready.connect(self._handle_frame)
        self.worker.worker_error.connect(self._handle_error)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_base_status("Starting OCR/translation...")
        self.ocr_runtime_label.setText("Starting...")
        self.runtime_debug_label.setText("Waiting for first frame..." if settings.runtime_debug_enabled else "Off")
        self._set_runtime_controls_locked(True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.worker.start()
        if self.overlay_active and settings.overlay_tracking_enabled:
            self._start_overlay_tracker(monitor)

    def _apply_overlay_render_options(self, settings: PipelineSettings) -> None:
        self.overlay_window.set_render_mode(settings.subtitle_render_mode)
        self.overlay_window.set_clean_patch_options(
            padding_px=settings.clean_patch_padding_px,
            mask_dilate_px=settings.clean_patch_mask_dilate_px,
            inpaint_radius=settings.clean_patch_inpaint_radius,
            max_crop_area=settings.clean_patch_max_crop_area,
        )

    def _stop_worker(self) -> None:
        self._overlay_started_worker = False
        self._stop_recording()
        if self.overlay_active:
            self._hide_overlay()
        if self.worker is None:
            return
        self.worker.stop()

    def _on_worker_finished(self) -> None:
        self._stop_recording()
        self._stop_overlay_tracker()
        self.worker = None
        self._set_runtime_controls_locked(False)
        self.start_button.setEnabled(bool(self.monitors))
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self._overlay_started_worker = False
        self._hover_target_mode_active = False
        if self._base_status != "Error":
            self._set_base_status("Stopped")
        self.ocr_runtime_label.setText("Stopped")
        self.runtime_debug_label.setText("Off")

    def _handle_error(self, message: str) -> None:
        self._set_base_status("Error")
        self.ocr_runtime_label.setText("Error")
        self.runtime_debug_label.setText("Error")
        QMessageBox.critical(self, "ScreenLens-Detection", message)
        self._stop_worker()

    def _handle_frame(self, analysis: FrameAnalysis) -> None:
        render_settings = self.worker.settings if self.worker is not None else PipelineSettings()
        translated_preview = self._translated_preview_frame(
            analysis,
            render_mode=render_settings.subtitle_render_mode,
            clean_patch_padding_px=render_settings.clean_patch_padding_px,
            clean_patch_mask_dilate_px=render_settings.clean_patch_mask_dilate_px,
            clean_patch_inpaint_radius=render_settings.clean_patch_inpaint_radius,
            clean_patch_max_crop_area=render_settings.clean_patch_max_crop_area,
        )
        analysis.translated_preview = translated_preview
        self.preview_label.setPixmap(self._frame_to_pixmap(analysis.annotated_frame, self.preview_label))
        self.mask_label.setPixmap(self._frame_to_pixmap(analysis.processed_preview, self.mask_label))
        self.translated_preview_label.setPixmap(
            self._frame_to_pixmap(translated_preview, self.translated_preview_label)
        )

        if analysis.fps < 1.0:
            self.fps_label.setText(f"{analysis.fps:.2f}")
        else:
            self.fps_label.setText(f"{analysis.fps:.1f}")
        self.detected_label.setText(str(len(analysis.boxes)))
        self.monitor_label.setText(analysis.monitor_label or "-")
        self._set_base_status(analysis.status)
        self.ocr_runtime_label.setText(analysis.ocr_runtime or "Unavailable")
        self.runtime_debug_label.setText(self._format_runtime_debug_text(analysis.runtime_timings_ms))

        if self.overlay_active:
            self.overlay_window.update_analysis(analysis)

        if self._recording_session is not None:
            try:
                self._recording_session.write_frame(analysis)
            except Exception as exc:
                self._handle_recording_error(str(exc))

        if analysis.boxes:
            lines = []
            for index, box in enumerate(analysis.boxes, start=1):
                lines.append(box.summary(index))
            self.text_output.setPlainText("\n\n".join(lines))
        else:
            self.text_output.setPlainText("No text regions detected in the current frame.")

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
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
        elif hotkey_id == 3:
            self._toggle_hover_target_mode()

    def _toggle_overlay_mode(self) -> None:
        if self.overlay_active:
            self._disable_overlay_mode()
            return
        self._enable_overlay_mode()

    def _toggle_hover_target_mode(self) -> None:
        if self._hover_target_mode_active:
            if self.worker is not None:
                self.worker.reset_hover_target()
            self._set_base_status("Hover target mode active")
            return

        self._enable_hover_target_mode()

    def _enable_hover_target_mode(self) -> None:
        if not self._select_hover_translation_region():
            self._set_base_status("Hover cursor region mode is unavailable")
            return

        if self.worker is not None:
            self.worker.set_translation_region_mode("hover")
            self.worker.reset_hover_target()

        if self.overlay_active:
            if self.worker is None:
                self._start_worker()
        else:
            self._enable_overlay_mode()

        if self.worker is None or not self.overlay_active:
            self._hover_target_mode_active = False
            self._set_base_status("Start processing before hover target mode")
            return

        self._hover_target_mode_active = True
        self.worker.set_translation_region_mode("hover")
        self.worker.reset_hover_target()
        self._set_base_status("Hover target mode active")

    def _select_hover_translation_region(self) -> bool:
        if self.translation_region_mode_combo.currentData() == "hover":
            return True

        hover_index = self.translation_region_mode_combo.findData("hover")
        if hover_index < 0:
            return False

        self.translation_region_mode_combo.setCurrentIndex(hover_index)
        return True

    def _enable_overlay_mode(self) -> None:
        monitor = self.monitor_combo.currentData()
        if monitor is None:
            QMessageBox.warning(self, "ScreenLens-Detection", "No monitor selected.")
            return

        render_settings = (
            self.worker.settings
            if self.worker is not None
            else PipelineSettings(subtitle_render_mode=self.subtitle_render_mode_combo.currentData())
        )
        self._apply_overlay_render_options(render_settings)
        self.overlay_window.show_for_monitor(monitor)
        self.overlay_window.set_tracking_mode(self.overlay_tracking_mode_combo.currentData())
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
        self._hover_target_mode_active = False
        self._hide_overlay()

        if owned_worker and self.worker is not None:
            self.worker.stop()

    def _hide_overlay(self) -> None:
        self._stop_overlay_tracker()
        self.overlay_active = False
        self._hover_target_mode_active = False
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
        if self._hover_target_mode_active:
            status = f"{status} | Hover target ON"
        if self.worker is not None and self.worker.hover_target_confirmed():
            status = f"{status} | Hover confirmed"
        if self._recording_session is not None:
            status = f"{status} | Recording ON"
        self.status_label.setText(status)

    def _toggle_recording(self) -> None:
        if self._recording_session is not None:
            self._stop_recording()
            return
        self._start_recording()

    def _start_recording(self) -> None:
        if self.worker is None:
            QMessageBox.warning(self, "ScreenLens-Detection", "Start processing before recording.")
            return

        try:
            self._recording_session = RecordingSession(fps=recording_fps_from_settings(self.worker.settings))
        except Exception as exc:
            self._handle_recording_error(str(exc))
            return

        self.record_button.setText("Stop Recording")
        self.recording_label.setText(str(self._recording_session.directory))
        self._refresh_status_label()

    def _stop_recording(self) -> None:
        session = self._recording_session
        self._recording_session = None
        if session is not None:
            session.close()
        self.record_button.setText("Start Recording")
        self.recording_label.setText("Off")
        self._refresh_status_label()

    def _handle_recording_error(self, message: str) -> None:
        session = self._recording_session
        self._recording_session = None
        if session is not None:
            session.close()
        self.record_button.setText("Start Recording")
        self.recording_label.setText("Error")
        self._refresh_status_label()
        QMessageBox.critical(self, "ScreenLens-Detection", f"Recording failed: {message}")

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
            self.hotkey_label.setText(self._hotkey_help_text(prefix="Global hotkeys active"))

    def _unregister_hotkeys(self) -> None:
        if not self._hotkeys_registered or sys.platform != "win32":
            return

        unregister_window_hotkeys(int(self.winId()))
        self._hotkeys_registered = False

    @staticmethod
    def _hotkey_help_text(*, prefix: str) -> str:
        return (
            f"{prefix}: {overlay_hotkey_labels()} toggle live screen overlay; "
            f"{hover_lock_hotkey_label()} start hover translate overlay"
        )

    def _update_ocr_boxes_label(self, value: int) -> None:
        self.ocr_boxes_value_label.setText(str(value))

    @staticmethod
    def _format_runtime_debug_text(timings_ms: dict[str, float]) -> str:
        if not timings_ms:
            return "Off"

        total_ms = timings_ms.get("total")
        stage_items = [(stage, value) for stage, value in timings_ms.items() if stage != "total"]
        slowest_stage, slowest_ms = max(stage_items, key=lambda item: item[1], default=("total", total_ms or 0.0))

        header_parts = []
        if total_ms is not None:
            header_parts.append(f"total {total_ms:.1f} ms")
        header_parts.append(f"slowest {MainWindow._runtime_stage_label(slowest_stage)} {slowest_ms:.1f} ms")

        lines = [" | ".join(header_parts)]
        for stage, value in stage_items:
            lines.append(f"{MainWindow._runtime_stage_label(stage)}: {value:.1f} ms")
        return "\n".join(lines)

    @staticmethod
    def _runtime_stage_label(stage: str) -> str:
        labels = {
            "scale_frame": "scale",
            "enhance_grayscale": "enhance",
            "full_frame_ocr": "full OCR",
            "hover_detection": "hover detect",
            "scanline_detection": "scanline detect",
            "opencv_detection": "opencv detect",
            "deep_text_detection": "deep detect",
            "ocr_grayscale": "OCR gray",
            "ocr_box_stability": "OCR stability",
            "motion_filter": "motion filter",
            "ocr_annotation": "OCR",
            "motion_offset": "motion offset",
            "translation": "translation",
            "cache_update": "cache",
            "state_update": "state",
            "draw_annotations": "draw boxes",
            "draw_mask_preview": "draw mask",
            "source_frame_copy": "source copy",
            "runtime_metadata": "metadata",
            "total": "total",
        }
        return labels.get(stage, stage.replace("_", " "))

    @staticmethod
    def _translated_preview_frame(
        analysis: FrameAnalysis,
        *,
        render_mode: str = "bubble",
        clean_patch_padding_px: int = 8,
        clean_patch_mask_dilate_px: int = 4,
        clean_patch_inpaint_radius: int = 3,
        clean_patch_max_crop_area: int = 120_000,
    ) -> np.ndarray:
        frame = analysis.source_frame
        if frame is None:
            frame = analysis.annotated_frame
        preview = np.asarray(frame).copy()
        if preview.ndim != 3 or preview.shape[2] != 3:
            return np.asarray(analysis.annotated_frame).copy()

        normalized_render_mode = normalize_subtitle_render_mode(render_mode)
        clean_text_rects: dict[tuple[int, int, int, int], QRect] = {}
        if normalized_render_mode == "clean_patch":
            for box in analysis.boxes:
                if not " ".join(box.translated_text.split()):
                    continue
                rect = (box.x, box.y, box.w, box.h)
                patch = clean_patch_for_box(
                    preview,
                    rect,
                    padding_px=clean_patch_padding_px,
                    mask_dilate_px=clean_patch_mask_dilate_px,
                    inpaint_radius=clean_patch_inpaint_radius,
                    max_crop_area=clean_patch_max_crop_area,
                )
                if patch is not None:
                    left, top, patch_width, patch_height = patch.rect
                    preview[top : top + patch_height, left : left + patch_width] = patch.image
                    clean_text_rects[rect] = QRect(left, top, patch_width, patch_height)
                else:
                    clean_text_rects[rect] = TranslationOverlay._fallback_clean_text_rect(
                        QRect(box.x, box.y, box.w, box.h),
                        bounds_width=preview.shape[1],
                        bounds_height=preview.shape[0],
                    )

        rgb = cv2.cvtColor(np.ascontiguousarray(preview), cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for box in analysis.boxes:
                text = overlay_text_for_box(box)
                if not text:
                    continue
                anchor_rect = QRect(box.x, box.y, box.w, box.h)
                bubble_rect = TranslationOverlay._expanded_bubble_rect(
                    anchor_rect,
                    text,
                    bounds_width=width,
                    bounds_height=height,
                )
                if normalized_render_mode == "clean_patch":
                    if not " ".join(box.translated_text.split()):
                        continue
                    MainWindow._paint_clean_preview_text(
                        painter,
                        clean_text_rects.get((box.x, box.y, box.w, box.h), anchor_rect),
                        text,
                        anchor_height=anchor_rect.height(),
                    )
                else:
                    MainWindow._paint_translated_preview_box(
                        painter,
                        bubble_rect,
                        text,
                        anchor_height=anchor_rect.height(),
                    )
        finally:
            painter.end()

        buffer = np.frombuffer(image.bits(), dtype=np.uint8)
        rgba = buffer.reshape((image.height(), image.bytesPerLine()))[:, : image.width() * 4]
        rgba = rgba.reshape((image.height(), image.width(), 4)).copy()
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    @staticmethod
    def _paint_translated_preview_box(
        painter: QPainter,
        rect: QRect,
        text: str,
        *,
        anchor_height: int | None = None,
    ) -> None:
        accent = QColor(48, 231, 149, 220)
        background = QColor(15, 23, 42, 212)
        text_color = QColor(248, 250, 252)
        font_anchor_height = rect.height() if anchor_height is None else anchor_height

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
        text_rect = bubble_rect.adjusted(horizontal_padding, vertical_padding, -horizontal_padding, -vertical_padding)
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            text_rect = bubble_rect

        painter.setFont(TranslationOverlay._font_for_text(text, text_rect, overlay_font_pixel_size(font_anchor_height)))
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    @staticmethod
    def _paint_clean_preview_text(
        painter: QPainter,
        rect: QRect,
        text: str,
        *,
        anchor_height: int | None = None,
    ) -> None:
        font_anchor_height = rect.height() if anchor_height is None else anchor_height
        horizontal_padding = max(min(rect.height() // 5, 10), 2)
        vertical_padding = max(min(rect.height() // 10, 5), 1)
        text_rect = rect.adjusted(horizontal_padding, vertical_padding, -horizontal_padding, -vertical_padding)
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            text_rect = rect

        painter.setFont(TranslationOverlay._font_for_text(text, text_rect, overlay_font_pixel_size(font_anchor_height)))
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap
        shadow = QColor(15, 23, 42, 230)
        text_color = QColor(248, 250, 252, 245)
        for offset_x, offset_y in ((-1, 0), (1, 0), (0, -1), (0, 1), (2, 2)):
            painter.setPen(shadow)
            painter.drawText(text_rect.translated(offset_x, offset_y), flags, text)

        painter.setPen(text_color)
        painter.drawText(text_rect, flags, text)

    @staticmethod
    def _create_image_label(placeholder: str) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(360, 240)
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
