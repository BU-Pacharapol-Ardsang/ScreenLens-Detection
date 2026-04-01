from __future__ import annotations

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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import cv2
import numpy as np

from ..capture import ScreenCapturer
from ..models import FrameAnalysis, MonitorSpec, PipelineSettings
from ..worker import ProcessingWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ScreenLens-Detection")
        self.resize(1400, 900)

        self.monitors: list[MonitorSpec] = []
        self.worker: ProcessingWorker | None = None

        self.monitor_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Monitors")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(80, 3000)
        self.interval_spin.setValue(250)
        self.interval_spin.setSuffix(" ms")

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 3.0)
        self.scale_spin.setSingleStep(0.25)
        self.scale_spin.setValue(1.5)

        self.area_spin = QSpinBox()
        self.area_spin.setRange(50, 10000)
        self.area_spin.setValue(250)

        self.language_edit = QLineEdit("eng")
        self.ocr_checkbox = QCheckBox("Enable OCR")
        self.ocr_checkbox.setChecked(True)

        self.fps_label = QLabel("0.0")
        self.detected_label = QLabel("0")
        self.monitor_label = QLabel("-")
        self.status_label = QLabel("Idle")

        self.preview_label = self._create_image_label("Annotated preview")
        self.mask_label = self._create_image_label("Segmentation preview")
        self.text_output = QPlainTextEdit()
        self.text_output.setReadOnly(True)

        self._build_ui()
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

        settings_box = QGroupBox("Pipeline Settings")
        settings_layout = QFormLayout(settings_box)
        settings_layout.addRow("Capture interval", self.interval_spin)
        settings_layout.addRow("Upscale factor", self.scale_spin)
        settings_layout.addRow("Min contour area", self.area_spin)
        settings_layout.addRow("OCR language", self.language_edit)
        settings_layout.addRow("", self.ocr_checkbox)

        stats_box = QGroupBox("Runtime Stats")
        stats_layout = QFormLayout(stats_box)
        stats_layout.addRow("FPS", self.fps_label)
        stats_layout.addRow("Detected boxes", self.detected_label)
        stats_layout.addRow("Monitor", self.monitor_label)
        stats_layout.addRow("Status", self.status_label)

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

    def _refresh_monitors(self) -> None:
        with ScreenCapturer() as capturer:
            self.monitors = capturer.list_monitors()

        self.monitor_combo.clear()
        for monitor in self.monitors:
            self.monitor_combo.addItem(monitor.label, userData=monitor)

        if not self.monitors:
            self.status_label.setText("No monitors detected")
            self.start_button.setEnabled(False)
            return

        self.status_label.setText("Ready")
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
            ocr_enabled=self.ocr_checkbox.isChecked(),
            ocr_language=self.language_edit.text().strip() or "eng",
        )

        self.worker = ProcessingWorker(monitor=monitor, settings=settings)
        self.worker.frame_ready.connect(self._handle_frame)
        self.worker.worker_error.connect(self._handle_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.status_label.setText("Running")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()

    def _on_worker_finished(self) -> None:
        self.worker = None
        self.start_button.setEnabled(bool(self.monitors))
        self.stop_button.setEnabled(False)
        if self.status_label.text() == "Running":
            self.status_label.setText("Stopped")

    def _handle_error(self, message: str) -> None:
        self.status_label.setText("Error")
        QMessageBox.critical(self, "ScreenLens-Detection", message)
        self._stop_worker()

    def _handle_frame(self, analysis: FrameAnalysis) -> None:
        self.preview_label.setPixmap(self._frame_to_pixmap(analysis.annotated_frame, self.preview_label))
        self.mask_label.setPixmap(self._frame_to_pixmap(analysis.processed_preview, self.mask_label))

        self.fps_label.setText(f"{analysis.fps:.1f}")
        self.detected_label.setText(str(len(analysis.boxes)))
        self.monitor_label.setText(analysis.monitor_label or "-")
        self.status_label.setText(analysis.status)

        if analysis.boxes:
            lines = []
            for index, box in enumerate(analysis.boxes, start=1):
                text = box.text if box.text else "<region detected>"
                lines.append(f"[{index}] x={box.x}, y={box.y}, w={box.w}, h={box.h} :: {text}")
            self.text_output.setPlainText("\n".join(lines))
        else:
            self.text_output.setPlainText("No text regions detected in the current frame.")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_worker()
        super().closeEvent(event)

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

