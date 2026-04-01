import cv2
import numpy as np

from screenlens_detection.models import PipelineSettings
from screenlens_detection.ocr import NoOpOCRBackend
from screenlens_detection.pipeline import TextDetectionPipeline


def test_pipeline_detects_text_like_regions() -> None:
    canvas = np.full((240, 720, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "ScreenLens Detection",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Realtime OCR Demo",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.0, min_contour_area=150, max_boxes=10, ocr_enabled=False),
        NoOpOCRBackend(),
    )
    analysis = pipeline.process(canvas, monitor_label="synthetic")

    assert analysis.annotated_frame.shape == canvas.shape
    assert analysis.processed_preview.shape[:2] == canvas.shape[:2]
    assert len(analysis.boxes) >= 1
