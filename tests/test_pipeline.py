import cv2
import numpy as np

from screenlens_detection.models import PipelineSettings
from screenlens_detection.ocr import NoOpOCRBackend
from screenlens_detection.pipeline import TextDetectionPipeline
from screenlens_detection.translation import NoOpTranslationBackend


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
        NoOpTranslationBackend(),
    )
    analysis = pipeline.process(canvas, monitor_label="synthetic")

    assert analysis.annotated_frame.shape == canvas.shape
    assert analysis.processed_preview.shape[:2] == canvas.shape[:2]
    assert len(analysis.boxes) >= 1


def test_pipeline_detects_document_lines_without_merging_whole_paragraph() -> None:
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1280, 90), (42, 42, 42), -1)
    cv2.rectangle(canvas, (0, 90), (180, 720), (56, 56, 56), -1)

    cv2.putText(
        canvas,
        "TBAC Knowledge",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    article_lines = [
        "Plastic pollution affects daily life and the environment.",
        "People can reduce waste by changing routine behavior.",
        "Using reusable bags and bottles helps lower plastic use.",
        "Small actions can influence communities over time.",
        "Sustainable choices improve long term environmental health.",
        "Collective action can turn awareness into measurable change.",
    ]
    for index, line in enumerate(article_lines):
        cv2.putText(
            canvas,
            line,
            (250, 170 + index * 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.5, min_contour_area=150, max_boxes=60, ocr_enabled=False),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )
    analysis = pipeline.process(canvas, monitor_label="synthetic-document")

    article_boxes = [box for box in analysis.boxes if box.x >= 220 and box.y >= 120]

    assert len(article_boxes) >= 5
    assert max(box.h for box in article_boxes) < 60
