import cv2
import numpy as np

from screenlens_detection.models import PipelineSettings
from screenlens_detection.ocr import NoOpOCRBackend, OCRBackend, OCRResult
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


class RecordingOCRBackend(OCRBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def is_available(self) -> bool:
        return True

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        self.calls.append((language, psm))
        return OCRResult(text="demo", confidence=95.0)


def test_pipeline_limits_ocr_boxes_per_frame() -> None:
    backend = RecordingOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=2,
        ),
        backend,
        NoOpTranslationBackend(),
    )

    working_boxes = [
        (0, 0, 40, 20),
        (20, 60, 120, 28),
        (40, 120, 220, 36),
        (60, 180, 320, 44),
    ]
    gray = np.full((280, 420), 255, dtype=np.uint8)

    detected = pipeline._annotate_with_ocr(working_boxes, gray, (280, 420, 3), 1.0)

    assert len(detected) == 2
    assert len(backend.calls) == 2


def test_pipeline_detects_actual_source_language_in_mixed_ocr_mode() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(source_language_code="tha+eng"),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    assert pipeline._resolve_source_language("Breaking news from BBC") == ("eng", "English")
    assert pipeline._resolve_source_language("ทดสอบภาษาไทย") == ("tha", "Thai")
    assert pipeline._resolve_source_language("BBC ภาษาไทย") == ("mixed", "Mixed (Thai + English)")


def test_pipeline_normalizes_stray_thai_noise_from_english_ocr() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    normalized = pipeline._normalize_recognized_text(
        "according to senior บ ร officials . It does not have nuclear weapons"
    )

    assert normalized == "according to senior officials. It does not have nuclear weapons"
