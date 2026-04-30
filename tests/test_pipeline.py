import cv2
import numpy as np

from screenlens_detection.models import DetectionBox, PipelineSettings
from screenlens_detection.ocr import NoOpOCRBackend, OCRBackend, OCRResult
from screenlens_detection.pipeline import TextDetectionPipeline
from screenlens_detection.translation import NoOpTranslationBackend, TranslationBackend


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


class BatchRecordingOCRBackend(RecordingOCRBackend):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[tuple[int, str, list[int]]] = []

    def recognize(self, image: object, *, language: str, psm: int) -> OCRResult:
        raise AssertionError("pipeline should use recognize_batch for frame OCR")

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        self.batch_calls.append((len(images), language, list(psms)))
        return [
            OCRResult(text=f"demo {index}", confidence=95.0)
            for index, _image in enumerate(images, start=1)
        ]


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


def test_pipeline_batches_ocr_crops_per_frame() -> None:
    backend = BatchRecordingOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=4,
        ),
        backend,
        NoOpTranslationBackend(),
    )

    working_boxes = [
        (10, 10, 120, 28),
        (20, 60, 160, 30),
        (30, 110, 240, 32),
    ]
    gray = np.full((180, 360), 255, dtype=np.uint8)

    detected = pipeline._annotate_with_ocr(working_boxes, gray, (180, 360, 3), 1.0)

    assert backend.batch_calls == [(3, "eng", [7, 7, 7])]
    assert [box.text for box in detected] == ["demo 1", "demo 2", "demo 3"]


def test_pipeline_keeps_detector_boxes_when_ocr_backend_is_unavailable() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.0, ocr_enabled=True),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )
    gray = np.full((80, 220), 255, dtype=np.uint8)

    detected = pipeline._annotate_with_ocr([(20, 20, 160, 28)], gray, (80, 220, 3), 1.0)

    assert len(detected) == 1
    assert detected[0].text == ""


def test_pipeline_waits_for_ocr_boxes_to_stabilize() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(stable_ocr_frames=2),
        RecordingOCRBackend(),
        NoOpTranslationBackend(),
    )

    assert pipeline._stabilize_ocr_boxes([(20, 40, 180, 32)]) == []
    assert pipeline._stabilize_ocr_boxes([(22, 41, 180, 32)]) == [(22, 41, 180, 32)]


def test_pipeline_rejects_moving_ocr_boxes_until_they_stabilize() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(stable_ocr_frames=2),
        RecordingOCRBackend(),
        NoOpTranslationBackend(),
    )

    assert pipeline._stabilize_ocr_boxes([(20, 40, 180, 32)]) == []
    assert pipeline._stabilize_ocr_boxes([(300, 40, 180, 32)]) == []
    assert pipeline._stabilize_ocr_boxes([(300, 40, 180, 32)]) == [(300, 40, 180, 32)]


def test_pipeline_filters_boxes_with_large_frame_to_frame_motion() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            motion_mean_threshold=10.0,
            motion_changed_ratio_threshold=0.10,
        ),
        RecordingOCRBackend(),
        NoOpTranslationBackend(),
    )
    box = (20, 10, 100, 20)
    pipeline._previous_motion_gray = np.zeros((80, 180), dtype=np.uint8)
    current = np.zeros((80, 180), dtype=np.uint8)
    current[10:30, 20:120] = 255

    assert pipeline._filter_motion_ocr_boxes([box], current) == []


def test_pipeline_keeps_static_boxes_when_motion_filter_is_active() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            motion_mean_threshold=10.0,
            motion_changed_ratio_threshold=0.10,
        ),
        RecordingOCRBackend(),
        NoOpTranslationBackend(),
    )
    box = (20, 10, 100, 20)
    pipeline._previous_motion_gray = np.zeros((80, 180), dtype=np.uint8)
    current = np.zeros((80, 180), dtype=np.uint8)

    assert pipeline._filter_motion_ocr_boxes([box], current) == [box]


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


class FirstOnlyTranslationBackend(TranslationBackend):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        self.calls.append(list(texts))
        return [f"translated:{texts[0]}"] + ([""] * (len(texts) - 1))


class OneShotTranslationBackend(TranslationBackend):
    def __init__(self) -> None:
        self.calls = 0

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        self.calls += 1
        if self.calls == 1:
            return [f"translated:{text}" for text in texts]
        return [""] * len(texts)


def test_pipeline_prioritizes_content_lines_for_translation_budget() -> None:
    backend = FirstOnlyTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        backend,
    )

    boxes = [
        DetectionBox(
            x=10,
            y=10,
            w=180,
            h=32,
            text="https://www.bbc.com/news/articles/c4g66p2q0750",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=20,
            y=80,
            w=1200,
            h=56,
            text="US Treasury Secretary Scott Bessent has told the BBC a small bit of economic pain is worthwhile",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=30,
            y=150,
            w=120,
            h=24,
            text="Chat",
            source_language_code="eng",
            source_language_label="English",
        ),
    ]

    translated = pipeline._apply_translations(boxes)

    assert backend.calls
    assert backend.calls[0][0].startswith("US Treasury Secretary Scott Bessent")
    assert translated[1].translated_text.startswith("translated:")
    assert translated[0].translated_text == ""


def test_pipeline_reuses_recent_translation_for_similar_box_in_next_frame() -> None:
    backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        backend,
    )

    first_frame = [
        DetectionBox(
            x=100,
            y=200,
            w=900,
            h=48,
            text="US Treasury Secretary Scott Bessent has told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    first_result = pipeline._apply_translations(first_frame)

    second_frame = [
        DetectionBox(
            x=102,
            y=202,
            w=905,
            h=50,
            text="US Treasury Secretary Scott Bessent told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    second_result = pipeline._apply_translations(second_frame)

    assert first_result[0].translated_text.startswith("translated:")
    assert second_result[0].translated_text == first_result[0].translated_text


def test_pipeline_reuses_tracked_translation_after_blank_and_scroll() -> None:
    backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(overlay_tracking_enabled=True),
        NoOpOCRBackend(),
        backend,
    )

    first_frame = [
        DetectionBox(
            x=100,
            y=200,
            w=900,
            h=48,
            text="US Treasury Secretary Scott Bessent has told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    first_result = pipeline._apply_translations(first_frame)

    assert first_result[0].translated_text.startswith("translated:")
    assert pipeline._apply_translations([]) == []

    moved_frame = [
        DetectionBox(
            x=100,
            y=520,
            w=900,
            h=48,
            text="US Treasury Secretary Scott Bessent has told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    moved_result = pipeline._apply_translations(moved_frame)

    assert moved_result[0].translated_text == first_result[0].translated_text
    assert backend.calls == 1


def test_pipeline_clears_recent_translation_without_overlay_tracking() -> None:
    backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(overlay_tracking_enabled=False),
        NoOpOCRBackend(),
        backend,
    )

    first_frame = [
        DetectionBox(
            x=100,
            y=200,
            w=900,
            h=48,
            text="US Treasury Secretary Scott Bessent has told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    pipeline._apply_translations(first_frame)
    pipeline._apply_translations([])

    moved_frame = [
        DetectionBox(
            x=100,
            y=520,
            w=900,
            h=48,
            text="US Treasury Secretary Scott Bessent has told the BBC",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    moved_result = pipeline._apply_translations(moved_frame)

    assert moved_result[0].translated_text == ""
    assert backend.calls == 2
