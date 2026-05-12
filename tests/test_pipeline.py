from dataclasses import replace

import cv2
import numpy as np

from screenlens_detection.models import DetectionBox, PipelineSettings
from screenlens_detection.ocr import NoOpOCRBackend, OCRBackend, OCRFrameResult, OCRResult
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
    assert analysis.runtime_timings_ms == {}


def test_pipeline_runtime_debug_timings_are_opt_in() -> None:
    canvas = np.full((120, 320, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Runtime Debug",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    disabled_pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.0, min_contour_area=80, ocr_enabled=False),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )
    disabled = disabled_pipeline.process(canvas, monitor_label="debug-off")

    enabled_pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.0, min_contour_area=80, ocr_enabled=False, runtime_debug_enabled=True),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )
    enabled = enabled_pipeline.process(canvas, monitor_label="debug-on")

    assert disabled.runtime_timings_ms == {}
    assert enabled.runtime_timings_ms["total"] > 0.0
    assert "opencv_detection" in enabled.runtime_timings_ms
    assert "ocr_annotation" in enabled.runtime_timings_ms


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


def test_pipeline_detects_overlay_text_on_complex_video_frame() -> None:
    rng = np.random.default_rng(5)
    canvas = np.zeros((360, 960, 3), dtype=np.uint8)
    canvas[:] = (8, 16, 25)
    for _index in range(160):
        start = (int(rng.integers(0, 960)), int(rng.integers(0, 360)))
        end = (int(rng.integers(0, 960)), int(rng.integers(0, 360)))
        color = tuple(int(value) for value in rng.integers(5, 80, size=3))
        cv2.line(canvas, start, end, color, int(rng.integers(1, 3)), cv2.LINE_AA)

    cv2.putText(
        canvas,
        "MISSION OBJECTIVE UPDATED",
        (120, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "MISSION OBJECTIVE UPDATED",
        (120, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    pipeline = TextDetectionPipeline(
        PipelineSettings(upscale_factor=1.0, min_contour_area=100, max_boxes=50, ocr_enabled=False),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )
    analysis = pipeline.process(canvas, monitor_label="synthetic-video")

    target = (110, 230, 430, 40)
    assert any(_target_coverage((box.x, box.y, box.w, box.h), target) >= 0.65 for box in analysis.boxes)


def _target_coverage(box: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> float:
    intersection = TextDetectionPipeline._intersection_area(box, target)
    return intersection / max(target[2] * target[3], 1)


def test_pipeline_scanline_mode_reuses_boxes_outside_active_band() -> None:
    canvas = np.full((240, 640, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Realtime OCR Demo",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Realtime OCR Demo",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    bottom_only = np.full_like(canvas, 255)
    cv2.putText(
        bottom_only,
        "Realtime OCR Demo",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            detection_scale=0.66,
            min_contour_area=80,
            max_boxes=20,
            ocr_enabled=False,
            scanline_roi_enabled=True,
            scanline_roi_band_count=2,
        ),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    top_target = (24, 49, 305, 55)
    bottom_target = (24, 168, 305, 56)

    first = pipeline.process(canvas, monitor_label="scanline-test")
    second = pipeline.process(canvas, monitor_label="scanline-test")
    third = pipeline.process(bottom_only, monitor_label="scanline-test")

    assert any(_target_coverage((box.x, box.y, box.w, box.h), top_target) >= 0.55 for box in first.boxes)
    assert any(_target_coverage((box.x, box.y, box.w, box.h), top_target) >= 0.55 for box in second.boxes)
    assert any(_target_coverage((box.x, box.y, box.w, box.h), bottom_target) >= 0.55 for box in second.boxes)
    assert not any(_target_coverage((box.x, box.y, box.w, box.h), top_target) >= 0.55 for box in third.boxes)
    assert any(_target_coverage((box.x, box.y, box.w, box.h), bottom_target) >= 0.55 for box in third.boxes)
    assert "scanline 1/2" in third.status


def test_pipeline_hover_region_selects_nearest_box_in_line_mode() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_region_mode="hover"),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    selected = pipeline._select_hover_source_boxes(
        [
            (40, 80, 180, 32),
            (420, 80, 180, 32),
            (40, 180, 180, 32),
        ],
        (60, 92),
    )

    assert selected == [(40, 80, 180, 32)]


def test_pipeline_hover_region_selects_strict_block_neighbors_for_wide_text() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_region_mode="hover", translation_block_mode="strict"),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    selected = pipeline._select_hover_source_boxes(
        [
            (100, 100, 620, 32),
            (102, 140, 630, 32),
            (101, 180, 590, 32),
            (780, 100, 520, 32),
            (30, 260, 150, 28),
        ],
        (160, 112),
    )

    assert selected == [
        (100, 100, 620, 32),
        (102, 140, 630, 32),
        (101, 180, 590, 32),
    ]


def test_pipeline_hover_region_uses_recent_ocr_cache_before_roi_detection() -> None:
    backend = BatchRecordingOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            translation_region_mode="hover",
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=4,
        ),
        backend,
        NoOpTranslationBackend(),
    )

    frame = np.full((300, 500, 3), 255, dtype=np.uint8)
    gray = np.full((300, 500), 255, dtype=np.uint8)
    pipeline._annotate_with_ocr([(100, 100, 300, 32)], gray, frame.shape, 1.0)

    detection_gray = pipeline._enhance_grayscale(frame)
    preview_boxes, line_mask, working_boxes = pipeline._hover_detection_pass(
        frame,
        detection_gray,
        frame.shape,
        1.0,
        (120, 112),
    )

    assert working_boxes == [(100, 100, 300, 32)]
    assert preview_boxes == [(100, 100, 300, 32)]
    assert line_mask[110, 120] == 255
    assert pipeline._last_hover_region_status == "hover cache"
    assert backend.batch_calls == [(1, "eng", [7])]


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


class FullFrameRecordingOCRBackend(OCRBackend):
    def __init__(self) -> None:
        self.frame_calls: list[tuple[tuple[int, ...], str]] = []
        self.batch_calls = 0

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return "Full frame test OCR"

    def supports_full_frame(self) -> bool:
        return True

    def recognize_frame(self, frame: np.ndarray, *, language: str) -> list[OCRFrameResult]:
        self.frame_calls.append((frame.shape, language))
        return [OCRFrameResult(rect=(20, 30, 120, 28), text="Hello World", confidence=95.0)]

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        self.batch_calls += 1
        raise AssertionError("full-frame OCR should bypass crop recognize_batch")


def test_pipeline_uses_full_frame_ocr_backend_directly() -> None:
    backend = FullFrameRecordingOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            min_contour_area=80,
        ),
        backend,
        NoOpTranslationBackend(),
    )
    frame = np.full((160, 260, 3), 255, dtype=np.uint8)

    analysis = pipeline.process(frame, monitor_label="full-frame")

    assert backend.frame_calls == [((160, 260, 3), "eng")]
    assert backend.batch_calls == 0
    assert [box.text for box in analysis.boxes] == ["Hello World"]
    assert [(box.x, box.y, box.w, box.h) for box in analysis.boxes] == [(20, 30, 120, 28)]
    assert "Native full-frame OCR detector" in analysis.status
    assert "Full frame test OCR | full-frame OCR | read 1" in analysis.status


class CountingBatchOCRBackend(BatchRecordingOCRBackend):
    def __init__(self) -> None:
        super().__init__()
        self._next_text_index = 1

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        self.batch_calls.append((len(images), language, list(psms)))
        results = []
        for _image in images:
            results.append(OCRResult(text=f"demo {self._next_text_index}", confidence=95.0))
            self._next_text_index += 1
        return results


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


def test_pipeline_skips_likely_browser_toolbar_before_ocr() -> None:
    backend = BatchRecordingOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=8,
        ),
        backend,
        NoOpTranslationBackend(),
    )

    working_boxes = [
        (540, 36, 1320, 80),
        (260, 620, 620, 46),
    ]
    gray = np.full((1080, 1920), 255, dtype=np.uint8)

    detected = pipeline._annotate_with_ocr(working_boxes, gray, (1080, 1920, 3), 1.0)

    assert backend.batch_calls == [(1, "eng", [7])]
    assert [(box.x, box.y, box.w, box.h) for box in detected] == [(260, 620, 620, 46)]


def test_pipeline_reuses_ocr_results_for_unchanged_crops() -> None:
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
    ]
    gray = np.full((140, 260), 255, dtype=np.uint8)

    first = pipeline._annotate_with_ocr(working_boxes, gray, (140, 260, 3), 1.0)
    second = pipeline._annotate_with_ocr(working_boxes, gray, (140, 260, 3), 1.0)

    assert backend.batch_calls == [(2, "eng", [7, 7])]
    assert [box.text for box in first] == ["demo 1", "demo 2"]
    assert [box.text for box in second] == ["demo 1", "demo 2"]


def test_pipeline_reused_ocr_cache_does_not_consume_per_frame_budget() -> None:
    backend = CountingBatchOCRBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=1,
        ),
        backend,
        NoOpTranslationBackend(),
    )

    gray = np.full((140, 260), 255, dtype=np.uint8)

    first = pipeline._annotate_with_ocr([(10, 10, 120, 28)], gray, (140, 260, 3), 1.0)
    second = pipeline._annotate_with_ocr(
        [(10, 10, 120, 28), (20, 70, 160, 30)],
        gray,
        (140, 260, 3),
        1.0,
    )

    assert [box.text for box in first] == ["demo 1"]
    assert [box.text for box in second] == ["demo 1", "demo 2"]
    assert backend.batch_calls == [(1, "eng", [7]), (1, "eng", [7])]
    assert pipeline._last_ocr_candidate_count == 2
    assert pipeline._last_ocr_reuse_count == 1
    assert pipeline._last_ocr_submitted_count == 1
    assert "1 new OCR/frame | submitted 1, reused 1/2" in pipeline._status_message()


def test_pipeline_reused_ocr_cache_keeps_translated_text() -> None:
    ocr_backend = CountingBatchOCRBackend()
    translation_backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(
            upscale_factor=1.0,
            ocr_enabled=True,
            ocr_language="eng",
            max_ocr_boxes_per_frame=1,
        ),
        ocr_backend,
        translation_backend,
    )

    gray = np.full((80, 220), 255, dtype=np.uint8)
    first = pipeline._annotate_with_ocr([(10, 10, 120, 28)], gray, (80, 220, 3), 1.0)
    first_translated = pipeline._apply_translations(first)
    pipeline._remember_ocr_translations(first_translated)

    second = pipeline._annotate_with_ocr([(10, 10, 120, 28)], gray, (80, 220, 3), 1.0)
    second_translated = pipeline._apply_translations(second)

    assert first_translated[0].translated_text == "translated:demo 1"
    assert second[0].translated_text == first_translated[0].translated_text
    assert second_translated[0].translated_text == first_translated[0].translated_text
    assert ocr_backend.batch_calls == [(1, "eng", [7])]
    assert translation_backend.calls == 1


def test_pipeline_invalidates_cached_ocr_when_crop_changes() -> None:
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

    working_boxes = [(10, 10, 120, 28)]
    gray = np.full((80, 180), 255, dtype=np.uint8)
    changed_gray = gray.copy()
    changed_gray[10:38, 10:130] = 0

    pipeline._annotate_with_ocr(working_boxes, gray, (80, 180, 3), 1.0)
    pipeline._annotate_with_ocr(working_boxes, gray, (80, 180, 3), 1.0)
    pipeline._annotate_with_ocr(working_boxes, changed_gray, (80, 180, 3), 1.0)

    assert backend.batch_calls == [(1, "eng", [7]), (1, "eng", [7])]


def test_pipeline_reuses_cached_ocr_for_motion_adjusted_box() -> None:
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

    gray = np.full((80, 220), 255, dtype=np.uint8)

    first = pipeline._annotate_with_ocr([(10, 10, 120, 28)], gray, (80, 220, 3), 1.0)
    pipeline._current_scaled_motion_offset = (20.0, 0.0, 0.24)
    second = pipeline._annotate_with_ocr([(30, 10, 120, 28)], gray, (80, 220, 3), 1.0)

    assert [box.text for box in first] == ["demo 1"]
    assert [box.text for box in second] == ["demo 1"]
    assert backend.batch_calls == [(1, "eng", [7])]


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
            motion_filter_enabled=True,
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
            motion_filter_enabled=True,
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


def test_pipeline_rejects_toolbar_icon_garbage_ocr() -> None:
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        NoOpTranslationBackend(),
    )

    assert pipeline._is_usable_text("8 O & 0 @ 60 Chat + auit", 64.0) is False


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


class RecordingRouteTranslationBackend(TranslationBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        self.calls.append((source_language_code, target_language_code, list(texts)))
        return [f"{target_language_code}:{text}" for text in texts]


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


def test_pipeline_deduplicates_repeated_text_before_translation() -> None:
    backend = RecordingRouteTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        backend,
    )

    boxes = [
        DetectionBox(
            x=10,
            y=20,
            w=500,
            h=42,
            text="Breaking news from BBC",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=10,
            y=90,
            w=500,
            h=42,
            text="Breaking news from BBC",
            source_language_code="eng",
            source_language_label="English",
        ),
    ]

    translated = pipeline._apply_translations(boxes)

    assert backend.calls == [("eng", "tha", ["Breaking news from BBC"])]
    assert translated[0].translated_text == "tha:Breaking news from BBC"
    assert translated[1].translated_text == "tha:Breaking news from BBC"


def test_pipeline_strict_block_translation_groups_paragraph_lines() -> None:
    backend = RecordingRouteTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_block_mode="strict"),
        NoOpOCRBackend(),
        backend,
    )

    boxes = [
        DetectionBox(
            x=100,
            y=100,
            w=720,
            h=32,
            text="Plastic pollution affects daily life and the environment.",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=102,
            y=140,
            w=735,
            h=32,
            text="People can reduce waste by changing routine behavior.",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=101,
            y=180,
            w=680,
            h=32,
            text="Using reusable bags and bottles helps lower plastic use.",
            source_language_code="eng",
            source_language_label="English",
        ),
    ]

    translated = pipeline._apply_translations(boxes)

    expected_text = "\n".join(box.text for box in boxes)
    assert backend.calls == [("eng", "tha", [expected_text])]
    assert len(translated) == 1
    assert translated[0].text == expected_text
    assert translated[0].translated_text == f"tha:{expected_text}"
    assert (translated[0].x, translated[0].y, translated[0].w, translated[0].h) == (100, 100, 737, 112)


def test_pipeline_strict_block_translation_reuses_recent_block_text() -> None:
    backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_block_mode="strict"),
        NoOpOCRBackend(),
        backend,
    )

    first_frame = [
        DetectionBox(
            x=100,
            y=100,
            w=720,
            h=32,
            text="Plastic pollution affects daily life and the environment.",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=102,
            y=140,
            w=735,
            h=32,
            text="People can reduce waste by changing routine behavior.",
            source_language_code="eng",
            source_language_label="English",
        ),
    ]
    first_result = pipeline._apply_translations(first_frame)

    second_frame = [replace(box, x=box.x + 2, y=box.y + 2) for box in first_frame]
    second_result = pipeline._apply_translations(second_frame)

    assert first_result[0].translated_text.startswith("translated:")
    assert second_result[0].translated_text == first_result[0].translated_text
    assert backend.calls == 1


def test_pipeline_strict_block_translation_leaves_menu_labels_as_lines() -> None:
    backend = RecordingRouteTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_block_mode="strict"),
        NoOpOCRBackend(),
        backend,
    )

    boxes = [
        DetectionBox(
            x=30,
            y=40,
            w=180,
            h=28,
            text="Settings",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=30,
            y=78,
            w=180,
            h=28,
            text="Save Game",
            source_language_code="eng",
            source_language_label="English",
        ),
        DetectionBox(
            x=30,
            y=116,
            w=180,
            h=28,
            text="Exit",
            source_language_code="eng",
            source_language_label="English",
        ),
    ]

    translated = pipeline._apply_translations(boxes)

    assert len(translated) == 3
    assert all("\n" not in box.text for box in translated)
    assert len(backend.calls) == 1
    assert len(backend.calls[0][2]) == 3


def test_pipeline_strict_block_translation_does_not_cross_columns() -> None:
    backend = RecordingRouteTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_block_mode="strict"),
        NoOpOCRBackend(),
        backend,
    )

    left_first = "Plastic pollution affects daily life and the environment."
    left_second = "People can reduce waste by changing routine behavior."
    right_first = "Economic reports describe a slower recovery this quarter."
    right_second = "Analysts expect demand to improve during the summer."
    boxes = [
        DetectionBox(x=100, y=100, w=520, h=32, text=left_first, source_language_code="eng"),
        DetectionBox(x=700, y=100, w=520, h=32, text=right_first, source_language_code="eng"),
        DetectionBox(x=100, y=140, w=520, h=32, text=left_second, source_language_code="eng"),
        DetectionBox(x=700, y=140, w=520, h=32, text=right_second, source_language_code="eng"),
    ]

    translated = pipeline._apply_translations(boxes)

    assert len(translated) == 2
    assert translated[0].text == f"{left_first}\n{left_second}"
    assert translated[1].text == f"{right_first}\n{right_second}"
    assert len(backend.calls) == 1
    assert backend.calls[0][2] == [translated[0].text, translated[1].text]


def test_pipeline_recent_translation_lookup_keeps_language_route_separate() -> None:
    backend = RecordingRouteTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(),
        NoOpOCRBackend(),
        backend,
    )

    thai_target = DetectionBox(
        x=10,
        y=20,
        w=500,
        h=42,
        text="Breaking news from BBC",
        source_language_code="eng",
        source_language_label="English",
        target_language_code="tha",
        target_language_label="Thai",
    )
    english_target = replace(
        thai_target,
        target_language_code="eng",
        target_language_label="English",
    )

    first = pipeline._apply_translations([thai_target])
    second = pipeline._apply_translations([english_target])

    assert first[0].translated_text == "tha:Breaking news from BBC"
    assert second[0].translated_text == "eng:Breaking news from BBC"
    assert backend.calls == [
        ("eng", "tha", ["Breaking news from BBC"]),
        ("eng", "eng", ["Breaking news from BBC"]),
    ]


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


def test_pipeline_reuses_recent_translation_for_stable_ocr_noise() -> None:
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

    noisy_frame = [
        DetectionBox(
            x=101,
            y=201,
            w=902,
            h=49,
            text="US Treasury Secretary Scott Bessent has told the B8C",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    noisy_result = pipeline._apply_translations(noisy_frame)

    assert noisy_result[0].translated_text == first_result[0].translated_text
    assert backend.calls == 1


def test_pipeline_does_not_reuse_similarity_when_numbers_change() -> None:
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
            w=500,
            h=48,
            text="HP 90 remaining after attack",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    pipeline._apply_translations(first_frame)

    changed_frame = [
        DetectionBox(
            x=101,
            y=201,
            w=500,
            h=48,
            text="HP 10 remaining after attack",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    changed_result = pipeline._apply_translations(changed_frame)

    assert changed_result[0].translated_text == ""
    assert backend.calls == 2


def test_pipeline_does_not_reuse_similarity_for_short_menu_labels() -> None:
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
            w=180,
            h=40,
            text="Continue",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    pipeline._apply_translations(first_frame)

    changed_frame = [
        DetectionBox(
            x=101,
            y=201,
            w=180,
            h=40,
            text="Continua",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    changed_result = pipeline._apply_translations(changed_frame)

    assert changed_result[0].translated_text == ""
    assert backend.calls == 2


def test_pipeline_can_disable_text_similarity_stability() -> None:
    backend = OneShotTranslationBackend()
    pipeline = TextDetectionPipeline(
        PipelineSettings(translation_similarity_stability_enabled=False),
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

    noisy_frame = [
        DetectionBox(
            x=101,
            y=201,
            w=902,
            h=49,
            text="US Treasury Secretary Scott Bessent has told the B8C",
            source_language_code="eng",
            source_language_label="English",
        )
    ]
    noisy_result = pipeline._apply_translations(noisy_frame)

    assert noisy_result[0].translated_text == ""
    assert backend.calls == 2


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


def test_pipeline_reuses_recent_translation_across_short_blank_without_overlay_tracking() -> None:
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

    assert moved_result[0].translated_text.startswith("translated:")
    assert backend.calls == 1


def test_pipeline_clears_recent_translation_after_extended_blank_without_overlay_tracking() -> None:
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
    pipeline._apply_translations([])
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


def test_pipeline_reuses_recent_translation_for_textless_moving_box() -> None:
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
    first_result = pipeline._apply_translations(first_frame)
    pipeline._current_motion_offset = (0.0, 36.0, 0.24)

    moving_frame = [
        DetectionBox(
            x=100,
            y=236,
            w=900,
            h=48,
            text="",
            target_language_code="tha",
            target_language_label="Thai",
        )
    ]
    moving_result = pipeline._apply_translations(moving_frame)

    assert moving_result[0].text == first_frame[0].text
    assert moving_result[0].translated_text == first_result[0].translated_text
    assert backend.calls == 1
