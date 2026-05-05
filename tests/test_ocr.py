import threading
from pathlib import Path
from time import sleep

import numpy as np

from screenlens_detection.ocr import EasyOCRBackend, OCRBackend, OCRResult, QueuedOCRBackend, TesseractOCRBackend


def test_resolve_binary_prefers_runtime_bundle(monkeypatch, tmp_path: Path) -> None:
    bundled_binary = tmp_path / "vendor" / "tesseract" / "tesseract.exe"
    bundled_binary.parent.mkdir(parents=True)
    bundled_binary.write_text("", encoding="utf-8")

    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr("screenlens_detection.ocr.application_roots", lambda: [tmp_path])
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert TesseractOCRBackend._resolve_binary() == str(bundled_binary)


def test_resolve_tessdata_dir_uses_binary_sibling_directory(tmp_path: Path, monkeypatch) -> None:
    binary_dir = tmp_path / "Tesseract-OCR"
    tessdata_dir = binary_dir / "tessdata"
    tessdata_dir.mkdir(parents=True)
    (tessdata_dir / "eng.traineddata").write_text("", encoding="utf-8")
    binary = binary_dir / "tesseract.exe"
    binary.write_text("", encoding="utf-8")

    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr("screenlens_detection.ocr.application_roots", lambda: [])

    assert TesseractOCRBackend._resolve_tessdata_dir(str(binary)) == str(tessdata_dir)


def test_resolve_tessdata_dir_uses_runtime_bundle_directory(tmp_path: Path, monkeypatch) -> None:
    tessdata_dir = tmp_path / "tesseract" / "tessdata"
    tessdata_dir.mkdir(parents=True)
    (tessdata_dir / "tha.traineddata").write_text("", encoding="utf-8")

    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr("screenlens_detection.ocr.application_roots", lambda: [tmp_path])

    assert TesseractOCRBackend._resolve_tessdata_dir(None) == str(tessdata_dir)


def test_easyocr_backend_maps_languages_and_aggregates_results(monkeypatch) -> None:
    class FakeReader:
        init_calls: list[tuple[tuple[str, ...], bool, bool]] = []
        read_calls: list[tuple[tuple[str, ...], tuple[int, ...], int, bool]] = []

        def __init__(self, languages: list[str], *, gpu: bool, verbose: bool) -> None:
            self.languages = tuple(languages)
            self.init_calls.append((self.languages, gpu, verbose))

        def readtext(self, image: np.ndarray, *, detail: int, paragraph: bool):
            self.read_calls.append((self.languages, image.shape, detail, paragraph))
            return [
                ([[40, 5], [80, 5], [80, 22], [40, 22]], "security", 0.82),
                ([[5, 5], [36, 5], [36, 22], [5, 22]], "BBC", 0.98),
            ]

    monkeypatch.setattr("screenlens_detection.ocr.EasyOCRReader", FakeReader)
    monkeypatch.setattr("screenlens_detection.ocr._nvidia_cuda_available", lambda: False)

    backend = EasyOCRBackend()
    prepared = backend.prepare_image(np.full((18, 90), 255, dtype=np.uint8))
    result = backend.recognize(prepared, language="tha+eng", psm=7)

    assert FakeReader.init_calls == [(("th", "en"), False, False)]
    assert FakeReader.read_calls[0][0] == ("th", "en")
    assert FakeReader.read_calls[0][2:] == (1, False)
    assert prepared.shape[0] > 18
    assert result.text == "BBC security"
    assert result.confidence == 90.0


def test_easyocr_backend_batches_pre_detected_crops(monkeypatch) -> None:
    class FakeReader:
        recognize_calls: list[tuple[tuple[int, ...], int, bool, list[list[int]]]] = []

        def __init__(self, languages: list[str], *, gpu: bool, verbose: bool) -> None:
            self.languages = tuple(languages)

        def recognize(
            self,
            image: np.ndarray,
            *,
            horizontal_list: list[list[int]],
            free_list: list[object],
            detail: int,
            paragraph: bool,
            batch_size: int,
            reformat: bool,
        ):
            self.recognize_calls.append((image.shape, batch_size, reformat, horizontal_list))
            return [
                ([[0, 0], [30, 0], [30, 10], [0, 10]], "first", 0.90),
                ([[0, 26], [20, 26], [20, 38], [0, 38]], "second", 0.80),
            ]

    monkeypatch.setattr("screenlens_detection.ocr.EasyOCRReader", FakeReader)
    monkeypatch.setattr("screenlens_detection.ocr._nvidia_cuda_available", lambda: True)

    backend = EasyOCRBackend(device_preference="gpu")
    results = backend.recognize_batch(
        [
            np.full((10, 30), 255, dtype=np.uint8),
            np.full((12, 20), 255, dtype=np.uint8),
        ],
        language="eng",
        psms=[7, 7],
    )

    assert [result.text for result in results] == ["first", "second"]
    assert [result.confidence for result in results] == [90.0, 80.0]
    assert FakeReader.recognize_calls == [
        ((38, 30), 2, False, [[0, 30, 0, 10], [0, 20, 26, 38]])
    ]


class RecordingOCRBackend(OCRBackend):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, list[int]]] = []
        self.closed = False

    def is_available(self) -> bool:
        return True

    def recognize_batch(self, images: list[object], *, language: str, psms: list[int]) -> list[OCRResult]:
        self.calls.append((len(images), language, list(psms)))
        return [
            OCRResult(text=f"text {index}", confidence=95.0)
            for index, _image in enumerate(images, start=1)
        ]

    def close(self) -> None:
        self.closed = True


def test_queued_ocr_backend_returns_cached_results_after_background_batch() -> None:
    backend = RecordingOCRBackend()
    queued = QueuedOCRBackend(backend, max_batch_size=8, synchronous_batch_size=0)

    try:
        image = np.full((16, 64), 255, dtype=np.uint8)
        first = queued.recognize_batch([image], language="eng", psms=[7])

        resolved = []
        for _attempt in range(20):
            resolved = queued.recognize_batch([image], language="eng", psms=[7])
            if resolved[0].text:
                break
            sleep(0.05)

        assert first == [OCRResult()]
        assert resolved == [OCRResult(text="text 1", confidence=95.0)]
        assert backend.calls == [(1, "eng", [7])]
    finally:
        queued.close()

    assert backend.closed is True


def test_queued_ocr_backend_processes_small_batches_synchronously() -> None:
    backend = RecordingOCRBackend()
    queued = QueuedOCRBackend(backend, max_batch_size=8, synchronous_batch_size=2)

    try:
        images = [
            np.full((16, 64), 255, dtype=np.uint8),
            np.full((18, 72), 255, dtype=np.uint8),
        ]

        result = queued.recognize_batch(images, language="eng", psms=[7, 7])

        assert result == [
            OCRResult(text="text 1", confidence=95.0),
            OCRResult(text="text 2", confidence=95.0),
        ]
        assert backend.calls == [(2, "eng", [7, 7])]
    finally:
        queued.close()


def test_queued_ocr_backend_processes_priority_subset_synchronously() -> None:
    backend = RecordingOCRBackend()
    queued = QueuedOCRBackend(backend, max_batch_size=8, synchronous_batch_size=2)

    try:
        images = [
            np.full((12, 24), 255, dtype=np.uint8),
            np.full((32, 240), 255, dtype=np.uint8),
            np.full((20, 160), 255, dtype=np.uint8),
        ]

        result = queued.recognize_batch(images, language="eng", psms=[7, 7, 7])

        assert result[0] == OCRResult()
        assert result[1] == OCRResult(text="text 1", confidence=95.0)
        assert result[2] == OCRResult(text="text 2", confidence=95.0)
        assert backend.calls[0] == (2, "eng", [7, 7])
    finally:
        queued.close()


def test_queued_ocr_backend_reports_configured_worker_count() -> None:
    backend = RecordingOCRBackend()
    queued = QueuedOCRBackend(backend, max_batch_size=8, synchronous_batch_size=0, worker_count=2)

    try:
        assert "async OCR queue, 2 workers" in queued.runtime_diagnostics()
    finally:
        queued.close()


def test_tesseract_backend_recognizes_batch_in_parallel() -> None:
    class ParallelTesseractBackend(TesseractOCRBackend):
        def __init__(self) -> None:
            self._binary = "tesseract.exe"
            self._tessdata_dir = None
            self._available_languages = {"eng"}
            self._max_workers = 2
            self._executor = None
            self._executor_worker_count = 0
            self._executor_lock = threading.Lock()
            self.barrier = threading.Barrier(2)
            self.thread_ids: set[int] = set()

        def is_available(self) -> bool:
            return True

        def _build_candidates(self, image: np.ndarray) -> list[np.ndarray]:
            return [image]

        def _recognize_candidate(self, image: np.ndarray, *, language: str, config: str) -> OCRResult:
            self.thread_ids.add(threading.get_ident())
            self.barrier.wait(timeout=1.0)
            return OCRResult(text=str(int(image[0, 0])), confidence=90.0)

    backend = ParallelTesseractBackend()
    try:
        results = backend.recognize_batch(
            [
                np.full((8, 8), 1, dtype=np.uint8),
                np.full((8, 8), 2, dtype=np.uint8),
            ],
            language="eng",
            psms=[7, 7],
        )

        assert [result.text for result in results] == ["1", "2"]
        assert len(backend.thread_ids) == 2
    finally:
        backend.close()
