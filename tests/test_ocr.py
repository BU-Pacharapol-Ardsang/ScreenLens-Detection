from pathlib import Path

import numpy as np

from screenlens_detection.ocr import EasyOCRBackend, TesseractOCRBackend


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

    backend = EasyOCRBackend()
    prepared = backend.prepare_image(np.full((18, 90), 255, dtype=np.uint8))
    result = backend.recognize(prepared, language="tha+eng", psm=7)

    assert FakeReader.init_calls == [(("th", "en"), False, False)]
    assert FakeReader.read_calls[0][0] == ("th", "en")
    assert FakeReader.read_calls[0][2:] == (1, False)
    assert prepared.shape[0] > 18
    assert result.text == "BBC security"
    assert result.confidence == 90.0
