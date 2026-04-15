from pathlib import Path

from screenlens_detection.ocr import TesseractOCRBackend


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
