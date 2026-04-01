# ScreenLens-Detection

`ScreenLens-Detection` is a Python + Qt desktop application for realtime screen-text detection. The current scaffold captures a monitor, runs preprocessing and text-region segmentation with OpenCV, optionally performs OCR with Tesseract, and visualizes the results in a desktop UI.

## Why this fits the project brief

This project aligns best with `Track 4: Vision + AI Integration` because it combines:

- `Screen capture / video-like frame stream`
- `Preprocessing`
- `Segmentation + contour-based text detection`
- `OCR inference`
- `Realtime desktop integration in a Qt application`

Suggested project title for presentations:

`ScreenLens-Detection: Real-time On-Screen Text Detection and OCR System`

## Pipeline

The current implementation uses this flow:

1. Capture a monitor in realtime with `mss`
2. Convert the frame to grayscale and enhance local contrast with `CLAHE`
3. Build a dual-polarity mask to detect both dark text on light backgrounds and light text on dark backgrounds
4. Segment likely text regions with morphology + contour filtering
5. Run OCR on each detected region when Tesseract is available
6. Draw detection boxes and stream the results to the Qt UI

## Features

- Realtime monitor capture
- Segmentation preview for demonstrations
- Bounding-box detection for on-screen text regions
- Optional OCR with `pytesseract`
- Adjustable capture interval, scale factor, contour area, and OCR language
- Clean Python package structure for future translation/overlay features

## Requirements

- Python `3.11+`
- Windows, Linux, or macOS with desktop screen-capture support
- Optional: Tesseract OCR installed and available on `PATH`

If `tesseract` is not available, the app still runs in detection-only mode.

### Tesseract on Windows

1. Install Tesseract OCR
2. Add the Tesseract install directory to `PATH`
3. If needed, set:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

To recognize Thai text, install Thai language data and set the app language field to `tha+eng`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

For tests:

```powershell
pip install -e ".[dev]"
pytest
```

## Run

```powershell
screenlens-detection
```

Or:

```powershell
python -m screenlens_detection
```

## Project structure

```text
src/screenlens_detection/
  capture.py
  main.py
  models.py
  ocr.py
  pipeline.py
  worker.py
  ui/main_window.py
tests/
  test_pipeline.py
```

## Next extension ideas

- Translation layer after OCR
- Click-and-drag region selection instead of full-monitor capture
- Better detector replacement with CRAFT / DBNet / EAST
- Overlay translated text directly over the source frame
- Result export for documentation and presentation demos
