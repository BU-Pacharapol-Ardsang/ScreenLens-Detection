# ScreenLens-Detection

`ScreenLens-Detection` is a Python + Qt desktop application for realtime screen-text detection. The current scaffold captures a monitor, detects text regions with a selectable OpenCV or optional deep detector, optionally performs OCR with EasyOCR or Tesseract, and visualizes the results in a desktop UI.

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
4. Segment likely text regions with the selected text detector
5. Run OCR on each detected region when an OCR backend is available
6. Draw detection boxes and stream the results to the Qt UI

## Features

- Realtime monitor capture
- Segmentation preview for demonstrations
- Selectable text detector: classic OpenCV morphology, optional PaddleOCR DBNet, or optional EasyOCR CRAFT
- Optional OCR with `EasyOCR` or `pytesseract`
- Selectable translation backend: `Argos Translate (Offline)`, `Google Translate (Online)`, or disabled
- Adjustable capture interval, scale factor, contour area, and OCR language
- Clean Python package structure for future translation/overlay features

## Requirements

- Python `3.11+`
- Windows, Linux, or macOS with desktop screen-capture support
- Optional: an OCR backend installed locally

If no OCR backend is available, the app still runs in detection-only mode.
By default the app prefers `EasyOCR` when it is installed, then falls back to `Tesseract`.
You can override that order with:

```powershell
$env:SCREENLENS_OCR_BACKEND="easyocr"   # or: tesseract / off
```

The Tesseract runtime can be discovered from any of these locations:

- `TESSERACT_CMD` / `TESSDATA_PREFIX`
- A bundled `tesseract` folder beside the built app
- A bundled `Tesseract-OCR` folder beside the built app
- System `PATH`

### Tesseract on Windows

1. Install Tesseract OCR
2. Add the Tesseract install directory to `PATH`
3. If needed, set:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

To recognize Thai text, install Thai language data and set the app language field to `tha+eng`.

### EasyOCR on Windows

For better mixed Thai/English OCR quality than Tesseract, install the optional EasyOCR backend:

```powershell
pip install -e ".[ocr_easy]"
```

EasyOCR is heavier because it pulls in PyTorch, but the app will automatically prefer it once installed.
If you want NVIDIA CUDA acceleration, use the Windows setup script instead so PyTorch is installed from the official CUDA wheel index:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime gpu
```

### Optional deep text detectors

The UI includes a `Text detector` dropdown:

- `Classic OpenCV (Morphology)` uses the original contour-based detector and remains the default.
- `PaddleOCR DBNet (Optional)` uses PaddleOCR detection when `paddleocr` and `paddlepaddle` are installed.
- `EasyOCR CRAFT (Optional)` reuses EasyOCR's CRAFT detector when `easyocr` is installed.

Install the optional detector packages with:

```powershell
pip install -e ".[detectors]"
```

If a selected deep detector is not installed, the app keeps running and reports the detector as unavailable in the status line.

### Translation backends

The app UI exposes three translation modes:

- `Argos Translate (Offline)` for fully local translation after the bundled `en<->th` model files are installed
- `Google Translate (Online)` for the previous `deep-translator` HTTP flow
- `Disabled` if you want OCR only

The Windows setup script downloads the `en->th` and `th->en` `.argosmodel` files into `vendor/argos/` so they can be bundled with the app and installed locally at runtime without requiring Google or a live network connection during use.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

For Windows local development with EasyOCR and automatic CPU/GPU torch selection:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime auto
```

Force a specific runtime:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime cpu
.\scripts\setup_windows.ps1 -TorchRuntime gpu
```

This setup also downloads the offline Argos Translate model files into `vendor/argos/`.

For tests:

```powershell
pip install -e ".[dev]"
pytest
```

If you want the upgraded OCR backend and optional deep detector backends in the same environment:

```powershell
pip install -e ".[dev,ocr_easy,detectors]"
```

For Windows packaging:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime auto -IncludeBuildTools
```

## Run

Fastest option for local use:

```powershell
python screenlens.py
```

On Windows, you can also double-click `screenlens.pyw` to open the GUI without typing a command.

If the package is installed with `pip install -e .`, a shorter command is also available:

```powershell
screenlens
```

The original command still works:

```powershell
screenlens-detection
```

Or:

```powershell
python -m screenlens_detection
```

## Package For A Clean Windows VM

Recommended deployment target: `PyInstaller` `onedir`

This avoids relying on a system Python install, `.pyw` file association, or a pre-existing `.venv` on the VM.

1. Prepare the build environment:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime auto -IncludeBuildTools
```

Use `-TorchRuntime gpu` if the build machine has an NVIDIA GPU and you want the packaged app to target CUDA explicitly.

2. Optional: bundle OCR for offline VMs by placing Tesseract files in:

```text
vendor/tesseract/
  tesseract.exe
  tessdata/
    eng.traineddata
    tha.traineddata
```

The setup script already downloads bundled offline translation models into:

```text
vendor/argos/
  translate-en_th.argosmodel
  translate-th_en.argosmodel
```

3. Build the app:

```powershell
.\scripts\build_windows.ps1 -Clean
```

One-click option from File Explorer or `cmd`:

```bat
build_screenlens_exe.bat
```

4. Copy the resulting folder to the VM and run:

```text
dist/ScreenLens/ScreenLens.exe
```

Notes:

- If `vendor/tesseract/tesseract.exe` exists, the build bundles it and the app prefers that copy automatically.
- If `vendor/argos/*.argosmodel` exists, the build bundles the offline translation models and installs them automatically at runtime.
- If no bundled or installed Tesseract is found, the app still opens in detection-only mode.
- Optional deep detector packages are not required for the classic OpenCV detector.
- `scripts/setup_windows.ps1` installs EasyOCR, RapidOCR/ONNXRuntime, and then pins `torch`/`torchvision` from the official PyTorch CPU or CUDA wheel index so the runtime matches your chosen device.
- `screenlens.py` and `screenlens.pyw` are still useful for local development, but the built `.exe` is the correct path for blank Windows VMs.
- `build_screenlens_exe.bat` is the simplest build entrypoint. It creates `.venv` automatically when missing, installs build tools, and then produces `dist\ScreenLens\ScreenLens.exe`.

## Project structure

```text
src/screenlens_detection/
  capture.py
  main.py
  models.py
  ocr.py
  pipeline.py
  text_detectors.py
  worker.py
  ui/main_window.py
tests/
  test_pipeline.py
```

## Next extension ideas

- Translation layer after OCR
- Click-and-drag region selection instead of full-monitor capture
- OCR consensus across multiple stable frames
- Overlay translated text directly over the source frame
- Result export for documentation and presentation demos
