# ScreenLens-Detection

`ScreenLens-Detection` is a Python + Qt desktop application for realtime screen-text detection, OCR, translation, overlay rendering, and recording. It captures a monitor, detects text with OpenCV or optional deep detectors, runs crop OCR or RapidOCR full-frame OCR, translates text with Argos or Google Translate, and visualizes results in the desktop UI or an on-screen overlay.

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
2. Keep only the newest captured frame in the worker queue to avoid accumulated latency
3. Scale the frame, enhance grayscale contrast, and select one of the runtime paths: full-frame OCR, hover full-frame OCR, hover ROI crop OCR, scanline ROI crop OCR, or full-frame detector crop OCR
4. For crop OCR, stabilize boxes, optionally filter motion, and use the queued OCR backend/cache
5. For full-frame OCR, filter, merge, validate, stabilize, and limit RapidOCR frame results
6. Reuse/cache translations, optionally translate strict text blocks, and build `FrameAnalysis`
7. Stream results to previews, overlay, recording, and runtime debug UI

## Features

- Realtime monitor capture
- Segmentation preview for demonstrations
- Selectable text detector: classic OpenCV morphology, optional RapidOCR ONNX DBNet, PaddleOCR DBNet, or EasyOCR CRAFT
- Optional OCR with `EasyOCR`, `RapidOCR`, or `pytesseract`
- RapidOCR full-frame OCR validation modes: `Fast`, `Balanced`, and `Strict`
- Selectable translation backend: `Argos Translate (Offline)`, `Google Translate (Online)`, or disabled
- Hover region and scanline ROI modes for reducing detector/OCR work
- Strict block translation for paragraph/subtitle-style text
- Bubble overlay and experimental clean patch subtitle rendering
- Overlay tracking with legacy motion or visual anchor lock
- Recording to annotated/segmentation/translated MP4 streams plus JSONL session logs
- Runtime debug timings per pipeline stage
- Adjustable capture interval, scale factor, contour area, OCR backend/device, detector, translation mode, preview, and overlay settings

## Requirements

- Python `3.11+`
- Windows, Linux, or macOS with desktop screen-capture support
- Optional: an OCR backend installed locally

If no OCR backend is available, the app still runs in detection-only mode.
By default the app prefers `EasyOCR` when it is installed, then falls back to `Tesseract`.
You can override that order with:

```powershell
$env:SCREENLENS_OCR_BACKEND="easyocr"   # or: rapidocr / tesseract / off
```

The UI also exposes an `OCR backend` dropdown. `RapidOCR full OCR` runs RapidOCR detection and recognition together on the frame, then sends the recognized text to the selected translation backend.

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

### RapidOCR full OCR

RapidOCR can be selected as a native full-frame OCR backend. It performs detection and recognition in one backend instead of sending detector boxes through EasyOCR crop recognition:

```powershell
pip install -e ".[ocr_rapid]"
```

For NVIDIA ONNX Runtime acceleration, install the GPU extra instead and select `GPU (NVIDIA CUDA)` or `Auto` in the `OCR device` dropdown:

```powershell
pip install -e ".[ocr_rapid_gpu]"
```

Do not install `onnxruntime` and `onnxruntime-gpu` into the same environment at the same time. The Windows setup script handles this automatically when `-TorchRuntime gpu` is used.
The current setup flow installs `rapidocr>=3.0.0` first, then force-reinstalls the selected ONNX Runtime provider so the CPU/GPU provider does not get silently replaced by dependency resolution.

### Optional deep text detectors

The UI includes a `Text detector` dropdown:

- `Classic OpenCV (Morphology)` uses the original contour-based detector and remains the default.
- `RapidOCR ONNX DBNet (Optional)` uses RapidOCR text detection with ONNX Runtime CPU/CUDA.
- `PaddleOCR DBNet (Optional)` uses PaddleOCR detection when `paddleocr` and `paddlepaddle` are installed.
- `EasyOCR CRAFT (Optional)` reuses EasyOCR's CRAFT detector when `easyocr` is installed.

Install the optional detector packages with:

```powershell
pip install -e ".[detectors]"
```

For RapidOCR DBNet detection with ONNX Runtime CUDA, use:

```powershell
pip install -e ".[detector_rapid_gpu]"
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
- `scripts/setup_windows.ps1` installs EasyOCR, RapidOCR/ONNX Runtime, and then pins `torch`/`torchvision` from the official PyTorch CPU or CUDA wheel index so the runtime matches your chosen device. With `-TorchRuntime gpu`, it installs `onnxruntime-gpu` and verifies `CUDAExecutionProvider`.
- `screenlens.py` and `screenlens.pyw` are still useful for local development, but the built `.exe` is the correct path for blank Windows VMs.
- `build_screenlens_exe.bat` is the simplest build entrypoint. It creates `.venv` automatically when missing, installs build tools, and then produces `dist\ScreenLens\ScreenLens.exe`.

## Project structure

```text
src/screenlens_detection/
  capture.py
  cursor.py
  languages.py
  launcher.py
  main.py
  models.py
  motion.py
  ocr.py
  onnxruntime_utils.py
  overlay.py
  overlay_tracker.py
  overlay_tracks.py
  pipeline.py
  recording.py
  runtime.py
  subtitle_cleaner.py
  text_detectors.py
  translation.py
  windows_capture_exclusion.py
  windows_hotkeys.py
  worker.py
  ui/main_window.py
tests/
  test_pipeline.py
  test_ocr.py
  test_translation.py
  test_text_detectors.py
  test_overlay.py
  test_recording.py
  test_subtitle_cleaner.py
```

## Next extension ideas

- Click-and-drag region selection instead of full-monitor capture
- Preset profiles for subtitles, games, documents, and web pages
- OCR quality metrics and confidence visualization
- Export detected/translated text to subtitle, CSV, or report formats
- Benchmark dashboard for comparing OpenCV, RapidOCR, PaddleOCR, EasyOCR, and Tesseract paths
