# ScreenLens-Detection - สรุปโครงการ

เอกสารนี้อ้างอิงโค้ดปัจจุบันบน branch `feature-optimization-efficency-4` ที่ HEAD `5f1cf9c` และสรุป pipeline, module, feature และ runtime flow ที่มีอยู่จริงใน repository ตอนนี้

## ภาพรวมโครงการ

**ScreenLens-Detection** เป็นแอปเดสก์ท็อป Python + Qt สำหรับจับภาพหน้าจอแบบเรียลไทม์ ตรวจจับข้อความบนภาพ ทำ OCR แปลภาษา และแสดงผลกลับเป็น preview หรือ overlay บนหน้าจอ ระบบถูกออกแบบให้เลือก backend ได้หลายแบบ เพื่อเทียบความเร็ว/ความแม่นยำของ OpenCV, EasyOCR, RapidOCR, PaddleOCR, Tesseract, Argos และ Google Translate

องค์ประกอบหลักในโค้ดปัจจุบัน:

- **Screen capture**: ใช้ `mss` ผ่าน `ScreenCapturer` เพื่อจับภาพ monitor ที่เลือก
- **Realtime worker**: ใช้ `ProcessingWorker(QThread)` และ capture thread แยก พร้อม `_LatestFrameQueue` ขนาด 1 ที่ทิ้ง frame เก่าเมื่อ process ไม่ทัน
- **Text detection**: ใช้ OpenCV morphology เป็น default และเลือก deep detector ได้ผ่าน RapidOCR ONNX DBNet, PaddleOCR DBNet หรือ EasyOCR CRAFT
- **OCR**: รองรับ crop OCR ผ่าน EasyOCR/Tesseract พร้อม `QueuedOCRBackend` และรองรับ RapidOCR full-frame OCR ผ่าน `recognize_frame()`
- **Translation**: รองรับ Argos Translate offline และ Google Translate online พร้อม queue/cache/reuse
- **Overlay**: แสดงคำแปลบนหน้าจอด้วย bubble mode หรือ clean patch mode พร้อม overlay tracking แบบ legacy motion หรือ visual anchor
- **Recording/debug**: บันทึก annotated/segmentation/translated preview และ `session_log.jsonl` พร้อม runtime timings ต่อ stage
- **Windows integration**: มี Windows hotkeys, overlay capture exclusion และ build flow สำหรับ PyInstaller onedir

**เวอร์ชัน**: 0.1.0  
**Python**: 3.11+  
**Runtime ที่รองรับ**:

- `python -m screenlens_detection`
- `screenlens` หรือ `screenlens-detection`
- `python screenlens.py`
- double-click `screenlens.pyw` บน Windows เมื่อมี `.venv`
- `build_screenlens_exe.bat` สำหรับสร้าง `dist\ScreenLens\ScreenLens.exe`

## Pipeline การทำงานปัจจุบัน

ลำดับหลักใน `TextDetectionPipeline.process()`:

```text
Capture selected monitor
  -> LatestFrameQueue keeps only newest frame
  -> ProcessingWorker receives frame
  -> Scale frame by effective detection_scale/upscale_factor
  -> Enhance grayscale
  -> Choose detection/OCR path
       A. hover + RapidOCR full-frame OCR
       B. RapidOCR full-frame OCR
       C. hover ROI + detector + crop OCR
       D. scanline ROI + detector + crop OCR
       E. full-frame detector + crop OCR
  -> Crop path: stabilize boxes + optional motion filter + queued OCR
  -> Full-frame path: filter/merge/validate/stabilize OCR frame results
  -> Estimate frame motion offset
  -> Translation reuse/cache + line or strict block translation
  -> Build previews, status, runtime timings and FrameAnalysis
  -> UI / overlay / recording
```

decision path สำคัญ:

- `ocr_backend.supports_full_frame()` และ `translation_region_mode == "hover"`: ใช้ `_annotate_hover_with_full_frame_ocr()`
- `ocr_backend.supports_full_frame()` และ full-screen mode: ใช้ `_annotate_with_full_frame_ocr()`
- hover mode ที่ไม่ใช่ full-frame OCR: ใช้ `_hover_detection_pass()` แล้ว crop OCR
- `scanline_roi_enabled=True`: ใช้ `_scanline_detection_pass()` เพื่อแบ่ง frame เป็น vertical bands
- default: ใช้ `_text_detection_pass()` โดยเลือก OpenCV หรือ deep detector ตาม `text_detector_mode`

ผลลัพธ์คือ `FrameAnalysis` ที่มี `boxes`, `source_frame`, preview frames, FPS, OCR runtime status, motion offset และ `runtime_timings_ms`

## โครงสร้างโครงการ

```text
ScreenLens-Detection/
├── build_screenlens_exe.bat
├── screenlens.py
├── screenlens.pyw
├── screenlens.spec
├── pyproject.toml
├── requirements.txt
├── README.md
├── PROJECT_SUMMARY.md
├── ARCHITECTURE_EVOLUTION.md
├── PIPELINE_MERMAID.md
├── PRESENTATION.md
├── scripts/
│   ├── setup_windows.ps1
│   ├── build_windows.ps1
│   ├── install_tesseract_vendor.ps1
│   └── download_argos_models.py
├── vendor/
│   ├── argos/
│   └── tesseract/
├── src/
│   └── screenlens_detection/
│       ├── app_entry.py
│       ├── capture.py
│       ├── cursor.py
│       ├── languages.py
│       ├── launcher.py
│       ├── main.py
│       ├── models.py
│       ├── motion.py
│       ├── ocr.py
│       ├── onnxruntime_utils.py
│       ├── overlay.py
│       ├── overlay_tracker.py
│       ├── overlay_tracks.py
│       ├── pipeline.py
│       ├── recording.py
│       ├── runtime.py
│       ├── subtitle_cleaner.py
│       ├── text_detectors.py
│       ├── translation.py
│       ├── windows_capture_exclusion.py
│       ├── windows_hotkeys.py
│       ├── worker.py
│       └── ui/main_window.py
└── tests/
    ├── test_pipeline.py
    ├── test_ocr.py
    ├── test_translation.py
    ├── test_text_detectors.py
    ├── test_overlay.py
    ├── test_motion.py
    ├── test_recording.py
    ├── test_subtitle_cleaner.py
    ├── test_main_window.py
    └── tests อื่นสำหรับ launcher/languages/motion/windows utilities
```

## โมดูลหลัก

### `models.py`

- `PipelineSettings` รวม setting ทั้ง pipeline เช่น detection scale, scanline ROI, hover ROI, translation block mode, subtitle render mode, OCR backend/device, full-frame validation, overlay tracking และ runtime debug
- `DetectionBox` เก็บตำแหน่งกล่อง, OCR text, translated text, language route และ confidence
- `FrameAnalysis` เป็น payload ที่ worker ส่งให้ UI/overlay/recording

### `worker.py`

- แยก UI ออกจาก processing ด้วย `QThread`
- แยก capture loop เป็น thread ชื่อ `ScreenLensCapture`
- `_LatestFrameQueue(maxsize=1)` จะ drop frame เก่าเพื่อให้ latency ต่ำ
- hover mode ใช้ dwell/tolerance ก่อนยืนยัน cursor target

### `pipeline.py`

- ทำ scale-aware preprocessing และ detection
- รองรับ OpenCV/deep detector path, scanline ROI, hover ROI, crop OCR และ full-frame OCR
- มี OCR cache จาก crop fingerprint + geometry + motion-adjusted matching
- full-frame OCR มี filter, merge line, validation mode, track stabilization และ output limit
- translation มี recent reuse, hover metadata filtering, subtitle line combining, strict block translation และ line translation

### `ocr.py`

- backend หลัก: `EasyOCRBackend`, `RapidOCRFullBackend`, `TesseractOCRBackend`
- `QueuedOCRBackend` ทำ cache และ worker queue สำหรับ crop OCR backend
- Tesseract ใช้ bundled runtime ได้จาก `vendor/tesseract` หรือ environment variables
- RapidOCR full OCR ใช้ ONNX Runtime CPU/CUDA ตาม provider ที่ติดตั้ง

### `text_detectors.py`

- OpenCV detector อยู่ใน `pipeline.py`
- optional deep detector อยู่ใน `text_detectors.py`: RapidOCR ONNX DBNet, PaddleOCR DBNet และ EasyOCR CRAFT
- RapidOCR detector มี CUDA fallback เป็น CPU ถ้า ONNX Runtime CUDA init ไม่สำเร็จ

### `translation.py`

- `ArgosTranslateBackend` สำหรับ offline translation และ bundled model `en<->th`
- `GoogleTranslateBackend` สำหรับ online translation ผ่าน `deep-translator`
- `QueuedTranslationBackend` ทำ cache, queue, sync budget และ batch ตาม route ภาษา

### `overlay.py`, `overlay_tracks.py`, `overlay_tracker.py`

- `TranslationOverlay` วาดคำแปลบนหน้าจอ
- bubble mode ขยายกล่องให้พอดีข้อความยาวและลด font เมื่อพื้นที่จำกัด
- clean patch mode ใช้ source frame เพื่อทำ patch ลบ subtitle เดิมก่อนวาดคำแปล
- `OverlayTrackManager` associate track จาก text similarity, IoU และ proximity
- `OverlayTrackingWorker` จับ frame grayscale ความละเอียดลดลงเพื่อช่วย track overlay ระหว่าง frame หลัก

### `subtitle_cleaner.py`

- สร้าง text mask จาก contrast, edge, HSV และ local delta
- ใช้ `cv2.inpaint()` เมื่อ mask เหมาะสม
- fallback เป็น soft background patch เมื่อ mask ratio ไม่เหมาะกับ inpainting

### `recording.py`

- สร้าง directory ใต้ `recordings/`
- เขียน `annotated_preview.mp4`, `segmentation_preview.mp4`, `translated_preview.mp4`
- เขียน `session_log.jsonl` พร้อม FPS, timings, OCR runtime, motion offset และ boxes ต่อ frame

### `ui/main_window.py`

- รวม control สำหรับ monitor, detector, OCR backend/device, full-frame validation, translation mode, hover/full region, strict block, subtitle render mode, overlay tracking, preview และ runtime debug
- รองรับ Windows hotkeys สำหรับ toggle overlay / hover target mode
- เชื่อม worker, overlay, overlay tracker และ recording session

## ฟีเจอร์ปัจจุบัน

- Realtime monitor capture พร้อม frame dropping
- OpenCV text detection แบบ dual-polarity และ scale-aware filtering
- Selectable deep detector: RapidOCR, PaddleOCR, EasyOCR
- Crop OCR queue/cache สำหรับ EasyOCR/Tesseract
- RapidOCR full-frame OCR พร้อม validation mode `fast`, `balanced`, `strict`
- Hover region translation และ scanline ROI
- Strict block translation สำหรับ paragraph/subtitle หลายบรรทัด
- Translation cache/reuse และ queued translation
- Bubble overlay ที่ขยายตามข้อความและไม่เล็กกว่า anchor
- Clean patch subtitle rendering แบบ experimental
- Overlay tracking แบบ legacy motion และ visual anchor lock
- Optional annotated/segmentation/translated previews
- Recording เป็น 3 video streams + JSONL session log
- Runtime debug timings ต่อ stage
- Windows hotkeys และ capture exclusion สำหรับ overlay window
- Windows setup/build ที่ตรวจ Torch และ ONNX Runtime provider หลังติดตั้ง

## Technology Stack

### Core Dependencies

| Library | Version | Purpose |
| --- | --- | --- |
| `PySide6` | `>=6.8.0` | Qt GUI framework |
| `opencv-python` | `>=4.10.0.84` | Computer vision, masks, tracking, recording |
| `numpy` | `>=2.1.0` | Numerical processing |
| `mss` | `>=10.0.0` | Screen capture |
| `Pillow` | `>=10.4.0` | Image bridge for OCR |
| `pytesseract` | `>=0.3.13` | Tesseract wrapper |
| `argostranslate` | `>=1.11.0` | Offline translation |
| `deep-translator` | `>=1.11.4` | Google translation backend |

### Optional / Build-time

- `easyocr>=1.7.2`
- `rapidocr>=3.0.0`
- `onnxruntime>=1.20.0` หรือ `onnxruntime-gpu>=1.20.0`
- `paddleocr>=3.0.0`, `paddlepaddle>=3.0.0`
- `torch==2.10.0`, `torchvision==0.25.0` ผ่าน setup script
- `pyinstaller>=6.14.0`
- `pytest>=8.3.0`

## Installation & Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

สำหรับ Windows setup ที่ติดตั้ง optional backend ครบ:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime cpu
```

สำหรับ GPU runtime:

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime gpu
```

`setup_windows.ps1` จะติดตั้ง project, EasyOCR, RapidOCR, Torch, PaddleOCR, bundled Tesseract และ Argos models จากนั้นตรวจ diagnostics ของ Torch และ ONNX Runtime ถ้าเลือก GPU แต่ `torch.cuda.is_available()` หรือ `CUDAExecutionProvider` ไม่พร้อม script จะ fail ทันที

## Build Windows EXE

```powershell
.\build_screenlens_exe.bat -TorchRuntime cpu
```

หรือ:

```powershell
.\build_screenlens_exe.bat -TorchRuntime gpu
```

build flow:

- `build_screenlens_exe.bat` forward option ไป `scripts\build_windows.ps1`
- `build_windows.ps1` สร้าง/ใช้ `.venv`, run setup และเรียก PyInstaller
- `setup_windows.ps1` ติดตั้ง `rapidocr>=3.0.0` แยกก่อน แล้ว force reinstall ONNX Runtime package ที่ตรงกับ CPU/GPU runtime
- output อยู่ที่ `dist\ScreenLens\ScreenLens.exe` แบบ onedir

## การตั้งค่าหลักใน UI

- Text Detector: OpenCV, RapidOCR, PaddleOCR, EasyOCR
- Scanline ROI: Full frame หรือ Sliding bands
- OCR Backend: Auto, EasyOCR, RapidOCR, Tesseract, Disabled
- OCR Device: Auto, CPU, GPU/CUDA
- Full-frame OCR Validation: Balanced, Fast, Strict
- Translation Mode: Argos Offline, Google Online, Disabled
- Translation Region: Full screen หรือ Hover cursor region
- Translation Block: Line mode หรือ Strict block
- Subtitle Render: Bubble overlay หรือ Clean patch
- Overlay Tracking: Legacy motion หรือ Visual anchor lock
- Runtime Debug: เปิด timings ต่อ stage
- Preview toggles: annotated, segmentation, translated

## Test Suite

รันทั้งหมด:

```powershell
pytest
```

รันเฉพาะกลุ่มสำคัญ:

```powershell
pytest tests/test_pipeline.py -v
pytest tests/test_ocr.py tests/test_translation.py tests/test_text_detectors.py -v
pytest tests/test_overlay.py tests/test_motion.py tests/test_recording.py -v
```

test ครอบคลุม pipeline path, full-frame OCR validation, hover ROI, strict block translation, OCR queue, Tesseract batch, text detector, translation cache, overlay rendering/tracking, clean patch, recording, launcher, hotkeys และ Windows capture exclusion

## สถานะโครงการ

- โค้ดปัจจุบันเป็น realtime OCR/translation desktop app ที่มี pipeline หลาย path สำหรับ benchmark ได้
- Full-frame OCR และ crop OCR อยู่ร่วมกันใน pipeline เดียว โดยตัดสินจาก backend capability
- Overlay/recording/runtime debug ทำให้ใช้ทดสอบ workflow จริงและเก็บ log เพื่อวิเคราะห์ latency ได้
- Build flow ปัจจุบันเน้น Windows clean VM และแยก CPU/GPU dependency ชัดเจนขึ้น โดยเฉพาะ ONNX Runtime provider

## ข้อจำกัดที่ควรทราบ

1. OCR quality ยังขึ้นกับภาพจริง เช่น font, contrast, motion blur, resolution และภาษา
2. Clean patch เป็น experimental และมี budget limit ผ่าน `clean_patch_max_crop_area`
3. Hover mode ต้องรอ cursor dwell ก่อน lock target ตาม `hover_dwell_ms`
4. Full-frame OCR แบบ strict ลด false positive ได้ แต่เพิ่มงาน OpenCV validation mask
5. GPU setup ต้องให้ Torch CUDA และ ONNX Runtime CUDA provider พร้อมทั้งคู่
6. Root launcher `screenlens.py` และ `screenlens.pyw` เน้น Windows `.venv\Scripts\...`

## เอกสารอ้างอิงภายใน repo

- [README.md](README.md)
- [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md)
- [PIPELINE_MERMAID.md](PIPELINE_MERMAID.md)
- `src/screenlens_detection/pipeline.py`
- `src/screenlens_detection/worker.py`
- `src/screenlens_detection/overlay.py`

---

*Last Updated: May 2026*  
*Code reference: `feature-optimization-efficency-4` / `5f1cf9c`*
