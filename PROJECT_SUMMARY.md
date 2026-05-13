# ScreenLens-Detection - สรุปโครงการ

## ภาพรวมโครงการ

**ScreenLens-Detection** เป็นแอปเดสก์ท็อปที่พัฒนาด้วย Python + Qt สำหรับจับภาพหน้าจอแบบเรียลไทม์ ตรวจจับบริเวณที่น่าจะเป็นข้อความด้วย OpenCV หรือ optional deep detector และสามารถทำ OCR กับแปลภาษาเพิ่มเติมผ่าน backend ที่เลือกได้

องค์ประกอบหลักของระบบในโค้ดปัจจุบันมีดังนี้:

- **การจับภาพหน้าจอ**: ใช้ `mss` สำหรับดึงภาพจากจอภาพที่เลือก
- **การประมวลผลภาพ**: ใช้ OpenCV สำหรับ grayscale, CLAHE, thresholding, morphology และการคัดกรองกล่องข้อความ พร้อม optional detector จาก RapidOCR, PaddleOCR และ EasyOCR
- **OCR**: รองรับ EasyOCR, RapidOCR full-frame OCR และ `pytesseract` ร่วมกับ Tesseract OCR
- **การแปลภาษา**: รองรับ Argos Translate แบบ offline และ Google Translate ผ่าน `deep-translator`
- **Desktop UI**: ใช้ PySide6 สำหรับแสดงภาพ preview, mask preview, translated preview, overlay และข้อความที่ตรวจจับได้

**เวอร์ชัน**: 0.1.0  
**Python**: 3.11+  
**ลักษณะการรัน**:

- ตัว package หลักรันได้ผ่าน `python -m screenlens_detection` หรือคำสั่ง `screenlens`
- launcher ไฟล์ `screenlens.py` และ `screenlens.pyw` ถูกเขียนให้พึ่งพา `.venv\Scripts\python(.w).exe` จึงเป็น flow ที่เน้น Windows
- สำหรับเครื่อง Windows ใหม่ให้ใช้ `build_screenlens_exe.bat` เพื่อสร้าง `dist\ScreenLens\ScreenLens.exe` พร้อม runtime ที่จำเป็น

---

## Pipeline การทำงาน

ลำดับการประมวลผลในโค้ดปัจจุบัน:

```text
1. Capture Frame
   ดึงภาพจาก monitor ที่ผู้ใช้เลือก

2. Preprocessing
   แปลงภาพเป็น grayscale และเพิ่ม contrast ด้วย CLAHE

3. Text Mask Detection
   สร้าง mask สำหรับข้อความทั้งแบบตัวอักษรมืดบนพื้นหลังสว่าง
   และตัวอักษรสว่างบนพื้นหลังมืด

4. Line Mask / Region Grouping
   ใช้ morphology เพื่อรวมกลุ่มพิกเซลที่น่าจะเป็นบรรทัดข้อความ

5. Text Box Extraction
   คัดกรอง candidate boxes ตามขนาด, aspect ratio,
   edge density และ foreground ratio

6. OCR Recognition
   เรียก OCR backend ที่เลือก เช่น EasyOCR, RapidOCR full OCR หรือ Tesseract

7. Translation
   แปลข้อความที่ OCR ได้ตาม source/target language ที่เลือก

8. Visualization
   วาดกรอบบนภาพจริง, แสดง segmentation preview
   และสรุปผลข้อความใน UI
```

---

## โครงสร้างโครงการ

```text
ScreenLens-Detection/
├── build_screenlens_exe.bat         # Build entrypoint สำหรับ Windows exe
├── screenlens.py                    # Launcher แบบมี console
├── screenlens.pyw                   # Launcher แบบไม่มี console บน Windows
├── screenlens.spec                  # PyInstaller spec
├── pyproject.toml                   # Project metadata และ dependencies
├── requirements.txt                 # Runtime dependencies
├── README.md                        # เอกสารภาพรวม
├── PROJECT_SUMMARY.md               # เอกสารสรุปโครงการ
├── scripts/
│   ├── setup_windows.ps1            # ติดตั้ง dependency, OCR runtimes และ model ที่ต้องใช้
│   ├── build_windows.ps1            # สร้าง PyInstaller onedir build
│   ├── install_tesseract_vendor.ps1 # เตรียม bundled Tesseract runtime
│   └── download_argos_models.py     # ดาวน์โหลด Argos model สำหรับ offline translation
├── vendor/
│   ├── argos/                       # bundled Argos model files
│   └── tesseract/                   # bundled Tesseract runtime สำหรับ packaged exe
├── src/
│   ├── screenlens_detection/
│   │   ├── __init__.py
│   │   ├── __main__.py              # python -m screenlens_detection
│   │   ├── main.py                  # Qt application entry point
│   │   ├── launcher.py              # ตรวจและบังคับใช้ interpreter ใน .venv
│   │   ├── capture.py               # Screen capture
│   │   ├── pipeline.py              # Text detection pipeline
│   │   ├── ocr.py                   # OCR backends
│   │   ├── translation.py           # Translation backends
│   │   ├── worker.py                # Background processing thread
│   │   ├── languages.py             # Language options และ language mapping
│   │   ├── models.py                # Data models
│   │   └── ui/
│   │       ├── __init__.py
│   │       └── main_window.py       # Main window UI
│   └── screenlens_detection.egg-info/
└── tests/
    ├── test_languages.py
    ├── test_launcher.py
    └── test_pipeline.py
```

---

## โมดูลหลัก

### 1. `capture.py`

- ใช้ `mss` สำหรับ enumerate monitor และจับภาพจาก monitor ที่เลือก
- คืนค่า frame ในรูปแบบ OpenCV BGR

### 2. `pipeline.py`

ศูนย์กลางของการประมวลผลภาพ ประกอบด้วย:

- ปรับขนาดภาพด้วย `upscale_factor`
- แปลงเป็น grayscale และเพิ่ม contrast ด้วย CLAHE
- ตรวจจับข้อความแบบ **dual-polarity**
  - ตัวอักษรมืดบนพื้นหลังสว่าง
  - ตัวอักษรสว่างบนพื้นหลังมืด
- ใช้ morphology และ connected components เพื่อรวม/คัดกรองบริเวณข้อความ
- merge กล่องที่อยู่บรรทัดเดียวกัน และ suppress กล่องที่ซ้อนกัน
- เรียก OCR และแปลภาษาเมื่อเปิดใช้งาน
- สร้าง annotated frame และ segmentation preview

### 3. `ocr.py`

- มี backend หลักคือ `EasyOCRBackend`, `RapidOCRFullBackend` และ `TesseractOCRBackend`
- รองรับ queued OCR เพื่อลด latency และลดการ OCR ซ้ำ
- fallback เป็น `NoOpOCRBackend` ถ้าไม่มี OCR backend พร้อมใช้งาน
- รองรับ bundled Tesseract runtime, `TESSERACT_CMD` และ `TESSDATA_PREFIX`

### 4. `translation.py`

- มี backend หลักคือ `ArgosTranslateBackend` สำหรับ offline translation และ `GoogleTranslateBackend` สำหรับ online translation
- ติดตั้ง bundled Argos `en<->th` model ที่ runtime เมื่อพบไฟล์ใน `vendor/argos`
- fallback เป็น `NoOpTranslationBackend` ถ้าไม่มี translation backend พร้อมใช้งาน
- มี cache และ queue สำหรับลดการแปลข้อความซ้ำ

### 5. `models.py`

- `MonitorSpec`: ข้อมูลของ monitor
- `PipelineSettings`: ค่าตั้งต้นของ pipeline
- `DetectionBox`: กล่องข้อความพร้อมผล OCR/translation
- `FrameAnalysis`: ผลลัพธ์ที่ worker ส่งกลับไปยัง UI

### 6. `worker.py`

- ใช้ `QThread` เพื่อประมวลผลแบบ background
- จับภาพ -> ส่งเข้า pipeline -> ส่งผลกลับ UI ผ่าน signal

### 7. `languages.py`

- จัดการ source/target language options
- map รหัสภาษาสำหรับ OCR และ translation
- ตรวจภาษาแบบง่ายจากตัวอักษรไทย/อังกฤษ

### 8. `ui/main_window.py`

- มี control สำหรับเลือก monitor และเริ่ม/หยุด worker
- แสดง annotated preview, segmentation preview และข้อความที่ตรวจจับได้
- แสดง runtime stats เช่น FPS, จำนวนกล่อง, monitor และสถานะระบบ

---

## ฟีเจอร์ที่มีอยู่ในโค้ดปัจจุบัน

✅ **Realtime Monitor Capture** - จับภาพจาก monitor ที่เลือกเป็นช่วงเวลา  
✅ **Dual-Polarity Text Detection** - ตรวจจับข้อความทั้งแบบตัวอักษรมืดบนพื้นหลังสว่าง และตัวอักษรสว่างบนพื้นหลังมืด  
✅ **Selectable Deep Detectors** - รองรับ RapidOCR ONNX DBNet, PaddleOCR DBNet และ EasyOCR CRAFT  
✅ **Segmentation Preview** - แสดง mask preview พร้อมกรอบของ region ที่ผ่านการคัดกรอง  
✅ **Selectable OCR** - รองรับ EasyOCR, RapidOCR full OCR และ Tesseract OCR  
✅ **Offline/Online Translation** - แปลผ่าน Argos Translate แบบ offline หรือ Google Translate แบบ online  
✅ **Language Routing** - รองรับ source language แบบ `Auto detect`, `English`, `Thai`, `Thai + English` และ target language แบบ `Thai` / `English`  
✅ **Adjustable Runtime Settings** - ปรับค่าได้จาก UI ได้แก่ detector, OCR backend/device, translation mode, source/target language, preview และ overlay/recording  
✅ **Clean Windows Build** - `build_screenlens_exe.bat` สร้าง PyInstaller onedir build พร้อม bundled Tesseract และ Argos model  
✅ **Windows Convenience Launchers** - ใช้งานผ่าน `screenlens.py` หรือ `screenlens.pyw` ได้เมื่อมี `.venv` ตามโครงสร้างที่โค้ดคาดไว้

---

## Technology Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `PySide6` | `>=6.8.0` | Qt GUI framework |
| `opencv-python` | `>=4.10.0.84` | Computer vision |
| `numpy` | `>=2.1.0` | Numerical processing |
| `mss` | `>=10.0.0` | Screen capture |
| `Pillow` | `>=10.4.0` | Image bridge for OCR |
| `pytesseract` | `>=0.3.13` | Python wrapper สำหรับ Tesseract |
| `argostranslate` | `>=1.11.0` | Offline translation |
| `deep-translator` | `>=1.11.4` | Translation backend |

### Optional / Build-time

- `pytest>=8.3.0` สำหรับ test suite
- `pyinstaller>=6.14.0` สำหรับสร้าง Windows exe
- `easyocr`, `rapidocr`, `onnxruntime`, `paddleocr`, `paddlepaddle`, `torch`, `torchvision` ถูกติดตั้งโดย `scripts/setup_windows.ps1`
- Tesseract OCR runtime ถูกเตรียมเข้า `vendor/tesseract` โดย `scripts/install_tesseract_vendor.ps1`

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd ScreenLens-Detection
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Runtime Dependencies

```powershell
pip install -e .
```

### 4. Windows Setup สำหรับ backend ครบ

```powershell
.\scripts\setup_windows.ps1 -TorchRuntime cpu
```

ใช้ `-TorchRuntime gpu` เมื่อต้องการ build/runtime ที่ target CUDA และเครื่อง build มี NVIDIA GPU พร้อม driver ที่รองรับ

### 5. Build Windows EXE สำหรับ Clean VM

```powershell
.\build_screenlens_exe.bat -TorchRuntime cpu
```

ผลลัพธ์อยู่ที่ `dist\ScreenLens\ScreenLens.exe` และต้องย้ายทั้งโฟลเดอร์ `dist\ScreenLens\` ไปทดสอบ เพราะเป็น PyInstaller แบบ onedir

ขั้นตอน build จะเตรียม bundled Tesseract runtime ใน `vendor\tesseract`, ดาวน์โหลด Argos model ไป `vendor\argos`, ติดตั้ง EasyOCR/RapidOCR/PaddleOCR และตรวจ runtime หลักก่อน build

### 6. Optional: Install Dev Dependencies

```powershell
pip install -e ".[dev]"
pytest
```

---

## วิธีใช้งาน

### วิธีที่ 1: รันแบบ package entry point

```powershell
screenlens
```

หรือ

```powershell
screenlens-detection
```

### วิธีที่ 2: รันเป็น Python module

```powershell
python -m screenlens_detection
```

### วิธีที่ 3: ใช้ launcher ใน root project

```powershell
python screenlens.py
```

หมายเหตุ:

- วิธีนี้อ้างอิง `.venv\Scripts\python.exe` ตาม logic ใน `launcher.py`
- เหมาะกับ Windows workflow ของโปรเจกต์นี้

### วิธีที่ 4: Double-click `screenlens.pyw` บน Windows

- เปิด GUI โดยไม่โชว์ console
- ต้องมี `.venv` และ dependencies ครบตามที่ launcher คาดไว้

---

## Test Suite

ชุดทดสอบที่มีอยู่ใน repo:

```powershell
pytest
pytest tests/test_pipeline.py -v
```

ไฟล์ test หลัก:

- `test_languages.py` - ตรวจ logic การแยกภาษาและ mapping ภาษา
- `test_launcher.py` - ตรวจ logic ของ launcher
- `test_pipeline.py` - ตรวจ text-region detection pipeline
- `test_ocr.py`, `test_translation.py`, `test_text_detectors.py` - ตรวจ OCR, translation และ detector backend behavior

สถานะที่ตรวจล่าสุดสำหรับชุด OCR/translation/text detector: `25 passed`

---

## Configuration

### Environment Variables

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"
```

สำหรับ packaged build ระบบจะ prefer bundled runtime ใต้ `dist\ScreenLens\_internal\vendor\tesseract` โดยอัตโนมัติ จึงไม่ต้องตั้ง environment variables บนเครื่อง clean VM

### Settings ผ่าน UI

- **Capture Interval**: ความถี่ในการจับภาพ
- **Upscale Factor**: ปรับขนาดภาพก่อนประมวลผล
- **Min Contour Area**: ค่าต่ำสุดของกล่องข้อความที่ยอมรับ
- **Text Detector**: เลือก OpenCV, RapidOCR, PaddleOCR หรือ EasyOCR detector
- **OCR Backend**: เลือก Auto, EasyOCR, RapidOCR, Tesseract หรือ Disabled
- **Translation Mode**: เลือก Argos Offline, Google Online หรือ Disabled
- **Source Language**: `Auto detect`, `English`, `Thai`, `Thai + English`
- **Target Language**: `Thai` หรือ `English`
- **Enable OCR**: เปิดหรือปิดการเรียก OCR

---

## สถานะโครงการ

- มี pipeline สำหรับ text-region detection ใช้งานได้แล้ว
- มี Qt UI สำหรับ preview แบบเรียลไทม์
- มี OCR backend, text detector backend และ translation backend แบบ selectable
- มี Windows build flow สำหรับ clean VM ผ่าน `build_screenlens_exe.bat`
- มี test suite สำหรับ launcher, language logic และ pipeline
- โค้ดปัจจุบันเหมาะกับการสาธิตงานด้าน screen text detection + OCR/translation workflow

---

## ข้อจำกัดที่ควรทราบ

1. **OCR quality ยังขึ้นกับสภาพภาพจริง**  
   ความแม่นยำขึ้นกับขนาดตัวอักษร, font, contrast, motion blur และภาษา แม้ backend จะถูก bundle ครบแล้ว

2. **Launcher ใน root project เน้น Windows**  
   `screenlens.py` และ `screenlens.pyw` อิง `.venv\Scripts\...` โดยตรง

3. **EasyOCR อาจต้องดาวน์โหลด model ครั้งแรก**  
   clean VM ที่มี internet สามารถดาวน์โหลด weights ได้ แต่ถ้าต้อง offline เต็มรูปแบบควรเตรียม EasyOCR model cache เพิ่มเติม

4. **Language options ใน UI ยังจำกัด**  
   ชุดภาษาที่เปิดให้เลือกใน UI ปัจจุบันเน้น Thai/English

5. **ประสิทธิภาพขึ้นกับเครื่อง**  
   ความเร็วจริงขึ้นกับความละเอียดจอ, capture interval และภาระของ OCR/translation

6. **จับภาพในระดับ monitor**  
   workflow ปัจจุบันยังไม่ได้รองรับการเลือกเฉพาะ region ด้วยเมาส์

---

## ความสอดคล้องกับโจทย์โครงการ

**Track**: Vision + AI Integration

องค์ประกอบที่มีในระบบ:

- Screen capture แบบต่อเนื่อง
- Image preprocessing
- Segmentation และ text-region detection
- OCR inference
- Desktop UI สำหรับ realtime visualization

---

## เอกสารอ้างอิงภายใน repo

- ดูภาพรวมเพิ่มเติมใน [README.md](README.md)
- โค้ด entry point อยู่ที่ `src/screenlens_detection/main.py`
- โค้ด pipeline อยู่ที่ `src/screenlens_detection/pipeline.py`

---

*Last Updated: May 2026*  
*Version: 0.1.0*
