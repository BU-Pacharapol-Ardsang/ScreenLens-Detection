# ScreenLens-Detection - สรุปโครงการ

## ภาพรวมโครงการ

**ScreenLens-Detection** เป็นแอปเดสก์ท็อปที่พัฒนาด้วย Python + Qt สำหรับจับภาพหน้าจอแบบเรียลไทม์ ตรวจจับบริเวณที่น่าจะเป็นข้อความด้วย OpenCV และสามารถทำ OCR กับแปลภาษาเพิ่มเติมได้เมื่อมี backend ที่พร้อมใช้งาน

องค์ประกอบหลักของระบบในโค้ดปัจจุบันมีดังนี้:

- **การจับภาพหน้าจอ**: ใช้ `mss` สำหรับดึงภาพจากจอภาพที่เลือก
- **การประมวลผลภาพ**: ใช้ OpenCV สำหรับ grayscale, CLAHE, thresholding, morphology และการคัดกรองกล่องข้อความ
- **OCR**: ใช้ `pytesseract` ร่วมกับ Tesseract OCR แบบ optional
- **การแปลภาษา**: ใช้ `deep-translator` ผ่าน Google Translate backend แบบ optional
- **Desktop UI**: ใช้ PySide6 สำหรับแสดงภาพ preview, mask preview และข้อความที่ตรวจจับได้

**เวอร์ชัน**: 0.1.0  
**Python**: 3.11+  
**ลักษณะการรัน**:

- ตัว package หลักรันได้ผ่าน `python -m screenlens_detection` หรือคำสั่ง `screenlens`
- launcher ไฟล์ `screenlens.py` และ `screenlens.pyw` ถูกเขียนให้พึ่งพา `.venv\Scripts\python(.w).exe` จึงเป็น flow ที่เน้น Windows

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
   เรียก Tesseract เมื่อเปิด OCR และมี backend พร้อมใช้งาน

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
├── screenlens.py                    # Launcher แบบมี console
├── screenlens.pyw                   # Launcher แบบไม่มี console บน Windows
├── pyproject.toml                   # Project metadata และ dependencies
├── requirements.txt                 # Runtime dependencies
├── README.md                        # เอกสารภาพรวม
├── PROJECT_SUMMARY.md               # เอกสารสรุปโครงการ
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

- มี backend หลักคือ `TesseractOCRBackend`
- fallback เป็น `NoOpOCRBackend` ถ้าไม่พบ Tesseract หรือ import `pytesseract` ไม่ได้
- รองรับการตั้งค่า `TESSERACT_CMD` และ `TESSDATA_PREFIX`

### 4. `translation.py`

- มี backend หลักคือ `GoogleTranslateBackend`
- fallback เป็น `NoOpTranslationBackend` ถ้า `deep-translator` ใช้งานไม่ได้
- มี cache สำหรับลดการแปลข้อความซ้ำ

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
✅ **Segmentation Preview** - แสดง mask preview พร้อมกรอบของ region ที่ผ่านการคัดกรอง  
✅ **Optional OCR** - เรียก Tesseract OCR ได้เมื่อระบบหา binary เจอ  
✅ **Optional Translation** - แปลข้อความที่ OCR ได้ผ่าน Google Translate backend  
✅ **Language Routing** - รองรับ source language แบบ `Auto detect`, `English`, `Thai`, `Thai + English` และ target language แบบ `Thai` / `English`  
✅ **Adjustable Runtime Settings** - ปรับค่าได้จาก UI ได้แก่ capture interval, upscale factor, min contour area, source language, target language และการเปิด/ปิด OCR  
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
| `deep-translator` | `>=1.11.4` | Translation backend |

### Optional / Development

- `pytest>=8.3.0` สำหรับ test suite
- Tesseract OCR สำหรับ OCR runtime จริง

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

### 4. Optional: Install Tesseract OCR

ถ้าต้องการ OCR จริง ต้องติดตั้ง Tesseract ให้ระบบหา binary เจอผ่าน `PATH` หรือกำหนดเองผ่าน `TESSERACT_CMD`

ตัวอย่างบน Windows:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

ถ้าต้องการ OCR ภาษาไทย ควรมีไฟล์ภาษาไทยใน `tessdata` และถ้าจำเป็นสามารถกำหนด:

```powershell
$env:TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"
```

### 5. Optional: Install Dev Dependencies

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

สถานะที่ตรวจล่าสุดในสภาพแวดล้อมปัจจุบัน: `8 passed`

---

## Configuration

### Environment Variables

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"
```

### Settings ผ่าน UI

- **Capture Interval**: ความถี่ในการจับภาพ
- **Upscale Factor**: ปรับขนาดภาพก่อนประมวลผล
- **Min Contour Area**: ค่าต่ำสุดของกล่องข้อความที่ยอมรับ
- **Source Language**: `Auto detect`, `English`, `Thai`, `Thai + English`
- **Target Language**: `Thai` หรือ `English`
- **Enable OCR**: เปิดหรือปิดการเรียก OCR

---

## สถานะโครงการ

- มี pipeline สำหรับ text-region detection ใช้งานได้แล้ว
- มี Qt UI สำหรับ preview แบบเรียลไทม์
- มี OCR backend และ translation backend แบบ optional
- มี test suite สำหรับ launcher, language logic และ pipeline
- โค้ดปัจจุบันเหมาะกับการสาธิตงานด้าน screen text detection + OCR/translation workflow

---

## ข้อจำกัดที่ควรทราบ

1. **OCR เป็น optional runtime dependency**  
   ถ้าไม่พบ Tesseract แอปยังรันได้ แต่จะอยู่ในโหมด detection-only

2. **Launcher ใน root project เน้น Windows**  
   `screenlens.py` และ `screenlens.pyw` อิง `.venv\Scripts\...` โดยตรง

3. **Language options ใน UI ยังจำกัด**  
   ชุดภาษาที่เปิดให้เลือกใน UI ปัจจุบันเน้น Thai/English

4. **ประสิทธิภาพขึ้นกับเครื่อง**  
   ความเร็วจริงขึ้นกับความละเอียดจอ, capture interval และภาระของ OCR/translation

5. **จับภาพในระดับ monitor**  
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

*Last Updated: April 2026*  
*Version: 0.1.0*
