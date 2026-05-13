# ScreenLens Architecture Evolution by Git Version

เอกสารนี้เน้นอธิบาย architecture ของแต่ละช่วงใน git ว่าใช้เทคโนโลยีอะไร, cache อย่างไร, model ทำงาน parallel หรือไม่, ใช้ CUDA/NVIDIA/ONNX Runtime หรือไม่ และ filter ภาพอย่างไร

## Quick Reference

| Version label | Git ref / commit | Architecture จุดเด่น |
| --- | --- | --- |
| V1.0 baseline | `ver1.0` / `f219626` | OpenCV detector + Tesseract crop OCR + Google Translate cache |
| V2.0 baseline | `ver2.0` / `fc03faa` | เพิ่ม latest-frame queue, EasyOCR GPU option, selectable detector, overlay tracking |
| OCR queue V1 | `2351918` | เพิ่ม `QueuedOCRBackend`, OCR cache จาก crop fingerprint |
| RapidOCR / translation backend | `590419e` | เพิ่ม RapidOCR ONNX DBNet detector, Argos offline, queued translation |
| OCR workers / Tesseract batch | `846290e` | เพิ่ม OCR worker count และ Tesseract `ThreadPoolExecutor` |
| Detection scale / cache | `95e538a` | แยก detection resolution กับ OCR source crop, cache match ด้วย motion/fingerprint |
| Scanline ROI | `637325f` | แบ่ง frame เป็น bands แล้ว detect ทีละส่วน |
| Hover ROI | `a2c4935` | จำกัด OCR เฉพาะพื้นที่ cursor hover |
| Full-frame OCR | `28b5202` | RapidOCR full-frame detection+recognition ใน backend เดียว |
| Hover full-frame OCR | `db63a8a` | เอา full-frame OCR ไปใช้เฉพาะ hover ROI |
| Current optimized | `35737c7` / `852565d` | เพิ่ม multiline/long subtitle logic, current main merge แล้ว |

## Architecture Layers ที่ใช้ร่วมกัน

```text
UI/MainWindow
  -> ProcessingWorker(QThread)
    -> Capture thread / LatestFrameQueue
    -> TextDetectionPipeline
      -> Text detector path
      -> OCR backend path
      -> Translation backend path
    -> FrameAnalysis
  -> TranslationOverlay / Preview / Recording
```

คำว่า parallel ในโปรเจกต์นี้มี 4 ระดับ:

1. UI กับ processing แยกกันด้วย `QThread`
2. Capture กับ process แยกกันด้วย capture thread + queue ขนาด 1 ตั้งแต่ V2.0
3. OCR/translation มี worker queue ใน optimization branch
4. ตัว model บางตัวใช้ GPU/ONNX/CUDA หรือ thread ภายใน backend เช่น EasyOCR/PyTorch, RapidOCR/ONNX Runtime, Tesseract thread pool

## V1.0 Baseline

Git:

- `ver1.0`
- commit `f219626`

### Pipeline

```text
Monitor frame
  -> resize/upscale
  -> grayscale
  -> CLAHE
  -> dual-polarity adaptive threshold
  -> morphology gradient + line mask
  -> connected components
  -> crop each text box
  -> Tesseract OCR
  -> Google Translate
  -> FrameAnalysis
```

### Image Filtering

- แปลง BGR เป็น grayscale
- เพิ่ม contrast ด้วย CLAHE
- ใช้ adaptive threshold 2 ขา:
  - dark text on light background
  - light text on dark background
- ใช้ morphology gradient เพื่อเน้น stroke/edge
- ใช้ Otsu threshold กับ gradient mask
- ใช้ morphology close/open เพื่อรวมตัวอักษรเป็น line region
- ใช้ connected components เพื่อแยกกล่องข้อความ

### OCR / Model

- ใช้ `TesseractOCRBackend`
- OCR เป็น crop-based: detect box ก่อน แล้ว crop box ส่งเข้า Tesseract ทีละกล่อง
- ไม่มี EasyOCR, RapidOCR, ONNX Runtime ใน baseline นี้
- ไม่มี CUDA/NVIDIA path

### Cache

- Translation มี cache ใน `GoogleTranslateBackend`
- key ประมาณ `(text, source_language, target_language)`
- OCR ยังไม่มี queue/cache จริงจัง

### Parallelism

- มี `ProcessingWorker(QThread)` แยกจาก UI
- capture และ processing ยังอยู่ใน loop เดียวของ worker
- OCR ไม่ได้ parallel ระดับ app; Tesseract ถูกเรียกตามกล่องที่เลือกใน frame นั้น

### Performance Character

- bottleneck หลักคือ OCR crop ทีละกล่องและ online translation
- ค่า `capture_interval_ms` เดิมค่อนข้างสูงกว่า current เพื่อกัน CPU/OCR หนักเกิน

## V2.0 Baseline

Git:

- `ver2.0`
- commit `fc03faa`

### สิ่งที่เปลี่ยนเชิง Architecture

- เพิ่ม `_LatestFrameQueue` ขนาด 1
- แยก capture loop ออกเป็น thread แยกจาก processing loop
- ถ้า process ไม่ทัน จะ drop frame เก่าแล้วเก็บ frame ล่าสุด ลด latency สะสม
- เพิ่ม EasyOCR backend พร้อม device preference
- เพิ่ม selectable text detector:
  - OpenCV morphology
  - PaddleOCR DBNet
  - EasyOCR CRAFT
- เพิ่ม overlay tracking และ recording path

### Image Filtering

ยังใช้ OpenCV path เดิมเป็น default:

- grayscale + CLAHE
- adaptive threshold dual polarity
- morphology line mask
- connected components
- filter ด้วย area, aspect ratio, foreground ratio, edge density

ถ้าเลือก deep detector:

- EasyOCR CRAFT ใช้ EasyOCR detector
- PaddleOCR DBNet ใช้ PaddleOCR text detection
- output จาก detector จะถูก filter/merge ต่อใน pipeline

### OCR / Model

- Auto OCR เริ่มจาก EasyOCR ก่อน ถ้ามี dependency
- fallback ไป Tesseract
- EasyOCR ใช้ PyTorch และสามารถใช้ NVIDIA CUDA ได้ถ้า `torch.cuda.is_available()`
- Tesseract ยังเป็น CPU process

### Cache

- Translation backend มี cache
- OCR crop ยังไม่มี queue/cache แบบ optimization branch
- มี recent translation reuse ในสาย `stably-for-dev` ก่อน/ช่วง V2 เพื่อ reuse แปลเดิมเมื่อข้อความใกล้เคียง

### Parallelism

- UI thread แยกจาก worker
- capture thread แยกจาก processing thread
- EasyOCR อาจใช้ GPU ผ่าน PyTorch ภายใน model
- OCR app-level ยังไม่ใช่ multi-worker queue

### Performance Character

- latency ดีขึ้นจาก latest-frame queue เพราะไม่ค้าง frame เก่า
- ถ้าเปิด EasyOCR GPU จะเร่ง OCR ได้ แต่ cost ต่อ batch/crop ยังสูง
- deep detector เพิ่มความสามารถ แต่ไม่จำเป็นต้องเร็วกว่า OpenCV

## OCR Queue V1

Git:

- branch ancestor: `origin/feature-optimization-efficiency`
- commit `2351918514534be0d0e1b46339996e26d0435e66`

### สิ่งที่เพิ่ม

- เพิ่ม `QueuedOCRBackend`
- เพิ่ม OCR cache ใน backend
- เพิ่ม OCR cache ใน pipeline สำหรับ crop result
- แนวคิดหลักคือ return frame ให้เร็วขึ้น โดยไม่ block ทุก crop OCR ใน frame เดียว

### Cache Design

มี 2 ชั้น:

1. `QueuedOCRBackend._cache`
   - key จาก digest ของ image bytes + shape + language + psm
   - ถ้า crop เหมือนเดิม จะคืน result ทันที

2. `TextDetectionPipeline._recent_ocr_results`
   - เก็บ rect, fingerprint, text, confidence
   - match จากตำแหน่งกล่อง + fingerprint difference
   - reuse OCR เมื่อกล่องและภาพ crop คล้ายของเดิม

### Parallelism

- มี OCR queue worker thread 1 ตัวในรุ่นแรก
- main pipeline ทำ synchronous OCR เฉพาะ batch เล็กแรกตาม budget
- crop ที่เหลือ enqueue ให้ worker ทำทีหลัง
- model เองไม่ได้ถูกทำ parallel หลาย worker ใน commit นี้

### Performance Character

- FPS เพิ่มได้ชัดถ้า frame มี text box หลายกล่อง
- output บางกล่องอาจเป็น translation pending ชั่วคราว เพราะ OCR อยู่ใน queue
- เหมาะกับ realtime เพราะไม่ยอมรอ OCR ทุกกล่องทุก frame

## RapidOCR Detector + Translation Backend

Git:

- branch ancestor: `origin/feature-optimization-efficiency`
- commit `590419ee41a0e2c8f46ca69ad0131f5f920efcab`

### Text Detector Architecture

เพิ่ม `RapidOCRTextDetector`:

- ใช้ RapidOCR DBNet detector
- ใช้ ONNX Runtime เป็น engine
- ตั้งค่า `Global.use_det=True`, `use_cls=False`, `use_rec=False`
- detector อย่างเดียว ไม่ recognize text
- model type เป็น mobile / PP-OCRv4 style config
- ใช้ `Det.limit_side_len=960`, `Det.max_candidates=300`, `Det.score_mode=fast`

### CUDA / NVIDIA / ONNX

- RapidOCR detector ใช้ ONNX Runtime
- ถ้า `SCREENLENS_OCR_DEVICE` หรือ UI device เป็น GPU/Auto และ provider มี `CUDAExecutionProvider` จะใช้ CUDA
- ถ้า CUDA init fail จะ fallback เป็น CPU ใน current lineage
- ONNX Runtime thread config:
  - `intra_op_num_threads=2`
  - `inter_op_num_threads=1`

### Translation Architecture

เพิ่ม/ขยาย backend:

- `ArgosTranslateBackend`
  - offline translation
  - ใช้ bundled `.argosmodel`
  - ใช้ `ctranslate2`
  - มี pair translator cache ต่อ language pair

- `GoogleTranslateBackend`
  - online translation
  - มี request budget, timeout, retry cooldown

- `QueuedTranslationBackend`
  - มี cache keyed by text/source/target
  - มี queue worker 1 ตัว
  - batch เฉพาะ route เดียวกัน

### Parallelism

- RapidOCR detector อาจใช้ CUDA/ONNX Runtime ภายใน model
- Translation queue มี worker thread แยก
- Argos ใช้ `ctranslate2.Translator.translate_batch()` เป็น batch inference

### Performance Character

- ลด dependency บน Google สำหรับ use case offline
- detector แบบ deep อาจแม่นขึ้น แต่ overhead สูงกว่า OpenCV ในบาง frame
- queued translation ลด frame blocking โดยเฉพาะตอนมีข้อความใหม่จำนวนมาก

## OCR Workers + Tesseract Batch

Git:

- branch ancestor: `origin/feature-optimization-efficiency2`
- commit `846290e7d22f8b92a9f5d1f483645be8ffb32d5c`

### สิ่งที่เพิ่ม

- `QueuedOCRBackend` รับ `worker_count`
- อ่านจาก env `SCREENLENS_OCR_WORKERS`
- Tesseract backend มี `ThreadPoolExecutor`
- Tesseract `recognize_batch()` ส่งหลาย crop เข้า executor ได้

### Parallelism

- App-level OCR queue สามารถมีหลาย worker
- Tesseract crop OCR สามารถ parallel ผ่าน thread pool
- EasyOCR batch path ใช้การ tile crop images เป็น canvas แล้วเรียก recognize เป็น batch
- ถ้าใช้ GPU กับ EasyOCR งานหนักไปอยู่ที่ PyTorch/CUDA ภายใน

### Cache

- cache เดิมยังอยู่:
  - image digest cache ใน queue backend
  - recent OCR result cache ใน pipeline
- worker หลายตัวใช้ cache เดียวกันผ่าน lock/condition

### Performance Character

- เหมาะกับ Tesseract เพราะ crop หลายกล่องแยกกันได้
- worker มากเกินอาจแย่ง CPU และทำให้ UI/processing ช้าลง
- ควร benchmark `SCREENLENS_OCR_WORKERS=1,2,4` กับเครื่องจริง

## Detection Scale + Motion-aware OCR Cache

Git:

- branch ancestor: `origin/feature-optimization-efficiency2`
- commit `95e538a67114f8843b58d5cc537f0c0528fab52e`

### สิ่งที่เพิ่ม

- เพิ่ม `detection_scale`
- detection ทำบนภาพเล็กลงได้
- OCR ยัง crop จาก source frame/high-res grayscale
- เพิ่ม motion offset เพื่อช่วย match cache เมื่อ content ขยับ

### Cache Design

pipeline cache เก็บ:

- rect
- crop fingerprint
- text/confidence
- OCR language
- psm
- translated text
- generation counters
- stable hits

การ match cache ใช้:

- IoU ของ rect
- motion-adjusted rect variants
- fingerprint difference
- center proximity
- size similarity
- stable hit bonus

### Image Filtering

- threshold/filter ปรับ scale-aware
- min area, min width/height, morphology length ถูกคูณตาม detection scale
- ลดงาน detector โดยไม่ลดความละเอียด crop ที่ส่ง OCR

### Parallelism

- ไม่ใช่ parallel ใหม่โดยตรง
- เป็น optimization ลด pixel workload ก่อนเข้า detector

### Performance Character

- เร็วขึ้นโดยเฉพาะจอใหญ่/fullscreen
- ระวัง scale ต่ำเกินทำให้กล่องหายหรือ merge ผิด

## Scanline ROI

Git:

- branch ancestor: `origin/feature-optimization-efficiency2`
- commit `637325f15a953fe40192cd1da8ba72a9e209c077`

### Pipeline

```text
frame
  -> split vertical bands
  -> process one active band per frame
  -> keep previous source boxes outside current band
  -> merge/deduplicate scanline boxes
  -> OCR selected boxes
```

### Cache / State

เก็บ state:

- `_scanline_source_boxes`
- `_scanline_frame_index`
- `_scanline_last_band_index`
- source/detection shape

เมื่อถึง band ใหม่:

- detect เฉพาะ band นั้นพร้อม overlap
- replace boxes ที่ center อยู่ใน core band
- retain boxes จาก band อื่น
- dedupe ด้วย IoU

### Image Filtering

- ใช้ filter เดิม แต่รันเฉพาะ ROI band
- มี overlap ratio เพื่อไม่ให้ข้อความตรงรอยต่อ band หาย

### Parallelism

- ไม่ได้ parallel
- เป็น temporal slicing: แบ่งงาน detector ข้ามหลาย frame

### Performance Character

- ลด cost ต่อ frame
- เหมาะกับ video/game ที่ยอมให้พื้นที่บางส่วน refresh ช้ากว่าได้
- ถ้า content เปลี่ยนเร็วทั้งจอ อาจเห็นกล่องเก่าค้างระหว่างรอวนครบทุก band

## Hover ROI

Git:

- branch ancestor: `origin/feature-optimization-efficiency2`
- commit `a2c493545d5e4b3081218410dca5b84ec3e094e2`

### Pipeline

```text
cursor position
  -> confirm/lock hover target
  -> compute ROI around cursor
  -> detect only ROI
  -> select boxes near cursor/anchor
  -> crop OCR selected boxes
  -> translate selected region
```

### Cache / State

- ใช้ recent OCR cache เดิม
- ใช้ recent translation reuse เดิม
- hover cache เลือกกล่องเดิมใกล้ cursor ถ้า ROI ยังเหมือนเดิม

### Parallelism

- ไม่เพิ่ม parallel
- ลด workload โดยลดพื้นที่ detector/OCR

### Performance Character

- FPS ดีขึ้นถ้าใช้งานแบบเล็งเฉพาะ subtitle/paragraph
- latency ต่ำกว่า full-screen OCR เพราะจำนวน candidate boxes ลดลง
- ความแม่นขึ้นกับ radius/margin/dwell/ตำแหน่ง cursor

## Full-frame OCR

Git:

- branch: `origin/feature-optimization-efficiency3`
- commit `28b5202ee4b80dd09e983258859e14e45801dd00`

### OCR / Model

เพิ่ม `RapidOCRFullBackend`:

- backend ทำ detection + recognition ใน pass เดียว
- `supports_full_frame()` เป็น `True`
- pipeline จะข้าม crop OCR path
- ใช้ `recognize_frame(frame)` แล้วคืน `OCRFrameResult(rect, text, confidence)`

### CUDA / NVIDIA / ONNX

- ใช้ RapidOCR + ONNX Runtime
- ใช้ CUDA เมื่อ `onnxruntime.get_available_providers()` มี `CUDAExecutionProvider`
- ถ้าไม่มีหรือเลือก CPU จะใช้ ONNX Runtime CPU
- ถ้า CUDA init fail จะ fallback CPU และเก็บ fallback reason ใน diagnostics
- config สำคัญ:
  - `Global.use_det=True`
  - `Global.use_rec=True`
  - `Global.use_cls=False`
  - `Rec.rec_batch_num=8`
  - `EngineConfig.onnxruntime.intra_op_num_threads=2`
  - `EngineConfig.onnxruntime.inter_op_num_threads=1`

### Cache / Stabilization

- ไม่มี crop OCR cache เพราะไม่ crop ทีละ box
- ใช้ full-frame track cache/state:
  - `_full_frame_ocr_tracks`
  - rect
  - text
  - confidence
  - stable_frames
  - missing_frames
- merge line results
- smooth rects
- prefer text จาก quality score/confidence
- require stable frames โดยเฉพาะ non-subtitle region

### Parallelism

- ไม่ใช้ OCR queue worker เพราะ full-frame backend return result เป็น frame-level
- model inference ใช้ ONNX Runtime CPU/CUDA ภายใน backend
- recognition batching ภายใน RapidOCR (`rec_batch_num=8`)

### Performance Character

- ลด overhead crop หลายกล่อง
- ดีเมื่อมีข้อความหลายจุดใน frame
- อาจหนักถ้าทำทั้งจอความละเอียดสูงโดยไม่มี ROI/scale

## Hover Full-frame OCR

Git:

- branch: `origin/feature-optimization-efficiency3`
- commit `db63a8a8bf1ad756706df2eca60a212966ebb14d`

### Pipeline

```text
confirmed hover cursor
  -> crop hover ROI
  -> RapidOCR full OCR on ROI
  -> map ROI rects back to source frame
  -> filter/merge/stabilize
  -> translate selected boxes
```

### Architecture

- ใช้ full-frame OCR backend แต่ input เป็น ROI
- ยังข้าม crop OCR queue
- ใช้ hover selection เพื่อเลือกกล่องที่เกี่ยวกับ cursor
- preview mask วาดเฉพาะ selected hover boxes

### CUDA / ONNX

- เหมือน full-frame OCR:
  - RapidOCR
  - ONNX Runtime CPU/CUDA
  - CUDA ผ่าน `CUDAExecutionProvider`

### Performance Character

- มักเร็วกว่า full-frame OCR ทั้งจอ เพราะลด pixel input
- เหมาะกับ subtitle/paragraph ที่ผู้ใช้ชี้ตำแหน่งได้

## Current Optimized Hover / Subtitle Logic

Git:

- branch: `origin/feature-optimization-efficiency3`
- commit `6768a5310dde03011f77ea1fda4489f5df87432f`
- commit `35737c77c3dbf6c69c21b6f35a467c6389473d60`
- merged main: `852565db13494210c339ae7b86dd011108ea6ce8`

### สิ่งที่เพิ่ม

- ตรวจ multiline subtitle block
- split/refine hover source boxes by active mask rows
- รวม subtitle lines ที่ align/contiguous
- filter metadata row เช่น title/channel/time ที่ไม่ใช่เนื้อหาหลัก
- รองรับ longer text blocks และ wider ROI

### Image Filtering

current OpenCV path มี filter เพิ่มจาก baseline:

- dual-polarity adaptive threshold
- stroke response mask จาก top-hat/black-hat
- local contrast mask จาก local max/min
- edge response mask จาก morphology gradient + Canny
- HSV overlay text mask สำหรับ white UI / saturated UI text
- suppress large mask components
- suppress large line components
- connected components filter ด้วย:
  - area
  - aspect ratio
  - foreground ratio
  - edge density
  - row/column coverage
  - largest component ratio

### Cache

ยังมีครบ:

- OCR backend digest cache
- pipeline recent OCR cache
- recent translation lookup/candidates
- queued translation cache
- full-frame OCR track state เมื่อใช้ RapidOCR full OCR

### Parallelism

ขึ้นกับ mode:

- OpenCV + crop OCR: ใช้ OCR queue workers และ Tesseract/EasyOCR batch ตาม backend
- RapidOCR full OCR: ไม่มี crop OCR queue แต่ใช้ ONNX Runtime CPU/CUDA ภายใน
- Translation: queued translation worker
- Capture: latest-frame capture thread
- UI: Qt main thread แยกจาก processing worker

## Technology Summary by Backend

| Backend | เทคโนโลยี | CPU/GPU | Parallel / batch |
| --- | --- | --- | --- |
| OpenCV detector | `cv2` threshold/morphology/components | CPU | vectorized OpenCV ภายใน |
| EasyOCR crop OCR | EasyOCR + PyTorch | CPU หรือ NVIDIA CUDA | batch/tiled crop, PyTorch ภายใน |
| Tesseract crop OCR | `pytesseract` + Tesseract binary | CPU | current lineage มี `ThreadPoolExecutor` |
| RapidOCR detector | RapidOCR DBNet + ONNX Runtime | CPU หรือ NVIDIA CUDA | ONNX Runtime threads |
| RapidOCR full OCR | RapidOCR det+rec + ONNX Runtime | CPU หรือ NVIDIA CUDA | recognition batch ภายใน (`rec_batch_num=8`) |
| Argos translation | Argos + CTranslate2 | CPU by default, CUDA possible if env/device supported | `translate_batch(max_batch_size=32)` |
| Google translation | deep-translator HTTP | network | budgeted request loop, no local model |

## Benchmark Notes

ถ้าจะเทียบความเร็วให้ตรง architecture:

- `fc03faa` เทียบ baseline V2.0 ก่อน OCR queue/cache
- `2351918` เทียบผล OCR queue/cache
- `846290e` เทียบผล `SCREENLENS_OCR_WORKERS`
- `95e538a` เทียบ detection scale + motion-aware cache
- `637325f` เทียบ scanline ROI
- `a2c4935` เทียบ hover ROI crop OCR
- `28b5202` เทียบ RapidOCR full-frame OCR
- `db63a8a` เทียบ hover + RapidOCR full-frame OCR
- `35737c7` หรือ `852565d` เทียบ current optimized

ควรเก็บทั้ง average FPS และ median frame time เพราะ queue/cache ทำให้บาง frame เร็วมาก แต่บาง frame spike ตอน backend ทำ OCR/translation จริง
