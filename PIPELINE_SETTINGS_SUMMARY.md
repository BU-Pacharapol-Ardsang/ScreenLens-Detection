# สรุปฟีเจอร์ใน Pipeline Settings

ไฟล์นี้สรุป control ในกล่อง **Pipeline Settings** ของหน้า UI เพื่อใช้ประกอบการนำเสนอหรืออธิบายตอน demo ระบบ `ScreenLens-Detection`

## ภาพรวมการแบ่งกลุ่ม

Pipeline Settings แบ่งหน้าที่ได้เป็น 6 กลุ่มหลัก:

| กลุ่ม | Setting ที่เกี่ยวข้อง | ใช้อธิบายว่า |
| --- | --- | --- |
| Capture / Performance | Capture interval, Upscale factor, Detector scale, Min contour area | ระบบควบคุมความเร็วและภาระประมวลผลอย่างไร |
| Detection | Text detector, Scan mode | เลือกวิธีหาบริเวณข้อความบนหน้าจอ |
| OCR | New OCR/frame, OCR backend, Full OCR validation, OCR device, Enable OCR | เลือกวิธีอ่านข้อความและควบคุมต้นทุน OCR |
| Language / Translation | Source language, Target language, Translation mode, Translation region, Translation grouping, Text similarity stability | route ภาษา แปลข้อความ และลดการแปลซ้ำ |
| Overlay / Preview | Subtitle style, Track overlay while scrolling, Previews, Overlay tracking | แสดงผลคำแปลบนจอและ preview ใน UI |
| Debug / Demo | Runtime debug timings | ใช้โชว์เวลาของแต่ละ stage ใน pipeline |

## รายละเอียดแต่ละ Setting

### 1. Capture interval

- **หน้าที่:** กำหนดรอบเวลาการจับภาพหน้าจอแต่ละ frame
- **ค่า default:** `40 ms` หรือประมาณ `25 FPS`
- **ผลต่อระบบ:** ค่ายิ่งต่ำ ระบบจับภาพถี่ขึ้นและดู realtime ขึ้น แต่ใช้ CPU/GPU/OCR มากขึ้น
- **ใช้พูดตอนนำเสนอ:** ระบบไม่ได้พยายามประมวลผลทุก frame ที่เป็นไปได้ แต่ตั้งรอบ capture ให้เหมาะกับ latency และภาระ OCR

### 2. Upscale factor

- **หน้าที่:** ขยายภาพก่อนเข้า pipeline เพื่อช่วยอ่านข้อความขนาดเล็ก
- **ค่า default:** `1.0`
- **ช่วงใน UI:** `1.0` ถึง `3.0`
- **ผลต่อระบบ:** ค่าสูงช่วยให้ OCR อ่านตัวอักษรเล็กดีขึ้น แต่ประมวลผลช้าลง
- **ใช้เมื่อ:** ข้อความในเกม วิดีโอ หรือเอกสารมีขนาดเล็กจน detector/OCR อ่านยาก

### 3. Detector scale

- **หน้าที่:** กำหนด scale ที่ใช้ตอน detect กล่องข้อความ
- **ค่า default:** `1.00`
- **ช่วงใน UI:** `0.40` ถึง `1.00`
- **ผลต่อระบบ:** ลดค่าเพื่อให้ detect เร็วขึ้น แล้ว map กล่องกลับไปยังภาพต้นฉบับสำหรับ OCR
- **ข้อควรระวัง:** ค่าต่ำเกินไปอาจพลาดข้อความเล็กหรือข้อความเส้นบาง

### 4. Min contour area

- **หน้าที่:** พื้นที่ขั้นต่ำของ candidate text region ใน OpenCV detector
- **ค่า default:** `150`
- **ผลต่อระบบ:** ค่าสูงช่วยตัด noise ออก แต่เสี่ยงพลาดตัวอักษรเล็ก
- **ใช้เมื่อ:** หน้าจอมี noise เยอะ เช่น texture, UI effect, video compression หรือ background ละเอียด

### 5. Text detector

- **หน้าที่:** เลือก engine สำหรับตรวจจับบริเวณข้อความ
- **ตัวเลือก:**
  - `Classic OpenCV (Morphology)`: default, ไม่ต้องใช้ model หนัก
  - `RapidOCR ONNX DBNet (Optional)`: detector แบบ deep learning ผ่าน ONNX Runtime
  - `PaddleOCR DBNet (Optional)`: detector จาก PaddleOCR
  - `EasyOCR CRAFT (Optional)`: detector จาก EasyOCR
- **ผลต่อระบบ:** deep detector อาจจับข้อความซับซ้อนได้ดีกว่า แต่ต้องติดตั้ง dependency/model เพิ่ม
- **ใช้พูดตอนนำเสนอ:** ถ้า dependency ยังไม่พร้อม แอปไม่ crash แต่แสดงสถานะว่า backend unavailable

### 6. Scan mode

- **หน้าที่:** เลือกขอบเขตการ scan ภาพสำหรับ detection
- **ตัวเลือก:**
  - `Full frame`: ตรวจทั้ง frame
  - `Sliding bands (video/game)`: แบ่งภาพเป็นแถบแนวนอนหลาย band พร้อม overlap
- **ผลต่อระบบ:** Sliding bands ช่วยลดงานต่อรอบและเหมาะกับ subtitle/game UI ที่ข้อความมักอยู่เป็นแถบ
- **ข้อควรระวัง:** ถ้าข้อความกระจายทั่วหน้าจอ Full frame จะครอบคลุมกว่า

### 7. New OCR/frame

- **หน้าที่:** จำกัดจำนวนกล่องใหม่ที่ส่งเข้า OCR ต่อ frame
- **ค่า default:** `12`
- **ช่วงใน UI:** `1` ถึง `256`
- **ผลต่อระบบ:** ค่าสูงอ่านข้อความได้หลายจุดขึ้น แต่ latency เพิ่ม ค่าต่ำช่วยให้ realtime ลื่นขึ้น
- **ใช้พูดตอนนำเสนอ:** เป็น safety valve สำคัญ เพราะ OCR เป็น stage ที่แพงที่สุดใน pipeline

### 8. Source language

- **หน้าที่:** ระบุภาษาต้นทางของข้อความบนหน้าจอ
- **ตัวเลือก:**
  - `Auto detect`
  - `English`
  - `Thai`
  - `Thai + English`
- **ผลต่อระบบ:** ใช้เลือกภาษา OCR และใช้ route การแปล
- **ใช้เมื่อ:** ถ้ารู้ชัดว่าหน้าจอเป็นภาษาเดียว ให้เลือกตรงภาษาเพื่อลดความคลาดเคลื่อนของ OCR/translation

### 9. Target language

- **หน้าที่:** ระบุภาษาปลายทางของคำแปล
- **ตัวเลือก:** `Thai`, `English`
- **ผลต่อระบบ:** ถ้าต้นทางและปลายทางเป็นภาษาเดียวกัน ระบบสามารถ reuse ข้อความเดิมแทนการส่งแปล
- **ใช้พูดตอนนำเสนอ:** ทำให้ pipeline ไม่แปลเกินจำเป็น และช่วยลด latency/ค่าเรียก backend

### 10. Translation mode

- **หน้าที่:** เลือก backend สำหรับแปลภาษา
- **ตัวเลือก:**
  - `Argos Translate (Offline)`: แปลในเครื่อง ไม่ต้องใช้อินเทอร์เน็ต
  - `Google Translate (Online)`: ใช้ `deep-translator` ผ่าน online service
  - `Disabled`: ปิดการแปล ใช้ OCR อย่างเดียว
- **ผลต่อระบบ:** Offline เหมาะกับ demo ที่ต้องการความเสถียร Online อาจได้คุณภาพดีแต่พึ่ง network และอาจเจอ rate limit

### 11. Translation region

- **หน้าที่:** เลือกบริเวณที่จะตรวจ/แปล
- **ตัวเลือก:**
  - `Full screen`: ใช้ทั้งหน้าจอ
  - `Hover cursor region`: แปลเฉพาะบริเวณรอบ cursor หลัง cursor นิ่งตามเวลาที่กำหนด
- **ผลต่อระบบ:** Hover region ลดงาน OCR/translation และเหมาะกับการ demo แบบชี้เฉพาะจุด
- **ใช้พูดตอนนำเสนอ:** ระบบรองรับทั้งโหมด scan ทั้งหน้าจอ และโหมด interactive ที่ผู้ใช้ชี้ตำแหน่งเอง

### 12. Translation grouping

- **หน้าที่:** เลือกวิธีจัดกลุ่มข้อความก่อนแปล
- **ตัวเลือก:**
  - `Line mode`: แปลแยกเป็นบรรทัดหรือกล่อง
  - `Block mode: Strict`: รวมข้อความหลายบรรทัดที่ align กันเป็น block ก่อนแปล
- **ผลต่อระบบ:** Block mode ช่วยให้ paragraph/subtitle หลายบรรทัดมีบริบทดีขึ้น แต่ใช้ logic รวมกลุ่มมากกว่า
- **ใช้เมื่อ:** subtitle, paragraph, dialogue หรือข้อความหลายบรรทัดที่ควรแปลเป็นประโยคเดียว

### 13. Subtitle style

- **หน้าที่:** เลือกรูปแบบการ render คำแปลบน on-screen overlay
- **ตัวเลือก:**
  - `Bubble overlay`: วาดกล่องคำแปลทับตำแหน่งเดิม
  - `Clean patch (experimental)`: สร้าง patch เพื่อลบ subtitle เดิม แล้ววาดคำแปลใหม่
- **ผลต่อระบบ:** Bubble เสถียรและเร็วกว่า Clean patch ดูเนียนกว่าในบางฉาก แต่เป็น experimental และต้องใช้ source frame/mask
- **ใช้พูดตอนนำเสนอ:** ระบบไม่ได้แค่ OCR/แปล แต่มีชั้น presentation ที่ทำให้คำแปลอยู่บนหน้าจอจริงแบบอ่านง่าย

### 14. Text similarity stability

- **หน้าที่:** ใช้ similarity ของข้อความและ geometry เพื่อ reuse คำแปลเดิมเมื่อ OCR ได้ข้อความใกล้เคียงกัน
- **ค่า default:** เปิด
- **ผลต่อระบบ:** ลดอาการคำแปลกระพริบ ลดการแปลซ้ำ และช่วยให้ overlay stable เมื่อ OCR มี noise เล็กน้อย
- **ข้อควรระวัง:** ถ้าหน้าจอเปลี่ยนเร็วมาก อาจต้องปิดเพื่อให้ระบบไม่ reuse ข้อความเก่าเกินไป

### 15. OCR backend

- **หน้าที่:** เลือก engine สำหรับอ่านข้อความ
- **ตัวเลือก:**
  - `Auto (EasyOCR, then Tesseract)`: ใช้ EasyOCR ก่อน ถ้าไม่มีจึง fallback ไป Tesseract
  - `EasyOCR crop OCR (Optional)`: OCR จาก crop แต่ละกล่อง
  - `RapidOCR full OCR (Optional)`: detect + recognize ทั้ง frame ใน backend เดียว
  - `Tesseract crop OCR`: OCR จาก crop ด้วย Tesseract
  - `Disabled`: ปิด OCR
- **ผลต่อระบบ:** Crop OCR เหมาะกับ detector pipeline ส่วน RapidOCR full OCR เหมาะกับการอ่านทั้ง frame แบบ native OCR engine
- **ใช้พูดตอนนำเสนอ:** Pipeline ออกแบบเป็น backend interface จึงสลับ engine ได้โดยไม่เปลี่ยน UI หลัก

### 16. Full OCR validation

- **หน้าที่:** กำหนดระดับการกรองผลลัพธ์จาก full-frame OCR โดยเฉพาะ RapidOCR full OCR
- **ตัวเลือก:**
  - `Fast (raw)`: รับผล OCR เร็วที่สุด กรองน้อย
  - `Balanced`: ค่า default กรองด้วย quality/confidence และ heuristic พื้นฐาน
  - `Strict`: ใช้ mask จาก preprocessing ช่วยยืนยันว่าผล OCR อยู่บนบริเวณที่เหมือนข้อความจริง
- **ผลต่อระบบ:** Fast เร็วแต่ noise มากกว่า Strict สะอาดกว่าแต่หนักกว่า
- **ใช้เมื่อ:** ถ้า demo ต้องการความเร็ว ใช้ Fast/Balanced ถ้าจอมี false positive เยอะ ใช้ Strict

### 17. OCR device

- **หน้าที่:** เลือก device preference สำหรับ OCR/deep detector backend ที่รองรับ
- **ตัวเลือก:** `Auto`, `CPU`, `GPU (NVIDIA CUDA)`
- **ผลต่อระบบ:** GPU ช่วยเร่ง EasyOCR/RapidOCR/ONNX Runtime เมื่อ environment พร้อม
- **ข้อควรระวัง:** ถ้า CUDA/ONNX Runtime provider ไม่พร้อม บาง backend จะ fallback หรือรายงานสถานะใน OCR runtime

### 18. Enable OCR

- **หน้าที่:** เปิดหรือปิด OCR
- **ค่า default:** เปิด
- **ผลต่อระบบ:** ถ้าปิด OCR ระบบยังทำ detection ได้ แต่ไม่มีข้อความอ่านจริงหรือคำแปลจาก OCR
- **ใช้พูดตอนนำเสนอ:** แยกชัดเจนระหว่าง stage detection กับ stage OCR ทำให้ทดสอบ segmentation/detection ได้โดยไม่ต้องใช้ OCR backend

### 19. Track overlay while scrolling

- **หน้าที่:** เปิดการ track กล่อง overlay ให้ตามเนื้อหาขณะ scroll หรือมี motion
- **ค่า default:** ปิด
- **ผลต่อระบบ:** เมื่อเปิด ระบบจะส่ง motion offset/confidence ไปให้ overlay และเก็บ recent translation มากขึ้นเพื่อช่วยตามตำแหน่ง
- **ใช้เมื่อ:** แปลหน้าเว็บ, subtitle, chat หรือ content ที่เลื่อน/ขยับระหว่าง frame

### 20. Runtime debug timings

- **หน้าที่:** แสดงเวลาที่ใช้ในแต่ละ stage ของ pipeline
- **ค่า default:** ปิด
- **ผลต่อระบบ:** เมื่อเปิด UI จะแสดง breakdown เช่น scale, OCR, translation, draw preview, metadata
- **ใช้พูดตอนนำเสนอ:** เหมาะสำหรับโชว์ว่า bottleneck อยู่ตรงไหน เช่น OCR หรือ full-frame OCR ใช้เวลามากสุด

### 21. Previews

- **หน้าที่:** เลือก preview ที่ต้องการแสดงใน UI
- **ตัวเลือก:**
  - `Annotated`: ภาพจริงพร้อมกรอบ detection/OCR
  - `Segmentation`: mask/segmentation preview
  - `Translated`: preview ที่วาดคำแปลลงตำแหน่งกล่อง
- **ผลต่อระบบ:** ปิด preview ที่ไม่ใช้ช่วยลดงานวาดภาพใน UI และทำให้ demo โฟกัสเฉพาะ output ที่ต้องการ
- **ใช้พูดตอนนำเสนอ:** Annotated แสดงผล detection, Segmentation แสดง image processing, Translated แสดงผลปลายทาง

### 22. Overlay tracking

- **หน้าที่:** เลือก algorithm สำหรับช่วยให้ overlay ตามตำแหน่งข้อความ
- **ตัวเลือก:**
  - `Legacy motion`: ใช้ motion offset และ local template tracking แบบเดิม
  - `Visual anchor lock`: ใช้ visual anchor template matching จาก frame tracking worker
- **ผลต่อระบบ:** ตัวเลือกนี้มีผลเมื่อเปิด `Track overlay while scrolling`
- **ใช้พูดตอนนำเสนอ:** เป็นส่วนที่ทำให้ overlay ไม่ใช่กล่องคงที่ แต่พยายามตามเนื้อหาบนจอจริง

## Runtime Stats ที่ใช้ประกอบการ Demo

ส่วนนี้ไม่ใช่ Pipeline Settings แต่ควรใช้ประกอบการอธิบายผลขณะ demo:

| Runtime Stats | ความหมาย |
| --- | --- |
| `FPS` | ความเร็วของ processing loop หลังผ่าน pipeline จริง |
| `Active boxes` | จำนวนกล่องข้อความที่ detect/OCR/translate ได้ใน frame ล่าสุด |
| `Monitor` | จอที่กำลัง capture |
| `Status` | สรุป route ภาษา, detector, OCR backend และ translation backend |
| `Recording` | path ของ recording session หรือสถานะปิด |
| `OCR runtime` | รายละเอียด backend เช่น RapidOCR / EasyOCR / Tesseract และ provider CPU/GPU |
| `Pipeline debug` | timing ของ stage ต่าง ๆ เมื่อเปิด Runtime debug timings |

## ประโยคสั้นสำหรับใช้พูดในสไลด์

- Pipeline Settings คือจุดที่ผู้ใช้ปรับ trade-off ระหว่าง **ความเร็ว**, **ความแม่นยำ**, และ **ความสวยของ overlay**
- ระบบออกแบบให้ backend สลับได้ ไม่ผูกกับ OCR หรือ detector ตัวเดียว
- Latency ถูกควบคุมด้วย capture interval, detector scale, OCR limit, cache และ frame dropping
- Translation ไม่ได้ส่งแปลทุก frame แต่มี reuse/cache และ text similarity stability เพื่อลดงานซ้ำ
- Overlay มีทั้งโหมดง่ายแบบ bubble และโหมด clean patch สำหรับ subtitle ที่ต้องการผลลัพธ์เนียนขึ้น
