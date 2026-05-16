# ScreenLens Pipeline Mermaid Diagrams

เอกสารนี้สรุป pipeline ปัจจุบันของ ScreenLens-Detection จากโค้ดบน branch `feature-optimization-efficency-4` ที่ HEAD `5f1cf9c` พร้อม diagram สำหรับใช้ benchmark FPS/latency และอธิบาย path สำคัญของ OCR, translation, overlay, recording และ build/runtime provider

## Git Reference

| Label | Branch / ref | Commit | ใช้เทียบอะไร |
| --- | --- | --- | --- |
| Current feature-4 | `feature-optimization-efficency-4` | `5f1cf9c` | setup/build แยก RapidOCR กับ ONNX Runtime provider และตรวจ diagnostics |
| Overlay bubble tuning | `feature-optimization-efficency-4` | `175fbd2`, `2650e9a` | compact/expanded bubble และ font fitting ใน overlay |
| Full-frame validation | `feature-optimization-efficency-4` | `3e2dc96` | validation mode `fast` / `balanced` / `strict` |
| Full-frame output stability | `feature-optimization-efficency-4` | `e0ec945` | output limit และ stability สำหรับ RapidOCR full-frame OCR |
| Main optimized merge | `origin/main` | `852565d` | merge optimization ชุดก่อนหน้าเข้า main |
| Feature 3 baseline | `origin/feature-optimization-efficiency3` | `35737c7` | hover/full-frame OCR และ subtitle block logic ก่อน feature-4 |
| V2.1 base | `ver2.1` | `ebbe24a` | จุดก่อน full-frame OCR tuning ชุดใหญ่ |
| V2.0 base | `ver2.0` | `fc03faa` | จุดก่อน queued OCR / scanline / hover optimization |

ตัวอย่าง checkout สำหรับ benchmark:

```powershell
git checkout 5f1cf9c
git checkout 3e2dc96
git checkout e0ec945
git checkout 852565d
git checkout fc03faa
```

## 1. Runtime Pipeline ปัจจุบัน

ไฟล์หลัก:

- `src/screenlens_detection/worker.py`
- `src/screenlens_detection/pipeline.py`
- `src/screenlens_detection/ocr.py`
- `src/screenlens_detection/translation.py`
- `src/screenlens_detection/overlay.py`
- `src/screenlens_detection/recording.py`

```mermaid
flowchart TD
    A["Input: selected monitor"] --> B["ScreenCapturer.grab()"]
    B --> C["_LatestFrameQueue maxsize=1"]
    C --> C1{"Queue full?"}
    C1 -->|Yes| C2["Drop old frame, keep newest"]
    C1 -->|No| D["ProcessingWorker.run()"]
    C2 --> D

    D --> E["TextDetectionPipeline.process(frame)"]
    E --> F["Scale frame by effective detection_scale/upscale_factor"]
    F --> G["Enhance grayscale"]
    G --> H{"Choose detection/OCR path"}

    H -->|Full OCR + hover| I["_annotate_hover_with_full_frame_ocr()"]
    H -->|Full OCR + full screen| J["_annotate_with_full_frame_ocr()"]
    H -->|Hover + crop OCR| K["_hover_detection_pass()"]
    H -->|Scanline ROI| L["_scanline_detection_pass()"]
    H -->|Default| M["_text_detection_pass()"]

    I --> N["Filter, merge, validate, stabilize OCRFrameResult"]
    J --> N
    K --> O["Source boxes"]
    L --> O
    M --> O
    O --> P["_stabilize_ocr_boxes()"]
    P --> Q["_filter_motion_ocr_boxes()"]
    Q --> R["_annotate_with_ocr() via QueuedOCRBackend"]
    R --> S["DetectionBox list with OCR text"]
    N --> S

    S --> T["_estimate_frame_offset()"]
    T --> U["_apply_translations()"]
    U --> V["_remember_ocr_translations()"]
    V --> W["Build FrameAnalysis"]

    W --> X["MainWindow previews and text panel"]
    W --> Y["TranslationOverlay"]
    W --> Z["RecordingSession"]

    X --> X1["Annotated / segmentation / translated preview"]
    Y --> Y1["Bubble or clean patch overlay"]
    Z --> Z1["MP4 streams + session_log.jsonl"]
```

Output หลักคือ `FrameAnalysis` ซึ่งมี `boxes`, `source_frame`, `annotated_frame`, `processed_preview`, `translated_preview`, `fps`, `ocr_runtime`, `content_offset_*` และ `runtime_timings_ms`

## 2. Detection Decision Tree

```mermaid
flowchart TD
    A["Pipeline.process(frame)"] --> B["detection_frame + detection_gray"]
    B --> C{"ocr_enabled and backend.supports_full_frame()?"}

    C -->|Yes| D{"translation_region_mode == hover?"}
    D -->|Yes| E["Path A: Hover RapidOCR full-frame OCR on ROI"]
    D -->|No| F["Path B: RapidOCR full-frame OCR on detection frame"]

    C -->|No| G{"translation_region_mode == hover?"}
    G -->|Yes| H["Path C: Hover ROI detector + crop OCR"]
    G -->|No| I{"scanline_roi_enabled?"}
    I -->|Yes| J["Path D: Scanline vertical band detector + crop OCR"]
    I -->|No| K{"text_detector_mode"}

    K -->|opencv| L["Path E1: OpenCV text mask + line mask + components"]
    K -->|rapidocr/paddleocr/easyocr| M["Path E2: Deep detector backend"]

    E --> N["Full-frame result pipeline"]
    F --> N
    H --> O["Crop result pipeline"]
    J --> O
    L --> O
    M --> O

    N --> P["DetectionBox list"]
    O --> P
    P --> Q["Translation and FrameAnalysis"]
```

Benchmark hint:

- Path A/B วัด RapidOCR full-frame OCR และ validation mode
- Path C วัด hover ROI ที่ลดพื้นที่ detector/OCR
- Path D วัด scanline ROI ที่แบ่งงาน detector ข้ามหลาย frame
- Path E1/E2 วัด OpenCV detector เทียบ deep detector

## 3. Full-frame OCR Validation

ใช้เมื่อ OCR backend เป็น `RapidOCRFullBackend` หรือ backend อื่นที่ `supports_full_frame()` เป็น `True`

```mermaid
flowchart TD
    A["ocr_backend.recognize_frame(frame_or_roi)"] --> B["OCRFrameResult rect/text/confidence"]
    B --> C["_filter_full_frame_ocr_results()"]
    C --> D["_limit_full_frame_ocr_results()"]
    D --> E["Normalize and _is_usable_text()"]
    E --> F["_merge_full_frame_line_results()"]
    F --> G{"full_frame_ocr_validation_mode"}

    G -->|fast| H["Return raw filtered/merged results"]
    G -->|balanced| I["Reject UI noise, short text and low confidence"]
    G -->|strict| J["Build OpenCV text_mask + line_mask"]
    J --> K["_full_frame_mask_support_score()"]
    K --> L["Keep only boxes with mask support"]

    H --> M["_stabilize_full_frame_ocr_results()"]
    I --> M
    L --> M
    M --> N["Track evidence, bbox smoothing, text quality preference"]
    N --> O["_select_full_frame_output_results()"]
    O --> P["DetectionBox output"]
```

Mode summary:

- `fast`: เหมาะกับวัด raw recall/latency เพราะไม่ใช้ validation หนัก
- `balanced`: default, ลด false positive จาก UI noise โดยไม่สร้าง mask เพิ่ม
- `strict`: เพิ่ม OpenCV validation mask เหมาะกับลด OCR หลุดจากภาพที่มี UI noise เยอะ

## 4. Crop OCR Queue และ Cache

```mermaid
flowchart TD
    A["Detected source boxes"] --> B["_prepare_ocr_candidates()"]
    B --> C["Crop high-res OCR grayscale"]
    C --> D["Create crop fingerprint"]
    D --> E{"Pipeline recent OCR cache hit?"}

    E -->|Hit| F["Reuse text/confidence/translation metadata"]
    E -->|Miss| G["_select_pending_ocr_candidates()"]
    G --> H["Limit by max_ocr_boxes_per_frame and priority"]
    H --> I["ocr_backend.prepare_image(crop)"]
    I --> J["QueuedOCRBackend.recognize_batch()"]

    J --> K{"Backend digest cache hit?"}
    K -->|Hit| L["Return cached OCRResult"]
    K -->|Miss + sync budget| M["Run OCR synchronously"]
    K -->|Miss + no sync budget| N["Append to OCR worker queue"]

    N --> O["OCR worker pops language-compatible batch"]
    O --> P["Underlying OCR backend recognize_batch()"]
    M --> P
    P --> Q["Store digest cache"]
    Q --> R["Normalize and filter OCR text"]
    F --> R
    R --> S["DetectionBox with OCR text"]
```

Crop OCR path ใช้กับ EasyOCR/Tesseract และ deep detector ที่คืนเฉพาะ boxes ส่วน RapidOCR full-frame OCR จะข้าม path นี้

## 5. Translation Pipeline

```mermaid
flowchart TD
    A["DetectionBox list with OCR text"] --> B["_apply_translations()"]
    B --> C{"Any boxes?"}
    C -->|No| D["Keep recent translations briefly, then clear if no overlay tracking"]
    C -->|Yes| E{"translation_region_mode == hover?"}

    E -->|Yes| F["_filter_hover_metadata_rows()"]
    F --> G["_combine_hover_subtitle_boxes()"]
    E -->|No| H["Use boxes as-is"]

    G --> I["_reuse_recent_translations()"]
    H --> I
    I --> J{"translation_block_mode == strict?"}
    J -->|Yes| K["_apply_strict_block_translations()"]
    J -->|No| L["_apply_line_translations()"]
    K --> L

    L --> M["translation_backend.translate_batch()"]
    M --> N["QueuedTranslationBackend"]
    N --> O{"Cache hit?"}
    O -->|Hit| P["Return cached translated_text"]
    O -->|Miss + sync budget| Q["Translate immediately"]
    O -->|Miss| R["Queue by source/target route"]
    R --> S["Worker batches same route"]
    S --> T["Argos offline or Google online"]
    Q --> T
    T --> U["Store translation cache"]
    P --> V["DetectionBox.translated_text"]
    U --> V
    V --> W["_remember_translations()"]
```

Strict block mode รวมกล่องที่ดูเป็น paragraph/subtitle หลายบรรทัดเป็น block เดียวก่อนแปล แล้วข้าม line translation ซ้ำของ member lines

## 6. Overlay, Clean Patch และ Tracking

```mermaid
flowchart TD
    A["FrameAnalysis"] --> B["TranslationOverlay.update_analysis()"]
    B --> C["Convert DetectionBox to OverlayBox"]
    C --> D{"overlay_tracking_enabled?"}

    D -->|No| E["OverlayTrackManager.replace_with_observations()"]
    D -->|Yes| F["Predict boxes from pipeline motion or realtime tracking"]
    F --> G["OverlayTrackManager.update_from_pipeline()"]
    G --> H{"tracking_mode"}

    H -->|legacy| I["Motion offset + local template tracking"]
    H -->|anchor| J["Visual anchor template matching"]

    E --> K{"subtitle_render_mode"}
    I --> K
    J --> K

    K -->|bubble| L["_expanded_bubble_rect() + _font_for_text()"]
    K -->|clean_patch| M["clean_patch_for_box() from source frame"]
    M --> N["OpenCV mask/inpaint or soft background patch"]
    L --> O["Paint overlay"]
    N --> O
```

เมื่อเปิด visual tracking, `OverlayTrackingWorker` จะ capture grayscale frame ที่ลด resolution แล้วส่ง `TrackingFrame` ให้ overlay เพื่อ track กล่องระหว่าง frame หลัก

## 7. Recording และ Runtime Debug

```mermaid
flowchart TD
    A["MainWindow._handle_frame(analysis)"] --> B["Update previews"]
    A --> C{"Recording active?"}
    C -->|No| D["Skip disk writes"]
    C -->|Yes| E["RecordingSession.write_frame()"]
    E --> F["annotated_preview.mp4"]
    E --> G["segmentation_preview.mp4"]
    E --> H["translated_preview.mp4"]
    E --> I["session_log.jsonl"]
    I --> J["fps, status, ocr_runtime, timings, motion, boxes"]

    A --> K{"runtime_debug_enabled?"}
    K -->|Yes| L["Format slowest stage from runtime_timings_ms"]
    K -->|No| M["Runtime debug Off"]
```

Fields ที่ควรดูใน log:

- `fps`
- `runtime_timings_ms.total`
- `runtime_timings_ms.scale_frame`
- `runtime_timings_ms.opencv_detection` หรือ `runtime_timings_ms.deep_detection`
- `runtime_timings_ms.hover_detection`
- `runtime_timings_ms.scanline_detection`
- `runtime_timings_ms.ocr_annotation`
- `runtime_timings_ms.full_frame_ocr`
- `runtime_timings_ms.hover_full_frame_ocr`
- `runtime_timings_ms.translation`
- `detected_boxes`
- `content_motion_confidence`

## 8. Build / Setup Provider Flow

```mermaid
flowchart TD
    A["build_screenlens_exe.bat"] --> B["scripts/build_windows.ps1 -Clean"]
    B --> C["Resolve or create .venv"]
    C --> D["scripts/setup_windows.ps1"]
    D --> E["Install project dependencies"]
    E --> F["Install EasyOCR"]
    F --> G["Install rapidocr>=3.0.0"]
    G --> H{"TorchRuntime"}
    H -->|cpu| I["Uninstall onnxruntime-gpu"]
    H -->|gpu| J["Uninstall onnxruntime"]
    I --> K["Force reinstall onnxruntime CPU"]
    J --> L["Force reinstall onnxruntime-gpu"]
    K --> M["Install Torch CPU"]
    L --> N["Install Torch CUDA"]
    M --> O["Install PaddleOCR"]
    N --> O
    O --> P["Sync Tesseract vendor"]
    P --> Q["Download Argos en-th/th-en models"]
    Q --> R["Torch diagnostics"]
    R --> S["ONNX Runtime diagnostics"]
    S --> T["PyInstaller onedir build"]
```

จุดสำคัญของ `5f1cf9c`: RapidOCR ถูกติดตั้งก่อน แล้วจึง force reinstall ONNX Runtime provider ที่เลือก เพื่อกัน dependency resolver ดึง provider ผิดตัวกลับมา

## 9. Optimization Evolution Timeline

```mermaid
flowchart TD
    A["4282d9e<br/>OCR + translation + overlay support"] --> B["2c9dc68<br/>recent translation reuse"]
    B --> C["3e3ac1e<br/>selectable text detectors"]
    C --> D["fc03faa<br/>tag ver2.0"]

    D --> E["2351918<br/>queued OCR + OCR cache"]
    E --> F["590419e<br/>RapidOCR detector + queued translation"]
    F --> G["846290e<br/>OCR workers + Tesseract batch"]
    G --> H["95e538a<br/>detection scale + motion cache"]
    H --> I["637325f<br/>scanline ROI"]
    I --> J["a2c4935<br/>hover ROI"]
    J --> K["ebbe24a<br/>tag ver2.1"]

    K --> L["28b5202<br/>full-frame OCR tracking"]
    L --> M["db63a8a<br/>hover full-frame OCR"]
    M --> N["6768a53<br/>multiline subtitle filtering"]
    N --> O["35737c7<br/>longer blocks + wider ROI"]
    O --> P["852565d<br/>main optimized merge"]
    P --> Q["889f7c<br/>drop old capture frames"]
    Q --> R["e0ec945<br/>full-frame output stability"]
    R --> S["3e2dc96<br/>full-frame validation modes"]
    S --> T["175fbd2<br/>compact overlay bubble"]
    T --> U["2650e9a<br/>bubble expansion tuning"]
    U --> V["5f1cf9c<br/>RapidOCR/ONNX setup flow"]
```

## 10. Suggested Benchmark Matrix

| Benchmark | Checkout commit | Focus |
| --- | --- | --- |
| Baseline V2.0 | `fc03faa` | ก่อน queue/cache/scanline/hover |
| Async OCR V1 | `2351918` | queued OCR + OCR cache |
| RapidOCR + translation backend | `590419e` | RapidOCR detector, Argos/Google queued translation |
| OCR workers | `846290e` | batch + worker count |
| Detection scale/cache | `95e538a` | FPS เมื่อปรับ detection scale |
| Scanline ROI | `637325f` | vertical band slicing |
| Hover ROI | `a2c4935` | OCR เฉพาะ cursor region |
| Full-frame OCR | `28b5202` | RapidOCR full-frame det+rec |
| Hover full-frame OCR | `db63a8a` | ROI + full-frame backend |
| Feature 3 optimized | `35737c7` | subtitle/hover block logic |
| Main optimized merge | `852565d` | หลัง merge เข้า main |
| Full-frame stability | `e0ec945` | output limit/stability |
| Validation modes | `3e2dc96` | `fast` vs `balanced` vs `strict` |
| Overlay compact bubble | `175fbd2` | compact text bubble behavior |
| Overlay expanded bubble | `2650e9a` | long translated text fitting |
| Current setup/build | `5f1cf9c` | ONNX Runtime CPU/GPU provider install flow |

สำหรับ realtime pipeline ควรเทียบ median frame time และ percentile spike คู่กับ average FPS เพราะ OCR/translation queue ทำให้บาง frame เร็วมาก แต่บาง frame spike ตอน backend ทำงานจริง
