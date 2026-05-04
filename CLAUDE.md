# IC Frame Laser-Mark Inspection System
### Note: This project puase, unknow duration

## Overview

An automated visual inspection system for IC (Integrated Circuit) frames with laser-marked characters. The system captures images via a Basler GigE camera, detects mold positions, verifies character integrity using contour analysis and HOG-based OCR, and outputs pass/fail signals via Raspberry Pi GPIO.

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Camera | Basler acA1300-60gc GigE, 1280×1024, link-local |
| Controller | Raspberry Pi (GPIO-capable) |
| GPIO Pin 3 | Inspect trigger (active-LOW input) |
| GPIO Pin 5 | Busy output |
| GPIO Pin 7 | Pass/Fail output |

---

## Software Stack

| Layer | Library |
|---|---|
| Language | Python 3 |
| Computer Vision | OpenCV |
| GUI | PyQt5 |
| Camera SDK | pypylon (Basler) |
| AI Inference | OpenVINO |
| GPIO | RPi.GPIO |

---

## Project File Layout

| Path | Purpose |
|---|---|
| `insp_exp.py` | Main application source (single-file architecture) |
| `Setup.json` | All settings — static thresholds + user-tunable params |
| `pin_recipe.json` | Frame/mold layout + slot shift references (v6) |
| `frame_layout.json` | Saved frame layout from FrameLayoutPanel |
| `templates/` | Per-character JSON inspection templates |
| `training_data/` | HOG OCR training crops (`training_data/<CHAR>/`) |
| `Mold_detector_openvino_model/` | OpenVINO IR model for mold bounding box detection |
| `image_source/` | DEBUG_MODE input images (BMP/JPG/PNG) |
| `Inspection_result/` | Failed image output archive |
| `debug/` | Debug PNGs from inspection pipeline steps |
| `search_mask.jpg` | Optional camera search mask |

---

## Operating Modes

```python
DEBUG_MODE = True   # Loops over image_source/ (BMP/JPG/PNG) — no camera required
DEBUG_MODE = False  # Live Basler camera — GrabStrategy_LatestImageOnly
```

Set in the CONFIG block at the top of `insp_exp.py`.

**Output files (on failure):**
- `Inspection_result/<timestamp>_R.png` — raw grayscale frame
- `Inspection_result/<timestamp>.png` — annotated BGR frame

---

## Configuration (`Setup.json`)

Two top-level sections:

| Section | Purpose | Restart Required |
|---|---|---|
| `["static"]` | Threshold constants — override code defaults | Yes |
| `["setup"]` | User-tunable values: exposure, pin score, grid params, grid letters | No |

Legacy `inspection_settings.txt` is automatically migrated on first run.

---

## Module Architecture

The entire system is implemented in `insp_exp.py` as a single file, ordered top to bottom:

```
InspectionResult        @dataclass — inspection result carrier
CONFIG block            Constants: DEBUG_MODE, PIN_*, FONT_*, OCR_*, CAMERA_*
_compute_hog            Module-level HOG helper (1764-dim L2-normalised vector)
SettingsManager         Reads/writes Setup.json (static + setup sections)
ImageIO                 Image save/load helpers
BaslerCamera            Basler GigE camera wrapper
MachineIO               GPIO trigger/busy/pass-fail output controller
ContourTemplate         JSON template read/write (contour + base64 PNG canvas)
cv2_draw_dashed_rect    Drawing utility
YOLOMoldDetector        OpenVINO IR inference — mold bounding box detection
InspectionEngine        Core CV logic (static methods only)
ResultAnnotator         Draws boxes/labels on display image (no Qt dependency)
InspectionController    Owns engine + template store + camera; run() entry point
RunWorker               QThread wrapper around InspectionController.run()
ImageView               PyQt5 QLabel subclass with zoom/pan
FrameTemplatePanel      Frame rect draw UI (Step 1 of template wizard)
FrameLayoutPanel        Mold A/B rect draw UI (Steps 2–3 of template wizard)
TemplatePreviewDialog   Modal for font template save wizard (3-step)
RightPanel              Settings + controls panel
MainWindow              Top-level window; wires all components
main()                  QApplication entry point
```

---

## Inspection Pipeline

The core inspection runs in `compare_roi` across 5 sequential steps:

| Step | Name | Logic |
|---|---|---|
| 1 | Presence check | Contours must exist in ROI |
| 2 | Shift detection | `shift_px / tmpl_diagonal ≤ FONT_SHIFT_RATIO_MAX (0.50)` |
| 3 | Hole/stroke integrity | Hole count + enclosed area ratio vs. template |
| 4 | Canvas IoU similarity | 64×64 centre-aligned canvas; weighted score: `similarity×0.70 + hole_score×0.20 + aspect_score×0.10` |
| 5 | Aspect ratio check | Bounding box ratio vs. template ± `FONT_ASPECT_TOLERANCE` |

---

## OCR System — HOG Cosine Similarity

Characters are read using HOG feature vectors compared via cosine similarity.

**Method:** `_compute_hog` → 1764-dim L2-normalised HOG vector → cosine similarity against stored template vectors

**Confidence thresholds:**

```python
OCR_CONF_EXPECTED = 0.88   # Fast-path return if confidence ≥ this
OCR_MIN_CONF      = 0.60   # Below this → report "?" (unreadable)
OCR_CONF_GAP_MIN  = 0.10   # Best must beat 2nd-best by this margin (prevents 0/O/8/2 confusion)
```

**Output fields:** `ocr_char`, `ocr_conf`, `ocr_string` (9-character mold ID)

**Training data:** 48×48 PNG crops stored in `training_data/<CHAR>/`

---

## Mold Detection (YOLO)

**Class:** `YOLOMoldDetector`

- Model: `Mold_detector_openvino_model/Mold_detector.xml` (YOLOv8, single class "IC")
- If model is missing or OpenVINO is unavailable: `is_ready()` returns `False`, detection is silently skipped
- Used by `InspectionController` to locate mold bounding boxes before slot inspection

---

## Pin Presence Gate

**Method:** `InspectionEngine.check_pin_presence()`

Detects IC lead/pin presence using Sobel-Y edge analysis within the lead ROI.

| Parameter | Value | Meaning |
|---|---|---|
| `PIN_SOBEL_MAG` | 40 | Edge magnitude threshold |
| `PIN_EDGE_RATIO` | 0.150 | Minimum edge-pixel fraction required |

No-lead molds receive automatic PASS (logged in gray), with no alarm raised.

---

## Template System

### Font Templates

Stored as `templates/<char>_<slot>.json` with the following fields:

| Field | Purpose |
|---|---|
| `contours` | Serialised contour points |
| `canvas_b64` | Base64-encoded 64×64 PNG canvas |
| `mold_size` | Mold dimensions (used for morphology kernel sizing) |
| `canvas_w/h` | Canvas dimensions |
| `tmpl_diagonal` | Reference diagonal for shift ratio |
| `tmpl_contour_count` | Expected number of contours |
| `tmpl_aspect` | Expected bounding box aspect ratio |
| `tmpl_bbox` | Reference bounding box |
| `shift_ref` | Shift reference point |
| `hog_vec` | HOG feature vector for OCR |

### Layout Files

- `pin_recipe.json` (v6) — frame/mold layout + slot shift references
- `frame_layout.json` — saved output from FrameLayoutPanel

---

## Template Setup Wizard (3-Step)

Run from the GUI to define inspection regions for a new IC type:

1. **Step 1 — `FrameTemplatePanel`:** Draw the outer frame bounding rectangle
2. **Step 2 — `FrameLayoutPanel`:** Draw Mold A bounding rectangle
3. **Step 3 — `FrameLayoutPanel`:** Draw Mold B bounding rectangle

The system auto-computes 9 slot positions at `mold_size / 3` pitch.
A constraint zone (1.3× frame-height square + `x_offset`) prevents molds from being placed on the wrong frame.

---

## IC / Frame Slot Numbering

- Ordering: column-first, Y-axis-first (9 slots per mold)
- `COL_SNAP = canvas_w // 2` — splits frame into left (Mold A) and right (Mold B) halves
- Mold A and Mold B at the same frame position receive sequential IC numbers
- Frame sort is applied before numbering

---

## Design Constraints

| Rule | Detail |
|---|---|
| No PCA rotation | Fixed camera mount — PCA rotation removed entirely; re-save all templates if the pipeline changes |
| CHAIN_APPROX_NONE | Never use `TC89_KCOS` — it collapses diagonal strokes (e.g., "7") |
| Canvas IoU only | 64×64 centre-aligned canvas IoU is the similarity metric; Hu moments are not used |
| Otsu inversion check | Bright-on-dark marks: tophat + Otsu + invert at template save; adaptive + unsharp at runtime |
| Save ↔ compare parity | Any change to extraction logic requires full template re-save |
| Mold-level kernel sizing | Morphology kernels are sized to mold dims (~150–160 px), not font ROI dims (~40–50 px) |
| Mold-level rotation | All letters rotate as one laser unit; per-letter rotation checks are incorrect |

---

## Debug Image Naming Convention

Intermediate debug PNGs written to `debug/` follow this naming scheme:

| Suffix | Stage |
|---|---|
| `_0_gray` | Grayscale input |
| `_1_thresh` | Thresholded binary |
| `_2_raw_contours` | All detected contours |
| `_3_filtered_contours` | Contours after size/area filtering |
| `_4_canvas` | Final 64×64 canvas |

---

## Known Issues

| Issue | Status |
|---|---|
| Large shifts cause false-empty detection in Step 2 | Open |
| Rotation detection via PCA on combined mold contours | Design agreed — not yet implemented |
| `_step1_find_frames` multi-scale template matching ~729 ms bottleneck | Open — fixed ROI or reduced scales proposed |
| Threshold tuning (pin presence, rotation angle) | Pending validation on real production images |
