# IC Frame Laser-Mark Inspection — Project Instructions

## Working Directory Layout

| Path | Purpose |
|---|---|
| `insp_exp.py` | **Main file to edit** |
| `Setup.json` | All settings (static thresholds + user-tunable); replaces legacy `inspection_settings.txt` |
| `pin_recipe.json` | Frame/mold layout + slot shift refs (v6) |
| `frame_layout.json` | Saved frame layout from FrameLayoutPanel |
| `templates/` | Per-character JSON templates |
| `training_data/` | HOG OCR training crops (`training_data/<CHAR>/`) |
| `Mold_detector_openvino_model/` | OpenVINO IR model for mold bbox detection |
| `image_source/` | DEBUG_MODE input images (BMP/JPG/PNG) |
| `Inspection_result/` | Failed image output |
| `debug/` | Debug PNGs from inspection steps |
| `search_mask.jpg` | Optional camera search mask |

---

## Project Identity

**Stack:** Python · OpenCV · PyQt5 · pypylon (Basler SDK) · OpenVINO · RPi.GPIO  
**Hardware:** Basler acA1300-60gc GigE, 1280×1024, link-local · Raspberry Pi GPIO

---

## Module Layout (top → bottom in file)

```
InspectionResult        @dataclass — result carrier
CONFIG block            constants (DEBUG_MODE, PIN_*, FONT_*, OCR_*, CAMERA_*)
_compute_hog            module-level HOG helper (1764-dim L2-normalised vector)
SettingsManager         persists Setup.json (static + setup sections)
ImageIO                 save/load helpers
BaslerCamera            Basler GigE camera wrapper
MachineIO               GPIO trigger/busy/pass-fail output
ContourTemplate         JSON template r/w (contour + base64 PNG canvas)
cv2_draw_dashed_rect    drawing utility
YOLOMoldDetector        OpenVINO IR inference — mold bbox detection
InspectionEngine        core CV logic (static methods only)
ResultAnnotator         draws boxes/labels on display image (no Qt)
InspectionController    owns engine + template store + camera; run() entry point
RunWorker               QThread wrapper around InspectionController.run()
ImageView               PyQt5 QLabel subclass with zoom/pan
FrameTemplatePanel      frame rect draw UI (step 1 of template wizard)
FrameLayoutPanel        mold A/B rect draw UI (steps 2–3 of template wizard)
TemplatePreviewDialog   modal for font template save wizard (3-step)
RightPanel              settings + controls panel
MainWindow              top-level window; wires everything
main()                  QApplication entry
```

---

## Critical Design Rules

| Rule | Detail |
|---|---|
| **No PCA rotation** | Fixed camera — removed entirely; re-save all templates if pipeline changes |
| **CHAIN_APPROX_NONE** | Never use TC89_KCOS — collapses diagonal strokes ("7") |
| **Canvas IoU, not Hu moments** | 64×64 centre-aligned canvas IoU is the similarity metric |
| **Otsu inversion check** | Bright-on-dark: tophat+Otsu+invert at template save; adaptive+unsharp at runtime |
| **Save ↔ compare must match** | Any extraction logic change requires full template re-save |
| **mold_size as kernel anchor** | Morph kernels sized to mold dims (~150–160 px), not font ROI (~40–50 px) |
| **Rotation is mold-level** | All letters rotate as one laser unit; per-letter rotation check is wrong |

---

## Inspection Pipeline (`compare_roi` — 5 steps)

```
Step 1  Presence check          contours must exist
Step 2  Shift detection         shift_px / tmpl_diagonal ≤ FONT_SHIFT_RATIO_MAX (0.50)
Step 3  Hole/stroke integrity   hole count + area ratio
Step 4  Canvas IoU similarity   64×64 centre-aligned; weighted score:
                                  similarity×0.70 + hole_score×0.20 + aspect_score×0.10
Step 5  Aspect ratio check      bbox ratio vs template ± FONT_ASPECT_TOLERANCE
```

---

## OCR — HOG Cosine Similarity (implemented)

**Method:** `_compute_hog` → cosine similarity against stored template HOG vectors  
**Constants:**
```python
OCR_CONF_EXPECTED = 0.88   # fast-path return if ≥ this
OCR_MIN_CONF      = 0.60   # below → report "?" (unreadable)
OCR_CONF_GAP_MIN  = 0.10   # best must beat 2nd-best by this (avoids 0/O/8/2 ties)
```
**Result fields:** `ocr_char`, `ocr_conf`, `ocr_string` (9-char mold ID)  
**Training data:** 48×48 PNGs in `training_data/<CHAR>/`

---

## Pin Presence Gate

**Method:** `InspectionEngine.check_pin_presence()`  
- Sobel-Y edge magnitude threshold: `PIN_SOBEL_MAG = 40`  
- Min edge-pixel fraction in lead ROI: `PIN_EDGE_RATIO = 0.150`  
- No-lead molds → PASS, skipped silently (logged gray), no alarm

---

## YOLO Mold Detection

**Class:** `YOLOMoldDetector`  
- Wraps `Mold_detector_openvino_model/Mold_detector.xml` (YOLO8, single class "IC")  
- Fallback: if model missing or OpenVINO unavailable → `is_ready()=False`, skipped silently  
- Used by `InspectionController` for mold bbox detection

---

## IC / Frame Numbering

- Column-first, Y-axis-first slot ordering (9 slots per mold)
- `COL_SNAP = canvas_w // 2` — frame left/right split
- Mold A + Mold B at same frame position → sequential IC numbers
- Frame sort applied before numbering

---

## Template Storage

**Font templates:** `templates/<char>_<slot>.json`  
**PIN/layout:** `pin_recipe.json` (v6) + `frame_layout.json`

JSON fields: `contours`, `canvas_b64`, `mold_size`, `canvas_w/h`, `tmpl_diagonal`, `tmpl_contour_count`, `tmpl_aspect`, `tmpl_bbox`, `shift_ref`, `hog_vec` (OCR)

---

## Settings (`Setup.json`)

Two sections:
- `["static"]` — threshold constants (override code defaults; require restart)
- `["setup"]` — user-tunable values (exposure, pin score, grid params, grid letters)

Legacy `inspection_settings.txt` is auto-migrated on first run.

---

## Operating Modes

```python
DEBUG_MODE = True   # loops image_source/ (BMP/JPG/PNG)
DEBUG_MODE = False  # live Basler — GrabStrategy_LatestImageOnly
```

**Output:** Failed images → `Inspection_result/`  
- `<timestamp>_R.png` raw grayscale · `<timestamp>.png` annotated BGR

**GPIO pins:** 3 = inspect trigger (active-LOW) · 5 = busy · 7 = pass/fail

---

## Template Wizard (3-step)

1. `FrameTemplatePanel` — draw frame rect
2. `FrameLayoutPanel` — draw Mold A rect
3. `FrameLayoutPanel` — draw Mold B rect

Auto-computes 9 slot positions at `mold_size/3` pitch.  
Constraint zone: 1.3× frame-height square + `x_offset` prevents wrong-frame mold placement.

---

## Known Open Issues

| Issue | Status |
|---|---|
| Shift detection: large shifts cause false-empty detection | Open |
| Rotation detection: PCA on combined mold contours | Design agreed, not coded |
| `_step1_find_frames` multi-scale TM ~729ms bottleneck | Open — fixed ROI or reduced scales proposed |
| Threshold tuning (pin presence, rotation) | Pending real-image testing |

---

## Coding Conventions (mandatory)

- **Before writing code:** read overall function context, adapt to fit; ask if better solution exists
- **Output only the changed block/function/class** — specify exactly where it goes (replace / insert after X)
- No assumptions on ambiguous requirements — ask first
- Parameters hardcoded; minimal user-facing controls
- Static method blocks: replace entire block when changing
- Explicit variable passing between steps; no context dicts
- Debug PNGs: `_0_gray`, `_1_thresh`, `_2_raw_contours`, `_3_filtered_contours`, `_4_canvas`
