# IC Frame Laser-Mark Inspection — Project Instructions

---

## Working Directory Layout

| Path | Purpose |
|---|---|
| `insp_exp.py` | **Main file to edit** |
| `insp_exp copy.py` | Manual backup of current stable version — rollback target |
| `Backup_Code/` | Backup files for experiment branches / manual version snapshots |
| `debug/` | Debug results: saved gray image and processed image outputs |
| `pin_recipe.json` | Frame/mold layout + slot shift refs |
| `inspection_settings.txt` | Numeric runtime settings |
| `search_mask.jpg` | Optional camera search mask |
| `templates/` | Per-character JSON templates |
| `image_source/` | DEBUG_MODE input images |
| `Inspection_result/` | Failed image output |
| `training_data/` | (to be created) SVM crop dataset |
| `svm_model.pkl` | (to be created) trained SVM model |
| `svm_collect.py` | (to be created) crop collector |
| `svm_train.py` | (to be created) model trainer |
| `svm_test.py` | (to be created) model evaluator |

---

## Project Identity

**Stack:** Python · OpenCV · PyQt5 · pypylon (Basler SDK) · RPi.GPIO  
**Hardware:** Basler acA1300-60gc GigE, 1280×1024, link-local · Raspberry Pi GPIO

---

## Module Layout (top → bottom in file)

```
InspectionResult        @dataclass — result carrier
CONFIG block            constants (DEBUG_MODE, PIN_*, FONT_*, CAMERA_*)
SettingsManager         persists inspection_settings.txt
ImageIO                 save/load helpers
ContourTemplate         JSON template r/w (contour + base64 PNG canvas)
cv2_draw_dashed_rect    drawing utility
InspectionEngine        core CV logic (static methods only)
InspectionController    owns engine + template store + camera; run() entry point
ImageView               PyQt5 QLabel subclass with zoom/pan
MaskingToolbar          toolbar widget
MaskConfirmDialog       modal for search-mask confirm
TemplatePreviewDialog   modal for template save wizard (3-step)
RightPanel              settings + controls panel
MainWindow              top-level window; wires everything
main()                  QApplication entry
```

---

## Critical Design Rules

| Rule | Detail |
|---|---|
| **No PCA rotation** | Fixed camera — removed entirely from save & compare paths; re-save all templates if pipeline changes |
| **CHAIN_APPROX_NONE** | Never use TC89_KCOS — collapses diagonal strokes ("7") causing Hu moment failure |
| **Canvas IoU, not Hu moments** | 64×64 centre-aligned canvas IoU is the similarity metric |
| **Otsu inversion check** | Bright-on-dark images: tophat+Otsu+invert used at template save; adaptive+unsharp at runtime |
| **Save ↔ compare must match** | Any extraction logic change requires full template re-save |
| **mold_size as kernel anchor** | Morph kernels sized to mold dims (~150–160px), not font ROI (~40–50px) |
| **Rotation is mold-level** | All letters rotate as one laser unit; per-letter rotation check is wrong |

---

## Inspection Pipeline (`compare_roi` — 5 steps)

```
Step 1  Presence check          contours must exist
Step 2  Shift detection         shift_px / tmpl_diagonal ≤ 0.20  (ratio-based)
Step 3  Hole/stroke integrity   hole count + area ratio
Step 4  Canvas IoU similarity   64×64 centre-aligned; weighted score:
                                  similarity×0.70 + hole_score×0.20 + aspect_score×0.10
Step 5  Aspect ratio check      bbox ratio vs template ± FONT_ASPECT_TOLERANCE
```

---

## Lead Presence Gate

**Method:** `InspectionEngine._check_lead_presence()`
- ROI boxes: 1/3 mold_width × full mold_height on each side
- White pixel ratio threshold: 20% at threshold=128
- No-lead molds → PASS result, skipped silently (logged gray), no alarm

---

## IC / Frame Numbering

- Column-first, Y-axis-first slot ordering (9 slots per mold)
- `COL_SNAP = canvas_w // 2` — frame left/right split
- Mold A + Mold B at same frame position → sequential IC numbers
- Frame sort applied before numbering

---

## Template Storage

**Font templates:** `templates/<char>_<slot>.json`  
**PIN template:** `pin_recipe.json`

JSON contains:
```json
{
  "contours": [...],
  "canvas_b64": "<base64 PNG>",
  "mold_size": [w, h],
  "canvas_w": 64, "canvas_h": 64,
  "tmpl_diagonal": ...,
  "tmpl_contour_count": ...,
  "tmpl_aspect": ...,
  "tmpl_bbox": [...],
  "shift_ref": [cx, cy]
}
```

---

## Operating Modes

```python
DEBUG_MODE = True   # loops image_source/ (BMP/JPG/PNG)
DEBUG_MODE = False  # live Basler — GrabStrategy_LatestImageOnly
```

**Output:** Failed images → `Inspection_result/`
- `<timestamp>_R.png` raw grayscale
- `<timestamp>.png`   annotated BGR

**GPIO pins:** 3 = inspect trigger (active-LOW) · 5 = busy · 7 = pass/fail

---

## Template Wizard (3-step)

1. Draw frame rect
2. Draw Mold A rect
3. Draw Mold B rect

Auto-computes 9 slot positions at `mold_size/3` pitch.  
Constraint zone: 1.3× frame-height square + `x_offset` param prevents wrong-frame mold placement.

---

## Next Work — SVM OCR Integration

**Plan file:** `SVM_OCR_Integration_Plan.md`  
**Status:** Designed, not yet coded.

### What to build (in order):

1. **`SVMClassifier` class** — before `InspectionEngine`
   - HOG: 48×48, blockSize 16×16, blockStride 8×8, cellSize 8×8, nbins 9 → 1764-dim vector
   - `predict(crop_gray)` → `(char: str, conf: float)`
   - Fallback: if `svm_model.pkl` missing → `is_ready()=False`, OCR step skipped silently

2. **Wire `_svm`** onto `InspectionController.__init__`

3. **Modify `_step3_inspect_fonts`** — OCR pass before defect pass per slot:
   ```
   recipe=""  AND ocr=""  → pass=True,  run_defect=False
   recipe=""  AND ocr="X" → pass=False, run_defect=False  (unexpected mark)
   recipe="X" AND ocr="X" → pass=True,  run_defect=True
   recipe="X" AND ocr="Y" → pass=False, run_defect=True   (wrong letter)
   recipe="X" AND ocr="?" → pass=None,  run_defect=True   (unreadable)
   ```

4. **Result dict new fields:** `ocr_char`, `ocr_conf`, `ocr_string` (9-char mold ID)

5. **Log format update:**
   ```
   F1-A [ic=1]  PASS  [45.2ms]  OCR:"290  4 BZ"
   F1-A [ic=1]  FAIL  [45.2ms]  OCR:"290  4 BZ"
     mismatch: slot2(exp=8,got=9)
     defect: slot3(low_conf=0.41)
   ```

6. **CSV export:** add `ocr_char`, `ocr_conf`, `ocr_string` columns

7. **`svm_collect.py`** — standalone script, not imported by main app
   - Iterates `image_source/`, crops all 9 slots per mold, saves 48×48 PNGs to `training_data/<CHAR>/`

### SVM Constants (hardcoded):
```python
SVM_CONF_FIRST  = 0.60
SVM_CONF_RETRY  = 0.45
SVM_ROI_EXPAND  = 1.15
```

---

## Known Open Issues

| Issue | Status |
|---|---|
| Shift detection: large shifts cause false-empty detection | Open |
| Rotation detection: PCA on combined mold contours, not per-letter | Design agreed, not coded |
| `_step1_find_frames` multi-scale TM ~729ms bottleneck | Open — fixed ROI or reduced scales proposed |
| Threshold tuning (lead presence, rotation) | Pending real-image testing |

---

## Coding Conventions (mandatory)

- **Before writing code:** read overall function context, adapt to fit; ask if better solution exists
- **Output only the changed block/function/class** — specify exactly where it goes (replace / insert after X)
- No assumptions on ambiguous requirements — ask first
- Parameters hardcoded; minimal user-facing controls
- Static method blocks: replace entire block when changing
- Explicit variable passing between steps; no context dicts
- Debug PNGs: `_0_gray`, `_1_thresh`, `_2_raw_contours`, `_3_filtered_contours`, `_4_canvas`
