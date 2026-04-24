"""
IC Frame Laser-Mark Inspection
================================
Structure
---------
  config       : constants + SettingsManager
  storage      : ImageIO, ContourTemplate, cv2_draw_dashed_rect
  engine       : InspectionEngine  (PIN search + font inspect)
  controller   : InspectionController  (owns engine + template store + cache)
  ui-widgets   : ImageView, MaskingToolbar, MaskConfirmDialog,
                 TemplatePreviewDialog, RightPanel, MainWindow
  main         : QApplication entry point

Template system
---------------
  Both PIN and font templates use contour-based storage:
    threshold -> findContours(RETR_TREE, CHAIN_APPROX_NONE)
    area filter >= MIN_CONTOUR_AREA px^2
    stored as contour points + pre-rendered filled canvas (base64 PNG)
    PCA rotation removed — fixed camera orientation, no normalisation needed

  PIN search  : TM_CCOEFF_NORMED on rendered canvas, multi-scale + IoU NMS
  Font inspect: pipeline
                  1. Presence   — contours must exist
                  2. Shift      — contour centre vs expected position
                  3. Holes      — hole count + area ratio (cleanup retry)
                  4. Similarity — centre-aligned IoU at 64x64
                  5. Aspect     — bounding-box ratio vs template
                expected shift reference stored per-slot in pin_recipe.json
"""

import sys
import os
import json
import csv
import base64
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
from dataclasses import dataclass, field
import time

# =========================================================
# INSPECTION RESULT
# =========================================================
@dataclass
class InspectionResult:
    """
    Returned by InspectionController.run().
    display    : annotated BGR image ready for _view.set_image().
    results    : list of per-letter result dicts.
    passed     : count of letters that passed.
    total      : total letters inspected.
    elapsed_ms : total wall-clock time for run() call.
    """
    display:       np.ndarray
    results:       list  = field(default_factory=list)
    passed:        int   = 0
    total:         int   = 0
    elapsed_ms:    float = 0.0
    last_lot:      bool  = False   # True when partial tray detected (not all columns filled)
    last_lot_cols: int   = 0       # number of leading columns that have chip recipe


# =========================================================
# CONFIG
# =========================================================
DEBUG_MODE     = True
PIN_ROI_W          = 120       # fixed capture size in image-space px
PIN_ROI_H          = 150
PIN_SOBEL_MAG      = 40        # minimum Sobel Y magnitude counted as a strong edge
PIN_EDGE_RATIO     = 0.150      # fraction of inner-ROI pixels that must be strong edges
PIN_TM_STRIDE   = 4      # coarse grid step (px) — skip every N pixels in result map
PIN_IOU_THR     = 0.50   # NMS overlap threshold

MIN_CONTOUR_AREA     = 1
MIN_CONTOUR_SOLIDITY = 0.35   # convex-hull fill ratio; rough surface noise < 0.35; serif I ≈ 0.40
MIN_CONTOUR_EXTENT   = 0.20   # area / bounding-rect; sparse blobs fail this
MIN_CONTOUR_REL_AREA = 0.003  # fraction of ROI area; rejects sub-0.3% specks
IMAGE_SOURCE_DIR  = "image_source/"
OUTPUT_DIR        = "Inspection_result"

# Camera
CAMERA_SERIAL        = "22202392"
CAMERA_WARMUP_FRAMES = 5
CAMERA_EXPOSURE_US   = 8000     # µs — overridden by RightPanel at runtime

# ---- Font Inspection Constants (hardcoded, not user-tunable) ----
FONT_CONFIDENCE_MIN        = 0.60
FONT_SHIFT_RATIO_MAX       = 0.50
FONT_ASPECT_TOLERANCE      = 0.25
FONT_HOLE_COUNT_TOLERANCE  = 1
FONT_HOLE_AREA_TOLERANCE   = 0.30

# ---- Last-lot detection ----
# Number of leading frame-columns (by X position) that must contain chips.
# Trailing columns must have frames detected but no chip contours.
LAST_LOT_CHIP_FRAME_COLS   = 1

# ---- Empty-slot guard (Otsu false-contour suppression) ----
# Minimum peak value in the white top-hat image required before Otsu runs.
# An empty (black) slot produces a near-zero top-hat; Otsu on that signal
# picks threshold ~2–5 and turns camera noise into false contours.
# Real laser marks produce top-hat peak >> 30.  Tune down if faint marks are
# missed; tune up if empty slots still produce false contours.
MIN_TOPHAT_SIGNAL          = 20

# ---- Reflection / False-mark Filter (empty slots only) ----
# Laser marks are thin strokes; IC surface reflections are blobs.
# Both checks must pass for a contour to be reported as unexpected mark.
MARK_MAX_FILL_RATIO       = 0.65  # non_zero / bbox_area  — blobs fill > 65%
MARK_MAX_THICKNESS_RATIO  = 0.18  # max dist-transform / slot_size — blobs > 18%

# ---- Dirty / Anomaly Detection ----
ANOMALY_MIN_AREA_RATIO    = 0.30  # secondary contour / roi_area — below → noise, ignore
DIRTY_EXTRA_RATIO_MAX     = 0.20  # extra pixels / union  > 15%  → foreign object / splatter
DIRTY_MISSING_RATIO_MAX   = 0.40  # missing pixels / union > 40% → severe stroke loss

# ---- HOG descriptor (shared: template load + runtime OCR) ----
# win 64×64 → 7×7 block positions × 4 cells × 9 bins = 1764-dim vector
_HOG_WIN  = (64, 64)
_HOG_DESC = cv2.HOGDescriptor(_HOG_WIN, (16, 16), (8, 8), (8, 8), 9)

def _compute_hog(canvas: np.ndarray) -> np.ndarray:
    """
    L2-normalised HOG vector from a binary canvas.

    Crops to the contour bounding box (+ 4px padding) before resizing,
    so the HOG window is filled with stroke signal regardless of the
    slot ROI size the canvas came from.  Makes HOG slot-size-invariant.

    Input : uint8 binary ndarray (any size).
    Output: float32 ndarray shape (1764,), L2-normalised.
            All-zero vector when canvas is empty or has no non-zero pixels.
    """
    if canvas is None or canvas.size == 0:
        return np.zeros(1764, dtype=np.float32)
    pts = cv2.findNonZero(canvas)
    if pts is None:
        return np.zeros(1764, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    pad = 4
    x1  = max(0, x - pad);              y1 = max(0, y - pad)
    x2  = min(canvas.shape[1], x+w+pad); y2 = min(canvas.shape[0], y+h+pad)
    crop    = canvas[y1:y2, x1:x2]
    resized = cv2.resize(crop, _HOG_WIN, interpolation=cv2.INTER_AREA)
    vec     = _HOG_DESC.compute(resized).flatten().astype(np.float32)
    norm    = np.linalg.norm(vec)
    return vec / max(float(norm), 1e-8)

# ---- OCR Constants (HOG cosine similarity — range 0.0–1.0) ----
OCR_CONF_EXPECTED  = 0.88   # fast path: return immediately if ≥ this
OCR_MIN_CONF       = 0.55   # below this → report "?" (unreadable)
OCR_CONF_GAP_MIN   = 0.10   # best must exceed 2nd-best by this — filters circular reflections
                             # that score similarly on "2","0","O","8" (small gap → "?")


# =========================================================
# SETTINGS MANAGER
# =========================================================
SETUP_FILE    = "Setup.json"
SETTINGS_FILE = "inspection_settings.txt"   # legacy — migrated to Setup.json on first run
MASK_FILE     = "search_mask.jpg"

# Static constants loaded from Setup.json ["static"] section.
# Values here are only the in-code fallback; Setup.json overrides them at runtime.
_SETUP_STATIC_DEFAULTS = {
    "font_confidence_min":        0.60,
    "font_shift_ratio_max":       0.50,
    "font_aspect_tolerance":      0.25,
    "font_hole_count_tolerance":  1,
    "font_hole_area_tolerance":   0.30,
    "last_lot_chip_frame_cols":   1,
    "min_tophat_signal":          20,
    "pin_edge_ratio":             0.150,
}

# User-tunable setup values — stored in Setup.json ["setup"] section.
# (header, default, min, max, is_float)
_SETTINGS_DEFAULTS = [
    ("pin_score_threshold",  0.75,  0.50,  1.00,  True ),
    ("max_matches",             6,     1,    12,  False),
    ("camera_exposure_us",   8000,   100, 100000,  False),
    ("grid_scale",            0.85,  0.50,  1.20,  True ),
    ("grid_x_frac",           0.00, -0.30,  0.30,  True ),
    ("grid_y_frac",           0.00, -0.30,  0.30,  True ),
]

_STRING_DEFAULTS = {
    "grid_letters": ",,,,,,,,,",
}

class SettingsManager:
    """
    Persists all settings to Setup.json with two sections:
      "static"  — inspection constants (loaded once at startup, apply to globals)
      "setup"   — user-tunable runtime values (bound to RightPanel spinboxes)

    Migration: if Setup.json is absent but inspection_settings.txt exists,
    the legacy txt file is read for setup values and Setup.json is written.
    """

    def __init__(self, path: str = SETUP_FILE):
        self.path = path

        # --- static section (overrides CONFIG-block constants) ---
        self._static: dict = dict(_SETUP_STATIC_DEFAULTS)

        # --- setup section (numeric) ---
        self._data: dict = {}
        for hdr, val, mn, mx, is_float in _SETTINGS_DEFAULTS:
            self._data[hdr] = {
                "value":    float(val) if is_float else int(val),
                "min":      float(mn)  if is_float else int(mn),
                "max":      float(mx)  if is_float else int(mx),
                "is_float": is_float,
            }

        # --- setup section (string) ---
        self._str_data: dict = dict(_STRING_DEFAULTS)

        # Load from file, migrate from legacy, or create fresh
        if os.path.exists(path):
            self._load_json(path)
        elif os.path.exists(SETTINGS_FILE):
            # Legacy file may be old txt format OR already new JSON format
            # (written by a previous run before the rename).  Try JSON first.
            if not self._load_json(SETTINGS_FILE):
                self._migrate_txt(SETTINGS_FILE)
            self.save()                         # write Setup.json
        else:
            self.save()                         # create Setup.json with defaults

        self._apply_statics()

    # ---- static access ---------------------------------------------------

    def get_static(self, key: str, default=None):
        return self._static.get(key, default)

    def _apply_statics(self):
        """Push static section values into module-level globals."""
        global FONT_CONFIDENCE_MIN, FONT_SHIFT_RATIO_MAX, FONT_ASPECT_TOLERANCE
        global FONT_HOLE_COUNT_TOLERANCE, FONT_HOLE_AREA_TOLERANCE
        global LAST_LOT_CHIP_FRAME_COLS, MIN_TOPHAT_SIGNAL, PIN_EDGE_RATIO
        s = self._static
        FONT_CONFIDENCE_MIN        = float(s.get("font_confidence_min",        FONT_CONFIDENCE_MIN))
        FONT_SHIFT_RATIO_MAX       = float(s.get("font_shift_ratio_max",       FONT_SHIFT_RATIO_MAX))
        FONT_ASPECT_TOLERANCE      = float(s.get("font_aspect_tolerance",      FONT_ASPECT_TOLERANCE))
        FONT_HOLE_COUNT_TOLERANCE  = int(  s.get("font_hole_count_tolerance",  FONT_HOLE_COUNT_TOLERANCE))
        FONT_HOLE_AREA_TOLERANCE   = float(s.get("font_hole_area_tolerance",   FONT_HOLE_AREA_TOLERANCE))
        LAST_LOT_CHIP_FRAME_COLS   = int(  s.get("last_lot_chip_frame_cols",   LAST_LOT_CHIP_FRAME_COLS))
        MIN_TOPHAT_SIGNAL          = int(  s.get("min_tophat_signal",          MIN_TOPHAT_SIGNAL))
        PIN_EDGE_RATIO             = float(s.get("pin_edge_ratio",             PIN_EDGE_RATIO))

    # ---- setup (numeric) access ------------------------------------------

    def get(self, header: str):
        return self._data[header]["value"]

    def get_min(self, header: str):
        return self._data[header]["min"]

    def get_max(self, header: str):
        return self._data[header]["max"]

    def set_value(self, header: str, value) -> bool:
        if header not in self._data:
            return False
        d = self._data[header]
        clamped = max(d["min"], min(d["max"], value))
        d["value"] = float(clamped) if d["is_float"] else int(round(clamped))
        return True

    # ---- setup (string) access -------------------------------------------

    def get_str(self, header: str) -> str:
        return self._str_data.get(header, "")

    def set_str(self, header: str, value: str):
        if header in self._str_data:
            self._str_data[header] = value.strip()

    # ---- persistence -----------------------------------------------------

    def save(self, path: str = None) -> str:
        target = path or self.path
        setup_section = {}
        for hdr, *_ in _SETTINGS_DEFAULTS:
            d = self._data[hdr]
            setup_section[hdr] = {
                "value": d["value"],
                "min":   d["min"],
                "max":   d["max"],
            }
        for hdr in _STRING_DEFAULTS:
            setup_section[hdr] = self._str_data[hdr]

        payload = {
            "static": dict(self._static),
            "setup":  setup_section,
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2)
        return target

    def load(self, path: str = None):
        self._load_json(path or self.path)
        self._apply_statics()

    def _load_json(self, path: str) -> bool:
        """Load Setup.json.  Returns True on success, False on any error."""
        try:
            with open(path, "r") as f:
                payload = json.load(f)

            # static section
            for key in _SETUP_STATIC_DEFAULTS:
                if key in payload.get("static", {}):
                    self._static[key] = payload["static"][key]

            # setup section (numeric)
            for hdr, _, _, _, is_float in _SETTINGS_DEFAULTS:
                entry = payload.get("setup", {}).get(hdr)
                if isinstance(entry, dict):
                    d = self._data[hdr]
                    d["min"]   = float(entry["min"]) if is_float else int(entry["min"])
                    d["max"]   = float(entry["max"]) if is_float else int(entry["max"])
                    val        = float(entry["value"])
                    clamped    = max(d["min"], min(d["max"], val))
                    d["value"] = float(clamped) if is_float else int(round(clamped))

            # setup section (string)
            for hdr in _STRING_DEFAULTS:
                entry = payload.get("setup", {}).get(hdr)
                if isinstance(entry, str):
                    self._str_data[hdr] = entry

            return True

        except Exception as e:
            print(f"[Settings] Load error ({path}): {e}")
            return False

    def _migrate_txt(self, path: str):
        """Read legacy inspection_settings.txt into the setup section."""
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    hdr = parts[0]
                    if hdr in self._str_data:
                        raw = line[len(hdr):].strip()
                        if raw.startswith('"') and raw.endswith('"'):
                            raw = raw[1:-1]
                        self._str_data[hdr] = raw.replace("\\n", "\n")
                        continue
                    if hdr not in self._data or len(parts) < 4:
                        continue
                    d = self._data[hdr]
                    try:
                        val = float(parts[1])
                        mn  = float(parts[2])
                        mx  = float(parts[3])
                    except ValueError:
                        continue
                    d["min"]   = float(mn)  if d["is_float"] else int(mn)
                    d["max"]   = float(mx)  if d["is_float"] else int(mx)
                    clamped    = max(d["min"], min(d["max"], val))
                    d["value"] = float(clamped) if d["is_float"] else int(round(clamped))
            print(f"[Settings] Migrated {path} → {self.path}")
        except Exception as e:
            print(f"[Settings] Migration error: {e}")


# =========================================================
# STORAGE — ImageIO
# =========================================================
class ImageIO:
    """
    Load / save images.
    Always normalises to TARGET_H x TARGET_W and returns GRAYSCALE.
    If source is 3-channel it is converted before resize.
    Save accepts gray or BGR — always writes as-is.
    """

    TARGET_H = 1024
    TARGET_W = 1280

    def load(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        h, w = img.shape[:2]
        if (h, w) != (self.TARGET_H, self.TARGET_W):
            img = cv2.resize(img, (self.TARGET_W, self.TARGET_H),
                             interpolation=cv2.INTER_AREA)
        return img

    def save(self, path: str, img: np.ndarray):
        cv2.imwrite(path, img)

    def list_images(self, folder: str) -> list:
        """Return sorted list of .bmp/.jpg/.png paths in folder."""
        exts = {".bmp", ".jpg", ".jpeg", ".png"}
        if not os.path.isdir(folder):
            return []
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
        return files
    
# =========================================================
# BASLER CAMERA
# =========================================================
class BaslerCamera:
    """
    Thin wrapper around pypylon InstantCamera.
    Returns GRAYSCALE frames (uint8, H×W).
    All methods are no-ops if pypylon is not installed.

    Usage
    -----
        cam = BaslerCamera(serial="", exposure_us=8000)
        cam.open()
        cam.warmup()          # discard CAMERA_WARMUP_FRAMES frames
        frame = cam.grab()    # returns np.ndarray or None on error
        cam.close()
    """

    def __init__(self, serial: str = "", exposure_us: int = CAMERA_EXPOSURE_US):
        self._serial      = serial
        self._exposure_us = exposure_us
        self._camera      = None
        self._converter   = None
        self._available   = False

        try:
            from pypylon import pylon
            self._pylon   = pylon
            self._available = True
        except ImportError:
            print("[Camera] pypylon not installed — camera disabled.")

    # ---- open / close ----
    def open(self) -> bool:
        if not self._available:
            return False
        try:
            pylon = self._pylon
            tl    = pylon.TlFactory.GetInstance()
            devs  = tl.EnumerateDevices()
            device = None
            for d in devs:
                if d.GetSerialNumber() == self._serial:
                    device = tl.CreateDevice(d)
                    break
            if device is None:
                print(f"[Camera] Serial '{self._serial}' not found.")
                return False

            self._camera = pylon.InstantCamera(device)
            self._camera.Open()
            self._camera.ExposureAuto.SetValue("Off")
            self._camera.ExposureTimeAbs.SetValue(float(self._exposure_us))
            self._camera.PixelFormat.SetValue("Mono8")
            self._camera.StartGrabbing(
                pylon.GrabStrategy_LatestImageOnly,)
                #pylon.GrabLoop_ProvidedByUser)
            print(f"[Camera] Opened. Exposure={self._exposure_us} µs")
            return True
        except Exception as e:
            print(f"[Camera] Open error: {e}")
            return False

    def set_exposure(self, us: int):
        self._exposure_us = int(us)
        if self._camera and self._camera.IsOpen():
            try:
                self._camera.ExposureTimeAbs.SetValue(float(us))
            except Exception as e:
                print(f"[Camera] Exposure set error: {e}")
    
    def warmup(self):
        """Grab and discard CAMERA_WARMUP_FRAMES frames to stabilise sensor."""
        for _ in range(CAMERA_WARMUP_FRAMES):
            self.grab()
        print(f"[Camera] Warmup done ({CAMERA_WARMUP_FRAMES} frames discarded).")

    def grab(self) -> np.ndarray | None:
        """Grab one frame. Returns H×W uint8 grayscale or None on failure."""
        if not self._available or self._camera is None:
            return None
        try:
            result = self._camera.RetrieveResult(
                5000, self._pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                img = result.GetArray().copy()  # H×W uint8 Mono8
            else:
                print(f"[Camera] Grab failed: {result.ErrorCode}")
                img = None
            result.Release()
            return img
        except Exception as e:
            print(f"[Camera] Grab error: {e}")
            return None

    def close(self):
        if self._camera:
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
            self._camera = None
        print("[Camera] Closed.")

    def is_open(self) -> bool:
        return self._camera is not None and self._camera.IsOpen()

# =========================================================
# MACHINE IO  (mockup — GPIO ports not yet defined)
# =========================================================
class MachineIO:
    """
    GPIO I/O using RPi.GPIO BOARD pin numbering.
    Matches IFLRMIV102 pin assignments.
    Falls back to mock (print-only) if RPi.GPIO not available.

    Pins (BOARD)
    ------------
    portStart = 3   INPUT  pull-up  active-LOW  — inspect trigger
    portBusy  = 5   OUTPUT          LOW=busy    HIGH=idle
    portCat   = 7   OUTPUT          HIGH=pass   LOW=fail
    """

    PORT_START = 3
    PORT_BUSY  = 5
    PORT_CAT   = 7

    def __init__(self):
        self._gpio_ok = False
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.PORT_START, GPIO.IN,  pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.PORT_BUSY,  GPIO.OUT)
            GPIO.setup(self.PORT_CAT,   GPIO.OUT)
            # idle state
            GPIO.output(self.PORT_BUSY, GPIO.HIGH)
            GPIO.output(self.PORT_CAT,  GPIO.HIGH)
            self._gpio_ok = True
            print("[IO] GPIO initialised (BOARD mode).")
        except Exception as e:
            print(f"[IO] GPIO not available — mock mode. ({e})")

    # ---- output helpers ----
    def _out(self, pin: int, high: bool):
        if self._gpio_ok:
            self._GPIO.output(pin, self._GPIO.HIGH if high else self._GPIO.LOW)

    def set_busy(self, state: bool):
        """Assert busy (LOW) or release (HIGH)."""
        self._out(self.PORT_BUSY, not state)   # LOW = busy
        if not self._gpio_ok:
            print(f"[IO-mock] busy={'ON' if state else 'OFF'}")

    def on_frame_result(self, passed: int, total: int):
        """Drive CAT pin: HIGH=pass, LOW=fail."""
        ok = (passed == total and total > 0)
        self._out(self.PORT_CAT, ok)
        if not self._gpio_ok:
            print(f"[IO-mock] result={'PASS' if ok else 'FAIL'} ({passed}/{total})")

    def on_run_start(self):
        self._out(self.PORT_CAT,  True)   # clear to HIGH
        self._out(self.PORT_BUSY, True)   # idle
        if not self._gpio_ok:
            print("[IO-mock] run start")

    def on_run_complete(self, passed: int, total: int):
        self._out(self.PORT_BUSY, True)   # release busy
        if not self._gpio_ok:
            print(f"[IO-mock] run complete {passed}/{total}")

    def on_last_lot(self, chip_cols: int):
        """Signal last-lot (partial tray) condition.
        Pulses CAT LOW→HIGH briefly so PLC can distinguish from normal FAIL.
        """
        if not self._gpio_ok:
            print(f"[IO-mock] *** LAST LOT signal — {chip_cols} col(s) filled ***")
            return
        self._GPIO.output(self.PORT_CAT, self._GPIO.LOW)
        time.sleep(0.2)
        self._GPIO.output(self.PORT_CAT, self._GPIO.HIGH)

    def wait_for_start(self, stop_flag_fn) -> bool:
        if not self._gpio_ok:
            # Mock: return immediately so the worker can simulate
            while not stop_flag_fn():
                time.sleep(0.05)
                return True
            return False

        GPIO = self._GPIO
        while not stop_flag_fn():
            
            if GPIO.input(self.PORT_START) == GPIO.LOW:
                time.sleep(0.005)                       # debounce
                if GPIO.input(self.PORT_START) == GPIO.LOW:
                    return True
            time.sleep(0.005)
        return False

    def cleanup(self):
        if self._gpio_ok:
            try:
                self._GPIO.cleanup()
            except Exception:
                pass
            
# =========================================================
# DRAW HELPERS
# =========================================================
def cv2_draw_dashed_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                          color: tuple, thickness: int = 1, dash: int = 8):
    """Draw a dashed rectangle using short cv2 line segments."""
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(abs(dx), abs(dy), 1)
        steps  = length // (dash * 2)
        for i in range(steps + 1):
            s  = i * 2 * dash / length
            e  = min((i * 2 + 1) * dash / length, 1.0)
            sx, sy = int(p1[0] + dx * s), int(p1[1] + dy * s)
            ex, ey = int(p1[0] + dx * e), int(p1[1] + dy * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))
    
def _touches_border(contour: np.ndarray, w: int, h: int) -> bool:
        """Return True if any contour point lies on the canvas edge."""
        pts = contour.reshape(-1, 2)
        return bool(
            np.any(pts[:, 0] == 0) or np.any(pts[:, 0] >= w - 1) or
            np.any(pts[:, 1] == 0) or np.any(pts[:, 1] >= h - 1)
        )
# =========================================================
# A.  ContourTemplate  — modular extraction pipeline
# =========================================================
class ContourTemplate:
    """
    Contour-based template store.

    Extraction pipeline (two named entry points)
    --------------------------------------------
      extract_frame_template(gray)
        _thresh_frame -> _morph_frame -> _find_contours

      extract_font_template(gray, mold_size)
        _thresh_font  -> _morph_font  -> _find_contours

    Each threshold / morph block is a separate @staticmethod.
    Swap any block independently without touching the others.

    Storage schema (JSON)  — unchanged from previous version.
    """

    TEMPLATE_DIR = "templates"

    def __init__(self):
        os.makedirs(self.TEMPLATE_DIR, exist_ok=True)

    # =========================================================
    # THRESHOLD BLOCKS
    # =========================================================

    @staticmethod
    def _thresh_frame(gray: np.ndarray) -> np.ndarray:
        """
        Frame ROI threshold.
        Input : grayscale ndarray
        Output: binary ndarray (0 / 255)
        Swap : replace body only — keep signature.
        """
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _hysteresis(tophat:     np.ndarray,
                    high_ratio: float = 0.60,
                    low_ratio:  float = 0.25) -> np.ndarray:
        """
        Two-threshold hysteresis on the top-hat image.

        Otsu is used to find a stable high anchor automatically — no manual
        tuning required.  The low threshold extends downward from there.

        high_val = otsu_val * high_ratio  →  definite stroke pixels
        low_val  = otsu_val * low_ratio   →  candidate pixels

        A candidate pixel is kept only if it belongs to a connected
        component that contains at least one definite pixel.
        Isolated dim noise (no strong neighbour) is discarded.

        high_ratio : 0.60  — fraction of Otsu value for "definite" gate
        low_ratio  : 0.25  — fraction of Otsu value for "candidate" gate
        """
        otsu_val, _ = cv2.threshold(
            tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        high_val  = max(1, int(otsu_val * high_ratio))
        low_val   = max(1, int(otsu_val * low_ratio))

        definite  = (tophat >= high_val).astype(np.uint8) * 255
        candidate = (tophat >= low_val ).astype(np.uint8) * 255

        n, labels = cv2.connectedComponents(candidate, connectivity=8)

        result = np.zeros_like(candidate)
        for i in range(1, n):
            comp = labels == i
            if np.any(definite[comp]):
                result[comp] = 255
        return result

    @staticmethod
    def _thresh_font(gray: np.ndarray, mold_size: int = 150) -> np.ndarray:
        h, w = gray.shape[:2]

        # Step 1: bilateral filter — edge-preserving noise reduction
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)

        # Step 2: white top-hat — isolates bright strokes on dark background
        k_size = max(9, (mold_size // 8) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)

        # Guard: Otsu on a uniform-dark (empty) slot degrades to thresholding
        # camera noise, producing false contours.  If the top-hat has no
        # meaningful signal, the slot is empty — skip Otsu entirely.
        if int(tophat.max()) < MIN_TOPHAT_SIGNAL:
            return np.zeros((h, w), dtype=np.uint8)

        # Step 3: Otsu threshold
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: border mask — suppress edge artifacts
        margin = max(3, mold_size // 50)
        mask   = np.zeros((h, w), dtype=np.uint8)
        mask[margin:h - margin, margin:w - margin] = 255
        binary = cv2.bitwise_and(binary, mask)

        return binary

    # =========================================================
    # MORPH BLOCKS
    # =========================================================

    @staticmethod
    def _morph_frame(binary: np.ndarray) -> np.ndarray:
        """
        Frame ROI morphology — open then close to remove noise.
        Input : binary ndarray from _thresh_frame
        Output: cleaned binary ndarray
        Swap : replace body only — keep signature.
        """
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        out = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
        out = cv2.morphologyEx(out,    cv2.MORPH_CLOSE, k)
        return out

    @staticmethod
    def _morph_font(binary: np.ndarray, mold_size: int = 150) -> np.ndarray:
        # OPEN first  — kills isolated speck noise before stitching
        # CLOSE after — bridges stroke gaps with a larger kernel
        k_open  = max(2, 3)#mold_size // 60)           # ~2px at mold=150
        k_close = max(2, 3) #mold_size // 50)           # ~3px at mold=150  — bridges small gaps only
        # CLOSE first — fills 1-2px breaks (serif junctions) without closing letter holes
        # OPEN after  — removes isolated noise blobs
        out = binary
        out = cv2.morphologyEx(
            out,    cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_open,  k_open)))
        out = cv2.morphologyEx(
            out, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)))
        
        return out

    # =========================================================
    # SHARED CONTOUR FINDER
    # =========================================================

    @staticmethod
    def _find_contours(binary: np.ndarray, h: int, w: int) -> tuple:
        raw_cnts, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # was TC89_KCOS

        empty_canvas = np.zeros((h, w), dtype=np.uint8)

        if not raw_cnts or hierarchy is None:
            return [], empty_canvas

        hierarchy = hierarchy[0]

        # Find largest top-level contour passing texture-noise filters
        roi_area  = max(h * w, 1)
        best_idx  = -1
        best_area = 0.0
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] != -1:
                continue                              # skip non-root contours

            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue

            # ── Relative area: reject sub-pixel specks ────────────
            if area / roi_area < MIN_CONTOUR_REL_AREA:
                continue

            # ── Solidity: reject jagged rough-surface noise ───────
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if area / max(hull_area, 1) < MIN_CONTOUR_SOLIDITY:
                continue

            # ── Extent: reject sparse irregular blobs ─────────────
            _, _, bw, bh = cv2.boundingRect(c)
            if area / max(bw * bh, 1) < MIN_CONTOUR_EXTENT:
                continue

            if area > best_area:
                best_area = area
                best_idx  = i

        if best_idx == -1:
            return [], empty_canvas

        # Root + direct children
        contours = [raw_cnts[best_idx]]
        children = []
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] == best_idx:
                children.append(c)
                contours.append(c)

        # Draw root filled, then punch holes for children
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(canvas, [contours[0]], -1, 255, cv2.FILLED)
        if children:
            cv2.drawContours(canvas, children, -1, 0, cv2.FILLED)

        return contours, canvas

    @staticmethod
    def _find_contours_all(binary: np.ndarray, h: int, w: int) -> tuple:
        """
        Font-path variant of _find_contours.

        Returns all valid root contours split into two groups:
          main_list — [largest_root, ...its_direct_hole_children]
                      → used for font inspection (presence, shift, shape)
          others    — all other valid roots
                      → passed to _check_dirty for anomaly detection

        Applies the same quality filters (area, solidity, extent) as
        _find_contours.  Canvas covers main_list only (primary root filled,
        holes punched) so the dirty check can subtract it cleanly.

        Returns ([], [], empty_canvas) when no valid root found.
        """
        raw_cnts, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        empty_canvas = np.zeros((h, w), dtype=np.uint8)

        if not raw_cnts or hierarchy is None:
            return [], [], empty_canvas

        hierarchy = hierarchy[0]
        roi_area  = max(h * w, 1)

        valid_roots = []   # (area, raw_index)
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] != -1:
                continue                              # skip non-root contours

            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            if area / roi_area < MIN_CONTOUR_REL_AREA:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(c))
            if area / max(hull_area, 1) < MIN_CONTOUR_SOLIDITY:
                continue

            _, _, bw, bh = cv2.boundingRect(c)
            if area / max(bw * bh, 1) < MIN_CONTOUR_EXTENT:
                continue

            valid_roots.append((area, i))

        if not valid_roots:
            return [], [], empty_canvas

        # Primary root = largest → feature anchor for signal / approx / HOG
        valid_roots.sort(key=lambda x: x[0], reverse=True)
        best_idx = valid_roots[0][1]

        # Main list: largest root + its direct children (holes)
        main_list = [raw_cnts[best_idx]]
        children  = []
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] == best_idx:
                children.append(c)
                main_list.append(c)

        # Others: remaining valid roots (potential dirty/foreign objects)
        others = [raw_cnts[idx] for _, idx in valid_roots[1:]]

        # Canvas: primary root filled, holes punched — secondary roots excluded
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(canvas, [main_list[0]], -1, 255, cv2.FILLED)
        if children:
            cv2.drawContours(canvas, children, -1, 0, cv2.FILLED)

        return main_list, others, canvas

    # =========================================================
    # NAMED ENTRY POINTS
    # =========================================================

    @staticmethod
    def extract_frame_template(gray: np.ndarray,
                               debug_prefix: str = "") -> tuple:
        """
        Full pipeline for FRAME / MOLD ROIs.
          _thresh_frame -> _morph_frame -> _find_contours

        Input : grayscale ndarray
        Output: (contours, canvas, thresh_binary)
                contours is [] when nothing found.
        """
        h, w   = gray.shape[:2]
        thresh = ContourTemplate._thresh_frame(gray)
        clean  = ContourTemplate._morph_frame(thresh)

        if debug_prefix:
            _write_debug(debug_prefix, gray, thresh, clean)

        contours, canvas = ContourTemplate._find_contours(clean, h, w)

        if debug_prefix and contours:
            _write_debug_contours(debug_prefix, gray, contours, canvas)

        return contours, canvas, thresh

    @staticmethod
    def extract_font_template(gray: np.ndarray,
                              mold_size:    int = 150,
                              debug_prefix: str = "") -> tuple:
        """
        Full pipeline for FONT / LETTER ROIs.
          _thresh_font -> _morph_font -> _find_contours_all

        Input : grayscale ndarray, mold_size (px) for kernel scaling
        Output: (main_list, canvas_main, clean_binary, others)
                main_list    — [primary_root, ...holes]  ([] when nothing found)
                canvas_main  — binary canvas of primary_root only
                clean_binary — post-morph binary (for dirty residual check)
                others       — secondary valid root contours (dirty detection)
        """
        h, w   = gray.shape[:2]
        thresh = ContourTemplate._thresh_font(gray, mold_size)
        clean  = ContourTemplate._morph_font(thresh, mold_size)

        if debug_prefix:
            _write_debug(debug_prefix, gray, thresh, clean)

        main_list, others, canvas = ContourTemplate._find_contours_all(clean, h, w)

        if debug_prefix and main_list:
            _write_debug_contours(debug_prefix, gray, main_list, canvas)

        return main_list, canvas, clean, others

    # =========================================================
    # ENCODE ROI  (frame / mold sections → recipe)
    # =========================================================

    @staticmethod
    def encode_roi(roi_gray:  np.ndarray,
                   rect_xywh: tuple,
                   offset:    tuple = None) -> dict:
        """
        Extract frame contours and return a JSON-serialisable section dict.
        Uses extract_frame_template internally.

        Input : roi_gray (grayscale), rect_xywh (x,y,w,h) capture position,
                offset (dx, dy) relative to frame centre — optional.
        Output: section dict  {contour, contours, canvas_b64, canvas_w, canvas_h, [offset]}
        Raises: RuntimeError when no contours found.
        """
        contours, canvas, _ = ContourTemplate.extract_frame_template(roi_gray)

        if not contours:
            x, y, w, h = rect_xywh
            raise RuntimeError(
                f"No contours found in region ({x},{y},{w},{h})")

        ch, cw     = canvas.shape[:2]
        canvas_b64 = _encode_canvas(canvas)

        section = {
            "contour":    list(rect_xywh),
            "contours":   [c.tolist() for c in contours],
            "canvas_b64": canvas_b64,
            "canvas_w":   cw,
            "canvas_h":   ch,
        }
        if offset is not None:
            section["offset"] = list(offset)
        return section

    # =========================================================
    # SAVE  (font template → JSON file)
    # =========================================================

    def save(self, name: str,
             roi_bgr:   np.ndarray,
             roi_rect:  tuple,
             mold_size: int = 150) -> str:
        """
        Extract font contours, compute metrics, write JSON template.

        Input : name (str), roi_bgr (BGR or gray ndarray),
                roi_rect (x,y,w,h), mold_size (px)
        Output: path to saved JSON file
        Raises: RuntimeError when no contours found or encode fails.
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) \
               if roi_bgr.ndim == 3 else roi_bgr.copy()

        contours, canvas, *_ = ContourTemplate.extract_font_template(
            gray, mold_size=mold_size,
            debug_prefix=f"debug/{name}")

        if not contours:
            raise RuntimeError(f"No contours found in ROI for '{name}'")

        ch, cw         = canvas.shape[:2]
        canvas_b64     = _encode_canvas(canvas)
        expected_pixels = int(np.count_nonzero(canvas))

        all_pts             = np.vstack([c.reshape(-1, 2) for c in contours])
        bx, by, bw, bh      = cv2.boundingRect(all_pts)
        tmpl_diagonal       = round(float(np.hypot(bw, bh)), 2)
        tmpl_aspect         = round(bw / max(bh, 1), 4)
        tmpl_contour_count  = len(contours)

        data = {
            "name":               name,
            "roi":                list(roi_rect),
            "type":               "contour",
            "pca_angle":          0.0,
            "mold_size":          mold_size,
            "contours":           [c.tolist() for c in contours],
            "canvas_b64":         canvas_b64,
            "canvas_w":           cw,
            "canvas_h":           ch,
            "expected_pixels":    expected_pixels,
            "tmpl_diagonal":      tmpl_diagonal,
            "tmpl_contour_count": tmpl_contour_count,
            "tmpl_aspect":        tmpl_aspect,
            "tmpl_bbox":          [bx, by, bw, bh],
        }
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    # =========================================================
    # LOAD
    # =========================================================

    def load(self, name: str) -> dict:
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("type") != "contour":
            raise ValueError(
                f"'{name}' is not a contour template "
                f"(got: {data.get('type')})")

        data["pca_angle"]  = float(data.get("pca_angle", 0.0))
        data["mold_size"]  = int(data.get("mold_size", 150))
        data["contours"]   = [np.array(c, dtype=np.int32)
                              for c in data["contours"]]
        raw             = base64.b64decode(data["canvas_b64"])
        arr             = np.frombuffer(raw, dtype=np.uint8)
        data["canvas"]  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        data["hog_vec"] = _compute_hog(data["canvas"])

        # Pre-compute shape descriptors — used by _compute_shape_score at runtime.
        # Computed here once at cache load so compare_roi pays no extraction cost.
        # InspectionEngine is defined later in the file but resolved at call time.
        outer = data["contours"][0] if data["contours"] else None
        data["approx_feat"]    = InspectionEngine._approx_features(outer) \
                                  if outer is not None else None
        data["contour_signal"] = InspectionEngine._resample_contour_signal(outer) \
                                  if outer is not None else None
        data["topo_feat"]      = InspectionEngine._skeleton_features(data["canvas"])

        # Backfill missing fields from older templates
        if not data.get("expected_pixels") and data["canvas"] is not None:
            data["expected_pixels"] = int(np.count_nonzero(data["canvas"]))

        if not data.get("tmpl_diagonal") or not data.get("tmpl_contour_count"):
            all_pts        = np.vstack(
                [c.reshape(-1, 2) for c in data["contours"]])
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            data["tmpl_diagonal"]      = round(float(np.hypot(bw, bh)), 2)
            data["tmpl_aspect"]        = round(bw / max(bh, 1), 4)
            data["tmpl_contour_count"] = len(data["contours"])
            data["tmpl_bbox"]          = [bx, by, bw, bh]

        return data

    # =========================================================
    # QUERIES
    # =========================================================

    def list_templates(self) -> list:
        return sorted(
            fn.replace("_template.json", "")
            for fn in os.listdir(self.TEMPLATE_DIR)
            if fn.endswith("_template.json")
        )


# =========================================================
# DEBUG WRITE HELPERS  (module-level, used by entry points)
# =========================================================

def _write_debug(prefix: str,
                 gray:   np.ndarray,
                 thresh: np.ndarray,
                 clean:  np.ndarray):
    """Write gray / thresh / clean binary images to debug folder."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    cv2.imwrite(f"{prefix}_0_gray.png",   gray)
    cv2.imwrite(f"{prefix}_1_thresh.png", thresh)
    cv2.imwrite(f"{prefix}_2_clean.png",  clean)


def _write_debug_contours(prefix:   str,
                          gray:     np.ndarray,
                          contours: list,
                          canvas:   np.ndarray):
    """Write filtered-contour overlay and filled canvas to debug folder."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 255, 80), 1)
    cv2.imwrite(f"{prefix}_3_contours.png", vis)
    cv2.imwrite(f"{prefix}_4_canvas.png",   canvas)
    print(f"[Debug] kept={len(contours)}  roi={gray.shape[1]}x{gray.shape[0]}")


def _encode_canvas(canvas: np.ndarray) -> str:
    """Encode a grayscale canvas ndarray to base64 PNG string."""
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Canvas PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# =========================================================
# B.  YOLOMoldDetector  — mold bbox detection via OpenVINO
# =========================================================
YOLO_MODEL_XML = "Mold_detector_openvino_model/Mold_detector.xml"

class YOLOMoldDetector:
    """
    Wraps Mold_detector OpenVINO IR model (YOLO8, single class "IC").
    detect_top_left() returns the top-left-most bbox as QRect.
    Falls back gracefully if model file is missing or inference fails.
    Output layout: [1, 5, 8400]  (cx, cy, w, h, conf) — no built-in NMS.
    """
    _INPUT_SIZE = 640

    def __init__(self):
        self._compiled = None
        self._ready    = False
        try:
            import openvino as ov
            if os.path.exists(YOLO_MODEL_XML):
                core  = ov.Core()
                model = core.read_model(YOLO_MODEL_XML)
                self._compiled = core.compile_model(model, "CPU", {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "PERFORMANCE_HINT":         "LATENCY",
                })
                self._ready = True
                print(f"[YOLO] OpenVINO model loaded: {YOLO_MODEL_XML}")
            else:
                print(f"[YOLO] Model not found: {YOLO_MODEL_XML}")
        except Exception as e:
            print(f"[YOLO] Load failed: {e}")

    def is_ready(self) -> bool:
        return self._ready

    _NMS_IOU_THR = 0.45   # IoU threshold for NMS deduplication

    def _run_inference(self, image_bgr: np.ndarray, conf_thr: float) -> list:
        """
        Run OpenVINO inference with NMS; return list of (x1, y1, w, h) in image coords.
        NMS is applied (IoU thr = _NMS_IOU_THR) so each physical mold yields one box.
        Returns [] if model not ready or no detections above conf_thr.
        """
        if not self._ready or self._compiled is None:
            return []
        try:
            if image_bgr.ndim == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            ih, iw = image_bgr.shape[:2]
            sz = self._INPUT_SIZE

            scale   = min(sz / iw, sz / ih)
            nw, nh  = int(iw * scale), int(ih * scale)
            resized = cv2.resize(image_bgr, (nw, nh))
            pad_buf = np.full((sz, sz, 3), 114, dtype=np.uint8)
            pad_x   = (sz - nw) // 2
            pad_y   = (sz - nh) // 2
            pad_buf[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

            blob   = pad_buf[:, :, ::-1].astype(np.float32) / 255.0
            blob   = blob.transpose(2, 0, 1)[np.newaxis]
            result = self._compiled(blob)
            preds  = result[0][0].T                          # [8400, 5]
            preds  = preds[preds[:, 4] >= conf_thr]
            if len(preds) == 0:
                return []

            raw_boxes, scores = [], []
            for row in preds:
                cx, cy, bw, bh = row[:4]
                x1 = max(0,  int((cx - bw / 2 - pad_x) / scale))
                y1 = max(0,  int((cy - bh / 2 - pad_y) / scale))
                x2 = min(iw, int((cx + bw / 2 - pad_x) / scale))
                y2 = min(ih, int((cy + bh / 2 - pad_y) / scale))
                if x2 > x1 and y2 > y1:
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(float(row[4]))

            if not raw_boxes:
                return []

            indices = cv2.dnn.NMSBoxes(
                raw_boxes, scores, conf_thr, self._NMS_IOU_THR)
            if len(indices) == 0:
                return []

            kept = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            return [tuple(raw_boxes[i]) for i in kept]
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            return []

    def detect_top_left(self, image_bgr: np.ndarray,
                        conf_thr: float = 0.40):
        """Return top-left-most detection as QRect (lowest x1+y1). None if none."""
        boxes = self._run_inference(image_bgr, conf_thr)
        if not boxes:
            return None
        x1, y1, w, h = min(boxes, key=lambda b: b[0] + b[1])
        from PyQt5 import QtCore
        return QtCore.QRect(x1, y1, w, h)

    def detect_all(self, image_bgr: np.ndarray,
                   conf_thr: float = 0.40) -> list:
        """
        Return all detections as list of QRect sorted X-col first then Y-row.
        Column snapping uses median bbox width for grouping.
        Returns [] if model not ready or no detections.
        """
        boxes = self._run_inference(image_bgr, conf_thr)
        if not boxes:
            return []
        median_w  = sorted(b[2] for b in boxes)[len(boxes) // 2]
        col_snap  = max(1, median_w // 2)
        boxes_sorted = sorted(boxes,
                              key=lambda b: (round(b[0] / col_snap), b[1]))
        from PyQt5 import QtCore
        return [QtCore.QRect(x, y, w, h) for x, y, w, h in boxes_sorted]


# =========================================================
# B.  InspectionEngine  — named check steps
# =========================================================
class InspectionEngine:
    """
    Two inspection primitives.

    find_all_pin_templates(image_gray, tmpl, ...)
        Multi-scale TM_CCOEFF_NORMED on the template canvas.
        Returns match list sorted best-score-first.

    compare_roi(roi_bgr, tmpl, ...)
        Three hard-fail gates + HOG confidence check.
        Returns a result dict; key "roi_canvas" carries the extracted
        canvas so the caller can draw contours without re-extracting.

    Check steps
    -----------
      1. _check_presence  (gray, mold_size)        -> (contours, canvas)  hard fail
      2. _check_shift     (contours, tmpl, ...)    -> (shift_px, ratio)   hard fail
      3. _check_holes     (gray, outer, holes ...) -> (score, ...)        hard fail
         HOG cosine       (_compute_hog + _hog_cosine) -> confidence      hard threshold
    """

    # =========================================================
    # CHECK STEPS
    # =========================================================

    @staticmethod
    def _check_presence(gray:      np.ndarray,
                        mold_size: int) -> tuple:
        """
        Step 1 — Presence.
        Extract font contours from the ROI.

        Input : gray (H×W uint8), mold_size (px)
        Output: (contours, canvas, clean_binary, others)
                contours     — main font contour list ([] → no mark found)
                canvas       — binary canvas of main contour only
                clean_binary — post-morph binary (for dirty residual analysis)
                others       — secondary valid roots (for dirty extra-contour check)
        """
        contours, canvas, clean_binary, others = ContourTemplate.extract_font_template(gray, mold_size=mold_size)
        return contours, canvas, clean_binary, others

    @staticmethod
    def _suppress_large_blobs(clean_binary: np.ndarray, mold_size: int) -> tuple:
        """
        Empty-slot reflection suppressor: binary white top-hat with a large kernel.

        TOPHAT(binary, k) = binary - OPEN(binary, k).
        Structures narrower than k survive (OPEN erases them → TOPHAT = binary).
        Large blobs wider than k are suppressed (OPEN ≈ blob → TOPHAT ≈ 0).

        Keeps thin laser strokes while removing wide specular reflection blobs.
        Returns (main_list, canvas, filtered_binary, others).
        """
        h, w = clean_binary.shape[:2]
        k = max(13, mold_size // 6)   # ~25px at mold=150; adjust up if marks still survive
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        filtered = cv2.morphologyEx(clean_binary, cv2.MORPH_TOPHAT, kernel)
        main_list, others, canvas = ContourTemplate._find_contours_all(filtered, h, w)
        return main_list, canvas, filtered, others

    @staticmethod
    def _check_shift(contours:  list,
                     tmpl:      dict,
                     exp_dx:    int,
                     exp_dy:    int,
                     mold_cx:   int,
                     mold_cy:   int,
                     roi_w:     int,
                     roi_h:     int) -> tuple:
        """
        Step 2 — Shift.
        Compare actual contour centre against expected position.

        Input : contours, tmpl dict (needs tmpl_diagonal),
                exp_dx / exp_dy (expected offset from mold centre),
                mold_cx / mold_cy (mold centre in image space),
                roi_w / roi_h (size of the cell ROI)
        Output: (shift_px: float, shift_ratio: float)
        """
        all_pts        = np.vstack([c.reshape(-1, 2) for c in contours])
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        local_cx       = bx + bw // 2
        local_cy       = by + bh // 2

        roi_ox   = mold_cx + exp_dx - roi_w // 2
        roi_oy   = mold_cy + exp_dy - roi_h // 2
        actual_dx = (roi_ox + local_cx) - mold_cx
        actual_dy = (roi_oy + local_cy) - mold_cy

        shift_px    = round(float(np.hypot(actual_dx - exp_dx,
                                           actual_dy - exp_dy)), 2)
        tmpl_diag   = float(tmpl.get("tmpl_diagonal", 1.0))
        shift_ratio = round(shift_px / max(tmpl_diag, 1.0), 4)
        return shift_px, shift_ratio
    
    @staticmethod
    def check_pin_presence(image_gray: np.ndarray,
                           anc_cx: int, anc_cy: int,
                           pin_sec: dict,
                           anc_sec: dict,
                           iw: int, ih: int) -> bool:
        """
        Detect thick horizontal pin stripes via Sobel Y strong-edge ratio.

        Method:
          1. Crop pin ROI using offset from anchor centre (save-time coords).
          2. Strip a horizontal margin (~12% each side) to exclude frame-cut
             artifacts at the ROI left/right edges.
          3. Compute Sobel Y; count pixels with |gradient| > PIN_SOBEL_MAG
             as a fraction of the inner ROI area.
          4. Thick pin stripes span the full width → high ratio (~0.30+).
             A few cut-tip marks are confined to edges (removed by margin)
             or too sparse → low ratio (~0.01-0.05).

        pin_sec : recipe["pin_a"] or recipe["pin_b"]
        anc_sec : recipe["anchor"]
        Returns True → pins present;  False → no pins (work dropped).
        """
        ax, ay, aw, ah = anc_sec["contour"]
        px, py, pw, ph = pin_sec["contour"]

        off_x = (px + pw // 2) - (ax + aw // 2)
        off_y = (py + ph // 2) - (ay + ah // 2)

        pcx = anc_cx + off_x
        pcy = anc_cy + off_y
        x1  = max(0,  pcx - pw // 2)
        y1  = max(0,  pcy - ph // 2)
        x2  = min(iw, x1 + pw)
        y2  = min(ih, y1 + ph)
        if x2 <= x1 or y2 <= y1:
            return False

        roi = image_gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # strip left/right margin to exclude frame-cut edge artifacts
        margin    = max(2, roi.shape[1] // 8)
        roi_inner = roi[:, margin : roi.shape[1] - margin]
        if roi_inner.shape[1] < 4:
            roi_inner = roi                              # fallback if ROI too narrow

        sobel_y    = cv2.Sobel(roi_inner, cv2.CV_64F, 0, 1, ksize=3)
        strong     = np.abs(sobel_y) > PIN_SOBEL_MAG
        edge_ratio = float(strong.sum()) / strong.size
        return edge_ratio >= PIN_EDGE_RATIO

    _ALIGN_SIZE = 64   # shared canvas resolution for alignment, IoU, dirty check

    @staticmethod
    def _centre_align(src: np.ndarray) -> np.ndarray:
        """
        Crop the white-pixel bounding box of src, scale to fit within
        _ALIGN_SIZE × _ALIGN_SIZE (aspect-preserving), and place centred.
        Returns a binary uint8 frame of size (_ALIGN_SIZE, _ALIGN_SIZE).
        """
        N = InspectionEngine._ALIGN_SIZE
        pts = cv2.findNonZero(src)
        if pts is None:
            return np.zeros((N, N), dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(pts)
        crop  = src[y:y + h, x:x + w]
        scale = min(N / max(w, 1), N / max(h, 1))
        sw    = max(1, int(round(w * scale)))
        sh    = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_NEAREST)
        frame   = np.zeros((N, N), dtype=np.uint8)
        ox = (N - sw) // 2
        oy = (N - sh) // 2
        frame[oy:oy + sh, ox:ox + sw] = resized
        return frame

    @staticmethod
    def _check_similarity(canvas:      np.ndarray,
                          tmpl_canvas: np.ndarray) -> float:
        """
        Centre-aligned IoU at _ALIGN_SIZE resolution.
        Positional offset removed — shift is checked separately.

        Input : canvas (from _check_presence), tmpl_canvas (from template dict)
        Output: float 0.0–1.0
        """
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return 0.0

        rc = InspectionEngine._centre_align(canvas)
        tc = InspectionEngine._centre_align(tmpl_canvas)

        intersection = np.count_nonzero(cv2.bitwise_and(rc, tc))
        union        = np.count_nonzero(cv2.bitwise_or(rc, tc))
        return round(float(intersection / max(union, 1)), 4)
    
    @staticmethod
    def _check_aspect(contours:    list,
                      tmpl_aspect: float) -> tuple:
        """
        Step 6 — Aspect ratio.
        Bounding-box aspect ratio vs template — catches horizontal / vertical distortion.

        Input : contours (from _check_presence), tmpl_aspect (from template dict)
        Output: (roi_aspect: float, aspect_diff: float)
                aspect_diff is fractional deviation from tmpl_aspect.
        """
        all_pts        = np.vstack([c.reshape(-1, 2) for c in contours])
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        roi_aspect     = round(bw / max(bh, 1), 4)
        aspect_diff    = abs(roi_aspect - tmpl_aspect) / max(tmpl_aspect, 1e-6)
        return roi_aspect, aspect_diff

    @staticmethod
    def _hog_cosine(q_hog: np.ndarray, tmpl_hog: np.ndarray) -> float:
        """
        Cosine similarity between two pre-normalised HOG vectors.
        Both must be L2-normalised float32 output of _compute_hog().
        Returns 0.0 when either vector is None or zero.
        """
        if q_hog is None or tmpl_hog is None:
            return 0.0
        return float(np.clip(np.dot(q_hog, tmpl_hog), 0.0, 1.0))

    # =========================================================
    # DESCRIPTOR: polygon approximation
    # =========================================================

    @staticmethod
    def _approx_features(contour: np.ndarray) -> tuple:
        """
        Douglas-Peucker approximation of outer contour.
        epsilon = 0.04 × perimeter — tuned for laser stroke widths 3-6 px.

        Input : outer contour (np.int32)
        Output: (vertex_count: int, sorted_angles: list[float])
                angles are interior turn angles in degrees, ascending.
        """
        peri   = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, closed=True)
        pts    = approx.reshape(-1, 2).astype(np.float32)
        n      = len(pts)
        angles = []
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            v1    = a - b
            v2    = c - b
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            cos_a = np.dot(v1, v2) / max(float(denom), 1e-6)
            angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
        return n, sorted(angles)

    @staticmethod
    def _approx_score(feat_q: tuple, feat_t: tuple) -> float:
        """
        Similarity between two _approx_features() outputs.

        Vertex count mismatch > 1  → returns 0.0 immediately (hard gate).
        Angle list L1 distance mapped to [0, 1].
        Shorter list padded with 180° (straight segment).

        Input : feat_q, feat_t — tuples from _approx_features()
        Output: float [0, 1]
        """
        if feat_q is None or feat_t is None:
            return 0.0
        n_q, ang_q = feat_q
        n_t, ang_t = feat_t
        if abs(n_q - n_t) > 1:
            return 0.0
        max_n = max(len(ang_q), len(ang_t))
        aq    = ang_q + [180.0] * (max_n - len(ang_q))
        at    = ang_t + [180.0] * (max_n - len(ang_t))
        l1    = sum(abs(a - b) for a, b in zip(aq, at)) / max(max_n * 180.0, 1.0)
        return round(float(max(0.0, 1.0 - l1)), 4)

    # =========================================================
    # DESCRIPTOR: resampled contour radius signal
    # =========================================================

    @staticmethod
    def _resample_contour_signal(contour: np.ndarray,
                                  n: int = 64) -> "np.ndarray | None":
        """
        Resample outer contour to N equally-spaced arc-length points.
        Returns normalised centroid-distance signal, shape (N,) float32.
        Returns None for degenerate contours (perimeter < 1 px).

        Input : contour (np.int32), n — number of sample points
        Output: float32 ndarray shape (n,), values in [0, 1]  or  None
        """
        if contour is None:
            return None
        pts    = contour.reshape(-1, 2).astype(np.float32)
        diffs  = np.diff(pts, axis=0, append=pts[:1])
        segs   = np.hypot(diffs[:, 0], diffs[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(segs)])   # shape N+1
        total  = float(cumlen[-1])
        if total < 1.0:
            return None
        # Extend pts to N+1 by wrapping first point — matches cumlen length
        pts_ext  = np.vstack([pts, pts[:1]])
        targets  = np.linspace(0.0, total, n, endpoint=False)
        resampled = np.column_stack([
            np.interp(targets, cumlen, pts_ext[:, 0]),
            np.interp(targets, cumlen, pts_ext[:, 1]),
        ])
        cx, cy  = resampled.mean(axis=0)
        radii   = np.hypot(resampled[:, 0] - cx, resampled[:, 1] - cy)
        max_r   = float(radii.max())
        if max_r < 1e-6:
            return None
        return (radii / max_r).astype(np.float32)

    @staticmethod
    def _contour_signal_sim(sig_q: "np.ndarray | None",
                             sig_t: "np.ndarray | None") -> float:
        """
        Rotation-invariant similarity between two radius signals.
        Uses circular cross-correlation via FFT — finds best rotational alignment.
        Returns 0.0 when either signal is None.

        Input : sig_q, sig_t — float32 ndarrays from _resample_contour_signal()
        Output: float [0, 1]
        """
        if sig_q is None or sig_t is None:
            return 0.0
        corr  = np.real(np.fft.ifft(
            np.fft.fft(sig_q) * np.conj(np.fft.fft(sig_t))))
        denom = np.sqrt(np.dot(sig_q, sig_q) * np.dot(sig_t, sig_t))
        return float(np.clip(np.max(corr) / max(float(denom), 1e-6), 0.0, 1.0))

    # =========================================================
    # DESCRIPTOR: skeleton topology
    # =========================================================

    @staticmethod
    def _skeleton_features(canvas: np.ndarray) -> "dict | None":
        """
        Zhang-Suen thinning → vectorised crossing-number topology.
        Requires opencv-contrib (cv2.ximgproc.thinning).
        Returns None silently when ximgproc is unavailable or canvas is empty.

        Crossing number (CN) per skeleton pixel (8-neighbourhood):
          CN = 1  → endpoint (stroke tip)
          CN ≥ 3  → branch / junction

        Input : filled binary canvas uint8
        Output: dict with keys:
                  skel        — uint8 skeleton image
                  n_endpoints — int
                  n_branches  — int
                  n_loops     — int  (interior connected components)
                  stroke_len  — int  (total skeleton px)
                  avg_width   — float (mean stroke width in px)
                or None
        """
        if canvas is None or not np.any(canvas):
            return None
        try:
            skel = cv2.ximgproc.thinning(
                canvas, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            return None

        p = (skel > 0).astype(np.int32)

        # Vectorised crossing-number over interior pixels
        nb = np.stack([
            p[:-2, 1:-1], p[:-2, 2:],  p[1:-1, 2:],  p[2:, 2:],
            p[2:,  1:-1], p[2:,  :-2], p[1:-1, :-2], p[:-2, :-2],
        ], axis=2)
        nb_shift = np.roll(nb, -1, axis=2)
        cn       = np.sum(np.abs(nb_shift - nb), axis=2) // 2
        center   = p[1:-1, 1:-1]

        n_endpoints = int(np.count_nonzero((cn == 1) & (center == 1)))
        n_branches  = int(np.count_nonzero((cn >= 3) & (center == 1)))
        stroke_len  = int(np.count_nonzero(skel))

        # Loop count via interior connected components of the filled canvas
        inv    = cv2.bitwise_not(canvas)
        n_cc, _ = cv2.connectedComponents(inv, connectivity=4)
        n_loops = max(0, n_cc - 1)   # subtract background component

        # Average stroke width from distance transform sampled at skeleton
        dist      = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
        skel_mask = skel > 0
        avg_width = float(dist[skel_mask].mean()) * 2.0 \
                    if np.any(skel_mask) else 0.0

        return {
            "skel":        skel,
            "n_endpoints": n_endpoints,
            "n_branches":  n_branches,
            "n_loops":     n_loops,
            "stroke_len":  stroke_len,
            "avg_width":   round(avg_width, 2),
        }

    @staticmethod
    def _topology_score(feat_q: "dict | None", feat_t: "dict | None") -> float:
        """
        Normalised similarity between two _skeleton_features() dicts.
        Four-component weighted vector:
          endpoints (35%) · branches (25%) · loops (25%) · stroke_len (15%)

        Count tolerance: endpoints/branches ±2, loops ±1.

        Input : feat_q, feat_t — dicts from _skeleton_features()
        Output: float [0, 1]  — 0.0 when either feat is None
        """
        if feat_q is None or feat_t is None:
            return 0.0

        def _cnt(a: int, b: int, tol: int) -> float:
            return max(0.0, 1.0 - abs(a - b) / max(tol, 1))

        ep_sim  = _cnt(feat_q["n_endpoints"], feat_t["n_endpoints"], 2)
        br_sim  = _cnt(feat_q["n_branches"],  feat_t["n_branches"],  2)
        lp_sim  = _cnt(feat_q["n_loops"],     feat_t["n_loops"],     1)
        sq, st  = feat_q["stroke_len"], feat_t["stroke_len"]
        len_sim = 1.0 - abs(sq - st) / max(sq + st, 1)

        score = ep_sim * 0.35 + br_sim * 0.25 + lp_sim * 0.25 + len_sim * 0.15
        return round(float(score), 4)

    # =========================================================
    # DESCRIPTOR: skeleton IoU
    # =========================================================

    @staticmethod
    def _skeleton_iou(canvas_q: np.ndarray,
                       canvas_t: np.ndarray,
                       dilate_px: int = 2) -> float:
        """
        IoU between centre-aligned dilated skeletons.
        Dilation adds sub-pixel tolerance without collapsing fine stroke detail.
        Returns 0.0 if ximgproc unavailable or skeletons are empty.

        Input : canvas_q / canvas_t — filled binary uint8 canvases
                dilate_px — dilation radius in pixels (default 2)
        Output: float [0, 1]
        """
        def _thin(c: np.ndarray) -> "np.ndarray | None":
            try:
                return cv2.ximgproc.thinning(
                    c, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            except AttributeError:
                return None

        def _centre_align(src: np.ndarray, size: int = 64) -> np.ndarray:
            pts = cv2.findNonZero(src)
            if pts is None:
                return np.zeros((size, size), dtype=np.uint8)
            x, y, w, h = cv2.boundingRect(pts)
            crop  = src[y:y + h, x:x + w]
            scale = min(size / max(w, 1), size / max(h, 1))
            sw    = max(1, int(round(w * scale)))
            sh    = max(1, int(round(h * scale)))
            rsz   = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_NEAREST)
            frame = np.zeros((size, size), dtype=np.uint8)
            frame[(size - sh) // 2:(size - sh) // 2 + sh,
                  (size - sw) // 2:(size - sw) // 2 + sw] = rsz
            return frame

        sq = _thin(canvas_q)
        st = _thin(canvas_t)
        if sq is None or st is None:
            return 0.0

        sq = _centre_align(sq)
        st = _centre_align(st)

        if dilate_px > 0:
            k  = cv2.getStructuringElement(
                cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            sq = cv2.dilate(sq, k)
            st = cv2.dilate(st, k)

        inter = np.count_nonzero(cv2.bitwise_and(sq, st))
        union = np.count_nonzero(cv2.bitwise_or(sq, st))
        return round(float(inter / max(union, 1)), 4)

    # =========================================================
    # COMPOSITE SCORER
    # =========================================================

    @staticmethod
    def _compute_shape_score(canvas_q:   np.ndarray,
                              contours_q: list,
                              tmpl:       dict) -> float:
        """
        Weighted combination of all shape descriptors.
        Each descriptor is independent — zero its weight to disable it.

        Descriptor weights (must sum to 1.0):
          W_HOG      0.25   gradient texture (HOG cosine)
          W_FILLED   0.15   global silhouette (filled canvas IoU)
          W_SKEL     0.25   stroke position (skeleton IoU, dilated)
          W_SIGNAL   0.20   outline shape (resampled radius signal)
          W_APPROX   0.15   corner topology (polygon approximation)

        Skeleton descriptors fall back gracefully when ximgproc is absent
        — their weight is redistributed to HOG + filled IoU.

        Input : canvas_q   — extracted binary canvas for query slot
                contours_q — contour list from _check_presence
                tmpl       — loaded template dict (pre-computed fields expected)
        Output: float [0, 1]
        """
        W_HOG    = 0.25
        W_FILLED = 0.15
        W_SKEL   = 0.25
        W_SIGNAL = 0.20
        W_APPROX = 0.15

        q_hog = _compute_hog(canvas_q)
        hog   = InspectionEngine._hog_cosine(q_hog, tmpl.get("hog_vec"))

        filled = InspectionEngine._check_similarity(canvas_q, tmpl.get("canvas"))

        skel = InspectionEngine._skeleton_iou(canvas_q, tmpl.get("canvas", np.zeros((1,1), np.uint8)))
        topo = InspectionEngine._topology_score(
            InspectionEngine._skeleton_features(canvas_q),
            tmpl.get("topo_feat"))

        outer_q  = contours_q[0] if contours_q else None
        sig_q    = InspectionEngine._resample_contour_signal(outer_q) \
                   if outer_q is not None else None
        signal   = InspectionEngine._contour_signal_sim(sig_q, tmpl.get("contour_signal"))

        af_q   = InspectionEngine._approx_features(outer_q) \
                 if outer_q is not None else None
        approx = InspectionEngine._approx_score(af_q, tmpl.get("approx_feat"))

        # Redistribute skeleton weight when unavailable
        if skel == 0.0 and topo == 0.0:
            w_hog    = W_HOG    + W_SKEL * 0.60
            w_filled = W_FILLED + W_SKEL * 0.40
            w_skel   = 0.0
        else:
            w_hog    = W_HOG
            w_filled = W_FILLED
            w_skel   = W_SKEL

        score = (hog    * w_hog  +
                 filled * w_filled +
                 skel   * w_skel  +
                 topo   * 0.0     +   # topology informs logging; weight here = 0
                 signal * W_SIGNAL +
                 approx * W_APPROX)

        return round(float(np.clip(score, 0.0, 1.0)), 4)

    @staticmethod
    def _is_laser_mark(canvas: np.ndarray) -> bool:
        """
        Return True if canvas looks like a genuine thin-stroke laser mark.
        Return False if it resembles an IC surface reflection (blob).

        Used only for EMPTY slot unexpected-mark detection — prevents
        medium-area specular reflections from being reported as marks.

        Two checks (both must pass):
          1. Fill ratio  — laser strokes fill < MARK_MAX_FILL_RATIO of bbox
          2. Stroke thinness — max inscribed circle radius / slot size
                               < MARK_MAX_THICKNESS_RATIO

        A ring-shaped reflection passes check 1 (ring ≈ low fill) but
        fails check 2 (ring wall is thicker than a laser stroke).
        A solid blob fails check 1 immediately.
        """
        if canvas is None:
            return False
        white = int(cv2.countNonZero(canvas))
        if white == 0:
            return False

        pts = cv2.findNonZero(canvas)
        if pts is None:
            return False
        _, _, bw, bh = cv2.boundingRect(pts)
        if bw == 0 or bh == 0:
            return False

        # Check 1: fill ratio
        fill_ratio = white / max(bw * bh, 1)
        if fill_ratio > MARK_MAX_FILL_RATIO:
            return False

        # Check 2: stroke thinness via distance transform
        dist      = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
        slot_size = float(max(canvas.shape[0], canvas.shape[1]))
        if dist.max() / max(slot_size, 1.0) > MARK_MAX_THICKNESS_RATIO:
            return False

        return True

    def _check_holes(self,
                    gray:             np.ndarray,
                    roi_outer:        np.ndarray,
                    roi_holes:        list,
                    canvas:           np.ndarray,
                    contours:         list,
                    tmpl_hole_count:  int,
                    tmpl_hole_ratios: list,
                    mold_size:        int) -> tuple:
        """
        Hole count + area ratio check with cleanup retry.
        Returns (score, roi_holes, canvas, contours).
        score = -1.0 → hard fail.
        """
        def _hole_score(holes: list) -> float:
            if tmpl_hole_count == 0 and len(holes) == 0:
                return 1.0
            count_diff = abs(len(holes) - tmpl_hole_count)
            if count_diff > FONT_HOLE_COUNT_TOLERANCE:
                return -1.0

            if not holes or not tmpl_hole_ratios:
                return 1.0 if tmpl_hole_count == 0 else 0.5

            outer_area = cv2.contourArea(roi_outer)
            roi_ratios = sorted(
                [cv2.contourArea(h) / max(outer_area, 1) for h in holes],
                reverse=True)
            tmpl_sorted = sorted(tmpl_hole_ratios, reverse=True)

            ratio_scores = []
            for r, t in zip(roi_ratios, tmpl_sorted):
                diff = abs(r - t) / max(t, 1e-6)
                ratio_scores.append(max(0.0, 1.0 - diff / max(FONT_HOLE_AREA_TOLERANCE, 1e-6)))

            return round(sum(ratio_scores) / max(len(ratio_scores), 1), 4)

        # First attempt
        score = _hole_score(roi_holes)
        if score >= 0:
            return score, roi_holes, canvas, contours

        # Hard-fail — cleanup retry with close+open
        h, w   = gray.shape[:2]
        roi_sz = min(h, w)
        k      = max(2, roi_sz // 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        thresh = ContourTemplate._thresh_font(gray, mold_size)
        clean  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel)

        new_contours, new_canvas = ContourTemplate._find_contours(clean, h, w)
        if not new_contours:
            return -1.0, roi_holes, canvas, contours

        new_outer = new_contours[0]
        new_holes = new_contours[1:] if len(new_contours) > 1 else []
        score2    = _hole_score(new_holes)

        if score2 < 0:
            return -1.0, roi_holes, canvas, contours

        return score2, new_holes, new_canvas, new_contours

    @staticmethod
    def ocr_identify(canvas:        np.ndarray,
                     contours:      list,
                     expected_char: str,
                     all_templates: dict) -> tuple:
        """
        OCR identification using the same multi-descriptor scorer as compare_roi.

        Query features pre-computed once — reused across all template comparisons.
        Skeleton descriptors omitted (ximgproc absent); weights redistributed to
        HOG, filled IoU, contour signal, and polygon approx.

        Effective weights (skeleton absent):
          HOG     0.40   filled IoU  0.25   signal  0.20   approx  0.15

        Three-stage search:

        Stage 1 — Expected char fast path
            Score expected template directly.
            If score >= OCR_CONF_EXPECTED (0.88) return immediately.

        Stage 2 — Group search (fallback)
            Alpha / numeric groups ordered by expected char type.
            Skip next group only when best already exceeds exp_conf.

        Returns (char: str, conf: float).
        Returns ("?", conf) if nothing clears OCR_MIN_CONF (0.55).
        """
        if not all_templates or canvas is None:
            return "?", 0.0

        # ── Pre-compute query features once ───────────────────────
        outer    = contours[0] if contours else None
        q_hog    = _compute_hog(canvas)
        q_signal = InspectionEngine._resample_contour_signal(outer) \
                   if outer is not None else None
        q_approx = InspectionEngine._approx_features(outer) \
                   if outer is not None else None

        def _score(tmpl: dict) -> float:
            hog    = InspectionEngine._hog_cosine(q_hog, tmpl.get("hog_vec"))
            filled = InspectionEngine._check_similarity(canvas, tmpl.get("canvas"))
            signal = InspectionEngine._contour_signal_sim(q_signal,
                                                          tmpl.get("contour_signal"))
            approx = InspectionEngine._approx_score(q_approx,
                                                    tmpl.get("approx_feat"))
            return hog * 0.40 + filled * 0.25 + signal * 0.20 + approx * 0.15

        # ── Empty slot: flat scan, no group ordering ───────────────
        if not expected_char:
            best_char   = "?"
            best_conf   = 0.0
            second_conf = 0.0
            for name, tmpl in all_templates.items():
                score = _score(tmpl)
                if score > best_conf:
                    second_conf = best_conf
                    best_conf   = score
                    best_char   = name
                elif score > second_conf:
                    second_conf = score
            if best_conf < OCR_MIN_CONF:
                return "?", round(best_conf, 4)
            if best_conf - second_conf < OCR_CONF_GAP_MIN:
                return "?", round(best_conf, 4)
            return best_char, round(best_conf, 4)

        # ── Stage 1: expected char fast path ──────────────────────
        exp_conf = 0.0
        exp_tmpl = all_templates.get(expected_char)
        if exp_tmpl is not None:
            exp_conf = _score(exp_tmpl)

        if exp_conf >= OCR_CONF_EXPECTED:
            return expected_char, round(exp_conf, 4)

        # ── Stage 2: group search — seeded with expected char ──────
        # Same-type group runs first (alpha→alpha, num→num).
        # Each group is fully scored before deciding whether to continue.
        # Skip remaining groups only when a template already beat exp_conf —
        # a different-type match can never be more reliable than same-type.
        best_char   = expected_char if exp_conf >= OCR_MIN_CONF else "?"
        best_conf   = exp_conf
        second_conf = 0.0

        alpha_group   = {k: v for k, v in all_templates.items()
                         if k.isalpha() and k != expected_char}
        numeric_group = {k: v for k, v in all_templates.items()
                         if not k.isalpha() and k != expected_char}

        groups = [alpha_group, numeric_group] if expected_char.isalpha() \
                 else [numeric_group, alpha_group]

        for group in groups:
            for name, tmpl in group.items():
                score = _score(tmpl)
                if score > best_conf:
                    second_conf = best_conf
                    best_conf   = score
                    best_char   = name
                elif score > second_conf:
                    second_conf = score
            # Full group scored — only skip next group if a winner beat exp_conf
            if best_conf > exp_conf:
                break

        if best_conf < OCR_MIN_CONF:
            return "?", round(best_conf, 4)
        if best_char != expected_char and best_conf - second_conf < OCR_CONF_GAP_MIN:
            return "?", round(best_conf, 4)

        return best_char, round(best_conf, 4)

    def find_all_pin_templates(self,
                           image_bgr:   np.ndarray,
                           tmpl:        dict,
                           score_thr:   float = 0.75,
                           max_matches: int   = 6,
                           mask:        np.ndarray = None) -> list:
        """
        Single-scale TM_CCOEFF_NORMED search with coarse grid stride.
        Scale fixed at 1.0 — camera and IC size are constant per machine.
        Returns list of (cx, cy, score, w, h, scale) sorted best-score-first.
        NMS applied with PIN_IOU_THR.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
            if image_bgr.ndim == 3 else image_bgr.copy()

        tmpl_canvas = tmpl.get("canvas")
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return []

        th, tw = tmpl_canvas.shape[:2]
        ih, iw = gray.shape[:2]

        if tw > iw or th > ih:
            return []

        search = cv2.bitwise_and(gray, mask) if mask is not None else gray
        result = cv2.matchTemplate(search, tmpl_canvas, cv2.TM_CCOEFF_NORMED)

        # Coarse grid — sample result map at PIN_TM_STRIDE intervals
        rh, rw = result.shape[:2]
        ys = np.arange(0, rh, PIN_TM_STRIDE)
        xs = np.arange(0, rw, PIN_TM_STRIDE)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        grid_scores = result[grid_y, grid_x]

        above = np.where(grid_scores >= score_thr)
        if above[0].size == 0:
            return []

        candidates = []
        for gi, gj in zip(above[0], above[1]):
            ry = int(ys[gi])
            rx = int(xs[gj])
            score = float(result[ry, rx])
            cx = rx + tw // 2
            cy = ry + th // 2
            candidates.append((cx, cy, score, tw, th, 1.0))

        if not candidates:
            return []

        # NMS
        candidates.sort(key=lambda c: c[2], reverse=True)
        kept = []
        for cand in candidates:
            cx, cy, score, w, h, scale = cand
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = x1 + w,      y1 + h
            suppressed = False
            for k in kept:
                kx, ky, _, kw, kh, _ = k
                kx1, ky1 = kx - kw // 2, ky - kh // 2
                kx2, ky2 = kx1 + kw,     ky1 + kh
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = w * h + kw * kh - inter
                if union > 0 and inter / union > PIN_IOU_THR:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)
            if len(kept) >= max_matches:
                break

        return kept

    @staticmethod
    def _check_dirty(others:      list,
                     canvas:      np.ndarray,
                     tmpl_canvas: np.ndarray,
                     roi_h:       int,
                     roi_w:       int) -> dict:
        """
        Step 3 — Dirty check.

        Aligns the ROI canvas and template canvas to _ALIGN_SIZE×_ALIGN_SIZE
        (same centre-align as _check_similarity), then computes:

          rc = centre_align(canvas)      — query
          tc = centre_align(tmpl_canvas) — template
          union_area = |rc ∪ tc|

          extra   = rc AND NOT tc   → pixels present in ROI but absent from template
                                      (residual: foreign object / splatter / stroke overshoot)
          missing = tc AND NOT rc   → pixels in template absent from ROI
                                      (union-gap: stroke loss / severe erosion)

        Also checks the 'others' list (secondary valid roots from _find_contours_all)
        for significant blobs that are completely outside the main contour bounding box
        and therefore invisible in the aligned 64×64 canvas.

        Returns dict:
            detected       — bool
            type           — "none" | "foreign_object" | "missing_stroke"
            extra_ratio    — count(extra)   / union_area
            missing_ratio  — count(missing) / union_area
            area_ratio     — max(extra_ratio, missing_ratio)  (for logging compat)
        """
        # ── Aligned canvas comparison ─────────────────────────────────
        rc = InspectionEngine._centre_align(canvas) \
             if canvas is not None else \
             np.zeros((InspectionEngine._ALIGN_SIZE,
                       InspectionEngine._ALIGN_SIZE), dtype=np.uint8)
        tc = InspectionEngine._centre_align(tmpl_canvas) \
             if tmpl_canvas is not None else \
             np.zeros((InspectionEngine._ALIGN_SIZE,
                       InspectionEngine._ALIGN_SIZE), dtype=np.uint8)

        union_area = max(int(np.count_nonzero(cv2.bitwise_or(rc, tc))), 1)

        extra   = cv2.subtract(rc, tc)   # residual  — in ROI, not in template
        missing = cv2.subtract(tc, rc)   # union-gap — in template, not in ROI

        extra_ratio   = round(int(np.count_nonzero(extra))   / union_area, 4)
        missing_ratio = round(int(np.count_nonzero(missing)) / union_area, 4)

        # ── Secondary-contour check (blobs outside aligned bbox) ─────
        roi_area          = max(roi_h * roi_w, 1)
        others_max_ratio  = 0.0
        for c in others:
            r = cv2.contourArea(c) / roi_area
            if r >= ANOMALY_MIN_AREA_RATIO:
                others_max_ratio = max(others_max_ratio, r)

        # ── Decision ─────────────────────────────────────────────────
        dirty_type = "none"
        if extra_ratio > DIRTY_EXTRA_RATIO_MAX or others_max_ratio >= ANOMALY_MIN_AREA_RATIO:
            dirty_type = "foreign_object"
        elif missing_ratio > DIRTY_MISSING_RATIO_MAX:
            dirty_type = "missing_stroke"

        area_ratio = round(max(extra_ratio, missing_ratio, others_max_ratio), 4)

        return {
            "detected":      dirty_type != "none",
            "type":          dirty_type,
            "extra_ratio":   extra_ratio,
            "missing_ratio": missing_ratio,
            "area_ratio":    area_ratio,
            "extra_map":     extra,    # 64×64 uint8 — pixels present in ROI but absent from template
            "missing_map":   missing,  # 64×64 uint8 — pixels in template absent from ROI
        }

    # =========================================================
    # COMPARE ROI  — orchestrates the pipeline
    # =========================================================

    def compare_roi(self,
                roi_bgr:      np.ndarray,
                tmpl:         dict,
                exp_dx:       int,
                exp_dy:       int,
                mold_cx:      int,
                mold_cy:      int,
                mold_size:    int   = 150,
                is_retry:     bool  = False,
                precomputed:  tuple = None) -> dict:
        """
        precomputed : (contours, canvas, clean_binary, others) from a prior
                      _check_presence call — skips Step 1 re-extraction.
                      Pass None to extract fresh.
        """

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) \
            if roi_bgr.ndim == 3 else roi_bgr.copy()

        orig_roi_h, orig_roi_w = gray.shape[:2]
        roi_h, roi_w = gray.shape[:2]
        roi_thresh = ContourTemplate._thresh_font(gray, mold_size)

        tmpl_contours = tmpl.get("contours", [])

        _no_dirty = {"detected": False, "type": "none", "area_ratio": 0.0, "contours": []}

        def _fail(step: int, reason: str, extras: dict = None) -> dict:
            base = {
                "pass":        False,
                "confidence":  0.0,
                "shift_px":    0.0,
                "shift_ratio": 0.0,
                "reason":      reason,
                "defect_step": step,
                "roi_canvas":  None,
                "roi_thresh":  roi_thresh,
                "tmpl_canvas": tmpl.get("canvas"),
                "orig_roi_w":  orig_roi_w,
                "orig_roi_h":  orig_roi_h,
                "dirty":       _no_dirty,
            }
            if extras:
                base.update(extras)
            return base

        if not tmpl_contours:
            return _fail(0, "no template contours")

        # Template geometry
        tmpl_outer      = tmpl_contours[0]
        tmpl_holes      = tmpl_contours[1:] if len(tmpl_contours) > 1 else []
        tmpl_hole_count = len(tmpl_holes)
        tmpl_outer_area = cv2.contourArea(tmpl_outer)
        tmpl_hole_ratios = [
            cv2.contourArea(h) / max(tmpl_outer_area, 1)
            for h in tmpl_holes
        ]

        # ── Step 1 : Presence (hard) ─────────────────────────────────
        if precomputed is not None:
            contours, canvas, clean_binary, others = precomputed
        else:
            contours, canvas, clean_binary, others = self._check_presence(gray, mold_size)

        if not contours:
            return _fail(1, "missing mark")

        roi_outer = contours[0]
        roi_holes = contours[1:] if len(contours) > 1 else []

        # ── Step 2 : Align + Dirty check (hard) ─────────────────────
        dirty = self._check_dirty(others, canvas, tmpl.get("canvas"), roi_h, roi_w)
        if dirty["detected"]:
            return _fail(2,
                f"dirty:{dirty['type']} area={dirty['area_ratio']:.3f}",
                {"roi_canvas": canvas, "dirty": dirty})

        # ── Step 3 : Shift (hard) ────────────────────────────────────
        shift_px, shift_ratio = self._check_shift(
            contours, tmpl, exp_dx, exp_dy, mold_cx, mold_cy, roi_w, roi_h)

        if shift_ratio > FONT_SHIFT_RATIO_MAX:
            return _fail(3,
                f"shift ratio={shift_ratio:.3f} > {FONT_SHIFT_RATIO_MAX}",
                {"shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "roi_canvas":  canvas})

        # ── Step 4 : Holes (hard, with cleanup retry) ────────────────
        hole_score, roi_holes, canvas, contours = self._check_holes(
            gray, roi_outer, roi_holes, canvas, contours,
            tmpl_hole_count, tmpl_hole_ratios, mold_size)

        if hole_score < 0:
            return _fail(4,
                f"hole mismatch got={len(roi_holes)} expected={tmpl_hole_count}",
                {"shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "roi_canvas":  canvas})

        # ── Step 5 : Shape score (confidence) ────────────────────────
        confidence = self._compute_shape_score(canvas, contours, tmpl)

        conf_thr = max(0.0, FONT_CONFIDENCE_MIN - 0.05) if is_retry \
                   else FONT_CONFIDENCE_MIN
        if confidence < conf_thr:
            return _fail(5,
                f"low shape_score={confidence:.3f} < {conf_thr:.3f}",
                {"confidence":  confidence,
                 "shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "roi_canvas":  canvas})

        return {
            "pass":        True,
            "confidence":  round(confidence, 4),
            "shift_px":    shift_px,
            "shift_ratio": shift_ratio,
            "reason":      "OK",
            "defect_step": 0,
            "roi_canvas":  canvas,
            "roi_thresh":  roi_thresh,
            "tmpl_canvas": tmpl.get("canvas"),
            "orig_roi_w":  orig_roi_w,
            "orig_roi_h":  orig_roi_h,
            "dirty":       _no_dirty,
        }


# =========================================================
# C.  ResultAnnotator  — standalone, no Qt dependency
# =========================================================
class ResultAnnotator:
    """
    Draws inspection annotations onto a BGR display image.
    All methods are static — no instance state.

    Each method takes the display image as first argument and
    modifies it in-place.  Returns None.

    Colour scheme
    -------------
      Frame box  : (0, 224, 255)  cyan dashed
      Mold box   : (0, 180, 200)  dim-cyan dashed
      Pass box   : (0, 200, 0)    green solid
      Fail box   : (0, 0, 200)    red solid
    """

    COLOR_FRAME = (0, 224, 255)
    COLOR_MOLD  = (0, 180, 200)
    COLOR_PASS  = (0, 200, 0)
    COLOR_FAIL  = (0, 0, 200)
    COLOR_OCR   = (0, 255, 255)   # yellow — OCR read label

    # ---- Frame -------------------------------------------------------

    @staticmethod
    def draw_frame(display: np.ndarray,
                   fcx:     int,
                   fcy:     int,
                   fw:      int,
                   fh:      int,
                   f_idx:   int,
                   fscore:  float,
                   frame_id: str = ""):
        """
        Draw dashed frame bounding box + score label.

        Input : display (BGR ndarray, modified in-place),
                fcx/fcy (frame centre), fw/fh (matched size),
                f_idx (0-based frame index), fscore (TM score 0–1),
                frame_id (layout id string, e.g. "F1")
        """
        ih, iw = display.shape[:2]
        fx1 = max(0,    fcx - fw // 2)
        fy1 = max(0,    fcy - fh // 2)
        fx2 = min(iw-1, fx1 + fw)
        fy2 = min(ih-1, fy1 + fh)
        cv2_draw_dashed_rect(
            display, (fx1, fy1), (fx2, fy2),
            ResultAnnotator.COLOR_FRAME, 1)
        lbl = f"{frame_id} {fscore:.2f}" if frame_id else f"F{f_idx+1} {fscore:.2f}"
        cv2.putText(display,
                    lbl,
                    (fx1 + 2, fy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    ResultAnnotator.COLOR_FRAME, 1)

    # ---- Mold --------------------------------------------------------

    @staticmethod
    def draw_mold(display:    np.ndarray,
                  acx:        int,
                  acy:        int,
                  aw:         int,
                  ah:         int,
                  f_idx:      int,
                  mold_label: str,
                  elapsed_ms: float = 0.0):
        """
        Draw dashed mold bounding box + label.

        Input : display (BGR ndarray, modified in-place),
                acx/acy (mold centre), aw/ah (mold size),
                f_idx (0-based), mold_label ("A" or "B"),
                elapsed_ms (optional timing label)
        """
        ih, iw = display.shape[:2]
        ax1 = max(0,    acx - aw // 2)
        ay1 = max(0,    acy - ah // 2)
        ax2 = min(iw-1, ax1 + aw)
        ay2 = min(ih-1, ay1 + ah)
        cv2_draw_dashed_rect(
            display, (ax1, ay1), (ax2, ay2),
            ResultAnnotator.COLOR_MOLD, 1)
        
        if mold_label == "A":
            f_idx = f_idx*2 +1
        else:
            f_idx = f_idx*2 +2
        cv2.putText(display,
                    f"F{f_idx}[{elapsed_ms:.1f}ms]",
                    (ax1 + 2, ay1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    ResultAnnotator.COLOR_MOLD, 1)
        

    # ---- Last-lot flag -----------------------------------------------
    @staticmethod
    def draw_last_lot_flag(display:   np.ndarray,
                           chip_cols: int,
                           acx:       int,
                           acy:       int,
                           aw:        int,
                           ah:        int):
        """
        Draw a prominent amber "LAST LOT" banner on the mold area.

        Input : display (BGR ndarray, modified in-place),
                chip_cols — number of leading columns that have chips (1 or 2),
                acx/acy — mold centre, aw/ah — mold size.
        """
        ih, iw = display.shape[:2]
        ax1 = max(0,    acx - aw // 2)
        ay1 = max(0,    acy - ah // 2)
        ax2 = min(iw-1, ax1 + aw)
        ay2 = min(ih-1, ay1 + ah)

        # Amber semi-transparent fill over mold area
        overlay = display.copy()
        cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (0, 140, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.22, display, 0.78, 0, display)

        # Solid amber border
        cv2.rectangle(display, (ax1, ay1), (ax2, ay2), (0, 165, 255), 2)

        # Label centred on mold
        lbl   = f"LAST LOT ({chip_cols}/3 col)"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscl  = 0.45
        thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = acx - tw // 2
        ty = acy + th // 2
        cv2.rectangle(display, (tx - 3, ty - th - bl), (tx + tw + 3, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 165, 255), thick)

    # ---- Ignored frame (last-lot empty column) ----------------------
    @staticmethod
    def draw_ignored_frame(display: np.ndarray,
                           acx: int, acy: int,
                           fw: int,  fh: int):
        """
        'IGNORED' text with black background, centred on an empty-column frame.
        No area overlay — the frame image remains visible beneath.
        """
        lbl  = "IGNORED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscl = 0.40
        thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = acx - tw // 2
        ty = acy + th // 2
        cv2.rectangle(display, (tx - 2, ty - th - bl), (tx + tw + 2, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, (160, 160, 160), thick, cv2.LINE_AA)

    # ---- Drop-work label (no-lead / Drop by Pin) --------------------
    @staticmethod
    def draw_drop_label(display: np.ndarray,
                        acx: int, acy: int,
                        aw: int, ah: int,
                        ic_id: str = ""):
        """Draw 'DROP' label centred on a mold whose leads are absent."""
        ih, iw = display.shape[:2]
        ax1 = max(0,    acx - aw // 2)
        ay1 = max(0,    acy - ah // 2)
        ax2 = min(iw-1, ax1 + aw)
        ay2 = min(ih-1, ay1 + ah)

        overlay = display.copy()
        cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (180, 180, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
        cv2.rectangle(display, (ax1, ay1), (ax2, ay2), (0, 200, 200), 1)

        lbl   = f"{ic_id} DROP" if ic_id else "DROP"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscl  = 0.55
        thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = acx - tw // 2
        ty = acy + th // 2
        cv2.rectangle(display, (tx - 3, ty - th - bl), (tx + tw + 3, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 200, 200), thick, cv2.LINE_AA)

    # ---- Missing frame (no TM match in layout ROI) ------------------
    @staticmethod
    def draw_missing_frame(display: np.ndarray,
                           rx: int, ry: int,
                           rw: int, rh: int,
                           frame_id: str = ""):
        """Draw MISSING annotation over a pre-defined frame ROI that had no TM hit."""
        ih, iw = display.shape[:2]
        x1 = max(0,    rx);      y1 = max(0,    ry)
        x2 = min(iw-1, rx + rw); y2 = min(ih-1, ry + rh)
        cv2_draw_dashed_rect(display, (x1, y1), (x2, y2), (0, 0, 180), 2)
        lbl  = f"{frame_id} MISSING" if frame_id else "MISSING"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscl = 0.50; thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = (x1 + x2) // 2 - tw // 2
        ty = (y1 + y2) // 2 + th // 2
        cv2.rectangle(display, (tx - 2, ty - th - bl), (tx + tw + 2, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 80, 255), thick, cv2.LINE_AA)

    # ---- Last-lot image-level banner --------------------------------
    @staticmethod
    def draw_last_lot_image_flag(display: np.ndarray,
                                 chip_cols: int,
                                 total_cols: int):
        """
        Amber banner at the top of the full display image for
        the image-level last-lot condition.

        chip_cols  — frame-columns that have physical chips (leading)
        total_cols — total frame-columns detected in this image
        """
        ih, iw = display.shape[:2]
        bar_h = 30
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (iw, bar_h), (0, 140, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)
        # solid border at bottom of bar
        cv2.line(display, (0, bar_h), (iw, bar_h), (0, 165, 255), 2)

        lbl  = f"*** LAST LOT IMAGE — {chip_cols}/{total_cols} frame-col(s) filled ***"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscl = 0.55
        thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = iw // 2 - tw // 2
        ty = bar_h // 2 + th // 2
        cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 255, 255), thick, cv2.LINE_AA)

    # ---- Letter ------------------------------------------------------
    @staticmethod
    def draw_letter(display: np.ndarray,
                    result:  dict):
        passed      = result["pass"]
        letter      = result["letter"]
        confidence  = result["confidence"]
        lx1         = result["lx1"]
        ly1         = result["ly1"]
        lx2         = result["lx2"]
        ly2         = result["ly2"]
        ocr_char    = result.get("ocr_char", "?")
        ocr_conf    = result.get("ocr_conf",  0.0)
        roi_thresh  = result.get("roi_thresh")
        tmpl_canvas = result.get("tmpl_canvas")

        col      = ResultAnnotator.COLOR_PASS if passed else ResultAnnotator.COLOR_FAIL
        cell_w   = lx2 - lx1
        cell_h   = ly2 - ly1

        # ── Dirty region red overlay ──────────────────────────────
        if not passed and cell_w > 0 and cell_h > 0:
            dirty_info = result.get("dirty", {})
            d_type     = dirty_info.get("type", "none")
            if d_type == "foreign_object":
                dmap = dirty_info.get("extra_map")
            elif d_type == "missing_stroke":
                dmap = dirty_info.get("missing_map")
            else:
                dmap = None

            if d_type in ("foreign_object", "missing_stroke"):
                cell_roi  = display[ly1:ly2, lx1:lx2]
                red_layer = np.zeros_like(cell_roi)
                if dmap is not None and dmap.size > 0:
                    dmap_rs = cv2.resize(dmap, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
                    red_layer[dmap_rs > 0] = (0, 0, 220)
                else:
                    red_layer[:] = (0, 0, 100)   # fallback: dim-red tint on whole cell
                cell_roi[:] = cv2.addWeighted(cell_roi, 0.55, red_layer, 0.45, 0)

        # ── Extracted edges — threshold contours in pass/fail color ──
        if roi_thresh is not None and roi_thresh.size > 0 and cell_w > 0 and cell_h > 0:
            rrs = cv2.resize(roi_thresh, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            rcnts, _ = cv2.findContours(rrs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display[ly1:ly2, lx1:lx2], rcnts, -1, col, 1)

        # ── Label with black background ───────────────────────────
        if letter:
            lbl = f"{letter}: {confidence:.2f}"
        else:
            lbl = f"!{ocr_char}: {ocr_conf:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.30
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(lbl, font, font_scale, thickness)
        tx = lx1
        ty = max(ly1 - 2, th + baseline)
        cv2.rectangle(display,
                      (tx, ty - th - baseline),
                      (tx + tw, ty + baseline),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, font_scale, col, thickness)
                
    
# =========================================================
# INSPECTION CONTROLLER
# =========================================================
class InspectionController:
    """
    Owns the inspection engine, template store, template cache, and results.

    Frame recipe  (pin_recipe.json)
    --------------------------------
    Stores FRAME + MOLD_A + MOLD_B as one unit.
    save_frame_recipe(image_bgr, frame_rect, mold_a_rect, mold_b_rect) -> bool
    has_frame_recipe() -> bool
    load_frame_recipe() -> dict
    get_frame_template() -> dict   (canvas ready for TM search)
    get_mold_offsets()  -> (mold_a_dict, mold_b_dict)

    Font templates  (templates/<NAME>_template.json)
    -------------------------------------------------
    save_font(name, roi_bgr, roi_rect, parent_widget) -> bool

    Cache  (load once before inspection loop)
    -----------------------------------------
    load_cache(names) -> list[str failed]

    Inspection stub
    ---------------
    run(image_bgr, mask) -> list
    """

    RECIPE_FILE = "pin_recipe.json"

    def __init__(self, sm: SettingsManager):
        self._sm    = sm
        self._ct    = ContourTemplate()
        self._eng   = InspectionEngine()
        self.cache:          dict[str, dict] = {}
        self.results:        list[dict]      = []
        self._ocr_templates: dict[str, dict] = {}
        self.yolo = YOLOMoldDetector()

    # ---- internal: encode one ROI section for the recipe ----
    @staticmethod
    def _encode_section(image_bgr: np.ndarray,
                        rect: QtCore.QRect,
                        offset: tuple = None) -> dict:
        """
        Qt-aware wrapper around ContourTemplate.encode_roi().
        Converts QRect → (x, y, w, h), clips to image bounds,
        crops the ROI, then delegates all extraction/encode logic.
        offset = (dx, dy) relative to frame centre — passed through to encode_roi.
        """
        ih, iw = image_bgr.shape[:2]
        x = max(0, rect.x())
        y = max(0, rect.y())
        w = min(rect.width(),  iw - x)
        h = min(rect.height(), ih - y)

        src     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
                  if image_bgr.ndim == 3 else image_bgr
        roi     = src[y:y + h, x:x + w].copy()

        return ContourTemplate.encode_roi(roi, (x, y, w, h), offset)
    
    # ---- internal: decode one section back to numpy ----
    @staticmethod
    def _decode_section(sec: dict) -> dict:
        sec = dict(sec)
        sec["contours"] = [np.array(c, dtype=np.int32) for c in sec["contours"]]
        raw             = base64.b64decode(sec["canvas_b64"])
        arr             = np.frombuffer(raw, dtype=np.uint8)
        sec["canvas"]   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return sec

    # ---- frame recipe save ----
    def save_frame_recipe(self,
                          image_bgr:    np.ndarray,
                          mold_a_rect:  QtCore.QRect,
                          mold_b_rect:  QtCore.QRect,
                          grid_letters: list) -> bool:
        """
        Build and save v6 pin_recipe.json from a detected mold pair (A above B).

        Layout:
          [pin_a] [A]
          [anchor]           ← diagonal midpoint between A and B
          [pin_b] [B]

        Anchor: centre X = mold_a_cx - aw*0.85, width = aw*0.7,
                centre Y = midpoint(A_centre_Y, B_centre_Y)
        Pin A/B: width = aw*0.20, height = ah, gap = aw*0.05 from mold left edge,
                 each at their respective mold's centre Y.

        grid_letters : list of exactly 9 strings (empty = skip), row-major.
        Returns True on success.
        """
        try:
            ih, iw = image_bgr.shape[:2]

            ax, ay = mold_a_rect.x(), mold_a_rect.y()
            aw, ah = mold_a_rect.width(), mold_a_rect.height()
            by = mold_b_rect.y()

            a_cx  = ax + aw // 2
            a_cy  = ay + ah // 2
            b_cy  = by + ah // 2

            # ── Anchor rect — mold width, right edge flush with mold A left edge ──
            anc_w   = aw
            anc_h   = ah
            anc_x   = max(0, ax - aw)
            anc_y   = max(0, (a_cy + b_cy) // 2 - anc_h // 2)
            anc_w   = min(anc_w, iw - anc_x)
            anc_h   = min(anc_h, ih - anc_y)
            anc_cx  = anc_x + anc_w // 2
            anc_cy  = anc_y + anc_h // 2
            anchor_rect = QtCore.QRect(anc_x, anc_y, anc_w, anc_h)

            mold_a_shift = [a_cx - anc_cx,  a_cy - anc_cy]
            mold_b_shift = [a_cx - anc_cx,  b_cy - anc_cy]

            # ── Per-mold pin ROI (left of each mold, narrow strip) ──────
            PIN_GAP_RATIO = 0.05
            PIN_W_RATIO   = 0.20
            pin_w  = max(4, int(aw * PIN_W_RATIO))
            pin_cx = ax - int(aw * PIN_GAP_RATIO) - pin_w // 2
            pin_lx = max(0, pin_cx - pin_w // 2)

            pin_a_rect = QtCore.QRect(pin_lx, max(0, ay),
                                      min(pin_w, iw - pin_lx),
                                      min(ah, ih - ay))
            pin_b_rect = QtCore.QRect(pin_lx, max(0, by),
                                      min(pin_w, iw - pin_lx),
                                      min(ah, ih - by))

            anchor_sec = self._encode_section(image_bgr, anchor_rect)
            pin_a_sec  = self._encode_section(image_bgr, pin_a_rect)
            pin_b_sec  = self._encode_section(image_bgr, pin_b_rect)

            recipe = {
                "version":      6,
                "mold_size":    [aw, ah],
                "mold_a_shift": mold_a_shift,
                "mold_b_shift": mold_b_shift,
                "anchor":       anchor_sec,
                "pin_a":        pin_a_sec,
                "pin_b":        pin_b_sec,
                "grid_letters": grid_letters,
            }
            with open(self.RECIPE_FILE, "w") as f:
                json.dump(recipe, f)

            self.cache.pop("ANCHOR", None)
            return True

        except Exception as e:
            print(f"[Controller] save_frame_recipe error: {e}")
            return False
        
    # ---- frame recipe queries ----
    def has_frame_recipe(self) -> bool:
        return os.path.exists(self.RECIPE_FILE)

    def load_frame_recipe(self) -> dict:
        with open(self.RECIPE_FILE, "r") as f:
            raw = json.load(f)
        version = raw.get("version", 1)
        if version < 6:
            raise ValueError(
                f"pin_recipe.json is version {version} — "
                f"please re-save the frame template (v6 required).")
        return {
            "version":      version,
            "mold_size":    raw["mold_size"],
            "mold_a_shift": raw["mold_a_shift"],
            "mold_b_shift": raw["mold_b_shift"],
            "anchor":       self._decode_section(raw["anchor"]),
            "pin_a":        self._decode_section(raw["pin_a"]),
            "pin_b":        self._decode_section(raw["pin_b"]),
            "grid_letters": raw.get("grid_letters", [""] * 9),
        }

    def get_frame_template(self) -> dict:
        """Return anchor section (canvas ready for TM search)."""
        return self.load_frame_recipe()["anchor"]

    def get_mold_offsets(self) -> tuple:
        """Return (mold_a_shift, mold_b_shift) as [dx, dy] lists."""
        recipe = self.load_frame_recipe()
        return recipe["mold_a_shift"], recipe["mold_b_shift"]

    # ---- frame layout (frame_layout.json) ----
    LAYOUT_FILE = "frame_layout.json"

    def has_frame_layout(self) -> bool:
        return os.path.exists(self.LAYOUT_FILE)

    def load_frame_layout(self) -> dict:
        with open(self.LAYOUT_FILE, "r") as f:
            return json.load(f)

    def save_frame_layout(self, layout: dict) -> bool:
        try:
            with open(self.LAYOUT_FILE, "w") as f:
                json.dump(layout, f, indent=2)
            return True
        except Exception as e:
            print(f"[Controller] save_frame_layout error: {e}")
            return False

    # ---- font template save ----
    def save_font(self, name, roi_bgr, roi_rect,
                parent_widget, mold_size: int = 150) -> bool:
        try:
            self._ct.save(name, roi_bgr, roi_rect, mold_size=mold_size)
            self.cache.pop(name, None)
            TemplatePreviewDialog(roi_bgr, name,  mold_size=mold_size, parent=parent_widget).exec_()
            return True
        except Exception as e:
            print(f"[Controller] Font '{name}' save error: {e}")
            return False

    def list_fonts(self) -> list:
        return self._ct.list_templates()

        # ---- cache ----
    def load_cache(self, names: list) -> list:
        failed = []
        for name in names:
            key = name.upper()
            try:
                self.cache[key] = self._ct.load(key)
            except Exception as e:
                print(f"[Controller] Cache load '{key}': {e}")
                failed.append(key)
        return failed
    
    def prepare(self, grid_letters: list) -> list:
        """
        Pre-flight: load recipe + layout + cache all templates for grid_letters.
        Call once before the inspection loop starts.

        Returns list of missing template names (empty = all good).
        Stores loaded recipe and layout in self._active_recipe / _active_layout.
        """
        self._active_recipe  = None
        self._active_grid    = []
        self._active_layout  = None

        if not self.has_frame_recipe():
            return ["__NO_RECIPE__"]

        try:
            recipe = self.load_frame_recipe()
        except Exception as e:
            print(f"[Prepare] Recipe load error: {e}")
            return ["__RECIPE_ERROR__"]

        if not self.has_frame_layout():
            return ["__NO_LAYOUT__"]

        try:
            layout = self.load_frame_layout()
        except Exception as e:
            print(f"[Prepare] Layout load error: {e}")
            return ["__LAYOUT_ERROR__"]

        active_grid = [l for l in grid_letters if l]
        if not active_grid:
            return ["__NO_GRID__"]

        unique = list(set(active_grid))
        failed = self.load_cache(unique)
        if failed:
            return failed

        # Load ALL saved templates for OCR — not just the active grid chars
        self._ocr_templates = {}
        for name in self._ct.list_templates():
            try:
                self._ocr_templates[name] = self._ct.load(name)
            except Exception as e:
                print(f"[OCR] Skipped template '{name}': {e}")

        # Patch active grid into recipe — used by run() without re-loading
        recipe["grid_letters"] = grid_letters
        self._active_recipe    = recipe
        self._active_grid      = grid_letters
        self._active_layout    = layout
        return []

    def set_run_params(self, pin_params: dict):
        """Store search params once before the loop. Used by run() per frame."""
        self._active_pin_params = pin_params
        
    # ---- last-lot detection (image-level, called after all frames processed) ----
    @staticmethod
    def _check_last_lot_image(matches: list,
                              all_results: list,
                              col_snap: int) -> tuple:
        """
        Post-inspection last-lot check across all detected frame-columns.

        Condition (using LAST_LOT_CHIP_FRAME_COLS = N):
          • Frame-columns 0 … N-1 must have physical chips (any pass/fail).
          • Frame-columns N … end must have frames detected but NO chip
            contours (defect_step == 1 on every letter-slot, or no letter-slots).
          • Total detected frame-columns must be > N.

        Physical chip = letter assigned AND defect_step != 1
                        AND reason != "dropped".

        Returns (is_last_lot: bool, chip_cols: int, total_cols: int).
        """
        if not matches:
            return False, 0, 0

        from collections import defaultdict
        cg_to_fidx: dict = defaultdict(list)
        for f_idx, (anc_cx, *_) in enumerate(matches):
            cg = round(anc_cx / max(col_snap, 1))
            cg_to_fidx[cg].append(f_idx)

        sorted_cg = sorted(cg_to_fidx.keys())
        n_cols    = len(sorted_cg)
        n_chip    = LAST_LOT_CHIP_FRAME_COLS

        if n_cols <= n_chip:
            return False, 0, n_cols, {}

        # Build reverse map: f_idx → 0-based column index
        fidx_to_col: dict = {}
        for col_i, cg in enumerate(sorted_cg):
            for fi in cg_to_fidx[cg]:
                fidx_to_col[fi] = col_i

        # Mark columns that have at least one physically-present chip
        col_has_chip = [False] * n_cols
        for r in all_results:
            fi = r.get("frame_idx", 0) - 1          # frame_idx is 1-based
            if fi not in fidx_to_col:
                continue
            if (r.get("letter", "") != ""
                    and r.get("reason") != "dropped"
                    and r.get("defect_step", 0) != 1):
                col_has_chip[fidx_to_col[fi]] = True

        # First n_chip columns must ALL have chips
        for c in range(n_chip):
            if not col_has_chip[c]:
                return False, 0, n_cols, {}

        # Remaining columns must ALL be empty (no chip contours)
        for c in range(n_chip, n_cols):
            if col_has_chip[c]:
                return False, 0, n_cols, {}

        return True, n_chip, n_cols, fidx_to_col

    # ---- inspection pipeline ----
    def run(self,
        image_bgr: np.ndarray,
        mask:      np.ndarray = None) -> "InspectionResult":
        self.results = []
        src     = image_bgr if image_bgr.ndim == 2 \
                else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        empty   = InspectionResult(display=display)

        recipe = getattr(self, "_active_recipe", None)
        layout = getattr(self, "_active_layout", None)
        if recipe is None or layout is None:
            print("[Pipeline] prepare() not called — no recipe/layout loaded.")
            return empty

        pin_params = getattr(self, "_active_pin_params", None)
        if pin_params is None:
            print("[Pipeline] set_run_params() not called — using defaults.")
            pin_params = {}

        t0_total = time.perf_counter()

        try:
            anchor_tmpl = recipe["anchor"]
        except KeyError:
            print("[Pipeline] Recipe missing 'anchor' key — re-save template.")
            return empty

        frame_results = self._step1_find_frames(
            image_bgr, anchor_tmpl, pin_params, layout)

        if not frame_results:
            print("[Pipeline] Frame layout is empty.")
            return empty

        ih, iw = image_bgr.shape[:2]

        # Build a fake matches list (cx, cy, score, fw, fh, scale) for last-lot
        # using expected ROI centres for MISSING frames so column grouping still works.
        fake_matches = []

        for f_idx, fentry in enumerate(frame_results):
            fid   = fentry["id"]
            found = fentry["found"]
            anc_cx, anc_cy = fentry["cx"], fentry["cy"]
            fscore = fentry["score"]
            fw, fh = fentry["fw"], fentry["fh"]

            fake_matches.append((anc_cx, anc_cy, fscore, fw, fh, 1.0))

            if not found:
                # Always draw annotation for missing frames
                rx, ry, rw, rh = fentry["roi"]
                ResultAnnotator.draw_missing_frame(display, rx, ry, rw, rh, fid)
                # Add placeholder results so every frame appears in the result set
                grid_letters = recipe.get("grid_letters", [""] * 9)
                for m_idx, mold_label in enumerate(["A", "B"]):
                    ic_num = f_idx * 2 + m_idx + 1
                    ic_id  = f"{fid}-{mold_label}"
                    for slot_idx, letter in enumerate(grid_letters):
                        if not letter:
                            continue
                        self.results.append({
                            "frame_idx":   f_idx + 1,
                            "frame_id":    fid,
                            "mold":        mold_label,
                            "slot":        slot_idx,
                            "letter":      letter,
                            "pass":        False,
                            "confidence":  0.0,
                            "shift_px":    0.0,
                            "shift_ratio": 0.0,
                            "defect_step": 0,
                            "reason":      "missing_frame",
                            "ic_num":      ic_num,
                            "cell_cx":     anc_cx,
                            "cell_cy":     anc_cy,
                            "elapsed_ms":  0.0,
                            "lx1": 0, "ly1": 0, "lx2": 0, "ly2": 0,
                            "roi_canvas":  None,
                        })
                continue

            ResultAnnotator.draw_frame(display, anc_cx, anc_cy, fw, fh, f_idx, fscore, fid)

            mold_areas = self._step2_locate_molds(
                recipe, anc_cx, anc_cy, 1.0, iw, ih)

            t0_frame = time.perf_counter()

            for m_idx, area in enumerate(mold_areas):
                acx, acy  = area["cx"],  area["cy"]
                aw,  ah   = area["w"],   area["h"]
                mold_size = min(aw, ah)
                ic_num    = f_idx * 2 + m_idx + 1   # A=2i+1, B=2i+2 (1-based)
                ic_id     = f"{fid}-{area['label']}"

                pin_key       = "pin_a" if area["label"] == "A" else "pin_b"
                leads_present = self._eng.check_pin_presence(
                    src, anc_cx, anc_cy,
                    recipe[pin_key], recipe["anchor"], iw, ih)

                t0_mold = time.perf_counter()
                letter_results = self._step3_inspect_fonts(
                    src, display, acx, acy,
                    area["grid"],
                    f_idx, area["label"], iw, ih,
                    mold_size     = mold_size,
                    mold_w        = aw,
                    mold_h        = ah,
                    leads_present = leads_present,
                    ic_id         = ic_id)
                mold_ms = (time.perf_counter() - t0_mold) * 1000

                for r in letter_results:
                    r["ic_num"]   = ic_num
                    r["frame_id"] = fid

                ResultAnnotator.draw_mold(display, acx, acy, aw, ah,
                                          f_idx, area["label"],
                                          elapsed_ms=mold_ms)

                self.results.extend(letter_results)

            frame_ms = (time.perf_counter() - t0_frame) * 1000

            for r in self.results:
                if r.get("frame_idx") == f_idx + 1 and "frame_ms" not in r:
                    r["frame_ms"] = round(frame_ms, 1)

        # ── Image-level last-lot check ──
        col_snap = max(1, anchor_tmpl.get("canvas_w", 60) // 2)
        img_ll, img_chip_cols, img_total_cols, fidx_to_col = \
            self._check_last_lot_image(fake_matches, self.results, col_snap)

        if img_ll:
            for r in self.results:
                r["last_lot"]      = True
                r["last_lot_cols"] = img_chip_cols
                fi = r.get("frame_idx", 0) - 1
                if fidx_to_col.get(fi, -1) >= img_chip_cols:
                    r["ignored"] = True

            for f_idx, fentry in enumerate(frame_results):
                if fidx_to_col.get(f_idx, -1) >= img_chip_cols:
                    anc_cx, anc_cy = fentry["cx"], fentry["cy"]
                    fw, fh = fentry["fw"], fentry["fh"]
                    ResultAnnotator.draw_ignored_frame(display, anc_cx, anc_cy, fw, fh)

            ResultAnnotator.draw_last_lot_image_flag(
                display, img_chip_cols, img_total_cols)
            print(f"[Pipeline] LAST LOT — {img_chip_cols}/{img_total_cols} "
                  f"frame-col(s) filled")

        total_ms = round((time.perf_counter() - t0_total) * 1000, 1)
        active      = [r for r in self.results if not r.get("ignored")]
        passed      = sum(1 for r in active if r["pass"])
        any_ll      = any(r.get("last_lot") for r in self.results)
        max_ll_cols = max((r.get("last_lot_cols", 0) for r in self.results), default=0)
        result = InspectionResult(
            display        = display,
            results        = self.results,
            passed         = passed,
            total          = len(active),
            elapsed_ms     = total_ms,
            last_lot       = any_ll,
            last_lot_cols  = max_ll_cols,
        )
        result.total_ms = total_ms
        return result

    def _step1_find_frames(self, image_bgr, anchor_tmpl, pin_params, layout):
        """
        For each frame in layout (F1→FN), crop the image at the pre-defined ROI
        and run a local TM to confirm presence.

        Returns list of dicts:
          { id, roi, found, cx, cy, score, fw, fh }
        Order is always the layout order — no sorting, no index reassignment.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
            if image_bgr.ndim == 3 else image_bgr.copy()

        tmpl_canvas = anchor_tmpl.get("canvas")
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return []

        th, tw = tmpl_canvas.shape[:2]
        ih, iw = gray.shape[:2]
        score_thr = pin_params.get("score_thr", 0.75)
        results = []

        for fentry in layout.get("frames", []):
            fid   = fentry["id"]
            x, y, w, h = fentry["roi"]
            x1 = max(0, x);      y1 = max(0, y)
            x2 = min(iw, x + w); y2 = min(ih, y + h)
            crop = gray[y1:y2, x1:x2]

            found = False
            cx    = x1 + (x2 - x1) // 2
            cy    = y1 + (y2 - y1) // 2
            score = 0.0

            if crop.shape[0] >= th and crop.shape[1] >= tw:
                result_map = cv2.matchTemplate(crop, tmpl_canvas, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result_map)
                if max_val >= score_thr:
                    found = True
                    score = float(max_val)
                    mx, my = max_loc
                    cx = x1 + mx + tw // 2
                    cy = y1 + my + th // 2

            results.append({
                "id":    fid,
                "roi":   [x, y, w, h],
                "found": found,
                "cx":    cx,
                "cy":    cy,
                "score": score,
                "fw":    tw,
                "fh":    th,
            })

        return results

    def _step2_locate_molds(self, recipe, anc_cx, anc_cy, fscale, iw, ih):
        """
        Derive mold A and mold B centres from anchor hit + stored shifts.
        Returns list of two area dicts (A first, then B).
        """
        aw, ah     = recipe["mold_size"]
        aw         = int(round(aw * fscale))
        ah         = int(round(ah * fscale))
        grid       = recipe.get("grid_letters", [])
        areas      = []

        for label, shift_key in [("A", "mold_a_shift"), ("B", "mold_b_shift")]:
            dx, dy = recipe[shift_key]
            acx = anc_cx + int(round(dx * fscale))
            acy = anc_cy + int(round(dy * fscale))
            acx = max(aw // 2, min(iw - aw // 2, acx))
            acy = max(ah // 2, min(ih - ah // 2, acy))
            areas.append({
                "label":    label,
                "cx":       acx,
                "cy":       acy,
                "w":        aw,
                "h":        ah,
                "fscale":   fscale,
                "grid":     grid,
            })
        return areas

    def _step3_inspect_fonts(self, image_bgr, display,
                         acx, acy, grid_letters,
                         f_idx, mold_label,
                         iw, ih,
                         mold_size:     int  = 150,
                         mold_w:        int  = 150,
                         mold_h:        int  = 150,
                         leads_present: bool = True,
                         ic_id:         str  = "") -> list:
        results = []

        # ── Grid geometry (tunable) ───────────────────────────────
        g_scale  = float(self._sm.get("grid_scale"))
        g_x_frac = float(self._sm.get("grid_x_frac"))
        g_y_frac = float(self._sm.get("grid_y_frac"))
        grid_cx  = acx + int(mold_w * g_x_frac)
        grid_cy  = acy + int(mold_h * g_y_frac)
        cell_w   = int(mold_w * g_scale / 3)
        cell_h   = int(mold_h * g_scale / 3)
        roi_w    = int(cell_w * 1.2)
        roi_h    = int(cell_h * 1.2)

        if not leads_present:
            # Work dropped — pass all slots, draw single mold-level label
            for slot_idx, letter in enumerate(grid_letters):
                if not letter:
                    continue
                row = slot_idx // 3
                col = slot_idx  % 3
                dx  = (col - 1) * cell_w
                dy  = (row - 1) * cell_h
                results.append({
                    "frame_idx":   f_idx + 1,
                    "mold":        mold_label,
                    "slot":        slot_idx,
                    "letter":      letter,
                    "pass":        True,
                    "confidence":  1.0,
                    "shift_px":    0.0,
                    "shift_ratio": 0.0,
                    "defect_step": 0,
                    "reason":      "dropped",
                    "cell_cx":     grid_cx + dx,
                    "cell_cy":     grid_cy + dy,
                    "elapsed_ms":  0.0,
                    "lx1": 0, "ly1": 0, "lx2": 0, "ly2": 0,
                    "roi_canvas":  None,
                })

            ResultAnnotator.draw_drop_label(display, acx, acy, mold_w, mold_h, ic_id=ic_id)
            return results

        # ── Normal font inspection ────────────────────────────────
        for slot_idx, letter in enumerate(grid_letters):
            letter = letter.upper() if letter else ""

            row = slot_idx // 3
            col = slot_idx  % 3

            dx = (col - 1) * cell_w
            dy = (row - 1) * cell_h

            cell_cx = grid_cx + dx
            cell_cy = grid_cy + dy

            half_w = roi_w // 2
            half_h = roi_h // 2

            lx1 = max(0,  cell_cx - half_w)
            ly1 = max(0,  cell_cy - half_h)
            lx2 = min(iw, lx1 + roi_w)
            ly2 = min(ih, ly1 + roi_h)

            if lx2 <= lx1 or ly2 <= ly1:
                continue

            roi = image_bgr[ly1:ly2, lx1:lx2]

            # ── Extract contours once — reused by defect check ───────
            gray_slot = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) \
                        if roi.ndim == 3 else roi
            slot_contours, slot_canvas, slot_clean_binary, slot_others = \
                self._eng._check_presence(gray_slot, mold_size)

            # ── Empty slot: foreign-object check first, then unexpected-mark ──
            if not letter:
                # Re-filter: tophat removes large reflection blobs, keeps thin strokes
                slot_contours, slot_canvas, slot_clean_binary, slot_others = \
                    InspectionEngine._suppress_large_blobs(slot_clean_binary, mold_size)
                slot_roi_area = (ly2 - ly1) * (lx2 - lx1)
                if slot_contours:
                    main_area  = cv2.contourArea(slot_contours[0])
                    area_ratio = main_area / max(slot_roi_area, 1)
                    if area_ratio >= ANOMALY_MIN_AREA_RATIO \
                            and not self._eng._is_laser_mark(slot_canvas):
                        results.append({
                            "frame_idx":   f_idx + 1,
                            "mold":        mold_label,
                            "slot":        slot_idx,
                            "letter":      "",
                            "pass":        False,
                            "confidence":  0.0,
                            "shift_px":    0.0,
                            "shift_ratio": 0.0,
                            "defect_step": 0,
                            "defect_type": "foreign_object_empty",
                            "reason":      f"foreign_object_in_empty_slot(area={area_ratio:.3f})",
                            "ocr_char":    "",
                            "ocr_conf":    0.0,
                            "cell_cx":     cell_cx,
                            "cell_cy":     cell_cy,
                            "elapsed_ms":  0.0,
                            "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                            "roi_canvas":  slot_canvas,
                            "dirty":       {"detected": True, "type": "foreign_object",
                                            "area_ratio": round(area_ratio, 4)},
                        })
                        ResultAnnotator.draw_letter(display, results[-1])
                        continue

                ocr_char, ocr_conf = self._eng.ocr_identify(
                    slot_canvas, slot_contours, "", self._ocr_templates)

                if not slot_contours or ocr_char == "?":
                    continue

                if self._eng._is_laser_mark(slot_canvas):
                    results.append({
                        "frame_idx":   f_idx + 1,
                        "mold":        mold_label,
                        "slot":        slot_idx,
                        "letter":      "",
                        "pass":        False,
                        "confidence":  ocr_conf,
                        "shift_px":    0.0,
                        "shift_ratio": 0.0,
                        "defect_step": 0,
                        "defect_type": "unexpected_mark",
                        "reason":      f"unexpected_mark:{ocr_char}",
                        "ocr_char":    ocr_char,
                        "ocr_conf":    ocr_conf,
                        "cell_cx":     cell_cx,
                        "cell_cy":     cell_cy,
                        "elapsed_ms":  0.0,
                        "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                        "roi_canvas":  slot_canvas,
                        "dirty":       {"detected": False, "type": "none", "area_ratio": 0.0},
                    })
                    ResultAnnotator.draw_letter(display, results[-1])
                continue

            # ── Letter slot: defect check ─────────────────────────────
            tmpl = self.cache.get(letter)
            if tmpl is None:
                continue

            exp_dx = dx
            exp_dy = dy

            tmpl_mold_size = tmpl.get("mold_size", mold_size)

            t0 = time.perf_counter()
            res = self._eng.compare_roi(
                roi, tmpl,
                exp_dx      = exp_dx,
                exp_dy      = exp_dy,
                mold_cx     = acx,
                mold_cy     = acy,
                mold_size   = tmpl_mold_size,
                precomputed = (slot_contours, slot_canvas,
                               slot_clean_binary, slot_others))

            # ── Retry on holes/shape failures (steps 4/5) ────────────
            if res["defect_step"] in {4, 5}:
                rx_w = int(roi_w * 1.15)
                rx_h = int(roi_h * 1.15)
                rlx1 = max(0,  cell_cx - rx_w // 2)
                rly1 = max(0,  cell_cy - rx_h // 2)
                rlx2 = min(iw, rlx1 + rx_w)
                rly2 = min(ih, rly1 + rx_h)
                if rlx2 > rlx1 and rly2 > rly1:
                    retry_roi = image_bgr[rly1:rly2, rlx1:rlx2]
                    res_retry = self._eng.compare_roi(
                        retry_roi, tmpl,
                        exp_dx    = exp_dx,
                        exp_dy    = exp_dy,
                        mold_cx   = acx,
                        mold_cy   = acy,
                        mold_size = tmpl_mold_size,
                        is_retry  = True)
                    if res_retry["pass"]:
                        res  = res_retry
                        lx1, ly1, lx2, ly2 = rlx1, rly1, rlx2, rly2

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── OCR on best canvas (after hole cleanup) ───────────────
            ocr_canvas = res.get("roi_canvas")
            if ocr_canvas is not None and np.any(ocr_canvas):
                h_oc, w_oc = ocr_canvas.shape[:2]
                ocr_cnts, *_ = ContourTemplate._find_contours_all(
                    ocr_canvas, h_oc, w_oc)
            else:
                ocr_canvas = slot_canvas
                ocr_cnts   = slot_contours
            ocr_char, ocr_conf = self._eng.ocr_identify(
                ocr_canvas, ocr_cnts, letter, self._ocr_templates)

            dirty_info = res.get("dirty", {"detected": False, "type": "none", "area_ratio": 0.0})

            results.append({
                "frame_idx":   f_idx + 1,
                "mold":        mold_label,
                "slot":        slot_idx,
                "letter":      letter,
                "pass":        res["pass"],
                "confidence":  res["confidence"],
                "shift_px":    res["shift_px"],
                "shift_ratio": res["shift_ratio"],
                "defect_step": res["defect_step"],
                "defect_type": res.get("defect_type", ""),
                "reason":      res["reason"],
                "ocr_char":    ocr_char,
                "ocr_conf":    ocr_conf,
                "cell_cx":     cell_cx,
                "cell_cy":     cell_cy,
                "elapsed_ms":  round(elapsed_ms, 2),
                "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                "roi_canvas":  res["roi_canvas"],
                "dirty":       dirty_info,
            })

            # ── Per-slot log (all slots in DEBUG_MODE) ────────────────
            if DEBUG_MODE:
                step   = res["defect_step"]
                passed = results[-1]["pass"]
                ic_f   = f_idx * 2 + (1 if mold_label == "A" else 2)
                print(
                    f"[SLOT] F{ic_f} s{slot_idx+1}({letter})"
                    f" {'OK  ' if passed else 'FAIL'}"
                    f" step={step}"
                    f" conf={res['confidence']:.3f}"
                    f" shift={res['shift_ratio']:.3f}"
                    f" dirty={dirty_info['type']}({dirty_info['area_ratio']:.3f})"
                    f" cnts={len(slot_contours)}"
                    f" | {res['reason']}")
                if not passed and step == 1:
                    pfx = f"debug/FAIL_F{f_idx+1}_{mold_label}_s{slot_idx}_{letter}"
                    ContourTemplate.extract_font_template(
                        gray_slot, mold_size=mold_size, debug_prefix=pfx)

            ResultAnnotator.draw_letter(display, results[-1])

        return results

# =========================================================
# RUN WORKER  — QThread for batch (debug) and camera runs
# =========================================================
class RunWorker(QtCore.QThread):
    """
    Runs inspection loop off the GUI thread.

    Signals
    -------
    sig_image(np.ndarray)               : display frame (BGR annotated)
    sig_result(str, str)                : (log_message, css_color)
    sig_done(int, int)                  : (total_passed, total_inspected)
    sig_error(str)                      : fatal error message
    """

    sig_image  = QtCore.pyqtSignal(object)
    sig_result = QtCore.pyqtSignal(str, str)
    sig_done   = QtCore.pyqtSignal(int, int)
    sig_error  = QtCore.pyqtSignal(str)

    def __init__(self,
                 ctrl:        "InspectionController",
                 io:          "MachineIO",
                 mask:        np.ndarray,
                 image_io:    "ImageIO",
                 run_from_io: bool = False,
                 io_recipe:   list = None,
                 ui_grid:     list = None,
                 camera:      "BaslerCamera | None" = None,
                 sm:          "SettingsManager | None" = None):
        super().__init__()
        self._ctrl        = ctrl
        self._io          = io
        self._mask        = mask
        self._image_io    = image_io
        self._run_from_io = run_from_io
        self._io_recipe   = io_recipe or []
        self._ui_grid     = ui_grid   or []
        self._camera      = camera
        self._sm          = sm
        self._stop_flag   = False

    def stop(self):
        self._stop_flag = True

    # ---- helpers ----
    def _inspect_one(self, img_gray: np.ndarray) -> "InspectionResult":
        t0     = time.perf_counter()
        result = self._ctrl.run(img_gray, mask=self._mask)
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _log(self, msg: str, color: str = "#dddddd"):
        self.sig_result.emit(msg, color)

    def _save_fail(self, img_gray: np.ndarray, display_bgr: np.ndarray):
        """
        Save raw gray + annotated BGR for a failed image.
        Files: Inspection_result/<ts>_R.png  and  <ts>.png
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = os.path.join(OUTPUT_DIR, f"{ts}_R.png")
        ann_path = os.path.join(OUTPUT_DIR, f"{ts}.png")
        cv2.imwrite(raw_path, img_gray)
        cv2.imwrite(ann_path, display_bgr)
        self._log(f"  NG saved: {ts}_R.png + {ts}.png", "#ffaa44")

    def _save_last_lot(self, img_gray: np.ndarray, display_bgr: np.ndarray,
                       chip_cols: int):
        """
        Save raw gray + annotated BGR for a last-lot image.
        Files: Inspection_result/lastlot_<ts>_R.png  and  lastlot_<ts>.png
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = os.path.join(OUTPUT_DIR, f"lastlot_{ts}_R.png")
        ann_path = os.path.join(OUTPUT_DIR, f"lastlot_{ts}.png")
        cv2.imwrite(raw_path, img_gray)
        cv2.imwrite(ann_path, display_bgr)
        self._log(
            f"  LAST LOT saved: lastlot_{ts}_R.png + lastlot_{ts}.png"
            f"  ({chip_cols}/3 col)",
            "#ffaa00")

    def _append_csv(self, ic_groups: dict, image_name: str,
                    last_lot: bool = False):
        """
        Auto-append one row per IC result to run_log/result_YYYYMMDD.csv.
        Skips no-lead molds. Writes header on first entry of the day.
        """
        os.makedirs("run_log", exist_ok=True)
        path = os.path.join("run_log", f"result_{datetime.now():%Y%m%d}.csv")
        write_header = not os.path.exists(path)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestamp", "image", "ic_num", "frame",
                                "mold", "verdict", "ocr_string",
                                "fail_causes", "elapsed_ms", "last_lot"])
                for ic_num in sorted(ic_groups.keys()):
                    group = ic_groups[ic_num]
                    if all(r.get("ignored") for r in group):
                        continue
                    if all(r.get("reason") == "dropped" for r in group):
                        continue
                    r0       = group[0]
                    passed   = all(r["pass"] for r in group)
                    ocr_map  = {r["slot"]: r.get("ocr_char", " ") for r in group}
                    ocr_str  = "".join(ocr_map.get(i, " ") for i in range(9))
                    causes   = ";".join(
                        f"slot{r['slot']}({r['letter']}):{r['reason']}"
                        for r in group if not r["pass"])
                    mold_ms  = sum(r.get("elapsed_ms", 0.0) for r in group)
                    mold_ll  = any(r.get("last_lot") for r in group)
                    w.writerow([ts, image_name, ic_num,
                                r0.get("frame_idx", ""), r0.get("mold", ""),
                                "PASS" if passed else "NG",
                                ocr_str, causes, f"{mold_ms:.1f}",
                                "1" if mold_ll else "0"])
        except Exception as e:
            self._log(f"  CSV write error: {e}", "#ff4444")

    # ---- debug (folder) mode ----
    def _run_debug(self):
        files = self._image_io.list_images(IMAGE_SOURCE_DIR)
        if not files:
            self.sig_error.emit(
                f"No images found in '{IMAGE_SOURCE_DIR}'")
            return

        self._log(
            f"=== DEBUG batch — {len(files)} images in '{IMAGE_SOURCE_DIR}' ===",
            "#00e5ff")
        self._io.on_run_start()
        self._io.set_busy(True)

        total_passed  = 0
        total_letters = 0

        for i, fpath in enumerate(files):
            if self._stop_flag:
                self._log("Batch stopped by user.", "#ffaa44")
                break

            fname = os.path.basename(fpath)
            self._log(f"── Image [{i+1}/{len(files)}] {fname} ──", "#cccccc")

            try:
                img = self._image_io.load(fpath)
            except Exception as e:
                self.sig_error.emit(f"Load error '{fname}': {e}")
                break

            t0_img = time.perf_counter()
            result = self._inspect_one(img)
            img_ms = round((time.perf_counter() - t0_img) * 1000, 1)

            self.sig_image.emit(result.display)

            all_pass = (result.passed == result.total and result.total > 0)

            # ── Per-mold summary ──────────────────────────────────
            ic_groups: dict = defaultdict(list)
            for r in result.results:
                ic_groups[r.get("ic_num", 0)].append(r)

            for ic_num in sorted(ic_groups.keys()):
                group    = ic_groups[ic_num]
                r0       = group[0]
                f_idx    = r0["frame_idx"]
                mold_lbl = r0["mold"]
                mold_ms  = sum(r["elapsed_ms"] for r in group)
                passed   = all(r["pass"] for r in group)
                verdict  = "PASS" if passed else "NG"
                color    = "#88ff88" if passed else "#ff4444"

                # Check no-lead / last-lot ignored
                no_lead = all(r.get("reason") == "dropped" for r in group)
                ignored = all(r.get("ignored") for r in group)
                if no_lead:
                    self._log(
                        f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                        f"DROP WORK — skipped (pass)  "
                        f"[{mold_ms:.1f}ms]",
                        "#888888")
                    continue
                if ignored:
                    self._log(
                        f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                        f"LAST LOT — ignored (empty col)  "
                        f"[{mold_ms:.1f}ms]",
                        "#666666")
                    continue

                # Build OCR string — slot-ordered 9 chars, space for inactive
                ocr_map = {r["slot"]: r.get("ocr_char", " ") for r in group}
                ocr_str = "".join(ocr_map.get(i, " ") for i in range(9))

                # Build fail causes
                fail_causes = []
                for r in group:
                    if not r["pass"]:
                        fail_causes.append(
                            f"slot{r['slot']}({r['letter']}):"
                            f"{r['reason']}")

                cause_str = "  " + "  ".join(fail_causes) if fail_causes else ""
                self._log(
                    f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  {verdict}"
                    f"  [{mold_ms:.1f}ms]  OCR:\"{ocr_str}\"{cause_str}",
                    color)

            # ── Image timing summary ──────────────────────────────
            n_ic = len(ic_groups)
            self._log(
                f"  ▸ Image total: {img_ms:.1f}ms  "
                f"ICs={n_ic}  passed={result.passed}/{result.total}",
                "#aaaaaa")

            if not all_pass:
                self._save_fail(img, result.display)

            if result.last_lot:
                self._log(
                    f"  *** LAST LOT — {result.last_lot_cols}/3 col(s) filled ***",
                    "#ffaa00")
                self._save_last_lot(img, result.display, result.last_lot_cols)
                self._io.on_last_lot(result.last_lot_cols)

            self._append_csv(ic_groups, fname, last_lot=result.last_lot)
            self._io.on_frame_result(result.passed, result.total)
            total_passed  += result.passed
            total_letters += result.total

        self._io.set_busy(False)
        self._io.on_run_complete(total_passed, total_letters)
        self.sig_done.emit(total_passed, total_letters)

    # ---- camera mode ----
    def _run_camera(self):
        if self._camera is None or not self._camera.is_open():
            self.sig_error.emit("Camera not open.")
            return

        self._log("=== CAMERA mode — warmup ... ===", "#00e5ff")
        self._camera.warmup()
        self._log("Warmup done. Waiting for trigger signal.", "#aaddff")

        self._io.on_run_start()
        total_passed  = 0
        total_letters = 0

        while not self._stop_flag:

            triggered = self._io.wait_for_start(lambda: self._stop_flag)
            if not triggered or self._stop_flag:
                break

            if self._sm is not None:
                self._sm.save()

            self._io.set_busy(True)

            img = self._camera.grab()
            if img is None:
                self.sig_error.emit("Camera grab failed — stopping run.")
                self._io.set_busy(False)
                break

            t0_img = time.perf_counter()
            result = self._inspect_one(img)
            img_ms = round((time.perf_counter() - t0_img) * 1000, 1)

            self.sig_image.emit(result.display)

            all_pass = (result.passed == result.total and result.total > 0)

            # ── Per-mold summary ──────────────────────────────────
            ic_groups: dict = defaultdict(list)
            for r in result.results:
                ic_groups[r.get("ic_num", 0)].append(r)

            for ic_num in sorted(ic_groups.keys()):
                group    = ic_groups[ic_num]
                r0       = group[0]
                f_idx    = r0["frame_idx"]
                mold_lbl = r0["mold"]
                mold_ms  = sum(r["elapsed_ms"] for r in group)
                passed   = all(r["pass"] for r in group)
                verdict  = "PASS" if passed else "NG"
                color    = "#88ff88" if passed else "#ff4444"

                no_lead = all(r.get("reason") == "dropped" for r in group)
                ignored = all(r.get("ignored") for r in group)
                if ignored:
                    self._log(
                        f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                        f"LAST LOT — ignored (empty col)  "
                        f"[{mold_ms:.1f}ms]",
                        "#666666")
                    continue
                if no_lead:
                    self._log(
                        f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                        f"DROP WORK — skipped (pass)  "
                        f"[{mold_ms:.1f}ms]",
                        "#888888")
                    continue

                # Build OCR string — slot-ordered 9 chars, space for inactive
                ocr_map = {r["slot"]: r.get("ocr_char", " ") for r in group}
                ocr_str = "".join(ocr_map.get(i, " ") for i in range(9))

                fail_causes = []
                for r in group:
                    if not r["pass"]:
                        fail_causes.append(
                            f"slot{r['slot']}({r['letter']}):"
                            f"{r['reason']}")

                cause_str = "  " + "  ".join(fail_causes) if fail_causes else ""
                self._log(
                    f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  {verdict}"
                    f"  [{mold_ms:.1f}ms]  OCR:\"{ocr_str}\"{cause_str}",
                    color)

            # ── Image timing summary ──────────────────────────────
            n_ic = len(ic_groups)
            self._log(
                f"  ▸ Image total: {img_ms:.1f}ms  "
                f"ICs={n_ic}  passed={result.passed}/{result.total}",
                "#aaaaaa")

            if not all_pass:
                self._save_fail(img, result.display)

            if result.last_lot:
                self._log(
                    f"  *** LAST LOT — {result.last_lot_cols}/3 col(s) filled ***",
                    "#ffaa00")
                self._save_last_lot(img, result.display, result.last_lot_cols)
                self._io.on_last_lot(result.last_lot_cols)

            ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._append_csv(ic_groups, f"cam_{ts_label}", last_lot=result.last_lot)
            self._io.on_frame_result(result.passed, result.total)
            total_passed  += result.passed
            total_letters += result.total
            self._io.set_busy(False)

        self._io.on_run_complete(total_passed, total_letters)
        self.sig_done.emit(total_passed, total_letters)

    # ---- QThread entry ----
    def run(self):
        try:
            if DEBUG_MODE:
                self._run_debug()
            else:
                self._run_camera()
        except Exception as e:
            self.sig_error.emit(f"Worker exception: {e}")
            
# =========================================================
# IMAGE VIEW  — zoomable label with rubber-band / place-mode ROI
# =========================================================
class ImageView(QtWidgets.QLabel):
    roi_selected = QtCore.pyqtSignal(QtCore.QRect)

    PLACE_W = PIN_ROI_W
    PLACE_H = PIN_ROI_H

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._draw_mode  = False
        self._place_mode = False
        self._mask_mode  = False
        self._mask_add   = True
        self._cursor_img = None
        self._start      = None
        self._rect       = None
        self._overlays    = []        # (QRect, QColor, label, style)
        self._scale       = 1.0
        self._offset      = QtCore.QPoint(0, 0)
        self._orig        = None
        self._stamp_mode  = False
        self._stamp_box   = 45
        self._stamp_label = ""
        self.setMouseTracking(True)
        self._constraint_rect: QtCore.QRect | None = None
        
    def set_image(self, img: np.ndarray):
        """Accept grayscale (H×W) or BGR (H×W×3) — stores as BGR for display."""
        if img.ndim == 2:
            self._orig = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self._orig = img.copy()
        QtCore.QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        if self._orig is None:
            return
        h, w = self._orig.shape[:2]
        rgb  = cv2.cvtColor(self._orig, cv2.COLOR_BGR2RGB)
        qi   = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qi)
        lw, lh = self.width(), self.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        self._scale  = pix.width() / w
        self._offset = QtCore.QPoint((lw - pix.width())  // 2,
                                     (lh - pix.height()) // 2)
        self.setPixmap(pix)

    # ---- modes ----
    def set_draw_mode(self, on: bool):
        self._draw_mode  = on
        self._place_mode = False
        self._mask_mode  = False
        self._rect = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    def set_place_mode(self, on: bool):
        self._place_mode = on
        self._draw_mode  = False
        self._mask_mode  = False
        self._cursor_img = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    def set_mask_draw_mode(self, on: bool, add: bool = True):
        self._mask_mode  = on
        self._mask_add   = add
        self._draw_mode  = False
        self._place_mode = False
        self._start      = None
        self._rect       = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    def set_stamp_mode(self, on: bool, box_size: int = 45, label: str = ""):
        """
        Stamp mode: a fixed-size rectangle follows the cursor.
        Click emits roi_selected with a QRect centred on the click point.
        """
        self._stamp_mode  = on
        self._stamp_box   = box_size
        self._stamp_label = label
        self._draw_mode   = False
        self._place_mode  = False
        self._mask_mode   = False
        self._cursor_img  = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()
        
    # ---- overlays ----
    def add_overlay(self, rect: QtCore.QRect, color: QtGui.QColor,
                    label: str = "", style: str = "solid"):
        self._overlays.append((rect, color, label, style))
        self.update()

    def clear_overlays(self):
        self._overlays.clear()
        self.update()

    def set_constraint_rect(self, rect: QtCore.QRect | None):
        """Show/hide the yellow dashed constraint zone for mold A/B drawing."""
        self._constraint_rect = rect
        self.update()
        
    # ---- coordinate helper ----
    def _to_img(self, pt: QtCore.QPoint) -> QtCore.QPoint:
        pix = self.pixmap()
        if pix is None or pix.width() == 0 or self._orig is None:
            return pt
        _, orig_w = self._orig.shape[:2]
        lw, lh = self.width(), self.height()
        scale  = pix.width() / orig_w
        off_x  = (lw - pix.width())  // 2
        off_y  = (lh - pix.height()) // 2
        return QtCore.QPoint(
            int((pt.x() - off_x) / scale),
            int((pt.y() - off_y) / scale),
        )

    def mousePressEvent(self, e):
        if e.button() != QtCore.Qt.LeftButton:
            return
        if self._place_mode:
            img_pt = self._to_img(e.pos())
            ix = img_pt.x() - self.PLACE_W // 2
            iy = img_pt.y() - self.PLACE_H // 2
            self.roi_selected.emit(
                QtCore.QRect(ix, iy, self.PLACE_W, self.PLACE_H))
            self._place_mode = False
            self._cursor_img = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.update()
        elif self._stamp_mode:
            img_pt = self._to_img(e.pos())
            half   = self._stamp_box // 2
            self.roi_selected.emit(
                QtCore.QRect(img_pt.x() - half, img_pt.y() - half,
                            self._stamp_box, self._stamp_box))
        elif self._draw_mode or self._mask_mode:
            self._start = e.pos()
            self._rect  = None

    def mouseMoveEvent(self, e):
        if self._place_mode or self._stamp_mode:
            self._cursor_img = self._to_img(e.pos())
            self.update()
        elif (self._draw_mode or self._mask_mode) and self._start:
            self._rect = QtCore.QRect(self._start, e.pos()).normalized()
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self)

        for rect, color, label, style in self._overlays:
            wr = QtCore.QRect(
                QtCore.QPoint(int(rect.x() * self._scale) + self._offset.x(),
                            int(rect.y() * self._scale) + self._offset.y()),
                QtCore.QSize(int(rect.width()  * self._scale),
                            int(rect.height() * self._scale)),
            )
            qt_style = QtCore.Qt.DashLine if style == "dash" \
                    else QtCore.Qt.SolidLine
            p.setPen(QtGui.QPen(color, 2, qt_style))
            p.drawRect(wr)
            if label:
                p.setFont(QtGui.QFont("Arial", 7, QtGui.QFont.Bold))
                p.setPen(QtGui.QPen(color, 1, QtCore.Qt.SolidLine))
                p.drawText(wr.topLeft() + QtCore.QPoint(2, 12), label)
                
        # Constraint zone (mold A/B draw boundary)
        if self._constraint_rect is not None:
            cr = self._constraint_rect
            wcr = QtCore.QRect(
                QtCore.QPoint(int(cr.x() * self._scale) + self._offset.x(),
                              int(cr.y() * self._scale) + self._offset.y()),
                QtCore.QSize(int(cr.width()  * self._scale),
                             int(cr.height() * self._scale)),
            )
            p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0), 2,
                                QtCore.Qt.DashLine))
            p.drawRect(wcr)
            p.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
            p.setPen(QtGui.QColor(255, 200, 0))
            p.drawText(wcr.topLeft() + QtCore.QPoint(3, -4), "DRAW MOLD HERE")
            
        if self._draw_mode and self._rect:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 1,
                                QtCore.Qt.DashLine))
            p.drawRect(self._rect)

        if self._mask_mode and self._rect:
            if self._mask_add:
                fill   = QtGui.QColor(0, 220, 80, 60)
                border = QtGui.QColor(0, 220, 80)
                tag    = "+ ADD"
            else:
                fill   = QtGui.QColor(220, 40, 40, 60)
                border = QtGui.QColor(220, 40, 40)
                tag    = "- SUB"
            p.fillRect(self._rect, fill)
            p.setPen(QtGui.QPen(border, 2, QtCore.Qt.SolidLine))
            p.drawRect(self._rect)
            p.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
            p.setPen(border)
            p.drawText(self._rect.topLeft() + QtCore.QPoint(4, 14), tag)

        # Stamp mode cursor
        if self._stamp_mode and self._cursor_img is not None:
            wcx = int(self._cursor_img.x() * self._scale) + self._offset.x()
            wcy = int(self._cursor_img.y() * self._scale) + self._offset.y()
            wb  = int(self._stamp_box * self._scale)
            wx  = wcx - wb // 2
            wy  = wcy - wb // 2
            p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0), 2,
                                QtCore.Qt.SolidLine))
            p.drawRect(wx, wy, wb, wb)
            p.drawLine(wcx - 8, wcy, wcx + 8, wcy)
            p.drawLine(wcx, wcy - 8, wcx, wcy + 8)
            if self._stamp_label:
                p.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                p.setPen(QtGui.QColor(255, 200, 0))
                p.drawText(wx + 3, wy + 13, self._stamp_label)

        # Place mode cursor (keep existing)
        if self._place_mode and self._cursor_img is not None:
            wcx = int(self._cursor_img.x() * self._scale) + self._offset.x()
            wcy = int(self._cursor_img.y() * self._scale) + self._offset.y()
            ww  = int(self.PLACE_W * self._scale)
            wh  = int(self.PLACE_H * self._scale)
            wx  = wcx - ww // 2
            wy  = wcy - wh // 2
            p.setPen(QtGui.QPen(QtGui.QColor(0, 180, 255), 2,
                                QtCore.Qt.SolidLine))
            p.drawRect(wx, wy, ww, wh)
            p.drawLine(wcx - 8, wcy, wcx + 8, wcy)
            p.drawLine(wcx, wcy - 8, wcx, wcy + 8)

        p.end()
    
    def mouseReleaseEvent(self, e):
        if self._draw_mode and self._start and self._rect:
            ir = QtCore.QRect(
                self._to_img(self._rect.topLeft()),
                self._to_img(self._rect.bottomRight())).normalized()
            self.roi_selected.emit(ir)
            self._start = None
        elif self._mask_mode and self._start and self._rect:
            ir = QtCore.QRect(
                self._to_img(self._rect.topLeft()),
                self._to_img(self._rect.bottomRight())).normalized()
            self.roi_selected.emit(ir)
            self._start = None
            self._rect  = None
            self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

# =========================================================
# MASKING TOOLBAR
# =========================================================
class MaskingToolbar(QtWidgets.QWidget):
    """Secondary toolbar shown only while drawing mask regions."""

    sig_add      = QtCore.pyqtSignal()
    sig_subtract = QtCore.pyqtSignal()
    sig_complete = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        lbl = QtWidgets.QLabel("  MASK EDIT MODE:")
        lbl.setStyleSheet("color:#ffcc00;font-weight:bold")
        lay.addWidget(lbl)

        self._btn_add = QtWidgets.QPushButton("Add Mask")
        self._btn_sub = QtWidgets.QPushButton("Subtract Mask")
        self._btn_ok  = QtWidgets.QPushButton("Complete")

        self._btn_add.setStyleSheet(
            "background:#1a6b2a;color:#fff;font-weight:bold;padding:4px 10px")
        self._btn_sub.setStyleSheet(
            "background:#6b1a1a;color:#fff;font-weight:bold;padding:4px 10px")
        self._btn_ok.setStyleSheet(
            "background:#4a4a00;color:#fff;font-weight:bold;padding:4px 10px")

        self._btn_add.setCheckable(True)
        self._btn_sub.setCheckable(True)
        self._btn_add.setChecked(True)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_sub.clicked.connect(self._on_sub)
        self._btn_ok.clicked.connect(self.sig_complete)

        for b in (self._btn_add, self._btn_sub, self._btn_ok):
            lay.addWidget(b)

        lay.addStretch()
        hint = QtWidgets.QLabel(
            "  Draw rectangles.  Add = allow search    Subtract = block search")
        hint.setStyleSheet("color:#999")
        lay.addWidget(hint)

    def _on_add(self):
        self._btn_add.setChecked(True)
        self._btn_sub.setChecked(False)
        self.sig_add.emit()

    def _on_sub(self):
        self._btn_sub.setChecked(True)
        self._btn_add.setChecked(False)
        self.sig_subtract.emit()

    def set_add_mode(self):
        self._btn_add.setChecked(True)
        self._btn_sub.setChecked(False)

    def set_sub_mode(self):
        self._btn_sub.setChecked(True)
        self._btn_add.setChecked(False)


# =========================================================
# MASK CONFIRM DIALOG
# =========================================================
class MaskConfirmDialog(QtWidgets.QDialog):
    """Colour-coded mask preview — confirm save or cancel."""

    def __init__(self, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Search Mask")
        self.setModal(True)
        lay = QtWidgets.QVBoxLayout(self)

        h, w  = mask.shape[:2]
        scale = min(1.0, 600 / max(w, 1))
        dw    = max(int(w * scale), 1)
        dh    = max(int(h * scale), 1)
        prev  = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        colour = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
        colour[prev == 255] = [40, 160,  40]
        colour[prev == 0]   = [60,  40, 140]

        rgb = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        qi  = QtGui.QImage(rgb.data, dw, dh, 3 * dw,
                           QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qi)

        img_lbl = QtWidgets.QLabel()
        img_lbl.setPixmap(pix)
        img_lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(img_lbl)

        white_pct = 100.0 * np.count_nonzero(mask) / max(mask.size, 1)
        info = QtWidgets.QLabel(
            f"<b>Mask preview</b>  |  "
            f"Image size: <b>{w}x{h}</b> px  |  "
            f"<span style='color:#80ff80'>"
            f"Searchable: {white_pct:.1f}%</span>  "
            f"<span style='color:#ff8080'>"
            f"Blocked: {100 - white_pct:.1f}%</span>"
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(info)

        note = QtWidgets.QLabel(
            "<i>Green = search allowed    Purple = search blocked</i>")
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setStyleSheet("color:#888")
        lay.addWidget(note)

        btn_row    = QtWidgets.QHBoxLayout()
        btn_save   = QtWidgets.QPushButton("Save Mask")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_save.setStyleSheet(
            "background:#1a6b2a;color:#fff;font-weight:bold;padding:4px 14px")
        btn_cancel.setStyleSheet(
            "background:#6b1a1a;color:#fff;font-weight:bold;padding:4px 14px")
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        self.adjustSize()


# =========================================================
# TEMPLATE PREVIEW DIALOG  — 3-panel: original | canvas | overlay
# =========================================================
class TemplatePreviewDialog(QtWidgets.QDialog):
    """
    Shows 3 panels after a template is saved:
      Left   : original BGR crop
      Centre : pre-rendered contour canvas (filled, grayscale)
      Right  : original crop with contour outlines overlaid in cyan
    """

    MAX_W = 320
    MAX_H = 320

    def __init__(self, roi_bgr: np.ndarray, name: str,
                 mold_size: int = 150, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Template Preview — {name}")
        self.setModal(True)
        lay = QtWidgets.QVBoxLayout(self)

        h, w  = roi_bgr.shape[:2]
        scale = max(1.0, min(4.0,
                            self.MAX_W / max(w, 1),
                            self.MAX_H / max(h, 1)))
        dw = max(int(w * scale), 1)
        dh = max(int(h * scale), 1)

        contours, canvas, *_ = ContourTemplate.extract_font_template(
            roi_bgr, mold_size=mold_size)   # was hardcoded 150

        orig_bgr  = roi_bgr if roi_bgr.ndim == 3 \
                    else cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        canvas_3c = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        overlay   = orig_bgr.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        orig_rs    = cv2.resize(orig_bgr,  (dw, dh),
                                interpolation=cv2.INTER_LINEAR)
        canvas_rs  = cv2.resize(canvas_3c, (dw, dh),
                                interpolation=cv2.INTER_LINEAR)
        overlay_rs = cv2.resize(overlay,   (dw, dh),
                                interpolation=cv2.INTER_LINEAR)

        gap      = np.zeros((dh, 4, 3), dtype=np.uint8)
        combined = np.hstack([orig_rs, gap, canvas_rs, gap, overlay_rs])

        ch, cw = combined.shape[:2]
        rgb    = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        qi     = QtGui.QImage(rgb.data, cw, ch, 3 * cw,
                              QtGui.QImage.Format_RGB888)
        pix    = QtGui.QPixmap.fromImage(qi.copy())

        img_lbl = QtWidgets.QLabel()
        img_lbl.setPixmap(pix)
        img_lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(img_lbl)

        info = QtWidgets.QLabel(
            f"<b>{name}</b>  |  "
            f"ROI: <b>{w}x{h} px</b>  |  "
            f"Contours kept: <b>{len(contours)}</b>  "
            f"(area >= {MIN_CONTOUR_AREA} px^2)<br>"
            f"<i style='color:#888'>Left: original  "
            f"Centre: canvas  Right: overlay</i>"
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(info)

        btn = QtWidgets.QPushButton("OK")
        btn.setFixedWidth(100)
        btn.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(); row.addWidget(btn); row.addStretch()
        lay.addLayout(row)
        self.adjustSize()


# =========================================================
# FRAME LAYOUT DIALOG
# =========================================================
class FrameLayoutDialog(QtWidgets.QDialog):
    """
    Step 4 of the frame template wizard.

    Displays the full image; user left-clicks to stamp expected anchor
    positions in order F1…FN.  Each stamp creates an ROI box sized to
    cover the complete frame (anchor + mold A + mold B) computed from
    the saved recipe.

    Undo removes the last stamp.  Confirm saves frame_layout.json.
    """

    LAYOUT_FILE = "frame_layout.json"
    MAX_DISP_W  = 920
    MAX_DISP_H  = 660

    def __init__(self, image_bgr: np.ndarray, recipe: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Layout — Stamp Expected Frame Positions")
        self.setModal(True)

        self._image_bgr = image_bgr
        self._recipe    = recipe
        self._stamps    = []   # list of (anchor_cx, anchor_cy) in image coords

        # Compute ROI dimensions from recipe
        aw, ah         = recipe["mold_size"]
        dx_a, dy_a     = recipe["mold_a_shift"]
        _dx_b, dy_b    = recipe["mold_b_shift"]
        self._anc_w    = aw
        self._anc_h    = ah
        self._dy_a     = dy_a
        self._dy_b     = dy_b
        # Full frame width: anchor half + mold-A x-shift + mold half
        self._roi_w    = aw // 2 + int(dx_a) + aw // 2
        # Full frame height: mold B bottom - mold A top
        self._roi_h    = int(dy_b - dy_a) + ah

        # Scale image to fit dialog display area
        ih, iw         = image_bgr.shape[:2]
        scale          = min(self.MAX_DISP_W / max(iw, 1),
                             self.MAX_DISP_H / max(ih, 1), 1.0)
        self._scale    = scale
        dw             = max(int(iw * scale), 1)
        dh             = max(int(ih * scale), 1)

        self.setFixedSize(dw + 24, dh + 90)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Hint label
        hint = QtWidgets.QLabel(
            "Left-click the anchor centre of each expected frame (F1, F2, …). "
            "Undo removes the last stamp.  Confirm saves the layout.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa; font-size:9px")
        root.addWidget(hint)

        # Image label — receives mouse clicks
        self._lbl = QtWidgets.QLabel()
        self._lbl.setFixedSize(dw, dh)
        self._lbl.setCursor(QtCore.Qt.CrossCursor)
        self._lbl.mousePressEvent = self._on_click
        root.addWidget(self._lbl)

        # Button row
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_undo    = QtWidgets.QPushButton("Undo (last stamp)")
        self._btn_confirm = QtWidgets.QPushButton("Confirm & Save")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")
        self._btn_confirm.setDefault(True)
        btn_row.addWidget(self._btn_undo)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_cancel)
        root.addLayout(btn_row)

        self._btn_undo.clicked.connect(self._undo)
        self._btn_confirm.clicked.connect(self._confirm)
        self._btn_cancel.clicked.connect(self.reject)

        self._refresh()

    # ---- mouse input ----

    def _on_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        s   = self._scale
        px  = int(ev.x() / s)
        py  = int(ev.y() / s)
        self._stamps.append((px, py))
        self._refresh()

    def _undo(self):
        if self._stamps:
            self._stamps.pop()
            self._refresh()

    # ---- save ----

    def _confirm(self):
        if not self._stamps:
            QtWidgets.QMessageBox.warning(
                self, "No Stamps",
                "Place at least one frame stamp before confirming.")
            return

        aw   = self._anc_w
        ah   = self._anc_h
        dy_a = self._dy_a
        rw   = self._roi_w
        rh   = self._roi_h

        frames = []
        for i, (cx, cy) in enumerate(self._stamps):
            # ROI top-left: anchor left edge, mold-A top edge
            rx = cx - aw // 2
            ry = cy + int(dy_a) - ah // 2
            frames.append({
                "id":  f"F{i + 1}",
                "roi": [rx, ry, rw, rh],
            })

        layout = {"version": 1, "frames": frames}
        try:
            with open(self.LAYOUT_FILE, "w") as f:
                json.dump(layout, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Could not save frame_layout.json:\n{e}")
            return

        self.accept()

    # ---- display refresh ----

    def _refresh(self):
        s    = self._scale
        aw   = self._anc_w
        ah   = self._anc_h
        dy_a = self._dy_a
        rw   = self._roi_w
        rh   = self._roi_h

        disp = cv2.resize(self._image_bgr,
                          (int(self._image_bgr.shape[1] * s),
                           int(self._image_bgr.shape[0] * s)),
                          interpolation=cv2.INTER_LINEAR)
        dh, dw = disp.shape[:2]

        for i, (cx, cy) in enumerate(self._stamps):
            # ROI in image coords
            rx = cx - aw // 2
            ry = cy + int(dy_a) - ah // 2

            # Convert to display coords
            drx  = max(0,    int(rx * s))
            dry  = max(0,    int(ry * s))
            drx2 = min(dw-1, int((rx + rw) * s))
            dry2 = min(dh-1, int((ry + rh) * s))

            cv2.rectangle(disp, (drx, dry), (drx2, dry2), (0, 224, 255), 2)

            # F-label top-left of box
            lbl = f"F{i + 1}"
            cv2.putText(disp, lbl,
                        (drx + 4, dry + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 224, 255), 2, cv2.LINE_AA)

            # Green cross at anchor centre
            dcx = max(0, min(dw-1, int(cx * s)))
            dcy = max(0, min(dh-1, int(cy * s)))
            cv2.drawMarker(disp, (dcx, dcy), (0, 255, 0),
                           cv2.MARKER_CROSS, 14, 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._lbl.setPixmap(QtGui.QPixmap.fromImage(qi.copy()))


# =========================================================
# FRAME RECIPE PREVIEW DIALOG
# =========================================================
class FrameRecipePreviewDialog(QtWidgets.QDialog):
    """
    Shows FRAME, MOLD A, and MOLD B each as a 3-panel strip:
      Left: original BGR crop | Centre: contour canvas | Right: contour overlay

    Stacked vertically — one strip per section.
    """

    PANEL_W = 240   # display width per individual panel image
    PANEL_H = 200   # display height per individual panel image

    def __init__(self, recipe: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Recipe Preview")
        self.setModal(True)
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        for key, label, color_bgr in [
            ("frame",  "FRAME",  (255, 200,   0)),
            ("mold_a", "MOLD A", (  0, 224, 255)),
            ("mold_b", "MOLD B", (  0, 180, 200)),
        ]:
            sec = recipe.get(key)
            if sec is None:
                continue

            # Decode section data
            contours = sec.get("contours", [])
            canvas   = sec.get("canvas")
            if canvas is None:
                continue

            # Reconstruct original BGR from canvas (grayscale → colour)
            # We use the canvas as "original" since the raw crop is not stored
            orig_gray = canvas
            h, w = orig_gray.shape[:2]

            orig_bgr  = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
            canvas_3c = cv2.cvtColor(canvas,    cv2.COLOR_GRAY2BGR)
            overlay   = orig_bgr.copy()
            cv2.drawContours(overlay, contours, -1, color_bgr, 1)

            # Scale to display size keeping aspect ratio
            scale = min(self.PANEL_W / max(w, 1), self.PANEL_H / max(h, 1))
            scale = max(1.0, scale)
            dw    = max(int(w * scale), 1)
            dh    = max(int(h * scale), 1)

            orig_rs    = cv2.resize(orig_bgr,  (dw, dh), interpolation=cv2.INTER_LINEAR)
            canvas_rs  = cv2.resize(canvas_3c, (dw, dh), interpolation=cv2.INTER_LINEAR)
            overlay_rs = cv2.resize(overlay,   (dw, dh), interpolation=cv2.INTER_LINEAR)

            gap      = np.zeros((dh, 4, 3), dtype=np.uint8)
            combined = np.hstack([orig_rs, gap, canvas_rs, gap, overlay_rs])

            ch, cw = combined.shape[:2]
            rgb    = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            qi     = QtGui.QImage(rgb.data, cw, ch, 3 * cw,
                                  QtGui.QImage.Format_RGB888)
            pix    = QtGui.QPixmap.fromImage(qi.copy())

            # Section title
            title = QtWidgets.QLabel(f"<b style='color:#00e5ff'>{label}</b>"
                                     f"  {w}x{h} px  "
                                     f"contours: {len(contours)}")
            title.setStyleSheet("font-size:10px")
            root.addWidget(title)

            img_lbl = QtWidgets.QLabel()
            img_lbl.setPixmap(pix)
            img_lbl.setAlignment(QtCore.Qt.AlignLeft)
            root.addWidget(img_lbl)

            note = QtWidgets.QLabel(
                "<i style='color:#666'>original  |  canvas  |  overlay</i>")
            note.setStyleSheet("font-size:9px")
            root.addWidget(note)

            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Sunken)
            root.addWidget(sep)

        btn = QtWidgets.QPushButton("Close")
        btn.setFixedWidth(100)
        btn.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(); row.addWidget(btn); row.addStretch()
        root.addLayout(row)
        self.adjustSize()


# =========================================================
# SETUP PREVIEW DIALOG
# =========================================================
class SetupPreviewDialog(QtWidgets.QDialog):
    """
    Renders the full image with mold A, mold B, and all letter cell
    boxes drawn on a copy — no changes to the live image view.

    Frame anchor taken from recipe["frame"]["contour"] (capture position).
    Mold centres derived using stored offsets.
    Letter cells placed using marking_pairs offsets from mold centre.
    Template canvas size used for each cell if available, else 45x45 px.

    Colour coding:
      FRAME box  — dashed cyan
      MOLD boxes — dashed cyan (dimmer)
      Cell boxes — yellow solid, labelled with template name
    """

    MAX_DIALOG_W = 1000
    MAX_DIALOG_H =  800

    def __init__(self, image_bgr: np.ndarray, recipe: dict,
                 ct: "ContourTemplate", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Setup Preview")
        self.setModal(True)

        ih, iw = image_bgr.shape[:2]
        display = image_bgr.copy()

        # ── frame ──────────────────────────────────────────────
        fx, fy, fw, fh = recipe["frame"]["contour"]
        fcx = fx + fw // 2
        fcy = fy + fh // 2
        cv2_draw_dashed_rect(display, (fx, fy), (fx + fw, fy + fh),
                             (0, 224, 255), 1)
        cv2.putText(display, "FRAME", (fx + 2, fy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 224, 255), 1)

        # ── molds + cells ──────────────────────────────────────
        grid_letters = recipe.get("grid_letters", [])
        for key, mold_label in [("mold_a", "MOLD A"), ("mold_b", "MOLD B")]:
            mold     = recipe[key]
            odx, ody = mold["offset"]
            mw, mh   = mold["canvas_w"], mold["canvas_h"]
            acx = max(mw // 2, min(iw - mw // 2, fcx + odx))
            acy = max(mh // 2, min(ih - mh // 2, fcy + ody))
            ax1, ay1 = acx - mw // 2, acy - mh // 2
            ax2, ay2 = ax1 + mw, ay1 + mh
            cv2_draw_dashed_rect(display, (ax1, ay1), (ax2, ay2),
                                 (0, 180, 200), 1)
            cv2.putText(display, mold_label, (ax1 + 2, ay1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 200), 1)

            cell_w = int(mw * 0.90 / 3)
            cell_h = int(mh * 0.85 / 3)
            for slot_idx, letter in enumerate(grid_letters):
                if not letter:
                    continue
                row = slot_idx // 3
                col = slot_idx  % 3
                dx  = (col - 1) * cell_w
                dy  = (row - 1) * cell_h
                ccx = acx + dx
                ccy = acy + dy
                roi_w = int(cell_w * 1.1)
                roi_h = int(cell_h * 1.1)
                lx1 = max(0,  ccx - roi_w // 2)
                ly1 = max(0,  ccy - roi_h // 2)
                lx2 = min(iw, lx1 + roi_w)
                ly2 = min(ih, ly1 + roi_h)
                cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (255, 220, 0), 1)
                cv2.putText(display, letter, (lx1 + 2, ly2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 220, 0), 1)
                
        # ── scale to dialog ────────────────────────────────────
        scale = min(self.MAX_DIALOG_W / max(iw, 1),
                    self.MAX_DIALOG_H / max(ih, 1), 1.0)
        dw = max(int(iw * scale), 1)
        dh = max(int(ih * scale), 1)
        disp_rs = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(disp_rs, cv2.COLOR_BGR2RGB)
        qi  = QtGui.QImage(rgb.data, dw, dh, 3 * dw,
                           QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qi.copy())

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        img_lbl = QtWidgets.QLabel()
        img_lbl.setPixmap(pix)
        img_lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(img_lbl)

        active = sum(1 for l in recipe.get("grid_letters", []) if l)
        self._panel.log(f"Setup preview — {active} active slots.", "#88ff88")
        
        info = QtWidgets.QLabel(
            f"Frame anchor: ({fcx},{fcy})  |  "
            f"Active slots: <b>{active}</b>"
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        info.setStyleSheet("font-size:10px")
        lay.addWidget(info)

        note = QtWidgets.QLabel(
            "<i style='color:#666'>Cyan dashed = frame/mold  "
            "Yellow solid = letter cells</i>")
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setStyleSheet("font-size:9px")
        lay.addWidget(note)

        btn = QtWidgets.QPushButton("Close")
        btn.setFixedWidth(100)
        btn.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(); row.addWidget(btn); row.addStretch()
        lay.addLayout(row)
        self.adjustSize()


# =========================================================
# FRAME TEMPLATE PANEL  — floating non-blocking step panel
# =========================================================
class FrameTemplatePanel(QtWidgets.QWidget):
    """
    Floating always-on-top panel — single-step YOLO mold detection confirm.

    Shows detected mold bbox overlay; user clicks Confirm to save or
    Retry to re-run detection.  Cancel aborts.
    """

    def __init__(self, on_confirm, on_retry, on_cancel, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("Create Frame Template")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setFixedWidth(340)

        self._on_confirm = on_confirm
        self._on_retry   = on_retry
        self._on_cancel  = on_cancel

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        self._lbl_title = QtWidgets.QLabel("Auto Mold Detection")
        self._lbl_title.setStyleSheet(
            "font-size:13px;font-weight:bold;color:#00e5ff")
        self._lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_title)

        self._lbl_status = QtWidgets.QLabel("Detecting…")
        self._lbl_status.setStyleSheet("color:#cccccc;font-size:11px")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_status.setWordWrap(True)
        lay.addWidget(self._lbl_status)

        lay.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_confirm = QtWidgets.QPushButton("Save Template")
        self._btn_retry   = QtWidgets.QPushButton("Retry")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_confirm.setStyleSheet(
            "background:#005f6b;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_retry.setStyleSheet(
            "background:#4a3a00;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_cancel.setStyleSheet(
            "background:#4a1a1a;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")

        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_retry.clicked.connect(self._on_retry)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_retry)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.adjustSize()

    def set_detected(self, rect_str: str):
        """Call after successful detection with a bbox description string."""
        self._lbl_status.setText(
            f"Mold detected:\n{rect_str}\n\nFrame (lead) box auto-derived.")
        self._lbl_status.setStyleSheet("color:#88ff88;font-size:11px")
        self._btn_confirm.setEnabled(True)

    def set_no_detection(self):
        """Call when YOLO finds nothing."""
        self._lbl_status.setText(
            "No mold detected.\nCheck image or model file,\nthen click Retry.")
        self._lbl_status.setStyleSheet("color:#ffaa44;font-size:11px")
        self._btn_confirm.setEnabled(False)

    def set_no_model(self):
        """Call when YOLO model is not available."""
        self._lbl_status.setText(
            "YOLO model not ready.\n"
            f"Place {YOLO_MODEL_XML} in the project folder.")
        self._lbl_status.setStyleSheet("color:#ff4444;font-size:11px")
        self._btn_confirm.setEnabled(False)
        self._btn_retry.setEnabled(False)

    def closeEvent(self, e):
        self._on_cancel()
        e.accept()
            
# =========================================================
# RIGHT PANEL
# =========================================================
class RightPanel(QtWidgets.QWidget):
    """
    Settings panel.
    Groups: Frame Search | Inspection | Grid Letters | Cell Box |
            Frame Recipe | Camera | Result Log
    Each spinbox is bound to SettingsManager and auto-saves on change.
    """

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self._sm      = settings
        self._grid_changed_cb = None
        self.setFixedWidth(380)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        # ── Inspection ────────────────────────────────────────
        # ── Pin Search ────────────────────────────────────────
        gb_pin = QtWidgets.QGroupBox("Pin Template Matching")
        fl_pin = QtWidgets.QFormLayout(gb_pin)
        self.spin_pin_score = self._bind_fspin("pin_score_threshold", fl_pin,
                                               "Min score")
        note_pin = QtWidgets.QLabel(
            "Min score: match threshold for anchor TM inside each frame ROI.\n"
            "Frame count is defined by the layout (step 4 of frame wizard).")
        note_pin.setStyleSheet("color:#888;font-size:9px")
        note_pin.setWordWrap(True)
        fl_pin.addRow("", note_pin)
        lay.addWidget(gb_pin)

        # ── Grid Position ─────────────────────────────────────
        gb_grid = QtWidgets.QGroupBox("Grid Position")
        fl_grid = QtWidgets.QFormLayout(gb_grid)
        self.spin_grid_scale = self._bind_fspin("grid_scale",  fl_grid, "Scale")
        self.spin_grid_x     = self._bind_fspin("grid_x_frac", fl_grid, "X offset")
        self.spin_grid_y     = self._bind_fspin("grid_y_frac", fl_grid, "Y offset")
        note_grid = QtWidgets.QLabel(
            "Scale: fraction of mold covered by 3×3 grid (0.85 = 85%).\n"
            "X/Y offset: shift grid centre as fraction of mold size.")
        note_grid.setStyleSheet("color:#888;font-size:9px")
        note_grid.setWordWrap(True)
        fl_grid.addRow("", note_grid)
        lay.addWidget(gb_grid)

        # ── Grid Letters — 3×3 slot grid ─────────────────────
        gb_gl = QtWidgets.QGroupBox("Expected Slot Letters")
        vl_gl = QtWidgets.QVBoxLayout(gb_gl)
        hint3 = QtWidgets.QLabel(
            "Fill each cell with the expected mark  (empty = skip)")
        hint3.setStyleSheet("color:#888;font-size:9px")
        vl_gl.addWidget(hint3)

        # Row-index labels + grid cells
        grid_container = QtWidgets.QWidget()
        grid_lay = QtWidgets.QGridLayout(grid_container)
        grid_lay.setSpacing(4)
        grid_lay.setContentsMargins(2, 2, 2, 2)

        # Column headers
        for col, lbl in enumerate(["Col 1", "Col 2", "Col 3"]):
            hdr = QtWidgets.QLabel(lbl)
            hdr.setAlignment(QtCore.Qt.AlignCenter)
            hdr.setStyleSheet("color:#888;font-size:8px")
            grid_lay.addWidget(hdr, 0, col + 1)

        self._slot_cells: list = []
        saved_parts = self._sm.get_str("grid_letters").split(",")
        while len(saved_parts) < 9:
            saved_parts.append("")

        for row in range(3):
            # Row label
            row_lbl = QtWidgets.QLabel(f"Row {row+1}")
            row_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row_lbl.setStyleSheet("color:#888;font-size:8px")
            grid_lay.addWidget(row_lbl, row + 1, 0)

            for col in range(3):
                slot_idx = row * 3 + col
                cell = QtWidgets.QLineEdit()
                cell.setMaxLength(3)
                cell.setFixedSize(54, 28)
                cell.setAlignment(QtCore.Qt.AlignCenter)
                cell.setFont(QtGui.QFont("Courier New", 10))
                cell.setToolTip(f"Slot {slot_idx + 1}  (row {row+1}, col {col+1})")
                val = saved_parts[slot_idx].strip()
                cell.setText(val)
                cell.textChanged.connect(self._on_grid_cell_changed)
                grid_lay.addWidget(cell, row + 1, col + 1)
                self._slot_cells.append(cell)

        vl_gl.addWidget(grid_container)
        self._io_recipe_lbl = QtWidgets.QLabel("IO recipe: —")
        self._io_recipe_lbl.setStyleSheet("color:#888;font-size:9px")
        vl_gl.addWidget(self._io_recipe_lbl)
        lay.addWidget(gb_gl)

        # ── Camera ────────────────────────────────────────────
        gb_cam = QtWidgets.QGroupBox("Camera")
        fl_cam = QtWidgets.QFormLayout(gb_cam)
        self.spin_exposure = self._bind_ispin("camera_exposure_us", fl_cam,
                                              "Exposure (µs)")
        mode_lbl = QtWidgets.QLabel(
            "DEBUG (folder)" if DEBUG_MODE else "CAMERA (Basler)")
        mode_lbl.setStyleSheet(
            "color:#ffcc00;font-size:9px" if DEBUG_MODE
            else "color:#88ff88;font-size:9px")
        fl_cam.addRow("Mode:", mode_lbl)
        lay.addWidget(gb_cam)

        # ── Result Log ────────────────────────────────────────
        gb_log = QtWidgets.QGroupBox("Result Log")
        fl_log = QtWidgets.QVBoxLayout(gb_log)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(200)
        self.log_box.setFont(QtGui.QFont("Courier New", 8))
        fl_log.addWidget(self.log_box)
        lay.addWidget(gb_log)

        lay.addStretch()

    # ---- spinbox factory helpers ----
    def _make_spin_row(self, spin_widget, key: str):
        sm = self._sm

        def _step(direction: int):
            step = spin_widget.singleStep()
            raw  = spin_widget.value() + direction * step
            sm.set_value(key, raw)
            clamped = sm.get(key)
            spin_widget.blockSignals(True)
            spin_widget.setValue(clamped)
            spin_widget.blockSignals(False)
            sm.save()

        def _on_changed(val):
            sm.set_value(key, val)
            clamped = sm.get(key)
            if abs(clamped - val) > 1e-9:
                spin_widget.blockSignals(True)
                spin_widget.setValue(clamped)
                spin_widget.blockSignals(False)
            sm.save()

        btn_up   = QtWidgets.QPushButton("▲")
        btn_down = QtWidgets.QPushButton("▼")
        for b in (btn_up, btn_down):
            b.setFixedSize(22, 20)
            b.setStyleSheet("font-size:9px;padding:0px")
        btn_up.clicked.connect(lambda: _step(+1))
        btn_down.clicked.connect(lambda: _step(-1))
        spin_widget.valueChanged.connect(_on_changed)

        row = QtWidgets.QWidget()
        rl  = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        rl.addWidget(spin_widget, stretch=1)
        rl.addWidget(btn_up)
        rl.addWidget(btn_down)
        return row

    def _bind_ispin(self, key: str, form_layout, label: str):
        sm = self._sm
        w  = QtWidgets.QSpinBox()
        w.setRange(int(sm.get_min(key)), int(sm.get_max(key)))
        w.setValue(int(sm.get(key)))
        form_layout.addRow(label, self._make_spin_row(w, key))
        return w

    def _bind_fspin(self, key: str, form_layout, label: str):
        sm = self._sm
        w  = QtWidgets.QDoubleSpinBox()
        w.setRange(float(sm.get_min(key)), float(sm.get_max(key)))
        w.setSingleStep(0.01)
        w.setDecimals(3)
        w.setValue(float(sm.get(key)))
        form_layout.addRow(label, self._make_spin_row(w, key))
        return w
    
    # ---- public API ----
    def log(self, msg: str, color: str = "#dddddd"):
        ts   = datetime.now().strftime("%H:%M:%S")
        html = f'<span style="color:{color}">[{ts}] {msg}</span>'
        self.log_box.append(html)

    def apply_settings(self):
        sm    = self._sm
        pairs = [
            (self.spin_pin_score, "pin_score_threshold"),
            (self.spin_exposure,  "camera_exposure_us"),
        ]
        for spin, key in pairs:
            spin.blockSignals(True)
            spin.setRange(sm.get_min(key), sm.get_max(key))
            spin.setValue(sm.get(key))
            spin.blockSignals(False)

        parts = sm.get_str("grid_letters").split(",")
        while len(parts) < 9:
            parts.append("")
        for i, cell in enumerate(self._slot_cells):
            cell.blockSignals(True)
            val = parts[i].strip()
            cell.setText(val)
            cell.blockSignals(False)

    def pin_search_params(self) -> dict:
        sm = self._sm
        return {
            "score_thr": sm.get("pin_score_threshold"),
        }

    def font_list(self, ct: "ContourTemplate") -> list:
        """
        Return all template names from the templates/ folder.
        Auto-detected at run time — no user input required.
        """
        return ct.list_templates()

    def _on_grid_cell_changed(self):
        letters = self.grid_letters()
        self._sm.set_str("grid_letters", ",".join(letters))
        self._sm.save()
        if self._grid_changed_cb:
            self._grid_changed_cb(letters)
        
    def set_io_recipe_label(self, letters: list):
        """Update the IO recipe status label."""
        text = ",".join(letters)
        self._io_recipe_lbl.setText(f"IO recipe: {text}")
        self._io_recipe_lbl.setStyleSheet("color:#aaddff;font-size:9px")

    def set_grid_changed_callback(self, cb):
        """Register callable(list[str]) invoked when grid letters text changes."""
        self._grid_changed_cb = cb

    def grid_letters(self) -> list:
        """Returns list of exactly 9 strings. Empty string = skip."""
        result = []
        for c in self._slot_cells:
            v = c.text().strip().upper()
            result.append("" if v == "" else v)
        return result

    def set_slot_cells(self, letters: list):
        """Populate all 9 slot cells from an external list (e.g. IO recipe)."""
        parts = list(letters)
        while len(parts) < 9:
            parts.append("")
        for i, cell in enumerate(self._slot_cells):
            cell.blockSignals(True)
            val = parts[i].strip().upper()
            cell.setText(val)
            cell.blockSignals(False)
        self._on_grid_cell_changed()


# =========================================================
# MAIN WINDOW  — UI mode switching only
# =========================================================
class MainWindow(QtWidgets.QWidget):
    def __init__(self, image: np.ndarray = None):
        super().__init__()
        self.setWindowTitle("IC Frame Inspection")

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width()  * 0.95), int(screen.height() * 0.95))
        self.move(screen.x() + int(screen.width()  * 0.025),
                  screen.y() + int(screen.height() * 0.025))

        self._image = image.copy() if image is not None else None
        self._io_obj    = ImageIO()
        self._sm    = SettingsManager(SETUP_FILE)
        self._grid_changed_cb = None
        self._ctrl  = InspectionController(self._sm)
        self._current_grid: list = []
        self._io_recipe:    list = []        # last recipe received from IO
        self._run_from_io:  bool = False     # True = run triggered by MachineIO
        
        # General draw mode
        self._mode    = None   # "frame" | "font" | "mask" | None
        self._pending = None   # pending font template name

        # Frame template creation state
        self._frame_rects: list             = [None, None, None]
        self._frame_panel: FrameTemplatePanel | None = None
        self._FRAME_TAGS  = ["MOLD_A", "MOLD_B", "ANCHOR", "PIN_A", "PIN_B"]
        self._yolo_pair_offset: int         = 0    # retry cycles through pairs

        # Mask work-in-progress
        self._mask_wip    = None
        self._mask_is_add = True
        
        # Run worker state
        self._worker:  RunWorker   | None = None
        self._camera:  BaslerCamera | None = None
        self._machine_io       = MachineIO()

        # Pre-load mask once
        self._run_mask: np.ndarray | None = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Camera init (production mode only)
        if not DEBUG_MODE:
            self._camera = BaslerCamera(
                serial      = CAMERA_SERIAL,
                exposure_us = self._sm.get("camera_exposure_us"),
            )
            if not self._camera.open():
                print("[Main] Camera open failed.")
                
        if not os.path.exists(SETTINGS_FILE):
            self._sm.save()

        self._build_ui()

        if image is not None:
            self._view.set_image(self._image)

        if self._ctrl.has_frame_recipe():
            self._panel.log("Frame recipe found on disk.", "#aaddff")

    # ----------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        left.addLayout(self._build_toolbar())

        self._mask_bar = MaskingToolbar()
        self._mask_bar.setVisible(False)
        self._mask_bar.sig_add.connect(self._mask_set_add)
        self._mask_bar.sig_subtract.connect(self._mask_set_sub)
        self._mask_bar.sig_complete.connect(self._mask_complete)
        left.addWidget(self._mask_bar)

        self._view = ImageView()
        self._view.roi_selected.connect(self._on_roi)
        self._view.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)
        left.addWidget(self._view)
        root.addLayout(left, stretch=1)

        self._panel = RightPanel(self._sm)
        self._panel.set_grid_changed_callback(self._on_grid_letters_updated)
        root.addWidget(self._panel)

    def _build_toolbar(self):
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)

        def btn(label, slot, bg=None):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            if bg:
                b.setStyleSheet(
                    f"background:{bg};color:#fff;"
                    f"font-weight:bold;padding:3px 8px")
            bar.addWidget(b)
            return b

        def sep():
            f = QtWidgets.QFrame()
            f.setFrameShape(QtWidgets.QFrame.VLine)
            f.setFrameShadow(QtWidgets.QFrame.Sunken)
            bar.addWidget(f)

        btn("Open Image",        self._open_image)
        sep()
        btn("Create Frame Tmpl", self._start_frame,   "#1155aa")
        btn("Create Font Tmpl",  self._start_font,    "#116611")
        btn("Masking Template",  self._start_masking, "#333388")
        sep()
        self._btn_run  = btn("▶ Start Run", self._start_run, "#1a6b2a")
        self._btn_stop = btn("■ Stop",      self._stop_run,  "#882200")
        self._btn_stop.setEnabled(False)
        btn("Clear", self._clear)
        sep()
        btn("⚙ Load Settings",  self._load_settings, "#555500")
        btn("Save Settings",  self._save_settings, "#555500")
        sep()
        bar.addStretch()
        return bar

    # ----------------------------------------------------------
    # Open image
    # ----------------------------------------------------------
    def _open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.bmp *.png *.jpg *.tif)")
        if not path:
            return
        try:
            img = self._io_obj.load(path)
            self._image = img.copy()
            self._view.set_image(self._image)
            self._view.clear_overlays()
            self._panel.log(
                f"Loaded: {os.path.basename(path)} "
                f"({img.shape[1]}x{img.shape[0]})", "#88ff88")
        except Exception as e:
            self._panel.log(f"Load error: {e}", "#ff4444")

        
    # ----------------------------------------------------------
    # Frame template creation — YOLO auto-detect single step
    # ----------------------------------------------------------
    def _start_frame(self):
        if self._image is None:
            self._panel.log("No image loaded.", "#ffaa44"); return
        if self._mode is not None:
            self._panel.log("Finish current operation first.", "#ffaa44"); return

        letters = self._panel.grid_letters()
        if not any(letters):
            QtWidgets.QMessageBox.warning(
                self, "Grid Letters Empty",
                "Please fill in at least one letter in the\n"
                "'Grid Letters' field before creating a template.\n\n"
                "Example:  ,9,H,B,,6,W,8,7")
            return

        self._frame_rects      = [None, None, None]
        self._yolo_pair_offset = 0
        self._mode             = "frame"
        self._clear_frame_overlays()

        self._frame_panel = FrameTemplatePanel(
            on_confirm = self._on_frame_confirm,
            on_retry   = self._on_frame_retry,
            on_cancel  = self._on_frame_cancel,
            parent     = self,
        )
        geo = self.geometry()
        pw  = self._frame_panel.sizeHint().width()
        self._frame_panel.move(geo.right() - pw - 20, geo.top() + 60)
        self._frame_panel.show()

        self._panel.log("Auto-detecting mold…", "#00e5ff")
        self._run_yolo_detection()

    def _run_yolo_detection(self):
        """Detect all molds; use top pair (A above B) to build anchor + pin overlays."""
        if not self._ctrl.yolo.is_ready():
            self._frame_panel.set_no_model()
            self._panel.log(
                f"YOLO model not ready — place {YOLO_MODEL_XML} in project folder.",
                "#ff4444")
            return

        all_rects = self._ctrl.yolo.detect_all(self._image)
        self._clear_frame_overlays()

        offset = self._yolo_pair_offset
        needed = offset + 2               # need at least offset+1 and offset+2

        if len(all_rects) < needed:
            self._frame_rects[0] = None
            self._frame_rects[1] = None
            self._frame_panel.set_no_detection()
            self._panel.log(
                f"YOLO: only {len(all_rects)} mold(s) detected, "
                f"need ≥{needed} for pair offset {offset}.", "#ffaa44")
            return

        rect_a = all_rects[offset]
        rect_b = all_rects[offset + 1]
        self._frame_rects[0] = rect_a
        self._frame_rects[1] = rect_b

        aw, ah  = rect_a.width(), rect_a.height()
        ax, ay  = rect_a.x(),    rect_a.y()
        by      = rect_b.y()
        a_cy    = ay + ah // 2
        b_cy    = by + ah // 2

        # anchor — mold width, right edge flush with mold A left edge
        anc_cy = (a_cy + b_cy) // 2
        anc_w  = aw
        anchor_rect = QtCore.QRect(
            max(0, ax - aw), max(0, anc_cy - ah // 2), anc_w, ah)

        # per-mold pin strips (left of each mold, narrow strip)
        pin_w  = max(4, int(aw * 0.20))
        pin_cx = ax - int(aw * 0.05) - pin_w // 2
        pin_lx = max(0, pin_cx - pin_w // 2)
        pin_a_rect = QtCore.QRect(pin_lx, ay, pin_w, ah)
        pin_b_rect = QtCore.QRect(pin_lx, by, pin_w, ah)

        self._view.add_overlay(rect_a,      QtGui.QColor(0,   224, 255), "MOLD_A", "dash")
        self._view.add_overlay(rect_b,      QtGui.QColor(0,   180, 255), "MOLD_B", "dash")
        self._view.add_overlay(anchor_rect, QtGui.QColor(255, 220, 0),   "ANCHOR", "dash")
        self._view.add_overlay(pin_a_rect,  QtGui.QColor(255, 120, 200), "PIN_A",  "dash")
        self._view.add_overlay(pin_b_rect,  QtGui.QColor(255, 120, 200), "PIN_B",  "dash")

        desc = (f"Pair [{offset},{offset+1}]  "
                f"A: {aw}×{ah}px  pitch={by - ay}px  "
                f"total={len(all_rects)} molds")
        self._frame_panel.set_detected(desc)
        self._panel.log(f"YOLO: {desc}", "#88ff88")

    def _on_frame_retry(self):
        self._yolo_pair_offset += 1
        self._panel.log(
            f"Retrying — trying pair offset {self._yolo_pair_offset}…", "#00e5ff")
        self._run_yolo_detection()

    def _on_frame_confirm(self):
        if self._frame_rects[0] is None or self._frame_rects[1] is None:
            self._panel.log("Need both mold A and B detected before confirming.", "#ffaa44")
            return
        self._finish_frame()

    def _on_frame_cancel(self):
        self._clear_frame_overlays()
        self._view.set_constraint_rect(None)
        self._mode        = None
        self._frame_rects = [None, None, None]
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None
        self._panel.log("Frame template creation cancelled.", "#888888")

    def _finish_frame(self):
        self._view.set_constraint_rect(None)
        self._mode = None
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None

        letters = self._panel.grid_letters()

        ok = self._ctrl.save_frame_recipe(
            self._image,
            self._frame_rects[0],
            self._frame_rects[1],
            grid_letters = letters,
        )
        if ok:
            self._panel.log("Frame recipe saved (auto-derived from YOLO bbox).", "#88ff88")
            try:
                recipe = self._ctrl.load_frame_recipe()
                FrameRecipePreviewDialog(recipe, parent=self).exec_()
            except Exception:
                pass
            # Step 4: stamp expected frame positions → saves frame_layout.json
            try:
                recipe = self._ctrl.load_frame_recipe()
                dlg = FrameLayoutDialog(self._image, recipe, parent=self)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    layout = self._ctrl.load_frame_layout()
                    n = len(layout.get("frames", []))
                    self._panel.log(
                        f"Frame layout saved — {n} frame(s) defined.", "#88ff88")
                else:
                    self._panel.log(
                        "Frame layout not saved (cancelled).", "#ffaa44")
            except Exception as e:
                self._panel.log(f"Frame layout step error: {e}", "#ff4444")
        else:
            self._panel.log("Frame recipe save failed.", "#ff4444")
            self._clear_frame_overlays()

    def _clear_frame_overlays(self):
        """Remove only the three frame step overlays, keep others."""
        self._view._overlays = [
            ov for ov in self._view._overlays
            if ov[2] not in self._FRAME_TAGS
        ]
        self._view.update()

    def _start_font(self):
        if self._image is None:
            self._panel.log("No image loaded.", "#ffaa44"); return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Font Template Name",
            "Letter / name (will be UPPER-CASED):")
        if not ok or not name.strip():
            return
        self._pending = name.strip().upper()
        self._mode    = "font"
        self._view.set_draw_mode(True)
        self._panel.log(
            f"Draw ROI for FONT '{self._pending}' ...", "#aaddff")
        
    # ----------------------------------------------------------
    # ROI committed by ImageView
    # ----------------------------------------------------------
    def _on_roi(self, rect: QtCore.QRect):
        if self._image is None or self._mode is None:
            return

        if self._mode == "mask":
            self._on_mask_roi(rect)
            return

        if self._mode == "frame":
            self._on_frame_roi(rect)
            return

        
        # ---- font mode ----
        self._view.set_draw_mode(False)
        x = max(0, rect.x())
        y = max(0, rect.y())
        w = min(rect.width(),  self._image.shape[1] - x)
        h = min(rect.height(), self._image.shape[0] - y)
        if w < 4 or h < 4:
            self._panel.log("ROI too small.", "#ffaa44")
            self._mode = None
            return

        roi  = self._image[y:y + h, x:x + w].copy()
        name = self._pending
        # MainWindow._on_roi() font path — derive mold_size from recipe
        mold_size = 150  # fallback
        if self._ctrl.has_frame_recipe():
            try:
                recipe   = self._ctrl.load_frame_recipe()
                mold_sec = recipe["mold_a"]
                mold_size = min(mold_sec["canvas_w"], mold_sec["canvas_h"])
            except Exception:
                pass

        ok = self._ctrl.save_font(name, roi, (x, y, w, h),
                                parent_widget = self,
                                mold_size     = mold_size)
        if ok:
            self._view.add_overlay(
                rect, QtGui.QColor(0, 210, 80), name, "solid")
            self._panel.log(
                f"Font '{name}' saved ({w}x{h} px).", "#88ff88")
        else:
            self._panel.log(f"Font '{name}' save failed.", "#ff4444")
        self._pending = None
        self._mode    = None

    def _on_frame_roi(self, _rect: QtCore.QRect):
        # Frame mode is now YOLO-only; rubber-band draws in frame mode are ignored.
        pass

    # ----------------------------------------------------------
    # Masking
    # ----------------------------------------------------------
    def _start_masking(self):
        if self._image is None:
            self._panel.log("No image loaded.", "#ffaa44"); return

        ih, iw = self._image.shape[:2]

        if os.path.exists(MASK_FILE):
            loaded = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)
            if loaded is not None and loaded.shape == (ih, iw):
                self._mask_wip = loaded.copy()
                self._panel.log("Existing mask loaded for editing.", "#aaddff")
            else:
                self._mask_wip = np.full((ih, iw), 255, dtype=np.uint8)
                self._panel.log(
                    "Mask size mismatch — starting fresh.", "#ffaa44")
        else:
            self._mask_wip = np.full((ih, iw), 255, dtype=np.uint8)
            self._panel.log(
                "Starting new mask (all white = fully searchable).", "#aaddff")

        self._mask_is_add = True
        self._mode        = "mask"
        self._mask_bar.setVisible(True)
        self._mask_bar.set_add_mode()
        self._view.set_mask_draw_mode(True, add=True)
        self._panel.log(
            "Masking mode: draw rectangles.  Add = allow  Subtract = block.",
            "#ffcc00")

    def _mask_set_add(self):
        self._mask_is_add = True
        self._view.set_mask_draw_mode(True, add=True)
        self._panel.log("Brush -> ADD (white / searchable)", "#80ff80")

    def _mask_set_sub(self):
        self._mask_is_add = False
        self._view.set_mask_draw_mode(True, add=False)
        self._panel.log("Brush -> SUBTRACT (black / blocked)", "#ff8080")

    def _on_mask_roi(self, rect: QtCore.QRect):
        if self._mask_wip is None:
            return
        ih, iw = self._mask_wip.shape[:2]
        x  = max(0, rect.x())
        y  = max(0, rect.y())
        x2 = min(iw, x + rect.width())
        y2 = min(ih, y + rect.height())
        if x2 <= x or y2 <= y:
            return
        self._mask_wip[y:y2, x:x2] = 255 if self._mask_is_add else 0
        op = "ADD" if self._mask_is_add else "SUBTRACT"
        self._panel.log(f"  Mask {op}: ({x},{y})->({x2},{y2})", "#cccccc")
        self._view.set_mask_draw_mode(True, add=self._mask_is_add)

    def _mask_complete(self):
        self._view.set_mask_draw_mode(False)
        self._mask_bar.setVisible(False)
        self._mode = None
        if self._mask_wip is None:
            return
        dlg = MaskConfirmDialog(self._mask_wip, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            ok = cv2.imwrite(MASK_FILE, self._mask_wip,
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                self._panel.log(
                    f"Search mask saved -> {MASK_FILE}", "#88ff88")
            else:
                self._panel.log(
                    f"ERROR: could not save mask to {MASK_FILE}", "#ff4444")
        else:
            self._panel.log(
                "Mask edit cancelled — no file written.", "#888888")
        self._mask_wip = None

    # ----------------------------------------------------------
    # Inspection run — threaded
    # ----------------------------------------------------------
    def _load_run_mask(self) -> np.ndarray | None:
        """Load and cache the binary search mask. Returns None if absent."""
        if not os.path.exists(MASK_FILE):
            return None
        ih  = ImageIO.TARGET_H
        iw  = ImageIO.TARGET_W
        m   = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (ih, iw):
            self._panel.log("Mask file missing or size mismatch — ignored.",
                            "#ffaa44")
            return None
        _, bm = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return bm

    def _start_run(self, from_io: bool = False):
        if self._worker and self._worker.isRunning():
            self._panel.log("Run already in progress.", "#ffaa44"); return
        if not self._ctrl.has_frame_recipe():
            self._panel.log("No frame recipe — create it first.", "#ffaa44"); return
        if not DEBUG_MODE and (self._camera is None or
                               not self._camera.is_open()):
            self._panel.log("Camera not open.", "#ff4444"); return

        self._run_from_io = from_io

        # Pick active grid
        grid = self._io_recipe if from_io and self._io_recipe \
               else self._panel.grid_letters()

        # Pre-flight: validate recipe + layout + templates exist
        missing = self._ctrl.prepare(grid)
        if missing:
            if "__NO_RECIPE__" in missing:
                msg = "No frame recipe found.\nCreate a frame template first."
            elif "__NO_LAYOUT__" in missing:
                msg = ("No frame layout found.\n"
                       "Create a frame template — the wizard will ask you to\n"
                       "stamp the expected frame positions (step 4).")
            elif "__LAYOUT_ERROR__" in missing:
                msg = "Failed to load frame_layout.json.\nRe-save the frame template."
            elif "__NO_GRID__" in missing:
                msg = "Grid letters are empty.\nEnter letters in the Grid Letters field."
            elif "__RECIPE_ERROR__" in missing:
                msg = "Failed to load frame recipe.\nThe file may be corrupt."
            else:
                names = "\n".join(f"  • {n}" for n in missing)
                msg = (
                    f"Missing font templates:\n{names}\n\n"
                    f"Create these templates using 'Create Font Tmpl'\n"
                    f"before starting the run."
                )
            QtWidgets.QMessageBox.warning(self, "Cannot Start Run", msg)
            self._panel.log(f"Run blocked — missing: {missing}", "#ff4444")
            return

        # Store search params once — worker reads from ctrl per frame
        sm = self._sm
        pin_params = {
            "score_thr": sm.get("pin_score_threshold"),
        }
        self._ctrl.set_run_params(pin_params)

        if not DEBUG_MODE and self._camera:
            self._camera.set_exposure(sm.get("camera_exposure_us"))

        mask = self._load_run_mask()

        self._worker = RunWorker(
            ctrl        = self._ctrl,
            io          = self._machine_io,
            mask        = mask,
            image_io    = self._io_obj,
            run_from_io = self._run_from_io,
            io_recipe   = list(self._io_recipe),
            ui_grid     = self._panel.grid_letters(),
            camera      = self._camera if not DEBUG_MODE else None,
            sm          = sm,
        )
        self._worker.sig_image.connect(self._view.set_image)
        self._worker.sig_result.connect(self._panel.log)
        self._worker.sig_done.connect(self._on_worker_done)
        self._worker.sig_error.connect(self._on_worker_error)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        mode = "DEBUG folder" if DEBUG_MODE else "CAMERA"
        src  = "IO" if from_io else "UI"
        self._panel.log(
            f"=== Start [{mode}] grid_src={src}"
            f"  pin_thr={pin_params['score_thr']:.2f}"
            f"  font_conf={FONT_CONFIDENCE_MIN:.2f}"
            f"  grid={','.join(grid)} ===", "#ffffff")
        self._worker.start()

    def _stop_run(self):
        if self._worker:
            self._worker.stop()
        self._panel.log("Stop requested.", "#ffaa44")
        self._btn_stop.setEnabled(False)

    def _on_worker_done(self, passed: int, total: int):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        color = "#88ff88" if passed == total and total > 0 else "#ff4444"
        self._panel.log(
            f"=== Run complete  {passed}/{total} passed ===", color)

    def _on_worker_error(self, msg: str):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._panel.log(f"ERROR: {msg}", "#ff4444")

    # ----------------------------------------------------------
    # Clear
    # ----------------------------------------------------------
    def _clear(self):
        self._view.clear_overlays()
        if self._image is not None:
            self._view.set_image(self._image)
        self._panel.log("Cleared.", "#888888")

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        if self._camera:
            self._camera.close()
        e.accept()
        
    # ----------------------------------------------------------
    # Grid letters — live update from UI or IO
    # ----------------------------------------------------------
    def _on_grid_letters_updated(self, letters: list):
        """Called whenever grid letters change (UI typing or IO signal)."""
        self._current_grid = letters
        if not self._ctrl.has_frame_recipe():
            return
        try:
            with open(self._ctrl.RECIPE_FILE, "r") as f:
                raw = json.load(f)
            raw["grid_letters"] = letters
            with open(self._ctrl.RECIPE_FILE, "w") as f:
                json.dump(raw, f)
            self._panel.log(f"Grid updated: {','.join(letters)}", "#aaddff")
        except Exception as e:
            self._panel.log(f"Grid update error: {e}", "#ff4444")

    def _on_io_recipe_received(self, letters: list):
        """Slot: called when external IO delivers a recipe."""
        self._io_recipe = list(letters)
        text = ",".join(letters)
        self._panel.set_slot_cells(letters)
        self._panel.set_io_recipe_label(letters)
        self._panel.log(f"IO recipe received: {text}", "#88ffcc")
        
    # ----------------------------------------------------------
    # Settings
    # ----------------------------------------------------------
    def _load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Settings File", "",
            "Text files (*.txt);;All files (*)")
        if not path:
            return
        try:
            self._sm.load(path)
            self._panel.apply_settings()
            self._panel.log(
                f"Settings loaded: {os.path.basename(path)}", "#aaddff")
        except Exception as e:
            self._panel.log(f"Settings load error: {e}", "#ff4444")

    def _save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Settings File", SETTINGS_FILE,
            "Text files (*.txt);;All files (*)")
        if not path:
            return
        try:
            saved = self._sm.save(path)
            self._panel.log(
                f"Settings saved: {os.path.basename(saved)}", "#88ff88")
        except Exception as e:
            self._panel.log(f"Settings save error: {e}", "#ff4444")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    pal = QtGui.QPalette()
    for role, col in [
        (QtGui.QPalette.Window,          (28,  28,  28)),
        (QtGui.QPalette.WindowText,      (220, 220, 220)),
        (QtGui.QPalette.Base,            (18,  18,  18)),
        (QtGui.QPalette.AlternateBase,   (38,  38,  38)),
        (QtGui.QPalette.Text,            (220, 220, 220)),
        (QtGui.QPalette.Button,          (48,  48,  48)),
        (QtGui.QPalette.ButtonText,      (220, 220, 220)),
        (QtGui.QPalette.Highlight,       (42,  130, 218)),
        (QtGui.QPalette.HighlightedText, (0,   0,   0)),
    ]:
        pal.setColor(role, QtGui.QColor(*col))
    app.setPalette(pal)

    io  = ImageIO()
    img = None
    candidates = io.list_images(IMAGE_SOURCE_DIR)
    if candidates:
        try:
            img = io.load(candidates[0])
            print(f"[Startup] Loaded {candidates[0]}")

        except Exception as e:
            print(f"[Startup] {e}")

    win = MainWindow(img)
    win.show()
    sys.exit(app.exec_())