"""
IC Frame Laser-Mark Inspection
================================
Structure
---------
  config       : constants + SettingsManager
  storage      : ImageIO, ContourTemplate, cv2_draw_dashed_rect
  engine       : InspectionEngine  (PIN search + font inspect)
  controller   : InspectionController  (owns engine + template store + cache)
  ui-widgets   : ImageView, FrameTemplatePanel, FrameLayoutPanel,
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
DEBUG_MODE     = True   # True = folder source; False = live camera
TEMPLATE_DEBUG = False  # True = write debug PNGs to debug/ on every template save/encode
PIN_SOBEL_MAG      = 40        # Sobel Y threshold for lead-edge detection
PIN_EDGE_RATIO     = 0.150     # min edge-pixel fraction in lead ROI
PIN_TM_STRIDE      = 4         # coarse TM grid step (px)
PIN_IOU_THR        = 0.50      # NMS overlap threshold

MIN_CONTOUR_AREA     = 1
MIN_CONTOUR_SOLIDITY = 0.35   # < 0.35 → noise; serif I ≈ 0.40
MIN_CONTOUR_EXTENT   = 0.20   # area / bbox; sparse blobs fail
MIN_CONTOUR_REL_AREA = 0.003  # sub-0.3% of ROI → speck
IMAGE_SOURCE_DIR  = "image_source/"
OUTPUT_DIR        = "Inspection_result"

CAMERA_SERIAL        = "22202392"
CAMERA_WARMUP_FRAMES = 5
CAMERA_EXPOSURE_US   = 8000   # µs — overridden by RightPanel

FONT_CONFIDENCE_MIN        = 0.70
FONT_SHIFT_RATIO_MAX       = 0.50
FONT_SHIFT_WIDE_FACTOR     = 1.5   # wide-ROI retry multiplier for shift detection
FONT_HOLE_COUNT_TOLERANCE  = 1
FONT_HOLE_AREA_TOLERANCE   = 0.30
LAST_LOT_CHIP_FRAME_COLS   = 1
MIN_TOPHAT_SIGNAL          = 20  # empty slot top-hat peak < 20 → skip Otsu

# P2 defect thresholds — normalised against canvas_area (letter pixels, not cell)
DIRTY_EXTRA_RATIO_MAX     = 0.15  # extra  > 15% → foreign object
DIRTY_MISSING_RATIO_MAX   = 0.25  # missing > 25% → broken stroke

# HOG: 64×64 win, 16×16 block, 8×8 stride, 8×8 cell, 9 bins → 1764-dim
_HOG_WIN  = (64, 64)
_HOG_DESC = cv2.HOGDescriptor(_HOG_WIN, (16, 16), (8, 8), (8, 8), 9)

def _compute_hog(canvas: np.ndarray) -> np.ndarray:
    """L2-normalised 1764-dim HOG vector; zero vector when canvas is empty."""
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
OCR_MIN_CONF       = 0.60   # below this → report "?" (unreadable)
OCR_CONF_GAP_MIN   = 0.10   # best must exceed 2nd-best by this — filters circular reflections
                             # that score similarly on "2","0","O","8" (small gap → "?")


# =========================================================
# SETTINGS MANAGER
# =========================================================
SETUP_FILE    = "Setup.json"
SETTINGS_FILE = "inspection_settings.txt"   # legacy — migrated to Setup.json on first run

# Static constants loaded from Setup.json ["static"] section.
# Values here are only the in-code fallback; Setup.json overrides them at runtime.
_SETUP_STATIC_DEFAULTS = {
    "font_confidence_min":        0.70,
    "font_shift_ratio_max":       0.50,
    "font_shift_wide_factor":     1.5,
    "font_hole_count_tolerance":  1,
    "font_hole_area_tolerance":   0.30,
    "last_lot_chip_frame_cols":   1,
    "min_tophat_signal":          20,
    "pin_edge_ratio":             0.150,
    "dirty_extra_ratio_max":      0.15,
    "dirty_missing_ratio_max":    0.25,
}

# User-tunable setup values — stored in Setup.json ["setup"] section.
# (header, default, min, max, is_float)
_SETTINGS_DEFAULTS = [
    ("pin_score_threshold",  0.75,  0.50,  1.00,  True ),
    ("ocr_conf_expected",    0.88,  0.50,  1.00,  True ),
    ("ocr_min_conf",         0.60,  0.30,  1.00,  True ),
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

    #     return self._static.get(key, default)

    def _apply_statics(self):
        """Push static section values into module-level globals."""
        global FONT_CONFIDENCE_MIN, FONT_SHIFT_RATIO_MAX, FONT_SHIFT_WIDE_FACTOR
        global FONT_HOLE_COUNT_TOLERANCE, FONT_HOLE_AREA_TOLERANCE
        global LAST_LOT_CHIP_FRAME_COLS, MIN_TOPHAT_SIGNAL, PIN_EDGE_RATIO
        global DIRTY_EXTRA_RATIO_MAX, DIRTY_MISSING_RATIO_MAX
        s = self._static
        FONT_CONFIDENCE_MIN        = float(s.get("font_confidence_min",        FONT_CONFIDENCE_MIN))
        FONT_SHIFT_RATIO_MAX       = float(s.get("font_shift_ratio_max",       FONT_SHIFT_RATIO_MAX))
        FONT_SHIFT_WIDE_FACTOR     = float(s.get("font_shift_wide_factor",     FONT_SHIFT_WIDE_FACTOR))
        FONT_HOLE_COUNT_TOLERANCE  = int(  s.get("font_hole_count_tolerance",  FONT_HOLE_COUNT_TOLERANCE))
        FONT_HOLE_AREA_TOLERANCE   = float(s.get("font_hole_area_tolerance",   FONT_HOLE_AREA_TOLERANCE))
        LAST_LOT_CHIP_FRAME_COLS   = int(  s.get("last_lot_chip_frame_cols",   LAST_LOT_CHIP_FRAME_COLS))
        MIN_TOPHAT_SIGNAL          = int(  s.get("min_tophat_signal",          MIN_TOPHAT_SIGNAL))
        PIN_EDGE_RATIO             = float(s.get("pin_edge_ratio",             PIN_EDGE_RATIO))
        DIRTY_EXTRA_RATIO_MAX      = float(s.get("dirty_extra_ratio_max",      DIRTY_EXTRA_RATIO_MAX))
        DIRTY_MISSING_RATIO_MAX    = float(s.get("dirty_missing_ratio_max",    DIRTY_MISSING_RATIO_MAX))

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
                    try:
                        val = float(entry["value"])
                    except Exception:
                        print(f"[Settings] Invalid value for {hdr}, using default")
                        continue
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
    def _morph_font(binary: np.ndarray, use_close: bool = True) -> np.ndarray:
        """
        use_close=True  (template saving):  OPEN → CLOSE — idealized clean shape.
        use_close=False (runtime detection): OPEN only    — preserves defect gaps/protrusions.
        """
        k = 3
        out = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        if use_close:
            out = cv2.morphologyEx(
                out, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        return out

    # =========================================================
    # SHARED CONTOUR FINDER
    # =========================================================

    @staticmethod
    def _filter_valid_roots(raw_cnts: list,
                            hierarchy: np.ndarray,
                            roi_area:  int) -> list:
        """
        Return (area, index) pairs for all top-level contours that pass
        the texture-noise filters (area, relative-area, solidity, extent).
        Shared by _find_contours and _find_contours_all.
        """
        result = []
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] != -1:
                continue
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
            result.append((area, i))
        return result

    @staticmethod
    def _find_contours(binary: np.ndarray, h: int, w: int) -> tuple:
        raw_cnts, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # was TC89_KCOS

        empty_canvas = np.zeros((h, w), dtype=np.uint8)

        if not raw_cnts or hierarchy is None:
            return [], empty_canvas

        hierarchy = hierarchy[0]

        valid = ContourTemplate._filter_valid_roots(raw_cnts, hierarchy, max(h * w, 1))
        if not valid:
            return [], empty_canvas

        _, best_idx = max(valid, key=lambda x: x[0])

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

        valid_roots = ContourTemplate._filter_valid_roots(
            raw_cnts, hierarchy, max(h * w, 1))

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
                              debug_prefix: str = "",
                              thresh:       "np.ndarray | None" = None,
                              use_close:    bool = True) -> tuple:
        """
        Full pipeline for FONT / LETTER ROIs.
          _thresh_font → _morph_font(use_close) → _find_contours_all

        thresh     : precomputed Otsu binary — pass to avoid redundant _thresh_font call.
        use_close  : True  (template saving)  → OPEN+CLOSE, idealized shape.
                     False (runtime detection) → OPEN only, defect gaps preserved.

        Output: (main_list, canvas_main, clean_binary, others)
        """
        h, w  = gray.shape[:2]
        if thresh is None:
            thresh = ContourTemplate._thresh_font(gray, mold_size)
        clean = ContourTemplate._morph_font(thresh, use_close=use_close)

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
        contours, canvas, _ = ContourTemplate.extract_frame_template(
            roi_gray,
            debug_prefix=f"debug/frame_{rect_xywh[0]}_{rect_xywh[1]}" if TEMPLATE_DEBUG else "")

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
            debug_prefix=f"debug/{name}" if TEMPLATE_DEBUG else "")

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

        # Pre-aligned 64×64 canvas — avoids repeated _centre_align(tmpl_canvas)
        # inside every compare_roi / _check_dirty / _check_similarity call.
        data["canvas_aligned"] = InspectionEngine._centre_align(data["canvas"]) \
                                  if data.get("canvas") is not None \
                                  else np.zeros((InspectionEngine._ALIGN_SIZE,
                                                 InspectionEngine._ALIGN_SIZE), dtype=np.uint8)

        # Pre-compute hole ratios — avoids recomputing contourArea per slot.
        if data["contours"]:
            _outer_area = cv2.contourArea(data["contours"][0])
            _holes      = data["contours"][1:]
            data["tmpl_outer_area"]  = _outer_area
            data["tmpl_hole_ratios"] = [
                cv2.contourArea(h) / max(_outer_area, 1) for h in _holes
            ]
        else:
            data["tmpl_outer_area"]  = 0.0
            data["tmpl_hole_ratios"] = []

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
                        mold_size: int,
                        thresh:    "np.ndarray | None" = None) -> tuple:
        """
        Step 1 — Presence.  Runtime path: OPEN-only morph preserves defect gaps.

        thresh : precomputed Otsu binary from _thresh_font — eliminates redundant call.
        Output : (contours, canvas, clean_binary, others)
        """
        contours, canvas, clean_binary, others = ContourTemplate.extract_font_template(
            gray, mold_size=mold_size, thresh=thresh, use_close=False)
        return contours, canvas, clean_binary, others

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
                          tmpl_canvas: np.ndarray,
                          tc:          np.ndarray = None) -> float:
        """
        Centre-aligned IoU at _ALIGN_SIZE resolution.
        Positional offset removed — shift is checked separately.

        Input : canvas (from _check_presence), tmpl_canvas (from template dict),
                tc — optional pre-aligned template canvas (from template["canvas_aligned"])
        Output: float 0.0–1.0
        """
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return 0.0

        rc = InspectionEngine._centre_align(canvas)
        if tc is None:
            tc = InspectionEngine._centre_align(tmpl_canvas)

        intersection = np.count_nonzero(cv2.bitwise_and(rc, tc))
        union        = np.count_nonzero(cv2.bitwise_or(rc, tc))
        return round(float(intersection / max(union, 1)), 4)
    
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

        sq = _thin(canvas_q)
        st = _thin(canvas_t)
        if sq is None or st is None:
            return 0.0

        sq = InspectionEngine._centre_align(sq)
        st = InspectionEngine._centre_align(st)

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

        filled = InspectionEngine._check_similarity(
            canvas_q, tmpl.get("canvas"), tmpl.get("canvas_aligned"))

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

        new_holes = new_contours[1:] if len(new_contours) > 1 else []
        score2    = _hole_score(new_holes)

        if score2 < 0:
            return -1.0, roi_holes, canvas, contours

        return score2, new_holes, new_canvas, new_contours

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
            cx, cy, score, w, h, _ = cand
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
    def _check_dirty(others:               list,
                     canvas:               np.ndarray,
                     tmpl_canvas:          np.ndarray,
                     roi_h:                int,
                     roi_w:                int,
                     tmpl_canvas_aligned:  np.ndarray = None) -> dict:
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
        N = InspectionEngine._ALIGN_SIZE
        _empty = np.zeros((N, N), dtype=np.uint8)

        # ── Aligned canvas comparison ─────────────────────────────────
        rc = InspectionEngine._centre_align(canvas) if canvas is not None else _empty

        # No template canvas → skip pixel diff; only secondary-contour check fires.
        # (Avoids extra_ratio=1.0 false-positive when template lacks a canvas field.)
        if tmpl_canvas is None or not np.any(tmpl_canvas):
            extra_ratio   = 0.0
            missing_ratio = 0.0
            extra         = _empty
            missing       = _empty
        else:
            tc = tmpl_canvas_aligned \
                 if tmpl_canvas_aligned is not None \
                 else InspectionEngine._centre_align(tmpl_canvas)

            union_area = max(int(np.count_nonzero(cv2.bitwise_or(rc, tc))), 1)

            # 2px dilation absorbs laser position drift without hiding real defects
            _dk    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            rc_cmp = cv2.dilate(rc, _dk)
            tc_cmp = cv2.dilate(tc, _dk)

            extra   = cv2.subtract(rc, tc_cmp)   # pixels in ROI outside dilated template
            missing = cv2.subtract(tc, rc_cmp)   # template pixels not covered by dilated ROI

            extra_ratio   = round(int(np.count_nonzero(extra))   / union_area, 4)
            missing_ratio = round(int(np.count_nonzero(missing)) / union_area, 4)

        # ── Secondary-contour check (blobs outside aligned bbox) ─────
        roi_area           = max(roi_h * roi_w, 1)
        others_max_ratio   = 0.0
        others_center_norm = None   # (nx, ny) of largest qualifying secondary contour
        for c in others:
            r = cv2.contourArea(c) / roi_area
            if r >= 0.30 and r > others_max_ratio:
                others_max_ratio = r
                M = cv2.moments(c)
                if M["m00"] > 0:
                    others_center_norm = (
                        float(M["m10"] / M["m00"]) / max(roi_w, 1),
                        float(M["m01"] / M["m00"]) / max(roi_h, 1))

        # ── Decision ─────────────────────────────────────────────────
        dirty_type = "none"
        if extra_ratio > DIRTY_EXTRA_RATIO_MAX or others_max_ratio >= 0.30:
            dirty_type = "foreign_object"
        elif missing_ratio > DIRTY_MISSING_RATIO_MAX:
            dirty_type = "missing_stroke"

        area_ratio = round(max(extra_ratio, missing_ratio, others_max_ratio), 4)

        return {
            "detected":           dirty_type != "none",
            "type":               dirty_type,
            "extra_ratio":        extra_ratio,
            "missing_ratio":      missing_ratio,
            "area_ratio":         area_ratio,
            "extra_map":          extra,    # 64×64 — pixels in ROI absent from template
            "missing_map":        missing,  # 64×64 — template pixels absent from ROI
            "others_center_norm": others_center_norm,
        }

    # =========================================================
    # P1  — Presence · Shape score · Shift
    # =========================================================

    @staticmethod
    def _identify_slot(contours:   list,
                       canvas:     np.ndarray,
                       tmpl:       dict,
                       exp_dx:     int,
                       exp_dy:     int,
                       mold_cx:    int,
                       mold_cy:    int,
                       roi_h:      int,
                       roi_w:      int) -> dict:
        """
        Input : contours + canvas from extract_font_template(use_close=False).
        Output: present, confidence, shift_px, shift_ratio
        """
        if not contours:
            return {"present": False, "confidence": 0.0,
                    "shift_px": 0.0, "shift_ratio": 0.0}

        confidence = InspectionEngine._compute_shape_score(canvas, contours, tmpl)
        shift_px, shift_ratio = InspectionEngine._check_shift(
            contours, tmpl, exp_dx, exp_dy, mold_cx, mold_cy, roi_w, roi_h)

        return {"present":     True,
                "confidence":  confidence,
                "shift_px":    shift_px,
                "shift_ratio": shift_ratio}

    # =========================================================
    # P2  — Defect check (clean vs canvas diff)
    # =========================================================

    @staticmethod
    def _defect_scan_slot(canvas: np.ndarray, clean: np.ndarray) -> dict:
        """
        Input : canvas (main contour fill) and clean (full morphed binary),
                both from extract_font_template(use_close=False) on the same slot.
        Steps:
          1. temp   = canvas - clean   → missing strokes (filled but no actual pixels)
          2. diff   = clean  - canvas  → extra material  (actual pixels outside fill)
          3. merged = temp | diff      → all difference pixels
          4. morph open (3×3)          → wipe single-pixel noise
        Ratios are normalised against canvas_area (letter size, not cell size).
        """
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        temp   = cv2.subtract(canvas, clean)           # 1 — missing
        diff   = cv2.subtract(clean,  canvas)           # 2 — extra
        merged = cv2.bitwise_or(temp, diff)             # 3 — all diff
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, _k)   # 4 — wipe noise

        canvas_area   = max(cv2.countNonZero(canvas), 1)
        missing_ratio = round(cv2.countNonZero(cv2.bitwise_and(temp, merged)) / canvas_area, 4)
        extra_ratio   = round(cv2.countNonZero(cv2.bitwise_and(diff, merged)) / canvas_area, 4)

        if extra_ratio > DIRTY_EXTRA_RATIO_MAX:
            defect_type = "foreign_object"
        elif missing_ratio > DIRTY_MISSING_RATIO_MAX:
            defect_type = "missing_stroke"
        else:
            defect_type = "none"

        return {
            "detected":      defect_type != "none",
            "type":          defect_type,
            "extra_ratio":   extra_ratio,
            "missing_ratio": missing_ratio,
            "area_ratio":    round(max(extra_ratio, missing_ratio), 4),
            "merged_map":    merged,
        }

    # =========================================================
    # COMPARE ROI  — orchestrates the pipeline
    # =========================================================


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

    COLOR_FRAME = (180, 150, 70)   # steel-blue (BGR) — template structure
    COLOR_MOLD  = (150, 110, 60)   # dim steel-blue (BGR) — template structure
    COLOR_PASS  = (50,  210, 50)   # green (BGR) — pass annotation
    COLOR_FAIL  = (50,  50,  210)  # red (BGR) — fail annotation
    COLOR_OCR   = (190, 190, 190)  # light gray (BGR) — OCR label

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
    # @staticmethod
    # def draw_last_lot_flag(display:   np.ndarray,
    #                        chip_cols: int,
    #                        acx:       int,
    #                        acy:       int,
    #                        aw:        int,
    #                        ah:        int):
    #     """
    #     Draw a prominent amber "LAST LOT" banner on the mold area.

    #     Input : display (BGR ndarray, modified in-place),
    #             chip_cols — number of leading columns that have chips (1 or 2),
    #             acx/acy — mold centre, aw/ah — mold size.
    #     """
    #     ih, iw = display.shape[:2]
    #     ax1 = max(0,    acx - aw // 2)
    #     ay1 = max(0,    acy - ah // 2)
    #     ax2 = min(iw-1, ax1 + aw)
    #     ay2 = min(ih-1, ay1 + ah)

    #     # Amber semi-transparent fill over mold area
    #     overlay = display.copy()
    #     cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (0, 140, 255), cv2.FILLED)
    #     cv2.addWeighted(overlay, 0.22, display, 0.78, 0, display)

    #     # Solid amber border
    #     cv2.rectangle(display, (ax1, ay1), (ax2, ay2), (0, 165, 255), 2)

    #     # Label centred on mold
    #     lbl   = f"LAST LOT ({chip_cols}/3 col)"
    #     font  = cv2.FONT_HERSHEY_SIMPLEX
    #     fscl  = 0.45
    #     thick = 1
    #     (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
    #     tx = acx - tw // 2
    #     ty = acy + th // 2
    #     cv2.rectangle(display, (tx - 3, ty - th - bl), (tx + tw + 3, ty + bl),
    #                   (0, 0, 0), cv2.FILLED)
    #     cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 165, 255), thick)

    # ---- Ignored frame (last-lot empty column) ----------------------
    @staticmethod
    def draw_ignored_frame(display: np.ndarray,
                           acx: int, acy: int):
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
        cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (80, 65, 45), cv2.FILLED)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.rectangle(display, (ax1, ay1), (ax2, ay2), ResultAnnotator.COLOR_MOLD, 1)

        lbl   = f"{ic_id} DROP" if ic_id else "DROP"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscl  = 0.55
        thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = acx - tw // 2
        ty = acy + th // 2
        cv2.rectangle(display, (tx - 3, ty - th - bl), (tx + tw + 3, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, ResultAnnotator.COLOR_MOLD, thick, cv2.LINE_AA)

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
        cv2_draw_dashed_rect(display, (x1, y1), (x2, y2), ResultAnnotator.COLOR_FAIL, 2)
        lbl  = f"{frame_id} MISSING" if frame_id else "MISSING"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscl = 0.50; thick = 1
        (tw, th), bl = cv2.getTextSize(lbl, font, fscl, thick)
        tx = (x1 + x2) // 2 - tw // 2
        ty = (y1 + y2) // 2 + th // 2
        cv2.rectangle(display, (tx - 2, ty - th - bl), (tx + tw + 2, ty + bl),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(display, lbl, (tx, ty), font, fscl, ResultAnnotator.COLOR_FAIL, thick, cv2.LINE_AA)

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
        _, iw = display.shape[:2]
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
        (tw, th), _ = cv2.getTextSize(lbl, font, fscl, thick)
        tx = iw // 2 - tw // 2
        ty = bar_h // 2 + th // 2
        cv2.putText(display, lbl, (tx, ty), font, fscl, (0, 255, 255), thick, cv2.LINE_AA)

    # ---- Defect circle highlight (extracted from draw_letter) -------

    @staticmethod
    def _draw_defect_highlight(display: np.ndarray, result: dict,
                               lx1: int, ly1: int,
                               cell_w: int, cell_h: int):
        """Draw one defect circle per contour cluster on a failed letter cell."""
        dirty_info  = result.get("dirty", {})
        d_type      = dirty_info.get("type", "none")
        roi_canvas  = result.get("roi_canvas")
        tmpl_canvas = result.get("tmpl_canvas")
        fail_color  = ResultAnnotator.COLOR_FAIL

        def _circles_from_map(dmap) -> bool:
            if dmap is None or not dmap.size or not np.any(dmap):
                return False
            cnts, _ = cv2.findContours(dmap, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            drew = False
            for cnt in cnts:
                if cv2.contourArea(cnt) < 2:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx = lx1 + int((x + w / 2) * cell_w / dmap.shape[1])
                cy = ly1 + int((y + h / 2) * cell_h / dmap.shape[0])
                r  = max(4, int(np.hypot(w * cell_w / dmap.shape[1],
                                         h * cell_h / dmap.shape[0]) * 0.40))
                cv2.circle(display, (cx, cy), r, fail_color, 2)
                drew = True
            return drew

        merged_map = dirty_info.get("merged_map")
        if merged_map is not None and np.any(merged_map):
            _circles_from_map(merged_map)
            return

        if d_type == "foreign_object":
            extra_map = dirty_info.get("extra_map")
            if extra_map is not None and np.any(extra_map):
                _circles_from_map(extra_map)
            else:
                norm_pt = dirty_info.get("others_center_norm")
                if norm_pt is not None:
                    cx = lx1 + int(norm_pt[0] * cell_w)
                    cy = ly1 + int(norm_pt[1] * cell_h)
                    cv2.circle(display, (cx, cy),
                               max(4, int(min(cell_w, cell_h) * 0.12)),
                               fail_color, 2)
        elif d_type == "missing_stroke":
            _circles_from_map(dirty_info.get("missing_map"))
        elif d_type == "hole_mismatch":
            if roi_canvas is not None and tmpl_canvas is not None:
                diff = cv2.subtract(InspectionEngine._centre_align(tmpl_canvas),
                                    InspectionEngine._centre_align(roi_canvas))
                if np.any(diff):
                    _circles_from_map(diff)

    # ---- Letter ------------------------------------------------------
    @staticmethod
    def draw_letter(display: np.ndarray, result: dict):
        passed     = result["pass"]
        letter     = result["letter"]
        confidence = result["confidence"]
        lx1        = result["lx1"]
        ly1        = result["ly1"]
        lx2        = result["lx2"]
        ly2        = result["ly2"]
        ocr_char   = result.get("ocr_char", "?")
        ocr_conf   = result.get("ocr_conf",  0.0)
        roi_thresh = result.get("roi_thresh")

        col    = ResultAnnotator.COLOR_PASS if passed else ResultAnnotator.COLOR_FAIL
        cell_w = lx2 - lx1
        cell_h = ly2 - ly1

        if not passed and cell_w > 0 and cell_h > 0:
            ResultAnnotator._draw_defect_highlight(
                display, result, lx1, ly1, cell_w, cell_h)

        # ── Extracted edges — threshold contours in pass/fail color ──
        if roi_thresh is not None and roi_thresh.size > 0 and cell_w > 0 and cell_h > 0:
            rrs = cv2.resize(roi_thresh, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            rcnts, _ = cv2.findContours(rrs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display[ly1:ly2, lx1:lx2], rcnts, -1, col, 1)

        # ── Label with black background ───────────────────────────
        lbl = f"{letter}: {confidence:.2f}" if letter else f"!{ocr_char}: {ocr_conf:.2f}"
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
        """Return anchor section. Uses cached recipe from prepare() when available."""
        r = self._active_recipe if getattr(self, "_active_recipe", None) else self.load_frame_recipe()
        return r["anchor"]

    def get_mold_offsets(self) -> tuple:
        """Return (mold_a_shift, mold_b_shift). Uses cached recipe from prepare()."""
        r = self._active_recipe if getattr(self, "_active_recipe", None) else self.load_frame_recipe()
        return r["mold_a_shift"], r["mold_b_shift"]

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
            TemplatePreviewDialog(roi_bgr, name, mold_size=mold_size, parent=parent_widget).exec_()
            return True
        except Exception as e:
            print(f"[Controller] Font '{name}' save error: {e}")
            return False

    # def list_fonts(self) -> list:
    #     return self._ct.list_templates()

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
    
    def _run_ocr(self, canvas: np.ndarray) -> tuple:
        """
        HOG cosine match of canvas against all loaded OCR templates.

        Returns (char, confidence):
          - Fast path  : best_conf ≥ OCR_CONF_EXPECTED → return immediately.
          - Gap check  : best − 2nd < OCR_CONF_GAP_MIN → ambiguous → "?".
          - Floor check: best_conf < OCR_MIN_CONF → unreadable → "?".
          - "?"        : no templates loaded, canvas is None, or above checks fail.
        """
        if not self._ocr_templates or canvas is None:
            return "?", 0.0

        q_hog  = _compute_hog(canvas)
        scores = sorted(
            ((InspectionEngine._hog_cosine(q_hog, t.get("hog_vec")), n)
             for n, t in self._ocr_templates.items()
             if t.get("hog_vec") is not None),
            reverse=True,
        )
        if not scores:
            return "?", 0.0

        best_conf, best_char = scores[0]
        conf_expected = self._sm.get("ocr_conf_expected")
        conf_min      = self._sm.get("ocr_min_conf")

        if best_conf >= conf_expected:
            return best_char, round(best_conf, 4)

        if len(scores) >= 2 and (best_conf - scores[1][0]) < OCR_CONF_GAP_MIN:
            return "?", round(best_conf, 4)

        if best_conf < conf_min:
            return "?", round(best_conf, 4)

        return best_char, round(best_conf, 4)

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

        # Deduplicate: ['A','B','A','A'] → ['A','B']
        unique = sorted(set(c.upper() for c in active_grid))
        print(f"[Prepare] Grid chars (unique): {unique}")

        # Explicit file-existence check before attempting to load
        tmpl_dir = self._ct.TEMPLATE_DIR
        missing_tmpls = [c for c in unique
                         if not os.path.exists(
                             os.path.join(tmpl_dir, f"{c}_template.json"))]
        if missing_tmpls:
            print(f"[Prepare] Missing templates: {missing_tmpls}")
            return missing_tmpls

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
        Last-frame-fed check.

        Condition (LAST_LOT_CHIP_FRAME_COLS = N):
          • Exactly N columns where EVERY active slot qualifies as last-fed:
              - reason == "dropped", OR
              - pass == False AND ocr_char in (" ", "?")
          • Total detected columns > N (at least one other column exists).
          • Column position is not constrained.
          • Other columns may contain PASS / normal NG — they stay active.

        Those N columns are marked ignored by the caller.
        Returns (is_last_lot, ll_count, total_cols, fidx_to_col, ll_col_set).
        """
        if not matches:
            return False, 0, 0, {}, set()

        cg_to_fidx: dict = defaultdict(list)
        for f_idx, (anc_cx, *_) in enumerate(matches):
            cg = round(anc_cx / max(col_snap, 1))
            cg_to_fidx[cg].append(f_idx)

        sorted_cg = sorted(cg_to_fidx.keys())
        n_cols    = len(sorted_cg)
        n_chip    = LAST_LOT_CHIP_FRAME_COLS

        if n_cols <= n_chip:
            return False, 0, n_cols, {}, set()

        # Build reverse map: f_idx → 0-based column index
        fidx_to_col: dict = {}
        for col_i, cg in enumerate(sorted_cg):
            for fi in cg_to_fidx[cg]:
                fidx_to_col[fi] = col_i

        # Collect active slots per column (letter assigned, contours found incl. dropped)
        col_slots: dict = {i: [] for i in range(n_cols)}
        for r in all_results:
            fi = r.get("frame_idx", 0) - 1          # frame_idx is 1-based
            if fi not in fidx_to_col:
                continue
            if r.get("letter", "") != "" and r.get("defect_step", 0) != 1:
                col_slots[fidx_to_col[fi]].append(r)

        # A column is last-fed when it has slots AND every slot is Drop or NG-unreadable
        ll_col_set: set = set()
        for col_i in range(n_cols):
            slots = col_slots[col_i]
            if not slots:
                continue
            all_last_fed = True
            for r in slots:
                if r.get("reason") == "dropped":
                    continue
                if not r.get("pass", True) and r.get("ocr_char", " ") in (" ", "?"):
                    continue
                all_last_fed = False
                break
            if all_last_fed:
                ll_col_set.add(col_i)

        if len(ll_col_set) != n_chip:
            return False, 0, n_cols, {}, set()

        return True, n_chip, n_cols, fidx_to_col, ll_col_set

    # ---- inspection pipeline ----
    def run(self,
        image_bgr: np.ndarray) -> "InspectionResult":
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
            src, anchor_tmpl, pin_params, layout)

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
        img_ll, img_chip_cols, img_total_cols, fidx_to_col, ll_col_set = \
            self._check_last_lot_image(fake_matches, self.results, col_snap)

        if img_ll:
            for r in self.results:
                r["last_lot"]      = True
                r["last_lot_cols"] = img_chip_cols
                fi = r.get("frame_idx", 0) - 1
                if fidx_to_col.get(fi, -1) in ll_col_set:
                    r["ignored"] = True

            for f_idx, fentry in enumerate(frame_results):
                if fidx_to_col.get(f_idx, -1) in ll_col_set:
                    anc_cx, anc_cy = fentry["cx"], fentry["cy"]
                    ResultAnnotator.draw_ignored_frame(display, anc_cx, anc_cy)

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

    def _step1_find_frames(self, image_gray, anchor_tmpl, pin_params, layout):
        """
        For each frame in layout (F1→FN), crop the image at the pre-defined ROI
        and run a local TM to confirm presence.

        Returns list of dicts:
          { id, roi, found, cx, cy, score, fw, fh }
        Order is always the layout order — no sorting, no index reassignment.
        """
        gray = image_gray  # caller (run()) already converts to grayscale

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

        # ── Grid geometry ─────────────────────────────────────────
        g_scale  = float(self._sm.get("grid_scale"))
        g_x_frac = float(self._sm.get("grid_x_frac"))
        g_y_frac = float(self._sm.get("grid_y_frac"))
        grid_cx  = acx + int(mold_w * g_x_frac)
        grid_cy  = acy + int(mold_h * g_y_frac)
        cell_w   = int(mold_w * g_scale / 3)
        cell_h   = int(mold_h * g_scale / 3)
        roi_w    = int(cell_w * 1.2)
        roi_h    = int(cell_h * 1.2)

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
                     if image_bgr.ndim == 3 else image_bgr

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

        mold_x1 = max(0,  acx - mold_w // 2)
        mold_x2 = min(iw, acx + mold_w // 2)
        mold_y1 = max(0,  acy - mold_h // 2)
        mold_y2 = min(ih, acy + mold_h // 2)

        for slot_idx, letter in enumerate(grid_letters):
            letter = letter.upper() if letter else ""
            if not letter:
                continue

            row = slot_idx // 3
            col = slot_idx  % 3
            dx  = (col - 1) * cell_w
            dy  = (row - 1) * cell_h

            cell_cx = grid_cx + dx
            cell_cy = grid_cy + dy

            lx1 = max(mold_x1, cell_cx - roi_w // 2)
            ly1 = max(mold_y1, cell_cy - roi_h // 2)
            lx2 = min(mold_x2, lx1 + roi_w)
            ly2 = min(mold_y2, ly1 + roi_h)

            if lx2 <= lx1 or ly2 <= ly1:
                continue

            tmpl = self.cache.get(letter)
            if tmpl is None:
                continue

            slot_gray = image_gray[ly1:ly2, lx1:lx2]

            t0 = time.perf_counter()
            contours, canvas, clean, _ = ContourTemplate.extract_font_template(
                slot_gray,
                mold_size = tmpl.get("mold_size", mold_size),
                use_close = False)

            p1 = InspectionEngine._identify_slot(
                contours, canvas, tmpl,
                dx, dy, acx, acy,
                slot_gray.shape[0], slot_gray.shape[1])

            p2 = (InspectionEngine._defect_scan_slot(canvas, clean)
                  if p1["present"]
                  else {"detected": False, "type": "none",
                        "extra_ratio": 0.0, "missing_ratio": 0.0, "area_ratio": 0.0})

            ocr_char, ocr_conf = self._run_ocr(canvas) if p1["present"] else ("?", 0.0)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Collect failures ──────────────────────────────────
            failures = []
            if not p1["present"]:
                # Wide-ROI retry: expands the search window by FONT_SHIFT_WIDE_FACTOR
                # to catch marks displaced outside the normal cell ROI by a large shift.
                # Clamped to full image bounds (not mold bounds) so nothing is missed.
                wide_w    = int(roi_w * FONT_SHIFT_WIDE_FACTOR)
                wide_h    = int(roi_h * FONT_SHIFT_WIDE_FACTOR)
                wlx1      = max(0,  cell_cx - wide_w // 2)
                wly1      = max(0,  cell_cy - wide_h // 2)
                wlx2      = min(iw, wlx1 + wide_w)
                wly2      = min(ih, wly1 + wide_h)
                shift_found = False

                if wlx2 > wlx1 and wly2 > wly1:
                    wide_gray = image_gray[wly1:wly2, wlx1:wlx2]
                    w_cnts, w_canvas, _, _ = ContourTemplate.extract_font_template(
                        wide_gray,
                        mold_size = tmpl.get("mold_size", mold_size),
                        use_close = False)

                    if w_cnts:
                        w_p1 = InspectionEngine._identify_slot(
                            w_cnts, w_canvas, tmpl,
                            dx, dy, acx, acy,
                            wide_gray.shape[0], wide_gray.shape[1])
                        # Promote wide-ROI data — mark is present but shifted
                        p1         = w_p1
                        canvas     = w_canvas
                        ocr_char, ocr_conf = self._run_ocr(w_canvas)
                        shift_found = True
                        failures.append((3,
                            f"shift ratio={w_p1['shift_ratio']:.3f}"
                            f" > {FONT_SHIFT_RATIO_MAX} (wide-retry)"))

                if not shift_found:
                    failures.append((1, "missing_mark"))
            else:
                if p2["detected"]:
                    failures.append((2,
                        f"{p2['type']}(area={p2['area_ratio']:.3f})"))
                if p1["shift_ratio"] > FONT_SHIFT_RATIO_MAX:
                    failures.append((3,
                        f"shift ratio={p1['shift_ratio']:.3f}"
                        f" > {FONT_SHIFT_RATIO_MAX}"))
                if p1["confidence"] < FONT_CONFIDENCE_MIN:
                    failures.append((4,
                        f"low_conf={p1['confidence']:.3f}"
                        f" < {FONT_CONFIDENCE_MIN}"))
                if ocr_char != "?" and ocr_char != letter and ocr_conf >= self._sm.get("ocr_min_conf"):
                    failures.append((5,
                        f"wrong_mark ocr={ocr_char}({ocr_conf:.2f})"
                        f" expected={letter}"))
            failures.sort(key=lambda x: x[0])

            passed      = len(failures) == 0
            defect_step = failures[0][0] if failures else 0
            reasons     = [r for _, r in failures] if failures else ["OK"]

            results.append({
                "frame_idx":    f_idx + 1,
                "mold":         mold_label,
                "slot":         slot_idx,
                "letter":       letter,
                "pass":         passed,
                "confidence":   p1["confidence"],
                "shift_px":     p1["shift_px"],
                "shift_ratio":  p1["shift_ratio"],
                "defect_step":  defect_step,
                "defect_steps": [s for s, _ in failures] if failures else [],
                "defect_type":  p2["type"] if p2["detected"] else "",
                "reason":       reasons[0],
                "reasons":      reasons,
                "ocr_char":     ocr_char,
                "ocr_conf":     ocr_conf,
                "cell_cx":      cell_cx,
                "cell_cy":      cell_cy,
                "elapsed_ms":   round(elapsed_ms, 2),
                "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                "roi_canvas":   canvas,
                "dirty": {
                    "detected":      p2["detected"],
                    "type":          p2["type"],
                    "area_ratio":    p2["area_ratio"],
                    "extra_ratio":   p2["extra_ratio"],
                    "missing_ratio": p2["missing_ratio"],
                    "merged_map":    p2.get("merged_map"),
                },
            })

            if DEBUG_MODE:
                ic_f = f_idx * 2 + (1 if mold_label == "A" else 2)
                print(
                    f"[SLOT] F{ic_f} s{slot_idx+1}({letter})"
                    f" {'OK  ' if passed else 'FAIL'}"
                    f" step={defect_step}"
                    f" conf={p1['confidence']:.3f}"
                    f" shift={p1['shift_ratio']:.3f}"
                    f" defect={p2['type']}({p2['area_ratio']:.3f})"
                    f" ocr={ocr_char}({ocr_conf:.2f})"
                    f" | {'; '.join(reasons)}")

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
                 image_io:    "ImageIO",
                 run_from_io: bool = False,
                 io_recipe:   list = None,
                 ui_grid:     list = None,
                 camera:      "BaslerCamera | None" = None,
                 sm:          "SettingsManager | None" = None):
        super().__init__()
        self._ctrl        = ctrl
        self._io          = io
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
        result = self._ctrl.run(img_gray)
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _log(self, msg: str, color: str = "#dddddd"):
        self.sig_result.emit(msg, color)

    def _save_result_images(self, img_gray: np.ndarray, display_bgr: np.ndarray,
                            prefix: str = "") -> str:
        """Write <prefix><ts>_R.png + <prefix><ts>.png; return base name."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base = f"{prefix}{ts}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_R.png"), img_gray)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}.png"),   display_bgr)
        return base

    def _save_fail(self, img_gray: np.ndarray, display_bgr: np.ndarray):
        base = self._save_result_images(img_gray, display_bgr)
        self._log(f"  NG saved: {base}_R.png + {base}.png", "#ffaa44")

    def _save_last_lot(self, img_gray: np.ndarray, display_bgr: np.ndarray,
                       chip_cols: int):
        base = self._save_result_images(img_gray, display_bgr, prefix="lastlot_")
        self._log(
            f"  LAST LOT saved: {base}_R.png + {base}.png  ({chip_cols}/3 col)",
            "#ffaa00")

    def _append_csv(self, ic_groups: dict, image_name: str):
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
                        f"slot{r['slot']}({r['letter']}):"
                        f"{'; '.join(r.get('reasons', [r['reason']]))}"
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

    # ---- shared per-image result handler ----
    def _handle_result(self,
                       result:      "InspectionResult",
                       img_gray:    np.ndarray,
                       img_ms:      float,
                       image_name:  str) -> tuple:
        """
        Log per-mold verdicts, save fail/last-lot images, write CSV, fire IO.
        Returns (passed_count, total_count) for caller accumulation.
        """
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

            ignored = all(r.get("ignored") for r in group)
            no_lead = all(r.get("reason") == "dropped" for r in group)
            if ignored:
                self._log(
                    f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                    f"LAST LOT — ignored (empty col)  [{mold_ms:.1f}ms]",
                    "#666666")
                continue
            if no_lead:
                self._log(
                    f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  "
                    f"DROP WORK — skipped (pass)  [{mold_ms:.1f}ms]",
                    "#888888")
                continue

            ocr_map    = {r["slot"]: r.get("ocr_char", " ") for r in group}
            ocr_str    = "".join(ocr_map.get(i, " ") for i in range(9))
            fail_causes = [
                f"slot{r['slot']}({r['letter']}):"
                f"{'; '.join(r.get('reasons', [r['reason']]))}"
                for r in group if not r["pass"]
            ]
            cause_str = "  " + "  ".join(fail_causes) if fail_causes else ""
            self._log(
                f"  F{f_idx}-{mold_lbl} [{ic_num:>2}]  {verdict}"
                f"  [{mold_ms:.1f}ms]  OCR:\"{ocr_str}\"{cause_str}",
                color)

        self._log(
            f"  ▸ Image total: {img_ms:.1f}ms  "
            f"ICs={len(ic_groups)}  passed={result.passed}/{result.total}",
            "#aaaaaa")

        if result.passed != result.total or result.total == 0:
            self._save_fail(img_gray, result.display)

        if result.last_lot:
            self._log(
                f"  *** LAST LOT — {result.last_lot_cols}/3 col(s) filled ***",
                "#ffaa00")
            self._save_last_lot(img_gray, result.display, result.last_lot_cols)
            self._io.on_last_lot(result.last_lot_cols)

        self._append_csv(ic_groups, image_name)
        self._io.on_frame_result(result.passed, result.total)
        return result.passed, result.total

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

            p, t = self._handle_result(result, img, img_ms, fname)
            total_passed  += p
            total_letters += t

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

            ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
            p, t = self._handle_result(result, img, img_ms, f"cam_{ts_label}")
            total_passed  += p
            total_letters += t
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
# IMAGE VIEW  — zoomable label with rubber-band / stamp-mode ROI
# =========================================================
class ImageView(QtWidgets.QLabel):
    roi_selected = QtCore.pyqtSignal(QtCore.QRect)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._draw_mode  = False
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
        self._stamp_box   = 45   # preview width
        self._stamp_h     = 45   # preview height (may differ from width)
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
        self._mask_mode  = False
        self._rect = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    # def set_mask_draw_mode(self, on: bool, add: bool = True):
    #     self._mask_mode  = on
    #     self._mask_add   = add
    #     self._draw_mode  = False
    #     self._start      = None
    #     self._rect       = None
    #     self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
    #     self.update()

    def set_stamp_mode(self, on: bool,
                       box_size: int = 45, box_h: int = 0,
                       label: str = ""):
        """
        Stamp mode: a fixed-size rectangle follows the cursor.
        Click emits roi_selected with a QRect centred on the click point.
        box_h = 0 means square (box_h = box_size).
        """
        self._stamp_mode  = on
        self._stamp_box   = box_size
        self._stamp_h     = box_h if box_h > 0 else box_size
        self._stamp_label = label
        self._draw_mode   = False
        self._mask_mode   = False
        self._cursor_img  = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    def set_stamp_label(self, label: str):
        self._stamp_label = label
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
        if self._stamp_mode:
            img_pt = self._to_img(e.pos())
            hw = self._stamp_box // 2
            hh = self._stamp_h   // 2
            self.roi_selected.emit(
                QtCore.QRect(img_pt.x() - hw, img_pt.y() - hh,
                             self._stamp_box, self._stamp_h))
        elif self._draw_mode or self._mask_mode:
            self._start = e.pos()
            self._rect  = None

    def mouseMoveEvent(self, e):
        if self._stamp_mode:
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
            p.setPen(QtGui.QPen(QtGui.QColor(210, 55, 200), 2,
                                QtCore.Qt.DashLine))
            p.drawRect(wcr)
            p.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
            p.setPen(QtGui.QColor(210, 55, 200))
            p.drawText(wcr.topLeft() + QtCore.QPoint(3, -4), "DRAW MOLD HERE")
            
        if self._draw_mode and self._rect:
            p.setPen(QtGui.QPen(QtGui.QColor(215, 65, 210), 1,
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

        # Stamp mode cursor — rectangular preview attached to cursor
        if self._stamp_mode and self._cursor_img is not None:
            wcx = int(self._cursor_img.x() * self._scale) + self._offset.x()
            wcy = int(self._cursor_img.y() * self._scale) + self._offset.y()
            ww  = int(self._stamp_box * self._scale)
            wh  = int(self._stamp_h   * self._scale)
            wx  = wcx - ww // 2
            wy  = wcy - wh // 2
            p.setPen(QtGui.QPen(QtGui.QColor(210, 55, 200), 2,
                                QtCore.Qt.DashLine))
            p.drawRect(wx, wy, ww, wh)
            p.setPen(QtGui.QPen(QtGui.QColor(220, 155, 215), 1,
                                QtCore.Qt.SolidLine))
            p.drawLine(wcx - 8, wcy, wcx + 8, wcy)
            p.drawLine(wcx, wcy - 8, wcx, wcy + 8)
            if self._stamp_label:
                p.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                p.setPen(QtGui.QColor(210, 55, 200))
                p.drawText(wx + 3, wy + 13, self._stamp_label)

        p.end()
    
    def mouseReleaseEvent(self, _):
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
# FRAME TEMPLATE PANEL
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
            "font-size:13px;font-weight:bold;color:#7ab8d8")
        self._lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_title)

        self._lbl_status = QtWidgets.QLabel("Detecting…")
        self._lbl_status.setStyleSheet("color:#b0bec8;font-size:11px")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_status.setWordWrap(True)
        lay.addWidget(self._lbl_status)

        lay.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_confirm = QtWidgets.QPushButton("Save Template")
        self._btn_retry   = QtWidgets.QPushButton("Retry")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_confirm.setStyleSheet(
            "background:#1a4070;color:#d8e8f0;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_retry.setStyleSheet(
            "background:#384858;color:#d8e8f0;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_cancel.setStyleSheet(
            "background:#5a2a30;color:#d8e8f0;font-weight:bold;"
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
        self._lbl_status.setText(
            f"Mold detected:\n{rect_str}\n\nFrame (lead) box auto-derived.")
        self._lbl_status.setStyleSheet("color:#88ff88;font-size:11px")
        self._btn_confirm.setEnabled(True)

    def set_no_detection(self):
        self._lbl_status.setText(
            "No mold detected.\nCheck image or model file,\nthen click Retry.")
        self._lbl_status.setStyleSheet("color:#ffaa44;font-size:11px")
        self._btn_confirm.setEnabled(False)

    def set_no_model(self):
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
# FRAME LAYOUT PANEL
# =========================================================
class FrameLayoutPanel(QtWidgets.QWidget):
    """
    Floating always-on-top panel for step 4 of the frame template wizard:
    stamping expected frame positions directly on the main ImageView.

    Provides stamp count feedback plus Undo / Confirm / Cancel controls.
    """

    def __init__(self, roi_w: int, roi_h: int,
                 on_undo, on_confirm, on_cancel, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("Frame Layout — Stamp Positions")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setFixedWidth(340)

        self._on_undo    = on_undo
        self._on_confirm = on_confirm
        self._on_cancel  = on_cancel

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        title = QtWidgets.QLabel("Stamp Expected Frame Positions")
        title.setStyleSheet("font-size:13px;font-weight:bold;color:#7ab8d8")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(title)

        hint = QtWidgets.QLabel(
            f"Left-click the anchor centre of each expected frame (F1, F2, …).\n"
            f"ROI box: {roi_w} × {roi_h} px  (1.5 × anchor canvas).")
        hint.setStyleSheet("color:#aaa;font-size:9px")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._lbl_count = QtWidgets.QLabel("Stamps placed: 0")
        self._lbl_count.setStyleSheet("color:#888888;font-size:11px")
        self._lbl_count.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_count)

        lay.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_undo    = QtWidgets.QPushButton("Undo")
        self._btn_confirm = QtWidgets.QPushButton("Confirm & Save")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_undo.setStyleSheet(
            "background:#384858;color:#d8e8f0;padding:5px 10px;border-radius:4px")
        self._btn_confirm.setStyleSheet(
            "background:#1a4070;color:#d8e8f0;font-weight:bold;"
            "padding:5px 12px;border-radius:4px")
        self._btn_cancel.setStyleSheet(
            "background:#5a2a30;color:#d8e8f0;padding:5px 10px;border-radius:4px")

        self._btn_confirm.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._btn_undo)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_cancel)
        lay.addLayout(btn_row)

        self.adjustSize()

    def update_count(self, n: int):
        self._lbl_count.setText(f"Stamps placed: {n}")
        self._lbl_count.setStyleSheet(
            f"color:{'#7ab8d8' if n > 0 else '#888898'};font-size:11px")
        self._btn_confirm.setEnabled(n > 0)

    def closeEvent(self, e):
        self._on_cancel()
        e.accept()


# =========================================================
# TEMPLATE PREVIEW DIALOG
# =========================================================
class TemplatePreviewDialog(QtWidgets.QDialog):
    """
    Modal shown after saving a font template.
    Displays the original ROI crop alongside the extracted 64×64 canvas.
    """
    _CANVAS_SCALE = 4   # 64×64 → 256×256

    def __init__(self, roi_bgr: np.ndarray, name: str,
                 mold_size: int = 150, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Template Saved — '{name}'")
        self.setModal(True)

        # ── Extract canvas from ROI ───────────────────────────────
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) \
               if roi_bgr.ndim == 3 else roi_bgr.copy()
        _, canvas, *_ = ContourTemplate.extract_font_template(
            gray, mold_size=mold_size, use_close=True)

        # ── Build layout ──────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        row    = QtWidgets.QHBoxLayout()

        def _make_lbl(img_bgr: np.ndarray, caption: str) -> QtWidgets.QWidget:
            box = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel()
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setFixedSize(256, 256)
            if img_bgr is not None and img_bgr.size > 0:
                h, w = img_bgr.shape[:2]
                scale = min(256 / max(w, 1), 256 / max(h, 1))
                sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = cv2.resize(img_bgr, (sw, sh))
                if resized.ndim == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                qimg = QtGui.QImage(resized.data, sw, sh, sw * 3,
                                    QtGui.QImage.Format_BGR888)
                lbl.setPixmap(QtGui.QPixmap.fromImage(qimg))
            cap = QtWidgets.QLabel(caption)
            cap.setAlignment(QtCore.Qt.AlignCenter)
            cap.setStyleSheet("color:#aaaaaa;font-size:11px")
            box.addWidget(lbl)
            box.addWidget(cap)
            w_wrap = QtWidgets.QWidget()
            w_wrap.setLayout(box)
            return w_wrap

        row.addWidget(_make_lbl(roi_bgr, "ROI crop"))
        canvas_vis = (canvas * 255).astype(np.uint8) \
                     if canvas is not None and canvas.max() <= 1 else canvas
        row.addWidget(_make_lbl(canvas_vis, "Extracted template"))

        btn = QtWidgets.QPushButton("OK")
        btn.setFixedWidth(80)
        btn.clicked.connect(self.accept)

        layout.addLayout(row)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignHCenter)
        self.adjustSize()


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

        # ── OCR Confidence ────────────────────────────────────
        gb_ocr = QtWidgets.QGroupBox("OCR Confidence")
        fl_ocr = QtWidgets.QFormLayout(gb_ocr)
        self.spin_ocr_expected = self._bind_fspin("ocr_conf_expected", fl_ocr, "Expected")
        self.spin_ocr_min      = self._bind_fspin("ocr_min_conf",      fl_ocr, "Min")
        note_ocr = QtWidgets.QLabel(
            "Expected: fast-path accept if score ≥ this.\n"
            "Min: below this → unreadable (reported as '?').")
        note_ocr.setStyleSheet("color:#888;font-size:9px")
        note_ocr.setWordWrap(True)
        fl_ocr.addRow("", note_ocr)
        lay.addWidget(gb_ocr)

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
            (self.spin_pin_score,    "pin_score_threshold"),
            (self.spin_ocr_expected, "ocr_conf_expected"),
            (self.spin_ocr_min,      "ocr_min_conf"),
            (self.spin_exposure,     "camera_exposure_us"),
            (self.spin_grid_scale,   "grid_scale"),
            (self.spin_grid_x,       "grid_x_frac"),
            (self.spin_grid_y,       "grid_y_frac"),
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

    # def pin_search_params(self) -> dict:
    #     sm = self._sm
    #     return {
    #         "score_thr": sm.get("pin_score_threshold"),
    #     }

    # def font_list(self, ct: "ContourTemplate") -> list:
    #     """
    #     Return all template names from the templates/ folder.
    #     Auto-detected at run time — no user input required.
    #     """
    #     return ct.list_templates()

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
        self._mode    = None   # "frame" | "font" | "frame_layout" | None
        self._pending = None   # pending font template name

        # Frame template creation state
        self._frame_rects: list             = [None, None, None]
        self._frame_panel: FrameTemplatePanel | None = None
        self._FRAME_TAGS  = ["MOLD_A", "MOLD_B", "ANCHOR", "PIN_A", "PIN_B"]
        self._yolo_pair_offset: int         = 0    # retry cycles through pairs

        # Frame layout stamping state (step 4)
        self._layout_stamps: list                = []   # list of QRect (image coords)
        self._layout_roi_w:  int                 = 0
        self._layout_roi_h:  int                 = 0
        self._layout_panel:  FrameLayoutPanel | None = None

        # Run worker state
        self._worker:  RunWorker   | None = None
        self._camera:  BaslerCamera | None = None
        self._machine_io       = MachineIO()

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
        btn("Create Frame Tmpl", self._start_frame,   "#1a4070")
        btn("Create Font Tmpl",  self._start_font,    "#1a4070")
        sep()
        self._btn_run  = btn("▶ Start Run", self._start_run, "#1a5530")
        self._btn_stop = btn("■ Stop",      self._stop_run,  "#7a2010")
        self._btn_stop.setEnabled(False)
        btn("Clear", self._clear)
        sep()
        btn("⚙ Load Settings",  self._load_settings, "#384858")
        btn("Save Settings",    self._save_settings,  "#384858")
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

        self._view.add_overlay(rect_a,      QtGui.QColor(200, 50,  255), "MOLD_A", "dash")
        self._view.add_overlay(rect_b,      QtGui.QColor(200, 50,  255), "MOLD_B", "dash")
        self._view.add_overlay(anchor_rect, QtGui.QColor(240, 180,  240), "ANCHOR", "dash")
        self._view.add_overlay(pin_a_rect,  QtGui.QColor(150, 255,  148), "PIN_A",  "dash")
        self._view.add_overlay(pin_b_rect,  QtGui.QColor(150, 255,  148), "PIN_B",  "dash")

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
            self._panel.log("Frame recipe saved. Now stamp expected frame positions.", "#88ff88")
            self._start_layout_stamp()
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

    # ----------------------------------------------------------
    # Frame layout stamping (step 4 — inline in main window)
    # ----------------------------------------------------------
    def _start_layout_stamp(self):
        """Enter frame-layout stamping mode on the main ImageView."""
        try:
            recipe = self._ctrl.load_frame_recipe()
        except Exception as e:
            self._panel.log(f"Layout stamp: recipe load error: {e}", "#ff4444")
            return

        anc_canvas = recipe["anchor"].get("canvas")
        if anc_canvas is None:
            self._panel.log("Layout stamp: anchor canvas not found.", "#ff4444")
            return

        canvas_h, canvas_w = anc_canvas.shape[:2]
        self._layout_roi_w = int(canvas_w * 1.5)
        self._layout_roi_h = int(canvas_h * 1.5)
        self._layout_stamps = []
        self._mode = "frame_layout"

        self._view.set_stamp_mode(
            True,
            box_size = self._layout_roi_w,
            box_h    = self._layout_roi_h,
            label    = "F1",
        )

        self._layout_panel = FrameLayoutPanel(
            roi_w      = self._layout_roi_w,
            roi_h      = self._layout_roi_h,
            on_undo    = self._on_layout_undo,
            on_confirm = self._on_layout_confirm,
            on_cancel  = self._on_layout_cancel,
            parent     = self,
        )
        geo = self.geometry()
        pw  = self._layout_panel.sizeHint().width()
        self._layout_panel.move(geo.right() - pw - 20, geo.top() + 60)
        self._layout_panel.show()
        self._panel.log(
            f"Step 4 — Click image to stamp frame positions "
            f"(ROI {self._layout_roi_w}×{self._layout_roi_h} px = 1.5× anchor).",
            "#aaddff")

    def _on_layout_stamp(self, rect: QtCore.QRect):
        n = len(self._layout_stamps) + 1
        self._layout_stamps.append(rect)
        self._view.add_overlay(
            rect, QtGui.QColor(210, 55, 200), f"F{n}", "dash")
        if self._layout_panel:
            self._layout_panel.update_count(n)
        self._view.set_stamp_label(f"F{n + 1}")
        self._panel.log(
            f"Stamp F{n} placed at ({rect.center().x()}, {rect.center().y()}).",
            "#88ddff")

    def _on_layout_undo(self):
        if not self._layout_stamps:
            return
        self._layout_stamps.pop()
        # Remove the last LAYOUT overlay (tag starts with "F")
        for i in range(len(self._view._overlays) - 1, -1, -1):
            lbl = self._view._overlays[i][2]
            if lbl.startswith("F") and lbl[1:].isdigit():
                del self._view._overlays[i]
                break
        self._view.update()
        n = len(self._layout_stamps)
        if self._layout_panel:
            self._layout_panel.update_count(n)
        self._view.set_stamp_label(f"F{n + 1}")
        self._panel.log(f"Undo — {n} stamp(s) remaining.", "#ffaa44")

    def _on_layout_confirm(self):
        frames = []
        for i, rect in enumerate(self._layout_stamps):
            frames.append({
                "id":  f"F{i + 1}",
                "roi": [rect.x(), rect.y(), rect.width(), rect.height()],
            })
        layout = {"version": 1, "frames": frames}
        ok = self._ctrl.save_frame_layout(layout)
        self._stop_layout_stamp()
        if ok:
            self._panel.log(
                f"Frame layout saved — {len(frames)} frame(s) defined.", "#88ff88")
        else:
            self._panel.log("Frame layout save failed.", "#ff4444")

    def _on_layout_cancel(self):
        self._stop_layout_stamp()
        self._panel.log("Frame layout stamping cancelled.", "#888888")

    def _stop_layout_stamp(self):
        self._mode = None
        self._view.set_stamp_mode(False)
        # Remove layout stamp overlays (tagged "F1", "F2", …)
        self._view._overlays = [
            ov for ov in self._view._overlays
            if not (ov[2].startswith("F") and ov[2][1:].isdigit())
        ]
        self._view.update()
        self._layout_stamps = []
        if self._layout_panel:
            self._layout_panel.hide()
            self._layout_panel = None

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

        if self._mode == "frame":
            self._on_frame_roi(rect)
            return

        if self._mode == "frame_layout":
            self._on_layout_stamp(rect)
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
                rect, QtGui.QColor(210, 55, 200), name, "solid")
            self._panel.log(
                f"Font '{name}' saved ({w}x{h} px).", "#88ff88")
        else:
            self._panel.log(f"Font '{name}' save failed.", "#ff4444")
        self._pending = None
        self._mode    = None

    def _on_frame_roi(self, *_):
        # Frame mode is now YOLO-only; rubber-band draws in frame mode are ignored.
        pass

    # ----------------------------------------------------------
    # Inspection run — threaded
    # ----------------------------------------------------------
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
                QtWidgets.QMessageBox.warning(self, "Missing Template", msg)
                self._panel.log(f"Run blocked — missing: {missing}", "#ff4444")
                return
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

        self._worker = RunWorker(
            ctrl        = self._ctrl,
            io          = self._machine_io,
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
            self, "Load Settings", SETUP_FILE,
            "JSON files (*.json);;All files (*)")
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
            self, "Save Settings", SETUP_FILE,
            "JSON files (*.json);;All files (*)")
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
        (QtGui.QPalette.Window,          (28,  33,  42)),
        (QtGui.QPalette.WindowText,      (200, 210, 222)),
        (QtGui.QPalette.Base,            (18,  22,  30)),
        (QtGui.QPalette.AlternateBase,   (35,  42,  54)),
        (QtGui.QPalette.Text,            (200, 210, 222)),
        (QtGui.QPalette.Button,          (44,  54,  70)),
        (QtGui.QPalette.ButtonText,      (200, 210, 222)),
        (QtGui.QPalette.Highlight,       (50,  105, 165)),
        (QtGui.QPalette.HighlightedText, (220, 230, 242)),
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