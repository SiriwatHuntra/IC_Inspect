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
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
from dataclasses import dataclass, field
import time
from scipy.signal import find_peaks

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
    display:    np.ndarray
    results:    list  = field(default_factory=list)
    passed:     int   = 0
    total:      int   = 0
    elapsed_ms: float = 0.0


# =========================================================
# CONFIG
# =========================================================
DEBUG_MODE     = True
PIN_SORT_ORDER = "tl_br"   # top-left -> bottom-right, row-first
PIN_ROI_W      = 120       # fixed capture size in image-space px
PIN_ROI_H      = 150

MIN_CONTOUR_AREA = 1
IMAGE_SOURCE_DIR  = "image_source"
OUTPUT_DIR        = "Inspection_result"

# Camera
CAMERA_SERIAL        = "22202392"
CAMERA_WARMUP_FRAMES = 5
CAMERA_EXPOSURE_US   = 8000     # µs — overridden by RightPanel at runtime

# ---- Font Inspection Constants (hardcoded, not user-tunable) ----
FONT_SHIFT_RATIO_MAX   = 0.20
FONT_ASPECT_TOLERANCE  = 0.25
FONT_HOLE_COUNT_TOLERANCE  = 1
FONT_HOLE_AREA_TOLERANCE   = 0.30
FONT_CONFIDENCE_MIN        = 0.60


# =========================================================
# SETTINGS MANAGER
# =========================================================
SETTINGS_FILE = "inspection_settings.txt"
MASK_FILE     = "search_mask.jpg"

# (header, default, min, max, is_float)
_SETTINGS_DEFAULTS = [
    ("pin_score_threshold",  0.75,  0.50,  1.00,  True ),
    ("iou_threshold",        0.50,  0.00,  1.00,  True ),
    ("max_matches",             6,     1,     6,  False),
    ("search_scale_min",     0.75,  0.30,  1.00,  True ),
    ("search_scale_max",     1.25,  1.00,  2.00,  True ),
    ("search_scale_steps",      7,     3,    21,  False),
    ("tm_threshold",         0.80,  0.50,  1.00,  True ),
    ("camera_exposure_us",  8000,   100, 100000,  False),
]

_STRING_DEFAULTS = {
    "grid_letters": ",,,,,,,,,",
}

class SettingsManager:
    """
    Persists numeric and string settings to a plain-text file.
    Numeric entries  ->  header  value  min  max
    String  entries  ->  header  "value"
    """

    def __init__(self, path: str = SETTINGS_FILE):
        self.path = path
        self._data: dict = {}
        for hdr, val, mn, mx, is_float in _SETTINGS_DEFAULTS:
            self._data[hdr] = {
                "value":    float(val) if is_float else int(val),
                "min":      float(mn)  if is_float else int(mn),
                "max":      float(mx)  if is_float else int(mx),
                "is_float": is_float,
            }
        self._str_data: dict = dict(_STRING_DEFAULTS)
        if os.path.exists(path):
            self._load()

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

    def get_str(self, header: str) -> str:
        return self._str_data.get(header, "")

    def set_str(self, header: str, value: str):
        if header in self._str_data:
            self._str_data[header] = value.strip()

    def save(self, path: str = None) -> str:
        target = path or self.path
        with open(target, "w") as f:
            for hdr, _, _, _, is_float in _SETTINGS_DEFAULTS:
                d = self._data[hdr]
                f.write(f"{hdr:<28} {d['value']}  {d['min']}  {d['max']}\n")
            for hdr, _ in _STRING_DEFAULTS.items():
                val = self._str_data[hdr].replace("\n", "\\n")
                f.write(f"{hdr:<28} \"{val}\"\n")
        return target
    
    def load(self, path: str = None):
        self._load(path or self.path)

    def _load(self, path: str = None):
        target = path or self.path
        try:
            with open(target, "r") as f:
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
        except Exception as e:
            print(f"[Settings] Load error: {e}")


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
        for i in range(CAMERA_WARMUP_FRAMES):
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
# MOCK IO WORKER  — simulates external recipe signal
# =========================================================
class MockIOWorker(QtCore.QThread):
    """
    Simulates an external IO signal that delivers a grid_letters recipe.
    In production this would read from a serial port, socket, or PLC.

    Signals
    -------
    sig_recipe(list)  : emits list of 9 strings when a recipe is received
    """
    sig_recipe = QtCore.pyqtSignal(list)

    # Hardcoded mock payload — replace with real IO read in production
    MOCK_RECIPE = ["7", "8", "W", "6", "", "8", "H", "9", ""]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        """
        Mock: wait 2 s then emit one recipe signal and exit.
        Production: loop reading IO, emit on each new recipe received.
        """
        for _ in range(40):          # 40 × 50 ms = 2 s
            if self._stop_flag:
                return
            self.msleep(50)
        if not self._stop_flag:
            self.sig_recipe.emit(list(self.MOCK_RECIPE))
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
    def _thresh_font(gray: np.ndarray, mold_size: int = 150) -> np.ndarray:
        # ── Step 1 : Noise filter ─────────────────────────────────
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # ── Step 2 : White top-hat — isolates bright strokes ─────
        # Kernel anchored to mold_size, not ROI size
        k_size = max(9, (mold_size // 8) | 1)          # odd, ~18px at mold=150
        kernel  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))
        tophat  = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)

        # ── Step 3 : Otsu on top-hat result ──────────────────────
        _, binary = cv2.threshold(
            tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Step 4 : Border mask — kill edge spikes ───────────────
        h, w   = binary.shape[:2]
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
        h, w   = binary.shape[:2]
        roi_sz = min(h, w)                          # anchor to actual input size
        k_size = max(2, roi_sz // 20)               # ~2-3px at 45-50px ROI
        out    = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size)))
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

        # Find largest top-level contour
        best_idx  = -1
        best_area = 0.0
        for i, c in enumerate(raw_cnts):
            if hierarchy[i][3] != -1:
                continue
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
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
          _thresh_font -> _morph_font -> _find_contours

        Input : grayscale ndarray, mold_size (px) for kernel scaling
        Output: (contours, canvas, thresh_binary)
                contours is [] when nothing found.
        """
        h, w   = gray.shape[:2]
        thresh = ContourTemplate._thresh_font(gray, mold_size)
        clean  = ContourTemplate._morph_font(thresh, mold_size)

        if debug_prefix:
            _write_debug(debug_prefix, gray, thresh, clean)

        contours, canvas = ContourTemplate._find_contours(clean, h, w)

        if debug_prefix and contours:
            _write_debug_contours(debug_prefix, gray, contours, canvas)

        return contours, canvas, thresh

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

        contours, canvas, _ = ContourTemplate.extract_font_template(
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

        data["pca_angle"] = float(data.get("pca_angle", 0.0))
        data["contours"]  = [np.array(c, dtype=np.int32)
                             for c in data["contours"]]
        raw            = base64.b64decode(data["canvas_b64"])
        arr            = np.frombuffer(raw, dtype=np.uint8)
        data["canvas"] = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

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
# B.  InspectionEngine  — named check steps
# =========================================================
class InspectionEngine:
    """
    Two inspection primitives.

    find_all_pin_templates(image_gray, tmpl, ...)
        Multi-scale TM_CCOEFF_NORMED on the template canvas.
        Returns match list sorted best-score-first.

    compare_roi(roi_bgr, tmpl, ...)
        Five hard-fail check steps in sequence.
        Returns a result dict; key "roi_canvas" carries the extracted
        canvas so the caller can draw contours without re-extracting.

    Check steps
    -----------
      1. _check_presence  (gray, mold_size)       -> (contours, canvas)
      2. _check_shift     (contours, tmpl, ...)   -> (shift_px, shift_ratio)
      3. _check_holes     (gray, outer, holes ...) -> (score, holes, canvas, contours)
      4. _check_similarity(canvas, tmpl_canvas)   -> confidence
      5. _check_aspect    (contours, tmpl_aspect) -> (roi_aspect, aspect_diff)
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
        Output: (contours: list, canvas: uint8 ndarray)
                contours is [] when nothing found → caller treats as fail.
        """
        contours, canvas, _ = ContourTemplate.extract_font_template(gray, mold_size=mold_size)
        
        return contours, canvas
    
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

    # @staticmethod
    # def _check_stroke(contours:          list,
    #                   tmpl_contour_count: int) -> int:
    #     """
    #     Step 3 — Stroke count.
    #     Detect dirty / extra marks by comparing contour count.

    #     Input : contours (from _check_presence),
    #             tmpl_contour_count (from template dict)
    #     Output: stroke_diff (int) — 0 is perfect match.
    #     """
    #     return abs(len(contours) - tmpl_contour_count)

    @staticmethod
    def _check_similarity(canvas:      np.ndarray,
                        tmpl_canvas: np.ndarray) -> float:
        """
        Step 4 — Similarity.
        Centre-aligned IoU at normalized 64x64 resolution.
        Positional offset removed before comparison — shift is checked separately.

        Input : canvas (from _check_presence), tmpl_canvas (from template dict)
        Output: confidence float 0.0–1.0  (higher = more similar)
        """
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return 0.0

        NORM_SIZE = 64

        def _centre_align(src: np.ndarray) -> np.ndarray:
            """Crop white-pixel bbox, place it centred in NORM_SIZE x NORM_SIZE frame."""
            pts = cv2.findNonZero(src)
            if pts is None:
                return np.zeros((NORM_SIZE, NORM_SIZE), dtype=np.uint8)
            x, y, w, h = cv2.boundingRect(pts)
            crop = src[y:y + h, x:x + w]
            # Scale crop to fit within NORM_SIZE keeping aspect ratio
            scale = min(NORM_SIZE / max(w, 1), NORM_SIZE / max(h, 1))
            sw = max(1, int(round(w * scale)))
            sh = max(1, int(round(h * scale)))
            resized = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_NEAREST)
            # Place centred
            frame = np.zeros((NORM_SIZE, NORM_SIZE), dtype=np.uint8)
            ox = (NORM_SIZE - sw) // 2
            oy = (NORM_SIZE - sh) // 2
            frame[oy:oy + sh, ox:ox + sw] = resized
            return frame

        rc = _centre_align(canvas)
        tc = _centre_align(tmpl_canvas)

        intersection = np.count_nonzero(cv2.bitwise_and(rc, tc))
        union        = np.count_nonzero(cv2.bitwise_or(rc, tc))
        return round(float(intersection / max(union, 1)), 4)

    # @staticmethod
    # def _check_coverage(canvas:          np.ndarray,
    #                     expected_pixels: int) -> float:
    #     """
    #     Step 5 — Coverage.
    #     Ratio of actual filled pixels to template reference pixels.
    #     > coverage_max → ghosting / over-exposure.
    #     < coverage_min → distortion / under-mark.

    #     Input : canvas (from _check_presence), expected_pixels (from template dict)
    #     Output: coverage float (1.0 = perfect match)
    #     """
    #     actual_pixels = int(np.count_nonzero(canvas))
    #     return round(actual_pixels / max(expected_pixels, 1), 4)

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
    
    # @staticmethod
    # def _get_normalized_vertices(contour:  np.ndarray,
    #                             bx: int, by: int,
    #                             bw: int, bh: int) -> np.ndarray:
    #     """
    #     approxPolyDP vertices normalized to bounding box (0.0–1.0).
    #     Epsilon = 5% of perimeter.
    #     """
    #     epsilon = 0.05 * cv2.arcLength(contour, True)
    #     approx  = cv2.approxPolyDP(contour, epsilon, True)
    #     pts     = approx.reshape(-1, 2).astype(np.float32)
    #     pts[:, 0] = (pts[:, 0] - bx) / max(bw, 1)
    #     pts[:, 1] = (pts[:, 1] - by) / max(bh, 1)
    #     return pts

    # @staticmethod
    # def _check_hu(roi_outer:  np.ndarray,
    #             tmpl_outer: np.ndarray) -> tuple:
    #     """
    #     Hu moment shape similarity.
    #     Returns (score 0-1, raw distance).
    #     """
    #     dist  = cv2.matchShapes(roi_outer, tmpl_outer,
    #                             cv2.CONTOURS_MATCH_I1, 0.0)
    #     score = max(0.0, 1.0 - dist / max(FONT_HU_THRESHOLD, 1e-6))
    #     return round(score, 4), round(dist, 6)

    # @staticmethod
    # def _check_vertices(roi_outer:   np.ndarray,
    #                     tmpl_verts:  np.ndarray,
    #                     tbx: int, tby: int,
    #                     tbw: int, tbh: int) -> float:
    #     """
    #     Nearest-match crossing point comparison (normalized bbox coords).
    #     Returns score 0.0–1.0.
    #     """
    #     if len(tmpl_verts) == 0:
    #         return 1.0

    #     # Get roi vertices normalized to template bbox space
    #     epsilon   = 0.05 * cv2.arcLength(roi_outer, True)
    #     approx    = cv2.approxPolyDP(roi_outer, epsilon, True)
    #     roi_pts   = approx.reshape(-1, 2).astype(np.float32)
    #     bx, by, bw, bh = cv2.boundingRect(roi_outer)
    #     roi_pts[:, 0] = (roi_pts[:, 0] - bx) / max(bw, 1)
    #     roi_pts[:, 1] = (roi_pts[:, 1] - by) / max(bh, 1)

    #     # Vertex count score
    #     count_diff  = abs(len(roi_pts) - len(tmpl_verts))
    #     count_score = max(0.0, 1.0 - count_diff / max(FONT_VERTEX_TOLERANCE * 2, 1))

    #     # Nearest match position score
    #     total_dist = 0.0
    #     for rv in roi_pts:
    #         dists      = np.linalg.norm(tmpl_verts - rv, axis=1)
    #         total_dist += float(np.min(dists))
    #     avg_dist    = total_dist / max(len(roi_pts), 1)
    #     pos_score   = max(0.0, 1.0 - avg_dist / max(FONT_VERTEX_POS_TOLERANCE, 1e-6))

    #     return round((count_score + pos_score) / 2.0, 4)

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

    def find_all_pin_templates(self,
                           image_bgr:    np.ndarray,
                           tmpl:         dict,
                           score_thr:    float = 0.75,
                           iou_thr:      float = 0.50,
                           max_matches:  int   = 6,
                           mask:         np.ndarray = None,
                           scale_min:    float = 0.75,
                           scale_max:    float = 1.25,
                           scale_steps:  int   = 7) -> list:
        """
        Multi-scale TM_CCOEFF_NORMED search for frame template.
        Returns list of (cx, cy, score, w, h, scale) sorted best-first.
        NMS applied with iou_thr.
        """
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
            if image_bgr.ndim == 3 else image_bgr.copy()

        tmpl_canvas = tmpl.get("canvas")
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return []

        th, tw = tmpl_canvas.shape[:2]
        scales = np.linspace(scale_min, scale_max, scale_steps)

        candidates = []

        for scale in scales:
            sw = max(8, int(round(tw * scale)))
            sh = max(8, int(round(th * scale)))

            scaled_tmpl = cv2.resize(tmpl_canvas, (sw, sh),
                                    interpolation=cv2.INTER_AREA)

            ih, iw = gray.shape[:2]
            if sw > iw or sh > ih:
                continue

            search = cv2.bitwise_and(gray, mask) if mask is not None else gray
            result = cv2.matchTemplate(search, scaled_tmpl,
                                    cv2.TM_CCOEFF_NORMED)
            locs   = np.where(result >= score_thr)

            for y, x in zip(*locs):
                score = float(result[y, x])
                cx    = x + sw // 2
                cy    = y + sh // 2
                candidates.append((cx, cy, score, sw, sh, float(scale)))

        if not candidates:
            return []

        # NMS — suppress overlapping boxes
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
                # IoU
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                iw_ = max(0, ix2 - ix1)
                ih_ = max(0, iy2 - iy1)
                inter = iw_ * ih_
                union = w * h + kw * kh - inter
                if union > 0 and inter / union > iou_thr:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)
            if len(kept) >= max_matches:
                break

        return kept
    # =========================================================
    # COMPARE ROI  — orchestrates the six steps
    # =========================================================

    def compare_roi(self,
                roi_bgr:      np.ndarray,
                tmpl:         dict,
                tm_thr:       float,
                exp_dx:       int,
                exp_dy:       int,
                mold_cx:      int,
                mold_cy:      int,
                mold_size:    int   = 150,
                coverage_min: float = 0.40,
                coverage_max: float = 2.00) -> dict:

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) \
            if roi_bgr.ndim == 3 else roi_bgr.copy()

        orig_roi_h, orig_roi_w = gray.shape[:2]
        roi_h, roi_w = gray.shape[:2]

        tmpl_aspect   = float(tmpl.get("tmpl_aspect", 1.0))
        tmpl_contours = tmpl.get("contours", [])

        def _fail(step: int, reason: str, extras: dict = None) -> dict:
            base = {
                "pass":         False,
                "confidence":   0.0,
                "stroke_diff":  0,
                "rotation_deg": 0.0,
                "shift_px":     0.0,
                "shift_ratio":  0.0,
                "coverage":     0.0,
                "aspect_ratio": 0.0,
                "tmpl_aspect":  tmpl_aspect,
                "reason":       reason,
                "defect_step":  step,
                "roi_canvas":   None,
                "orig_roi_w":   orig_roi_w,
                "orig_roi_h":   orig_roi_h,
            }
            if extras:
                base.update(extras)
            return base

        if not tmpl_contours:
            return _fail(0, "no template contours")

        # Template outer + holes
        tmpl_outer      = tmpl_contours[0]
        tmpl_holes      = tmpl_contours[1:] if len(tmpl_contours) > 1 else []
        tmpl_hole_count = len(tmpl_holes)

        # Template hole area ratios
        tmpl_outer_area  = cv2.contourArea(tmpl_outer)
        tmpl_hole_ratios = [
            cv2.contourArea(h) / max(tmpl_outer_area, 1)
            for h in tmpl_holes
        ]

        # ── Step 1 : Presence ────────────────────────────────
        contours, canvas = self._check_presence(gray, mold_size)

        if not contours:
            return _fail(1, "missing mark")

        roi_outer = contours[0]
        roi_holes = contours[1:] if len(contours) > 1 else []

        # ── Step 2 : Shift ───────────────────────────────────
        shift_px, shift_ratio = self._check_shift(
            contours, tmpl, exp_dx, exp_dy, mold_cx, mold_cy, roi_w, roi_h)

        if shift_ratio > FONT_SHIFT_RATIO_MAX:
            return _fail(2,
                f"shift ratio={shift_ratio:.3f} > {FONT_SHIFT_RATIO_MAX}",
                {"shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "roi_canvas":  canvas})

        # ── Step 3 : Holes (hard-fail with cleanup retry) ────
        hole_score, roi_holes, canvas, contours = self._check_holes(
            gray, roi_outer, roi_holes, canvas, contours,
            tmpl_hole_count, tmpl_hole_ratios, mold_size)

        if hole_score < 0:
            return _fail(3,
                f"hole mismatch got={len(roi_holes)} "
                f"expected={tmpl_hole_count}",
                {"shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "roi_canvas":  canvas})

        # ── Step 4 : Similarity (canvas IoU 64×64) ───────────
        similarity = self._check_similarity(canvas, tmpl.get("canvas"))

        # ── Step 5 : Aspect ratio ────────────────────────────
        roi_aspect, aspect_diff = self._check_aspect(contours, tmpl_aspect)
        aspect_score = max(0.0, 1.0 - aspect_diff / max(FONT_ASPECT_TOLERANCE, 1e-6))

        if aspect_diff > FONT_ASPECT_TOLERANCE:
            return _fail(5,
                f"distortion aspect={roi_aspect:.3f} "
                f"tmpl={tmpl_aspect:.3f} diff={aspect_diff:.2%}",
                {"shift_px":    shift_px,
                 "shift_ratio": shift_ratio,
                 "aspect_ratio": roi_aspect,
                 "roi_canvas":  canvas})

        # ── Weighted confidence ───────────────────────────────
        confidence = (
            similarity   * 0.70 +
            hole_score   * 0.20 +
            aspect_score * 0.10
        )

        if confidence < FONT_CONFIDENCE_MIN:
            return _fail(4,
                f"low confidence={confidence:.3f} < {FONT_CONFIDENCE_MIN}",
                {"confidence":   confidence,
                 "shift_px":     shift_px,
                 "shift_ratio":  shift_ratio,
                 "aspect_ratio": roi_aspect,
                 "roi_canvas":   canvas})

        return {
            "pass":         True,
            "confidence":   round(confidence, 4),
            "stroke_diff":  0,
            "rotation_deg": 0.0,
            "shift_px":     shift_px,
            "shift_ratio":  shift_ratio,
            "coverage":     0.0,
            "aspect_ratio": roi_aspect,
            "tmpl_aspect":  tmpl_aspect,
            "reason":       "OK",
            "defect_step":  0,
            "roi_canvas":   canvas,
            "orig_roi_w":   orig_roi_w,
            "orig_roi_h":   orig_roi_h,
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

    # ---- Frame -------------------------------------------------------

    @staticmethod
    def draw_frame(display: np.ndarray,
                   fcx:     int,
                   fcy:     int,
                   fw:      int,
                   fh:      int,
                   f_idx:   int,
                   fscore:  float):
        """
        Draw dashed frame bounding box + score label.

        Input : display (BGR ndarray, modified in-place),
                fcx/fcy (frame centre), fw/fh (matched size),
                f_idx (0-based frame index), fscore (TM score 0–1)
        """
        ih, iw = display.shape[:2]
        fx1 = max(0,    fcx - fw // 2)
        fy1 = max(0,    fcy - fh // 2)
        fx2 = min(iw-1, fx1 + fw)
        fy2 = min(ih-1, fy1 + fh)
        cv2_draw_dashed_rect(
            display, (fx1, fy1), (fx2, fy2),
            ResultAnnotator.COLOR_FRAME, 1)
        cv2.putText(display,
                    f"F{f_idx+1} {fscore:.2f}",
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
        cv2.putText(display,
                    f"F{f_idx+1}-{mold_label} [{elapsed_ms:.1f}ms]",
                    (ax1 + 2, ay1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    ResultAnnotator.COLOR_MOLD, 1)

    # ---- Letter ------------------------------------------------------
    @staticmethod
    def draw_letter(display: np.ndarray,
                    result:  dict):
        passed     = result["pass"]
        letter     = result["letter"]
        confidence = result["confidence"]
        lx1        = result["lx1"]
        ly1        = result["ly1"]
        lx2        = result["lx2"]
        ly2        = result["ly2"]
        roi_canvas = result.get("roi_canvas")

        col = ResultAnnotator.COLOR_PASS if passed else ResultAnnotator.COLOR_FAIL
        lbl = f"{letter}+ {confidence:.2f}" if passed \
            else f"{letter}- {confidence:.2f}"

        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), col, 1)

        if roi_canvas is not None and roi_canvas.size > 0:
            canvas_h, canvas_w = roi_canvas.shape[:2]
            cell_w = lx2 - lx1
            cell_h = ly2 - ly1
            scale_x = cell_w / max(canvas_w, 1)
            scale_y = cell_h / max(canvas_h, 1)

            # Outer contours
            outer, _ = cv2.findContours(
                roi_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Inner contours (holes) — invert, exclude any touching canvas border
            inner_all, _ = cv2.findContours(
                cv2.bitwise_not(roi_canvas), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            inner = [
                c for c in inner_all
                if cv2.contourArea(c) >= MIN_CONTOUR_AREA
                and not _touches_border(c, canvas_w, canvas_h)
            ]

            for cnts in (outer, inner):
                if not cnts:
                    continue
                shifted = []
                for c in cnts:
                    scaled = c.astype(np.float32).copy()
                    scaled[..., 0] = scaled[..., 0] * scale_x + lx1
                    scaled[..., 1] = scaled[..., 1] * scale_y + ly1
                    shifted.append(scaled.astype(np.int32))
                cv2.drawContours(display, shifted, -1, col, 1)
                
    
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
        self.cache:   dict[str, dict] = {}
        self.results: list[dict]      = []

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

    @staticmethod
    def _find_mold_centre(image_bgr:       np.ndarray,
                          constraint_rect: QtCore.QRect,
                          canvas_w:        int,
                          canvas_h:        int) -> tuple | None:
        """
        Locate mold centre within constraint_rect using rim edge detection.
        Camera orientation: pins always on top and bottom of mold body.

        Steps
        -----
        X — vertical edge (Sobel-X) column profile:
              leftmost strong peak = left rim
              cx = left_rim_abs_x + canvas_w // 2

        Y — horizontal edge (Sobel-Y) row profile within mold-width band:
              first strong peak  = top rim
              closest peak to (top + canvas_h) within ±35px = bottom rim
              cy = (top_abs + bot_abs) // 2
              fallback: cy = top_abs + canvas_h // 2

        Returns (cx, cy) in image space, or None if no left rim found.
        """
        ih, iw = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
               if image_bgr.ndim == 3 else image_bgr.copy()

        rx  = max(0, constraint_rect.x())
        ry  = max(0, constraint_rect.y())
        rw  = min(constraint_rect.width(),  iw - rx)
        rh  = min(constraint_rect.height(), ih - ry)
        if rw < 16 or rh < 16:
            return None

        roi  = gray[ry:ry + rh, rx:rx + rw]
        blur = cv2.GaussianBlur(roi, (5, 5), 0)

        # ── Step 1 : X from left vertical rim ────────────────────────
        sobelx   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        snx      = cv2.normalize(np.abs(sobelx), None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        _, ebx   = cv2.threshold(snx, 50, 255, cv2.THRESH_BINARY)
        col_sum  = np.sum(ebx, axis=0).astype(float)
        col_s    = np.convolve(col_sum, np.ones(5) / 5, mode='same')

        peaks_x, _ = find_peaks(col_s,
                                 height=col_s.max() * 0.30,
                                 distance=max(10, canvas_w // 5))

        if len(peaks_x) == 0:
            print(f"[MoldFind] No V-edge peaks found — cannot detect mold.")
            return None

        left_roi_x = int(peaks_x[0])

        # Try to find right rim = leftmost peak that is ~canvas_w away from left
        right_candidates = [int(p) for p in peaks_x
                            if abs(int(p) - left_roi_x - canvas_w) < canvas_w * 0.30
                            and int(p) > left_roi_x + canvas_w * 0.5]
        if right_candidates:
            right_roi_x = min(right_candidates)
            actual_w    = right_roi_x - left_roi_x
        else:
            right_roi_x = left_roi_x + canvas_w
            actual_w    = canvas_w

        found_cx = rx + (left_roi_x + right_roi_x) // 2
        print(f"[MoldFind] Left rim={rx+left_roi_x} Right rim={rx+right_roi_x} "
              f"actual_w={actual_w} → cx={found_cx}")

        # ── Step 2 : Y from top/bottom horizontal rims ───────────────
        band_x1 = max(0,  rx + left_roi_x)
        band_x2 = min(iw, band_x1 + canvas_w)
        band    = gray[ry:ry + rh, band_x1:band_x2]

        sobely  = cv2.Sobel(cv2.GaussianBlur(band, (5, 5), 0),
                            cv2.CV_64F, 0, 1, ksize=3)
        sny     = cv2.normalize(np.abs(sobely), None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
        _, eby  = cv2.threshold(sny, 50, 255, cv2.THRESH_BINARY)
        row_sum = np.sum(eby, axis=1).astype(float)
        row_s   = np.convolve(row_sum, np.ones(5) / 5, mode='same')

        peaks_y, _ = find_peaks(row_s,
                                 height=row_s.max() * 0.20,
                                 distance=max(10, canvas_h // 6))

        if len(peaks_y) == 0:
            found_cy = ry + rh // 2
            print(f"[MoldFind] No H-edge peaks — using zone centre cy={found_cy}")
            return (found_cx, found_cy, actual_w, canvas_h)

        top_roi      = int(peaks_y[0])
        expected_bot = top_roi + canvas_h
        window       = 35

        # Closest peak to expected bottom (not strongest)
        candidates = [
            (int(p), abs(int(p) - expected_bot))
            for p in peaks_y
            if abs(int(p) - expected_bot) <= window
            and int(p) > top_roi + canvas_h // 2
        ]

        if candidates:
            bot_roi   = min(candidates, key=lambda x: x[1])[0]
            found_cy  = ry + (top_roi + bot_roi) // 2
            actual_h  = bot_roi - top_roi
            print(f"[MoldFind] Top roi_y={top_roi}(abs={ry+top_roi}) "
                  f"Bot roi_y={bot_roi}(abs={ry+bot_roi}) "
                  f"actual_h={actual_h} → cy={found_cy}")
        else:
            found_cy = ry + top_roi + canvas_h // 2
            actual_h = canvas_h
            print(f"[MoldFind] Top roi_y={top_roi}(abs={ry+top_roi}) "
                  f"no bottom match → cy={found_cy}")

        print(f"[MoldFind] Result: ({found_cx},{found_cy}) body={actual_w}x{actual_h}")
        return (found_cx, found_cy, actual_w, actual_h)
    
    # ---- frame recipe save ----
    def save_frame_recipe(self,
                          image_bgr:         np.ndarray,
                          frame_rect:        QtCore.QRect,
                          grid_letters:      list,
                          mold_a_constraint: QtCore.QRect,
                          mold_b_constraint: QtCore.QRect) -> tuple:
        """
        Auto-locate mold A/B using blob search within constraint zones,
        then extract contours and write pin_recipe.json.

        Returns (ok: bool, a_found: bool, b_found: bool)
          ok      — recipe written successfully
          a_found — True = blob found, False = fallback used
          b_found — same for mold B
        Caller should prompt retry if a_found or b_found is False.
        """
        try:
            fcx = frame_rect.x() + frame_rect.width()  // 2
            fcy = frame_rect.y() + frame_rect.height() // 2

            def _mold_offset(constraint: QtCore.QRect,
                             cw: int, ch: int) -> tuple:
                result = InspectionController._find_mold_centre(
                    image_bgr, constraint, cw, ch)
                if result is not None:
                    mcx, mcy, body_w, body_h = result
                    found = True
                    print(f"[Recipe] Mold found: ({mcx},{mcy}) body={body_w}x{body_h}")
                else:
                    mcx    = constraint.x() + constraint.width()  // 2
                    mcy    = constraint.y() + constraint.height() // 2
                    body_w = cw
                    body_h = ch
                    found  = False
                    print(f"[Recipe] Mold fallback centre: ({mcx},{mcy})")

                # Build encode rect using ACTUAL body dimensions
                rect = QtCore.QRect(mcx - body_w // 2,
                                    mcy - body_h // 2,
                                    body_w, body_h)
                return (mcx - fcx, mcy - fcy), found, rect
            
            frame_sec = self._encode_section(image_bgr, frame_rect)

            # Use constraint rect size as canvas_w/h estimate before encode
            
            fh     = frame_rect.height()
            hint_w = int(fh * 0.8)
            hint_h = int(fh * 0.8)
            a_offset, a_found, a_rect = _mold_offset(mold_a_constraint, hint_w, hint_h)
            b_offset, b_found, b_rect = _mold_offset(mold_b_constraint, hint_w, hint_h)

            mold_a_sec = self._encode_section(image_bgr, a_rect, offset=a_offset)
            mold_b_sec = self._encode_section(image_bgr, b_rect, offset=b_offset)

            # Store fixed grid dimensions from actual detected body size
            # These drive cell layout regardless of frame draw size
            _, _, a_body_w, a_body_h = \
                InspectionController._find_mold_centre(
                    image_bgr, mold_a_constraint,
                    int(frame_rect.height() * 0.85),
                    int(frame_rect.height() * 0.85)) or \
                (0, 0, a_rect.width(), a_rect.height())

            recipe = {
                "version":       4,
                "frame":         frame_sec,
                "mold_a":        mold_a_sec,
                "mold_b":        mold_b_sec,
                "grid_letters":  grid_letters,
                "mold_grid_w":   int(a_rect.width()  * 0.90),
                "mold_grid_h":   int(a_rect.height() * 0.85),
            }
            
            with open(self.RECIPE_FILE, "w") as f:
                json.dump(recipe, f)

            self.cache.pop("FRAME", None)
            return True, a_found, b_found

        except Exception as e:
            print(f"[Controller] save_frame_recipe error: {e}")
            return False, False, False
        
    # ---- frame recipe queries ----
    def has_frame_recipe(self) -> bool:
        return os.path.exists(self.RECIPE_FILE)

    def load_frame_recipe(self) -> dict:
        with open(self.RECIPE_FILE, "r") as f:
            raw = json.load(f)
        result = {
            "version":      raw.get("version", 1),
            "frame":        self._decode_section(raw["frame"]),
            "mold_a":       self._decode_section(raw["mold_a"]),
            "mold_b":       self._decode_section(raw["mold_b"]),
            "grid_letters": raw.get("grid_letters", [""] * 9),
        }
        return result

    def get_frame_template(self) -> dict:
        """Return frame section only (canvas ready for TM search)."""
        recipe = self.load_frame_recipe()
        return recipe["frame"]

    def get_mold_offsets(self) -> tuple:
        """
        Return (mold_a, mold_b) dicts each containing:
          offset   : [dx, dy] from frame match centre
          contours : list of np.int32 arrays
          canvas   : uint8 grayscale ndarray
        """
        recipe = self.load_frame_recipe()
        return recipe["mold_a"], recipe["mold_b"]

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
        Pre-flight: load recipe + cache all templates for grid_letters.
        Call once before the inspection loop starts.

        Returns list of missing template names (empty = all good).
        Stores loaded recipe in self._active_recipe for use by run().
        """
        self._active_recipe  = None
        self._active_grid    = []

        if not self.has_frame_recipe():
            return ["__NO_RECIPE__"]

        try:
            recipe = self.load_frame_recipe()
        except Exception as e:
            print(f"[Prepare] Recipe load error: {e}")
            return ["__RECIPE_ERROR__"]

        active_grid = [l for l in grid_letters if l]
        if not active_grid:
            return ["__NO_GRID__"]

        unique = list(set(active_grid))
        failed = self.load_cache(unique)
        if failed:
            return failed

        # Patch active grid into recipe — used by run() without re-loading
        recipe["grid_letters"] = grid_letters
        self._active_recipe    = recipe
        self._active_grid      = grid_letters
        return []

    def set_run_params(self, pin_params: dict, tm_thr: float):
        """Store search params once before the loop. Used by run() per frame."""
        self._active_pin_params = pin_params
        self._active_tm_thr     = tm_thr
        
    # ---- inspection pipeline ----
    def run(self,
            image_bgr: np.ndarray,
            mask:      np.ndarray = None) -> "InspectionResult":
        """
        Run inspection using pre-loaded recipe and cache from prepare().
        No disk reads. prepare() must be called before the loop starts.
        """
        self.results = []
        src     = image_bgr if image_bgr.ndim == 2 \
                  else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        empty   = InspectionResult(display=display)

        recipe = getattr(self, "_active_recipe", None)
        if recipe is None:
            print("[Pipeline] prepare() not called or failed.")
            return empty

        pin_params = getattr(self, "_active_pin_params", None)
        tm_thr     = getattr(self, "_active_tm_thr",    0.80)

        matches = self._step1_find_frames(
            image_bgr, recipe["frame"], pin_params, mask)

        if not matches:
            print("[Pipeline] No frame matches found.")
            return empty

        ih, iw = image_bgr.shape[:2]

        for f_idx, (fcx, fcy, fscore, fw, fh, fscale) in enumerate(matches):
            ResultAnnotator.draw_frame(display, fcx, fcy, fw, fh, f_idx, fscore)

            mold_areas = self._step2_locate_molds(
                recipe, fcx, fcy, fscale, iw, ih, f_idx)

            for area in mold_areas:
                acx, acy  = area["cx"],  area["cy"]
                aw,  ah   = area["w"],   area["h"]
                mold_size = min(aw, ah)

                t0_mold = time.perf_counter()
                letter_results = self._step3_inspect_fonts(
                    src, display, acx, acy,
                    area["grid"],
                    tm_thr,
                    f_idx, area["label"], iw, ih,
                    mold_size = mold_size,
                    mold_w    = aw,
                    mold_h    = ah,
                    grid_w    = area["grid_w"],
                    grid_h    = area["grid_h"])
                mold_ms = (time.perf_counter() - t0_mold) * 1000

                ResultAnnotator.draw_mold(display, acx, acy, aw, ah,
                                          f_idx, area["label"], elapsed_ms=mold_ms)
                self.results.extend(letter_results)

        passed = sum(1 for r in self.results if r["pass"])
        return InspectionResult(
            display = display,
            results = self.results,
            passed  = passed,
            total   = len(self.results),
        )

    def _step1_find_frames(self, image_bgr, frame_tmpl, pin_params, mask):
        matches = self._eng.find_all_pin_templates(
            image_bgr, frame_tmpl,
            score_thr   = pin_params["score_thr"],
            iou_thr     = pin_params["iou_thr"],
            max_matches = pin_params["max_matches"],
            mask        = mask,
            scale_min   = pin_params["scale_min"],
            scale_max   = pin_params["scale_max"],
            scale_steps = pin_params["scale_steps"],
        )
        if not matches:
            return []
        ROW_SNAP = frame_tmpl.get("canvas_h", 60) // 2
        if PIN_SORT_ORDER == "tl_br":
            matches = sorted(matches,
                            key=lambda m: (round(m[1] / max(ROW_SNAP, 1)), m[0]))
        return matches

    def _step2_locate_molds(self, recipe, fcx, fcy, fscale, iw, ih, f_idx):
        areas = []
        for key, label in [("mold_a", "A"), ("mold_b", "B")]:
            mold = recipe[key]
            odx, ody    = mold["offset"]
            canvas_w    = mold["canvas_w"]
            canvas_h    = mold["canvas_h"]
            acx = fcx + int(round(odx * fscale))
            acy = fcy + int(round(ody * fscale))
            aw  = int(round(canvas_w * fscale))
            ah  = int(round(canvas_h * fscale))
            acx = max(aw // 2, min(iw - aw // 2, acx))
            acy = max(ah // 2, min(ih - ah // 2, acy))
            areas.append({
                "label":      label,
                "cx":         acx,
                "cy":         acy,
                "w":          aw,
                "h":          ah,
                "fscale":     fscale,
                "canvas_w":   canvas_w,   # raw (unscaled) stored dims
                "canvas_h":   canvas_h,
                "contours":   mold["contours"],
                "canvas":     mold["canvas"],
                "grid":       recipe.get("grid_letters", []),
                "grid_w":     recipe.get("mold_grid_w", int(aw * 0.90)),
                "grid_h":     recipe.get("mold_grid_h", int(ah * 0.85)),
            })
        return areas

    def _step3_inspect_fonts(self, image_bgr, display,
                         acx, acy, grid_letters,
                         tm_thr,
                         f_idx, mold_label,
                         iw, ih,
                         mold_size: int = 150,
                         mold_w:    int = 150,
                         mold_h:    int = 150,
                         grid_w:    int = 0,
                         grid_h:    int = 0) -> list:
        results = []

        # Use stored grid dims if available — decouples cell size from frame draw size
        if grid_w <= 0:
            grid_w = int(mold_w * 0.90)
        if grid_h <= 0:
            grid_h = int(mold_h * 0.85)
        cell_w   = int(grid_w / 3)
        cell_h   = int(grid_h / 3)

        roi_w    = int(cell_w * 1.2)
        roi_h    = int(cell_h * 1.2)

        for slot_idx, letter in enumerate(grid_letters):
            if not letter:
                continue

            tmpl = self.cache.get(letter)
            if tmpl is None:
                continue

            row = slot_idx // 3
            col = slot_idx  % 3

            dx = (col - 1) * cell_w
            dy = (row - 1) * cell_h

            cell_cx = acx + dx
            cell_cy = acy + dy

            half_w = roi_w // 2
            half_h = roi_h // 2

            lx1 = max(0,  cell_cx - half_w)
            ly1 = max(0,  cell_cy - half_h)
            lx2 = min(iw, lx1 + roi_w)
            ly2 = min(ih, ly1 + roi_h)

            if lx2 <= lx1 or ly2 <= ly1:
                continue

            exp_dx = dx
            exp_dy = dy

            roi = image_bgr[ly1:ly2, lx1:lx2]

            t0 = time.perf_counter()
            res = self._eng.compare_roi(
                roi, tmpl, tm_thr,
                exp_dx    = exp_dx,
                exp_dy    = exp_dy,
                mold_cx   = acx,
                mold_cy   = acy,
                mold_size = mold_size)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results.append({
                "frame_idx":    f_idx + 1,
                "mold":         mold_label,
                "slot":         slot_idx,
                "letter":       letter,
                "pass":         res["pass"],
                "confidence":   res["confidence"],
                "stroke_diff":  res["stroke_diff"],
                "rotation_deg": res["rotation_deg"],
                "shift_px":     res["shift_px"],
                "shift_ratio":  res["shift_ratio"],
                "coverage":     res["coverage"],
                "aspect_ratio": res["aspect_ratio"],
                "tmpl_aspect":  res["tmpl_aspect"],
                "defect_step":  res["defect_step"],
                "reason":       res["reason"],
                "cell_cx":      cell_cx,
                "cell_cy":      cell_cy,
                "elapsed_ms":   round(elapsed_ms, 2),
                "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                "roi_canvas":   res["roi_canvas"],
            })

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
                 camera:      "BaslerCamera | None" = None):
        super().__init__()
        self._ctrl        = ctrl
        self._io          = io
        self._mask        = mask
        self._image_io    = image_io
        self._run_from_io = run_from_io
        self._io_recipe   = io_recipe or []
        self._ui_grid     = ui_grid   or []
        self._camera      = camera
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

        total_passed = 0
        total_letters = 0

        for i, fpath in enumerate(files):
            if self._stop_flag:
                self._log("Batch stopped by user.", "#ffaa44")
                break

            fname = os.path.basename(fpath)
            self._log(f"[{i+1}/{len(files)}] {fname}", "#cccccc")

            try:
                img = self._image_io.load(fpath)
            except Exception as e:
                self.sig_error.emit(f"Load error '{fname}': {e}")
                break

            result = self._inspect_one(img)
            self.sig_image.emit(result.display)

            all_pass = (result.passed == result.total and result.total > 0)

            self._log(
                f"  ⏱ {result.elapsed_ms:.1f}ms  "
                f"letters={result.total}  passed={result.passed}",
                "#aaaaaa")

            for r in result.results:
                verdict = "PASS" if r["pass"] else "FAIL"
                color   = "#88ff88" if r["pass"] else "#ff4444"
                self._log(
                    f"  F{r['frame_idx']}-{r['mold']} [{r['letter']}]"
                    f" slot{r['slot']} {verdict}"
                    f"  conf={r['confidence']:.3f}"
                    f"  rot={r['rotation_deg']:.1f}°"
                    f"  shift={r['shift_px']:.1f}px({r['shift_ratio']:.3f})"
                    f"  cov={r['coverage']:.3f}"
                    f"  asp={r['aspect_ratio']:.3f}"
                    f"  step={r['defect_step']}"
                    f"  [{r['elapsed_ms']:.1f}ms]"
                    f"  {r['reason']}",
                    color)

            if not all_pass:
                self._save_fail(img, result.display)

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

            # Step 3/4: wait for inspect signal from machine
            triggered = self._io.wait_for_start(lambda: self._stop_flag)
            if not triggered or self._stop_flag:
                break

            # Step 5: assert busy, grab frame
            self._io.set_busy(True)

            img = self._camera.grab()
            if img is None:
                self.sig_error.emit("Camera grab failed — stopping run.")
                self._io.set_busy(False)
                break

            # Step 6: inspect
            result = self._inspect_one(img)
            self.sig_image.emit(result.display)

            all_pass = (result.passed == result.total and result.total > 0)

            self._log(
                f"  ⏱ {result.elapsed_ms:.1f}ms  "
                f"letters={result.total}  passed={result.passed}",
                "#aaaaaa")

            for r in result.results:
                verdict = "PASS" if r["pass"] else "FAIL"
                color   = "#88ff88" if r["pass"] else "#ff4444"
                self._log(
                    f"  F{r['frame_idx']}-{r['mold']} [{r['letter']}]"
                    f" slot{r['slot']} {verdict}"
                    f"  conf={r['confidence']:.3f}"
                    f"  rot={r['rotation_deg']:.1f}°"
                    f"  shift={r['shift_px']:.1f}px({r['shift_ratio']:.3f})"
                    f"  cov={r['coverage']:.3f}"
                    f"  asp={r['aspect_ratio']:.3f}"
                    f"  step={r['defect_step']}"
                    f"  [{r['elapsed_ms']:.1f}ms]"
                    f"  {r['reason']}",
                    color)

            if not all_pass:
                self._save_fail(img, result.display)

            # Step 7: output result, release busy
            self._io.on_frame_result(result.passed, result.total)
            total_passed  += result.passed
            total_letters += result.total

            self._io.set_busy(False)
            # Step 8: loop back to wait for next trigger (step 4)

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
        orig_h, orig_w = self._orig.shape[:2]
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

        contours, canvas, _ = ContourTemplate.extract_font_template(
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
    Floating always-on-top panel — guides through 3 sequential steps.

    Step 0 — FRAME  : rubber-band draw
    Step 1 — MOLD A : rubber-band draw (constrained near frame)
    Step 2 — MOLD B : rubber-band draw (constrained near frame)
    """

    _STEP_LABELS = [
        ("Step 1 / 1  —  FRAME",
         "Draw the lead frame region on the image.\n"
         "Mold A/B will be located automatically."),
    ]

    def __init__(self, on_confirm, on_cancel, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("Create Frame Template")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setFixedWidth(340)

        self._on_confirm = on_confirm
        self._on_cancel  = on_cancel

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        self._lbl_title = QtWidgets.QLabel()
        self._lbl_title.setStyleSheet(
            "font-size:13px;font-weight:bold;color:#00e5ff")
        self._lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_title)

        self._lbl_inst = QtWidgets.QLabel()
        self._lbl_inst.setStyleSheet("color:#cccccc;font-size:11px")
        self._lbl_inst.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_inst.setWordWrap(True)
        lay.addWidget(self._lbl_inst)

        self._lbl_status = QtWidgets.QLabel("Draw a rectangle on the image.")
        self._lbl_status.setStyleSheet("color:#888888;font-size:10px")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_status)

        lay.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_confirm = QtWidgets.QPushButton("Confirm")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_confirm.setStyleSheet(
            "background:#005f6b;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_cancel.setStyleSheet(
            "background:#4a1a1a;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")

        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.adjustSize()
        self.set_step(0, can_confirm=False)

    def set_step(self, step: int, can_confirm: bool):
        title, inst = self._STEP_LABELS[0]
        self._lbl_title.setText(title)
        self._lbl_inst.setText(inst)

        if can_confirm:
            self._lbl_status.setText("Rect drawn.  Click Confirm to auto-locate molds.")
            self._lbl_status.setStyleSheet("color:#00e5ff;font-size:10px")
        else:
            self._lbl_status.setText("Draw a rectangle on the image.")
            self._lbl_status.setStyleSheet("color:#888888;font-size:10px")

        self._btn_confirm.setEnabled(can_confirm)
        self._btn_confirm.setText("Confirm & Auto-Find Molds")

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
        self._setup_cb = None
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
        gb_insp = QtWidgets.QGroupBox("Inspection")
        fl_insp = QtWidgets.QFormLayout(gb_insp)
        self.spin_tm_threshold = self._bind_fspin("tm_threshold", fl_insp,
                                                   "Similarity threshold")
        note_insp = QtWidgets.QLabel(
            "Similarity: higher = stricter match required.")
        note_insp.setStyleSheet("color:#888;font-size:9px")
        note_insp.setWordWrap(True)
        fl_insp.addRow("", note_insp)
        lay.addWidget(gb_insp)

        # ── Grid Letters ──────────────────────────────────────────
        gb_gl = QtWidgets.QGroupBox("Grid Letters  (9 slots, comma-separated)")
        fl_gl = QtWidgets.QVBoxLayout(gb_gl)

        hint3 = QtWidgets.QLabel(
            "Slots:  [1,2,3] / [4,5,6] / [7,8,9]   empty = skip")
        hint3.setStyleSheet("color:#888;font-size:9px")
        fl_gl.addWidget(hint3)

        self.grid_letters_edit = QtWidgets.QLineEdit(
            self._sm.get_str("grid_letters"))
        self.grid_letters_edit.setFont(QtGui.QFont("Courier New", 9))
        self.grid_letters_edit.setPlaceholderText(
            "e.g.  A,B,C,,E,,G,,   (9 comma-sep, empty=skip)")
        self.grid_letters_edit.textChanged.connect(self._on_grid_letters_changed)
        fl_gl.addWidget(self.grid_letters_edit)

        self._io_recipe_lbl = QtWidgets.QLabel("IO recipe: —")
        self._io_recipe_lbl.setStyleSheet("color:#888;font-size:9px")
        fl_gl.addWidget(self._io_recipe_lbl)

        lay.addWidget(gb_gl)

        # ── Frame Recipe Info (read-only) ─────────────────────
        gb_ri = QtWidgets.QGroupBox("Frame Recipe")
        fl_ri = QtWidgets.QVBoxLayout(gb_ri)
        self._recipe_lbl = QtWidgets.QLabel("No recipe on disk.")
        self._recipe_lbl.setStyleSheet("color:#888;font-size:9px")
        self._recipe_lbl.setWordWrap(True)
        fl_ri.addWidget(self._recipe_lbl)
        lay.addWidget(gb_ri)

        # ── Camera Settings ───────────────────────────────────────
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
        self.log_box.setFixedHeight(200)
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

    def _on_preview_setup(self):
        if self._setup_cb:
            self._setup_cb()

    def set_setup_callback(self, cb):
        """Register callable invoked when Preview Setup Points is clicked."""
        self._setup_cb = cb

    # ---- public API ----
    def log(self, msg: str, color: str = "#dddddd"):
        ts   = datetime.now().strftime("%H:%M:%S")
        html = f'<span style="color:{color}">[{ts}] {msg}</span>'
        self.log_box.append(html)

    def apply_settings(self):
        sm    = self._sm
        pairs = [
            (self.spin_tm_threshold, "tm_threshold"),
            (self.spin_exposure,     "camera_exposure_us"),
        ]
        for spin, key in pairs:
            spin.blockSignals(True)
            spin.setRange(sm.get_min(key), sm.get_max(key))
            spin.setValue(sm.get(key))
            spin.blockSignals(False)

        self.grid_letters_edit.blockSignals(True)
        self.grid_letters_edit.setText(sm.get_str("grid_letters"))
        self.grid_letters_edit.blockSignals(False)

    def pin_search_params(self) -> dict:
        sm = self._sm
        return {
            "score_thr":   sm.get("pin_score_threshold"),
            "iou_thr":     sm.get("iou_threshold"),
            "max_matches": int(sm.get("max_matches")),
            "scale_min":   sm.get("search_scale_min"),
            "scale_max":   sm.get("search_scale_max"),
            "scale_steps": int(sm.get("search_scale_steps")),
        }

    def tm_threshold(self) -> float:
        return float(self.spin_tm_threshold.value())

    def font_list(self, ct: "ContourTemplate") -> list:
        """
        Return all template names from the templates/ folder.
        Auto-detected at run time — no user input required.
        """
        return ct.list_templates()

    def refresh_recipe_info(self, recipe: dict | None):
        """
        Update the Frame Recipe read-only display.
        Pass None to show 'No recipe on disk.'
        """
        if recipe is None:
            self._recipe_lbl.setText("No recipe on disk.")
            self._recipe_lbl.setStyleSheet("color:#888;font-size:9px")
            return

        def _sec_info(sec: dict, label: str) -> str:
            x, y, w, h = sec.get("contour", [0, 0, 0, 0])
            off = sec.get("offset")
            if off:
                return (f"{label}: {w}x{h} px  "
                        f"offset ({off[0]:+d}, {off[1]:+d})")
            return f"{label}: {w}x{h} px  at ({x},{y})"

        lines = [
            _sec_info(recipe["frame"],  "FRAME"),
            _sec_info(recipe["mold_a"], "MOLD A"),
            _sec_info(recipe["mold_b"], "MOLD B"),
        ]
        self._recipe_lbl.setText("\n".join(lines))
        self._recipe_lbl.setStyleSheet("color:#aaddff;font-size:9px")

    def _on_grid_letters_changed(self, text: str):
        self._sm.set_str("grid_letters", text)
        self._sm.save()
        if self._grid_changed_cb:
            self._grid_changed_cb(self.grid_letters())
        
    def set_io_recipe_label(self, letters: list):
        """Update the IO recipe status label."""
        text = ",".join(letters)
        self._io_recipe_lbl.setText(f"IO recipe: {text}")
        self._io_recipe_lbl.setStyleSheet("color:#aaddff;font-size:9px")

    def set_grid_changed_callback(self, cb):
        """Register callable(list[str]) invoked when grid letters text changes."""
        self._grid_changed_cb = cb

    def grid_letters(self) -> list:
        """
        Parse grid_letters field.
        Returns list of exactly 9 strings. Empty string = skip.
        """
        raw   = self.grid_letters_edit.text()
        parts = raw.split(",")
        # Pad or trim to exactly 9
        parts = parts[:9]
        while len(parts) < 9:
            parts.append("")
        return [p.strip().upper() for p in parts]


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
        self._sm    = SettingsManager(SETTINGS_FILE)
        self._setup_cb        = None
        self._grid_changed_cb = None
        self._ctrl  = InspectionController(self._sm)
        self._mock_io_worker: MockIOWorker | None = None
        self._current_grid: list = []
        self._io_recipe:    list = []        # last recipe received from IO
        self._run_from_io:  bool = False     # True = run triggered by MachineIO
        
        # General draw mode
        self._mode    = None   # "frame" | "font" | "mask" | None
        self._pending = None   # pending font template name

        # Frame template creation — 3-step state
        self._frame_step:  int              = 0
        self._frame_rects: list             = [None, None, None]
        self._frame_panel: FrameTemplatePanel | None = None
        # Overlay tag keys for each step so re-draw replaces the right box
        self._FRAME_TAGS = ["FRAME", "MOLD_A", "MOLD_B"]

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
            try:
                self._panel.refresh_recipe_info(
                    self._ctrl.load_frame_recipe())
            except Exception:
                pass

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
        btn("Export CSV", self._export_csv)
        btn("Mock IO Recipe", self._trigger_mock_io, "#334455")
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
    # Frame template creation — 3-step sequence
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

        self._frame_step  = 0
        self._frame_rects = [None, None, None]
        self._mode        = "frame"
        self._clear_frame_overlays()

        self._frame_panel = FrameTemplatePanel(
            on_confirm = self._on_frame_confirm,
            on_cancel  = self._on_frame_cancel,
            parent     = self,
        )
        geo = self.geometry()
        pw  = self._frame_panel.sizeHint().width()
        self._frame_panel.move(geo.right() - pw - 20, geo.top() + 60)
        self._frame_panel.show()

        self._view.set_draw_mode(True)
        self._view.set_constraint_rect(None)
        self._panel.log("Frame template — Draw FRAME rect, molds auto-detected.", "#00e5ff")

    def _on_frame_confirm(self):
        if self._frame_rects[0] is None:
            return
        self._finish_frame()

    def _mold_a_constraint_rect(self) -> QtCore.QRect | None:
        """
        Constraint zone for MOLD A (above frame centre).
        Size    : 1.3 * frame_height  (square)
        X       : left = frame right edge
        Y       : bottom = frame centre Y  (zone sits above centre)
        """
        fr = self._frame_rects[0]
        if fr is None:
            return None
        ih, iw = self._image.shape[:2]
        zone = int(1.3 * fr.height())
        fcy  = int(fr.y() + fr.height()/ 2.2)
        x1 = max(0,  fr.x() + fr.width())
        y2 = fcy                               # bottom = frame centre Y
        y1 = max(0,  y2 - zone)
        x2 = min(iw, x1 + zone)
        return QtCore.QRect(x1, y1, x2 - x1, y2 - y1)

    def _mold_b_constraint_rect(self) -> QtCore.QRect | None:
        """
        Constraint zone for MOLD B (below frame centre).
        Same size and X as MOLD A, Y mirrored around frame centre.
        Y       : top = frame centre Y  (zone sits below centre)
        """
        fr = self._frame_rects[0]
        if fr is None:
            return None
        ih, iw = self._image.shape[:2]
        zone = int(1.3 * fr.height())
        fcy  =  int(fr.y() + fr.height()/ 2.2)
        x1 = max(0,  fr.x() + fr.width())
        y1 = fcy                               # top = frame centre Y
        y2 = min(ih, y1 + zone)
        x2 = min(iw, x1 + zone)
        return QtCore.QRect(x1, y1, x2 - x1, y2 - y1)

    def _on_frame_cancel(self):
        self._clear_frame_overlays()
        self._view.set_draw_mode(False)
        self._view.set_constraint_rect(None)
        self._mode        = None
        self._frame_rects = [None, None, None]
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None
        self._panel.log("Frame template creation cancelled.", "#888888")

    def _finish_frame(self):
        self._view.set_draw_mode(False)
        self._view.set_constraint_rect(None)
        self._mode = None
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None

        letters      = self._panel.grid_letters()
        a_constraint = self._mold_a_constraint_rect()
        b_constraint = self._mold_b_constraint_rect()

        if a_constraint is None or b_constraint is None:
            self._panel.log("Frame rect not set — cannot compute mold zones.", "#ff4444")
            return

        self._panel.log("Auto-searching mold A/B ...", "#aaddff")
        ok, a_found, b_found = self._ctrl.save_frame_recipe(
            self._image,
            self._frame_rects[0],
            grid_letters      = letters,
            mold_a_constraint = a_constraint,
            mold_b_constraint = b_constraint,
        )

        if not ok:
            self._panel.log("Frame recipe save failed.", "#ff4444")
            self._clear_frame_overlays()
            return

        # Report mold search results
        if a_found and b_found:
            self._panel.log("Frame recipe saved — Mold A/B auto-located.", "#88ff88")
        else:
            missing = []
            if not a_found: missing.append("Mold A")
            if not b_found: missing.append("Mold B")
            names = " and ".join(missing)
            self._panel.log(
                f"Recipe saved but {names} used fallback — verify result.", "#ffaa44")
            QtWidgets.QMessageBox.warning(
                self, "Mold Auto-Detect Failed",
                f"{names} could not be auto-located.\n\n"
                f"The constraint zone centre was used as fallback.\n"
                f"Please verify the recipe preview, then retry if incorrect.")

        try:
            recipe = self._ctrl.load_frame_recipe()
            self._panel.refresh_recipe_info(recipe)
            FrameRecipePreviewDialog(recipe, parent=self).exec_()
        except Exception:
            pass

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

    def _on_frame_roi(self, rect: QtCore.QRect):
        x = max(0, rect.x())
        y = max(0, rect.y())
        w = min(rect.width(),  self._image.shape[1] - x)
        h = min(rect.height(), self._image.shape[0] - y)
        if w < 4 or h < 4:
            return

        clipped = QtCore.QRect(x, y, w, h)
        tag     = "FRAME"
        self._view._overlays = [
            ov for ov in self._view._overlays if ov[2] != tag]
        self._view.add_overlay(clipped, QtGui.QColor(0, 224, 255), tag, "dash")

        self._frame_rects[0] = clipped
        if self._frame_panel:
            self._frame_panel.set_step(0, can_confirm=True)
        self._view.set_draw_mode(True)
        
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

        # Pre-flight: validate recipe + templates exist
        missing = self._ctrl.prepare(grid)
        if missing:
            if "__NO_RECIPE__" in missing:
                msg = "No frame recipe found.\nCreate a frame template first."
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
            "score_thr":   sm.get("pin_score_threshold"),
            "iou_thr":     sm.get("iou_threshold"),
            "max_matches": int(sm.get("max_matches")),
            "scale_min":   sm.get("search_scale_min"),
            "scale_max":   sm.get("search_scale_max"),
            "scale_steps": int(sm.get("search_scale_steps")),
        }
        tm_thr = sm.get("tm_threshold")
        self._ctrl.set_run_params(pin_params, tm_thr)

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
            f"=== Start [{mode}] grid_src={src}  tm_thr={tm_thr:.3f}"
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

    def _trigger_mock_io(self):
        """Fire the mock IO worker — simulates external recipe delivery."""
        if self._mock_io_worker and self._mock_io_worker.isRunning():
            self._panel.log("Mock IO already running.", "#ffaa44")
            return
        self._mock_io_worker = MockIOWorker(self)
        self._mock_io_worker.sig_recipe.connect(self._on_io_recipe_received)
        self._mock_io_worker.start()
        self._panel.log("Mock IO started — recipe arrives in ~2s.", "#aaddff")

    def _on_io_recipe_received(self, letters: list):
        """Slot: called when MockIOWorker delivers a recipe."""
        self._io_recipe = list(letters)
        text = ",".join(letters)
        self._panel.grid_letters_edit.setText(text)
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

    # ----------------------------------------------------------
    # Export / save
    # ----------------------------------------------------------
    def _export_csv(self):
        if not self._ctrl.results:
            self._panel.log("No results yet.", "#ffaa44"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV",
            f"result_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "CSV (*.csv)")
        if not path:
            return
        keys = list(self._ctrl.results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._ctrl.results)
        self._panel.log(f"CSV -> {path}", "#88ff88")

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