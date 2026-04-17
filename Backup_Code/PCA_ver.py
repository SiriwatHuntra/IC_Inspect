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
    Otsu threshold -> findContours(RETR_TREE, TC89_KCOS)
    area filter >= 4 px^2
    stored as contour points + pre-rendered filled canvas (base64 PNG)
    font templates additionally store pca_angle (rotation normalised at save)

  PIN search  : TM_CCOEFF_NORMED on rendered canvas, multi-scale + IoU NMS
  Font inspect: 4-step hard-fail pipeline
                  1. Presence   — contours must exist
                  2. Shift      — contour centre vs expected position
                  3. Rotation   — PCA angle must be within FONT_ROTATION_LIMIT
                  4. Similarity — multi-scale TM_CCOEFF_NORMED vs stored canvas
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
FONT_SHIFT_THRESHOLD = 60       # px  — max allowed mark centre shift
FONT_ROTATION_LIMIT  = 15.0     # deg — max PCA angle before hard fail
FONT_TM_SCALE_MIN    = 0.80     # multi-scale TM range for font matching
FONT_TM_SCALE_MAX    = 1.3
FONT_TM_SCALE_STEPS  = 5
FONT_CANVAS_SIZE     = 64       # internal normalised canvas (px, not displayed)

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
    ("tm_threshold",         0.80,  0.30,  1.00,  True ),
    ("cell_box_size",          45,    10,   200,  False),
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
        lines  = ["# IC Inspection Settings\n",
                  "# Header  value  min  max\n#\n"]
        for hdr, _v, _mn, _mx, is_float in _SETTINGS_DEFAULTS:
            d = self._data[hdr]
            if is_float:
                line = (f"{hdr:<25} {d['value']:>10.4f}"
                        f"  {d['min']:>10.4f}  {d['max']:>10.4f}\n")
            else:
                line = (f"{hdr:<25} {int(d['value']):>10d}"
                        f"  {int(d['min']):>10d}  {int(d['max']):>10d}\n")
            lines.append(line)
        lines.append("#\n# String settings\n#\n")
        for k, v in self._str_data.items():
            safe = v.replace("\n", "\\n")
            lines.append(f"{k:<25} \"{safe}\"\n")
        with open(target, "w") as f:
            f.writelines(lines)
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
            #devs  = tl.EnumerateDevices()

            #device = None
            if self._serial:
                devs = tl.EnumerateDevices()
                device = None
                for d in devs:
                    if d.GetSerialNumber() == self._serial:
                        device = tl.CreateDevice(d)
                        break
                if device is None:
                    print(f"[Camera] Serial '{self._serial}' not found.")
                    return False
            else:
                if not devs:
                    print("[Camera] No Basler cameras detected.")
                    return False
                device = tl.CreateDevice(devs[0])

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
        """
        Poll portStart until active-LOW pulse detected or stop requested.
        Debounced: signal must hold LOW for two reads 5 ms apart.
        Returns True = trigger received, False = stop requested.
        stop_flag_fn : callable returning bool — True means stop.
        """
        if not self._gpio_ok:
            # Mock: return immediately so the worker can simulate
            import time
            while not stop_flag_fn():
                time.sleep(0.05)
                return True
            return False

        import time
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
# STORAGE — ContourTemplate
# =========================================================
class ContourTemplate:
    """
    Contour-based template store for both PIN and font letter templates.

    Extraction
    ----------
    Otsu threshold -> findContours(RETR_TREE, TC89_KCOS)
    Keep all contours with area >= MIN_CONTOUR_AREA (outer + holes).

    Font templates additionally apply PCA rotation normalisation at save time:
      all contour points merged -> PCA principal axis -> rotate by -θ
      normalised canvas resized to FONT_CANVAS_SIZE for TM comparison.

    Storage  (JSON per template)
    ----------------------------
    {
      "name":        str,
      "roi":         [x, y, w, h],
      "type":        "contour",
      "contours":    [[[x,y], ...], ...],
      "canvas_b64":  str,                  filled grayscale canvas (base64 PNG)
      "canvas_w":    int,
      "canvas_h":    int,
      "pca_angle":   float,                degrees applied at save (0.0 for PIN)
    }

    Load returns
    ------------
    {
      "name", "roi", "pca_angle",
      "contours": list of np.int32 arrays,
      "canvas":   uint8 grayscale ndarray
    }
    """

    TEMPLATE_DIR = "templates"

    def __init__(self):
        os.makedirs(self.TEMPLATE_DIR, exist_ok=True)

    # ---- contour extraction (shared) ----
    @staticmethod
    def extract_contours(roi) -> tuple:
        """
        Accepts gray (H×W) or BGR (H×W×3).
        Returns (contours, canvas, thresh_img).
        """
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_cnts, _ = cv2.findContours(th, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_TC89_KCOS)
        contours = [c for c in raw_cnts
                    if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        h, w = gray.shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(canvas, contours, -1, 255, cv2.FILLED)
        return contours, canvas, th

    # ---- PCA rotation helper ----
    @staticmethod
    def _pca_angle(contours: list) -> float:
        """
        Compute principal axis angle (degrees) from merged contour points.
        Returns angle in (-90, 90]. Returns 0.0 if computation fails.
        """
        pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
        if len(pts) < 5:
            return 0.0
        _, eigenvectors = cv2.PCACompute(pts, mean=np.array([]))
        angle = float(np.degrees(np.arctan2(eigenvectors[0, 1],
                                            eigenvectors[0, 0])))
        # Normalise to (-90, 90]
        if angle <= -90.0:
            angle += 180.0
        elif angle > 90.0:
            angle -= 180.0
        return round(angle, 2)

    # ---- rotate gray image around centre ----
    @staticmethod
    def _rotate_gray(gray: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    # ---- save (font template — PCA normalised) ----
    def save(self, name: str, roi_bgr: np.ndarray, roi_rect: tuple) -> str:
        """
        Extract contours, PCA-normalise rotation, resize canvas to
        FONT_CANVAS_SIZE, and save template JSON.
        name must be UPPER-CASED by the caller.
        Returns the saved file path.
        Raises RuntimeError if no contours found.
        """
        if roi_bgr.ndim == 3:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_bgr.copy()

        contours, _, _ = self.extract_contours(gray)
        if not contours:
            raise RuntimeError(f"No contours found in ROI for '{name}'")

        # PCA rotation correction
        angle = self._pca_angle(contours)
        rotated = self._rotate_gray(gray, -angle)

        # Re-extract on rotated image
        contours_rot, canvas_rot, _ = self.extract_contours(rotated)
        if not contours_rot:
            # Fallback: use original if rotation broke contours
            contours_rot = contours
            canvas_rot   = cv2.resize(
                np.zeros_like(gray), (gray.shape[1], gray.shape[0]))
            cv2.drawContours(canvas_rot, contours, -1, 255, cv2.FILLED)

        # Resize canvas to fixed internal size
        norm_canvas = cv2.resize(canvas_rot,
                                 (FONT_CANVAS_SIZE, FONT_CANVAS_SIZE),
                                 interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".png", norm_canvas)
        if not ok:
            raise RuntimeError("Failed to encode canvas PNG")
        canvas_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        data = {
            "name":       name,
            "roi":        list(roi_rect),
            "type":       "contour",
            "pca_angle":  angle,
            "contours":   [c.tolist() for c in contours_rot],
            "canvas_b64": canvas_b64,
            "canvas_w":   FONT_CANVAS_SIZE,
            "canvas_h":   FONT_CANVAS_SIZE,
        }
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    # ---- save_pin (PIN template — no PCA, raw canvas) ----
    def save_pin(self, name: str, roi_bgr: np.ndarray, roi_rect: tuple) -> str:
        """
        Save a PIN/frame template without PCA normalisation.
        Used by _encode_section in InspectionController.
        """
        contours, canvas, _ = self.extract_contours(roi_bgr)
        if not contours:
            raise RuntimeError(f"No contours found in ROI for PIN '{name}'")
        h, w = canvas.shape[:2]
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("Failed to encode canvas PNG")
        canvas_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        data = {
            "name":       name,
            "roi":        list(roi_rect),
            "type":       "contour",
            "pca_angle":  0.0,
            "contours":   [c.tolist() for c in contours],
            "canvas_b64": canvas_b64,
            "canvas_w":   w,
            "canvas_h":   h,
        }
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    # ---- load ----
    def load(self, name: str) -> dict:
        """
        Load a contour template from disk.
        Returns dict with decoded numpy arrays ready for use.
        Raises ValueError if the file is not a contour template.
        """
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("type") != "contour":
            raise ValueError(
                f"'{name}' is not a contour template (got: {data.get('type')})")

        data["pca_angle"] = float(data.get("pca_angle", 0.0))
        data["contours"]  = [np.array(c, dtype=np.int32)
                             for c in data["contours"]]
        raw            = base64.b64decode(data["canvas_b64"])
        arr            = np.frombuffer(raw, dtype=np.uint8)
        data["canvas"] = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return data

    # ---- queries ----
    def is_saved(self, name: str) -> bool:
        path = os.path.join(self.TEMPLATE_DIR, f"{name}_template.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("type") == "contour"
        except Exception:
            return False

    def list_templates(self) -> list:
        return sorted(
            fn.replace("_template.json", "")
            for fn in os.listdir(self.TEMPLATE_DIR)
            if fn.endswith("_template.json")
        )


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
# INSPECTION ENGINE  — pure cv2, no Qt, no file I/O
# =========================================================
class InspectionEngine:
    """
    Two inspection primitives.

    find_all_pin_templates(image_bgr, tmpl, ...)
        Multi-scale TM_CCOEFF_NORMED on the template's pre-rendered canvas.
        Accepts a pre-loaded template dict — no disk I/O inside.
        Returns match list sorted best-first.

    compare_roi(roi_bgr, tmpl, tm_thr, exp_dx, exp_dy, mold_cx, mold_cy)
        Four-step hard-fail font inspection:
          1. Presence  2. Shift  3. Rotation+Normalize  4. Canvas TM
        Returns {pass, similarity, threshold, shift_px, pca_angle,
                 scale_used, reason}.
    """

    # ---- IoU (NMS helper) ----
    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw  = max(0, ix2 - ix1)
        ih  = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    # ---- PIN search ----
    def find_all_pin_templates(self,
                               image_bgr:   np.ndarray,
                               tmpl:        dict,
                               score_thr:   float = 0.75,
                               iou_thr:     float = 0.50,
                               max_matches: int   = 10,
                               mask:        np.ndarray = None,
                               scale_min:   float = 0.75,
                               scale_max:   float = 1.25,
                               scale_steps: int   = 7) -> list:
        """
        Slide the template canvas over the image at each pyramid scale.
        Collect all locations >= score_thr, IoU-NMS across all scales.

        Parameters
        ----------
        tmpl : pre-loaded dict from ContourTemplate.load() — must have 'canvas'.

        Returns
        -------
        list of (cx, cy, score, matched_w, matched_h, best_scale)
        sorted best-score-first, capped at max_matches.
        """
        canvas_orig = tmpl.get("canvas")
        if canvas_orig is None or canvas_orig.size == 0:
            print("[PinSearch] Template has no canvas.")
            return []

        ph0, pw0 = canvas_orig.shape[:2]
        if ph0 < 2 or pw0 < 2:
            return []

        gray = image_bgr if image_bgr.ndim == 2 \
               else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape[:2]

        # Build scale pyramid — always include 1.0
        if scale_steps < 2:
            scales = [1.0]
        else:
            scales = list(np.linspace(scale_min, scale_max, scale_steps))
            if 1.0 not in scales:
                scales.append(1.0)
            scales = sorted(set(round(s, 4) for s in scales))

        # Accumulate: (score, x1, y1, x2, y2, cx, cy, scale)
        all_candidates = []

        for sc in scales:
            pw = max(4, int(round(pw0 * sc)))
            ph = max(4, int(round(ph0 * sc)))
            if pw > iw or ph > ih:
                continue

            patch = cv2.resize(canvas_orig, (pw, ph),
                               interpolation=cv2.INTER_AREA if sc < 1.0
                               else cv2.INTER_LINEAR)

            res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)

            if mask is not None:
                rh, rw  = res.shape[:2]
                mask_rs = cv2.resize(mask, (rw, rh),
                                     interpolation=cv2.INTER_NEAREST)
                res     = res * (mask_rs.astype(np.float32) / 255.0)

            locs = np.argwhere(res >= score_thr)
            for (r, c) in locs:
                score = float(res[r, c])
                x1, y1 = int(c), int(r)
                x2, y2 = x1 + pw, y1 + ph
                cx, cy = x1 + pw // 2, y1 + ph // 2
                all_candidates.append((score, x1, y1, x2, y2, cx, cy, sc))

        if not all_candidates:
            return []

        all_candidates.sort(key=lambda v: v[0], reverse=True)

        kept       = []
        suppressed = set()
        for i, ci in enumerate(all_candidates):
            if i in suppressed:
                continue
            kept.append(ci)
            box_i = (ci[1], ci[2], ci[3], ci[4])
            for j, cj in enumerate(all_candidates[i + 1:], start=i + 1):
                if j in suppressed:
                    continue
                if self._iou(box_i, (cj[1], cj[2], cj[3], cj[4])) > iou_thr:
                    suppressed.add(j)

        return [
            (k[5], k[6], round(k[0], 4),
             k[3] - k[1],   # matched width
             k[4] - k[2],   # matched height
             k[7])           # best_scale
            for k in kept[:max_matches]
        ]

    # ---- font inspect — 4-step hard-fail ----
    def compare_roi(self,
                    roi_bgr:  np.ndarray,
                    tmpl:     dict,
                    tm_thr:   float,
                    exp_dx:   int,
                    exp_dy:   int,
                    mold_cx:  int,
                    mold_cy:  int) -> dict:
        """
        Four-step hard-fail font inspection.

        Steps
        -----
        1. Presence   — contours must exist in ROI.
        2. Shift      — contour bbox centre vs expected position.
        3. Normalise  — rotate ROI by -pca_angle to match saved canvas orientation.
                        PCA has 180° ambiguity, so both 0° and 180° are tried;
                        best TM score wins (handles 6/9, b/d ambiguity).
        4. Similarity — multi-scale TM_CCOEFF_NORMED vs stored canvas.
                        scale < 1.0 : shrink template, slide over FONT_CANVAS_SIZE roi
                        scale >= 1.0: pad roi, slide FONT_CANVAS_SIZE template over it

        Returns
        -------
        {pass, similarity, threshold, shift_px, pca_angle, scale_used, reason}
        """
        _fail = lambda reason: {
            "pass":       False,
            "similarity": 0.0,
            "threshold":  tm_thr,
            "shift_px":   0.0,
            "pca_angle":  0.0,
            "scale_used": 1.0,
            "reason":     reason,
        }

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) \
            if roi_bgr.ndim == 3 else roi_bgr.copy()

        # ── STEP 1 : Presence ──────────────────────────────────
        contours, _, _ = ContourTemplate.extract_contours(gray)
        if not contours:
            return _fail("no mark detected")

        # ── STEP 2 : Shift ─────────────────────────────────────
        all_pts        = np.vstack([c.reshape(-1, 2) for c in contours])
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        local_cx       = bx + bw // 2
        local_cy       = by + bh // 2

        roi_h, roi_w = gray.shape[:2]
        roi_ox = mold_cx + exp_dx - roi_w // 2
        roi_oy = mold_cy + exp_dy - roi_h // 2

        actual_dx = (roi_ox + local_cx) - mold_cx
        actual_dy = (roi_oy + local_cy) - mold_cy
        shift_px  = float(np.hypot(actual_dx - exp_dx, actual_dy - exp_dy))

        if shift_px > FONT_SHIFT_THRESHOLD:
            return {**_fail(
                f"shift {shift_px:.1f}px exceeds {FONT_SHIFT_THRESHOLD}px"),
                "shift_px": round(shift_px, 2)}

        # ── STEP 3 : PCA normalise ─────────────────────────────
        # Compute ROI PCA angle and rotate to 0° to match saved canvas.
        # Try both pca_angle and pca_angle+180° — take candidate with better TM.
        pts       = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
        pca_angle = 0.0
        if len(pts) >= 5:
            _, eigenvectors = cv2.PCACompute(pts, mean=np.array([]))
            angle = float(np.degrees(np.arctan2(eigenvectors[0, 1],
                                                eigenvectors[0, 0])))
            if angle <= -90.0:
                angle += 180.0
            elif angle > 90.0:
                angle -= 180.0
            pca_angle = round(angle, 2)

        # Build two candidate normalised ROIs (0° and 180° flip)
        def _normalise(g: np.ndarray, extra_rot: float) -> np.ndarray:
            rot_angle = -(pca_angle + extra_rot)
            if abs(rot_angle) > 0.5:
                g = ContourTemplate._rotate_gray(g, rot_angle)
            return cv2.resize(g, (FONT_CANVAS_SIZE, FONT_CANVAS_SIZE),
                            interpolation=cv2.INTER_AREA)

        candidates = [_normalise(gray, 0.0), _normalise(gray, 180.0)]

        # ── STEP 4 : Canvas TM multi-scale ─────────────────────
        tmpl_canvas = tmpl.get("canvas")
        if tmpl_canvas is None or tmpl_canvas.size == 0:
            return {**_fail("template canvas missing"),
                    "shift_px":  round(shift_px, 2),
                    "pca_angle": pca_angle}

        CS = FONT_CANVAS_SIZE
        scales = list(np.linspace(FONT_TM_SCALE_MIN, FONT_TM_SCALE_MAX,
                                FONT_TM_SCALE_STEPS))
        if 1.0 not in scales:
            scales.append(1.0)
        scales = sorted(set(round(s, 4) for s in scales))

        def _best_sim_for_roi(norm_roi: np.ndarray) -> tuple:
            best_sim   = -1.0
            best_scale = 1.0
            for sc in scales:
                if sc < 1.0:
                    tw    = max(4, int(round(CS * sc)))
                    th    = max(4, int(round(CS * sc)))
                    patch = cv2.resize(tmpl_canvas, (tw, th),
                                    interpolation=cv2.INTER_AREA)
                    res   = cv2.matchTemplate(norm_roi, patch,
                                            cv2.TM_CCOEFF_NORMED)
                else:
                    pad   = max(1, int(round((sc - 1.0) * CS / 2)))
                    roi_p = cv2.copyMakeBorder(norm_roi, pad, pad, pad, pad,
                                            cv2.BORDER_CONSTANT, value=0)
                    if roi_p.shape[0] < CS or roi_p.shape[1] < CS:
                        continue
                    res = cv2.matchTemplate(roi_p, tmpl_canvas,
                                            cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_sim:
                    best_sim   = max_val
                    best_scale = sc
            return best_sim, best_scale

        sim0, sc0 = _best_sim_for_roi(candidates[0])
        sim1, sc1 = _best_sim_for_roi(candidates[1])

        if sim0 >= sim1:
            best_sim, best_scale = sim0, sc0
        else:
            best_sim, best_scale = sim1, sc1
            pca_angle = round(pca_angle + 180.0, 2)   # reflect actual used angle

        similarity = round(float(max(0.0, best_sim)), 4)
        passed     = similarity >= tm_thr

        return {
            "pass":       passed,
            "similarity": similarity,
            "threshold":  tm_thr,
            "shift_px":   round(shift_px, 2),
            "pca_angle":  pca_angle,
            "scale_used": best_scale,
            "reason":     "OK" if passed
                        else f"similarity {similarity:.3f} < {tm_thr:.3f}",
        }


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
        Crop rect from image_bgr, extract contours, encode canvas.
        offset = (dx, dy) relative to frame centre — stored for mold sections.
        Returns a JSON-serialisable dict.
        """
        ih, iw = image_bgr.shape[:2]
        x  = max(0, rect.x())
        y  = max(0, rect.y())
        w  = min(rect.width(),  iw - x)
        h  = min(rect.height(), ih - y)
        src = image_bgr if image_bgr.ndim == 2 \
              else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        roi = src[y:y + h, x:x + w].copy()

        contours, canvas, _ = ContourTemplate.extract_contours(roi)
        if not contours:
            raise RuntimeError(f"No contours found in region ({x},{y},{w},{h})")

        ch, cw = canvas.shape[:2]
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("Canvas PNG encode failed")
        canvas_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        section = {
            "contour":    [x, y, w, h],
            "contours":   [c.tolist() for c in contours],
            "canvas_b64": canvas_b64,
            "canvas_w":   cw,
            "canvas_h":   ch,
        }
        if offset is not None:
            section["offset"] = list(offset)
        return section

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
                        frame_rect:   QtCore.QRect,
                        mold_a_rect:  QtCore.QRect,
                        mold_b_rect:  QtCore.QRect,
                        grid_letters: list,
                        grid_a:       list,
                        grid_b:       list) -> bool:
        """
        Extract contours for all three regions and write pin_recipe.json.
        Mold A/B offsets stored relative to frame centre.
        grid_letters : list of 9 strings (empty string = skip).
        grid_a/b     : list of up to 9 (dx, dy, exp_dx, exp_dy) tuples
                       — one per non-empty slot, in row-major order.
                       None entry = skipped slot.
        Returns True on success.
        """
        try:
            fcx = frame_rect.x() + frame_rect.width()  // 2
            fcy = frame_rect.y() + frame_rect.height() // 2

            def _mold_offset(r: QtCore.QRect) -> tuple:
                mcx = r.x() + r.width()  // 2
                mcy = r.y() + r.height() // 2
                return (mcx - fcx, mcy - fcy)

            frame_sec  = self._encode_section(image_bgr, frame_rect)
            mold_a_sec = self._encode_section(
                image_bgr, mold_a_rect, offset=_mold_offset(mold_a_rect))
            mold_b_sec = self._encode_section(
                image_bgr, mold_b_rect, offset=_mold_offset(mold_b_rect))

            # Normalise grid: always 9 entries, None for skipped slots
            # Each shift entry: (dx, dy, exp_dx, exp_dy)
            def _norm(shifts: list, letters: list) -> list:
                result = []
                s_idx  = 0
                for letter in letters:
                    if letter:
                        if s_idx < len(shifts):
                            entry = shifts[s_idx]
                            if entry is not None:
                                dx, dy, exp_dx, exp_dy = entry
                                result.append({
                                    "letter": letter,
                                    "dx":     int(dx),
                                    "dy":     int(dy),
                                    "exp_dx": int(exp_dx),
                                    "exp_dy": int(exp_dy),
                                })
                            else:
                                result.append(None)
                            s_idx += 1
                        else:
                            result.append(None)
                    else:
                        result.append(None)
                while len(result) < 9:
                    result.append(None)
                return result[:9]

            recipe = {
                "version":      3,
                "frame":        frame_sec,
                "mold_a":       mold_a_sec,
                "mold_b":       mold_b_sec,
                "grid_letters": grid_letters,
                "grid_a":       _norm(grid_a, grid_letters),
                "grid_b":       _norm(grid_b, grid_letters),
            }
            with open(self.RECIPE_FILE, "w") as f:
                json.dump(recipe, f)

            self.cache.pop("FRAME", None)
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
        result = {
            "version":      raw.get("version", 1),
            "frame":        self._decode_section(raw["frame"]),
            "mold_a":       self._decode_section(raw["mold_a"]),
            "mold_b":       self._decode_section(raw["mold_b"]),
            "grid_letters": raw.get("grid_letters", [""] * 9),
            "grid_a":       raw.get("grid_a",       [None] * 9),
            "grid_b":       raw.get("grid_b",       [None] * 9),
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
    def save_font(self, name: str, roi_bgr: np.ndarray, roi_rect: tuple,
                  parent_widget) -> bool:
        try:
            self._ct.save(name, roi_bgr, roi_rect)
            self.cache.pop(name, None)
            TemplatePreviewDialog(roi_bgr, name, parent=parent_widget).exec_()
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

    # ---- inspection pipeline ----
    def run(self,
            image_bgr:  np.ndarray,
            pin_params: dict,
            tm_thr:     float,
            cell_box:   int,
            mask:       np.ndarray = None) -> "InspectionResult":
        """
        Run the full inspection pipeline using grid data stored in recipe.

        Parameters
        ----------
        image_bgr  : source image (1280x1024 BGR).
        pin_params : dict from RightPanel.pin_search_params().
        tm_thr     : TM similarity threshold (0–1, higher = stricter).
        cell_box   : letter crop size in px (square).
        mask       : optional binary search mask.
        """
        self.results = []
        # Promote to BGR for colour annotation regardless of source depth
        src     = image_bgr if image_bgr.ndim == 2 \
                  else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        empty        = InspectionResult(display=display)

        if not self.has_frame_recipe():
            print("[Pipeline] No frame recipe.")
            return empty

        recipe = self.load_frame_recipe()

        # Collect unique non-None letter names from both grids
        all_letters = set()
        for slot in (recipe.get("grid_a", []) + recipe.get("grid_b", [])):
            if slot is not None:
                all_letters.add(slot["letter"])

        if not all_letters:
            print("[Pipeline] No grid slots defined in recipe.")
            return empty

        failed = self.load_cache(list(all_letters))
        if failed:
            print(f"[Pipeline] Missing font templates: {failed}")
            return empty

        # Step 1 — find frame matches
        matches = self._step1_find_frames(
            image_bgr, recipe["frame"], pin_params, mask)

        if not matches:
            print("[Pipeline] No frame matches found.")
            return empty

        ih, iw = image_bgr.shape[:2]

        for f_idx, (fcx, fcy, fscore, fw, fh, fscale) in enumerate(matches):
            fx1 = max(0,    fcx - fw // 2)
            fy1 = max(0,    fcy - fh // 2)
            fx2 = min(iw-1, fx1 + fw)
            fy2 = min(ih-1, fy1 + fh)
            cv2_draw_dashed_rect(display, (fx1, fy1), (fx2, fy2),
                                (0, 224, 255), 1)
            cv2.putText(display, f"F{f_idx+1} {fscore:.2f}",
                        (fx1+2, fy1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 224, 255), 1)

            mold_areas = self._step2_locate_molds(
                recipe, fcx, fcy, fscale, iw, ih, f_idx)

            for area in mold_areas:
                acx, acy = area["cx"], area["cy"]
                aw,  ah  = area["w"],  area["h"]
                ax1 = max(0,    acx - aw // 2)
                ay1 = max(0,    acy - ah // 2)
                ax2 = min(iw-1, ax1 + aw)
                ay2 = min(ih-1, ay1 + ah)
                cv2_draw_dashed_rect(display, (ax1, ay1), (ax2, ay2),
                                    (0, 180, 200), 1)

                t0_mold    = time.perf_counter()
                letter_results = self._step3_inspect_fonts(
                    src, display, acx, acy,
                    area["grid"], tm_thr, cell_box,
                    f_idx, area["label"], iw, ih)
                mold_ms = (time.perf_counter() - t0_mold) * 1000

                cv2.putText(display, f"F{f_idx+1}-{area['label']} [{mold_ms:.1f}ms]",
                            (ax1+2, ay1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 200), 1)

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
            odx, ody = mold["offset"]
            acx = fcx + int(round(odx * fscale))
            acy = fcy + int(round(ody * fscale))
            aw  = int(round(mold["canvas_w"] * fscale))
            ah  = int(round(mold["canvas_h"] * fscale))
            acx = max(aw // 2, min(iw - aw // 2, acx))
            acy = max(ah // 2, min(ih - ah // 2, acy))
            grid_key = "grid_a" if key == "mold_a" else "grid_b"
            areas.append({
                "label":    label,
                "cx":       acx,
                "cy":       acy,
                "w":        aw,
                "h":        ah,
                "contours": mold["contours"],
                "canvas":   mold["canvas"],
                "grid":     recipe.get(grid_key, []),
            })
        return areas

    def _step3_inspect_fonts(self,
                        image_bgr, display,
                        acx, acy,
                        grid: list,
                        tm_thr, cell_box,
                        f_idx, mold_label,
                        iw, ih) -> list:
        results = []
        half    = cell_box // 2

        for slot_idx, slot in enumerate(grid):
            if slot is None:
                continue
            letter  = slot["letter"]
            dx      = slot["dx"]
            dy      = slot["dy"]
            exp_dx  = slot.get("exp_dx", dx)
            exp_dy  = slot.get("exp_dy", dy)

            cell_cx = acx + dx
            cell_cy = acy + dy

            lx1 = max(0,  cell_cx - half)
            ly1 = max(0,  cell_cy - half)
            lx2 = min(iw, lx1 + cell_box)
            ly2 = min(ih, ly1 + cell_box)

            if lx2 <= lx1 or ly2 <= ly1:
                continue

            tmpl = self.cache.get(letter)
            if tmpl is None:
                continue

            roi = image_bgr[ly1:ly2, lx1:lx2]

            t0  = time.perf_counter()
            res = self._eng.compare_roi(
                roi, tmpl, tm_thr,
                exp_dx=exp_dx, exp_dy=exp_dy,
                mold_cx=acx,   mold_cy=acy)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            passed     = res["pass"]
            similarity = res["similarity"]
            reason     = res["reason"]

            col_cv = (0, 200, 0) if passed else (200, 0, 0)
            lbl    = f"{letter}+ {similarity:.2f}" if passed \
                     else f"{letter}- {similarity:.2f}"
            cv2.rectangle(display, (lx1, ly1), (lx2, ly2), col_cv, 2)
            cv2.putText(display, lbl, (lx1 + 2, ly2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_cv, 1)

            results.append({
                "frame_idx":  f_idx + 1,
                "mold":       mold_label,
                "slot":       slot_idx,
                "letter":     letter,
                "pass":       passed,
                "similarity": similarity,
                "threshold":  tm_thr,
                "shift_px":   res["shift_px"],
                "pca_angle":  res["pca_angle"],
                "scale_used": res["scale_used"],
                "reason":     reason,
                "cell_cx":    cell_cx,
                "cell_cy":    cell_cy,
                "elapsed_ms": round(elapsed_ms, 2),
            })

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
                 ctrl:       "InspectionController",
                 io:         "MachineIO",
                 pin_params: dict,
                 tm_thr:     float,
                 cell_box:   int,
                 mask:       np.ndarray,
                 image_io:   "ImageIO",
                 camera:     "BaslerCamera | None" = None):
        super().__init__()
        self._ctrl       = ctrl
        self._io         = io
        self._pin_params = pin_params
        self._tm_thr     = tm_thr
        self._cell_box   = cell_box
        self._mask       = mask
        self._image_io   = image_io
        self._camera     = camera
        self._stop_flag  = False

    def stop(self):
        self._stop_flag = True

    # ---- helpers ----
    def _inspect_one(self, img_gray: np.ndarray) -> "InspectionResult":
        t0     = time.perf_counter()
        result = self._ctrl.run(
            img_gray,
            pin_params = self._pin_params,
            tm_thr     = self._tm_thr,
            cell_box   = self._cell_box,
            mask       = self._mask,
        )
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
                    f"  sim={r['similarity']:.3f}"
                    f"  shift={r['shift_px']:.1f}px"
                    f"  rot={r['pca_angle']:.1f}°"
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
                    f"  sim={r['similarity']:.3f}"
                    f"  shift={r['shift_px']:.1f}px"
                    f"  rot={r['pca_angle']:.1f}°"
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

    def __init__(self, roi_bgr: np.ndarray, name: str, parent=None):
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

        # Re-extract using same logic as ContourTemplate.save
        contours, canvas, _ = ContourTemplate.extract_contours(roi_bgr)

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
        for key, mold_label, grid_key in [
            ("mold_a", "MOLD A", "grid_a"),
            ("mold_b", "MOLD B", "grid_b"),
        ]:
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

            for slot in recipe.get(grid_key, []):
                if slot is None:
                    continue
                name = slot["letter"]
                dx   = slot["dx"]
                dy   = slot["dy"]
                ccx  = acx + dx
                ccy  = acy + dy
                try:
                    td = ct.load(name)
                    lw = td.get("canvas_w", 45)
                    lh = td.get("canvas_h", 45)
                except Exception:
                    lw, lh = 45, 45
                lx1 = max(0, ccx - lw // 2)
                ly1 = max(0, ccy - lh // 2)
                lx2 = min(iw, lx1 + lw)
                ly2 = min(ih, ly1 + lh)
                cv2.rectangle(display, (lx1, ly1), (lx2, ly2),
                            (255, 220, 0), 1)
                cv2.putText(display, name, (lx1 + 2, ly2 - 2),
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

        active = sum(
            1 for g in (recipe.get("grid_a", []) + recipe.get("grid_b", []))
            if g is not None
        )
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
    Floating always-on-top panel — guides through 5 sequential steps.

    Step 0 — FRAME  : rubber-band draw
    Step 1 — MOLD A : rubber-band draw
    Step 2 — MOLD B : rubber-band draw
    Step 3 — Stamp MOLD A slots (one per non-empty grid letter)
    Step 4 — Stamp MOLD B slots (one per non-empty grid letter)
    """

    _STEP_LABELS = [
        ("Step 1 / 5  —  FRAME",
         "Draw the lead frame region on the image."),
        ("Step 2 / 5  —  MOLD A",
         "Draw Mold A region on the image."),
        ("Step 3 / 5  —  MOLD B",
         "Draw Mold B region on the image."),
        ("Step 4 / 5  —  Stamp MOLD A",
         "Click to stamp each letter slot position on the image."),
        ("Step 5 / 5  —  Stamp MOLD B",
         "Click to stamp each letter slot position on the image."),
    ]

    def __init__(self, on_confirm, on_cancel, on_skip, parent=None):
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
        self._on_skip    = on_skip

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

        # Slot progress label — shown only during stamp steps
        self._lbl_slot = QtWidgets.QLabel()
        self._lbl_slot.setStyleSheet(
            "color:#ffcc00;font-size:12px;font-weight:bold")
        self._lbl_slot.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_slot)

        self._lbl_status = QtWidgets.QLabel("Draw a rectangle on the image.")
        self._lbl_status.setStyleSheet("color:#888888;font-size:10px")
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._lbl_status)

        lay.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_confirm = QtWidgets.QPushButton("Confirm")
        self._btn_skip    = QtWidgets.QPushButton("Skip")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_confirm.setStyleSheet(
            "background:#005f6b;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_skip.setStyleSheet(
            "background:#555500;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")
        self._btn_cancel.setStyleSheet(
            "background:#4a1a1a;color:#fff;font-weight:bold;"
            "padding:6px 14px;border-radius:4px")

        self._btn_confirm.setEnabled(False)
        self._btn_skip.setVisible(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_skip.clicked.connect(self._on_skip)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_skip)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.adjustSize()
        self.set_step(0, can_confirm=False)

    def set_step(self, step: int, can_confirm: bool,
                 slot_label: str = "", slot_progress: str = ""):
        title, inst = self._STEP_LABELS[step]
        self._lbl_title.setText(title)
        self._lbl_inst.setText(inst)

        is_stamp = step >= 3
        self._btn_skip.setVisible(is_stamp)
        self._btn_confirm.setVisible(not is_stamp)

        if is_stamp:
            self._lbl_slot.setText(
                f"Slot: [{slot_label}]  {slot_progress}")
            self._lbl_status.setText(
                "Click on the image to stamp this slot's position.")
            self._lbl_status.setStyleSheet("color:#ffcc00;font-size:10px")
        else:
            self._lbl_slot.setText("")
            if can_confirm:
                self._lbl_status.setText(
                    "Rect drawn.  Click Confirm to proceed.")
                self._lbl_status.setStyleSheet("color:#00e5ff;font-size:10px")
            else:
                self._lbl_status.setText(
                    "Draw a rectangle on the image.")
                self._lbl_status.setStyleSheet("color:#888888;font-size:10px")
            self._btn_confirm.setEnabled(can_confirm)
            self._btn_confirm.setText("Finish" if step == 2 else "Confirm")

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        if self._camera:
            self._camera.close()
        self._machine_io.cleanup()   # ← add this
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

        # ── Frame Search ──────────────────────────────────────
        gb_pin = QtWidgets.QGroupBox("Frame Search")
        fl_pin = QtWidgets.QFormLayout(gb_pin)
        self.spin_pin_score   = self._bind_fspin("pin_score_threshold", fl_pin,
                                                  "Score threshold")
        self.spin_iou_thr     = self._bind_fspin("iou_threshold",       fl_pin,
                                                  "IoU threshold")
        self.spin_max_match   = self._bind_ispin("max_matches",         fl_pin,
                                                  "Max matches")
        self.spin_scale_min   = self._bind_fspin("search_scale_min",    fl_pin,
                                                  "Scale min")
        self.spin_scale_max   = self._bind_fspin("search_scale_max",    fl_pin,
                                                  "Scale max")
        self.spin_scale_steps = self._bind_ispin("search_scale_steps",  fl_pin,
                                                  "Scale steps")
        lay.addWidget(gb_pin)

        # ── Inspection ────────────────────────────────────────
        gb_insp = QtWidgets.QGroupBox("Inspection")
        fl_insp = QtWidgets.QFormLayout(gb_insp)
        self.spin_tm_threshold = self._bind_fspin("tm_threshold", fl_insp,
                                                   "Similarity threshold")
        note_insp = QtWidgets.QLabel(
            "Higher = stricter.  1.0 = perfect match required.")
        note_insp.setStyleSheet("color:#888;font-size:9px")
        note_insp.setWordWrap(True)
        fl_insp.addRow("", note_insp)
        lay.addWidget(gb_insp)

        # ── Grid Letters ──────────────────────────────────────────
        gb_gl = QtWidgets.QGroupBox("Grid Letters  (9 slots, comma-separated)")
        fl_gl = QtWidgets.QVBoxLayout(gb_gl)
        self._btn_preview_setup = QtWidgets.QPushButton("Preview Setup Points")
        self._btn_preview_setup.setStyleSheet(
            "background:#1a3a5c;color:#fff;padding:3px 8px;font-size:10px")
        self._btn_preview_setup.clicked.connect(self._on_preview_setup)
        fl_gl.addWidget(self._btn_preview_setup)

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

        lay.addWidget(gb_gl)

        # ── Cell Box Size ─────────────────────────────────────────
        gb_cb = QtWidgets.QGroupBox("Letter Cell Box Size")
        fl_cb = QtWidgets.QFormLayout(gb_cb)
        self.spin_cell_box = self._bind_ispin("cell_box_size", fl_cb, "Box size (px)")
        lay.addWidget(gb_cb)

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
            (self.spin_pin_score,    "pin_score_threshold"),
            (self.spin_iou_thr,      "iou_threshold"),
            (self.spin_max_match,    "max_matches"),
            (self.spin_scale_min,    "search_scale_min"),
            (self.spin_scale_max,    "search_scale_max"),
            (self.spin_scale_steps,  "search_scale_steps"),
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
        return {
            "score_thr":   float(self.spin_pin_score.value()),
            "iou_thr":     float(self.spin_iou_thr.value()),
            "max_matches": int(self.spin_max_match.value()),
            "scale_min":   float(self.spin_scale_min.value()),
            "scale_max":   float(self.spin_scale_max.value()),
            "scale_steps": int(self.spin_scale_steps.value()),
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


    def cell_box_size(self) -> int:
        return int(self.spin_cell_box.value())

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
        self._ctrl  = InspectionController(self._sm)

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

        # Stamp step state
        self._stamp_letters:  list = []   # non-empty grid letters in order
        self._stamp_shifts_a: list = []   # (dx, dy) per stamped slot, mold A
        self._stamp_shifts_b: list = []   # (dx, dy) per stamped slot, mold B
        self._stamp_idx:      int  = 0    # current slot index within current mold
        self._stamp_mold:     str  = "A"  # "A" or "B"
        
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
        self._panel.set_setup_callback(self._preview_setup_points)
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

        btn("📂 Open Image",        self._open_image)
        sep()
        btn("📌 Create Frame Tmpl", self._start_frame,   "#1155aa")
        btn("🔤 Create Font Tmpl",  self._start_font,    "#116611")
        btn("🗺 Masking Template",  self._start_masking, "#333388")
        sep()
        self._btn_run  = btn("▶ Start Run", self._start_run, "#1a6b2a")
        self._btn_stop = btn("■ Stop",      self._stop_run,  "#882200")
        self._btn_stop.setEnabled(False)
        btn("🧹 Clear", self._clear)
        sep()
        btn("⚙ Load Settings",  self._load_settings, "#555500")
        btn("💾 Save Settings",  self._save_settings, "#555500")
        sep()
        btn("💾 Export CSV", self._export_csv)
        btn("🖼 Save Image",  self._save_img)
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
    # Inspection setup point preview — dedicated dialog
    # ----------------------------------------------------------
    def _preview_setup_points(self):
        if self._image is None:
            self._panel.log("No image loaded.", "#ffaa44"); return
        if not self._ctrl.has_frame_recipe():
            self._panel.log("No frame recipe — create it first.", "#ffaa44"); return

        try:
            recipe = self._ctrl.load_frame_recipe()
        except Exception as e:
            self._panel.log(f"Setup preview error: {e}", "#ff4444"); return

        SetupPreviewDialog(
            image_bgr = self._image,
            recipe    = recipe,
            ct        = self._ctrl._ct,
            parent    = self,
        ).exec_()
        active = sum(
            1 for g in (recipe.get("grid_a", []) + recipe.get("grid_b", []))
            if g is not None
        )
        self._panel.log(f"Setup preview — {active} active slots.", "#88ff88")
        
    # ----------------------------------------------------------
    # Frame template creation — 3-step sequence
    # ----------------------------------------------------------
    def _start_frame(self):
        if self._image is None:
            self._panel.log("No image loaded.", "#ffaa44"); return
        if self._mode is not None:
            self._panel.log("Finish current operation first.", "#ffaa44"); return

        # Validate grid letters — must have at least one non-empty slot
        letters = self._panel.grid_letters()
        non_empty = [l for l in letters if l]
        if not non_empty:
            QtWidgets.QMessageBox.warning(
                self, "Grid Letters Empty",
                "Please fill in at least one letter in the\n"
                "'Grid Letters' field before creating a template.\n\n"
                "Example:  A,B,C,,E,,G,,")
            return

        self._frame_step      = 0
        self._frame_rects     = [None, None, None]
        self._stamp_letters   = non_empty
        self._stamp_shifts_a  = []
        self._stamp_shifts_b  = []
        self._stamp_idx       = 0
        self._stamp_mold      = "A"
        self._mode            = "frame"

        self._clear_frame_overlays()

        self._frame_panel = FrameTemplatePanel(
            on_confirm = self._on_frame_confirm,
            on_cancel  = self._on_frame_cancel,
            on_skip    = self._on_stamp_skip,
            parent     = self,
        )
        geo = self.geometry()
        pw  = self._frame_panel.sizeHint().width()
        self._frame_panel.move(geo.right() - pw - 20, geo.top() + 60)
        self._frame_panel.show()

        self._view.set_draw_mode(True)
        self._panel.log(
            "Frame template — Step 1/5: Draw FRAME rect.", "#00e5ff")
            
    def _on_frame_confirm(self):
        if self._frame_step < 2:
            if self._frame_rects[self._frame_step] is None:
                return
            self._frame_step += 1
            self._frame_panel.set_step(self._frame_step, can_confirm=False)
            self._view.set_draw_mode(True)
            step_name = self._FRAME_TAGS[self._frame_step]
            self._panel.log(
                f"Frame template — Step {self._frame_step+1}/5:"
                f" Draw {step_name} rect.", "#00e5ff")
        elif self._frame_step == 2:
            if self._frame_rects[2] is None:
                return
            # Advance to stamp MOLD A
            self._frame_step  = 3
            self._stamp_mold  = "A"
            self._stamp_idx   = 0
            self._stamp_shifts_a = []
            self._start_stamp_step()

    def _on_frame_cancel(self):
        self._clear_frame_overlays()
        self._view.set_draw_mode(False)
        self._view.set_stamp_mode(False)
        self._mode        = None
        self._frame_rects = [None, None, None]
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None
        self._panel.log("Frame template creation cancelled.", "#888888")

    def _finish_frame(self):
        self._view.set_stamp_mode(False)
        self._view.set_draw_mode(False)
        self._mode = None
        if self._frame_panel:
            self._frame_panel.hide()
            self._frame_panel = None

        letters = self._panel.grid_letters()

        ok = self._ctrl.save_frame_recipe(
            self._image,
            self._frame_rects[0],
            self._frame_rects[1],
            self._frame_rects[2],
            grid_letters = letters,
            grid_a       = self._stamp_shifts_a,
            grid_b       = self._stamp_shifts_b,
        )
        if ok:
            self._panel.log(
                "Frame recipe saved (FRAME + MOLD A + MOLD B + grids).", "#88ff88")
            try:
                recipe = self._ctrl.load_frame_recipe()
                self._panel.refresh_recipe_info(recipe)
                FrameRecipePreviewDialog(recipe, parent=self).exec_()
            except Exception:
                pass
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

    def _start_stamp_step(self):
        """Enter stamp mode for current mold / slot index."""
        mold    = self._stamp_mold
        letters = self._stamp_letters
        idx     = self._stamp_idx
        total   = len(letters)

        if idx >= total:
            # Done with this mold
            if mold == "A":
                self._stamp_mold = "B"
                self._stamp_idx  = 0
                self._stamp_shifts_b = []
                self._frame_step = 4
                self._start_stamp_step()
            else:
                self._finish_frame()
            return

        letter   = letters[idx]
        progress = f"{idx+1}/{total}"
        step_num = 3 if mold == "A" else 4

        self._frame_panel.set_step(step_num, can_confirm=False,
                                slot_label=letter,
                                slot_progress=progress)
        self._panel.log(
            f"Stamp MOLD {mold} slot {idx+1}/{total}: [{letter}]  "
            f"— click on image to place.", "#ffcc00")

        box = self._sm.get("cell_box_size")

        # Show mold centre overlay for reference
        mold_key = "mold_a" if mold == "A" else "mold_b"
        try:
            recipe = self._ctrl.load_frame_recipe() \
                    if self._ctrl.has_frame_recipe() else None
        except Exception:
            recipe = None

        self._view.set_stamp_mode(True, box_size=int(box), label=letter)    

    def _on_stamp_click(self, rect: QtCore.QRect):
        """
        Called when user stamps a slot position.
        rect is centred on the click point (size = cell_box_size).
        Computes:
          dx/dy     — stamp centre offset from mold centre (position reference)
          exp_dx/dy — actual contour bbox centre offset from mold centre
                      (used as expected shift reference during inspection)
        Stores (dx, dy, exp_dx, exp_dy) 4-tuple per slot.
        """
        mold      = self._stamp_mold
        mold_step = 1 if mold == "A" else 2   # index into _frame_rects
        mold_rect = self._frame_rects[mold_step]

        # Mold centre in image space
        mcx = mold_rect.x() + mold_rect.width()  // 2
        mcy = mold_rect.y() + mold_rect.height() // 2

        # Stamp centre in image space
        scx = rect.x() + rect.width()  // 2
        scy = rect.y() + rect.height() // 2

        dx = scx - mcx
        dy = scy - mcy

        # Compute expected contour centre from actual image content
        exp_dx, exp_dy = dx, dy   # fallback = stamp centre
        if self._image is not None:
            ih, iw = self._image.shape[:2]
            box    = int(self._sm.get("cell_box_size"))
            half   = box // 2
            lx1 = max(0,  scx - half)
            ly1 = max(0,  scy - half)
            lx2 = min(iw, lx1 + box)
            ly2 = min(ih, ly1 + box)
            if lx2 > lx1 and ly2 > ly1:
                roi_gray = self._image[ly1:ly2, lx1:lx2] \
                           if self._image.ndim == 2 \
                           else cv2.cvtColor(
                               self._image[ly1:ly2, lx1:lx2],
                               cv2.COLOR_BGR2GRAY)
                cnts, _, _ = ContourTemplate.extract_contours(roi_gray)
                if cnts:
                    all_pts = np.vstack([c.reshape(-1, 2) for c in cnts])
                    bx, by, bw, bh = cv2.boundingRect(all_pts)
                    # Contour bbox centre in image space
                    ccx = lx1 + bx + bw // 2
                    ccy = ly1 + by + bh // 2
                    exp_dx = ccx - mcx
                    exp_dy = ccy - mcy

        letter = self._stamp_letters[self._stamp_idx]
        if mold == "A":
            self._stamp_shifts_a.append((dx, dy, exp_dx, exp_dy))
        else:
            self._stamp_shifts_b.append((dx, dy, exp_dx, exp_dy))

        # Draw stamped overlay
        tag   = f"{mold}{self._stamp_idx}"
        color = QtGui.QColor(255, 200, 0)
        self._view._overlays = [
            ov for ov in self._view._overlays if ov[2] != tag]
        box  = int(self._sm.get("cell_box_size"))
        half = box // 2
        stamp_rect = QtCore.QRect(scx - half, scy - half, box, box)
        self._view.add_overlay(
            stamp_rect, color,
            f"{letter}({dx:+d},{dy:+d}) exp({exp_dx:+d},{exp_dy:+d})",
            "solid")

        self._panel.log(
            f"  Mold {mold} [{letter}] stamp ({dx:+d},{dy:+d})"
            f"  exp_shift ({exp_dx:+d},{exp_dy:+d})",
            "#ffcc00")

        self._stamp_idx += 1
        self._start_stamp_step()

    def _on_stamp_skip(self):
        """Skip current stamp slot."""
        mold   = self._stamp_mold
        letter = self._stamp_letters[self._stamp_idx]
        if mold == "A":
            self._stamp_shifts_a.append(None)
        else:
            self._stamp_shifts_b.append(None)
        self._panel.log(
            f"  Mold {mold} [{letter}] skipped.", "#888888")
        self._stamp_idx += 1
        self._start_stamp_step()
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
            # Stamp steps handled separately
            if self._frame_step >= 3:
                self._on_stamp_click(rect)
                return
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
        ok   = self._ctrl.save_font(
            name, roi, (x, y, w, h), parent_widget=self)
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
        """
        Called when user releases a rubber-band draw during frame mode.
        Stores the rect for the current step, updates overlay, enables Confirm.
        User can re-draw — previous overlay for this step is replaced.
        """
        x = max(0, rect.x())
        y = max(0, rect.y())
        w = min(rect.width(),  self._image.shape[1] - x)
        h = min(rect.height(), self._image.shape[0] - y)
        if w < 4 or h < 4:
            return   # too small — ignore, keep draw mode active

        clipped = QtCore.QRect(x, y, w, h)
        tag     = self._FRAME_TAGS[self._frame_step]

        # Replace overlay for this step (remove old, add new)
        self._view._overlays = [
            ov for ov in self._view._overlays if ov[2] != tag
        ]
        self._view.add_overlay(clipped, QtGui.QColor(0, 224, 255), tag, "dash")

        # Store rect and enable Confirm
        self._frame_rects[self._frame_step] = clipped
        if self._frame_panel:
            self._frame_panel.set_step(self._frame_step, can_confirm=True)

        # Keep draw mode active so user can re-draw if needed
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
        ref = self._image if self._image is not None else None
        ih  = ImageIO.TARGET_H
        iw  = ImageIO.TARGET_W
        m   = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (ih, iw):
            self._panel.log("Mask file missing or size mismatch — ignored.",
                            "#ffaa44")
            return None
        _, bm = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return bm

    def _start_run(self):
        if self._worker and self._worker.isRunning():
            self._panel.log("Run already in progress.", "#ffaa44"); return
        if not self._ctrl.has_frame_recipe():
            self._panel.log("No frame recipe — create it first.", "#ffaa44")
            return
        if not DEBUG_MODE and (self._camera is None or
                               not self._camera.is_open()):
            self._panel.log("Camera not open.", "#ff4444"); return

        # Update camera exposure if changed in UI
        if not DEBUG_MODE and self._camera:
            self._camera.set_exposure(
                self._sm.get("camera_exposure_us"))

        pin_params = self._panel.pin_search_params()
        tm_thr     = self._panel.tm_threshold()
        cell_box   = self._panel.cell_box_size()
        mask       = self._load_run_mask()

        self._worker = RunWorker(
            ctrl       = self._ctrl,
            io         = self._machine_io,
            pin_params = pin_params,
            tm_thr     = tm_thr,
            cell_box   = cell_box,
            mask       = mask,
            image_io   = self._io_obj,
            camera     = self._camera if not DEBUG_MODE else None,
        )
        self._worker.sig_image.connect(self._view.set_image)
        self._worker.sig_result.connect(self._panel.log)
        self._worker.sig_done.connect(self._on_worker_done)
        self._worker.sig_error.connect(self._on_worker_error)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        mode = "DEBUG folder" if DEBUG_MODE else "CAMERA"
        self._panel.log(
            f"=== Start [{mode}]  score_thr={pin_params['score_thr']:.2f}"
            f"  tm_thr={tm_thr:.3f}  box={cell_box}px ===", "#ffffff")
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

    def _save_img(self):
        if self._view.pixmap() is None:
            self._panel.log("Nothing to save.", "#ffaa44"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Result Image",
            f"result_{datetime.now():%Y%m%d_%H%M%S}.png",
            "PNG (*.png);;BMP (*.bmp)")
        if not path:
            return
        self._view.pixmap().save(path)
        self._panel.log(f"Image -> {path}", "#88ff88")


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
    for candidate in [
        "image_source/Image__2026-03-11__13-34-39.bmp",
        "Image__2026-03-11__13-34-39.bmp",
    ]:
        if os.path.exists(candidate):
            try:
                img = io.load(candidate)
                print(f"[Startup] Loaded {candidate}")
            except Exception as e:
                print(f"[Startup] {e}")
            break

    win = MainWindow(img)
    win.show()
    sys.exit(app.exec_())