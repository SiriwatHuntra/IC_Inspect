"""
SVM OCR Training Pipeline
==========================
Trains a 38-class SVM (0-9, A-Z, dash, space) on 48x48 grayscale crops.

Dataset layout expected:
    dataset/
        train/
            0/   *.png / *.jpg
            A/   ...
            dash/
            space/
        test/
            (same structure — used by svm_test.py only)

Output:
    svm_model.onnx          — ONNX model (skl2onnx)
    svm_model.xml           — OpenVINO IR (via openvino ovc)
    svm_model.bin
    label_map.json          — index -> display label, e.g. {0: "0", 36: "-", 37: " "}

Usage:
    python svm_train.py --data dataset/train
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_SIZE     = 48          # resize target (square)
MOLD_SIZE    = 150         # passed to _thresh_font for kernel anchoring

# Folder name -> display label mapping
FOLDER_LABEL_MAP = {
    "dash":  "-",
    "space": " ",
}

# ──────────────────────────────────────────────
# THRESHOLD BLOCK  (copied from ContourTemplate)
# Same pipeline as runtime inspection — must stay in sync.
# ──────────────────────────────────────────────

def _thresh_font(gray: np.ndarray, mold_size: int = MOLD_SIZE) -> np.ndarray:
    """
    Identical to ContourTemplate._thresh_font.
    Keep both in sync when algorithm changes.
    """
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    k_size = max(9, (mold_size // 8) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w   = binary.shape[:2]
    margin = max(3, mold_size // 50)
    mask   = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    binary = cv2.bitwise_and(binary, mask)
    return binary


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────

def _center_crop(gray: np.ndarray) -> np.ndarray:
    """
    Find largest contour bounding box, center it on IMG_SIZE canvas, stretch to fill.
    Falls back to direct resize if no contour found.
    """
    binary = _thresh_font(gray)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        # Largest contour bounding box
        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Clamp to image bounds
        ih, iw = gray.shape
        x  = max(0, x)
        y  = max(0, y)
        w  = min(w, iw - x)
        h  = min(h, ih - y)

        if w > 0 and h > 0:
            crop = gray[y:y + h, x:x + w]
            return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Fallback — no contour found
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def preprocess(img_path: str) -> np.ndarray:
    """
    Load image → grayscale → center largest contour → 48x48 → flatten float32.
    Returns 1-D feature vector of length IMG_SIZE*IMG_SIZE.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load: {img_path}")
    centered = _center_crop(img)
    normed   = centered.astype(np.float32) / 255.0
    return normed.flatten()


# ──────────────────────────────────────────────
# DATASET LOADER
# ──────────────────────────────────────────────

def load_dataset(root: str):
    """
    Walk root/<class_folder>/*.png|jpg
    Returns X (N, 2304), y (N,), label_map {int -> str}, classes [str]
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Gather all valid class folders, sort for deterministic index
    raw_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    if not raw_folders:
        raise ValueError(f"No class subfolders found in {root}")

    # Build label list (display labels, sorted)
    folder_to_label = {}
    for folder in raw_folders:
        name = folder.name
        folder_to_label[name] = FOLDER_LABEL_MAP.get(name, name.upper())

    # Sort by display label for consistent indexing: 0-9 A-Z - space
    sorted_labels = sorted(
        set(folder_to_label.values()),
        key=lambda s: (
            0 if s.isdigit() else
            1 if s.isalpha() else
            2
        )
    )
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    X, y = [], []
    skipped = 0

    for folder in raw_folders:
        display_label = folder_to_label[folder.name]
        class_idx     = label_to_idx[display_label]

        img_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        ]

        for p in img_paths:
            try:
                feat = preprocess(str(p))
                X.append(feat)
                y.append(class_idx)
            except Exception as e:
                print(f"  [SKIP] {p.name}: {e}")
                skipped += 1

    if skipped:
        print(f"  Skipped {skipped} files due to load errors.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"  Loaded {len(X)} samples across {len(sorted_labels)} classes.")
    for lbl in sorted_labels:
        cnt = int(np.sum(y == label_to_idx[lbl]))
        print(f"    [{label_to_idx[lbl]:2d}] '{lbl}'  -> {cnt} samples")

    return X, y, idx_to_label, sorted_labels


# ──────────────────────────────────────────────
# ONNX EXPORT  (skl2onnx)
# ──────────────────────────────────────────────

def export_onnx(model, n_features: int, out_path: str):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[ONNX] skl2onnx not installed. Run: pip install skl2onnx")
        return False

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model   = convert_sklearn(model, initial_types=initial_type)

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[ONNX] Saved -> {out_path}")
    return True


# ──────────────────────────────────────────────
# OPENVINO CONVERT  (ovc CLI)
# ──────────────────────────────────────────────

def convert_openvino(onnx_path: str, out_dir: str):
    import subprocess
    out_name = Path(onnx_path).stem
    cmd = [
        sys.executable, "-m", "openvino.tools.ovc",
        onnx_path,
        "--output_model", os.path.join(out_dir, out_name),
    ]
    print(f"[OV] Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OV] IR saved -> {out_dir}/{out_name}.xml / .bin")
        return True
    else:
        print(f"[OV] Conversion failed:\n{result.stderr}")
        print("     Install: pip install openvino")
        return False


# ──────────────────────────────────────────────
# MAIN TRAIN
# ──────────────────────────────────────────────

def train(data_root: str, out_dir: str = "."):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────
    print(f"\n[Data] Loading from: {data_root}")
    X, y, idx_to_label, classes = load_dataset(data_root)

    # ── Train ─────────────────────────────────
    print(f"\n[Train] Fitting SVM (rbf kernel) on {len(X)} samples ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       decision_function_shape="ovr", probability=False)),
    ])
    pipe.fit(X, y)

    y_pred = pipe.predict(X)
    acc    = accuracy_score(y, y_pred)
    print(f"[Train] Training accuracy: {acc*100:.2f}%")

    # ── Save label map ────────────────────────
    label_map_path = os.path.join(out_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)
    print(f"[Label] Saved -> {label_map_path}")

    # ── ONNX export ───────────────────────────
    onnx_path = os.path.join(out_dir, "svm_model.onnx")
    ok = export_onnx(pipe, X.shape[1], onnx_path)

    # ── OpenVINO IR convert ───────────────────
    if ok:
        convert_openvino(onnx_path, out_dir)

    print("\n[Done] Training complete.")
    return pipe, idx_to_label


# ──────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM OCR Training Pipeline")
    parser.add_argument("--data",    default="dataset/train",
                        help="Path to training dataset root (default: dataset/train)")
    parser.add_argument("--out",     default=".",
                        help="Output directory for model files (default: current dir)")
    args = parser.parse_args()

    train(data_root=args.data, out_dir=args.out)