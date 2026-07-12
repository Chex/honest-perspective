"""Apply PF (roll, pitch, vfov) -> H rectification, build side-by-side comparison.

Layout per row: current RANSAC output (or freshly computed) | PF rectified.
Original dropped per user request (recompression hurts comparison quality).
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from perspective2d import PerspectiveFields

VERSION = "Paramnet-360Cities-edina-centered"
PROJECT = Path(__file__).parent.parent
SAMPLES = PROJECT / "samples"
OUT_DIR = Path(__file__).parent / "comparison"
RANSAC_CACHE = OUT_DIR / "ransac_cache"
OUT_DIR.mkdir(exist_ok=True)
RANSAC_CACHE.mkdir(exist_ok=True)

# (src, optional cached RANSAC reference filename in samples/)
PAIRS = [
    ("IMG_5587.JPG", "IMG_5587_out.jpg"),
    ("IMG_5606.JPG", "IMG_5606_out.jpg"),
    ("IMG_5792.JPG", "IMG_5792_out.jpg"),
    ("IMG_5980.jpeg", "IMG_5980_fixed.jpg"),
    ("IMG_5984.jpeg", "IMG_5984_fixed.jpg"),
    # New challenge images — no pre-saved RANSAC reference, will be generated.
    ("IMG_5302.JPG", None),
    ("IMG_5362.JPG", None),
    ("IMG_5893.JPG", None),
    ("IMG_4501.jpeg", None),
]


def get_ransac_output(src_name):
    """Return a BGR image of perspective_fix's RANSAC output for this src, or None."""
    src_path = SAMPLES / src_name
    cached = RANSAC_CACHE / f"{src_path.stem}_ransac.jpg"
    if cached.exists():
        return cv2.imread(str(cached))
    print(f"  computing RANSAC for {src_name} ...")
    try:
        subprocess.run(
            ["uv", "run", "python", "fix.py", str(src_path), str(cached), "--mode", "both"],
            cwd=str(PROJECT), check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        print(f"  RANSAC failed: {e.stderr[:200]}")
        return None
    return cv2.imread(str(cached)) if cached.exists() else None


def to_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().squeeze())
    return float(x)


def rectify_homography(roll_deg, pitch_deg, vfov_deg, w, h):
    """Compute H = K @ R.T @ K^-1. PF convention: R = R_z(roll) @ R_x(pitch)."""
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    f = h / (2 * np.tan(np.radians(vfov_deg) / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_z = np.array([
        [np.cos(roll), np.sin(roll), 0],
        [-np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    R = R_z @ R_x
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv
    return H


def warp_with_autosize(img, H):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x_min, y_min = warped_corners.min(axis=0)
    x_max, y_max = warped_corners.max(axis=0)
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))
    out_w = min(out_w, w * 2)
    out_h = min(out_h, h * 2)
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_final = T @ H
    return cv2.warpPerspective(img, H_final, (out_w, out_h), borderValue=(20, 20, 20))


def stack_horizontally(images, target_h=900, gap=12):
    resized = []
    for img in images:
        if img is None:
            placeholder = np.full((target_h, target_h, 3), 60, dtype=np.uint8)
            cv2.putText(placeholder, "N/A", (target_h // 3, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            resized.append(placeholder)
            continue
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))
    sep = np.full((target_h, gap, 3), 40, dtype=np.uint8)
    parts = []
    for i, im in enumerate(resized):
        parts.append(im)
        if i < len(resized) - 1:
            parts.append(sep)
    return np.hstack(parts)


def annotate(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return out


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = PerspectiveFields(VERSION).to(device)
    model.train(False)

    for src_name, ransac_name in PAIRS:
        src_path = SAMPLES / src_name
        img = cv2.imread(str(src_path))
        h, w = img.shape[:2]

        with torch.inference_mode():
            pred = model.inference(img_bgr=img)
        roll = to_scalar(pred["pred_roll"])
        pitch = to_scalar(pred["pred_pitch"])
        vfov = to_scalar(pred["pred_vfov"])

        H = rectify_homography(roll, pitch, vfov, w, h)
        rectified = warp_with_autosize(img, H)

        # RANSAC reference: prefer pre-saved file in samples/, otherwise generate
        ransac_img = None
        ransac_label = "RANSAC"
        if ransac_name:
            rpath = SAMPLES / ransac_name
            if rpath.exists():
                ransac_img = cv2.imread(str(rpath))
                ransac_label = f"RANSAC ({ransac_name})"
        if ransac_img is None:
            ransac_img = get_ransac_output(src_name)
            ransac_label = "RANSAC (auto-generated, mode=both)"

        side_by_side = stack_horizontally([
            annotate(ransac_img, ransac_label) if ransac_img is not None else None,
            annotate(rectified, f"PF  roll={roll:+.2f} pitch={pitch:+.2f} vfov={vfov:.1f}"),
        ])

        out_path = OUT_DIR / f"compare_edina_{src_path.stem}.jpg"
        cv2.imwrite(str(out_path), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(f"wrote {out_path.name}  PF: roll={roll:+.2f}deg pitch={pitch:+.2f}deg vfov={vfov:.1f}deg")


if __name__ == "__main__":
    main()
