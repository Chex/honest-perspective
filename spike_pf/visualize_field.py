"""Visualize PF's per-pixel up-vector field on each test image.

Goal: see whether PF's confidence (variance of up-vector across the image)
correlates with how trustworthy the global (roll, pitch) estimate is.

For each image, produces a 3-panel output:
  1. original (downsampled)
  2. up-vector arrows on a coarse grid, colored by deviation from mean
  3. text panel: roll/pitch/vfov + confidence metrics
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from perspective2d import PerspectiveFields

VERSION = "PersNet_Paramnet-GSV-centered"
SAMPLES = Path(__file__).parent.parent / "samples"
OUT_DIR = Path(__file__).parent / "field_viz"
OUT_DIR.mkdir(exist_ok=True)

IMAGES = [
    "IMG_5587.JPG", "IMG_5606.JPG", "IMG_5792.JPG",
    "IMG_5980.jpeg", "IMG_5984.jpeg",
    "IMG_5302.JPG", "IMG_5362.JPG", "IMG_5893.JPG", "IMG_4501.jpeg",
]


def to_scalar(x):
    return float(x.detach().cpu().squeeze()) if torch.is_tensor(x) else float(x)


def field_metrics(gravity, central_crop=0.5):
    """Compute confidence metrics on the up-vector field.

    gravity: tensor (2, H, W) — components (gx, gy) per pixel.
    Returns:
      mean_dir_deg: dominant direction in degrees (0 = down, 90 = right, etc.)
      circ_std_deg: circular standard deviation of direction (degrees)
      central_std_deg: same but only in central crop (where user's subject usually is)
    """
    g = gravity.detach().cpu().numpy()  # (2, H, W)
    H, W = g.shape[1], g.shape[2]

    # Direction angle at each pixel (radians)
    ang = np.arctan2(g[1], g[0])  # (H, W)

    # Circular stats: convert to unit vectors, average, compute resultant length
    def circ_std(angles):
        # circ_std = sqrt(-2 ln R) where R = |mean(unit vectors)|
        c = np.cos(angles).mean()
        s = np.sin(angles).mean()
        R = np.sqrt(c * c + s * s)
        R = max(R, 1e-6)
        return np.degrees(np.sqrt(-2 * np.log(min(R, 1.0))))

    full_std = circ_std(ang.ravel())
    full_mean = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))

    cy0 = int(H * (1 - central_crop) / 2)
    cy1 = int(H - cy0)
    cx0 = int(W * (1 - central_crop) / 2)
    cx1 = int(W - cx0)
    central_ang = ang[cy0:cy1, cx0:cx1]
    central_std_val = circ_std(central_ang.ravel())

    return full_mean, full_std, central_std_val


def overlay_arrows(img_bgr, gravity, grid=24, scale=0.6):
    """Draw up-vector arrows on a coarse grid over the image."""
    H_img, W_img = img_bgr.shape[:2]
    g = gravity.detach().cpu().numpy()  # (2, H_g, W_g)
    H_g, W_g = g.shape[1], g.shape[2]

    out = img_bgr.copy()
    step_y = H_img // grid
    step_x = W_img // grid
    arrow_len = min(step_y, step_x) * scale

    # Compute mean direction for color mapping
    ang_full = np.arctan2(g[1], g[0])
    mean_c = np.cos(ang_full).mean()
    mean_s = np.sin(ang_full).mean()

    for gy in range(grid):
        for gx in range(grid):
            iy = int((gy + 0.5) * step_y)
            ix = int((gx + 0.5) * step_x)
            # Sample gravity field at this point
            sy = int((gy + 0.5) / grid * H_g)
            sx = int((gx + 0.5) / grid * W_g)
            vy = g[1, sy, sx]
            vx = g[0, sy, sx]
            mag = np.sqrt(vx * vx + vy * vy)
            if mag < 0.05:
                continue
            # Deviation from mean direction in degrees
            dev = abs(np.degrees(np.arctan2(vy * mean_c - vx * mean_s,
                                            vx * mean_c + vy * mean_s)))
            # Green if aligned with mean, red if deviates >30°
            t = min(dev / 30.0, 1.0)
            color = (int(80 * (1 - t)), int(220 * (1 - t)), int(60 + 195 * t))
            dx = vx / mag * arrow_len
            dy = vy / mag * arrow_len
            p1 = (ix, iy)
            p2 = (int(ix + dx), int(iy + dy))
            cv2.arrowedLine(out, p1, p2, color, 2, tipLength=0.3)
    return out


def text_panel(w, h, lines):
    panel = np.full((h, w, 3), 30, dtype=np.uint8)
    for i, (line, color) in enumerate(lines):
        cv2.putText(panel, line, (16, 38 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return panel


def stack_horizontally(images, target_h=720, gap=10):
    parts = []
    sep = np.full((target_h, gap, 3), 40, dtype=np.uint8)
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        scale = target_h / h
        parts.append(cv2.resize(im, (int(w * scale), target_h)))
        if i < len(images) - 1:
            parts.append(sep)
    return np.hstack(parts)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = PerspectiveFields(VERSION).to(device)
    model.train(False)

    summary = []
    for name in IMAGES:
        p = SAMPLES / name
        img = cv2.imread(str(p))
        with torch.inference_mode():
            pred = model.inference(img_bgr=img)

        roll = to_scalar(pred["pred_roll"])
        pitch = to_scalar(pred["pred_pitch"])
        vfov = to_scalar(pred["pred_vfov"])
        mean_dir, full_std, central_std = field_metrics(pred["pred_gravity"])

        # Verdict heuristic: low central_std → confident; high → ambiguous
        if central_std < 15:
            verdict, vcolor = "CONFIDENT", (60, 220, 80)
        elif central_std < 30:
            verdict, vcolor = "OK", (60, 200, 220)
        else:
            verdict, vcolor = "AMBIGUOUS", (60, 80, 240)

        arrows = overlay_arrows(img, pred["pred_gravity"])
        info = text_panel(720, 720, [
            (f"{name}", (240, 240, 240)),
            (f"roll  = {roll:+.2f} deg", (200, 200, 200)),
            (f"pitch = {pitch:+.2f} deg", (200, 200, 200)),
            (f"vfov  = {vfov:.1f} deg", (200, 200, 200)),
            ("", (0, 0, 0)),
            (f"field mean dir: {mean_dir:+.1f} deg", (180, 180, 240)),
            (f"  (0=down, ~90=right)", (140, 140, 180)),
            (f"field std (all):     {full_std:.1f} deg", (180, 180, 240)),
            (f"field std (central): {central_std:.1f} deg", (180, 180, 240)),
            ("", (0, 0, 0)),
            (f"verdict: {verdict}", vcolor),
            ("(green arrows = agree", (160, 160, 160)),
            (" with mean direction;", (160, 160, 160)),
            (" red = disagree)", (160, 160, 160)),
        ])

        side = stack_horizontally([arrows, info])
        out_path = OUT_DIR / f"field_{p.stem}.jpg"
        cv2.imwrite(str(out_path), side, [cv2.IMWRITE_JPEG_QUALITY, 92])
        summary.append((name, roll, pitch, vfov, full_std, central_std, verdict))
        print(f"{name:<18} roll={roll:+6.2f} pitch={pitch:+6.2f} "
              f"std_all={full_std:5.1f} std_central={central_std:5.1f}  {verdict}")

    print("\n--- summary ---")
    print(f"{'image':<18} {'roll':>7} {'pitch':>7} {'vfov':>6} "
          f"{'std_all':>8} {'std_ctr':>8}  verdict")
    for r in summary:
        n, ro, pi, vf, sa, sc, v = r
        print(f"{n:<18} {ro:>+6.2f}  {pi:>+6.2f}  {vf:>5.1f}  "
              f"{sa:>7.1f}  {sc:>7.1f}  {v}")


if __name__ == "__main__":
    main()
