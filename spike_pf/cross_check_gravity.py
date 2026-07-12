"""Cross-check EXIF-derived gravity against PF-derived gravity.

PF outputs (roll, pitch). Convert to image-frame gravity unit vector and
compare to gravity.json from spike_gravity/dump_gravity.py.

Coordinate convention: image y axis points down. For an upright camera
(roll=0, pitch=0), gravity in camera frame = (0, 1, 0).

For PF's R = R_z(roll) @ R_x(pitch), world up = (0, -1, 0)_world expressed
in camera frame is R @ (0, -1, 0). Gravity (down) is the negative of that:
    g_cam = (sin(roll)*cos(pitch),
             cos(roll)*cos(pitch),
            -sin(pitch))
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from perspective2d import PerspectiveFields

VERSION = "PersNet_Paramnet-GSV-centered"  # the better-behaved checkpoint
SPIKE_PF = Path(__file__).parent
SAMPLES = SPIKE_PF.parent / "samples"
GRAVITY_JSON = SPIKE_PF.parent / "spike_gravity" / "gravity.json"


def to_scalar(x):
    return float(x.detach().cpu().squeeze()) if torch.is_tensor(x) else float(x)


def pf_to_g_img(roll_deg, pitch_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    return np.array([
        np.sin(r) * np.cos(p),
        np.cos(r) * np.cos(p),
        -np.sin(p),
    ])


def angle_deg(a, b):
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def main():
    entries = json.loads(GRAVITY_JSON.read_text())
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = PerspectiveFields(VERSION).to(device)
    model.train(False)

    print(f"{'image':<18} {'roll':>7} {'pitch':>7}  "
          f"{'g_exif':>26}  {'g_pf':>26}  {'diff':>6}")
    print("-" * 110)
    rows = []
    for e in entries:
        name = e["name"]
        g_exif = np.array(e["g_img"])
        img = cv2.imread(str(SAMPLES / name))
        with torch.inference_mode():
            pred = model.inference(img_bgr=img)
        roll = to_scalar(pred["pred_roll"])
        pitch = to_scalar(pred["pred_pitch"])
        g_pf = pf_to_g_img(roll, pitch)
        diff = angle_deg(g_exif, g_pf)
        rows.append((name, roll, pitch, g_exif, g_pf, diff))
        print(f"{name:<18} {roll:>+6.2f}  {pitch:>+6.2f}  "
              f"[{g_exif[0]:+.2f},{g_exif[1]:+.2f},{g_exif[2]:+.2f}]  "
              f"[{g_pf[0]:+.2f},{g_pf[1]:+.2f},{g_pf[2]:+.2f}]  "
              f"{diff:>5.2f}°")


if __name__ == "__main__":
    main()
