"""Perspective Fields spike on perspective_fix sample images.

Goal: get (roll, pitch, vfov) for each sample image, see if values match the
camera tilt humans perceive.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from perspective2d import PerspectiveFields

VERSION = "PersNet_Paramnet-GSV-centered"
SAMPLES = Path(__file__).parent.parent / "samples"
IMAGES = [
    "IMG_5587.JPG",
    "IMG_5606.JPG",
    "IMG_5792.JPG",
    "IMG_5980.jpeg",
    "IMG_5984.jpeg",
]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().squeeze())
    return float(x)


def main():
    device = pick_device()
    print(f"device: {device}\nmodel: {VERSION}\n")

    t = time.time()
    model = PerspectiveFields(VERSION).to(device)
    model.eval()
    print(f"load: {time.time()-t:.2f}s\n")

    print(f"{'image':<18} {'roll':>8} {'pitch':>8} {'vfov':>8} {'f@h':>8} {'time':>6}")
    print("-" * 64)
    for name in IMAGES:
        p = SAMPLES / name
        img_bgr = cv2.imread(str(p))
        h, w = img_bgr.shape[:2]
        t = time.time()
        with torch.inference_mode():
            pred = model.inference(img_bgr=img_bgr)
        dt = time.time() - t

        roll = to_scalar(pred["pred_roll"])
        pitch = to_scalar(pred["pred_pitch"])
        vfov = to_scalar(pred["pred_vfov"])
        f_px = h / (2 * np.tan(np.radians(vfov) / 2))
        print(f"{name:<18} {roll:>7.2f}° {pitch:>7.2f}° {vfov:>7.2f}° {f_px:>8.0f} {dt:>5.2f}s")


if __name__ == "__main__":
    main()
