"""Dump per-image Orientation tag + mapped g_img to JSON for cross-check.

Run from perspective_fix/ root with its venv (has fix.py + PIL).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import fix  # noqa: E402

IMAGES = [
    "IMG_5587.JPG", "IMG_5606.JPG", "IMG_5792.JPG",
    "IMG_5980.jpeg", "IMG_5984.jpeg",
    "IMG_5302.JPG", "IMG_5362.JPG", "IMG_5893.JPG", "IMG_4501.jpeg",
]


def get_image_gravity(acc):
    Ax, Ay, Az = acc
    if abs(Ax) > abs(Ay):
        if Ax < 0:
            Ix, Iy, Iz = Ay, -Ax, -Az
        else:
            Ix, Iy, Iz = -Ay, Ax, -Az
        layout = "landscape"
    else:
        if Ay < 0:
            Ix, Iy, Iz = Ax, -Ay, -Az
        else:
            Ix, Iy, Iz = -Ax, Ay, -Az
        layout = "portrait"
    v = np.array([Ix, Iy, Iz], dtype=np.float64)
    return v / np.linalg.norm(v), layout


def main():
    out = []
    print(f"{'image':<18} {'Orient':>6} {'layout':>10} {'|Ax|':>6} {'|Ay|':>6} {'|Az|':>6}  g_img")
    print("-" * 100)
    for name in IMAGES:
        path = ROOT / "samples" / name
        _, _, exif = fix.load_image(str(path))
        info = fix.apple_acceleration_from_exif(exif)
        if info is None:
            print(f"{name:<18}  no gravity info")
            continue
        acc = info["vector"]
        orient = info["orientation"]
        g_img, layout = get_image_gravity(acc)
        print(f"{name:<18} {orient:>6} {layout:>10} "
              f"{abs(acc[0]):>6.3f} {abs(acc[1]):>6.3f} {abs(acc[2]):>6.3f}  "
              f"[{g_img[0]:+.3f}, {g_img[1]:+.3f}, {g_img[2]:+.3f}]")
        out.append({
            "name": name,
            "orientation": orient,
            "acc": list(acc),
            "g_img": g_img.tolist(),
            "layout": layout,
        })

    json_path = Path(__file__).parent / "gravity.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
