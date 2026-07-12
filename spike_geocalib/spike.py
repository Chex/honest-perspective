"""Compare GeoCalib intrinsics/gravity with EXIF and the current pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from geocalib import GeoCalib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fix  # noqa: E402


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _scalar(value):
    array = _array(value).reshape(-1)
    return float(array[0]) if len(array) else None


def _field_summary(value):
    array = _array(value).astype(np.float64)
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_result(result, source_scale):
    camera = _array(result["camera"]).reshape(-1)
    gravity = _array(result["gravity"]).reshape(-1, 3)[0]
    summary = {
        "camera_vector": camera.tolist(),
        "focal_px": float(camera[2] / source_scale),
        "gravity": gravity.tolist(),
        "covariance": _array(result["covariance"]).tolist(),
    }
    for key, value in result.items():
        if "uncertainty" in key:
            number = _scalar(value)
            if key == "focal_uncertainty":
                number /= source_scale
            summary[key] = number
        elif "confidence" in key:
            summary[key] = _field_summary(value)
    return summary


def apple_gravity_ray(gravity, intrinsics, width, height):
    if gravity is None:
        return None
    vp = fix._gravity_vertical_vp(
        gravity["vector"], gravity["orientation"], width, height,
        intrinsics=intrinsics,
    )
    return fix._vp_ray(vp, width, height, intrinsics=intrinsics) if vp is not None else None


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    images = args.images or [ROOT / "samples" / "IMG_5984.jpeg"]
    device = choose_device(args.device)
    started = time.perf_counter()
    model = GeoCalib().to(device).eval()
    load_seconds = time.perf_counter() - started
    rows = []

    for path in images:
        bgr, _icc, exif = fix.load_image(path)
        height, width = bgr.shape[:2]
        exif_intrinsics = fix.camera_intrinsics_from_exif(exif, width, height)
        apple_gravity = fix.apple_acceleration_from_exif(exif)
        image = model.load_image(path).to(device)
        source_scale = min(1.0, 768.0 / max(width, height))
        if source_scale < 1.0:
            resized = [round(height * source_scale), round(width * source_scale)]
            image = F.interpolate(
                image[None], size=resized, mode="bilinear", align_corners=False,
                antialias=True,
            )[0]
        torch.mps.synchronize() if device.type == "mps" else None
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.calibrate(image)
        torch.mps.synchronize() if device.type == "mps" else None
        elapsed = time.perf_counter() - started
        geocalib = summarize_result(result, source_scale)
        apple_ray = apple_gravity_ray(apple_gravity, exif_intrinsics, width, height)
        geocalib_ray = np.asarray(geocalib["gravity"], dtype=np.float64)
        gravity_angle = None
        if apple_ray is not None:
            dot = abs(float(np.dot(apple_ray, geocalib_ray) /
                            (np.linalg.norm(apple_ray) * np.linalg.norm(geocalib_ray))))
            gravity_angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        rows.append({
            "image": path.name,
            "size": [width, height],
            "device": str(device),
            "model_load_seconds": load_seconds,
            "inference_seconds": elapsed,
            "exif_intrinsics": exif_intrinsics,
            "apple_gravity": apple_gravity,
            "gravity_angle_deg": gravity_angle,
            "focal_error_percent": (
                100.0 * (geocalib["focal_px"] - exif_intrinsics["fx"])
                / exif_intrinsics["fx"]
            ),
            "geocalib": geocalib,
        })

    payload = {"rows": rows}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
