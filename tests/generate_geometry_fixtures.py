"""Generate the shared geometry-contract fixtures for Python / JS / Swift.

The Python implementation in geometry.py is the reference; this script
records its outputs for randomized inputs so every other implementation
(webapp/geometry.js, Sources/HonestGeometry) can be checked against the
same JSON file instead of shelling out to each other.

Regenerate after any intentional contract change:

    python tests/generate_geometry_fixtures.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry import (  # noqa: E402
    compose_rotation,
    compute_view,
    interpolate_rotation,
    make_intrinsics,
    rotation_from_vector,
)

FIXTURES_PATH = ROOT / "tests" / "HonestGeometryTests" / "Fixtures" / "geometry_contract.json"
SEED = 20260710


def _intrinsics_values(intrinsics):
    return {key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")}


def build_fixtures():
    rng = np.random.default_rng(SEED)
    fixtures = {
        "version": 1,
        "seed": SEED,
        "generator": "tests/generate_geometry_fixtures.py",
        "makeIntrinsics": [],
        "computeView": [],
        "drag": [],
        "interpolate": [],
    }

    for width, height, focal in [
        (5712, 4284, 26.0),
        (4032, 3024, 24.0),
        (800, 600, None),
        (600, 800, 52.0),
    ]:
        intrinsics = make_intrinsics(width, height, focal_35mm=focal)
        expected = _intrinsics_values(intrinsics)
        expected["source"] = intrinsics["source"]
        expected["focal35mm"] = intrinsics["focal_35mm"]
        fixtures["makeIntrinsics"].append({
            "width": width,
            "height": height,
            "focal35mm": focal,
            "expected": expected,
        })

    sizes = [(800, 600), (4032, 3024), (3024, 4032)]
    for index in range(24):
        width, height = sizes[index % len(sizes)]
        intrinsics = make_intrinsics(
            width, height, focal_35mm=float(rng.choice([24, 26, 35, 52]))
        )
        vector = np.radians(rng.uniform(-12, 12, size=3))
        rotation = rotation_from_vector(vector)
        crop = None
        if index % 4 == 3:
            # A custom crop strictly inside the canonical one (10% shrink
            # toward its center keeps it inside the transformed quad by
            # convexity, and preserves the aspect ratio).
            x0, y0, x1, y1 = compute_view(rotation, intrinsics, [width, height])["canonical_crop"]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            crop = [
                cx + (x0 - cx) * 0.9,
                cy + (y0 - cy) * 0.9,
                cx + (x1 - cx) * 0.9,
                cy + (y1 - cy) * 0.9,
            ]
        view = compute_view(rotation, intrinsics, [width, height], crop=crop)
        case = {
            "rotation": rotation.reshape(-1).tolist(),
            "intrinsics": _intrinsics_values(intrinsics),
            "imageSize": [width, height],
            "expected": {
                "matrix": view["matrix"].reshape(-1).tolist(),
                "homography": view["homography"].reshape(-1).tolist(),
                "crop": [float(value) for value in view["crop"]],
                "canonicalCrop": [float(value) for value in view["canonical_crop"]],
                "transformedCorners": np.asarray(view["transformed_corners"]).tolist(),
                "thetaDeg": float(view["theta_deg"]),
                "projectiveWRatio": float(view["projective_w_ratio"]),
            },
        }
        if crop is not None:
            case["crop"] = [float(value) for value in crop]
        fixtures["computeView"].append(case)

    drag_intrinsics = {"fx": 600.925, "fy": 600.925, "cx": 400.0, "cy": 300.0}
    starts = [np.eye(3)] + [
        rotation_from_vector(np.radians(rng.uniform(-10, 10, size=3))) for _ in range(2)
    ]
    for start in starts:
        for dx, dy in [(-120, 0), (0, 90), (-120, 90), (37.5, -64.25)]:
            pitch = -math.atan(dy / drag_intrinsics["fy"])
            yaw = math.atan(dx / drag_intrinsics["fx"])
            expected = compose_rotation(np.array([pitch, yaw, 0.0]), start)
            fixtures["drag"].append({
                "startRotation": start.reshape(-1).tolist(),
                "displacement": [dx, dy],
                "intrinsics": dict(drag_intrinsics),
                "expectedRotation": expected.reshape(-1).tolist(),
            })

    for _ in range(6):
        vector = np.radians(rng.uniform(-12, 12, size=3))
        rotation = rotation_from_vector(vector)
        for strength in (0.0, 0.35, 0.7, 1.0):
            expected = interpolate_rotation(rotation, strength)
            fixtures["interpolate"].append({
                "rotation": rotation.reshape(-1).tolist(),
                "strength": strength,
                "expectedRotation": expected.reshape(-1).tolist(),
            })

    return fixtures


def main():
    fixtures = build_fixtures()
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURES_PATH.open("w") as handle:
        json.dump(fixtures, handle, indent=1)
        handle.write("\n")
    print(f"wrote {FIXTURES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
