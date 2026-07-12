"""DA3 monocular pose spike on perspective_fix sample images.

Goals:
  1. Confirm DA3-Small loads + runs on Mac (MPS or CPU) without xformers.
  2. Get camera extrinsics/intrinsics for a single image.
  3. Compare DA3's R against perspective_fix's current solvePnP-derived R.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "da3_repo" / "src"))

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

REPO = "depth-anything/DA3-Small"
SAMPLES = Path(__file__).parent.parent / "samples"
TEST_IMG = SAMPLES / "IMG_5587.JPG"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = pick_device()
    print(f"device: {device}")

    t = time.time()
    print(f"loading {REPO} …")
    model = DepthAnything3.from_pretrained(REPO)
    model = model.to(device=device).eval()
    print(f"  load: {time.time()-t:.2f}s")

    t = time.time()
    with torch.inference_mode():
        pred = model.inference([str(TEST_IMG)])
    print(f"  inference: {time.time()-t:.2f}s")

    print("--- prediction ---")
    print(f"depth shape: {pred.depth.shape}, dtype: {pred.depth.dtype}")
    print(f"extrinsics shape: {pred.extrinsics.shape}")
    print(f"intrinsics shape: {pred.intrinsics.shape}")
    print(f"extrinsics[0]:\n{pred.extrinsics[0]}")
    print(f"intrinsics[0]:\n{pred.intrinsics[0]}")

    # Rotation matrix R (camera-to-world is typically R.T of extrinsics[:3,:3])
    R = pred.extrinsics[0, :3, :3]
    print(f"\nR:\n{R}")
    print(f"det(R) = {np.linalg.det(R):.4f} (should be ~1)")

    # World-up direction in camera frame = R @ [0,0,1] if z is world-up,
    # OR R @ [0,1,0] if y is world-up. DA3 uses OpenCV convention so let's
    # extract the gravity direction = world's down (or up) projected to camera.
    print("\ncolumn vectors of R (camera frame axes in world frame):")
    print(f"  R[:,0] = {R[:,0]}")
    print(f"  R[:,1] = {R[:,1]}")
    print(f"  R[:,2] = {R[:,2]}")


if __name__ == "__main__":
    main()
