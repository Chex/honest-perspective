"""Shared physical camera model for preview, export, and auto correction."""

from __future__ import annotations

import math

import cv2
import numpy as np


FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
STATE_VERSION = 1
CROP_POLICY = "conservative_same_aspect_v1"


def make_intrinsics(width, height, focal_35mm=None):
    """Return centered square-pixel intrinsics, preferring EXIF 35mm focal length."""
    width = float(width)
    height = float(height)
    if focal_35mm is not None and np.isfinite(focal_35mm) and focal_35mm > 0:
        focal_px = float(focal_35mm) * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
        source = "exif_35mm"
    else:
        focal_px = max(width, height)
        source = "fallback_max_dimension"
    return {
        "fx": focal_px,
        "fy": focal_px,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "source": source,
        "focal_35mm": float(focal_35mm) if focal_35mm is not None else None,
    }


def intrinsics_matrix(intrinsics):
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_to_rotation(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation contains non-finite values")
    u, _singular, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_deg(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_from_homography(homography, intrinsics):
    k = intrinsics_matrix(intrinsics)
    candidate = np.linalg.inv(k) @ np.asarray(homography, dtype=np.float64) @ k
    scale = np.cbrt(abs(float(np.linalg.det(candidate))))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("homography is singular")
    return project_to_rotation(candidate / scale)


def interpolate_rotation(rotation, strength):
    """Interpolate identity -> rotation on SO(3), never in homography space."""
    strength = float(strength)
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    rotation = project_to_rotation(rotation)
    rotvec, _ = cv2.Rodrigues(rotation)
    interpolated, _ = cv2.Rodrigues(rotvec * strength)
    return project_to_rotation(interpolated)


def rotation_from_vector(rotation_vector):
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector contains non-finite values")
    rotation, _ = cv2.Rodrigues(vector)
    return project_to_rotation(rotation)


def compose_rotation(rotation_vector, rotation):
    return project_to_rotation(rotation_from_vector(rotation_vector) @ rotation)


def _apply_homography(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    denominators = projected[:, 2]
    if np.any(np.abs(denominators) < 1e-9):
        raise ValueError("projection crosses infinity")
    return projected[:, :2] / denominators[:, None], denominators


def _fit_aspect_float(crop, target_aspect):
    x0, y0, x1, y1 = [float(value) for value in crop]
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("crop is empty")
    if width / height > target_aspect:
        new_width = height * target_aspect
        center = (x0 + x1) / 2.0
        x0, x1 = center - new_width / 2.0, center + new_width / 2.0
    else:
        new_height = width / target_aspect
        center = (y0 + y1) / 2.0
        y0, y1 = center - new_height / 2.0, center + new_height / 2.0
    return [x0, y0, x1, y1]


def conservative_crop(transformed_corners, target_aspect):
    tl, tr, br, bl = np.asarray(transformed_corners, dtype=np.float64).reshape(4, 2)
    crop = [
        max(tl[0], bl[0]),
        max(tl[1], tr[1]),
        min(tr[0], br[0]),
        min(br[1], bl[1]),
    ]
    return _fit_aspect_float(crop, float(target_aspect))


def _crop_inside_quad(crop, quad, tolerance=1e-5):
    x0, y0, x1, y1 = crop
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    contour = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(contour, point, False) >= -tolerance for point in corners)


def compute_view(rotation, intrinsics, image_size, crop=None):
    """Build a fixed-size warp and a crop containing only source pixels."""
    width, height = [float(value) for value in image_size]
    if width <= 1 or height <= 1:
        raise ValueError("image_size must be positive")
    rotation = project_to_rotation(rotation)
    k = intrinsics_matrix(intrinsics)
    homography = k @ rotation @ np.linalg.inv(k)
    image_corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    transformed, denominators = _apply_homography(homography, image_corners)
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("projection flips across the camera plane")
    w_ratio = float(np.min(np.abs(denominators)) / np.max(np.abs(denominators)))

    canonical_crop = conservative_crop(transformed, width / height)
    if crop is None:
        crop = canonical_crop
    else:
        crop = [float(value) for value in crop]
        if len(crop) != 4 or not np.all(np.isfinite(crop)):
            raise ValueError("crop must contain four finite values")
        crop = _fit_aspect_float(crop, width / height)
        if not _crop_inside_quad(crop, transformed):
            raise ValueError("crop extends outside transformed source pixels")

    x0, y0, x1, y1 = crop
    crop_width = x1 - x0
    crop_height = y1 - y0
    sx = width / crop_width
    sy = height / crop_height
    if abs(sx - sy) > 1e-6 * max(1.0, sx, sy):
        raise ValueError("crop would require non-uniform scaling")
    view = np.array(
        [[sx, 0.0, -sx * x0], [0.0, sy, -sy * y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ) @ homography
    return {
        "matrix": view,
        "homography": homography,
        "crop": crop,
        "canonical_crop": canonical_crop,
        "transformed_corners": transformed,
        "theta_deg": rotation_angle_deg(rotation),
        "projective_w_ratio": w_ratio,
    }


def make_correction_state(rotation, intrinsics, image_size, crop=None):
    rotation = project_to_rotation(rotation)
    view = compute_view(rotation, intrinsics, image_size, crop=crop)
    return {
        "version": STATE_VERSION,
        "rotation": rotation.reshape(-1).tolist(),
        "intrinsics": dict(intrinsics),
        "crop": list(view["crop"]),
        "crop_policy": CROP_POLICY,
    }


def validate_correction_state(state, image_size, max_rotation_deg=45.0,
                              min_projective_w_ratio=0.22):
    try:
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise ValueError("unsupported correction state version")
        rotation_input = np.asarray(state.get("rotation"), dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(rotation_input)):
            raise ValueError("rotation contains non-finite values")
        orthogonality_error = float(np.linalg.norm(rotation_input.T @ rotation_input - np.eye(3)))
        determinant = float(np.linalg.det(rotation_input))
        if orthogonality_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
            raise ValueError("rotation is not in SO(3)")
        rotation = project_to_rotation(rotation_input)
        theta_deg = rotation_angle_deg(rotation)
        if theta_deg > float(max_rotation_deg):
            raise ValueError("rotation exceeds the safety envelope")
        intrinsics = state.get("intrinsics")
        if not isinstance(intrinsics, dict):
            raise ValueError("intrinsics are missing")
        view = compute_view(rotation, intrinsics, image_size, crop=state.get("crop"))
        if view["projective_w_ratio"] < float(min_projective_w_ratio):
            raise ValueError("projection is too close to degeneracy")
        return {
            "accepted": True,
            "reason": None,
            "theta_deg": theta_deg,
            "orthogonality_error": orthogonality_error,
            "determinant": determinant,
            "projective_w_ratio": view["projective_w_ratio"],
            "view": view,
        }
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            "accepted": False,
            "reason": str(error),
            "theta_deg": None,
            "orthogonality_error": None,
            "determinant": None,
            "projective_w_ratio": None,
            "view": None,
        }


def warp_with_state(image, state, max_rotation_deg=45.0):
    height, width = image.shape[:2]
    validation = validate_correction_state(
        state,
        [width, height],
        max_rotation_deg=max_rotation_deg,
    )
    if not validation["accepted"]:
        raise ValueError(validation["reason"])
    matrix = validation["view"]["matrix"]
    return cv2.warpPerspective(image, matrix, (width, height))
