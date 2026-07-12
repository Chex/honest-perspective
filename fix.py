"""
Perspective fix.

CLI:
    python fix.py input.jpg output.jpg [--strength 1.0] [--mode vertical|horizontal|both]

Library:
    from perspective_fix.fix import auto_correct, warp_with_corners, load_image, save_image
"""
import sys
import argparse
import struct
import numpy as np
import cv2
from PIL import ExifTags, Image

from geometry import (
    compute_view,
    interpolate_rotation,
    intrinsics_matrix,
    make_correction_state,
    make_intrinsics,
    rotation_from_homography,
    validate_correction_state,
    warp_with_state,
)

# Cap on the recovered camera rotation in the manual/web path.
#
# EXIF-derived intrinsics make this a real axis-angle bound. Frontend and
# backend intentionally share 35 degrees so a server candidate never opens in
# a state that the manual interaction cannot edit.
MAX_CAMERA_ROT_DEG = 35.0

# Apple MakerNote tag exposed by exiftool as "AccelerationVector".
# This is not standard EXIF, so every parser has to be defensive.
APPLE_ACCELERATION_TAG = 8
GRAVITY_NORM_ACCEPT_MIN = 0.75
GRAVITY_NORM_ACCEPT_MAX = 1.25
GRAVITY_NORM_TRUST_TOL = 0.10
BOTH_MODE_AREA_TOLERANCE = 0.95
SOURCE_STRAIGHT_HORIZONTAL_DEG = 0.8
SOURCE_STRAIGHT_VERTICAL_DEG = 1.5
STRAIGHTNESS_IMPROVEMENT_DEG = 0.25
MIN_VP_WEIGHTED_INLIER_RATIO = 0.50
MAX_GRAVITY_VISUAL_AUTO_ANGLE_DEG = 8.0

# Optional HEIC support — iPhone "High Efficiency" format. Web app expects this; CLI
# works without it for JPG/PNG.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def load_image(path):
    """Read via Pillow to keep ICC profile and EXIF; return (bgr_array, icc, exif)."""
    pil = Image.open(path)
    icc = pil.info.get("icc_profile")
    exif = pil.info.get("exif")
    # Keep stored pixels as-is; current gravity mapping only trusts Orientation=1.
    pil = pil.convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, icc, exif


def save_image(path, bgr, icc=None, exif=None, quality=95):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    save_kwargs = {"quality": quality}
    if icc:
        save_kwargs["icc_profile"] = icc
    if exif:
        save_kwargs["exif"] = exif
    pil.save(path, **save_kwargs)


def _load_exif(exif_bytes):
    if not exif_bytes:
        return None
    try:
        exif = Image.Exif()
        exif.load(exif_bytes)
        return exif
    except Exception:
        return None


def camera_intrinsics_from_exif(exif_bytes, width, height):
    """Build K from EXIF 35mm-equivalent focal length, with a safe fallback."""
    exif = _load_exif(exif_bytes)
    focal_35mm = None
    if exif is not None:
        focal_35mm = exif.get(41989)
        if focal_35mm is None:
            try:
                focal_35mm = exif.get_ifd(ExifTags.IFD.Exif).get(41989)
            except Exception:
                try:
                    focal_35mm = exif.get_ifd(34665).get(41989)
                except Exception:
                    focal_35mm = None
    try:
        focal_35mm = float(focal_35mm) if focal_35mm is not None else None
    except (TypeError, ValueError):
        focal_35mm = None
    return make_intrinsics(width, height, focal_35mm=focal_35mm)


def apple_acceleration_from_exif(exif_bytes):
    """
    Return Apple's gravity-ish acceleration vector from EXIF bytes, if present.

    iPhone stores this in Apple MakerNote tag 8 as three signed rationals.
    Pillow exposes the MakerNote as opaque bytes, so we parse only the tiny
    Apple IFD subset we need. Return (x, y, z) plus orientation, or None.
    """
    exif = _load_exif(exif_bytes)
    if exif is None:
        return None
    if exif.get(271) != "Apple":
        return None
    orientation = int(exif.get(274, 1) or 1)
    try:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        try:
            exif_ifd = exif.get_ifd(34665)
        except Exception:
            return None
    maker = exif_ifd.get(37500)
    if not isinstance(maker, (bytes, bytearray)):
        return None
    maker = bytes(maker)
    if not maker.startswith(b"Apple iOS\x00\x00\x01"):
        return None
    # Apple MakerNote starts with a 14-byte header, then a big-endian IFD.
    if len(maker) < 16:
        return None
    try:
        count = struct.unpack(">H", maker[14:16])[0]
        for i in range(count):
            off = 16 + i * 12
            if off + 12 > len(maker):
                break
            tag, typ, n, value = struct.unpack(">HHII", maker[off:off + 12])
            if tag != APPLE_ACCELERATION_TAG:
                continue
            # TIFF type 10 = signed rational; Apple stores x/y/z here.
            if typ != 10 or n != 3 or value + 24 > len(maker):
                return None
            vals = []
            for j in range(3):
                num, den = struct.unpack(">ii", maker[value + j * 8:value + j * 8 + 8])
                if den == 0:
                    return None
                vals.append(num / den)
            norm = float(np.linalg.norm(vals))
            if not GRAVITY_NORM_ACCEPT_MIN <= norm <= GRAVITY_NORM_ACCEPT_MAX:
                return None
            norm_deviation = abs(norm - 1.0)
            return {
                "vector": tuple(vals),
                "orientation": orientation,
                "norm": norm,
                "norm_deviation": norm_deviation,
                "trusted": norm_deviation <= GRAVITY_NORM_TRUST_TOL,
            }
    except Exception:
        return None
    return None


def _gravity_vertical_vp(acceleration, orientation, w, h, intrinsics=None):
    """
    Convert Apple acceleration vector to a vertical vanishing point.

    iOS stores many photos with pixels already rotated upright and EXIF
    Orientation reset to 1, while Apple's AccelerationVector remains in the
    physical device coordinate system. Infer the capture layout from the
    dominant acceleration axis, then map device gravity into image coordinates.
    """
    if orientation != 1 or acceleration is None:
        return None
    Ax, Ay, Az = acceleration
    if abs(Ax) > abs(Ay):
        if Ax < 0:
            Ix, Iy, Iz = Ay, -Ax, -Az
        else:
            Ix, Iy, Iz = -Ay, Ax, -Az
    else:
        if Ay < 0:
            Ix, Iy, Iz = Ax, -Ay, -Az
        else:
            Ix, Iy, Iz = -Ax, Ay, -Az

    norm = float(np.linalg.norm([Ix, Iy, Iz]))
    if norm < 1e-6:
        return None
    gx, gy, gz = Ix / norm, Iy / norm, Iz / norm
    intrinsics = intrinsics or make_intrinsics(w, h)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    denom = gz
    if abs(denom) < 1e-4:
        denom = 1e-4 if denom >= 0 else -1e-4
    return np.array([
        (fx * gx + cx * gz) / denom,
        (fy * gy + cy * gz) / denom,
        1.0,
    ], dtype=np.float64)


def _vp_ray(vp, w, h, intrinsics=None):
    Kinv = np.linalg.inv(intrinsics_matrix(intrinsics or make_intrinsics(w, h)))
    ray = Kinv @ np.array([vp[0], vp[1], 1.0])
    return ray / np.linalg.norm(ray)


def _vp_angle_deg(a, b, w, h, intrinsics=None):
    ra = _vp_ray(a, w, h, intrinsics=intrinsics)
    rb = _vp_ray(b, w, h, intrinsics=intrinsics)
    dot = abs(float(np.clip(np.dot(ra, rb), -1.0, 1.0)))
    return float(np.degrees(np.arccos(dot)))


def _gravity_is_trusted(gravity, gravity_mode):
    if gravity is None:
        return False
    if gravity_mode == "force":
        return True
    return gravity_mode == "auto" and bool(gravity.get("trusted"))


def fit_aspect(x0, y0, x1, y1, target_aspect, max_w, max_h):
    """Center-crop an axis-aligned rect to match target_aspect (= w/h)."""
    w = x1 - x0
    h = y1 - y0
    cur = w / h
    if cur > target_aspect:
        new_w = h * target_aspect
        cx = (x0 + x1) / 2
        x0 = max(0, int(round(cx - new_w / 2)))
        x1 = min(max_w, int(round(cx + new_w / 2)))
    else:
        new_h = w / target_aspect
        cy = (y0 + y1) / 2
        y0 = max(0, int(round(cy - new_h / 2)))
        y1 = min(max_h, int(round(cy + new_h / 2)))
    return x0, y0, x1, y1


def detect_line_segments(gray):
    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return np.empty((0, 4))
    return lines.reshape(-1, 4)


def line_to_homogeneous(p1, p2):
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    return l


def cluster_by_angle(segs, vertical_tol_deg=25, horizontal_tol_deg=25):
    verticals, horizontals = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        a = abs(angle)
        if a > 90:
            a = 180 - a
        if a > 90 - vertical_tol_deg:
            verticals.append((x1, y1, x2, y2))
        elif a < horizontal_tol_deg:
            horizontals.append((x1, y1, x2, y2))
    return verticals, horizontals


def ransac_vanishing_point(segs, iters=2000, inlier_thresh=2.0):
    if len(segs) < 2:
        return None
    lines = [line_to_homogeneous((x1, y1), (x2, y2)) for x1, y1, x2, y2 in segs]
    best_vp, best_inliers = None, -1
    rng = np.random.default_rng(0)
    n = len(lines)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-9:
            continue
        vp = vp / vp[2]
        inliers = 0
        for (x1, y1, x2, y2), l in zip(segs, lines):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < 1:
                continue
            vx, vy = vp[0] - mx, vp[1] - my
            vl = np.hypot(vx, vy)
            if vl < 1:
                continue
            # angular error between segment direction and (midpoint -> vp)
            cross = abs(dx * vy - dy * vx) / (length * vl)
            err = cross * length  # perpendicular distance from segment to vp ray
            if err < inlier_thresh:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_vp = vp
    return best_vp


def _vanishing_point_support(segs, vp, inlier_thresh=2.0):
    if vp is None or not segs:
        return None
    total_weight = 0.0
    inlier_weight = 0.0
    inlier_count = 0
    errors = []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        vx, vy = vp[0] - mx, vp[1] - my
        ray_length = float(np.hypot(vx, vy))
        if length < 1.0 or ray_length < 1.0:
            continue
        error = abs(dx * vy - dy * vx) / ray_length
        total_weight += length
        errors.append(error)
        if error < inlier_thresh:
            inlier_count += 1
            inlier_weight += length
    return {
        "line_count": len(segs),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_count / len(segs),
        "weighted_inlier_ratio": inlier_weight / total_weight if total_weight else 0.0,
        "median_error_px": float(np.median(errors)) if errors else None,
    }


def homography_from_single_vp(vp, w, h, axis, intrinsics=None):
    """
    Build a homography that sends a single vanishing point to infinity along
    `axis` ('vertical' or 'horizontal'). Only corrects convergence along that axis.
    """
    K = intrinsics_matrix(intrinsics or make_intrinsics(w, h))
    Kinv = np.linalg.inv(K)
    v = Kinv @ np.array([vp[0], vp[1], 1.0])
    v = v / np.linalg.norm(v)
    if axis == "vertical":
        target = np.array([0.0, -1.0, 0.0]) if v[1] < 0 else np.array([0.0, 1.0, 0.0])
    elif axis == "horizontal":
        target = np.array([-1.0, 0.0, 0.0]) if v[0] < 0 else np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"axis must be 'vertical' or 'horizontal', got {axis!r}")
    # Rotation axis = cross(v, target), normalized
    axis = np.cross(v, target)
    s = np.linalg.norm(axis)
    c = float(np.clip(v.dot(target), -1.0, 1.0))
    if s < 1e-8:
        R = np.eye(3)
    else:
        axis = axis / s
        K_cross = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K_cross + (1 - c) * (K_cross @ K_cross)
    # R maps v -> target; the image homography needs the inverse direction,
    # so we use R (not R.T) here: H = K @ R_cam.T @ Kinv with R_cam = R.T.
    return K @ R @ Kinv


def homography_from_vps(vp_v, vp_h, w, h, intrinsics=None):
    """
    Build a homography that sends vp_v to (cx, -inf) and vp_h to (+inf, cy),
    i.e. makes vertical lines truly vertical and horizontal lines truly horizontal.
    Constructed as H = inv(K) where K maps an undistorted image back to the photo.
    Practical approach: use the two finite vanishing points to recover the rotation
    that levels the image, then convert to a homography.
    """
    K = intrinsics_matrix(intrinsics or make_intrinsics(w, h))
    Kinv = np.linalg.inv(K)

    def ray(vp):
        v = Kinv @ np.array([vp[0], vp[1], 1.0])
        return v / np.linalg.norm(v)

    rh0 = ray(vp_h)  # x-axis in world, sign ambiguous
    rv0 = ray(vp_v)  # y-axis in world, sign ambiguous

    # A vanishing point gives an axis, not a directed axis. Try the four sign
    # choices and keep the one closest to the current camera orientation. This
    # avoids 180° branch flips on near-level images such as IMG_5984.
    candidates = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            rh = sx * rh0
            rv = sy * rv0
            # orthogonalize: keep rv as up, project rh onto plane perpendicular to rv
            rh = rh - rh.dot(rv) * rv
            n = np.linalg.norm(rh)
            if n < 1e-9:
                continue
            rh = rh / n
            rz = np.cross(rh, rv)
            R = np.column_stack([rh, rv, rz])
            # Ensure R is a proper rotation
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            theta = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
            candidates.append((theta, R))
    if not candidates:
        return np.eye(3)
    _theta, R = min(candidates, key=lambda x: x[0])
    H = K @ R.T @ Kinv
    return H


def blend_with_identity(H, strength, intrinsics):
    """Compatibility wrapper that now interpolates on SO(3)."""
    rotation = rotation_from_homography(H, intrinsics)
    rotation = interpolate_rotation(rotation, strength)
    k = intrinsics_matrix(intrinsics)
    return k @ rotation @ np.linalg.inv(k)


def warp_and_get_quad(img, H):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    pts = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)
    # Translate so all coords are positive
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    H2 = T @ H
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    warped = cv2.warpPerspective(img, H2, (new_w, new_h))
    quad = (pts - [min_x, min_y]).astype(np.float64)
    return warped, quad


def largest_inscribed_rect(quad, img_shape, downsample=4):
    """
    Largest axis-aligned rectangle inside the quadrilateral.
    Approach: rasterize quad as binary mask (downsampled for speed), then run
    the classic "largest rectangle in binary matrix" histogram algorithm.
    """
    h, w = img_shape[:2]
    ds = downsample
    small_h, small_w = h // ds, w // ds
    mask = np.zeros((small_h, small_w), dtype=np.uint8)
    poly = (quad / ds).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 1)

    # Largest rectangle of 1s using histogram method.
    heights = np.zeros(small_w, dtype=np.int32)
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1
    for row in range(small_h):
        heights = (heights + 1) * mask[row]
        stack = []
        for col in range(small_w + 1):
            cur_h = heights[col] if col < small_w else 0
            start = col
            while stack and stack[-1][1] > cur_h:
                s_col, s_h = stack.pop()
                area = s_h * (col - s_col)
                if area > best[0]:
                    best = (area, s_col, row - s_h + 1, col, row + 1)
                start = s_col
            stack.append((start, cur_h))
    _, x0, y0, x1, y1 = best
    return int(x0 * ds), int(y0 * ds), int(x1 * ds), int(y1 * ds)


def _detect_and_cluster(gray, w, h):
    """LSD → length filter → angle clustering, with a single adaptive
    re-pass when the first pass produces too few segments for RANSAC.

    The default 3% length filter (171px on 5712-wide iPhone shots) is
    well-tuned for buildings — long unbroken edges. But brick-wall /
    tile-wall scenes have lots of strong VISUAL structure made of short
    segments (mortar joints ~13px on a typical brick photo). 3% wipes
    those out (21,745 → 4 on IMG_5984). If either cluster ends up
    starved for RANSAC, re-cluster at 1.5%.
    """
    segs = detect_line_segments(gray)
    if not len(segs):
        return [], []
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    short_side = min(w, h)
    keep = segs[lengths > short_side * 0.03]
    verticals, horizontals = cluster_by_angle(keep)
    MIN_FOR_RANSAC = 5
    if min(len(verticals), len(horizontals)) < MIN_FOR_RANSAC:
        keep = segs[lengths > short_side * 0.015]
        verticals, horizontals = cluster_by_angle(keep)
    return verticals, horizontals


def _line_straightness_quality(bgr, min_len_ratio=0.01):
    """
    Lightweight "already straight?" signal for auto mode selection.

    This intentionally measures only local line orientation. It is not a VP
    solver; it answers the product question "would auto-correction make the
    visible horizontal/vertical structure more level than the source?"
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    segs = detect_line_segments(gray)
    if not len(segs):
        return {"score_deg": None, "horizontal_deg": None, "vertical_deg": None,
                "horizontal_count": 0, "vertical_count": 0}
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[lengths > min(w, h) * min_len_ratio]
    h_devs = []
    v_devs = []
    for x1, y1, x2, y2 in segs:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        if abs(angle) < 25:
            h_devs.append(float(angle))
        if abs(abs(angle) - 90) < 25:
            v_devs.append(float(angle - 90 if angle > 0 else angle + 90))
    h_deg = abs(float(np.median(h_devs))) if h_devs else None
    v_deg = abs(float(np.median(v_devs))) if v_devs else None
    vals = [v for v in (h_deg, v_deg) if v is not None]
    return {
        "score_deg": max(vals) if vals else None,
        "horizontal_deg": h_deg,
        "vertical_deg": v_deg,
        "horizontal_count": len(h_devs),
        "vertical_count": len(v_devs),
    }


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values) or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _segment_alignment(segs, homography, axis):
    if not segs:
        return None
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    points = segs.reshape(-1, 2)
    transformed = cv2.perspectiveTransform(
        points.reshape(1, -1, 2), np.asarray(homography, dtype=np.float64)
    ).reshape(-1, 4)
    deviations = []
    weights = []
    for original, corrected in zip(segs, transformed):
        x1, y1, x2, y2 = corrected
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = ((angle + 90.0) % 180.0) - 90.0
        deviation = abs(angle) if axis == "horizontal" else abs(90.0 - abs(angle))
        length = float(np.hypot(original[2] - original[0], original[3] - original[1]))
        if np.isfinite(deviation) and length > 0:
            deviations.append(deviation)
            weights.append(length)
    return _weighted_median(deviations, weights)


def _line_alignment_quality(verticals, horizontals, homography, mode):
    identity = np.eye(3, dtype=np.float64)
    before_v = _segment_alignment(verticals, identity, "vertical")
    before_h = _segment_alignment(horizontals, identity, "horizontal")
    after_v = _segment_alignment(verticals, homography, "vertical")
    after_h = _segment_alignment(horizontals, homography, "horizontal")
    axes = ("vertical", "horizontal") if mode == "both" else (mode,)

    def score(vertical, horizontal):
        lookup = {"vertical": vertical, "horizontal": horizontal}
        values = [lookup[axis] for axis in axes if lookup[axis] is not None]
        return max(values) if values else None

    before = score(before_v, before_h)
    after = score(after_v, after_h)
    improvement = before - after if before is not None and after is not None else None
    return {
        "before_deg": before,
        "after_deg": after,
        "improvement_deg": improvement,
        "before_vertical_deg": before_v,
        "before_horizontal_deg": before_h,
        "after_vertical_deg": after_v,
        "after_horizontal_deg": after_h,
        "vertical_count": len(verticals),
        "horizontal_count": len(horizontals),
    }


def _identity_corners(w, h):
    return [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]


def _compute_homography(bgr, mode, gravity=None, gravity_mode="auto", intrinsics=None,
                        line_evidence=None, vp_evidence=None):
    """
    Run line detection → VP estimation → build H.

    Each mode is STRICT about its requirements:
      - "vertical": needs vp_v, else returns None
      - "horizontal": needs vp_h, else returns None
      - "both": needs BOTH vp_v and vp_h, else returns None

    No silent fallback. The old "both → single-axis fallback" path was a
    leftover from when auto_correct_pick_best ran the algorithm once;
    nowadays pick_best runs all three modes independently, so a "both"
    that secretly degrades to horizontal is redundant with the horizontal
    candidate already in the list — and worse, it lies about what
    actually ran. When a user explicitly picks "Full correction" from the
    menu, they mean both-axes-corrected, not "any correction that fits."

    Returns (H, mode, meta) on success or (None, None, meta) on failure.
    """
    h, w = bgr.shape[:2]
    intrinsics = intrinsics or make_intrinsics(w, h)
    if line_evidence is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        verticals, horizontals = _detect_and_cluster(gray, w, h)
    else:
        verticals, horizontals = line_evidence
    if vp_evidence is None:
        vp_v = ransac_vanishing_point(verticals) if mode in ("vertical", "both") else None
        vp_h = ransac_vanishing_point(horizontals) if mode in ("horizontal", "both") else None
    else:
        vp_v = vp_evidence.get("vertical") if mode in ("vertical", "both") else None
        vp_h = vp_evidence.get("horizontal") if mode in ("horizontal", "both") else None
    gravity_vp = None
    gravity_angle = None
    gravity_used = False
    if gravity is not None and gravity_mode != "off":
        gravity_vp = _gravity_vertical_vp(
            gravity.get("vector"), gravity.get("orientation", 1), w, h,
            intrinsics=intrinsics,
        )
        if gravity_vp is not None and vp_v is not None:
            gravity_angle = _vp_angle_deg(
                gravity_vp, vp_v, w, h, intrinsics=intrinsics
            )

    meta = {
        "gravity_available": gravity_vp is not None,
        "gravity_trusted": bool(gravity.get("trusted")) if gravity else False,
        "gravity_norm": gravity.get("norm") if gravity else None,
        "gravity_norm_deviation": gravity.get("norm_deviation") if gravity else None,
        "gravity_used": False,
        "gravity_visual_angle_deg": gravity_angle,
        "vertical_lines": len(verticals),
        "horizontal_lines": len(horizontals),
        "vertical_source": None,
    }

    def choose_vertical_vp():
        trust_gravity = gravity_vp is not None and _gravity_is_trusted(gravity, gravity_mode)
        if not trust_gravity:
            return vp_v, False, "visual" if vp_v is not None else None
        if vp_v is None or gravity_angle is None:
            return gravity_vp, True, "gravity"
        support = _vanishing_point_support(verticals, vp_v) or {}
        if (
            gravity_angle <= MAX_GRAVITY_VISUAL_AUTO_ANGLE_DEG
            and support.get("weighted_inlier_ratio", 0.0) >= MIN_VP_WEIGHTED_INLIER_RATIO
        ):
            gravity_h = homography_from_single_vp(
                gravity_vp, w, h, "vertical", intrinsics=intrinsics
            )
            visual_h = homography_from_single_vp(
                vp_v, w, h, "vertical", intrinsics=intrinsics
            )
            gravity_quality = _line_alignment_quality(
                verticals, horizontals, gravity_h, "vertical"
            )["after_deg"]
            visual_quality = _line_alignment_quality(
                verticals, horizontals, visual_h, "vertical"
            )["after_deg"]
            if (
                visual_quality is not None
                and gravity_quality is not None
                and visual_quality + STRAIGHTNESS_IMPROVEMENT_DEG < gravity_quality
            ):
                return vp_v, False, "visual"
        return gravity_vp, True, "gravity"

    if mode == "vertical":
        vertical_vp, gravity_used, vertical_source = choose_vertical_vp()
        if vertical_vp is None:
            return None, None, meta
        meta["gravity_used"] = gravity_used
        meta["vertical_source"] = vertical_source
        return homography_from_single_vp(
            vertical_vp, w, h, "vertical", intrinsics=intrinsics
        ), "vertical", meta
    if mode == "horizontal":
        if vp_h is None:
            return None, None, meta
        return homography_from_single_vp(
            vp_h, w, h, "horizontal", intrinsics=intrinsics
        ), "horizontal", meta
    if mode == "both":
        vertical_vp, gravity_used, vertical_source = choose_vertical_vp()

        if vertical_vp is None or vp_h is None:
            return None, None, meta
        meta["gravity_used"] = gravity_used
        meta["vertical_source"] = vertical_source
        return homography_from_vps(
            vertical_vp, vp_h, w, h, intrinsics=intrinsics
        ), "both", meta
    return None, None, meta


def auto_correct(bgr, mode="vertical", strength=1.0, keep_aspect=True, gravity=None,
                 gravity_mode="auto", intrinsics=None, line_evidence=None,
                 vp_evidence=None):
    """
    Full auto pipeline. Returns (cropped_bgr, source_corners_or_None, actual_mode).

    source_corners is a list of 4 [x, y] points in SOURCE image pixel space,
    clockwise from top-left, representing the quadrilateral that maps onto
    the output rectangle. It remains part of the legacy corners API and the
    web frontend's compatibility fallback; canonical manual adjustment uses
    the correction state instead. None when detection failed.

    actual_mode is what was applied: "vertical" / "horizontal" / "both", or
    None when the requested mode lacks the required evidence.
    """
    h, w = bgr.shape[:2]
    intrinsics = intrinsics or make_intrinsics(w, h)
    H, actual_mode, meta = _compute_homography(
        bgr, mode, gravity=gravity, gravity_mode=gravity_mode,
        intrinsics=intrinsics, line_evidence=line_evidence, vp_evidence=vp_evidence,
    )
    if H is None:
        return bgr, None, None, meta
    rotation = interpolate_rotation(rotation_from_homography(H, intrinsics), strength)
    try:
        state = make_correction_state(rotation, intrinsics, [w, h])
    except ValueError as error:
        meta = dict(meta)
        meta["reason"] = f"invalid state: {error}"
        return bgr, None, None, meta
    physics = validate_correction_state(state, [w, h], max_rotation_deg=MAX_CAMERA_ROT_DEG)
    meta = dict(meta)
    if line_evidence is not None:
        verticals, horizontals = line_evidence
        meta["alignment"] = _line_alignment_quality(
            verticals,
            horizontals,
            compute_view(rotation, intrinsics, [w, h])["homography"],
            actual_mode,
        )
    meta["physics"] = {key: value for key, value in physics.items() if key != "view"}
    meta["state"] = state
    if not physics["accepted"]:
        meta["reason"] = physics["reason"] or "beyond safe rotation range"
        return bgr, None, None, meta
    cropped = warp_with_state(bgr, state, max_rotation_deg=MAX_CAMERA_ROT_DEG)
    view_inv = np.linalg.inv(physics["view"]["matrix"])
    output_corners = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64
    )
    source_corners = cv2.perspectiveTransform(
        output_corners.reshape(1, -1, 2), view_inv
    ).reshape(-1, 2).tolist()
    return cropped, source_corners, actual_mode, meta


def auto_correct_all_modes(bgr, strength=1.0, keep_aspect=True, gravity=None,
                           gravity_mode="auto", intrinsics=None):
    """
    Run all three modes and return per-mode results. This is the primitive
    behind both pick_best (machine choice) and the manual menu (user choice).

    Returns dict keyed by mode name → {
        "cropped":    np.ndarray | None,   # warped+cropped image bytes
        "corners":    list[[x,y]*4] | None,
        "area_ratio": float | None,        # output area / source area
        "reason":     str | None,          # human-facing rejection reason
    }

    `reason` values:
      None                 → viable, use as-is
      "no vanishing point" → required vanishing point(s) not found
      "over-corrected"     → output area > 105% source (wrong VP on noisy lines)
      "crops too much"     → output area < 50% source (extreme foreshortening)
    """
    source_area = bgr.shape[0] * bgr.shape[1]
    h, w = bgr.shape[:2]
    intrinsics = intrinsics or make_intrinsics(w, h)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    line_evidence = _detect_and_cluster(gray, w, h)
    verticals, horizontals = line_evidence
    vp_evidence = {
        "vertical": ransac_vanishing_point(verticals),
        "horizontal": ransac_vanishing_point(horizontals),
    }
    vp_quality = {
        axis: _vanishing_point_support(
            verticals if axis == "vertical" else horizontals,
            vp,
        )
        for axis, vp in vp_evidence.items()
    }
    identity_state = make_correction_state(np.eye(3), intrinsics, [w, h])
    results = {
        "none": {
            "cropped": bgr,
            "corners": _identity_corners(w, h),
            "area_ratio": 1.0,
            "reason": None,
            "meta": {
                "quality": None,
                "identity": True,
                "state": identity_state,
                "physics": {
                    "accepted": True,
                    "reason": None,
                    "theta_deg": 0.0,
                    "orthogonality_error": 0.0,
                    "determinant": 1.0,
                    "projective_w_ratio": 1.0,
                },
            },
        }
    }
    for m in ("both", "vertical", "horizontal"):
        cropped, corners, _actual_mode, meta = auto_correct(
            bgr, mode=m, strength=strength, keep_aspect=keep_aspect, gravity=gravity,
            gravity_mode=gravity_mode, intrinsics=intrinsics,
            line_evidence=line_evidence, vp_evidence=vp_evidence,
        )
        meta = dict(meta)
        meta["vp_quality"] = vp_quality
        if corners is None:
            results[m] = {
                "cropped": None, "corners": None,
                "area_ratio": None,
                "reason": meta.get("reason") or "no vanishing point",
                "meta": meta,
            }
            continue
        area_ratio = abs(cv2.contourArea(np.asarray(corners, dtype=np.float32))) / source_area
        reason = None
        if area_ratio > 1.05:
            reason = "over-corrected"
        elif area_ratio < 0.50:
            reason = "crops too much"
        alignment = meta.get("alignment") or {}
        meta["quality"] = {
            "score_deg": alignment.get("after_deg"),
            "source_score_deg": alignment.get("before_deg"),
        }
        results[m] = {
            "cropped": cropped, "corners": corners,
            "area_ratio": area_ratio, "reason": reason, "meta": meta,
        }
    return results


def auto_correct_pick_best(bgr, strength=1.0, keep_aspect=True, gravity=None,
                           gravity_mode="auto", intrinsics=None):
    """
    Try each mode and return whichever preserves the most output pixels.
    Thin wrapper over auto_correct_all_modes — same selection rule
    (max area among viable candidates), shared per-mode runs.

    Returns (cropped_bgr, source_corners_or_None, chosen_mode_or_None).
    """
    results = auto_correct_all_modes(
        bgr, strength=strength, keep_aspect=keep_aspect, gravity=gravity,
        gravity_mode=gravity_mode, intrinsics=intrinsics,
    )
    viable = [(m, r) for m, r in results.items()
              if r["corners"] is not None and r["reason"] is None]
    if not viable:
        return bgr, None, None
    best_mode, best_r = choose_auto_mode(results)
    return best_r["cropped"], best_r["corners"], best_mode


def choose_auto_mode(results):
    """
    Pick the auto winner from auto_correct_all_modes() output.

    Area still matters, but "both" is the product goal when it is viable: a
    small crop sacrifice is worth real horizontal+vertical correction.
    """
    viable = [(m, r) for m, r in results.items()
              if r["corners"] is not None and r["reason"] is None]
    if not viable:
        return None, None
    source = results.get("none")
    corrected_viable = [(m, r) for m, r in viable if m != "none"]
    if not corrected_viable:
        return "none", source
    auto_eligible = []
    for mode, result in corrected_viable:
        meta = result.get("meta", {})
        alignment = meta.get("alignment") or {}
        improvement = alignment.get("improvement_deg")
        # No visual evidence is different from evidence of no improvement:
        # gravity/model-only candidates may still be useful in line-poor scenes.
        if improvement is not None and improvement < STRAIGHTNESS_IMPROVEMENT_DEG:
            continue
        vp_quality = meta.get("vp_quality") or {}
        vertical_quality = vp_quality.get("vertical") or {}
        horizontal_quality = vp_quality.get("horizontal") or {}
        vertical_confident = (
            vertical_quality.get("weighted_inlier_ratio", 0.0)
            >= MIN_VP_WEIGHTED_INLIER_RATIO
        )
        horizontal_confident = (
            horizontal_quality.get("weighted_inlier_ratio", 0.0)
            >= MIN_VP_WEIGHTED_INLIER_RATIO
        )
        if mode in ("horizontal", "both") and not horizontal_confident:
            continue
        if mode in ("vertical", "both") and not meta.get("gravity_used") and not vertical_confident:
            continue
        gravity_angle = meta.get("gravity_visual_angle_deg")
        if (
            meta.get("gravity_used")
            and vertical_confident
            and gravity_angle is not None
            and gravity_angle > MAX_GRAVITY_VISUAL_AUTO_ANGLE_DEG
        ):
            continue
        auto_eligible.append((mode, result))
    if not auto_eligible:
        return "none", source
    best_mode, best_r = max(auto_eligible, key=lambda x: x[1]["area_ratio"])
    both = results.get("both")
    if (
        both
        and any(mode == "both" for mode, _result in auto_eligible)
        and both["corners"] is not None
        and both["reason"] is None
        and both["area_ratio"] >= best_r["area_ratio"] * BOTH_MODE_AREA_TOLERANCE
    ):
        return "both", both
    return best_mode, best_r


def _adj_flat(m):
    """3×3 adjugate, flat row-major 9-list in and out. Verbatim port of the
    frontend's adj() (index.html). Pure-Python so float64 ops happen in the
    same order as JS — keeps backend↔frontend bit-equivalence on the same
    inputs."""
    return [
        m[4]*m[8] - m[5]*m[7], m[2]*m[7] - m[1]*m[8], m[1]*m[5] - m[2]*m[4],
        m[5]*m[6] - m[3]*m[8], m[0]*m[8] - m[2]*m[6], m[2]*m[3] - m[0]*m[5],
        m[3]*m[7] - m[4]*m[6], m[1]*m[6] - m[0]*m[7], m[0]*m[4] - m[1]*m[3],
    ]


def _multmm_flat(a, b):
    """3×3 × 3×3 in flat row-major. Matches JS multmm()."""
    c = [0.0] * 9
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += a[3*i + k] * b[3*k + j]
            c[3*i + j] = s
    return c


def _multmv_flat(m, v):
    """3×3 × 3-vector in flat. Matches JS multmv()."""
    return [
        m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
        m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
        m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
    ]


def _basis_to_points_flat(x1, y1, x2, y2, x3, y3, x4, y4):
    """Matches JS basisToPoints()."""
    m = [x1, x2, x3, y1, y2, y3, 1.0, 1.0, 1.0]
    v = _multmv_flat(_adj_flat(m), [x4, y4, 1.0])
    return _multmm_flat(m, [v[0], 0.0, 0.0, 0.0, v[1], 0.0, 0.0, 0.0, v[2]])


def _projective_3x3(src, dst):
    """4-point homography src→dst, bit-identical to JS projective3x3For().

    src, dst: sequence of 4 (x, y) pairs in clockwise-from-top-left order.
    Replaces cv2.getPerspectiveTransform in the manual/web path so that
    /validate's frontend↔backend rectify_error delta is exactly 0 instead
    of drifting at the float-precision level. The auto VP-based paths in
    fix.py still construct H from rotations (not via this helper), so CLI
    byte-identity is unaffected.
    """
    s = _basis_to_points_flat(
        src[0][0], src[0][1], src[1][0], src[1][1],
        src[3][0], src[3][1], src[2][0], src[2][1],
    )
    d = _basis_to_points_flat(
        dst[0][0], dst[0][1], dst[1][0], dst[1][1],
        dst[3][0], dst[3][1], dst[2][0], dst[2][1],
    )
    t = _multmm_flat(d, _adj_flat(s))
    inv = 1.0 / t[8]
    t = [v * inv for v in t]
    return np.array([
        [t[0], t[1], t[2]],
        [t[3], t[4], t[5]],
        [t[6], t[7], t[8]],
    ], dtype=np.float64)


def _aspect_from_corners(src):
    top_w = float(np.linalg.norm(src[1] - src[0]))
    bot_w = float(np.linalg.norm(src[2] - src[3]))
    left_h = float(np.linalg.norm(src[3] - src[0]))
    right_h = float(np.linalg.norm(src[2] - src[1]))
    avg_w = max(1.0, (top_w + bot_w) / 2.0)
    avg_h = max(1.0, (left_h + right_h) / 2.0)
    return avg_w / avg_h


def _rotation_from_corners(src, w, h, object_aspect=None):
    """
    Estimate the same pure camera-rotation homography the web preview uses.

    This intentionally mirrors solveCameraRotation() in index.html instead of
    using solvePnP. Four dragged corners are often only approximately a valid
    projection of one physical rectangle; DLT + Gram-Schmidt gives us one
    deterministic closest rotation for both preview and final export.
    """
    aspect = float(object_aspect) if object_aspect else _aspect_from_corners(src)
    if not np.isfinite(aspect) or aspect <= 0:
        return None, None, None

    world = [
        (0.0,    0.0),
        (aspect, 0.0),
        (aspect, 1.0),
        (0.0,    1.0),
    ]
    src_pairs = [(float(p[0]), float(p[1])) for p in src]
    try:
        Hwi = _projective_3x3(world, src_pairs)
    except (ZeroDivisionError, ValueError):
        return None, None, None

    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)
    M = K_inv @ Hwi

    m1 = M[:, 0]
    m2 = M[:, 1]
    n1 = np.linalg.norm(m1)
    n2 = np.linalg.norm(m2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None, None, None

    # IPPE planar pose is sign-ambiguous (the world plane can sit on either
    # side of the camera). Try both branches and keep the smaller-rotation
    # one — see the matching block in webapp/index.html's solveCameraRotation
    # for the geometric explanation. The wrong branch typically gives a
    # rotation near 180° about a world-plane axis; the rotated quad ends up
    # mirrored and downstream cropping collapses.
    def build_branch(sign):
        r1 = sign * m1 / n1
        r2 = sign * m2 / n2
        r2 = r2 - r1 * float(r1.dot(r2))
        n2b = np.linalg.norm(r2)
        if n2b < 1e-9:
            return None
        r2 = r2 / n2b
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R

    candidates = [R for R in (build_branch(+1), build_branch(-1)) if R is not None]
    if not candidates:
        return None, None, None

    def theta_of(R):
        trace = float(np.trace(R))
        return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))

    candidates.sort(key=theta_of)
    R = candidates[0]
    theta_deg = theta_of(R)
    if theta_deg > MAX_CAMERA_ROT_DEG:
        return None, None, theta_deg

    return R, K @ R.T @ K_inv, theta_deg


def _rotation_homography_from_corners(src, w, h, object_aspect=None):
    _R, H, _theta_deg = _rotation_from_corners(src, w, h, object_aspect=object_aspect)
    return H


def validate_corner_physics(source_corners, image_size, object_aspect=None,
                            max_rectify_error=0.035):
    w, h = [float(v) for v in image_size]
    src = np.asarray(source_corners, dtype=np.float64).reshape(4, 2)
    R, H, theta_deg = _rotation_from_corners(src, w, h, object_aspect=object_aspect)
    if H is None:
        return {
            "accepted": False,
            "theta_deg": theta_deg,
            "rectify_error": None,
        }

    rotated = cv2.perspectiveTransform(src.reshape(1, -1, 2), H).reshape(-1, 2)
    tl, tr, br, bl = rotated
    err = max(
        abs(tl[0] - bl[0]),
        abs(tr[0] - br[0]),
        abs(tl[1] - tr[1]),
        abs(bl[1] - br[1]),
    ) / max(1.0, min(w, h))
    return {
        "accepted": bool(err <= max_rectify_error),
        "theta_deg": theta_deg,
        "rectify_error": float(err),
    }


def warp_with_corners(bgr, source_corners, keep_aspect=True, object_aspect=None):
    """
    Camera-rotation perspective correction. The user-edited path.

    The 4 source corners are interpreted as the image projection of a real
    planar rectangle. We estimate the closest camera rotation with the same
    DLT + Gram-Schmidt path used by the web preview, then apply the inverse
    rotation as a pure-rotation homography H = K @ R.T @ K^-1. By construction
    the output is what a real camera at a different orientation would have shot:
    content cannot be sheared or non-uniformly stretched.

    source_corners: list of 4 [x, y] in source pixel space, clockwise from
        top-left (the order auto_correct returns and the web frontend uses).
    """
    h, w = bgr.shape[:2]
    src = np.asarray(source_corners, dtype=np.float32).reshape(4, 2)

    H_rot = _rotation_homography_from_corners(src, w, h, object_aspect=object_aspect)
    if H_rot is None:
        # Preserve the physical-authenticity contract: if the dragged corners
        # cannot produce a bounded camera rotation, do not fall back to a free
        # 8-DOF warp that can shear the subject.
        return bgr

    # Warp the whole image. Translate so all output coords are non-negative,
    # same trick as warp_and_get_quad does for the auto path.
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    pts = cv2.perspectiveTransform(img_corners.reshape(1, -1, 2), H_rot).reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    T = np.array([[1, 0, -min_xy[0]],
                  [0, 1, -min_xy[1]],
                  [0, 0, 1]], dtype=np.float64)
    H2 = T @ H_rot
    new_w = int(np.ceil(max_xy[0] - min_xy[0]))
    new_h = int(np.ceil(max_xy[1] - min_xy[1]))
    if new_w < 2 or new_h < 2:
        return bgr
    warped = cv2.warpPerspective(bgr, H2, (new_w, new_h))

    # The user's source quad, viewed through H2, lands as a near-rectangle in
    # warped space. Crop to the largest axis-aligned rect inside it.
    user_quad = cv2.perspectiveTransform(
        src.reshape(1, -1, 2).astype(np.float64), H2
    ).reshape(-1, 2)
    x0, y0, x1, y1 = largest_inscribed_rect(user_quad, warped.shape)
    if x1 <= x0 or y1 <= y0:
        return warped

    if keep_aspect:
        x0, y0, x1, y1 = fit_aspect(x0, y0, x1, y1, w / h, new_w, new_h)

    return warped[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--mode", choices=["vertical", "horizontal", "both"], default="vertical")
    ap.add_argument("--keep-aspect", action=argparse.BooleanOptionalAction, default=True,
                    help="Center-crop output to match input aspect ratio (default on)")
    args = ap.parse_args()

    try:
        img, icc, exif = load_image(args.input)
    except Exception as e:
        print(f"Cannot read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    gravity = apple_acceleration_from_exif(exif)
    h, w = img.shape[:2]
    intrinsics = camera_intrinsics_from_exif(exif, w, h)
    cropped, corners, _actual_mode, meta = auto_correct(
        img, mode=args.mode, strength=args.strength, keep_aspect=args.keep_aspect,
        gravity=gravity, intrinsics=intrinsics,
    )
    if corners is None:
        print(f"Correction skipped: {meta.get('reason') or 'VP missing'}.", file=sys.stderr)
        retained_ratio = 1.0
    else:
        retained_ratio = abs(cv2.contourArea(np.asarray(corners, dtype=np.float32))) / (w * h)
    save_image(args.output, cropped, icc=icc, exif=exif)
    print(f"Saved {args.output}  ({cropped.shape[1]}x{cropped.shape[0]}, "
          f"{100 * retained_ratio:.0f}% of source area retained)")


if __name__ == "__main__":
    main()
