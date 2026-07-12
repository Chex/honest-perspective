# Gravity Prior Spike — Findings

**Date**: 2026-05-23
**Status**: Research complete, integration pending

## Context
Previous attempts to use Apple's `AccelerationVector` MakerNote from HEIC/JPEG EXIF were abandoned because the device-to-image coordinate mapping appeared inconsistent. The `gravity_mode` was defaulted to `off`. 

## The Core Discovery: The iOS Auto-Rotation Trap
The inconsistency wasn't due to bad sensor data, but rather an iOS image processing behavior:
1. Apple records the `AccelerationVector` in the **physical hardware coordinate system** at the exact moment of capture (e.g., -Y is the charging port, -X is the volume buttons).
2. However, when iOS saves the photo, it **auto-rotates the pixel matrix** to be upright (portrait or landscape depending on the dominant tilt) and **resets the EXIF `Orientation` tag to 1**.
3. Crucially, the `AccelerationVector` in the EXIF is **NOT rotated** to match the new pixel matrix. 

This means reading the `Orientation` tag is useless because it is always 1 for processed photos, causing the physical gravity vector and the image coordinate system to decouple.

## The Solution: Self-Describing Orientation
Since the gravity vector always points physically "down", its `(X, Y)` components inherently describe how the phone was held:
- If `|Ax| > |Ay|`, the phone was held in landscape.
- If `|Ay| > |Ax|`, the phone was held in portrait.

By checking the signs and magnitudes of the raw acceleration vector, we can deduce exactly how iOS rotated the pixels, allowing us to map the hardware acceleration vector into the image's coordinate space dynamically.

### Coordinate Mapping Logic
```python
def get_image_gravity(acc):
    Ax, Ay, Az = acc
    # Deduce phone orientation from gravity
    if abs(Ax) > abs(Ay): # Landscape
        if Ax < 0:
            Ix, Iy, Iz = Ay, -Ax, -Az
        else:
            Ix, Iy, Iz = -Ay, Ax, -Az
    else: # Portrait
        if Ay < 0:
            Ix, Iy, Iz = Ax, -Ay, -Az
        else:
            Ix, Iy, Iz = -Ax, Ay, -Az
            
    v = np.array([Ix, Iy, Iz], dtype=np.float64)
    return v / np.linalg.norm(v)
```

## Data Validation
We compared the dynamically mapped gravity vector against the RANSAC-derived visual vanishing point ray across our sample images.

| Image | Scene | Mapped Gravity | Visual VP Ray | Angle Diff | Notes |
|-------|-------|----------------|---------------|------------|-------|
| 5893 | Forest | `[ 0.00,  1.00, -0.01]` | `[ 0.01,  1.00, -0.01]` | 0.75° | Near perfect match in a line-poor scene. |
| 5302 | Cup on desk | `[ 0.00,  1.00,  0.07]` | `[ 0.02,  0.99,  0.10]` | 2.02° | Near perfect match. |
| 5587 | Suburban house | `[-0.05,  1.00, -0.01]` | `[ 0.03,  1.00, -0.04]` | 4.52° | Very close. |
| 5980 | Bread shelves | `[-0.00,  1.00,  0.05]` | `[ 0.02,  1.00,  0.06]` | 1.68° | Match. |
| **5984** | **Brick wall** | **`[ 0.02,  1.00,  0.02]`** | **`[-0.05,  0.83,  0.55]`** | **32.47°** | **Catastrophic RANSAC failure. Gravity reveals the phone was plumb (Z≈0.02).** |

## Conclusion
Gravity data is the ultimate "free" prior for vertical perspective correction. It successfully handles scenes where line-based RANSAC fails entirely (e.g., dense brick walls) or degrades (e.g., forests, curved objects), achieving similar benefits to the `Perspective Fields` model but with **zero inference cost**.

## Validation Round 2 (2026-05-23)

Two additional checks performed after initial findings:

### Step 1: Orientation tag sanity check
All 9 sample images return EXIF `Orientation=1`. The load-bearing assumption
("iOS resets Orientation to 1 after auto-rotating pixels") holds for this
dataset. No image takes the unhandled branch.

Branch decisions are also robust on this sample — no `|Ax|≈|Ay|` near-ties,
even for the most extreme tilt (5362 pagoda with `|Az|=0.679`, where
`|Ax|=0.735 >> |Ay|=0.012`).

### Step 2: Cross-validation against Perspective Fields
For each image, converted PF's `(roll, pitch)` to an image-frame gravity
unit vector (`g_pf`) and compared against EXIF-derived `g_exif`.

| Image | PF roll | PF pitch | g_exif | g_pf | Diff |
|-------|---------|----------|--------|------|------|
| 5587 | +1.40° | +4.18° | [-0.05, +1.00, -0.01] | [+0.02, +1.00, -0.07] | 5.40° |
| 5606 | -1.28° | -5.08° | [+0.01, +0.99, +0.11] | [-0.02, +1.00, +0.09] | 1.82° |
| **5792** | -1.68° | +5.90° | [+0.20, +0.98, -0.09] | [-0.03, +0.99, -0.10] | **12.98°** |
| 5980 | +0.94° | -4.87° | [-0.00, +1.00, +0.05] | [+0.02, +1.00, +0.08] | 2.56° |
| **5984** | +0.08° | +4.59° | [+0.02, +1.00, +0.02] | [+0.00, +1.00, -0.08] | 5.91° |
| 5302 | -0.62° | -4.76° | [+0.00, +1.00, +0.07] | [-0.01, +1.00, +0.08] | 0.92° |
| **5362** | +0.28° | +40.21° | [+0.01, +0.73, -0.68] | [+0.00, +0.76, -0.65] | 2.56° |
| 5893 | -1.77° | +2.92° | [+0.00, +1.00, -0.01] | [-0.03, +1.00, -0.05] | 3.02° |
| 4501 | +4.15° | -5.00° | [-0.01, +0.99, +0.15] | [+0.07, +0.99, +0.09] | 6.02° |

8 of 9 images agree to within 6°; two independent algorithms (EXIF sensor
fusion vs neural-net visual inference) converge on the same gravity
direction. Most importantly:

- **5984 (the brick-wall win-case)**: PF independently confirms "phone was
  plumb" — corroborates that RANSAC's 32° disagreement in the original
  validation was RANSAC failing, not gravity failing. The Round 1 claim
  now has external support.
- **5362 (40° pagoda tilt)**: both methods agree the camera is looking up
  ~40°. The `|Ax|>|Ay|` orientation branch survives extreme Z tilt.

### The 5792 outlier — a real limitation

5792's raw acceleration vector has norm `1.207`, well above 1.0. Apple's
`AccelerationVector` is total acceleration (gravity + user motion), not
gravity alone. When the phone is moving during capture (walking, shutter
press jitter), the gravity reading is contaminated.

The fix is a confidence gate using the raw norm:

```python
if abs(np.linalg.norm(acc) - 1.0) > 0.1:  # motion-contaminated
    # do not trust gravity; fall back to PF or RANSAC
```

This catches 5792 (norm=1.207, deviation 0.207) cleanly. The other 8
images all have norm within ±0.1 of unity and produce agreeing
predictions.

The Round 1 `apple_acceleration_from_exif` accepts norm ∈ [0.75, 1.25],
which is too loose for this use — it accepts 5792's contaminated reading.
Tightening the acceptance window (or surfacing the norm as a confidence
output) is required before integration.

### Round 2 conclusion

Method is sound. Three things needed before shipping:
1. Tighten norm-deviation tolerance, or expose norm as confidence.
2. Continue to bail when Orientation ≠ 1 (already in `_gravity_vertical_vp`).
3. Test on deliberately upside-down / 45°-rotated samples (not yet covered).

## Spike Artifacts
- `test_gravity.py` - Initial investigation script testing raw gravity vectors.
- `test_gravity_mapping.py` - The script that implements the dynamic mapping and computes Angle Diff against RANSAC.
- `dump_gravity.py` - Round 2 Step 1: prints Orientation tag + raw acc + mapped g_img for all samples, writes gravity.json.
- `gravity.json` - Per-image EXIF gravity data, consumed by spike_pf/cross_check_gravity.py.
- (in spike_pf/) `cross_check_gravity.py` - Round 2 Step 2: runs PF, converts (roll, pitch) → g_pf, compares to gravity.json.
