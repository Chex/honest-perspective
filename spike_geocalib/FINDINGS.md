# GeoCalib Spike Findings

Date: 2026-07-10

## Question

Can GeoCalib improve the project's focal-length estimate, validate Apple
`AccelerationVector`, and provide a useful confidence signal for automatic
correction?

## Setup

- Official GeoCalib commit `97b8968e7798a66bf04fcf791fb535624241bda7`
- Pinhole checkpoint, 111 MB
- Mac MPS, input downscaled to at most 768 px before GeoCalib's own 320 px pass
- Nine existing project samples
- EXIF focal length remains the reference where available
- Gravity comparison is axis-based, so opposite up/down signs are equivalent

## Results

| Image | Apple vs GeoCalib gravity | GeoCalib gravity uncertainty | Focal error vs EXIF | Warm inference |
|---|---:|---:|---:|---:|
| 4501 | 9.42 deg | 6.19 deg | -40.3% | 1.13 s |
| 5302 | 1.67 deg | 5.55 deg | -22.3% | 0.83 s |
| 5362 | 0.18 deg | 5.54 deg | -0.5% | 0.46 s |
| 5587 | 5.56 deg | 3.00 deg | +34.0% | 0.48 s |
| 5606 | 1.02 deg | 3.59 deg | +21.9% | 0.48 s |
| 5792 | 11.95 deg | 1.34 deg | -10.3% | 0.45 s |
| 5893 | 1.45 deg | 2.67 deg | +9.7% | 0.88 s |
| 5980 | 1.27 deg | 2.03 deg | -15.5% | 0.42 s |
| 5984 | 1.64 deg | 3.61 deg | +58.2% | 0.38 s |

Cached model construction took about 0.51 s. The first run also downloaded the
checkpoint and took about 19 s to construct the model. Running GeoCalib on the
full-resolution input is wasteful because its public wrapper upsamples confidence
fields back to the input size; downscaling first reduced warm inference to
0.38-1.13 s.

## Conclusions

### Keep EXIF as the source of camera intrinsics

GeoCalib focal estimates vary from -40% to +58% relative to EXIF and correctly
report large focal uncertainty on difficult images such as 5984. It is useful as
a fallback when EXIF is absent, but should not replace trustworthy EXIF.

### GeoCalib is a useful independent gravity witness

Six images agree with Apple gravity within 1.7 degrees. The strongest cases are:

- 5984: 1.64-degree agreement independently supports the near-level Apple pose.
- 5362: 0.18-degree agreement confirms the large intentional upward camera pose.
- 5792: 11.95-degree disagreement confirms the existing acceleration-norm gate;
  Apple norm is 1.207 and motion-contaminated, while GeoCalib is confident.

However, 5587 and 4501 show that a near-unit acceleration norm is not sufficient
to trust Apple gravity for subtle correction. They disagree with GeoCalib by
5.56 and 9.42 degrees despite passing the norm gate.

### Agreement is not permission to auto-correct

5362 is the important counterexample: Apple and GeoCalib agree almost perfectly,
but fully correcting a deliberate 40-degree upward view would destroy the
photographer's intent. Estimator confidence answers "is this pose real?", not
"should the product remove it?" Large corrections still need an intent/crop-loss
gate or explicit user confirmation.

## Recommended integration

Do not put GeoCalib in the manual rendering or save path. Keep it as a lazy-loaded
automatic estimator returning a gravity proposal and uncertainty:

1. Use EXIF focal length whenever present.
2. Reject Apple gravity when its acceleration norm is contaminated.
3. When Apple and GeoCalib agree within their uncertainty, fuse the two axes.
4. When they disagree, preserve both proposals and prefer no correction unless
   visual line evidence clearly supports one.
5. When Apple gravity is absent or rejected, GeoCalib can supply the gravity
   proposal.
6. Apply a separate semantic safety rule for large/intentional corrections.

Raw measurements are in `results.json`; reproduction is `uv run python spike.py`.
