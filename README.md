# honest-perspective

> Perspective correction that never invents scene content — and would rather do nothing than guess.

[中文说明 →](README.zh.md)

![Original street photo transitioning to automatic vertical perspective correction](docs/assets/demo.gif)

A prototype for fixing perspective distortion in photos. Same problem space as the perspective tools in Snapseed or Lightroom, with two deliberately honest constraints:

- **Crop-only, never fill.** The warp is cropped to the valid source-image footprint. It resamples the source normally, but never fills missing regions with black borders, content-aware synthesis, or AI-generated "fake background".
- **No correction beats a wrong correction.** Auto mode treats "leave the photo alone" as a first-class candidate, and picks it whenever the evidence isn't convincing.

Especially useful for photos of buildings: the converging verticals from tilting the camera up or down can be straightened.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python **3.11** is recommended. On Windows, activate the environment with `.venv\Scripts\activate` instead.

If you get an error that `cv2.createLineSegmentDetector` doesn't exist, install `opencv-contrib-python` instead.

## CLI usage

```bash
python fix.py input.jpg output.jpg
```

By default only **vertical** lines are corrected (like Lightroom's Auto Vertical), preserving the natural perspective of horizontal lines. For architecture this is usually the best-looking result.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--mode {vertical,horizontal,both}` | `vertical` | Which direction to correct |
| `--strength FLOAT` | `1.0` | Correction strength: `0.0` = untouched, `1.0` = full correction, in between = interpolated |
| `--keep-aspect / --no-keep-aspect` | on | Re-crop to the source aspect ratio, centered |

### The three modes

- **`vertical`** (default): make near-vertical lines truly vertical, keep horizontal perspective. The most robust mode; best for buildings.
- **`horizontal`**: the reverse — make near-horizontal lines level. Good for objects shot from the side and tabletop scenes.
- **`both`**: correct both directions for a fully "orthographic" look. Needs two clearly visible vanishing-point families; otherwise it over-corrects (symptom: the frame gets stretched into a strange trapezoid and the crop loss is large).

### Using `--strength`

If `1.0` feels over-corrected (the building looks unnaturally regular), try `0.6`–`0.7`. Interpolation happens on the rotation manifold, so intermediate values are still physically valid camera rotations.

## Web app

Run a local web editor on your LAN:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- Desktop: http://127.0.0.1:8000/
- iPhone / iPad on the same Wi‑Fi: use your machine's LAN IP, e.g. `http://192.168.x.x:8000/`

Supports JPG / PNG / **HEIC** (iPhone's "High Efficiency" format, decoded server-side via `pillow-heif`). If the browser can't decode a format itself, the preview falls back to a server render (`POST /preview`); the full-resolution original is still what gets warped on save.

### Hosted demo (GitHub Pages)

The frontend degrades gracefully when no backend is reachable: it switches to a manual-only demo mode where Save renders the correction with WebGL in the browser instead of calling `/warp`. That is what the GitHub Pages demo runs — photos never leave your device. `.github/workflows/pages.yml` publishes `webapp/` on every push to `main` (repo Settings → Pages → Source: GitHub Actions).

Demo-mode trade-offs vs the local backend:

- no auto correction (LSD / RANSAC / the gravity prior live in Python)
- focal length falls back to `max(w, h)` — no EXIF parsing in the browser yet
- the exported JPEG is untagged sRGB and carries no EXIF/ICC
- HEIC only works in browsers that decode it natively (Safari)
- images beyond the GPU texture limit are downscaled to fit

One thing the demo does *better* than the plain-HTTP LAN setup: it is served over HTTPS, so the iOS share sheet ("Save to Photos") actually works.

### Interaction

With the local backend running:

- Drop or pick a photo → the server extracts line segments and vanishing points once, builds four candidates (no-change / vertical / horizontal / both), and only auto-applies one when the lines measurably improve and enough source pixels survive the crop
- The auto pick is shown by default; "no change" is a first-class candidate you can also select manually
- **Drag** re-aims the camera: horizontal displacement is pure yaw, vertical displacement is pure pitch. A one-finger drag never introduces roll, and the same gesture means the same thing no matter where on the image it starts
- **Hold the "original" button** to peek at the uncorrected source; release to return
- **Save**: on desktop as a JPG download. On iOS, HTTPS enables the Web Share API and its "Save to Photos" action; the plain-HTTP LAN URL above falls back to a download, which Safari normally saves to Files. The local backend preserves EXIF (capture time, camera, GPS, lens) and the ICC profile; the hosted demo has the limitations listed above
- The preview is downscaled in the browser to ~`max(viewport) × DPR` (capped at 2500px) so `matrix3d` stays at 60fps on large photos; local-backend saves use the full-resolution original, while hosted-demo exports remain subject to the GPU texture limit

### The math behind manual correction

Manual mode is a **strict camera-rotation model**. The authoritative state is not four freely draggable corners, but:

1. `intrinsics`: K computed from EXIF `FocalLengthIn35mmFilm` when present, falling back to `max(w, h)`
2. `rotation`: a 3×3 matrix on SO(3) (`RᵀR = I`, `det R = 1`)
3. `crop`: a same-aspect crop inside the rotated source quad, uniform scaling only
4. Final warp = `S_crop · K · R · K⁻¹` — there are no independent arbitrary-shear or non-uniform-scale controls; every projective change comes from a physically valid camera rotation

A drag computes `yaw = atan(dx / fx)` and `pitch = -atan(dy / fy)` from the pose at pointer-down, then composes on SO(3). A two-dimensional gesture drives exactly two camera axes; no roll is mixed in to make a corner track the finger, which is why the gesture is position-independent.

The frontend (`webapp/geometry.js`) and backend (`geometry.py`) are the same geometry contract; randomized-rotation contract tests compare matrices, crops, and angles across both implementations (`tests/test_geometry.py`). The save endpoint sends the correction state directly; a legacy corners API remains only as a compatibility path.

Two safety valves guard the manual path:

| Constant | Value | Rejects when |
|---|---|---|
| `MAX_ROT_RAD` | 35° | the state's true axis-angle exceeds 35° |
| `MIN_PROJECTIVE_W_RATIO` | 0.22 | the projective w min/max ratio drops below 0.22 (near-degenerate) |

`window.__rejectStats` counts how often each threshold fires, so real usage can calibrate these numbers over time.

## How auto mode works

1. **Find segments**: `cv2.createLineSegmentDetector` extracts salient line segments
2. **Cluster by angle**: split into "near-vertical" and "near-horizontal" groups (±25° tolerance)
3. **RANSAC vanishing points**: estimate a vanishing point per group
4. **Build the homography**: send each vanishing point to infinity in its canonical direction
5. **Warp**: the frame edges become an irregular quad
6. **Largest inscribed rectangle**: the key to "crop, never fill"
7. **(Optional)** center-crop back to the source aspect ratio

Candidate selection tracks how the *source* segments would align after each candidate rotation (no re-detection on warped pixels, so no resampling noise); if the improvement is under 0.25°, "no change" wins.

### Where the algorithm honestly fails

The auto detector is a **line-segment aligner**. Its hidden assumption is *"the dominant line directions in the frame are the world's orthogonal axes."* Ranked by how dangerously that assumption breaks:

- **Buildings, documents, screens, signs, tabletops** (strong man-made orthogonal lines): ✅ the sweet spot
- **Pyramids / tents / domes** (no vertical lines, though humans perceive an implied "up"): the algorithm can't see the implied vertical — it either fails to find a VP or corrects only horizontally. A relatively **safe** failure: no improvement, but no damage
- **Cups / sculpture / curved objects** (almost no straight lines): LSD finds nothing, the tool returns the original. The **safest** failure
- **Forests / grass / crowds** (many weakly parallel lines): the **most dangerous** — LSD finds many short near-vertical segments, RANSAC finds a barely-plausible VP, and the algorithm confidently applies a correction that may point the wrong way. This is exactly why "no change" is a first-class candidate

### Gravity prior

iPhones keep the accelerometer running while shooting — **Apple Maker Notes in the EXIF record the gravity vector at the moment of capture**. In other words, the phone already knows which way is down, independent of image content.

The project parses that vector, maps it into image coordinates according to device orientation, and gates it on the acceleration norm to reject motion-contaminated samples. It is a single total-acceleration sample, not unconditional ground truth.

| Scenario | LSD + RANSAC | Gravity prior |
|---|---|---|
| Buildings | ✅ works | ✅ independent orientation estimate |
| Pyramids / cups / sculpture | ❌ no solution | ✅ can estimate tilt without visible lines |
| Forests / weak structure | ⚠️ confidently wrong | ✅ doesn't depend on segments |
| Lens distortion | not corrected | not corrected |
| Screenshots / no EXIF | works | no data → falls back to LSD |

A GeoCalib cross-check on 9 samples showed most images agree with Apple's gravity within ~1.7°, but two conflicted by 5.6° and 9.4° even with a healthy norm. So the plan is *not* "gravity overrides visual VPs" — it's three estimators (Apple gravity, visual VPs, GeoCalib) each producing proposals with confidence, defaulting to no-change on conflict. See `spike_geocalib/FINDINGS.md`.

The `spike_*/FINDINGS.md` files document the research behind these decisions, including the dead ends.

## Tests

```bash
python -m unittest discover -s tests -v
```

Node.js is required for the browser/backend geometry contract tests.

## Color

iPhone JPGs are usually **Display P3** with an embedded ICC profile. IO goes through Pillow and **preserves the ICC profile and EXIF as-is**, so colors don't wash out after processing.

## Known limitations

- `--mode both` is unstable in visually busy scenes (stray lines get mistaken for horizontals); fall back to `vertical` when in doubt
- If no vanishing point can be found at all (e.g. pure nature scenes with no straight lines), the tool saves the original unchanged
- **EXIF Orientation is only fully tested for Orientation=1**; other values still need complete pixel-transpose + gravity-remap + tag-cleanup tests
- **HDR is not preserved**: iPhone HDR photos store a gainmap auxiliary layer in the HEIC container. The pipeline decodes only the primary image, and JPEG output has nowhere to put a gainmap, so iOS Photos renders the result as SDR.
  - Upstream blocker: libheif / pillow-heif / libvips can currently *read* Apple gainmaps but not *write* them. Google's libultrahdr roadmap adds HEIC gainmap support in 2026; libvips is expected to follow.
  - Apple's official HDR editing API (`CIContext.writeHEIFRepresentation` + `kCGImageAuxiliaryDataTypeISOGainMap`) is Swift/Obj-C only. Truly preserving HDR likely requires a native iOS Photo Editing Extension.

## Roadmap

- [ ] Fuse Apple gravity / visual VP / GeoCalib proposals with per-estimator confidence
- [ ] Full tests for EXIF Orientation ≠ 1
- [ ] HDR gainmap preservation (waiting on libvips / libultrahdr HEIC gainmap writing)
- [ ] Multiple vanishing-point families for complex scenes
- [ ] Try M-LSD or DeepLSD in place of classic LSD
- [ ] Mac Photos / iOS Shortcut integration
- [ ] Native iOS Photo Editing Extension (the only official path to in-place edits in the photo library with HDR intact)

## License

[MIT](LICENSE)
