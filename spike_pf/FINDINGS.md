# PF (Perspective Fields) Spike — Findings

**Date**: 2026-05-22
**Branch**: develop
**Spike code**: `perspective_fix/spike_pf/`
**Status**: superseded (2026-07-10) — GeoCalib provides the same signal class
(gravity + focal + per-image uncertainty) without the Adobe non-commercial
license; see `../spike_geocalib/FINDINGS.md`. The std-based gating idea and
`cross_check_gravity.py` remain referenced by the gravity validation chain.

## Context

`perspective_fix` is line-based (LSD + RANSAC + solvePnP). It struggles on:
- Line-poor scenes (forest, hand-held cup, blank wall)
- Mixed-line scenes where RANSAC picks the wrong cluster as "vertical"
- Subtle hand-held tilts where there's no clear architectural reference

Two prior dead-ends informed this spike:
1. EXIF Apple `AccelerationVector` gravity — reads correctly but the
   device→image coordinate mapping is hard to get right; abandoned (gated
   `gravity_mode=off` by default).
2. Depth Anything 3 — depth/intrinsics work but single-image `extrinsics`
   is always identity (camera pose is undefined without a second view).

This spike investigates **Perspective Fields** (PF, CVPR 2023 Highlight,
Adobe Research). PF directly predicts per-pixel "up-vector" + (roll, pitch,
vfov) for a single image — exactly the rectification primitive we need.

## What we tried

### Setup
- 9 test images across 5 scene types (architectural, indoor product,
  forest, hand-held object, intentional looking-up).
- 2 PF model checkpoints:
  - `PersNet_Paramnet-GSV-centered` — trained on Google Street View
  - `Paramnet-360Cities-edina-centered` — trained on 360cities + EDINA,
    advertised as the broader "indoor/outdoor/natural/egocentric" model
- Inference on Mac MPS, ~230ms/image after warmup, 47s cold load.

### Rectification math
PF's `R = R_z(roll) @ R_x(pitch)` is **non-standard**: their rotation
matrices' sin signs are flipped relative to the textbook convention, which
makes their `R` effectively the inverse of the standard rotation. So the
rectification homography is `H = K @ R @ K_inv` (not `R.T`).

### Comparison
For each image: side-by-side current RANSAC output vs PF-derived
rectification, using PF's vfov to set focal length.

## Findings

### 1. Visual quality: PF matches RANSAC on RANSAC-friendly images

| Image | Scene | RANSAC vs PF |
|-------|-------|--------------|
| 5587 | suburban house | tie |
| 5606 | toothpaste shelf | PF slightly more level |
| 5792 | city street | tie |
| 5980 | bread shelves | tie |
| 5984 | brick wall + sticker | tie |
| 5302 | drink + box on table | PF gives subtle correction, RANSAC barely moves |
| 4501 | hand-held yogurt | PF auto-levels, RANSAC doesn't move |
| **5893** | **dense forest** | **PF clean, RANSAC misfires badly** |
| **5362** | **intentional looking-up at pagoda** | **PF over-corrects (pitch=40°), RANSAC preserves intent** |

The "PF wins on line-poor scenes" hypothesis holds (5893, 4501, 5302). The
"PF can be too aggressive on intentional perspective" tradeoff also holds
(5362).

### 2. Choice of PF checkpoint matters but is not a free upgrade

Switching `GSV-centered` → `EDINA-centered` changed outputs significantly,
but introduced new failures:

| Image | GSV pitch | EDINA pitch | Visual |
|-------|-----------|-------------|--------|
| 5984 brick wall | +4.6° | **+77.9°** | EDINA catastrophic (image rotated out of canvas) |
| 5362 pagoda | +40.2° | +48.9° | EDINA more aggressive |
| 4501 yogurt | -5.0° | -12.5° | EDINA exceeds practical threshold |
| 5893 forest | +2.9° | +5.0° | both fine |
| others | similar | similar | minor diffs |

Selecting a different checkpoint moves the failure mode but doesn't
eliminate it. We need a per-image **runtime confidence** signal, not
better priors.

### 3. The per-pixel field IS that confidence signal

PF outputs (`pred_gravity`) are a (2, H, W) tensor — per-pixel up-vectors.
The (roll, pitch, vfov) scalars from ParamNet are a derived summary that
loses this information.

Computed circular standard deviation of the up-vector angle across the
image (`std_all`) and the central 50% crop (`std_central`):

| Image | Model | std_all | std_central | Quality |
|-------|-------|---------|-------------|---------|
| 5587 | GSV | 1.5° | 0.6° | confident & right |
| 5606 | GSV | 1.6° | 0.7° | confident & right |
| 5792 | GSV | 2.3° | 1.1° | confident & right |
| 5980 | GSV | 1.9° | 1.1° | confident & right |
| 5984 | GSV | 1.3° | 0.7° | confident & right |
| 5302 | GSV | 2.6° | 1.2° | confident & right |
| 5893 | GSV | 1.9° | 1.0° | confident & right |
| 4501 | GSV | 5.0° | 1.7° | confident & right (auto-level) |
| **5362** | GSV | **17.6°** | 8.7° | confident & physically right, artistically wrong |
| 5984 | **EDINA** | **108°** | **94.6°** | **incoherent / catastrophic** |
| 5362 | EDINA | 20.2° | 8.5° | over-corrects |
| 4501 | EDINA | 4.8° | 2.3° | borderline |

Clear separation:
- **`std_central < 5°`** ⇒ field is coherent, trust the (roll, pitch)
- **`std_central > 30°`** ⇒ field is incoherent, model is grasping at
  straws (5984+EDINA only — happens with checkpoint/domain mismatch)
- **`std_all > 12°` with `std_central < 12°`** ⇒ field is coherent but has
  high variation across image — extreme perspective, likely intentional
  (5362 pagoda)

### 4. PF inference is fast enough for backend use

- Cold model load (downloads weights): 47s for 399MB GSV / 798MB EDINA
- Warm inference: 0.23–0.26s per image on Mac MPS (M-series GPU)
- 9 images in ~3s total after first warmup
- Memory: well within unified memory budget

## Proposed integration: A++

The pre-spike "plan A+" was `RANSAC fallback only when vp_v is None, +
hard cap |pitch|>10°`. The spike data lets us refine to **A++** with a
data-driven gate:

```python
def gate_pf_correction(pred):
    """Returns one of: APPLY, REJECT_INCOHERENT, REJECT_INTENTIONAL."""
    g = pred["pred_gravity"].cpu().numpy()       # (2, H, W)
    ang = np.arctan2(g[1], g[0])
    H, W = ang.shape
    central = ang[H//4:3*H//4, W//4:3*W//4]
    std_all = circ_std(ang.ravel())
    std_central = circ_std(central.ravel())
    pitch = abs(float(pred["pred_pitch"]))

    if std_central > 30:
        return "REJECT_INCOHERENT"   # 5984+EDINA pattern
    if std_all > 12 and pitch > 10:
        return "REJECT_INTENTIONAL"  # 5362 pagoda pattern
    return "APPLY"
```

This rule correctly classifies all 18 measured (image, checkpoint) pairs.
Thresholds (30°, 12°, 10°) are calibrated to the data; further field
testing may shift them slightly.

### Where PF plugs into fix.py
- Add lazy-loaded PF model wrapper.
- In `auto_correct` / `choose_auto_mode`, when RANSAC fails to find a
  valid `vp_v`, call PF.
- Apply gate. If `APPLY`, compose H from (roll, pitch) and PF's vfov;
  else fall through to "no correction".
- Keep existing CLI byte-identity on the 3 reference JPGs (RANSAC path
  unchanged on those images).

### Cold-start cost
PF's 47s cold load is a problem for a request-per-photo backend. Mitigation:
- Lazy-load on first PF-requiring request (only pay cost when needed).
- Keep model resident in the uvicorn process (current single-worker setup
  already does this).
- For batch CLI use, model loads once per invocation — acceptable.

## License caveats — must read before any non-self-use

- **Perspective Fields**: Adobe Research License, **non-commercial only**.
  Same restriction tier as DSINE.
- Self-use prototype OK. Distributing this as a product, SaaS, or open
  source = must contact Adobe (research contracts) first, or train a
  drop-in replacement.
- The principles we'd extract for integration (std-based gating, field
  output usage) are not encumbered — only PF's weights and code are.

## Open questions

1. **Production focal length**: PF's vfov estimate disagrees with our
   `f = max(w, h)` heuristic by up to 31%. Even outside the PF integration,
   we could use PF's f to improve RANSAC's solvePnP precision. Worth
   experimenting independently.
2. **Smoothing the integration**: PF's H may produce visually different
   corrections than RANSAC's even on RANSAC-friendly images. Need to
   decide if the fallback-only path is enough, or if we want PF to also
   sanity-check RANSAC's output (disagreement → caution).
3. **Subject-aware gating**: if the user has a clear subject (e.g.,
   tappable plane), restrict the up-vector averaging to that region —
   this would beat the global std_central heuristic. Requires SAM2 or
   user interaction, out of scope for now.
4. **Model swap**: if Adobe releases a v2 / a different model later,
   the std-based gating logic carries over — only the weights file path
   changes.

## Spike artifacts (in `spike_pf/`)

- `spike.py` — minimal "load PF, print (roll, pitch, vfov)" smoke test
- `compare.py` — full rectification + side-by-side vs current RANSAC
- `visualize_field.py` — overlays up-vector arrows + computes std metrics
- `comparison/` — side-by-side JPGs (GSV + EDINA labelled)
- `field_viz/` — arrow overlays per image
- `pyproject.toml` / `.venv/` — uv environment, ~5GB on disk (mostly
  PyTorch + downloaded model weights)

## Reproduction

```bash
cd perspective_fix/spike_pf
uv run python compare.py        # side-by-side comparison
uv run python visualize_field.py # field visualization + std metrics
```
