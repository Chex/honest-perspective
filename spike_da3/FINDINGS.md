# Depth Anything 3 Spike — Findings

**Date**: 2026-05-21
**Status**: dead end, archived

## What we tried
Depth Anything 3 (Aug 2025, ByteDance / TikTok) advertises single-image
depth + camera intrinsics + extrinsics. The extrinsics output would, if
real, give us camera pose directly — the perspective rectification primitive
we need.

## Findings

1. **Setup viable**: DA3-Small loads on Mac MPS without xformers (its
   requirement is optional despite being listed; the code path has a
   try/except fallback to pure PyTorch). About 5 GB of deps, including
   moviepy, e3nn, pycolmap, evo. `KMP_DUPLICATE_LIB_OK=TRUE` needed
   on Mac due to pycolmap's OpenMP duplicate library.
2. **Inference works**: depth and intrinsics outputs look sane.
3. **Extrinsics are identity**: for any single-image input, the returned
   extrinsics matrix is the identity. This is mathematically correct —
   camera pose is undefined without a second view or a prior frame; the
   model has no signal to predict it from. The "single-image extrinsics"
   capability advertised on the model card refers to multi-image
   inference with a defined reference frame.

## Conclusion

DA3 cannot help with perspective rectification from a single photo. It
would only become useful if we had a second image (e.g., burst frames)
to give it baseline. Out of scope for this project.

For the path forward (Perspective Fields + EXIF gravity), see
`../spike_pf/FINDINGS.md` and `../spike_gravity/FINDINGS.md`.

## Spike Artifacts
- `spike.py` — smoke test that loads DA3, runs inference, dumps the
  identity extrinsics output that killed this direction.
- `pyproject.toml` / `uv.lock` — environment record. Removed `.venv/` and
  cloned `da3_repo/` to reclaim 1 GB.
