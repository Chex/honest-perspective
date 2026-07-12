"""FastAPI backend for the perspective_fix web app.

Two stateless endpoints (per design decision in plan-eng-review):
  POST /detect → auto-detect the source quad for a given mode
  POST /warp   → warp source image with user-edited corners, return JPEG

Static frontend served from ./webapp/.

Run:
  uv run uvicorn server:app --host 0.0.0.0 --port 8000
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from fix import (
    MAX_CAMERA_ROT_DEG,
    apple_acceleration_from_exif,
    auto_correct,
    auto_correct_all_modes,
    camera_intrinsics_from_exif,
    choose_auto_mode,
    load_image,
    validate_corner_physics,
    warp_with_corners,
)
from geometry import validate_correction_state, warp_with_state

app = FastAPI(title="perspective_fix")
STATIC_DIR = Path(__file__).parent / "webapp"


# Self-use dev server: never let the browser cache anything. Avoids the
# "I edited index.html but Safari still shows the old version" trap.
@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


def _jpeg_bytes(bgr, icc=None, exif=None, quality=95) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    kwargs = {"format": "JPEG", "quality": quality}
    if icc:
        kwargs["icc_profile"] = icc
    if exif:
        kwargs["exif"] = exif
    pil.save(buf, **kwargs)
    return buf.getvalue()


def _read_upload(data: bytes):
    try:
        return load_image(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    gravity_mode: str = Form("auto"),
):
    if mode not in ("auto", "vertical", "horizontal", "both"):
        raise HTTPException(status_code=400,
                            detail="mode must be auto/vertical/horizontal/both")
    if gravity_mode not in ("auto", "force", "off"):
        raise HTTPException(status_code=400,
                            detail="gravity_mode must be auto/force/off")
    data = await file.read()
    bgr, _icc, exif = _read_upload(data)
    gravity = apple_acceleration_from_exif(exif)
    h, w = bgr.shape[:2]
    intrinsics = camera_intrinsics_from_exif(exif, w, h)
    alternatives = None
    mode_meta = None
    if mode == "auto":
        # Run all 3 modes ONCE; pick_best is now a thin selector on top.
        # Expose every mode's result so the frontend's menu can show
        # "Full correction 95.4%" etc. without re-fetching.
        all_results = auto_correct_all_modes(
            bgr,
            gravity=gravity,
            gravity_mode=gravity_mode,
            intrinsics=intrinsics,
        )
        viable = {m: r for m, r in all_results.items()
                  if r["corners"] is not None and r["reason"] is None}
        if viable:
            chosen, chosen_result = choose_auto_mode(all_results)
            corners = chosen_result["corners"]
            state = chosen_result.get("meta", {}).get("state")
        else:
            chosen = None
            corners = None
            state = None
        alternatives = {
            m: {
                "corners":    r["corners"],
                "area_ratio": r["area_ratio"],
                "reason":     r["reason"],
                "meta":       r.get("meta"),
                "state":      r.get("meta", {}).get("state"),
            }
            for m, r in all_results.items()
        }
    else:
        # Explicit mode (no alternatives needed — caller already picked).
        _cropped, corners, chosen, mode_meta = auto_correct(
            bgr,
            mode=mode,
            gravity=gravity,
            gravity_mode=gravity_mode,
            intrinsics=intrinsics,
        )
        state = mode_meta.get("state") if mode_meta else None
    if corners is None:
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        detected = False
    else:
        detected = True
    payload = {
        "detected": detected,
        "corners": corners,
        "image_size": [w, h],
        "mode": chosen,
        "state": state,
        "intrinsics": intrinsics,
        "gravity": {
            "available": gravity is not None,
            "orientation": gravity.get("orientation") if gravity else None,
            "norm": gravity.get("norm") if gravity else None,
            "norm_deviation": gravity.get("norm_deviation") if gravity else None,
            "trusted": gravity.get("trusted") if gravity else False,
            "mode": gravity_mode,
        },
    }
    if alternatives is not None:
        payload["alternatives"] = alternatives
    if mode_meta is not None:
        payload["meta"] = mode_meta
    return payload


@app.post("/warp")
async def warp(
    file: UploadFile = File(...),
    corners: str | None = Form(None),
    state: str | None = Form(None),
    keep_aspect: bool = Form(True),
    object_aspect: float | None = Form(None),
):
    if state is None and corners is None:
        raise HTTPException(status_code=400, detail="state or corners is required")
    data = await file.read()
    bgr, icc, exif = _read_upload(data)
    h, w = bgr.shape[:2]
    if state is not None:
        try:
            correction_state = json.loads(state)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail=f"Invalid state: {error}")
        expected_intrinsics = camera_intrinsics_from_exif(exif, w, h)
        supplied_intrinsics = correction_state.get("intrinsics", {})
        for key in ("fx", "fy", "cx", "cy"):
            expected = float(expected_intrinsics[key])
            try:
                supplied = float(supplied_intrinsics[key])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"Invalid state intrinsics: {key}")
            if abs(supplied - expected) > 1e-6 * max(1.0, abs(expected)):
                raise HTTPException(status_code=422, detail="State intrinsics do not match image")
        validation = validate_correction_state(
            correction_state, [w, h], max_rotation_deg=MAX_CAMERA_ROT_DEG
        )
        if not validation["accepted"]:
            raise HTTPException(
                status_code=422,
                detail={"message": "Correction state rejected", "validation": {
                    key: value for key, value in validation.items() if key != "view"
                }},
            )
        out = warp_with_state(
            bgr, correction_state, max_rotation_deg=MAX_CAMERA_ROT_DEG
        )
    else:
        try:
            c = json.loads(corners)
            if not (isinstance(c, list) and len(c) == 4 and all(len(p) == 2 for p in c)):
                raise ValueError("expected [[x,y],[x,y],[x,y],[x,y]]")
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid corners: {error}")
        validation = validate_corner_physics(c, [w, h], object_aspect=object_aspect)
        if not validation["accepted"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Corners violate the camera-rotation constraint",
                    "validation": validation,
                },
            )
        out = warp_with_corners(bgr, c, keep_aspect=keep_aspect, object_aspect=object_aspect)
    body = _jpeg_bytes(out, icc=icc, exif=exif)
    stem = Path(file.filename or "result").stem
    headers = {"Content-Disposition": f'attachment; filename="{stem}_fixed.jpg"'}
    return StreamingResponse(io.BytesIO(body), media_type="image/jpeg", headers=headers)


@app.post("/validate")
async def validate(
    corners: str | None = Form(None),
    state: str | None = Form(None),
    image_size: str = Form(...),
    object_aspect: float | None = Form(None),
    frontend: str | None = Form(None),
):
    try:
        size = json.loads(image_size)
        if not (isinstance(size, list) and len(size) == 2):
            raise ValueError("image_size must be [w,h]")
        if state is not None:
            correction_state = json.loads(state)
            raw_result = validate_correction_state(
                correction_state, size, max_rotation_deg=MAX_CAMERA_ROT_DEG
            )
            result = {key: value for key, value in raw_result.items() if key != "view"}
        elif corners is not None:
            c = json.loads(corners)
            if not (isinstance(c, list) and len(c) == 4 and all(len(p) == 2 for p in c)):
                raise ValueError("corners must be [[x,y],[x,y],[x,y],[x,y]]")
            result = validate_corner_physics(c, size, object_aspect=object_aspect)
        else:
            raise ValueError("state or corners is required")
        frontend_result = json.loads(frontend) if frontend else None
        print(
            "PHYSICS_VALIDATE",
            json.dumps({
                "frontend": frontend_result,
                "backend": result,
                "object_aspect": object_aspect,
            }, ensure_ascii=False),
            flush=True,
        )
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid validation input: {e}")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# Static frontend last (so API routes take precedence).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
