# TODO: implement
# app/routes/system.py
import os
from fastapi import APIRouter

from ..camera.camera_manager import manager
from ..core.config import (
    PLC_IP,
    PLC_PORT,
    PLC_BIT,
    POLL_INTERVAL_S,
    DEAD_TIME_S,
    GAMMA_DEFAULT,
    CLAHE_CLIP_DEFAULT,
    CLAHE_TILE_DEFAULT,
    DO_NORM_DEFAULT,
    DO_RL_DEFAULT,
    RL_ITER_DEFAULT,
    RL_PSF_SIZE_DEFAULT,
    SPEC_SUPPRESS_DEFAULT,
    SPEC_THR_DEFAULT,
    SPEC_MAXAREA_DEFAULT,
    PREVIEW_DEB_DEFAULT,
    DEFAULT_ROI_NORM,
    SAVE_DIR,
    MODEL_DIR,
)

router = APIRouter()


@router.get("/api/status")
def status():
    return {
        "cameras": len(manager.cameras),
        "device": getattr(manager, "device", None),
        "save_dir": SAVE_DIR,
        "plc": {
            "ip": PLC_IP,
            "port": PLC_PORT,
            "bit": PLC_BIT,
            "poll_ms": int(POLL_INTERVAL_S * 1000),
            "dead_ms": int(DEAD_TIME_S * 1000),
        },
        "defaults": {
            "gamma": GAMMA_DEFAULT,
            "clahe_clip": CLAHE_CLIP_DEFAULT,
            "clahe_tile": CLAHE_TILE_DEFAULT,
            "norm": DO_NORM_DEFAULT,
            "rl": DO_RL_DEFAULT,
            "rl_iter": RL_ITER_DEFAULT,
            "rl_psf": RL_PSF_SIZE_DEFAULT,
            "spec_suppress": SPEC_SUPPRESS_DEFAULT,
            "spec_thr": SPEC_THR_DEFAULT,
            "spec_maxarea": SPEC_MAXAREA_DEFAULT,
            "preview_deblur": PREVIEW_DEB_DEFAULT,
            "roi_norm_default": DEFAULT_ROI_NORM,
        },
    }


@router.get("/api/models")
def list_models():
    files = []
    try:
        for fname in os.listdir(MODEL_DIR):
            if not fname.lower().endswith((".pt", ".onnx")):
                continue
            full = MODEL_DIR / fname
            if full.is_file():
                files.append({"name": fname, "path": str(full)})
    except Exception as e:
        return {"ok": False, "err": str(e), "models": []}
    return {"ok": True, "models": files}


@router.post("/api/connect")
def connect():
    manager.stop_all()
    manager.cameras.clear()
    manager.discover_and_setup()
    return {"ok": True, "cameras": len(manager.cameras)}


@router.post("/api/disconnect")
def disconnect():
    manager.stop_all()
    manager.cameras.clear()
    return {"ok": True}


@router.post("/api/start")
def start_all():
    manager.start_all()
    return {"ok": True}


@router.post("/api/stop")
def stop_all():
    manager.stop_all()
    return {"ok": True}


@router.post("/api/reset_counters")
def reset_counters():
    manager.reset_counters()
    return {"ok": True}
