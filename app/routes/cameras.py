# TODO: implement
# app/routes/cameras.py
import os
from fastapi import APIRouter

from ..camera.camera_manager import manager
from ..core.config import (
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
)


router = APIRouter()


def _build_cameras_payload():
    cams = []
    for i, c in enumerate(manager.cameras):
        roi_norm_str = ""
        if getattr(c, "roi_norm", None) is not None:
            x, y, w, h = c.roi_norm
            roi_norm_str = f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}"

        cp = getattr(c, "camera_params", {}) or {}
        exposure = cp.get("ExposureTime", None)
        gain = cp.get("Gain", None)
        fps = cp.get("AcquisitionFrameRate", None)

        cams.append(
            {
                "id": i,
                "name": getattr(c, "device_name", f"Cam{i}"),
                "serial": getattr(c.device, "GetSerialNumber", lambda: "Unknown")(),
                "status": "active" if getattr(c, "capture_on", False) else "inactive",
                "total": getattr(c, "total_count", 0),
                "pass": getattr(c, "pass_count", 0),
                "reject": getattr(c, "reject_count", 0),
                "model": os.path.basename(getattr(c, "model_path", "") or "") or "Not Selected",
                "neg_label": getattr(c, "neg_label", "Residue"),
                "last_nuts": getattr(c, "last_nuts", 0),
                "last_residue": getattr(c, "last_residue", 0),
                "last_defects": getattr(c, "last_defects", 0),
                "last_state": getattr(c, "last_state", "idle"),
                "last_trigger_ts": getattr(c, "last_trigger_ts", ""),
                "last_save_path": getattr(c, "last_save_path", ""),
                "trigger_count": getattr(c, "trigger_count", 0),
                "conf": getattr(c, "conf", 0.95),
                "iou": getattr(c, "iou", 0.30),
                "roi_norm": roi_norm_str,
                "gamma": getattr(c, "gamma", GAMMA_DEFAULT),
                "clahe_clip": getattr(c, "clahe_clip", CLAHE_CLIP_DEFAULT),
                "clahe_tile": getattr(c, "clahe_tile", CLAHE_TILE_DEFAULT),
                "do_norm": getattr(c, "do_norm", DO_NORM_DEFAULT),
                "do_rl": getattr(c, "do_rl", DO_RL_DEFAULT),
                "rl_iter": getattr(c, "rl_iter", RL_ITER_DEFAULT),
                "rl_psf_size": getattr(c, "rl_psf_size", RL_PSF_SIZE_DEFAULT),
                "spec_suppress": getattr(c, "spec_suppress", SPEC_SUPPRESS_DEFAULT),
                "spec_thr": getattr(c, "spec_thr", SPEC_THR_DEFAULT),
                "spec_maxarea": getattr(c, "spec_maxarea", SPEC_MAXAREA_DEFAULT),
                "preview_deblur": getattr(c, "preview_deblur", PREVIEW_DEB_DEFAULT),
                "exposure": exposure,
                "gain": gain,
                "fps": fps,
            }
        )
    return {"cameras": cams}


@router.get("/api/cameras")
def get_cameras():
    return _build_cameras_payload()


@router.post("/api/camera/{cam_id}/start")
def start_cam(cam_id: int):
    ok = manager.start_cam(cam_id)
    return {"ok": ok}


@router.post("/api/camera/{cam_id}/stop")
def stop_cam(cam_id: int):
    ok = manager.stop_cam(cam_id)
    return {"ok": ok}


@router.post("/api/camera/{cam_id}/reconnect")
def reconnect_cam(cam_id: int):
    ok, err = manager.reconnect_cam(cam_id)
    return {"ok": ok, "err": None if ok else err}


@router.post("/api/camera/{cam_id}/param")
async def set_param(cam_id: int, payload: dict):
    key = payload.get("key")
    value = payload.get("value")
    ok = manager.set_param(cam_id, key, value)
    return {"ok": ok}
