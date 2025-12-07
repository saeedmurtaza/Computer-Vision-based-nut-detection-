# TODO: implement
# app/routes/detect.py
from typing import Any, Dict
import os
from fastapi import APIRouter, UploadFile, File, Form

from ..camera.camera_manager import manager
from ..core.config import MODEL_DIR

router = APIRouter()


@router.post("/api/detect")
async def set_detection(payload: Dict[str, Any]):
    enabled = bool(payload.get("enabled", True))
    conf = float(payload.get("conf", 0.95))
    iou = float(payload.get("iou", 0.30))
    manager.set_detect(enabled, conf, iou)
    return {"ok": True}


@router.post("/api/camera/{cam_id}/detect")
async def set_detection_cam(cam_id: int, payload: Dict[str, Any]):
    enabled = bool(payload.get("enabled", True))
    conf = float(payload.get("conf", 0.95))
    iou = float(payload.get("iou", 0.30))
    ok = manager.set_detect_for_cam(cam_id, enabled, conf, iou)
    return {"ok": ok}


@router.post("/api/camera/{cam_id}/model")
async def set_model(cam_id: int, payload: Dict[str, Any]):
    path = payload.get("path")
    conf = float(payload.get("conf", 0.95))
    iou = float(payload.get("iou", 0.30))
    ok, err = manager.set_model(cam_id, path, conf, iou)
    return {"ok": ok, "err": None if ok else err}


@router.post("/api/camera/{cam_id}/model_upload")
async def upload_model(
    cam_id: int,
    file: UploadFile = File(...),
    conf: float = Form(0.95),
    iou: float = Form(0.30),
):
    dst = MODEL_DIR / file.filename
    with open(dst, "wb") as f:
        f.write(await file.read())
    ok, err = manager.set_model(cam_id, str(dst), conf, iou)
    return {"ok": ok, "path": str(dst), "err": None if ok else err}
