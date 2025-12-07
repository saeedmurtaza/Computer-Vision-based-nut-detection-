# TODO: implement
# app/routes/snapshot.py
import time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..camera.camera_manager import manager
from ..camera.roi import crop_by_norm, mask_outside_bbox
from ..core.utils import slugify_model_name
from ..core.config import SAVE_DIR, ROI_MASK_OUTSIDE_SAVES

router = APIRouter()


@router.get("/api/snapshot/{cam_id}")
def snapshot(cam_id: int):
    if cam_id >= len(manager.cameras):
        return JSONResponse({"ok": False, "err": "invalid camera"}, status_code=400)

    cam = manager.cameras[cam_id]
    img = cam.capture_image()
    if img is None:
        return JSONResponse({"ok": False, "err": "no frame"}, status_code=404)

    if getattr(cam, "roi_norm", None) is not None:
        if ROI_MASK_OUTSIDE_SAVES:
            H, W = img.shape[:2]
            x, y, w, h = cam.roi_norm
            x1, y1, x2, y2 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
            img = mask_outside_bbox(img, (x1, y1, x2, y2))
        else:
            img, _ = crop_by_norm(img, cam.roi_norm)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_token = slugify_model_name(getattr(cam, "model_path", None))
    filename = f"snapshot_cam{cam_id}_{model_token}_{stamp}.jpg"
    out_path = Path(SAVE_DIR) / filename
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        return JSONResponse({"ok": False, "err": "write_failed"}, status_code=500)
    return {"ok": True, "path": str(out_path)}


@router.post("/api/burst/{cam_id}")
def burst(cam_id: int, n: int = 5):
    if cam_id >= len(manager.cameras):
        return JSONResponse({"ok": False, "err": "invalid camera"}, status_code=400)
    cam = manager.cameras[cam_id]

    saved = []
    for _ in range(max(1, int(n))):
        img = cam.capture_image()
        if img is None:
            continue

        if getattr(cam, "roi_norm", None) is not None:
            if ROI_MASK_OUTSIDE_SAVES:
                H, W = img.shape[:2]
                x, y, w, h = cam.roi_norm
                x1, y1, x2, y2 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
                img = mask_outside_bbox(img, (x1, y1, x2, y2))
            else:
                img, _ = crop_by_norm(img, cam.roi_norm)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_token = slugify_model_name(getattr(cam, "model_path", None))
        filename = f"burst_cam{cam_id}_{model_token}_{stamp}.jpg"
        out_path = Path(SAVE_DIR) / filename
        if cv2.imwrite(str(out_path), img):
            saved.append(str(out_path))
        time.sleep(0.05)
    return {"ok": True, "saved": saved}
