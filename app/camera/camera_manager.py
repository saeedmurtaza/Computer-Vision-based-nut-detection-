# TODO: implement
# app/camera/camera_manager.py
import os
from pathlib import Path
from typing import Any, List, Optional, Dict

import torch
from pypylon import pylon

from ..core.config import (
    SAVE_DIR,
    DEFAULT_MODELS,
    PER_CAMERA_CONFIG,
    DEFAULT_ROI_NORM,
    PLC_IP,
    PLC_PORT,
    PLC_BIT,
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
from ..core.utils import parse_roi_norm
from .camera_worker import ProximityCameraTrigger

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CameraManager:
    def __init__(self, save_dir: str = SAVE_DIR, model_paths: Optional[List[str]] = None, device: str = TORCH_DEVICE):
        self.save_dir = save_dir
        self.model_paths = model_paths or DEFAULT_MODELS
        self.device = device
        self.cameras: List[ProximityCameraTrigger] = []
        self.devices: List[Any] = []

    def discover_and_setup(self):
        self.cameras.clear()
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        self.devices = list(devices)
        if not devices:
            print("❌ No cameras found.")
            return False
        print(f"✅ Found {len(devices)} camera(s).")

        for idx, dev in enumerate(devices):
            dev_name = dev.GetFriendlyName()
            cam_dir = os.path.join(self.save_dir, f"cam_{idx}")
            Path(cam_dir).mkdir(parents=True, exist_ok=True)

            model_path = self.model_paths[idx] if idx < len(self.model_paths) else None
            if model_path and not os.path.exists(model_path):
                print(f"⚠️ Model path missing for cam {idx}: {model_path}. Skipping model load at start.")
                model_path = None

            cfg = PER_CAMERA_CONFIG[idx] if idx < len(PER_CAMERA_CONFIG) else {}
            roi_str = cfg.get("roi_norm", DEFAULT_ROI_NORM)
            roi_norm = parse_roi_norm(roi_str) if roi_str else None

            cam = ProximityCameraTrigger(
                save_dir=cam_dir,
                device=dev,
                device_name=cfg.get("name", dev_name),
                torch_device=self.device,
                model_path=model_path,
                plc_ip=PLC_IP,
                plc_port=PLC_PORT,
                trigger_bit=PLC_BIT,
                roi_norm=roi_norm,
                camera_params=cfg.get("camera_params"),
                gamma=cfg.get("gamma", GAMMA_DEFAULT),
                clahe_clip=cfg.get("clahe_clip", CLAHE_CLIP_DEFAULT),
                clahe_tile=cfg.get("clahe_tile", CLAHE_TILE_DEFAULT),
                do_norm=cfg.get("do_norm", DO_NORM_DEFAULT),
                do_rl=cfg.get("do_rl", DO_RL_DEFAULT),
                rl_iter=cfg.get("rl_iter", RL_ITER_DEFAULT),
                rl_psf_size=cfg.get("rl_psf_size", RL_PSF_SIZE_DEFAULT),
                spec_suppress=cfg.get("spec_suppress", SPEC_SUPPRESS_DEFAULT),
                spec_thr=cfg.get("spec_thr", SPEC_THR_DEFAULT),
                spec_maxarea=cfg.get("spec_maxarea", SPEC_MAXAREA_DEFAULT),
                preview_deblur=cfg.get("preview_deblur", PREVIEW_DEB_DEFAULT),
                conf=cfg.get("conf", 0.95),
                iou=cfg.get("iou", 0.30),
            )
            self.cameras.append(cam)

        print(f"✅ {len(self.cameras)} cameras initialized on {self.device}.")
        return True

    def start_all(self):
        for i, c in enumerate(self.cameras):
            c.start(i)

    def stop_all(self):
        for c in self.cameras:
            c.stop()

    def start_cam(self, cam_id: int) -> bool:
        if cam_id >= len(self.cameras):
            return False
        self.cameras[cam_id].start(cam_id)
        return True

    def stop_cam(self, cam_id: int) -> bool:
        if cam_id >= len(self.cameras):
            return False
        self.cameras[cam_id].stop()
        return True

    def reconnect_cam(self, cam_id: int):
        if cam_id >= len(self.devices):
            return False, "No such device index"

        if cam_id < len(self.cameras):
            try:
                self.cameras[cam_id].stop()
            except Exception:
                pass

        dev = self.devices[cam_id]
        dev_name = dev.GetFriendlyName()
        cam_dir = os.path.join(self.save_dir, f"cam_{cam_id}")
        Path(cam_dir).mkdir(parents=True, exist_ok=True)

        prev_model_path = None
        if cam_id < len(self.cameras):
            prev_model_path = getattr(self.cameras[cam_id], "model_path", None)
        if not prev_model_path and cam_id < len(self.model_paths):
            prev_model_path = self.model_paths[cam_id]

        cfg = PER_CAMERA_CONFIG[cam_id] if cam_id < len(PER_CAMERA_CONFIG) else {}
        roi_str = cfg.get("roi_norm", DEFAULT_ROI_NORM)
        roi_norm = parse_roi_norm(roi_str) if roi_str else None

        cam = ProximityCameraTrigger(
            save_dir=cam_dir,
            device=dev,
            device_name=cfg.get("name", dev_name),
            torch_device=self.device,
            model_path=prev_model_path if prev_model_path and os.path.exists(prev_model_path) else None,
            plc_ip=PLC_IP,
            plc_port=PLC_PORT,
            trigger_bit=PLC_BIT,
            roi_norm=roi_norm,
            camera_params=cfg.get("camera_params"),
            gamma=cfg.get("gamma", GAMMA_DEFAULT),
            clahe_clip=cfg.get("clahe_clip", CLAHE_CLIP_DEFAULT),
            clahe_tile=cfg.get("clahe_tile", CLAHE_TILE_DEFAULT),
            do_norm=cfg.get("do_norm", DO_NORM_DEFAULT),
            do_rl=cfg.get("do_rl", DO_RL_DEFAULT),
            rl_iter=cfg.get("rl_iter", RL_ITER_DEFAULT),
            rl_psf_size=cfg.get("rl_psf_size", RL_PSF_SIZE_DEFAULT),
            spec_suppress=cfg.get("spec_suppress", SPEC_SUPPRESS_DEFAULT),
            spec_thr=cfg.get("spec_thr", SPEC_THR_DEFAULT),
            spec_maxarea=cfg.get("spec_maxarea", SPEC_MAXAREA_DEFAULT),
            preview_deblur=cfg.get("preview_deblur", PREVIEW_DEB_DEFAULT),
            conf=cfg.get("conf", 0.95),
            iou=cfg.get("iou", 0.30),
        )

        if cam_id < len(self.cameras):
            self.cameras[cam_id] = cam
        else:
            self.cameras.append(cam)

        return True, "ok"

    def reset_counters(self):
        for c in self.cameras:
            c.total_count = 0
            c.pass_count = 0
            c.reject_count = 0
            c.last_nuts = 0
            c.last_residue = 0
            c.last_defects = 0
            c.last_state = "idle"

    def set_detect(self, enabled: bool, conf: float, iou: float):
        for c in self.cameras:
            c.detect_enabled = enabled
            c.conf = conf
            c.iou = iou
            if getattr(c, "detector", None):
                c.detector.conf_threshold = conf
                c.detector.iou_threshold = iou

    def set_detect_for_cam(self, cam_id: int, enabled: bool, conf: float, iou: float):
        if cam_id >= len(self.cameras):
            return False
        c = self.cameras[cam_id]
        c.detect_enabled = enabled
        c.conf = conf
        c.iou = iou
        if getattr(c, "detector", None):
            c.detector.conf_threshold = conf
            c.detector.iou_threshold = iou
        return True

    def set_param(self, cam_id: int, key: str, value):
        if cam_id >= len(self.cameras):
            return False
        self.cameras[cam_id].set_camera_parameters(**{key: value})
        return True

    def set_model(self, cam_id: int, path: str, conf: float, iou: float):
        if cam_id >= len(self.cameras) or not os.path.exists(path):
            return False, "Invalid camera or model path"
        cam = self.cameras[cam_id]
        try:
            if getattr(cam, "detector", None) and cam.model_path != path and self.device == "cuda":
                torch.cuda.empty_cache()
            cam.setup_yolo(path, conf, iou)
            return True, "ok"
        except Exception as e:
            return False, str(e)

# Global manager instance used by main + streams
manager = CameraManager()

 # Test block for CameraManager class
# if __name__ == "__main__":
#     """
#     sanity-test for CameraManager.

#     - If NO Basler cameras are connected:
#         * prints config
#         * tests manager.discover_and_setup()
#         * verifies that camera list is empty (expected)
    
#     - If one or more Basler cameras ARE connected:
#         * discovers cameras
#         * creates ProximityCameraTrigger instances
#         * calls setup_yolo() if a model exists
#         * starts ONE camera for 2 seconds (take_image)
#         * stops camera cleanly

#     This does NOT run the full proximity-trigger loop.
#     """

#     print("\n=== CameraManager sanity Test ===")

#     print("\n[CONFIG SUMMARY]")
#     print("SAVE_DIR:", SAVE_DIR)
#     print("DEVICE:", TORCH_DEVICE)
#     print("DEFAULT_MODELS:", DEFAULT_MODELS)
#     print("CAMERA PRESETS:", PER_CAMERA_CONFIG)

#     # 1. Instantiate manager
#     mgr = CameraManager(save_dir=SAVE_DIR, model_paths=DEFAULT_MODELS, device=TORCH_DEVICE)

#     print("\n[1] Discovering cameras …")
#     ok = mgr.discover_and_setup()

#     if not ok or len(mgr.cameras) == 0:
#         print("⚠️ No cameras detected → running config-only test.")
#         print("=== sanity test completed (no hardware) ===\n")
#     else:
#         print(f"✅ {len(mgr.cameras)} camera(s) discovered.")

#         # Test ONLY first camera to avoid triggering all cameras
#         cam0 = mgr.cameras[0]

#         print("\n[2] Testing camera setup …")
#         if not cam0.setup_camera():
#             print("❌ setup_camera() failed on cam0")
#         else:
#             print("✅ Camera hardware initialized")

#         # Test YOLO load
#         model_path = DEFAULT_MODELS[0] if DEFAULT_MODELS else None
#         if model_path and os.path.exists(model_path):
#             print("\n[3] Testing YOLO load …")
#             try:
#                 cam0.setup_yolo(model_path, conf=0.5, iou=0.4)
#                 print("✅ YOLO model loaded")
#             except Exception as e:
#                 print("⚠️ YOLO load failed (this may be OK if model not present):", e)
#         else:
#             print("⚠️ Model for cam0 not found → skipping YOLO test")

#         # Test single image capture (basically: run take_image without trigger loop)
#         print("\n[4] Testing one image capture (no trigger loop) …")
#         try:
#             # We pass sr=None because SR model is not needed for sanity test
#             cam0.take_image(0, sr=None)

#             img = cam0.capture_image()
#             if img is not None:
#                 print(f"✅ Image capture OK, shape={img.shape}")
#             else:
#                 print("⚠️ No preview image returned")
#         except Exception as e:
#             print("⚠️ Error during test-capture:", e)

#         print("\n[5] Stopping camera …")
#         try:
#             cam0.stop()
#             print("✅ Camera stopped")
#         except Exception:
#             print("⚠️ Stop failed (ignored)")

#         print("\n=== sanity test completed (hardware path) ===\n")
