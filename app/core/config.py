# TODO: implement
# app/core/config.py
import os
from pathlib import Path
import torch

# ------------------------------
# Paths
# ------------------------------
APP_DIR = Path(__file__).resolve().parent.parent     # .../o_hive_clean/app
PROJECT_ROOT = APP_DIR.parent                        # .../o_hive_clean

STATIC_DIR = APP_DIR / "static" #paths needs to be changed according to new structure
MODEL_DIR = PROJECT_ROOT / "models" #.../punkang_camera_system_refactor/models

HOME_SNAPSHOT_DIR = Path.home() / "O-HIVE" / "snapshots"
HOME_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = str(HOME_SNAPSHOT_DIR)

# ------------------------------
# PLC / Trigger config
# ------------------------------
PLC_IP   = os.environ.get("OHIVE_PLC_IP",   "192.168.1.2")
PLC_PORT = int(os.environ.get("OHIVE_PLC_PORT", "2004"))
PLC_BIT  = os.environ.get("OHIVE_PLC_BIT",  "D7100")   # rising edge source

POLL_INTERVAL_S = float(os.environ.get("OHIVE_POLL_S",  "0.000001"))
DEAD_TIME_S     = float(os.environ.get("OHIVE_DEAD_S",  "0.000001"))

# Time (in ms) to wait after PLC rising edge before running detection.
# 0 = immediate capture at the rising edge.
DET_DELAY_MS    = float(os.environ.get("OHIVE_DET_DELAY_MS", "-25"))

# ------------------------------
# Image / enhancement defaults
# ------------------------------
DEFAULT_ROI_NORM = "0.250,0.8,0.28,0.95"

ROI_NORM_CAM1 = "0.60,0.25,0.33,0.60"  # adjusted default for Residue

GAMMA_DEFAULT       = float(os.environ.get("OHIVE_GAMMA", "2"))
CLAHE_CLIP_DEFAULT  = float(os.environ.get("OHIVE_CLAHE_CLIP", "3.0"))
CLAHE_TILE_DEFAULT  = int(os.environ.get("OHIVE_CLAHE_TILE", "8"))
DO_NORM_DEFAULT     = os.environ.get("OHIVE_NORM", "1") == "1"

DO_RL_DEFAULT       = os.environ.get("OHIVE_RL", "1") == "1"
RL_ITER_DEFAULT     = int(os.environ.get("OHIVE_RL_ITER", "6"))
RL_PSF_SIZE_DEFAULT = int(os.environ.get("OHIVE_RL_PSF", "3"))
PREVIEW_DEB_DEFAULT = os.environ.get("OHIVE_PREVIEW_DEB", "0") == "1"

SPEC_SUPPRESS_DEFAULT = os.environ.get("OHIVE_SPEC_SUPPRESS", "1") == "1"
SPEC_THR_DEFAULT      = int(os.environ.get("OHIVE_SPEC_THR", "245"))
SPEC_MAXAREA_DEFAULT  = int(os.environ.get("OHIVE_SPEC_MAXAREA", "200"))

# Saved-image shape options:
#   0 = tight crop (image becomes smaller, just the nut)
#   1 = keep 640×640 and black out everything outside ROI (uniform dataset size)
ROI_MASK_OUTSIDE_SAVES = os.environ.get("OHIVE_ROI_MASK_SAVE", "1") == "1"

# ------------------------------
# Per-camera presets
# ------------------------------
PER_CAMERA_CONFIG = [
    {
        # Camera 0 (Threads – your active camera)
        "name": "Cam0 Threads",
        "camera_params": {
            "ExposureTime": 300.0,
            "Gain": 20.0,
            "PixelFormat": "Mono8",
            "AcquisitionFrameRateEnable": True,
            "AcquisitionFrameRate": 10,
        },
        "roi_norm": os.environ.get("OHIVE_ROI_CAM0", DEFAULT_ROI_NORM),
        "gamma": GAMMA_DEFAULT,
        "clahe_clip": CLAHE_CLIP_DEFAULT,
        "clahe_tile": CLAHE_TILE_DEFAULT,
        "do_norm": DO_NORM_DEFAULT,
        "do_rl": DO_RL_DEFAULT,
        "rl_iter": RL_ITER_DEFAULT,
        "rl_psf_size": RL_PSF_SIZE_DEFAULT,
        "spec_suppress": SPEC_SUPPRESS_DEFAULT,
        "spec_thr": SPEC_THR_DEFAULT,
        "spec_maxarea": SPEC_MAXAREA_DEFAULT,
        "preview_deblur": PREVIEW_DEB_DEFAULT,
        "conf": 0.95,
        "iou": 0.30,
    },
    {
        # Camera 1 (e.g. Residue) – currently unused, but kept

        "name": "Cam1 Residue",
        "camera_params": {
            "ExposureTime": 2000.0,
            "Gain": 6.0,
            "PixelFormat": "Mono8",
            "AcquisitionFrameRateEnable": True,
            "AcquisitionFrameRate": 49,
        },
        "roi_norm": os.environ.get("OHIVE_ROI_CAM1", ROI_NORM_CAM1),
        "gamma": GAMMA_DEFAULT,
        "clahe_clip": CLAHE_CLIP_DEFAULT,
        "clahe_tile": CLAHE_TILE_DEFAULT,
        "do_norm": DO_NORM_DEFAULT,
        "do_rl": DO_RL_DEFAULT,
        "rl_iter": RL_ITER_DEFAULT,
        "rl_psf_size": RL_PSF_SIZE_DEFAULT,
        "spec_suppress": SPEC_SUPPRESS_DEFAULT,
        "spec_thr": SPEC_THR_DEFAULT,
        "spec_maxarea": SPEC_MAXAREA_DEFAULT,
        "preview_deblur": PREVIEW_DEB_DEFAULT,
        "conf": 0.95,
        "iou": 0.30,
    },
]

# ------------------------------
# Torch / models
# ------------------------------
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODELS = [
    str(MODEL_DIR / "best.pt"),      # cam0 //this will be replaced with best.pt later
    str(MODEL_DIR / "residue_defect.pt"),  # cam1
]

# print(f" Reached config: SAVE_DIR={SAVE_DIR}, TORCH_DEVICE={TORCH_DEVICE}")

# ------------------------------
# Settings object (for convenient imports)
# ------------------------------
class Settings:
    def __init__(self):
        self.APP_DIR = APP_DIR
        self.PROJECT_ROOT = PROJECT_ROOT
        self.STATIC_DIR = STATIC_DIR
        self.MODEL_DIR = MODEL_DIR
        self.SAVE_DIR = SAVE_DIR

        self.PLC_IP = PLC_IP
        self.PLC_PORT = PLC_PORT
        self.PLC_BIT = PLC_BIT
        self.POLL_INTERVAL_S = POLL_INTERVAL_S
        self.DEAD_TIME_S = DEAD_TIME_S
        self.DET_DELAY_MS = DET_DELAY_MS

        self.DEFAULT_ROI_NORM = DEFAULT_ROI_NORM
        self.ROI_NORM_CAM1 = ROI_NORM_CAM1

        self.GAMMA_DEFAULT = GAMMA_DEFAULT
        self.CLAHE_CLIP_DEFAULT = CLAHE_CLIP_DEFAULT
        self.CLAHE_TILE_DEFAULT = CLAHE_TILE_DEFAULT
        self.DO_NORM_DEFAULT = DO_NORM_DEFAULT
        self.DO_RL_DEFAULT = DO_RL_DEFAULT
        self.RL_ITER_DEFAULT = RL_ITER_DEFAULT
        self.RL_PSF_SIZE_DEFAULT = RL_PSF_SIZE_DEFAULT
        self.PREVIEW_DEB_DEFAULT = PREVIEW_DEB_DEFAULT
        self.SPEC_SUPPRESS_DEFAULT = SPEC_SUPPRESS_DEFAULT
        self.SPEC_THR_DEFAULT = SPEC_THR_DEFAULT
        self.SPEC_MAXAREA_DEFAULT = SPEC_MAXAREA_DEFAULT
        self.ROI_MASK_OUTSIDE_SAVES = ROI_MASK_OUTSIDE_SAVES

        self.PER_CAMERA_CONFIG = PER_CAMERA_CONFIG
        self.TORCH_DEVICE = TORCH_DEVICE
        self.DEFAULT_MODELS = DEFAULT_MODELS

settings = Settings()

