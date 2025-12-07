# Punkang Camera System – Refactored O-HIVE Backend

This repository contains a **refactored, modular version** of the Punkang nut-inspection system (O-HIVE backend).  
It combines:

- **Basler industrial cameras** (via `pypylon`)
- **YOLO-based defect detection** (threads / no-threads, residue, etc.)
- **Arduino proximity sensor** trigger (via serial)
- **PLC handshake** for pass / fail signals
- A **FastAPI backend + HTML dashboard** for live monitoring and control

The goal of this refactor is to make the system **clean, testable, and easier to extend** while keeping the original production logic.

---

## Folder Structure

```text
Punkang_camera_system_refactor/
│
├── app/                     # All runtime Python code (backend + logic)
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint
│   │
│   ├── core/                # Config, logging, small helpers
│   │   ├── __init__.py
│   │   ├── config.py        # ENV vars, paths, constants, camera defaults
│   │   ├── logger.py        # Central logging setup
│   │   └── utils.py         # Tiny helpers (slugify, ROI parsing, etc.)
│   │
│   ├── camera/              # Basler camera + per-camera worker
│   │   ├── __init__.py
│   │   ├── basler_driver.py # Thin wrapper around pypylon for open/grab
│   │   ├── camera_worker.py # ProximityCameraTrigger (per-camera logic)
│   │   ├── camera_manager.py# CameraManager (multi-camera orchestration)
│   │   ├── roi.py           # crop_by_norm, mask_outside_bbox, ROI helpers
│   │   └── enhancer.py      # enhance_for_yolo, deblur, RL, beautify_preview
│   │
│   ├── detection/           # YOLO + class logic
│   │   ├── __init__.py
│   │   ├── yolo_wrapper.py  # RealYOLO, model loading on device
│   │   ├── classify.py      # detect_classes_from_image, detect_classes
│   │   └── postprocess.py   # Post-filtering, primary box, label utilities
│   │
│   ├── plc/                 # PLC client (PyXGT.LS)
│   │   ├── __init__.py
│   │   └── plc_client.py    # connection, XGB read/write, bits
│   │
│   ├── sensor/              # Arduino proximity sensor (serial)
│   │   ├── __init__.py
│   │   └── serial_sensor.py # sensor_init(), JSON reading helpers
│   │
│   ├── streams/             # MJPEG & ROI streaming
│   │   ├── __init__.py
│   │   ├── stabilizer.py    # StreamStabilizer (no black frames)
│   │   └── generators.py    # frame_generator(), frame_generator_roi()
│   │
│   ├── routes/              # API endpoints
│   │   ├── __init__.py
│   │   ├── system.py        # /api/status, /api/connect, /api/start, /api/stop, /api/reset_counters
│   │   ├── cameras.py       # /api/camera/{id}/start, stop, reconnect, param
│   │   ├── detect.py        # /api/detect, /api/camera/{id}/detect, /api/camera/{id}/model, model_upload
│   │   ├── snapshot.py      # /api/snapshot/{id}, /api/burst/{id}
│   │   └── stream.py        # /video_feed/{id}, /video_feed_roi/{id}, /ws
│   │
│   └── static/              # HTML frontend
│       └── index.html       # Web dashboard
│
├── tools/                   # Dataset tools + training code
│   ├── __init__.py
│   ├── dataset_split.py     # Train/val split helper
│   ├── train_yolo.py        # YOLO training script
│   └── metadata/
│       └── notes.json       # Category info (e.g., "Fail", "Pass")
│
├── data/                    # Raw dataset / exports (optional, for training)
│   └── project-8-at-2025-11-30-16-49-788a90b5/
│       ├── images/
│       ├── labels/
│       └── notes.json       # Can mirror tools/metadata/notes.json
│
├── models/                  # Trained YOLO weights
│   ├── best.pt              # Current training output (from tools/train_yolo.py)
│   ├── residue_defect.pt    # Optional residue model
│   └── no_threads.pt        # Optional earlier thread model
│
├── launcher/                # Desktop launcher (Windows)
│   └── launcher.py          # Small script to start the backend (+ browser)
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
