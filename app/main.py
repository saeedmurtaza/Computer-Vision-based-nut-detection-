# app/main.py — O-HIVE (Windows, Basler + PLC trigger, modular)
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .core.config import STATIC_DIR, DEFAULT_MODELS
from .camera.camera_manager import manager
from .routes import (
    system_router,
    cameras_router,
    detect_router,
    snapshot_router,
    stream_router,
)

# ---------- FastAPI app ----------
app = FastAPI(title="O-HIVE AI Detection System (Windows, Basler + PLC trigger)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent     # .../o_hive_clean/app
STATIC_DIR = APP_DIR / "static"              # .../o_hive_clean/app/static

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------- Routers ----------
app.include_router(system_router)
app.include_router(cameras_router)
app.include_router(detect_router)
app.include_router(snapshot_router)
app.include_router(stream_router)


# ---------- Lifecycle ----------
@app.on_event("startup")
def _startup():
    from .core.config import DEFAULT_MODELS  # keep import local to avoid cycles

    manager.discover_and_setup()
    for cam_idx, mp in enumerate(DEFAULT_MODELS[:2]):
        if os.path.exists(mp) and cam_idx < len(manager.cameras):
            try:
                ok, err = manager.set_model(cam_idx, mp, conf=0.95, iou=0.30)
                if not ok:
                    print(f"⚠️ Could not apply default model to cam {cam_idx}: {err}")
                else:
                    print(f"✅ Default model applied to cam {cam_idx}: {Path(mp).name}")
            except Exception as e:
                print(f"⚠️ Could not apply default model to cam {cam_idx}: {e}")
    manager.start_all()
    print("🚀 O-HIVE backend running — open http://127.0.0.1:8000")


@app.on_event("shutdown")
def _shutdown():
    manager.stop_all()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
