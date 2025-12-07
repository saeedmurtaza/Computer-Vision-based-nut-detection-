# app/routes/__init__.py
from .system import router as system_router
from .cameras import router as cameras_router
from .detect import router as detect_router
from .snapshot import router as snapshot_router
from .stream import router as stream_router

__all__ = [
    "system_router",
    "cameras_router",
    "detect_router",
    "snapshot_router",
    "stream_router",
]

