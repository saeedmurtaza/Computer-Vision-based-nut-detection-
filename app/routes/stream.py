# TODO: implement
# app/routes/stream.py
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..streams.generators import frame_generator, frame_generator_roi
from .cameras import _build_cameras_payload

router = APIRouter()


@router.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    return StreamingResponse(
        frame_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video_feed_roi/{cam_id}")
async def video_feed_roi(cam_id: int):
    return StreamingResponse(
        frame_generator_roi(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = _build_cameras_payload()
            await ws.send_json(data)
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        print(f"WS closed: {e}")
