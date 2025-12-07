# TODO: implement
# app/streams/generators.py
import time
import asyncio
from typing import AsyncGenerator
from ..camera.camera_manager import manager
from .stabilizer import StreamStabilizer

async def frame_generator(cam_id: int) -> AsyncGenerator[bytes, None]:
    from queue import Empty

    if cam_id >= len(manager.cameras):
        yield b""
        return

    cam = manager.cameras[cam_id]
    last_yield = time.time()
    HEARTBEAT_S = 1.0
    stabilizer = StreamStabilizer()

    while True:
        try:
            try:
                img = cam.capture_image()
            except Empty:
                img = None

            if img is not None and not stabilizer.is_black(img):
                jb = stabilizer.encode_jpeg(img, quality=75)
                if jb:
                    stabilizer.last_good_jpeg = jb
                    stabilizer.last_frame_ts = time.time()
                    yield stabilizer.wrap_chunk(jb)
                    last_yield = time.time()
            else:
                if stabilizer.last_good_jpeg is not None and (time.time() - last_yield) >= HEARTBEAT_S:
                    yield stabilizer.wrap_chunk(stabilizer.last_good_jpeg)
                    last_yield = time.time()

            await asyncio.sleep(0.02)

        except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
            break
        except Exception as e:
            print(f"⚠️ frame_generator error (cam {cam_id}): {e}")
            if stabilizer.last_good_jpeg is not None:
                try:
                    yield stabilizer.wrap_chunk(stabilizer.last_good_jpeg)
                except Exception:
                    break
            await asyncio.sleep(0.1)

async def frame_generator_roi(cam_id: int) -> AsyncGenerator[bytes, None]:
    if cam_id >= len(manager.cameras):
        yield b""
        return

    cam = manager.cameras[cam_id]
    HEARTBEAT_S = 1.0
    last_yield = time.time()
    stabilizer = StreamStabilizer()

    while True:
        try:
            with cam.frame_lock:
                img = cam.latest_det_input.copy() if cam.latest_det_input is not None else None

            if img is not None and img.size > 0:
                jb = stabilizer.encode_jpeg(img, quality=80)
                if jb:
                    stabilizer.last_good_jpeg = jb
                    stabilizer.last_frame_ts = time.time()
                    yield stabilizer.wrap_chunk(jb)
                    last_yield = time.time()
                    await asyncio.sleep(0.02)
                    continue

            if stabilizer.last_good_jpeg is not None and (time.time() - last_yield) >= HEARTBEAT_S:
                yield stabilizer.wrap_chunk(stabilizer.last_good_jpeg)
                last_yield = time.time()

            await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ frame_generator_roi error (cam {cam_id}): {e}")
            await asyncio.sleep(0.1)
