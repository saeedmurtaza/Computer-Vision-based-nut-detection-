# TODO: implement
# app/streams/stabilizer.py
from typing import Optional
import cv2
import numpy as np

class StreamStabilizer:
    def __init__(self):
        self.last_good_jpeg: Optional[bytes] = None
        self.last_frame_ts: float = 0.0

    def is_black(self, img: Optional[np.ndarray], thr_mean: float = 2.0) -> bool:
        if img is None or img.size == 0:
            return True
        return float(img.mean()) < thr_mean

    def encode_jpeg(self, img: np.ndarray, quality: int = 75) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def wrap_chunk(self, jpeg_bytes: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
