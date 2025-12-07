# TODO: implement
# app/detection/post_process.py
from typing import List, Tuple, Optional
import numpy as np


BoxTuple = Tuple[int, int, int, int, float, int, str]
# (x1, y1, x2, y2, conf, cls_id, cls_name)


def filter_allowed(
    detections: List[BoxTuple],
    allowed: Optional[List[str]] = None,
    conf_thr: float = 0.0,
) -> List[BoxTuple]:
    """
    Keep only detections whose class is in `allowed` and conf >= conf_thr.
    If allowed is None, only conf_thr is applied.
    """
    if allowed is not None:
        allowed_set = set(allowed)
        return [
            d for d in detections
            if d[6] in allowed_set and d[4] >= conf_thr
        ]
    return [d for d in detections if d[4] >= conf_thr]


def select_primary(detections: List[BoxTuple]) -> Optional[BoxTuple]:
    """
    Select the detection with the highest confidence (or None if empty).
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d[4])


def to_mask(
    shape: Tuple[int, int, int],
    detections: List[BoxTuple],
    min_conf: float = 0.0,
) -> np.ndarray:
    """
    Build a binary mask from the union of the boxes with conf >= min_conf.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2, conf, _, _ in detections:
        if conf < min_conf:
            continue
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        mask[y1:y2, x1:x2] = 255
    return mask
