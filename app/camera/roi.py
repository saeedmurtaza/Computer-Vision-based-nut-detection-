# TODO: implement
# app/camera/roi.py
import numpy as np

def crop_by_norm(img: np.ndarray, roi):
    """roi=(x,y,w,h) normalized on current image size; returns cropped image and bbox (x1,y1,x2,y2)."""
    H, W = img.shape[:2]
    x, y, w, h = roi
    x1 = max(0, min(W - 1, int(round(x * W))))
    y1 = max(0, min(H - 1, int(round(y * H))))
    x2 = max(0, min(W, int(round((x + w) * W))))
    y2 = max(0, min(H, int(round((y + h) * H))))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return img, (0, 0, W, H)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def mask_outside_bbox(img: np.ndarray, bbox):
    """Return image with everything outside bbox blacked out (same size as input)."""
    x1, y1, x2, y2 = bbox
    out = np.zeros_like(img)
    out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out

def crop_bottom_half(frame):
    h, w = frame.shape[:2]
    cropped = frame[h // 2 : h, :]
    return cropped
