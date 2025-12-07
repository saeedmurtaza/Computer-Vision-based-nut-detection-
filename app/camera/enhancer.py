# TODO: implement
# app/camera/enhancer.py
from typing import Optional
import cv2
import numpy as np

def sharpen_lines(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    _, binary_lines = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_BINARY)
    return binary_lines

def enhance_lines_np(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    denoised = cv2.GaussianBlur(gray, (3,3), 0)
    blur = cv2.GaussianBlur(denoised, (5,5), 0)
    high_pass = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    edges = cv2.adaptiveThreshold(
        high_pass, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    enhanced = cv2.bitwise_or(gray, edges)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def sharpen_strong(img):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    sharp = cv2.addWeighted(img, 1.8, blur, -0.8, 0)
    return sharp

def deblur_motion(img, length=15, angle=0):
    kernel = np.zeros((length, length))
    kernel[int((length-1)/2), :] = np.ones(length)
    kernel = cv2.warpAffine(
        kernel,
        cv2.getRotationMatrix2D((length/2, length/2), angle, 1.0),
        (length, length),
    )
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def enhance_dark_areas_simple(img, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def detect_dark_motion(frame, fgbg, dark_thresh=80):
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    motion_mask = fgbg.apply(gray)
    _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    combined_mask = cv2.bitwise_and(motion_mask, dark_mask)
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    return combined_mask

def deblur_numpy(frame, sharpen_strength=1.5, blur_kernel=5):
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    sharpened = cv2.addWeighted(gray, 1 + sharpen_strength, blur, -sharpen_strength, 0)
    return sharpened

def load_sr_model(model_path="./ESPCN_x4.pb", scale=4):
    if not cv2.__dict__.get("dnn_superres"):
        raise RuntimeError("OpenCV dnn_superres module not available.")
    if not (model_path and isinstance(model_path, str)):
        raise ValueError("Invalid SR model path.")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", scale)
    print(f"Loaded ESPCN SR model, scale={scale}")
    return sr

def enhance_resolution(frame, sr):
    return sr.upsample(frame)

def enhance_for_yolo(
    img_bgr: np.ndarray,
    gamma: float,
    clahe_clip: float,
    clahe_tile: int,
    do_norm: bool,
) -> np.ndarray:
    x = img_bgr.astype(np.float32) / 255.0
    x = np.power(x, 1.0 / max(1e-6, gamma))
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_tile, clahe_tile))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    if do_norm:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out

def deblur_and_denoise(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    return sharpened

def suppress_specular(bgr: np.ndarray, thr: int, max_area: int) -> np.ndarray:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, thr, 255)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) <= max_area:
                cv2.drawContours(spot, [c], -1, 255, -1)
        if spot.max() == 0:
            return bgr
        return cv2.inpaint(bgr, spot, 3, cv2.INPAINT_TELEA)
    except Exception:
        return bgr

def _gaussian_psf(size: int) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    k = cv2.getGaussianKernel(size, 0)
    psf = k @ k.T
    psf /= max(psf.sum(), 1e-8)
    return psf

def richardson_lucy_y(bgr: np.ndarray, iterations: int, psf_size: int) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = y.astype(np.float32) / 255.0
    psf = _gaussian_psf(psf_size)
    psf_flip = cv2.flip(psf, -1)
    est = y.copy()
    for _ in range(max(1, iterations)):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
        conv[conv <= 1e-6] = 1e-6
        rel = y / conv
        est *= cv2.filter2D(rel, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est, 0.0, 1.0)
    y_out = (est * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y_out, cr, cb]), cv2.COLOR_YCrCb2BGR)

def beautify_preview(img_bgr: np.ndarray) -> np.ndarray:
    try:
        sm = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=7)
        blur = cv2.GaussianBlur(sm, (0, 0), 1.0)
        sharp = cv2.addWeighted(sm, 1.35, blur, -0.60, 0)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return img_bgr

# def draw_pass_overlay(img: np.ndarray) -> np.ndarray:
#     h, w = img.shape[:2]
#     out = img.copy()
#     cv2.rectangle(out, (3, 3), (w - 3, h - 3), (40, 200, 90), 3)
#     cv2.putText(out, "PASS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 200, 90), 2, cv2.LINE_AA)
#     return out

def draw_reject_overlay(img: np.ndarray) -> np.ndarray:
    # Border/text intentionally minimal
    return img.copy()
