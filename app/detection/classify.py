# TODO: implement
# app/detection/classify.py
import os
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

"""
Inline detection model used by detect_classes_from_image / detect_classes.

We can override via OHIVE_INLINE_MODEL, otherwise it keeps our original path.
"""

DEFAULT_INLINE_MODEL = os.environ.get(
    "OHIVE_INLINE_MODEL",
    r"C:\Users\MurtazaSaeed\Documents\Punkang_camera_system_refactor\models\best.pt",
) #the path needs to be updated according to newly trained model on site

if not os.path.exists(DEFAULT_INLINE_MODEL):
    print(f"⚠️ Inline YOLO model path not found: {DEFAULT_INLINE_MODEL}")
model = YOLO(DEFAULT_INLINE_MODEL)
model_classes = list(model.names.values())
print("----------------------------model class---------------------:", model_classes)


def detect_classes_from_image(image: np.ndarray, conf: float = 0.4, iou: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Detects classes in a NumPy image (HWC, BGR or grayscale).
    Returns list of tuples: [(class_name, confidence), ...]
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    print("--------------------reached here for detection---------------------")
    results = model(image, conf=conf, iou=iou, save=True, agnostic_nms=True)
    print("----------------MODEL CLASSES INSIDE DETECTION FUNC:", results)
    detected_classes: List[Tuple[str, float]] = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        class_name = results[0].names[cls_id]
        detected_classes.append((class_name, conf_score))

    print(detected_classes)
    return detected_classes


def detect_classes(image_path: str, conf: float = 0.4, iou: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Runs YOLO detection on an image path with configurable thresholds.
    Returns a list of (class_name, confidence).
    """
    print("--------------------reached here for detection---------------------")
    results = model(image_path, conf=conf, iou=iou, agnostic_nms=True)
    print("--------------------Not reached here for detection---------------------")
    print("----------------MODEL CLASSES:", results)
    detected: List[Tuple[str, float]] = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            class_name = model.names.get(cls_id, f"unknown_{cls_id}")
            detected.append((class_name, conf_score))

    return detected
