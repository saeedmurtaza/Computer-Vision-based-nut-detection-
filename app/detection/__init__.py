# app/detection/__init__.py
from .yolo_wrapper import RealYOLO
from .classify import detect_classes_from_image, detect_classes

__all__ = ["RealYOLO", "detect_classes_from_image", "detect_classes"]
