# TODO: implement
# app/detection/yolo_wrapper.py
from typing import List, Tuple, Optional
import os
import cv2
from ultralytics import YOLO

class RealYOLO:
    """
    Per-camera decision: if any allowed class box >= conf → REJECT, else PASS.
    Kept for compatibility with set_model() API even though our inline
    detect_classes_from_image() is used for actual decision logic.
    """
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.95,
        iou_threshold: float = 0.30,
        device: str = "cpu",
        allowed_labels: Optional[List[str]] = None,
    ):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found.")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        self.allowed = set(allowed_labels or ["Residue"])
        print(f"✅ Loaded model on {device}: {os.path.basename(model_path)}")

    def detect_once(
        self, img_bgr, imgsz: int = 640
    ) -> Tuple[List[Tuple[int, int, int, int, float, int, str]], "np.ndarray"]:
        res = self.model(img_bgr, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=imgsz, verbose=False)[0]
        detections = []
        annotated = img_bgr.copy()
        if res.boxes is not None:
            for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                conf_val = float(conf)
                cls_id = int(cls)
                cls_name = self.class_names[cls_id]
                detections.append((x1, y1, x2, y2, conf_val, cls_id, cls_name))
                if (cls_name in self.allowed) and conf_val >= self.conf_threshold:
                    color = (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"{cls_name}:{conf_val:.2f}",
                        (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
        return detections, annotated

    def decision_is_defect(self, detections):
        for _, _, _, _, conf, _, name in detections:
            if name in self.allowed and conf >= self.conf_threshold:
                return True
        return False


    # ... Test block...

# if __name__ == "__main__":
#     import numpy as np
#     import cv2
#     from pathlib import Path

#     # NOTE: We are directly using the RealYOLO class defined above.
    
#     # 1. Check if a dummy model path exists (or create a path to a known model)
#     # The Path must point to a real model file for the constructor to succeed.
#     model_path = Path("models/best.pt") 
    
#     # Assert checks if the model file is actually present
#     if not model_path.exists():
#         print(f"❌ Model file not found at {model_path}. Please adjust path or train a dummy model.")
#         exit()

#     # 2. INSTANTIATION: Call the constructor correctly with required arguments
#     #    The constructor signature is: RealYOLO(model_path, conf_threshold, iou_threshold, device, allowed_labels)
#     try:
#         yolo = RealYOLO(
#             model_path=str(model_path), 
#             conf_threshold=0.75, # Example threshold
#             device="cpu"
#         )
#     except FileNotFoundError as e:
#         print(f"❌ Initialization failed: {e}")
#         exit()
    
#     print("✅ RealYOLO instance created.")

#     # 3. TEST DETECTION LOGIC
#     # Create a dummy image (black image)
#     dummy = np.zeros((640, 640, 3), dtype=np.uint8)

#     # Run detection on the dummy image
#     dets, ann = yolo.detect_once(dummy)
    
#     # Run decision logic
#     is_defect = yolo.decision_is_defect(dets)

#     print("-" * 30)
#     print(f"Detections found: {len(dets)}")
#     print(f"Decision: Is Defect? {is_defect}")
#     # print(f"Raw Detections:", dets) # Uncomment if you want to see the full output tuple
#     print("-" * 30)
