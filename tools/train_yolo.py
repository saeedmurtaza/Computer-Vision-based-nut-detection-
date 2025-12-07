# TODO: implement
import os
import yaml
from ultralytics import YOLO

# ----------------------------------------------------
# 1. PATHS — update ONLY this one line
# ----------------------------------------------------
DATASET_DIR = r'C:\Users\User\Documents\GitHub\pungkang_camera_system\data\project-8-at-2025-11-30-16-49-788a90b5'

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
TRAIN_IMAGES = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES = os.path.join(IMAGES_DIR, "val")

DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

# ----------------------------------------------------
# 2. Create data.yaml if missing
# ----------------------------------------------------
if not os.path.exists(DATA_YAML):
    data_dict = {
        "train": TRAIN_IMAGES,
        "val": VAL_IMAGES,
        "nc": 2,
        "names": ["pass", "fail"]
    }

    with open(DATA_YAML, "w") as f:
        yaml.dump(data_dict, f)

    print(f"[INFO] data.yaml created at: {DATA_YAML}")
else:
    print("[INFO] data.yaml already exists")

# ----------------------------------------------------
# 3. Load YOLO model (fresh pretrained backbone)
# ----------------------------------------------------
print("[INFO] Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")   # IMPORTANT: fresh model, not old pt

# ----------------------------------------------------
# 4. Train
# ----------------------------------------------------
print("[INFO] Training started...")

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=1280,
    batch=4,
    device=0,       # GPU
    workers=0       # Windows fix (multiprocessing issue)
)

print("[INFO] Training completed.")

# ----------------------------------------------------
# 5. Export ONNX (optional)
# ----------------------------------------------------
print("[INFO] Exporting ONNX model (FP16)...")
model.export(format="onnx", half=True)

print("[INFO] All tasks complete!")
