# TODO: implement
import os
import random
import shutil

DATASET_DIR = "./project-8-at-2025-11-30-16-49-788a90b5"  # your Label Studio export folder
TRAIN_RATIO = 0.8  # 80% train, 20% val

images_dir = os.path.join(DATASET_DIR, "images")
labels_dir = os.path.join(DATASET_DIR, "labels")

train_img = os.path.join(DATASET_DIR, "images/train")
val_img = os.path.join(DATASET_DIR, "images/val")
train_lbl = os.path.join(DATASET_DIR, "labels/train")
val_lbl = os.path.join(DATASET_DIR, "labels/val")

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

train_count = int(len(all_images) * TRAIN_RATIO)

for idx, img in enumerate(all_images):
    img_src = os.path.join(images_dir, img)
    lbl_src = os.path.join(labels_dir, img.replace('.jpg', '.txt').replace('.png', '.txt'))

    if idx < train_count:
        shutil.move(img_src, os.path.join(train_img, img))
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(train_lbl, os.path.basename(lbl_src)))
    else:
        shutil.move(img_src, os.path.join(val_img, img))
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(val_lbl, os.path.basename(lbl_src)))

print("Dataset successfully split into train/val!")
