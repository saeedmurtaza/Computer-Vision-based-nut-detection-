# TODO: implement
# app/core/utils.py
import os
from typing import Optional, Tuple

def slugify_model_name(path_or_name: Optional[str]) -> str:
    if not path_or_name:
        return "no_model"
    name = os.path.basename(path_or_name)
    name = os.path.splitext(name)[0]
    name = name.strip().replace(" ", "_")
    name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return name.lower() or "no_model"

def parse_roi_norm(s: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        xs = [float(v.strip()) for v in s.split(",")]
        if len(xs) != 4:
            return None
        x, y, w, h = xs
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        return (x, y, w, h)
    except Exception:
        return None

# --- Corrected Test Execution ---

# 1. Test slugify_model_name
# model_path = " models/best model.pt "
# slugified_name = slugify_model_name(model_path)
# print(f"Input: {model_path}")
# print(f"Output (slugify_model_name): {slugified_name}")

# # 2. Test parse_roi_norm
# roi_string = "0.1, 0.2, 0.5, 0.5"
# parsed_roi = parse_roi_norm(roi_string)
# print(f"\nInput: {roi_string}")
# print(f"Output (parse_roi_norm): {parsed_roi}")