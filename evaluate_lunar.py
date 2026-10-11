"""
evaluate_lunar.py - Evaluate best weights on full val set and shadow subset.
"""

import pathlib
import shutil
from ultralytics import YOLO

best_pt = pathlib.Path("runs/detect/runs/lunar_rocks/weights/best.pt")
print(f"Loading weights from {best_pt}")
model = YOLO(str(best_pt))

# Copy to ./lunar_best.pt
shutil.copy2(best_pt, "lunar_best.pt")
print("Saved best model to ./lunar_best.pt")

# Full Val Metrics
print("\n--- Evaluating on Full Val Set (300 images) ---")
val_res = model.val(data="data/yolo/data.yaml", imgsz=416, device="cpu", workers=0)
print("\n=== VAL SET METRICS ===")
print(f"Precision: {val_res.box.mp:.4f}")
print(f"Recall:    {val_res.box.mr:.4f}")
print(f"mAP50:     {val_res.box.map50:.4f}")
print(f"mAP50-95:  {val_res.box.map:.4f}")

# Shadow Subset Metrics
print("\n--- Evaluating on Shadow Subset (100 images) ---")
shadow_yaml = pathlib.Path("data/yolo/shadow_eval.yaml")
base_dir = pathlib.Path("data/yolo").resolve().as_posix()
shadow_yaml.write_text(f"""path: {base_dir}
train: images/train
val: val_shadow.txt
names:
  0: rock
""", encoding="utf-8")

shadow_res = model.val(data=str(shadow_yaml), imgsz=416, device="cpu", workers=0)
print("\n=== SHADOW SET METRICS ===")
print(f"Shadow Precision: {shadow_res.box.mp:.4f}")
print(f"Shadow Recall:    {shadow_res.box.mr:.4f}")
print(f"Shadow mAP50:     {shadow_res.box.map50:.4f}")
print(f"Shadow mAP50-95:  {shadow_res.box.map:.4f}")
