"""
train_lunar.py - Fast Hackathon CPU Training for Project C.R.A.T.E.R.
Trains YOLOv8n on lunar rocks dataset, evaluates on val and shadow subsets,
and exports best weights.
"""

import pathlib
import time
import shutil
import torch
from ultralytics import YOLO

DATA_YAML = pathlib.Path("data/yolo/data.yaml")
VAL_SHADOW_TXT = pathlib.Path("data/yolo/val_shadow.txt")
PROJECT = "runs"
NAME = "lunar_rocks"
IMGSZ = 416
EPOCHS = 5
BATCH = 16
DEVICE = "cpu"


def train_and_evaluate():
    print(f"=== TRAINING YOLOV8N ON LUNAR ROCKS ===")
    print(f"Device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")
    print(f"Image Size: {IMGSZ} | Epochs: {EPOCHS} | Batch: {BATCH}")

    model = YOLO("yolov8n.pt")

    t_start = time.time()
    results = model.train(
        data=str(DATA_YAML.resolve()),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        patience=10,
        workers=0,
        project=PROJECT,
        name=NAME,
        device=DEVICE,
        verbose=True
    )
    t_end = time.time()
    print(f"\nTraining completed in {t_end - t_start:.1f}s ({(t_end - t_start)/60:.2f} mins)")

    # Path to best weights
    best_weights = pathlib.Path(PROJECT) / "detect" / NAME / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback to last.pt if best.pt is not created
        best_weights = pathlib.Path(PROJECT) / "detect" / NAME / "weights" / "last.pt"
    print(f"Best weights located at: {best_weights}")

    # Validate on standard val set
    print("\n=== VALIDATING ON FULL VAL SET ===")
    trained_model = YOLO(str(best_weights))
    val_metrics = trained_model.val(
        data=str(DATA_YAML.resolve()),
        imgsz=IMGSZ,
        device=DEVICE,
        workers=0
    )

    print("\n=== VAL SET METRICS ===")
    print(f"Precision:    {val_metrics.box.mp:.4f}")
    print(f"Recall:       {val_metrics.box.mr:.4f}")
    print(f"mAP50:        {val_metrics.box.map50:.4f}")
    print(f"mAP50-95:     {val_metrics.box.map:.4f}")

    # Evaluate on shadow subset
    print("\n=== EVALUATING ON SHADOW SUBSET ===")
    # Create temporary shadow data.yaml
    shadow_yaml = pathlib.Path("data/yolo/shadow_eval.yaml")
    shadow_content = f"""path: {pathlib.Path('data/yolo').resolve().as_posix()}
train: images/train
val: val_shadow.txt
names:
  0: rock
"""
    with open(shadow_yaml, "w", encoding="utf-8") as f:
        f.write(shadow_content)

    shadow_metrics = trained_model.val(
        data=str(shadow_yaml.resolve()),
        imgsz=IMGSZ,
        device=DEVICE,
        workers=0
    )
    print("\n=== SHADOW SET METRICS ===")
    print(f"Shadow Precision: {shadow_metrics.box.mp:.4f}")
    print(f"Shadow Recall:    {shadow_metrics.box.mr:.4f}")
    print(f"Shadow mAP50:     {shadow_metrics.box.map50:.4f}")
    print(f"Shadow mAP50-95:  {shadow_metrics.box.map:.4f}")

    # Copy best weights to ./lunar_best.pt
    dest_pt = pathlib.Path("lunar_best.pt")
    shutil.copy2(best_weights, dest_pt)
    print(f"\nCopied {best_weights} -> {dest_pt}")


if __name__ == "__main__":
    train_and_evaluate()
