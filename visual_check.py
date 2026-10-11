"""
visual_check.py - Visual evaluation on val, shadow, and real moon images.
Saves annotated prediction outputs to ./data/preview/predictions/.
"""

import pathlib
import cv2
from ultralytics import YOLO

PRED_DIR = pathlib.Path("data/preview/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "lunar_best.pt"
CONF_THRESH = 0.25

def main():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    val_imgs = sorted(list(pathlib.Path("data/yolo/images/val").glob("*.png")))[:10]
    shadow_imgs = sorted(list(pathlib.Path("data/yolo/images/shadow").glob("*.png")))[:5]
    real_imgs = sorted(list(pathlib.Path("Dataset/real_moon_images").glob("*.png")))[:10]

    print(f"\nRunning predictions on 10 val images...")
    for p in val_imgs:
        res = model(str(p), conf=CONF_THRESH, device="cpu", verbose=False)[0]
        out_p = PRED_DIR / f"val_{p.name}"
        res.save(filename=str(out_p))
        print(f"  Saved {out_p.name} ({len(res.boxes)} detections)")

    print(f"\nRunning predictions on 5 shadow images...")
    for p in shadow_imgs:
        res = model(str(p), conf=CONF_THRESH, device="cpu", verbose=False)[0]
        out_p = PRED_DIR / f"shadow_{p.name}"
        res.save(filename=str(out_p))
        print(f"  Saved {out_p.name} ({len(res.boxes)} detections)")

    print(f"\nRunning predictions on 10 real moon photos...")
    real_det_counts = []
    for p in real_imgs:
        res = model(str(p), conf=CONF_THRESH, device="cpu", verbose=False)[0]
        out_p = PRED_DIR / f"real_{p.name}"
        res.save(filename=str(out_p))
        real_det_counts.append(len(res.boxes))
        confs = [f"{b.conf.item():.2f}" for b in res.boxes]
        print(f"  Saved {out_p.name} ({len(res.boxes)} detections, confs: {confs})")

    print(f"\nTotal real moon detections across 10 images: {sum(real_det_counts)}")
    print(f"Average detections per real image: {sum(real_det_counts)/len(real_imgs):.1f}")


if __name__ == "__main__":
    main()
