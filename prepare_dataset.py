"""
prepare_dataset.py - Dataset preparation for Project C.R.A.T.E.R.
Processes Kaggle Artificial Lunar Rocky Landscape Dataset into YOLOv8 format.
"""

import pathlib
import csv
import shutil
import cv2
import numpy as np

# Configuration & Paths
DATASET_DIR = pathlib.Path("Dataset")
OUTPUT_DIR = pathlib.Path("data/yolo")
PREVIEW_DIR = pathlib.Path("data/preview")
SEED = 42

IMG_WIDTH = 720
IMG_HEIGHT = 480
TOTAL_AREA = IMG_WIDTH * IMG_HEIGHT
MIN_BOX_DIM = 12
MIN_AREA_RATIO = 0.0002  # 0.02% of 345,600 = 69.12 px
MIN_AREA = TOTAL_AREA * MIN_AREA_RATIO

# Target dataset size (CPU-optimized ~1500, GPU ~3000)
TARGET_TOTAL = 1500


def load_id_file(filepath: pathlib.Path) -> set:
    """Load IDs from text file and return set of integer IDs."""
    if not filepath.exists():
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def main():
    np.random.seed(SEED)
    print("=== STEP 1: LOADING EXCLUSIONS & METADATA ===")
    cam_anomaly_ids = load_id_file(DATASET_DIR / "cam_anomaly_IDs.txt")
    ground_facing_ids = load_id_file(DATASET_DIR / "ground_facing_IDs.txt")
    mismatch_ids = load_id_file(DATASET_DIR / "mismatch_IDs.txt")
    shadow_ids = load_id_file(DATASET_DIR / "shadow_IDs.txt")
    top200_ids = load_id_file(DATASET_DIR / "top200_largerocks_IDs.txt")

    print(f"Loaded {len(cam_anomaly_ids)} cam_anomaly IDs")
    print(f"Loaded {len(ground_facing_ids)} ground_facing IDs")
    print(f"Loaded {len(mismatch_ids)} mismatch IDs")
    print(f"Loaded {len(shadow_ids)} shadow IDs")
    print(f"Loaded {len(top200_ids)} top200_largerocks IDs")

    # Exclusions set
    excluded_ids = cam_anomaly_ids | ground_facing_ids | mismatch_ids
    print(f"Total unique excluded IDs: {len(excluded_ids)}")

    print("\n=== STEP 2: PARSING BOUNDING BOXES ===")
    boxes_by_frame = {}
    csv_path = DATASET_DIR / "bounding_boxes.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["Frame"])
            x = float(row["TopLeftCornerX"])
            y = float(row["TopLeftCornerY"])
            w = float(row["Length"])
            h = float(row["Height"])

            # Clip to image bounds
            x1 = max(0.0, min(float(IMG_WIDTH), x))
            y1 = max(0.0, min(float(IMG_HEIGHT), y))
            x2 = max(0.0, min(float(IMG_WIDTH), x + w))
            y2 = max(0.0, min(float(IMG_HEIGHT), y + h))

            w_clip = x2 - x1
            h_clip = y2 - y1
            area_clip = w_clip * h_clip

            # Filter tiny boxes (< 12px or < 0.02% area)
            if w_clip < MIN_BOX_DIM or h_clip < MIN_BOX_DIM or area_clip < MIN_AREA:
                continue

            # Normalized YOLO format (cx, cy, w, h)
            cx = (x1 + x2) / (2.0 * IMG_WIDTH)
            cy = (y1 + y2) / (2.0 * IMG_HEIGHT)
            nw = w_clip / IMG_WIDTH
            nh = h_clip / IMG_HEIGHT

            boxes_by_frame.setdefault(fid, []).append((0, cx, cy, nw, nh, (x1, y1, x2, y2)))

    print(f"Frames with valid boxes: {len(boxes_by_frame)}")

    # Filter available frames
    render_dir = DATASET_DIR / "images" / "render"
    valid_frames = []
    cam_removed = 0
    ground_removed = 0
    mismatch_removed = 0
    no_file_removed = 0

    for fid, boxes in boxes_by_frame.items():
        img_path = render_dir / f"render{fid:04d}.png"
        if not img_path.exists():
            no_file_removed += 1
            continue
        if fid in cam_anomaly_ids:
            cam_removed += 1
            continue
        if fid in ground_facing_ids:
            ground_removed += 1
            continue
        if fid in mismatch_ids:
            mismatch_removed += 1
            continue
        if len(boxes) > 0:
            valid_frames.append(fid)

    valid_frames.sort()
    print(f"\nExclusion breakdown:")
    print(f"  Removed by cam_anomaly: {cam_removed}")
    print(f"  Removed by ground_facing: {ground_removed}")
    print(f"  Removed by mismatch: {mismatch_removed}")
    print(f"  Remaining candidate frames with valid boxes: {len(valid_frames)}")

    print(f"\n=== STEP 3: SELECTING DETERMINISTIC SUBSET ({TARGET_TOTAL}) ===")
    top200_valid = [fid for fid in valid_frames if fid in top200_ids]
    non_top_valid = [fid for fid in valid_frames if fid not in top200_ids]
    print(f"Remaining top200 frames included: {len(top200_valid)}")

    needed_from_pool = TARGET_TOTAL - len(top200_valid)
    pool_indices = np.linspace(0, len(non_top_valid) - 1, needed_from_pool, dtype=int)
    selected_pool = [non_top_valid[i] for i in pool_indices]

    subset_frames = sorted(list(set(top200_valid + selected_pool)))
    print(f"Selected subset total frames: {len(subset_frames)}")

    print("\n=== STEP 4: CONTIGUOUS BLOCK TRAIN/VAL SPLIT (80/20) ===")
    total_count = len(subset_frames)
    val_count = int(total_count * 0.20)
    train_count = total_count - val_count

    # 4 contiguous val blocks of ~75 frames spread across index range
    num_val_blocks = 4
    block_size = val_count // num_val_blocks
    val_set = set()

    for b in range(num_val_blocks):
        center = int((b + 0.5) / num_val_blocks * total_count)
        start = max(0, center - block_size // 2)
        end = min(total_count, start + block_size)
        for idx in range(start, end):
            val_set.add(subset_frames[idx])

    val_frames = sorted(list(val_set))
    train_frames = sorted([f for f in subset_frames if f not in val_set])
    print(f"Train frames count: {len(train_frames)} ({len(train_frames)/total_count*100:.1f}%)")
    print(f"Val frames count: {len(val_frames)} ({len(val_frames)/total_count*100:.1f}%)")

    # Shadow evaluation subset
    val_shadow_frames = [f for f in val_frames if f in shadow_ids]
    print(f"Val frames in shadow_IDs: {len(val_shadow_frames)}")
    if len(val_shadow_frames) < 100:
        extra_shadow = [f for f in valid_frames if f in shadow_ids and f not in train_frames and f not in val_shadow_frames]
        needed_extra = min(100 - len(val_shadow_frames), len(extra_shadow))
        val_shadow_frames.extend(extra_shadow[:needed_extra])
        print(f"Supplemented shadow evaluation set to {len(val_shadow_frames)} frames.")

    print("\n=== STEP 5: WRITING DATASET FILES ===")
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    def write_split(frames, split_name):
        for fid in frames:
            img_name = f"render{fid:04d}.png"
            src_img = render_dir / img_name
            dst_img = OUTPUT_DIR / "images" / split_name / img_name
            shutil.copy2(src_img, dst_img)

            lbl_name = f"render{fid:04d}.txt"
            dst_lbl = OUTPUT_DIR / "labels" / split_name / lbl_name
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for box in boxes_by_frame[fid]:
                    cls_id, cx, cy, nw, nh, _ = box
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print("Copying train images and labels...")
    write_split(train_frames, "train")
    print("Copying val images and labels...")
    write_split(val_frames, "val")

    # Write val_shadow.txt
    shadow_txt_path = OUTPUT_DIR / "val_shadow.txt"
    with open(shadow_txt_path, "w", encoding="utf-8") as sf:
        for fid in val_shadow_frames:
            sf.write(str((OUTPUT_DIR / "images" / "val" / f"render{fid:04d}.png").resolve()) + "\n")
    print(f"Wrote {shadow_txt_path}")

    # Write data.yaml
    data_yaml_path = OUTPUT_DIR / "data.yaml"
    abs_out_dir = OUTPUT_DIR.resolve().as_posix()
    yaml_content = f"""path: {abs_out_dir}
train: images/train
val: images/val
names:
  0: rock
"""
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)
    print(f"Wrote {data_yaml_path}")

    print("\n=== STEP 6: GENERATING 8 PREVIEW IMAGES ===")
    preview_candidates = []
    # 2 shadow images
    preview_shadow = [f for f in val_frames if f in shadow_ids][:2]
    # 2 top200 images
    preview_top200 = [f for f in val_frames if f in top200_ids and f not in preview_shadow][:2]
    # 4 standard val images
    preview_std = [f for f in val_frames if f not in preview_shadow and f not in preview_top200][:4]
    preview_fids = preview_shadow + preview_top200 + preview_std

    for fid in preview_fids:
        tag = "shadow" if fid in shadow_ids else ("top200" if fid in top200_ids else "std")
        img_file = render_dir / f"render{fid:04d}.png"
        img = cv2.imread(str(img_file))
        for box in boxes_by_frame[fid]:
            _, _, _, _, _, (x1, y1, x2, y2) = box
            ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.putText(img, "rock", (ix1, max(15, iy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_name = PREVIEW_DIR / f"preview_frame{fid:04d}_{tag}.png"
        cv2.imwrite(str(out_name), img)
        print(f"Saved preview: {out_name}")

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
