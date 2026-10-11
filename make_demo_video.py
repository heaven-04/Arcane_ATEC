"""
make_demo_video.py - Generate demo.mp4 from consecutive lunar render images.
Creates ~60s video at 10 fps (600 frames) excluding corrupted/glitched renders.
"""

import pathlib
import cv2
import config

DATASET_RENDER_DIR = pathlib.Path("Dataset/images/render")
OUTPUT_VIDEO = pathlib.Path("demo.mp4")
FPS = 10.0
TARGET_SECONDS = 60
TARGET_FRAMES = int(FPS * TARGET_SECONDS)  # 600 frames

logger = config.setup_logger("make_demo_video")


def load_exclusions() -> set:
    excl = set()
    for name in ["cam_anomaly_IDs.txt", "ground_facing_IDs.txt", "mismatch_IDs.txt"]:
        p = pathlib.Path("Dataset") / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().isdigit():
                        excl.add(int(line.strip()))
    return excl


def build_video():
    logger.info("Collecting valid render frames...")
    exclusions = load_exclusions()
    logger.info(f"Loaded {len(exclusions)} exclusion IDs.")

    # Find candidate consecutive frames
    candidate_frames = []
    for p in sorted(DATASET_RENDER_DIR.glob("render*.png")):
        try:
            fid = int(p.stem.replace("render", ""))
            if fid not in exclusions:
                candidate_frames.append(p)
        except Exception:
            continue

    if not candidate_frames:
        raise RuntimeError(f"No render frames found in {DATASET_RENDER_DIR}")

    selected = candidate_frames[:TARGET_FRAMES]
    logger.info(f"Selected {len(selected)} consecutive frames for demo video.")

    # Read first image to get dimensions
    sample = cv2.imread(str(selected[0]))
    h, w, _ = sample.shape
    logger.info(f"Frame dimensions: {w}x{h} at {FPS} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (w, h))

    count = 0
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            out.write(img)
            count += 1

    out.release()
    logger.info(f"Successfully generated {OUTPUT_VIDEO} with {count} frames ({count/FPS:.1f}s).")


if __name__ == "__main__":
    build_video()
