"""
export_and_benchmark.py - Export best weights to ONNX and benchmark PyTorch vs ONNX.
"""

import pathlib
import time
import numpy as np
import cv2
from ultralytics import YOLO

PT_MODEL_PATH = pathlib.Path("lunar_best.pt")
ONNX_MODEL_PATH = pathlib.Path("lunar_best.onnx")
NUM_WARMUP = 10
NUM_FRAMES = 100
IMGSZ = 640

def main():
    print(f"=== EXPORTING {PT_MODEL_PATH} TO ONNX (imgsz={IMGSZ}) ===")
    model = YOLO(str(PT_MODEL_PATH))
    export_path = model.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True)
    print(f"Exported to: {export_path}")

    # Generate dummy frame for benchmark (matching image shape)
    dummy_frame = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)

    print(f"\n=== BENCHMARKING PYTORCH (.pt) OVER {NUM_FRAMES} FRAMES (CPU) ===")
    pt_model = YOLO(str(PT_MODEL_PATH))
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = pt_model(dummy_frame, imgsz=IMGSZ, device="cpu", verbose=False)

    pt_times = []
    for _ in range(NUM_FRAMES):
        t0 = time.perf_counter()
        _ = pt_model(dummy_frame, imgsz=IMGSZ, device="cpu", verbose=False)
        t1 = time.perf_counter()
        pt_times.append((t1 - t0) * 1000.0) # ms

    pt_avg_ms = np.mean(pt_times)
    pt_fps = 1000.0 / pt_avg_ms
    print(f"PyTorch (.pt) Device: CPU")
    print(f"PyTorch (.pt) Latency: {pt_avg_ms:.2f} ms/frame")
    print(f"PyTorch (.pt) Throughput: {pt_fps:.2f} FPS")

    print(f"\n=== BENCHMARKING ONNX (.onnx) OVER {NUM_FRAMES} FRAMES (CPU) ===")
    onnx_model = YOLO(str(ONNX_MODEL_PATH))
    # Warmup
    for _ in range(NUM_WARMUP):
        _ = onnx_model(dummy_frame, imgsz=IMGSZ, device="cpu", verbose=False)

    onnx_times = []
    for _ in range(NUM_FRAMES):
        t0 = time.perf_counter()
        _ = onnx_model(dummy_frame, imgsz=IMGSZ, device="cpu", verbose=False)
        t1 = time.perf_counter()
        onnx_times.append((t1 - t0) * 1000.0) # ms

    onnx_avg_ms = np.mean(onnx_times)
    onnx_fps = 1000.0 / onnx_avg_ms
    print(f"ONNX (.onnx) Device: CPU")
    print(f"ONNX (.onnx) Latency: {onnx_avg_ms:.2f} ms/frame")
    print(f"ONNX (.onnx) Throughput: {onnx_fps:.2f} FPS")


if __name__ == "__main__":
    main()
