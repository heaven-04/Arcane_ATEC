"""
rover_edge.py - Project C.R.A.T.E.R. Perception & Kinematics Node
Detects objects via YOLOv8, projects bottom-center pixel to spatial coords (x_m, y_m),
packs telemetry and streams via UDP to 127.0.0.1:5000.
"""

import sys
import socket
import argparse
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO

import config
from telemetry import pack_telemetry, MAX_HAZARDS

logger = config.setup_logger("rover_edge")

# Homography Matrix (Image Plane [u, v, 1] -> Spatial Ground Plane [x_m, y_m, 1])
H_MATRIX = np.array([
    [0.005, -0.001, -1.6],
    [0.000,  0.007, -2.2],
    [0.000,  0.002,  1.0]
], dtype=np.float32)


def project_pixel_to_ground(u: float, v: float, h_mat: np.ndarray):
    """Project bottom-center pixel (u, v) into spatial ground coordinates (x_m, y_m)."""
    p_img = np.array([u, v, 1.0], dtype=np.float32)
    p_world = h_mat @ p_img
    z = p_world[2] if p_world[2] != 0 else 1.0
    return float(p_world[0] / z), float(p_world[1] / z)


def parse_args():
    parser = argparse.ArgumentParser(description="Rover Edge Perception Node")
    parser.add_argument(
        "--video",
        default=config.DEFAULT_VIDEO,
        help="Path to video file or camera index (default: demo.mp4)"
    )
    parser.add_argument(
        "--rover-id",
        type=int,
        default=config.DEFAULT_ROVER_ID,
        help="Rover identifier (0-15)"
    )
    parser.add_argument(
        "--weights",
        default=config.DEFAULT_WEIGHTS,
        help="YOLO model weights path (default: lunar_best.pt)"
    )
    return parser.parse_args()


def run_rover_edge():
    args = parse_args()
    rover_id = args.rover_id
    weights_path = pathlib.Path(args.weights)

    logger.info(f"Initializing Rover Edge Node (Rover ID: {rover_id})...")
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    try:
        model = YOLO(str(weights_path))
        num_classes = len(model.names)
        logger.info(f"Loaded YOLO weights from {weights_path} with {num_classes} classes.")
        assert num_classes <= MAX_HAZARDS, (
            f"Model class count ({num_classes}) exceeds maximum allowed hazard classes ({MAX_HAZARDS})"
        )
    except Exception as exc:
        logger.error(f"Failed to load YOLO model: {exc}")
        sys.exit(1)

    # Determine video source
    video_src = args.video
    is_file = False
    if str(video_src).isdigit():
        cap = cv2.VideoCapture(int(video_src))
    else:
        v_path = pathlib.Path(video_src)
        if not v_path.exists():
            logger.error(f"Video source does not exist: {v_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(v_path.resolve()))
        is_file = True

    if not cap.isOpened():
        logger.error(f"Unable to open video source: {video_src}")
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_addr = (config.PROXY_FORWARD_IP if False else "127.0.0.1", config.PROXY_LISTEN_PORT)
    logger.info(f"Streaming telemetry packets to {target_addr[0]}:{target_addr[1]}...")

    seq_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to rewind video file.")
                        break
                else:
                    logger.warning("Camera stream interrupted.")
                    break

            results = model(frame, verbose=False, conf=config.CONFIDENCE_THRESHOLD)
            for r in results:
                for box in r.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        u = (x1 + x2) / 2.0
                        v = float(y2)

                        x_m, y_m = project_pixel_to_ground(u, v, H_MATRIX)
                        seq_num = (seq_num + 1) & 0xFFFF
                        payload = pack_telemetry(rover_id, cls_id, x_m, y_m, conf, seq_num)
                        if payload:
                            sock.sendto(payload, target_addr)
                    except Exception as err:
                        logger.error(f"Box processing error: {err}")
    except KeyboardInterrupt:
        logger.info("Stopping Rover Edge node...")
    finally:
        cap.release()
        sock.close()


if __name__ == "__main__":
    run_rover_edge()
