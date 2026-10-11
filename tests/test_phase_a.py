"""
test_phase_a.py - Test suite for Phase A bug fixes and upgrades.
"""

import pathlib
import time
import cv2
import pytest

import config
from telemetry import pack_telemetry, unpack_telemetry, CLASS_NAMES, CLASS_PRIORITY, MAX_HAZARDS
from backend import associate_and_update_hazard, hazard_state


def test_requirements_file():
    """Verify requirements.txt exists and lists key dependencies."""
    req_path = pathlib.Path("requirements.txt")
    assert req_path.exists(), "requirements.txt must exist"
    content = req_path.read_text(encoding="utf-8")
    assert "ultralytics" in content
    assert "fastapi" in content
    assert "pytest" in content
    assert "psutil" in content


def test_demo_video_generated():
    """Verify demo.mp4 exists and is a valid video file."""
    video_path = pathlib.Path(config.DEFAULT_VIDEO)
    assert video_path.exists(), f"Video file {video_path} does not exist"

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Could not open demo.mp4 with cv2"
    ret, frame = cap.read()
    assert ret, "Could not read frame from demo.mp4"
    assert frame.shape[0] == 480
    assert frame.shape[1] == 720
    cap.release()


def test_telemetry_class_contract():
    """Verify hazard classes and priority definitions."""
    assert MAX_HAZARDS == 14, "MAX_HAZARDS must be 14 (reserving 15 for heartbeat)"
    assert 0 in CLASS_NAMES, "Class 0 must be in CLASS_NAMES"
    assert 0 in CLASS_PRIORITY, "Class 0 must have an assigned transmission priority"
    assert 15 in CLASS_PRIORITY, "Heartbeat class 15 must have an assigned transmission priority"

    # Packing roundtrip test
    payload = pack_telemetry(rover_id=2, hazard_class=0, x_m=15.4, y_m=-8.2, confidence=0.88, seq_num=42)
    assert len(payload) == 8, "Payload must be exactly 8 bytes"
    unpacked = unpack_telemetry(payload)
    assert unpacked is not None
    assert unpacked["rover_id"] == 2
    assert unpacked["hazard_class"] == 0
    assert abs(unpacked["x_m"] - 15.4) < 0.1
    assert abs(unpacked["y_m"] - (-8.2)) < 0.1
    assert abs(unpacked["confidence"] - 0.88) < 0.05
    assert unpacked["seq_num"] == 42


def test_spatial_hazard_association():
    """Verify detections within 0.6m update same hazard; >0.6m creates new hazard."""
    hazard_state.clear()

    # Initial detection at (10.0, 20.0)
    det1 = {
        "rover_id": 1,
        "hazard_class": 0,
        "x_m": 10.0,
        "y_m": 20.0,
        "confidence": 0.85,
        "seq_num": 1
    }
    associate_and_update_hazard(det1)
    assert len(hazard_state) == 1
    h1_key = list(hazard_state.keys())[0]
    assert h1_key.startswith("H")

    # Second detection 0.3m away -> must associate to same hazard
    det2 = {
        "rover_id": 1,
        "hazard_class": 0,
        "x_m": 10.2,
        "y_m": 20.1,
        "confidence": 0.90,
        "seq_num": 2
    }
    associate_and_update_hazard(det2)
    assert len(hazard_state) == 1, "Nearby detection (<0.6m) should update existing hazard"
    assert hazard_state[h1_key]["confidence"] == 0.90

    # Third detection 2.5m away -> must create new hazard
    det3 = {
        "rover_id": 1,
        "hazard_class": 0,
        "x_m": 12.5,
        "y_m": 20.0,
        "confidence": 0.75,
        "seq_num": 3
    }
    associate_and_update_hazard(det3)
    assert len(hazard_state) == 2, "Distant detection (>0.6m) must create a new distinct hazard"
