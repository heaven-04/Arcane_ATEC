"""
config.py - Centralized configuration and logging for Project C.R.A.T.E.R.
"""

import pathlib
import logging
from typing import Optional

# Base directories
BASE_DIR = pathlib.Path(__file__).parent.resolve()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Network Addresses & Ports
PROXY_LISTEN_IP = "0.0.0.0"
PROXY_LISTEN_PORT = 5000
PROXY_FORWARD_IP = "127.0.0.1"
PROXY_FORWARD_PORT = 5001
PROXY_CONTROL_IP = "127.0.0.1"
PROXY_CONTROL_PORT = 5002

BACKEND_HOST = "0.0.0.0"
BACKEND_HTTP_PORT = 8000
BACKEND_UDP_IP = "0.0.0.0"
BACKEND_UDP_PORT = 5001

# Network Degradation Parameters
DEFAULT_LOSS_RATE = 0.20
DEFAULT_LATENCY = 0.05       # 50 ms
DEFAULT_JITTER = 0.01        # 10 ms
DEFAULT_CORRUPT_RATE = 0.0   # 0% bit corruption
DEFAULT_BURST_MODE = 0       # 0 = uniform, 1 = Gilbert-Elliott

# Perception & Edge Parameters
DEFAULT_ROVER_ID = 1
DEFAULT_WEIGHTS = "lunar_best.pt"
DEFAULT_VIDEO = "demo.mp4"
CONFIDENCE_THRESHOLD = 0.35

# Protocol & Class Definitions
MAX_HAZARDS = 14             # 0..13 reserved for hazards
HEARTBEAT_CLASS = 15         # 15 reserved for heartbeat

# Backend State & Association
HAZARD_ASSOCIATION_DIST_M = 0.6  # 0.6 m spatial radius for track association
AGING_PURGE_SECONDS = 120.0
AGING_REFRESH_INTERVAL = 0.5


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Configure a logger with timestamps to both console and a log file."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        file_path = LOGS_DIR / log_file
    else:
        file_path = LOGS_DIR / f"{name}.log"

    fh = logging.FileHandler(str(file_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
