import subprocess
import time
import sys
import config

logger = config.setup_logger("main")
logger.info("STARTING MISSION CONTROL PIPELINE - PROJECT C.R.A.T.E.R...")

# 1. Start FastAPI Mission Control Backend
logger.info("[1/3] Launching Mission Control Backend (HTTP :8000, UDP :5001)...")
backend_proc = subprocess.Popen([sys.executable, "backend.py"])
time.sleep(2)

# 2. Start Degraded Network Proxy
logger.info("[2/3] Launching Degraded Network Proxy (Port 5000 -> 5001)...")
proxy_proc = subprocess.Popen([sys.executable, "proxy.py"])
time.sleep(1)

# 3. Start Rover Edge Perception Node
logger.info("[3/3] Launching Rover Edge Node (YOLOv8 & demo.mp4)...")
edge_proc = subprocess.Popen([sys.executable, "rover_edge.py"])

logger.info("All pipeline services are running successfully!")
logger.info("Press Ctrl+C to terminate all services.\n")

try:
    backend_proc.wait()
except KeyboardInterrupt:
    logger.info("Shutting down Project C.R.A.T.E.R. pipeline...")
    edge_proc.terminate()
    proxy_proc.terminate()
    backend_proc.terminate()
