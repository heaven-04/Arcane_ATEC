"""
backend.py - Project C.R.A.T.E.R. Mission Control Bridge
FastAPI backend streaming live telemetry over WebSockets (/ws),
serving static dashboard at "/", and providing REST endpoints for network and goal control.
"""

import socket
import threading
import asyncio
import time
import math
import pathlib
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import config
from telemetry import unpack_telemetry

logger = config.setup_logger("backend")

# Active WebSocket connections
active_connections: List[WebSocket] = []
main_loop: asyncio.AbstractEventLoop = None

# Stateful Tracking
hazard_state: Dict[str, Dict[str, Any]] = {}

# Single Rover state starting at origin
rover_state: Dict[str, Dict[str, Any]] = {
    "1": {
        "x": 0.0, "y": 0.0, "heading": 50.2, "battery": 100.0, "state": "NOMINAL",
        "link": {
            "received": 0, "lost": 0, "loss_pct": 0.0, "duplicates": 0, "bad_crc": 0, "out_of_order": 0,
            "bytes_per_s": 16.0, "packets_per_s": 2.0, "heartbeat_age": 0.0, "json_equiv_bytes_per_s": 195.2
        }
    }
}

network_config = {
    "loss": config.DEFAULT_LOSS_RATE,
    "latency_ms": int(config.DEFAULT_LATENCY * 1000),
    "jitter_ms": int(config.DEFAULT_JITTER * 1000),
    "corrupt": config.DEFAULT_CORRUPT_RATE,
    "burst": config.DEFAULT_BURST_MODE
}

current_goal = [35.0, 45.0]
# Archimedean Spiral Trajectory (Expanding Survey within 28m Radar Radius)
planned_path = [
    [0.0, 0.0],       # Origin Base
    [1.5, 0.8],       # Loop 1
    [1.0, 3.0],
    [-2.5, 2.5],
    [-4.0, -1.0],
    [-1.5, -5.0],
    [3.0, -6.0],
    [7.0, -2.0],      # Loop 2
    [8.0, 4.0],
    [3.0, 9.0],
    [-5.0, 10.0],
    [-11.0, 3.0],
    [-11.0, -7.0],
    [-3.0, -13.0],
    [7.0, -14.0],
    [14.0, -5.0],     # Loop 3
    [16.0, 6.0],
    [8.0, 17.0],
    [-5.0, 19.0],
    [-19.0, 9.0],
    [-20.0, -8.0],
    [-8.0, -22.0],
    [8.0, -24.0],
    [22.0, -12.0],    # Outer Ring
    [25.0, 5.0],
    [15.0, 24.0],
    [-8.0, 26.0],
    [-26.0, 12.0],
    [-27.0, -10.0],
    [-12.0, -27.0],
    [10.0, -28.0],
    [28.0, 0.0]       # Final Perimeter Point
]
current_waypoint_idx = 1

state_lock = threading.Lock()
next_hazard_counter = 0


def build_full_state_message() -> dict:
    """Build canonical telemetry dictionary according to schema."""
    now = time.time()
    with state_lock:
        hazards_copy = {}
        for hid, h in list(hazard_state.items()):
            age = round(now - h["last_seen"], 1)
            state_label = "FRESH"
            display_conf = h["confidence"]
            uncertainty_m = 0.45
            if age >= 20.0:
                state_label = "STALE"
                display_conf = max(0.05, h["confidence"] * 0.4)
                uncertainty_m = min(4.5, 2.5 + (age - 20.0) * 0.05)
            elif age >= 6.0:
                state_label = "AGING"
                display_conf = max(0.1, h["confidence"] * math.exp(-(age - 6.0) / 14.0))
                uncertainty_m = min(3.0, 0.45 + (age - 6.0) * 0.15)

            hazards_copy[hid] = {
                "id": hid,
                "class": h["hazard_class"],
                "name": "rock",
                "x_m": h["x_m"],
                "y_m": h["y_m"],
                "confidence": h["confidence"],
                "display_confidence": round(display_conf, 2),
                "age": age,
                "state": state_label,
                "uncertainty_m": round(uncertainty_m, 2),
                "rover_id": h["rover_id"]
            }

        return {
            "ts": now,
            "rovers": {k: dict(v) for k, v in rover_state.items()},
            "hazards": hazards_copy,
            "path": list(planned_path),
            "network": dict(network_config)
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager starting background threads and tasks."""
    global main_loop
    main_loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    udp_thread = threading.Thread(target=udp_receiver_thread, args=(stop_event,), daemon=True)
    udp_thread.start()
    broadcast_task = asyncio.create_task(periodic_broadcast_loop(stop_event))

    logger.info(f"Mission Control Backend active on UDP :{config.BACKEND_UDP_PORT}")
    try:
        yield
    finally:
        stop_event.set()
        broadcast_task.cancel()
        logger.info("Mission Control Backend shut down cleanly.")


# L'initialisation de l'application FastAPI SE TROUVE APRÈS la fonction lifespan
app = FastAPI(title="Project C.R.A.T.E.R. Mission Control", lifespan=lifespan)

# Middleware CORS pour autoriser les requêtes OPTIONS du serveur de prévisualisation IDE (Port 55152)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint broadcasting live telemetry JSON to Mission Control."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Dashboard client connected. Total clients: {len(active_connections)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as err:
        logger.error(f"WebSocket client error: {err}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"Dashboard client disconnected. Remaining: {len(active_connections)}")


async def broadcast_state(state: dict):
    """Broadcast JSON state to all active WebSocket connections."""
    disconnected = []
    for ws in list(active_connections):
        try:
            await ws.send_json(state)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


async def periodic_broadcast_loop(stop_event: threading.Event):
    """Broadcast full telemetry state, update rover kinematics, and purge dead hazards."""
    global current_waypoint_idx
    while not stop_event.is_set():
        try:
            await asyncio.sleep(config.AGING_REFRESH_INTERVAL)
            now = time.time()
            with state_lock:
                if current_waypoint_idx < len(planned_path):
                    target_x, target_y = planned_path[current_waypoint_idx]
                    rx = rover_state["1"]["x"]
                    ry = rover_state["1"]["y"]
                    dist = math.hypot(target_x - rx, target_y - ry)

                    if dist < 0.6:
                        current_waypoint_idx += 1
                        # Auto-loop back to origin if spiral completes
                        if current_waypoint_idx >= len(planned_path):
                            current_waypoint_idx = 0
                            rover_state["1"]["x"] = 0.0
                            rover_state["1"]["y"] = 0.0
                    else:
                        speed = 1.2
                        step = speed * config.AGING_REFRESH_INTERVAL
                        theta = math.atan2(target_y - ry, target_x - rx)

                        rover_state["1"]["x"] = round(rx + step * math.cos(theta), 2)
                        rover_state["1"]["y"] = round(ry + step * math.sin(theta), 2)
                        rover_state["1"]["heading"] = round(math.degrees(theta), 1)
                        rover_state["1"]["battery"] = max(0.0, round(rover_state["1"]["battery"] - 0.02, 2))

                to_purge = [hid for hid, h in hazard_state.items() if (now - h["last_seen"]) > config.AGING_PURGE_SECONDS]
                for hid in to_purge:
                    del hazard_state[hid]
                    logger.info(f"Hazard {hid} purged after {config.AGING_PURGE_SECONDS}s.")

            payload = build_full_state_message()
            if active_connections:
                await broadcast_state(payload)
        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error(f"Broadcast loop error: {err}")


def forward_to_proxy_control(cmd: str):
    """Send command string to Proxy UDP control port 5002."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(cmd.encode("utf-8"), (config.PROXY_CONTROL_IP, config.PROXY_CONTROL_PORT))
    except Exception as exc:
        logger.error(f"Failed to send control command to proxy: {exc}")
    finally:
        sock.close()


@app.post("/api/network")
async def api_network(payload: dict):
    """Dynamically update proxy degradation parameters."""
    with state_lock:
        if "loss" in payload:
            network_config["loss"] = float(payload["loss"])
            forward_to_proxy_control(f"LOSS:{network_config['loss']}")
        if "latency_ms" in payload:
            network_config["latency_ms"] = int(payload["latency_ms"])
            forward_to_proxy_control(f"LATENCY:{network_config['latency_ms'] / 1000.0}")
        if "jitter_ms" in payload:
            network_config["jitter_ms"] = int(payload["jitter_ms"])
            forward_to_proxy_control(f"JITTER:{network_config['jitter_ms'] / 1000.0}")
        if "corrupt" in payload:
            network_config["corrupt"] = float(payload["corrupt"])
            forward_to_proxy_control(f"CORRUPT:{network_config['corrupt']}")
        if "burst" in payload:
            network_config["burst"] = int(payload["burst"])
            forward_to_proxy_control(f"BURST:{network_config['burst']}")
    return {"status": "ok", "network": network_config}


@app.post("/api/blackout")
async def api_blackout(payload: dict):
    """Simulate total communications blackout for N seconds."""
    seconds = int(payload.get("seconds", 10))
    logger.warning(f"BLACKOUT SIMULATION TRIGGERED: {seconds} seconds")

    prev_loss = network_config.get("loss", 0.20)
    with state_lock:
        for r in rover_state.values():
            r["state"] = "BLACKOUT"
    forward_to_proxy_control("LOSS:1.0")

    async def restore_after_delay():
        await asyncio.sleep(seconds)
        forward_to_proxy_control(f"LOSS:{prev_loss}")
        with state_lock:
            for r in rover_state.values():
                r["state"] = "NOMINAL"
        logger.info(f"Blackout ended. Restored LOSS={prev_loss*100:.1f}%")

    asyncio.create_task(restore_after_delay())
    return {"status": "blackout_active", "seconds": seconds}


@app.post("/api/goal")
async def api_goal(payload: dict):
    """Update destination goal coordinates [x, y]."""
    x = float(payload.get("x", 35.0))
    y = float(payload.get("y", 45.0))
    global current_goal
    with state_lock:
        current_goal = [x, y]
        if planned_path:
            planned_path[-1] = [x, y]
    logger.info(f"Updated navigation goal to [{x:.2f}, {y:.2f}]")
    return {"status": "ok", "goal": current_goal}


@app.get("/api/state")
async def api_state():
    """Return full state snapshot matching WebSocket schema."""
    return build_full_state_message()


def associate_and_update_hazard(unpacked: dict):
    """Project camera detection to global map frame & cluster within 2.5m radius."""
    global next_hazard_counter
    rover_id = str(unpacked["rover_id"])
    cls_id = unpacked["hazard_class"]
    rel_x = unpacked["x_m"]
    rel_y = unpacked["y_m"]
    conf = unpacked["confidence"]
    seq_num = unpacked["seq_num"]

    with state_lock:
        if rover_id in rover_state:
            rover_state[rover_id]["link"]["received"] += 1
            rx = rover_state[rover_id]["x"]
            ry = rover_state[rover_id]["y"]
            heading_deg = rover_state[rover_id].get("heading", 0.0)
            theta = math.radians(heading_deg)

            abs_x = rx + (rel_x * math.cos(theta) - rel_y * math.sin(theta))
            abs_y = ry + (rel_x * math.sin(theta) + rel_y * math.cos(theta))
        else:
            abs_x, abs_y = rel_x, rel_y

        cluster_x = round(abs_x / 1.5) * 1.5
        cluster_y = round(abs_y / 1.5) * 1.5

        best_id = None
        min_dist = float("inf")

        for hid, h in hazard_state.items():
            if h["hazard_class"] == cls_id:
                dist = math.hypot(h["x_m"] - cluster_x, h["y_m"] - cluster_y)
                if dist <= 2.5 and dist < min_dist:
                    min_dist = dist
                    best_id = hid

        now = time.time()
        if best_id is not None:
            h = hazard_state[best_id]
            h["x_m"] = round(0.8 * h["x_m"] + 0.2 * cluster_x, 2)
            h["y_m"] = round(0.8 * h["y_m"] + 0.2 * cluster_y, 2)
            h["confidence"] = max(h["confidence"], conf)
            h["last_seen"] = now
            h["seq_num"] = seq_num
            h["rover_id"] = int(rover_id)
        else:
            next_hazard_counter += 1
            new_id = f"H-{next_hazard_counter}"
            hazard_state[new_id] = {
                "id": new_id,
                "rover_id": int(rover_id),
                "hazard_class": cls_id,
                "x_m": round(cluster_x, 2),
                "y_m": round(cluster_y, 2),
                "confidence": conf,
                "seq_num": seq_num,
                "last_seen": now,
                "age": 0.0
            }


def udp_receiver_thread(stop_event: threading.Event):
    """Receive UDP telemetry packets, unpack, and associate hazards."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((config.BACKEND_UDP_IP, config.BACKEND_UDP_PORT))
        sock.settimeout(0.5)
        logger.info(f"UDP listener active on {config.BACKEND_UDP_IP}:{config.BACKEND_UDP_PORT}")
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(1024)
                unpacked = unpack_telemetry(data)
                if unpacked:
                    associate_and_update_hazard(unpacked)
            except socket.timeout:
                continue
            except Exception as loop_err:
                logger.error(f"UDP processing error: {loop_err}")
    finally:
        sock.close()


dashboard_dir = pathlib.Path(__file__).parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


if __name__ == "__main__":
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_HTTP_PORT)