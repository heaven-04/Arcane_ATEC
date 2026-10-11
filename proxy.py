"""
proxy.py - Project C.R.A.T.E.R. Degraded Network Emulator
Simulates lunar regolith link degradation:
- Non-blocking heap-based delivery queue with jitter & reordering
- Gilbert-Elliott burst loss model & uniform loss
- Bit-corruption simulation to exercise receiver CRC
- Dynamic UDP control API on 127.0.0.1:5002
"""

import socket
import threading
import time
import random
import heapq
from typing import List, Tuple

import config

logger = config.setup_logger("proxy")

# Network Settings
LISTEN_IP = config.PROXY_LISTEN_IP
LISTEN_PORT = config.PROXY_LISTEN_PORT
FORWARD_IP = config.PROXY_FORWARD_IP
FORWARD_PORT = config.PROXY_FORWARD_PORT

CONTROL_IP = config.PROXY_CONTROL_IP
CONTROL_PORT = config.PROXY_CONTROL_PORT

# Degradation State
state_lock = threading.Lock()
loss_rate = config.DEFAULT_LOSS_RATE
latency_s = config.DEFAULT_LATENCY
jitter_s = config.DEFAULT_JITTER
corrupt_rate = config.DEFAULT_CORRUPT_RATE
burst_mode = bool(config.DEFAULT_BURST_MODE)
burst_state_bad = False  # False = GOOD, True = BAD

# Scheduled Delivery Queue (Min-Heap of (release_time, counter, data, addr))
delivery_heap: List[Tuple[float, int, bytes, Tuple[str, int]]] = []
heap_cond = threading.Condition()
packet_counter = 0
MAX_QUEUE_SIZE = 1000

# Statistics
stats = {
    "received": 0,
    "forwarded": 0,
    "dropped": 0,
    "corrupted": 0,
    "reordered": 0
}


def corrupt_payload(data: bytes) -> bytes:
    """Flip a single random bit in the payload to exercise CRC validation."""
    if not data:
        return data
    ba = bytearray(data)
    byte_idx = random.randrange(len(ba))
    bit_idx = random.randrange(8)
    ba[byte_idx] ^= (1 << bit_idx)
    return bytes(ba)


def should_drop_packet() -> bool:
    """Evaluate drop decision using uniform or Gilbert-Elliott burst model."""
    global burst_state_bad
    with state_lock:
        if burst_mode:
            # Gilbert-Elliott 2-state Markov chain
            if burst_state_bad:
                # In BAD state: 90% loss, 30% chance to recover to GOOD
                if random.random() < 0.30:
                    burst_state_bad = False
                return random.random() < 0.90
            else:
                # In GOOD state: 2% background loss, 5% chance to drop into BAD
                if random.random() < 0.05:
                    burst_state_bad = True
                return random.random() < 0.02
        else:
            return random.random() < loss_rate


def delivery_worker(client_sock: socket.socket, stop_event: threading.Event):
    """Worker thread releasing scheduled packets from the delivery heap."""
    last_release_time = 0.0
    while not stop_event.is_set():
        packet_to_send = None
        with heap_cond:
            while not delivery_heap and not stop_event.is_set():
                heap_cond.wait(timeout=0.1)
            if stop_event.is_set():
                break

            now = time.time()
            release_time, _, data, addr = delivery_heap[0]
            if now >= release_time:
                heapq.heappop(delivery_heap)
                packet_to_send = (data, addr, release_time)
            else:
                wait_time = max(0.001, release_time - now)
                heap_cond.wait(timeout=wait_time)
                continue

        if packet_to_send:
            p_data, p_addr, r_time = packet_to_send
            if r_time < last_release_time:
                stats["reordered"] += 1
            last_release_time = r_time
            try:
                client_sock.sendto(p_data, p_addr)
                stats["forwarded"] += 1
            except Exception as send_err:
                logger.error(f"Failed to forward packet: {send_err}")


def control_listener(stop_event: threading.Event):
    """UDP listener on port 5002 for dynamic network degradation commands."""
    global loss_rate, latency_s, jitter_s, corrupt_rate, burst_mode
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((CONTROL_IP, CONTROL_PORT))
        sock.settimeout(1.0)
        logger.info(f"Control API active on {CONTROL_IP}:{CONTROL_PORT}")
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(512)
                cmd = data.decode("utf-8").strip().upper()
                with state_lock:
                    if cmd.startswith("LOSS:"):
                        loss_rate = max(0.0, min(1.0, float(cmd.split(":", 1)[1])))
                        logger.info(f"[CONTROL] LOSS_RATE updated: {loss_rate*100:.1f}%")
                    elif cmd.startswith("LATENCY:"):
                        latency_s = max(0.0, float(cmd.split(":", 1)[1]))
                        logger.info(f"[CONTROL] LATENCY updated: {latency_s*1000:.1f}ms")
                    elif cmd.startswith("JITTER:"):
                        jitter_s = max(0.0, float(cmd.split(":", 1)[1]))
                        logger.info(f"[CONTROL] JITTER updated: {jitter_s*1000:.1f}ms")
                    elif cmd.startswith("CORRUPT:"):
                        corrupt_rate = max(0.0, min(1.0, float(cmd.split(":", 1)[1])))
                        logger.info(f"[CONTROL] CORRUPT_RATE updated: {corrupt_rate*100:.1f}%")
                    elif cmd.startswith("BURST:"):
                        val = int(cmd.split(":", 1)[1])
                        burst_mode = bool(val)
                        logger.info(f"[CONTROL] BURST_MODE updated: {burst_mode}")
                    else:
                        logger.warning(f"Unrecognized control command: {cmd}")
            except socket.timeout:
                continue
            except Exception as parse_err:
                logger.error(f"Control command parse error: {parse_err}")
    finally:
        sock.close()


def run_proxy():
    global packet_counter
    stop_event = threading.Event()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Start worker and control threads
    worker_th = threading.Thread(target=delivery_worker, args=(client_sock, stop_event), daemon=True)
    control_th = threading.Thread(target=control_listener, args=(stop_event,), daemon=True)
    worker_th.start()
    control_th.start()

    target_addr = (FORWARD_IP, FORWARD_PORT)
    try:
        server_sock.bind((LISTEN_IP, LISTEN_PORT))
        logger.info(f"Relay online: UDP {LISTEN_IP}:{LISTEN_PORT} -> {target_addr[0]}:{target_addr[1]}")
        while True:
            data, addr = server_sock.recvfrom(1024)
            stats["received"] += 1

            if should_drop_packet():
                stats["dropped"] += 1
                continue

            with state_lock:
                c_rate = corrupt_rate
                lat = latency_s
                jit = jitter_s

            if c_rate > 0 and random.random() < c_rate:
                data = corrupt_payload(data)
                stats["corrupted"] += 1

            # Apply latency + jitter
            delay = max(0.0, lat + random.uniform(-jit, jit))
            release_time = time.time() + delay

            with heap_cond:
                if len(delivery_heap) < MAX_QUEUE_SIZE:
                    packet_counter += 1
                    heapq.heappush(delivery_heap, (release_time, packet_counter, data, target_addr))
                    heap_cond.notify()
                else:
                    stats["dropped"] += 1  # Queue overflow drop
    except KeyboardInterrupt:
        logger.info("Shutting down proxy emulator...")
    finally:
        stop_event.set()
        with heap_cond:
            heap_cond.notify_all()
        server_sock.close()
        client_sock.close()


if __name__ == "__main__":
    run_proxy()
