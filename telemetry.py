"""
telemetry.py - Project C.R.A.T.E.R. Data Contract
Ultra-compact 8-byte binary telemetry protocol over UDP.
Format: !BhhBH
- B: Header / Class (High nibble: rover_id, Low nibble: hazard_class)
     0..13: Hazard classes
     15: Heartbeat packet
- h: X coordinate (decimeters, signed 16-bit int)
- h: Y coordinate (decimeters, signed 16-bit int)
- B: Confidence (0-100%, unsigned 8-bit int)
- H: Sequence Number (unsigned 16-bit int)
"""

import struct
from typing import Optional, Dict, Any

TELEMETRY_FORMAT = "!BhhBH"
PAYLOAD_SIZE = struct.calcsize(TELEMETRY_FORMAT)  # Exactly 8 bytes

# Trained detection classes (0..13 reserved for hazards, 15 for heartbeat)
CLASS_NAMES = {0: "rock"}
HEARTBEAT_CLASS = 15
MAX_HAZARDS = 14

# Transmission priority: higher number = higher priority for bandwidth limiter
CLASS_PRIORITY = {
    0: 2,               # Rock hazard: high priority
    HEARTBEAT_CLASS: 1  # Heartbeat / telemetry: baseline priority
}


def pack_telemetry(
    rover_id: int,
    hazard_class: int,
    x_m: float,
    y_m: float,
    confidence: float,
    seq_num: int
) -> bytes:
    """Pack telemetry fields into an 8-byte binary payload."""
    try:
        # Encode rover_id (4 bits, 0-15) and hazard_class (4 bits, 0-15)
        header_byte = ((int(rover_id) & 0x0F) << 4) | (int(hazard_class) & 0x0F)

        # Scale spatial coordinates from meters to decimeters (multiply by 10)
        x_dm = int(round(float(x_m) * 10))
        y_dm = int(round(float(y_m) * 10))

        # Clamp coordinates to signed 16-bit integer range (-32768 to 32767)
        x_dm = max(-32768, min(32767, x_dm))
        y_dm = max(-32768, min(32767, y_dm))

        # Convert confidence to 0-100 or 0-255 byte
        if isinstance(confidence, float) and confidence <= 1.0:
            conf_val = int(round(confidence * 100))
        else:
            conf_val = int(confidence)
        conf_byte = max(0, min(255, conf_val))

        # Sequence number packed as unsigned 16-bit integer
        seq_val = int(seq_num) & 0xFFFF

        return struct.pack(TELEMETRY_FORMAT, header_byte, x_dm, y_dm, conf_byte, seq_val)
    except Exception as exc:
        print(f"[Telemetry] Pack error: {exc}")
        return b""


def unpack_telemetry(payload: bytes) -> Optional[Dict[str, Any]]:
    """Unpack an 8-byte binary payload into a dictionary."""
    try:
        if not payload or len(payload) != PAYLOAD_SIZE:
            return None

        header_byte, x_dm, y_dm, conf_byte, seq_val = struct.unpack(
            TELEMETRY_FORMAT, payload
        )

        rover_id = (header_byte >> 4) & 0x0F
        hazard_class = header_byte & 0x0F
        x_m = round(x_dm / 10.0, 2)
        y_m = round(y_dm / 10.0, 2)
        confidence = round(conf_byte / 100.0, 2)

        return {
            "rover_id": rover_id,
            "hazard_class": hazard_class,
            "x_m": x_m,
            "y_m": y_m,
            "confidence": confidence,
            "seq_num": seq_val
        }
    except Exception as exc:
        print(f"[Telemetry] Unpack error: {exc}")
        return None
