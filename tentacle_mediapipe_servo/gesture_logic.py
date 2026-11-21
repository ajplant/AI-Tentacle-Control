"""Hardware-free index-finger arrow math and continuous hold-to-run selection."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Sequence

STOP = "STOP"
LEFT = "LEFT"
RIGHT = "RIGHT"
UP = "UP"
AWAY = "AWAY"
AWAY_MODES = ("disabled", "down", "from-camera", "toward-camera")
ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Settings:
    serial_port: str = "COM3"
    baudrate: int = 115200
    camera: int = 0
    mirror: bool = True
    hand_model: str = "hand_landmarker.task"
    away_mode: str = "toward-camera"
    speed: float = 0.18
    servo_forward_signs: tuple[int, int, int] = (1, 1, 1)
    direction_tolerance_degrees: float = 25.0
    up_tolerance_degrees: float = 20.0
    depth_tolerance_degrees: float = 25.0
    maximum_frame_age_seconds: float = 0.15
    maximum_loop_gap_seconds: float = 0.18
    minimum_hand_confidence: float = 0.70
    minimum_palm_pixels: float = 30.0
    arrow_neutral_palm_ratio: float = 0.35
    minimum_arrow_pixels: float = 12.0

    def validate(self) -> "Settings":
        bounds = {
            "speed": (0.05, 0.28),
            "direction_tolerance_degrees": (10, 30),
            "up_tolerance_degrees": (10, 25),
            "depth_tolerance_degrees": (10, 30),
            "maximum_frame_age_seconds": (0.05, 0.15),
            "maximum_loop_gap_seconds": (0.08, 0.18),
            "minimum_hand_confidence": (0.65, 0.95),
            "minimum_palm_pixels": (20, 150),
            "arrow_neutral_palm_ratio": (0.20, 0.60),
            "minimum_arrow_pixels": (8, 40),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}.")
        if type(self.camera) is not int or not 0 <= self.camera <= 20:
            raise ValueError("camera must be an integer from 0 through 20.")
        if type(self.mirror) is not bool or self.baudrate != 115200:
            raise ValueError("mirror must be true/false and baudrate must be 115200.")
        if self.away_mode not in AWAY_MODES:
            raise ValueError(f"away_mode must be one of {AWAY_MODES}.")
        if len(self.servo_forward_signs) != 3 or any(type(v) is not int or v not in (-1, 1) for v in self.servo_forward_signs):
            raise ValueError("servo_forward_signs must contain three integers, each +1 or -1.")
        for name in ("serial_port", "hand_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        return self


def load_settings(path: Path, **overrides) -> Settings:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object.")
    # Old configurations remain usable. These retired duration/travel settings
    # have NO effect in any mode; communication deadlines are separate.
    for name in ("maximum_session_seconds", "maximum_pulse_seconds",
                 "maximum_signed_travel_seconds", "maximum_cumulative_motion_seconds",
                 "gesture_hold_seconds", "neutral_hold_seconds"):
        data.pop(name, None)
    unknown = set(data) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError(f"Use the supplied config_gestures.json; unexpected settings: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    if "servo_forward_signs" in data:
        data["servo_forward_signs"] = tuple(data["servo_forward_signs"])
    return Settings(**data).validate()


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def length(a):
    return math.sqrt(sum(x*x for x in a))


def angular_distance(a, b):
    return abs((a-b+180.0) % 360.0-180.0)


@dataclass(frozen=True)
class Observation:
    valid_hand: bool = False
    arrow_active: bool = False
    gesture: str = STOP
    angle_degrees: float | None = None
    depth_angle_degrees: float | None = None
    hand_label: str = ""
    reason: str = "No single reliable hand"


def is_neutral(observation: Observation) -> bool:
    """Compatibility helper: only STOP is stationary. UP is reverse."""
    return observation.valid_hand and observation.gesture == STOP


def classify_hand(points: Sequence[Sequence[float]], width: int, height: int,
                  settings: Settings, hand_label: str = "", world_points=None) -> Observation:
    """Angles are in the displayed image: right=0, up=90, left=180, down=270.

    Direction depends ONLY on the MCP-to-tip arrow, never joint straightness.
    world_points is accepted for compatibility but is not used for this math.
    Depth uses the image landmarks' approximate z scaled like image x; it is
    NOT a physical depth measurement. Positive tip-minus-base z is farther away.
    """
    if width <= 0 or height <= 0 or len(points) != 21:
        return Observation(reason="Invalid landmark count or image dimensions")
    try:
        p = [tuple(float(v) for v in row) for row in points]
        if any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in p):
            return Observation(reason="Invalid hand landmarks")
    except (TypeError, ValueError):
        return Observation(reason="Invalid hand landmarks")
    if any(not (0.01 <= x <= 0.99 and 0.01 <= y <= 0.99) for x, y, z in p):
        return Observation(reason="Hand cropped or near image edge")
    scaled = [(x*width, y*height, z*width) for x, y, z in p]
    palm_size = length(sub(scaled[0][:2], scaled[9][:2]))
    if palm_size < settings.minimum_palm_pixels:
        return Observation(reason="Hand too small; move closer")
    dx, dy, dz = sub(scaled[8], scaled[5])
    projected = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0 if projected >= 2.0 else None
    minimum_length = max(settings.minimum_arrow_pixels,
                         settings.arrow_neutral_palm_ratio*palm_size)
    if length((dx, dy, dz)) < minimum_length:
        return Observation(valid_hand=True, angle_degrees=angle, hand_label=hand_label,
                           reason="Short arrow: STOP (tip close to index knuckle)")
    depth_angle = math.degrees(math.atan2(projected, abs(dz)))
    gesture = STOP
    reason = "Between direction sectors: stop"
    if depth_angle <= settings.depth_tolerance_degrees:
        matches = ((settings.away_mode == "from-camera" and dz > 0)
                   or (settings.away_mode == "toward-camera" and dz < 0))
        if matches:
            gesture, reason = AWAY, "Depth gesture (approximate)"
        else:
            reason = "Depth direction disabled or not selected"
    elif depth_angle >= 60.0 and angle is not None:
        if angular_distance(angle, 90.0) <= settings.up_tolerance_degrees:
            gesture = UP
        elif angular_distance(angle, 180.0) <= settings.direction_tolerance_degrees:
            gesture = LEFT
        elif angular_distance(angle, 0.0) <= settings.direction_tolerance_degrees:
            gesture = RIGHT
        elif settings.away_mode == "down" and angular_distance(angle, 270.0) <= settings.direction_tolerance_degrees:
            gesture = AWAY
        if gesture == UP:
            reason = "UP arrow: reverse ALL three motors"
        elif gesture != STOP:
            reason = "Arrow direction accepted (finger may be bent)"
    return Observation(True, True, gesture, angle, depth_angle, hand_label, reason)


def speeds_for(gesture: str, settings: Settings) -> tuple[float, float, float]:
    speed = settings.speed
    raw = {STOP: ZERO, LEFT: (0.0, speed, 0.0), RIGHT: (0.0, 0.0, speed),
           UP: (-speed, -speed, -speed), AWAY: (speed, 0.0, 0.0)}
    if gesture not in raw:
        raise ValueError("Unknown gesture")
    if gesture == AWAY and settings.away_mode == "disabled":
        return ZERO
    return tuple(v*sign for v, sign in zip(raw[gesture], settings.servo_forward_signs))


class PulseGate:
    """Historical name; direct current-gesture selection, with no dwell/release gate."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.active = STOP
        self.status = "STOP: all motors stationary"

    def update(self, observation: Observation, now: float) -> str:
        if not math.isfinite(now):
            raise ValueError("Invalid timestamp")
        if not observation.valid_hand:
            self.active = STOP
            self.status = "Tracking unavailable: STOP"
            return STOP
        gesture = observation.gesture
        if gesture not in (STOP, LEFT, RIGHT, UP, AWAY):
            raise ValueError("Unknown gesture")
        if not observation.arrow_active or (gesture == AWAY and self.settings.away_mode == "disabled"):
            gesture = STOP
        self.active = gesture
        self.status = ("STOP: all motors stationary" if gesture == STOP else
                       "UP: ALL motors reverse continuously" if gesture == UP else
                       f"{gesture}: continuous; no release gesture required")
        return gesture
