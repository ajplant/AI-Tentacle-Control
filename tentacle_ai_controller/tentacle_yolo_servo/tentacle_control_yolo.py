"""Single-webcam, closed-loop controller for a three-tendon tentacle.

One unmirrored webcam frame supplies both index-finger targets and YOLO pose
tentacle feedback. Hand and tentacle must both be visible. A short calibration
learns a local 2x3 visual Jacobian for three continuous-rotation servos.

Continuous-rotation servos are velocity devices: this program never knows their
absolute shaft angles. Camera feedback, command-time budgets, and firmware
watchdogs reduce risk, but physical travel stops or slip clutches are still
required.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import queue
import threading
import time
from typing import Any

import numpy as np
from yolo_pose_tracker import YoloTentacleTracker, TentacleObservation, tracker_signature


DEFAULT_CONFIG: dict[str, Any] = {
    "serial_port": "COM3",
    "baudrate": 115200,
    "camera": 0,
    "hand_model": "hand_landmarker.task",
    "visual_model": "visual_model_yolo.json",
    "yolo_model": "best.pt",
    "yolo_imgsz": 640,
    "yolo_device": "cpu",
    "yolo_box_confidence": 0.50,
    "yolo_keypoint_confidence": 0.60,
    "minimum_base_axis_pixels": 12.0,
    "keypoint_border_pixels": 5.0,
    "maximum_keypoint_jump_fraction": 0.10,
    "maximum_axis_jump_degrees": 15.0,
    "maximum_frame_age_seconds": 0.15,
    "control_hz": 20.0,
    "gain": 0.8,
    "damping": 0.08,
    "nullspace_gain": 0.25,
    "maximum_speed": 0.28,
    "maximum_acceleration": 1.2,
    "error_deadband": 0.025,
    "stable_frames_before_arm": 12,
    "maximum_signed_travel_seconds": 0.30,
    "maximum_cumulative_motion_seconds": 1.20,
    "maximum_session_seconds": 30.0,
    "target_lateral_gain": 0.75,
    "target_axial_gain": 0.80,
    "target_axial_offset": 0.20,
    "target_smoothing": 0.20,
    "calibration_speed": 0.18,
    "calibration_pulse_seconds": 0.18,
    "calibration_repeats": 2,
    "calibration_settle_seconds": 0.35,
    "maximum_loop_gap_seconds": 0.18,
    "maximum_target_offset": 0.30,
}


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    settings = dict(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as source:
        loaded = json.load(source)
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a JSON object.")
    if "aruco_dictionary" in loaded:
        raise ValueError("Use config_yolo.json, not the old ArUco configuration.")
    if "hand_camera" in loaded or "tentacle_camera" in loaded or "mirror_hand_camera" in loaded:
        raise ValueError(
            'Old two-camera config: use the new config.json with "camera": 0, '
            'remove hand_camera/tentacle_camera/mirror_hand_camera, and retrain.'
        )
    settings.update(loaded)
    validate_config(settings)
    return settings, path.parent


def validate_config(settings: dict[str, Any]) -> None:
    bounds = {
        "control_hz": (10.0, 30.0), "gain": (0.01, 2.0),
        "damping": (0.01, 2.0), "nullspace_gain": (0.0, 1.0),
        "maximum_speed": (0.01, 0.35), "maximum_acceleration": (0.01, 3.0),
        "error_deadband": (0.001, 0.2), "target_smoothing": (0.01, 1.0),
        "maximum_signed_travel_seconds": (0.01, 1.0),
        "maximum_cumulative_motion_seconds": (0.1, 5.0),
        "maximum_session_seconds": (1.0, 120.0),
        "maximum_loop_gap_seconds": (0.08, 0.18),
        "calibration_speed": (0.05, 0.30),
        "calibration_pulse_seconds": (0.05, 0.30),
        "calibration_settle_seconds": (0.2, 2.0),
        "maximum_target_offset": (0.01, 0.5),
        "target_lateral_gain": (-1.0, 1.0),
        "target_axial_gain": (-1.0, 1.0),
        "target_axial_offset": (-0.5, 1.0),
    }
    for name, (low, high) in bounds.items():
        value = float(settings[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and between {low} and {high}.")
    for name, low, high in (
        ("stable_frames_before_arm", 5, 100), ("calibration_repeats", 1, 3),
        ("camera", 0, 20),
    ):
        value = settings[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}.")
    if settings["baudrate"] != 115200:
        raise ValueError("This firmware requires baudrate 115200.")
    for key in ("serial_port", "hand_model", "visual_model", "yolo_model", "yolo_device"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    if type(settings["yolo_imgsz"]) is not int or settings["yolo_imgsz"] not in (320, 416, 512, 640, 768, 960):
        raise ValueError("yolo_imgsz must be 320, 416, 512, 640, 768, or 960.")
    for name, low, high in (
        ("yolo_box_confidence", .25, .95), ("yolo_keypoint_confidence", .50, .95),
        ("minimum_base_axis_pixels", 8., 200.), ("keypoint_border_pixels", 2., 50.),
        ("maximum_keypoint_jump_fraction", .01, .15),
        ("maximum_axis_jump_degrees", 2., 20.),
        ("maximum_frame_age_seconds", .05, .15),
    ):
        value = float(settings[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and between {low} and {high}.")


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-9:
        return None
    return vector / length


class Camera:
    def __init__(self, cv2_module: Any, index: int, name: str):
        self.cv2 = cv2_module
        self.name = name
        self.capture = cv2_module.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open {name} camera index {index}.")
        self.capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.has_frame = threading.Event()
        self.first_read = True
        self.last_timestamp: float | None = None
        self.failure: str | None = None
        self.startup_deadline = time.monotonic() + 5.0
        self.worker = threading.Thread(target=self._capture, daemon=True)
        self.worker.start()

    def _capture(self) -> None:
        while not self.closed.is_set():
            ok, frame = self.capture.read()
            if not ok:
                if not self.has_frame.is_set() and time.monotonic() < self.startup_deadline:
                    self.closed.wait(0.05)
                    continue
                self.failure = f"{self.name} camera stopped returning frames."
                return
            self.has_frame.set()
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait((time.monotonic(), frame))
            except queue.Full:
                pass

    def read(self) -> np.ndarray:
        if self.failure:
            raise RuntimeError(self.failure)
        try:
            timestamp, frame = self.frames.get(timeout=5.0 if self.first_read else 0.15)
        except queue.Empty as exc:
            raise RuntimeError(self.failure or f"{self.name} camera timed out.") from exc
        if time.monotonic() - timestamp > 0.15:
            raise RuntimeError(f"{self.name} camera frame is stale.")
        self.first_read = False
        self.last_timestamp = timestamp
        return frame

    def assert_fresh(self, settings: dict[str, Any]) -> None:
        if (self.last_timestamp is None or time.monotonic() - self.last_timestamp >
                float(settings["maximum_frame_age_seconds"])):
            raise RuntimeError("Frame too old after inference; stopping. Do not extend the watchdog.")

    def close(self) -> None:
        self.closed.set()
        self.capture.release()
        self.worker.join(timeout=0.25)


class ESP32Link:
    """Serial dependency for the ESP32 velocity controller."""

    def __init__(self, serial_module: Any, port: str, baudrate: int):
        self.connection = serial_module.Serial()
        self.connection.port = port
        self.connection.baudrate = baudrate
        self.connection.timeout = 0
        self.connection.write_timeout = 0.5
        self.connection.rtscts = False
        self.connection.dsrdtr = False
        self.connection.dtr = False
        self.connection.rts = False
        self.connection.open()
        time.sleep(1.0)
        self.connection.reset_input_buffer()
        self.sequence = 0
        self.armed = False
        self.receive_buffer = b""
        try:
            self._exchange("STOP", "OK,STOP", timeout=1.0)
            self._exchange("PING", "PONG", timeout=1.0)
        except Exception:
            self.connection.close()
            raise

    def _write(self, command: str) -> None:
        payload = (command + "\n").encode("ascii")
        if self.connection.write(payload) != len(payload):
            raise RuntimeError("Incomplete serial write.")
        self.connection.flush()

    def _exchange(self, command: str, expected: str, timeout: float = 0.10) -> None:
        self._write(command)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if expected in self.drain_responses():
                return
            time.sleep(0.001)
        self.armed = False
        raise RuntimeError(f"Missing ESP32 acknowledgement: {expected}")

    def arm(self) -> None:
        self._exchange("ARM", "OK,ARM", timeout=1.0)
        self.armed = True

    def send_speeds(self, speeds: np.ndarray) -> None:
        if not self.armed:
            raise RuntimeError("Cannot command servos until the ESP32 is armed.")
        values = np.asarray(speeds, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("Exactly three finite servo speeds are required.")
        if np.any(np.abs(values) > 0.35):
            raise ValueError("This prototype limits servo speeds to +/-0.35.")
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        self._exchange(
            f"V,{self.sequence},{values[0]:.4f},{values[1]:.4f},{values[2]:.4f}",
            f"OK,V,{self.sequence}",
        )

    def drain_responses(self) -> list[str]:
        if not self.connection.in_waiting:
            return []
        self.receive_buffer += self.connection.read(self.connection.in_waiting)
        if len(self.receive_buffer) > 8192:
            raise RuntimeError("ESP32 response buffer overflow.")
        chunks = self.receive_buffer.split(b"\n")
        self.receive_buffer = chunks.pop()
        responses = [chunk.decode("ascii", errors="replace").strip() for chunk in chunks]
        errors = [line for line in responses if line.startswith("ERR,")]
        if errors:
            self.armed = False
            raise RuntimeError("ESP32 reported: " + "; ".join(errors))
        return responses

    def stop(self) -> None:
        if self.connection.is_open:
            try:
                self._exchange("STOP", "OK,STOP", timeout=1.0)
            finally:
                self.armed = False

    def close(self) -> None:
        if self.connection.is_open:
            try:
                self.stop()
            finally:
                self.connection.close()


class GuardedDrive:
    """Tracks previous commands in real time and applies a watchdog reserve."""

    def __init__(self, link: ESP32Link, settings: dict[str, Any]):
        self.link = link
        self.settings = settings
        self.budget = MotionBudget(
            settings["maximum_signed_travel_seconds"],
            settings["maximum_cumulative_motion_seconds"],
        )
        self.previous_command = np.zeros(3)
        self.accounted_at = time.monotonic()
        self.started_at: float | None = None
        self.armed = False

    def _account(self) -> None:
        now = time.monotonic()
        elapsed = now - self.accounted_at
        self.accounted_at = now
        self.budget.update(self.previous_command, elapsed)
        if self.armed and elapsed > float(self.settings["maximum_loop_gap_seconds"]):
            raise RuntimeError("Control loop stalled; inspect and manually re-home.")
        if self.started_at is not None and now - self.started_at > float(
            self.settings["maximum_session_seconds"]
        ):
            raise RuntimeError("Session time limit reached; inspect and re-home.")

    def arm(self) -> None:
        if self.armed:
            raise RuntimeError("Drive is already armed; stop before arming again.")
        self.link.arm()
        self.armed = True
        self.accounted_at = time.monotonic()
        if self.started_at is None:
            self.started_at = self.accounted_at

    def send_speeds(self, speeds: np.ndarray) -> None:
        self._account()
        self.budget.check_future(speeds, 0.35)
        # Count the entire write/ACK interval as the new command, conservatively.
        self.previous_command = np.asarray(speeds, dtype=float).copy()
        self.link.send_speeds(speeds)

    def stop(self) -> None:
        try:
            self._account()
        finally:
            try:
                self.link.stop()
            finally:
                self.armed = False
                now = time.monotonic()
                try:
                    self.budget.update(self.previous_command, now - self.accounted_at)
                finally:
                    self.previous_command[:] = 0.0
                    self.accounted_at = now


class HandTargetTracker:
    """Maps the index finger into a normalized 2D target in the palm frame."""

    def __init__(self, model_path: Path, settings: dict[str, Any]):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise RuntimeError(f"MediaPipe model not found: {model_path}")
        self.mp = mp
        self.vision = vision
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.tracker = vision.HandLandmarker.create_from_options(options)
        self.previous_timestamp = -1
        self.lateral_gain = float(settings["target_lateral_gain"])
        self.axial_gain = float(settings["target_axial_gain"])
        self.axial_offset = float(settings["target_axial_offset"])
        self.smoothing = float(settings["target_smoothing"])
        self.smoothed: np.ndarray | None = None

    def process(self, cv2_module: Any, frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(
            self.previous_timestamp + 1, time.monotonic_ns() // 1_000_000
        )
        self.previous_timestamp = timestamp
        result = self.tracker.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            self.smoothed = None
            return None

        hand = result.hand_landmarks[0]
        height, width = frame.shape[:2]
        points = np.array([[point.x * width, point.y * height] for point in hand], dtype=float)
        if not np.isfinite(points).all():
            self.smoothed = None
            return None
        wrist, index_mcp, index_pip, index_dip, index_tip = (
            points[0], points[5], points[6], points[7], points[8]
        )
        forward = unit(index_mcp - wrist)
        if forward is None:
            return None
        lateral_axis = np.array([-forward[1], forward[0]])
        finger_length = (
            np.linalg.norm(index_pip - index_mcp)
            + np.linalg.norm(index_dip - index_pip)
            + np.linalg.norm(index_tip - index_dip)
        )
        if finger_length < 1e-6:
            return None

        displacement = index_tip - index_mcp
        lateral = float(np.dot(displacement, lateral_axis) / finger_length)
        axial = float(np.dot(displacement, forward) / finger_length)
        target = np.array(
            [
                np.clip(lateral * self.lateral_gain, -0.85, 0.85),
                np.clip(
                    self.axial_offset + axial * self.axial_gain, -0.10, 1.10
                ),
            ],
            dtype=float,
        )
        if self.smoothed is None:
            self.smoothed = target
        else:
            self.smoothed += self.smoothing * (target - self.smoothed)

        for index in (0, 5, 6, 7, 8):
            location = tuple(np.rint(points[index]).astype(int))
            cv2_module.circle(frame, location, 5, (0, 220, 255), -1)
        return self.smoothed.copy()

    def close(self) -> None:
        self.tracker.close()


class VisualJacobianModel:
    """Learned local map from command-seconds to visible tip displacement."""

    def __init__(self, jacobian: np.ndarray, straight_length_pixels: float,
                 home_state: np.ndarray | None = None, camera_index: int = 0,
                 frame_size: tuple[int, int] | None = None, tracker_signature: str | None = None):
        matrix = np.asarray(jacobian, dtype=float)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("The visual Jacobian must be a finite 2x3 matrix.")
        if np.max(np.abs(matrix)) > 100:
            raise ValueError("Implausibly large Jacobian; recalibrate with stable YOLO keypoints.")
        self.jacobian = matrix
        self.straight_length_pixels = float(straight_length_pixels)
        if type(camera_index) is not int or not 0 <= camera_index <= 20:
            raise ValueError("Invalid model camera index.")
        if frame_size is not None and (
            len(frame_size) != 2 or any(type(v) is not int or v <= 0 for v in frame_size)
        ):
            raise ValueError("Invalid calibration frame dimensions.")
        self.camera_index = camera_index
        self.frame_size = frame_size
        self.tracker_signature = tracker_signature
        self.home_state = np.asarray([0.0, 1.0] if home_state is None else home_state, dtype=float)
        if self.home_state.shape != (2,) or not np.all(np.isfinite(self.home_state)):
            raise ValueError("Invalid model home state.")
        if not math.isfinite(self.straight_length_pixels) or self.straight_length_pixels < 20:
            raise ValueError("Straight tentacle reference length must be at least 20 pixels.")
        if self.rank_and_condition()[0] < 1:
            raise ValueError("No usable visible response was learned.")

    def decomposition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left, singular, right = np.linalg.svd(self.jacobian, full_matrices=False)
        threshold = max(0.03, float(singular[0]) * 0.10)
        keep = singular > threshold
        return left, singular, right, keep

    def projected_error(self, error: np.ndarray) -> np.ndarray:
        left, _, _, keep = self.decomposition()
        basis = left[:, keep]
        return basis @ (basis.T @ np.asarray(error, dtype=float))

    @classmethod
    def load(cls, path: Path, expected_signature: str) -> "VisualJacobianModel":
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("model_type") != "local_visual_jacobian_yolo_five_points_v1":
            raise ValueError("Old/ArUco calibration rejected; calibrate with the YOLO tracker first.")
        if payload.get("tracker_signature") != expected_signature:
            raise ValueError("YOLO weights/settings changed; recalibrate before operation.")
        return cls(
            np.asarray(payload["jacobian"], dtype=float),
            float(payload["straight_length_pixels"]),
            np.asarray(payload.get("home_state", [0.0, 1.0]), dtype=float),
            payload.get("camera_index", 0),
            tuple(payload["frame_size"]) if payload.get("frame_size") else None,
            payload.get("tracker_signature"),
        )

    def save(self, path: Path) -> None:
        payload = {
            "model_type": "local_visual_jacobian_yolo_five_points_v1",
            "tracker_signature": self.tracker_signature,
            "geometry": "point2_root_axis_point1_to_point2_tip_point5_v1",
            "jacobian": self.jacobian.tolist(),
            "straight_length_pixels": self.straight_length_pixels,
            "home_state": self.home_state.tolist(),
            "camera_index": self.camera_index,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "created_unix_time": time.time(),
            "warning": "Camera-only estimate; no absolute spool position feedback.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")

    def rank_and_condition(self) -> tuple[int, float]:
        _, singular, _, keep = self.decomposition()
        rank = int(np.sum(keep))
        usable = singular[keep]
        condition = float(usable[0] / usable[-1]) if rank else float("inf")
        return rank, condition

    def command(
        self,
        error: np.ndarray,
        estimated_travel: np.ndarray,
        gain: float,
        damping: float,
        nullspace_gain: float,
        maximum_speed: float,
    ) -> np.ndarray:
        error = np.asarray(error, dtype=float)
        estimated_travel = np.asarray(estimated_travel, dtype=float)
        if (error.shape != (2,) or estimated_travel.shape != (3,) or
                not np.all(np.isfinite(error)) or not np.all(np.isfinite(estimated_travel)) or
                not all(math.isfinite(v) for v in (gain, damping, nullspace_gain, maximum_speed)) or
                damping <= 0 or gain < 0 or nullspace_gain < 0 or maximum_speed <= 0):
            raise ValueError("Invalid visual-controller input.")
        left, singular, right, keep = self.decomposition()
        factors = np.where(keep, singular / (singular**2 + damping**2), 0.0)
        pseudo_inverse = (right.T * factors) @ left.T
        command = pseudo_inverse @ (gain * np.asarray(error, dtype=float))
        # Exact retained-subspace null projector; damping must not leak this term.
        nullspace = np.eye(3) - right[keep].T @ right[keep]
        command -= nullspace_gain * (nullspace @ estimated_travel)
        largest = float(np.max(np.abs(command)))
        if largest > maximum_speed:
            command *= maximum_speed / largest
        return command

class MotionBudget:
    """Command-time estimate only; it is not a replacement for encoders."""

    def __init__(self, signed_limit: float, cumulative_limit: float):
        self.signed_limit = float(signed_limit)
        self.cumulative_limit = float(cumulative_limit)
        if (not math.isfinite(self.signed_limit) or self.signed_limit <= 0 or
                not math.isfinite(self.cumulative_limit) or self.cumulative_limit <= 0):
            raise ValueError("Motion-budget limits must be finite and positive.")
        self.signed = np.zeros(3, dtype=float)
        self.cumulative = np.zeros(3, dtype=float)

    def update(self, command: np.ndarray, elapsed: float) -> None:
        values = np.asarray(command, dtype=float)
        if (values.shape != (3,) or not np.all(np.isfinite(values)) or
                not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Invalid motion-budget input.")
        delta = values * elapsed
        self.signed += delta
        self.cumulative += np.abs(delta)
        if np.any(np.abs(self.signed) > self.signed_limit):
            raise RuntimeError("Estimated spool-travel limit reached; manually re-home.")
        if np.any(self.cumulative > self.cumulative_limit):
            raise RuntimeError("Cumulative motion budget reached; inspect and re-home.")

    def check_future(self, command: np.ndarray, seconds: float) -> None:
        values = np.asarray(command, dtype=float)
        if (values.shape != (3,) or not np.all(np.isfinite(values)) or
                not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Invalid future-budget input.")
        delta = values * seconds
        if (np.any(np.abs(self.signed + delta) > self.signed_limit) or
                np.any(self.cumulative + np.abs(delta) > self.cumulative_limit)):
            raise RuntimeError("Insufficient motion budget for another watchdog interval.")


def put_lines(cv2_module: Any, frame: np.ndarray, lines: list[str]) -> None:
    for row, text in enumerate(lines):
        location = (10, 25 + 25 * row)
        cv2_module.putText(
            frame, text, location, cv2_module.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 3, cv2_module.LINE_AA
        )
        cv2_module.putText(
            frame, text, location, cv2_module.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2_module.LINE_AA
        )


def resized(cv2_module: Any, frame: np.ndarray) -> np.ndarray:
    return cv2_module.resize(frame, (640, 480))


class CalibrationTrainer:
    def __init__(
        self, cv2_module: Any, camera: Camera, tracker: YoloTentacleTracker,
        link: GuardedDrive, settings: dict[str, Any]
    ):
        self.cv2 = cv2_module
        self.camera = camera
        self.tracker = tracker
        self.link = link
        self.settings = settings
        self.reference_length = 0.0
        self.frame_size: tuple[int, int] | None = None

    def wait_for_straight_home(self) -> None:
        recent_lengths: list[float] = []
        print("Manually straighten the tentacle and lightly tension all lines.")
        print("Press S in the preview once all five points are stable and the tentacle is straight.")
        while True:
            frame = self.camera.read()
            observation = self.tracker.observe(frame, None)
            # The plate-reference axis may have a fixed offset from the shaft;
            # learn that bias as home_state rather than demanding a zero x value.
            at_home = (observation is not None and observation.straightness > 0.92 and
                       abs(observation.state[0]) < 0.40 and observation.state[1] > 0.85)
            if at_home:
                recent_lengths.append(observation.base_to_tip_pixels)
                recent_lengths = recent_lengths[-30:]
            else:
                recent_lengths.clear()
            put_lines(
                self.cv2,
                frame,
                [
                    "CALIBRATION: manually straight and lightly tensioned",
                    "Press S when straight AND aligned with base direction; Q aborts",
                    f"Valid frames: {len(recent_lengths)}/15",
                    f"Pose gate: {self.tracker.last_rejection}",
                ],
            )
            self.cv2.imshow("Tentacle calibration", frame)
            key = self.cv2.waitKey(1) & 0xFF
            if (key in (ord("q"), ord("Q"), 27) or
                    self.cv2.getWindowProperty("Tentacle calibration", self.cv2.WND_PROP_VISIBLE) < 1):
                raise RuntimeError("Calibration aborted.")
            if (key in (ord("s"), ord("S")) and at_home and
                    len(recent_lengths) >= 15 and
                    np.std(recent_lengths) / np.mean(recent_lengths) < 0.02):
                self.camera.assert_fresh(self.settings)
                self.reference_length = float(np.median(recent_lengths))
                self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                return

    def wait_for_visual_arm(self) -> None:
        # Do not block GUI updates with terminal input while waiting to arm.
        print("Calibration will move each servo in both directions. Keep a physical cutoff ready.")
        print("In the live window: A permits calibration pulses; Q cancels.")
        stable = 0
        previous = None
        while True:
            frame = self.camera.read()
            observation = self.tracker.observe(frame, self.reference_length)
            valid = (observation is not None and observation.straightness > .92 and
                     abs(observation.state[0]) < .40 and observation.state[1] > .85)
            stable = stable+1 if (valid and (previous is None or
                np.linalg.norm(observation.state-previous) < .02)) else 0
            previous = observation.state.copy() if valid else None
            put_lines(self.cv2, frame, ["S HOME SET - NO MOTION YET",
                "A: permit small AUTOMATIC MOTOR PULSES | Q: cancel",
                f"Stable straight frames: {stable}/15", f"Pose gate: {self.tracker.last_rejection}"])
            self.cv2.imshow("Tentacle calibration", frame)
            key = self.cv2.waitKey(1) & 0xFF
            if (key in (ord("q"),ord("Q"),27) or
                    self.cv2.getWindowProperty("Tentacle calibration",self.cv2.WND_PROP_VISIBLE) < 1):
                raise RuntimeError("Calibration cancelled before arming.")
            if key in (ord("a"),ord("A")) and stable >= 15:
                self.camera.assert_fresh(self.settings)
                return

    def sample_state(self, seconds: float) -> np.ndarray:
        samples: list[np.ndarray] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.camera.read()
            observation = self.tracker.observe(frame, self.reference_length)
            if observation is None:
                self.link.stop()
                raise RuntimeError("Required YOLO pose lost during calibration.")
            self.camera.assert_fresh(self.settings)
            self.link.send_speeds(np.zeros(3))
            samples.append(observation.state)
            put_lines(self.cv2, frame, ["Calibration sampling", "Q aborts"])
            self.cv2.imshow("Tentacle calibration", frame)
            if (self.cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27) or
                    self.cv2.getWindowProperty("Tentacle calibration",self.cv2.WND_PROP_VISIBLE) < 1):
                self.link.stop()
                raise RuntimeError("Calibration aborted.")
            time.sleep(1.0 / float(self.settings["control_hz"]))
        if len(samples) < 3:
            raise RuntimeError("Too few camera samples during calibration.")
        # Ignore the first half of settling where possible.
        return np.median(np.asarray(samples[-max(3, len(samples) // 2):]), axis=0)

    def pulse(self, servo_index: int, speed: float, seconds: float) -> float:
        command = np.zeros(3, dtype=float)
        command[servo_index] = speed
        period = 1.0 / float(self.settings["control_hz"])
        starting_travel = self.link.budget.signed[servo_index]
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.camera.read()
            observation = self.tracker.observe(frame, self.reference_length)
            if observation is None:
                self.link.stop()
                raise RuntimeError("Required YOLO pose lost during a motor pulse.")
            self.camera.assert_fresh(self.settings)
            self.link.send_speeds(command)
            put_lines(
                self.cv2,
                frame,
                [f"Servo {servo_index + 1}: {speed:+.2f}", "Q aborts"],
            )
            self.cv2.imshow("Tentacle calibration", frame)
            if (self.cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27) or
                    self.cv2.getWindowProperty("Tentacle calibration",self.cv2.WND_PROP_VISIBLE) < 1):
                self.link.stop()
                raise RuntimeError("Calibration aborted.")
            time.sleep(period)
        self.link.send_speeds(np.zeros(3))
        return float(self.link.budget.signed[servo_index] - starting_travel)

    def train(self) -> VisualJacobianModel:
        self.wait_for_straight_home()
        self.wait_for_visual_arm()

        speed = float(self.settings["calibration_speed"])
        duration = float(self.settings["calibration_pulse_seconds"])
        settle = float(self.settings["calibration_settle_seconds"])
        repeats = int(self.settings["calibration_repeats"])
        self.link.arm()
        baseline = self.sample_state(settle)
        columns: list[np.ndarray] = []

        for servo_index in range(3):
            estimates: list[np.ndarray] = []
            for repeat in range(repeats):
                print(f"Calibrating servo {servo_index + 1}, pass {repeat + 1}/{repeats}")
                before = self.sample_state(settle)
                for direction in (1.0, -1.0, -1.0, 1.0):
                    exposure = self.pulse(servo_index, direction * speed, duration)
                    after = self.sample_state(settle)
                    displacement = after - before
                    if np.linalg.norm(displacement) > 0.15:
                        raise RuntimeError("Calibration moved too far; reduce pulse settings.")
                    if abs(exposure) > 1e-5:
                        estimates.append(displacement / exposure)
                    before = after
                if np.linalg.norm(before - baseline) > 0.08:
                    raise RuntimeError(
                        "Tentacle did not return near home; inspect and manually re-home."
                    )
            estimates_array = np.asarray(estimates)
            column = np.median(estimates_array, axis=0)
            scatter = float(np.median(np.linalg.norm(estimates_array - column, axis=1)))
            if np.linalg.norm(column) < 0.03 or scatter > max(0.06, 1.5 * np.linalg.norm(column)):
                raise RuntimeError(
                    "Inconsistent or invisible calibration response. Check slack, "
                    "neutral pulse, keypoints, and camera plane before retraining."
                )
            columns.append(column)

        self.link.stop()
        model = VisualJacobianModel(np.column_stack(columns), self.reference_length,
                                    baseline, int(self.settings["camera"]), self.frame_size,
                                    self.tracker.signature)
        rank, condition = model.rank_and_condition()
        if rank < 1:
            raise RuntimeError(
                f"Calibration is unusable (rank={rank}, condition={condition:.1f}). "
                "Check tendon spacing, keypoint orientation, and camera view."
            )
        print(f"Calibration accepted: rank={rank}, condition={condition:.2f}")
        if rank == 1:
            print("Only one visible direction is controllable here; other errors are ignored.")
        return model


class SharedFrameTracker:
    """Both detectors receive untouched pixels from one captured frame.

    Trackers annotate their inputs, so use independent copies. Never mirror the
    capture: both detectors and the calibrated coordinates use the raw image.
    """

    def __init__(self, cv2_module: Any, hand_tracker: HandTargetTracker,
                 tentacle_tracker: YoloTentacleTracker):
        self.cv2 = cv2_module
        self.hand_tracker = hand_tracker
        self.tentacle_tracker = tentacle_tracker

    def process(self, frame: np.ndarray, reference_length: float | None):
        hand_frame = frame.copy()
        tentacle_frame = frame.copy()
        target = self.hand_tracker.process(self.cv2, hand_frame)
        observation = self.tentacle_tracker.observe(tentacle_frame, reference_length)
        return target, observation, hand_frame, tentacle_frame


class TentacleApplication:
    def __init__(
        self, cv2_module: Any, camera: Camera,
        hand_tracker: HandTargetTracker, tentacle_tracker: YoloTentacleTracker,
        model: VisualJacobianModel, link: GuardedDrive | None, settings: dict[str, Any]
    ):
        self.cv2 = cv2_module
        self.camera = camera
        self.tracking = SharedFrameTracker(cv2_module, hand_tracker, tentacle_tracker)
        self.model = model
        self.link = link
        self.settings = settings
        self.armed = False
        self.stable_frames = 0
        self.hard_fault = False
        self.fault = "Not armed"
        self.previous_command = np.zeros(3)
        self.previous_state: np.ndarray | None = None
        self.previous_time = time.monotonic()
        self.no_progress_since = time.monotonic()
        self.best_error = float("inf")
        self.previous_target: np.ndarray | None = None
        self.previous_tracking_state: np.ndarray | None = None

    def disarm(self, reason: str, hard: bool = False) -> None:
        if self.link is not None and (self.armed or self.link.armed):
            try:
                self.link.stop()
            except (RuntimeError, OSError) as exc:
                print(f"Stop warning: {exc}", file=sys.stderr)
        self.armed = False
        self.fault = reason
        self.hard_fault = self.hard_fault or hard
        self.previous_command[:] = 0.0
        self.previous_state = None
        self.stable_frames = 0

    def rate_limit(self, requested: np.ndarray, elapsed: float) -> np.ndarray:
        maximum_change = float(self.settings["maximum_acceleration"]) * elapsed
        change = np.clip(
            requested - self.previous_command, -maximum_change, maximum_change
        )
        return self.previous_command + change

    def run(self) -> None:
        print("Keep the tentacle manually homed before starting this session.")
        print("A: arm  X: stop  Q/Esc: quit")
        while True:
            iteration_started = time.monotonic()
            raw_frame = self.camera.read()
            if (self.model.frame_size is not None and
                    (raw_frame.shape[1], raw_frame.shape[0]) != self.model.frame_size):
                self.disarm("Camera resolution changed; retrain", hard=True)
                raise RuntimeError("Camera resolution differs from calibration; retrain.")
            target, observation, hand_frame, tentacle_frame = self.tracking.process(
                raw_frame, self.model.straight_length_pixels
            )
            try:
                self.camera.assert_fresh(self.settings)
            except RuntimeError:
                self.disarm("Stale image after inference", hard=True)
                raise
            now = time.monotonic()
            elapsed = now - self.previous_time
            self.previous_time = now
            if self.armed and elapsed > float(self.settings["maximum_loop_gap_seconds"]):
                self.disarm("Slow/stalled camera or inference loop", hard=True)
            valid = (target is not None and observation is not None and
                     np.asarray(target).shape == (2,) and np.isfinite(target).all())
            stable = (valid and observation is not None and
                      (self.previous_tracking_state is None or
                       np.linalg.norm(observation.state - self.previous_tracking_state) < 0.03))
            self.stable_frames = self.stable_frames + 1 if stable else 0
            self.previous_tracking_state = observation.state.copy() if observation else None

            command = np.zeros(3)
            error = None
            if not valid:
                if self.armed:
                    missing = "hand" if target is None else "tentacle pose"
                    self.disarm(f"Lost {missing}; inspect and re-home before restart", hard=True)
            elif observation is not None and target is not None:
                offset = target - self.model.home_state
                distance = float(np.linalg.norm(offset))
                maximum_offset = float(self.settings["maximum_target_offset"])
                if distance > maximum_offset:
                    target = self.model.home_state + offset * maximum_offset / distance
                error = target - observation.state
                if (self.armed and self.previous_state is not None and
                        np.linalg.norm(observation.state - self.previous_state) > 0.12):
                    self.disarm("Implausible pose jump", hard=True)
                if (self.armed and
                        np.linalg.norm(observation.state - self.model.home_state) > maximum_offset + 0.12):
                    self.disarm("Tentacle left calibrated local workspace", hard=True)
                if self.armed and self.link is not None:
                    projected_error = float(np.linalg.norm(self.model.projected_error(error)))
                    if (self.previous_target is None or
                            np.linalg.norm(target - self.previous_target) > 0.02):
                        self.best_error = projected_error
                        self.no_progress_since = now
                    self.previous_target = target.copy()
                    if projected_error < self.best_error - 0.005:
                        self.best_error = projected_error
                        self.no_progress_since = now
                    elif projected_error < float(self.settings["error_deadband"]):
                        self.no_progress_since = now
                    elif now - self.no_progress_since > 2.0:
                        self.disarm("No visible progress; inspect tendons and model", hard=True)

                if self.armed and self.link is not None:
                    if projected_error >= float(self.settings["error_deadband"]):
                        command = self.model.command(
                            error,
                            self.link.budget.signed,
                            float(self.settings["gain"]),
                            float(self.settings["damping"]),
                            float(self.settings["nullspace_gain"]),
                            float(self.settings["maximum_speed"]),
                        )
                    command = self.rate_limit(command, elapsed)
                    try:
                        self.link.send_speeds(command)
                    except (RuntimeError, OSError, ValueError) as exc:
                        self.disarm(str(exc), hard=True)
                        command[:] = 0.0

                if self.armed:
                    self.previous_state = observation.state.copy()

            self.previous_command = command.copy()
            status = "ARMED" if self.armed else "STOPPED"
            common = [
                "ONE WEBCAM: same captured frame in both panels",
                f"Status: {status}",
                f"Stable detections: {self.stable_frames}",
                f"Fault: {self.fault}",
                "A arm | X stop | Q quit",
            ]
            if target is not None:
                put_lines(
                    self.cv2,
                    hand_frame,
                    [f"Hand target: {target[0]:+.2f}, {target[1]:+.2f}"] + common,
                )
            else:
                put_lines(self.cv2, hand_frame, ["HAND NOT DETECTED"] + common)
            if observation is not None:
                extra = [
                    f"Tentacle: {observation.state[0]:+.2f}, {observation.state[1]:+.2f}",
                    f"Motor speeds: {command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}",
                ]
                if error is not None:
                    extra.append(f"Error norm: {np.linalg.norm(error):.3f}")
                put_lines(self.cv2, tentacle_frame, extra + common)
            else:
                put_lines(
                    self.cv2, tentacle_frame,
                    ["REQUIRED TENTACLE POSE NOT DETECTED"] + common
                )

            display = np.hstack(
                [resized(self.cv2, hand_frame), resized(self.cv2, tentacle_frame)]
            )
            self.cv2.imshow("Hand target | Tentacle feedback", display)
            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.disarm("Exiting")
                break
            if key in (ord("x"), ord("X")):
                self.disarm("Stopped by operator; inspect and re-home before restart", hard=True)
            if key in (ord("a"), ord("A")):
                required = int(self.settings["stable_frames_before_arm"])
                if self.link is None:
                    self.fault = "Preview only: use run --enable-motion for powered operation"
                elif self.armed:
                    pass  # Do not reset accounting or neutralize a running drive.
                elif self.hard_fault:
                    self.fault = "Hard fault: restart only after inspection and re-home"
                elif self.stable_frames < required:
                    self.fault = f"Need {required} stable detection frames before arming"
                else:
                    if (observation is None or
                            np.linalg.norm(observation.state - self.model.home_state) > 0.08):
                        self.fault = "Manually straighten tentacle to the trained home first"
                    else:
                        self.camera.assert_fresh(self.settings)
                        self.link.arm()
                        self.armed = True
                        self.fault = "None"
                        self.previous_command[:] = 0.0
                        self.previous_state = None
                        self.best_error = float("inf")
                        self.no_progress_since = time.monotonic()
                        self.previous_time = self.no_progress_since
            if self.cv2.getWindowProperty("Hand target | Tentacle feedback", self.cv2.WND_PROP_VISIBLE) < 1:
                self.disarm("Preview window closed")
                break
            time.sleep(max(0.0, 1.0 / float(self.settings["control_hz"]) -
                               (time.monotonic() - iteration_started)))


def preview_cameras(cv2_module: Any, settings: dict[str, Any], base: Path) -> None:
    """Motor-free shared-frame hand/pose verification, before calibration."""
    with ExitStack() as cleanup:
        cleanup.callback(cv2_module.destroyAllWindows)
        camera = Camera(cv2_module, int(settings["camera"]), "shared webcam")
        cleanup.callback(camera.close)
        hand_tracker = HandTargetTracker(resolve_from(base, str(settings["hand_model"])), settings)
        cleanup.callback(hand_tracker.close)
        tracking = SharedFrameTracker(cv2_module, hand_tracker,
                                      YoloTentacleTracker(cv2_module, settings, base))
        while True:
            started = time.monotonic()
            target, observation, hand_frame, tentacle_frame = tracking.process(camera.read(), None)
            age = time.monotonic()-camera.last_timestamp
            put_lines(cv2_module, hand_frame,
                      ["ONE WEBCAM - hand tracking", "Hand detected" if target is not None else "No hand detected",
                       "NO MOTOR COMMANDS - Q quits"])
            put_lines(cv2_module, tentacle_frame,
                      ["SAME FRAME - tentacle tracking",
                       "All five keypoints accepted" if observation else tracking.tentacle_tracker.last_rejection,
                       f"Processing: {(time.monotonic()-started)*1000:.0f} ms | frame age: {age*1000:.0f} ms",
                       "Frame age OK for motion" if age <= settings["maximum_frame_age_seconds"] else
                       "TOO SLOW FOR MOTION - do not extend the watchdog"])
            cv2_module.imshow("Camera check", np.hstack(
                [resized(cv2_module, hand_frame), resized(cv2_module, tentacle_frame)]))
            if cv2_module.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2_module.getWindowProperty("Camera check", cv2_module.WND_PROP_VISIBLE) < 1:
                break


def check_neutral(serial_module: Any, settings: dict[str, Any]) -> None:
    print("Disconnect all fishing lines from the servo spools. Keep power cutoff ready.")
    if input("Type UNLOADED after disconnecting the tendons: ").strip() != "UNLOADED":
        raise RuntimeError("Neutral check cancelled.")
    link = ESP32Link(serial_module, str(settings["serial_port"]), int(settings["baudrate"]))
    try:
        link.arm()
        deadline = time.monotonic() + 10.0
        print("Sending zero speed for 10 seconds. Ctrl+C stops. No spool should creep.")
        while time.monotonic() < deadline:
            link.send_speeds(np.zeros(3))
            time.sleep(0.05)
    finally:
        safe_close(link)


def load_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Run: python -m pip install -r requirements.txt"
        ) from exc
    return cv2, serial


def run_training(
    cv2_module: Any, serial_module: Any, settings: dict[str, Any], base: Path
) -> None:
    with ExitStack() as cleanup:
        cleanup.callback(cv2_module.destroyAllWindows)
        camera = Camera(cv2_module, int(settings["camera"]), "shared webcam")
        cleanup.callback(camera.close)
        tracker = YoloTentacleTracker(cv2_module, settings, base)
        link = ESP32Link(serial_module, str(settings["serial_port"]), int(settings["baudrate"]))
        cleanup.callback(safe_close, link)
        guarded = GuardedDrive(link, settings)
        cleanup.callback(safe_close, guarded)
        model = CalibrationTrainer(
            cv2_module, camera, tracker, guarded, settings
        ).train()
        model_path = resolve_from(base, str(settings["visual_model"]))
        model.save(model_path)
        print(f"Saved learned visual model to {model_path}")


def safe_close(resource: Any) -> None:
    try:
        if isinstance(resource, GuardedDrive):
            resource.stop()
        else:
            resource.close()
    except Exception as exc:
        print(f"Cleanup warning: {exc}", file=sys.stderr)


def run_controller(
    cv2_module: Any, serial_module: Any, settings: dict[str, Any], base: Path,
    enable_motion: bool = False,
) -> None:
    model_path = resolve_from(base, str(settings["visual_model"]))
    if not model_path.exists():
        raise RuntimeError(f"Train first; visual model does not exist: {model_path}")
    signature = tracker_signature(settings, base)
    model = VisualJacobianModel.load(model_path, signature)
    if model.camera_index != settings["camera"] or model.frame_size is None:
        raise RuntimeError("Camera selection/model metadata changed; retrain with this webcam.")
    if enable_motion:
        print("POWER OFF, manually straighten the tentacle, and reset spool positions.")
        print("Lightly tension all lines, restore power, and keep a power cutoff ready.")
        if input("Type HOMED only after this physical reset: ").strip() != "HOMED":
            raise RuntimeError("Manual session home was not confirmed.")
    with ExitStack() as cleanup:
        cleanup.callback(cv2_module.destroyAllWindows)
        camera = Camera(cv2_module, int(settings["camera"]), "shared webcam")
        cleanup.callback(camera.close)
        hand_tracker = HandTargetTracker(resolve_from(base, str(settings["hand_model"])), settings)
        cleanup.callback(hand_tracker.close)
        tentacle_tracker = YoloTentacleTracker(cv2_module, settings, base)
        drive = None
        if enable_motion:
            link = ESP32Link(serial_module, str(settings["serial_port"]), int(settings["baudrate"]))
            cleanup.callback(safe_close, link)
            drive = GuardedDrive(link, settings)
            cleanup.callback(safe_close, drive)
        TentacleApplication(
            cv2_module, camera, hand_tracker,
            tentacle_tracker, model, drive, settings
        ).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config_yolo.json"),
        help="JSON configuration file (default: config_yolo.json beside this script)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("calibrate", aliases=["train"], help="Learn servo response; NOT YOLO detector training")
    run_parser = subparsers.add_parser("run", help="Preview or run shared-webcam control")
    run_parser.add_argument("--enable-motion", action="store_true",
                            help="Enable serial output; requires manual home and A to arm")
    subparsers.add_parser("preview", help="Check hand and YOLO keypoints in one webcam (no serial)")
    subparsers.add_parser("neutral", help="10-second unloaded stop-pulse check")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings, base = load_config(args.config)
    cv2_module, serial_module = load_dependencies()
    if args.mode in ("train", "calibrate"):
        run_training(cv2_module, serial_module, settings, base)
    elif args.mode == "run":
        run_controller(cv2_module, serial_module, settings, base, args.enable_motion)
    elif args.mode == "preview":
        preview_cameras(cv2_module, settings, base)
    elif args.mode == "neutral":
        check_neutral(serial_module, settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by operator.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError, ImportError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

