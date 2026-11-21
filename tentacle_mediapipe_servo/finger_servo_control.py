"""MediaPipe-only hand-angle control of three continuous-rotation servos.

Default mode is camera preview: it never imports pyserial or opens a port.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import queue
import sys
import threading
import time

from gesture_logic import (AWAY_MODES, Observation, PulseGate, Settings, STOP,
                           ZERO, classify_hand, load_settings, speeds_for)
from servo_link import ESP32Link, GuardedDrive, SafetyFault, WindingEstimate

ROOT = Path(__file__).resolve().parent
WINDOW = "MediaPipe finger-angle servo control"


class Camera:
    """Capture thread keeps only the latest frame, never repeats an old frame."""
    def __init__(self, cv2, index):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera {index}; close other camera applications.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frames = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.failure = None
        self.first_read = True
        self.startup_deadline = time.monotonic()+5.0
        self.worker = threading.Thread(target=self._capture, daemon=True)
        self.worker.start()

    def _capture(self):
        seen_frame = False
        try:
            while not self.closed.is_set():
                ok, frame = self.capture.read()
                timestamp = time.monotonic()
                if not ok:
                    if not seen_frame and timestamp < self.startup_deadline:
                        self.closed.wait(0.02)
                        continue
                    self.failure = "Camera stopped delivering frames."
                    return
                seen_frame = True
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait((timestamp, frame))
                except queue.Full:
                    pass
        except Exception as error:
            self.failure = f"Camera failure: {error}"

    def read(self):
        if self.failure:
            raise SafetyFault(self.failure)
        try:
            result = self.frames.get(timeout=5.0 if self.first_read else 0.15)
        except queue.Empty as error:
            raise SafetyFault(self.failure or "Camera timed out.") from error
        self.first_read = False
        return result

    def close(self):
        self.closed.set()
        self.capture.release()
        self.worker.join(timeout=0.25)


class HandTracker:
    def __init__(self, model_path, settings):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        self.mp, self.settings = mp, settings
        self.last_ms = -1
        self.tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=settings.minimum_hand_confidence,
            min_hand_presence_confidence=settings.minimum_hand_confidence,
            min_tracking_confidence=settings.minimum_hand_confidence))

    def process(self, cv2, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(self.last_ms+1, time.monotonic_ns()//1_000_000)
        self.last_ms = timestamp
        result = self.tracker.detect_for_video(image, timestamp)
        if len(result.hand_landmarks) != 1:
            return Observation(reason="Show exactly ONE whole hand"), None
        if not result.handedness or not result.handedness[0]:
            return Observation(reason="Unreliable hand identity"), None
        category = result.handedness[0][0]
        if category.score < self.settings.minimum_hand_confidence:
            return Observation(reason="Uncertain hand identity"), None
        points = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        height, width = frame.shape[:2]
        observation = classify_hand(points, width, height, self.settings,
                                    category.category_name)
        return observation, points

    def close(self):
        self.tracker.close()


def frame_is_fresh(captured_at, settings, now=None):
    now = time.monotonic() if now is None else now
    age = now-captured_at
    return math.isfinite(age) and 0 <= age <= settings.maximum_frame_age_seconds


def assert_fresh(captured_at, settings, now=None):
    if not frame_is_fresh(captured_at, settings, now):
        raise SafetyFault("Frame stale after inference; stopped. Do not extend motion deadlines.")


class FrameReadiness:
    """Warm up while DISARMED; never authorize stale input for motor control."""
    REQUIRED_FRAMES = 8

    def __init__(self, settings):
        self.settings = settings
        self.consecutive = 0
        self.last_completed = None
        self.frame_age_ms = 0.0
        self.inference_ms = 0.0
        self.loop_ms = 0.0
        self.last_frame_ok = False
        self.reason = "Warming up: show one whole hand"

    @property
    def ready(self):
        return self.consecutive >= self.REQUIRED_FRAMES

    def finish(self, captured_at, completed_at, inference_seconds, valid_hand):
        gap = 0.0 if self.last_completed is None else completed_at-self.last_completed
        self.frame_age_ms = (completed_at-captured_at)*1000
        self.inference_ms = inference_seconds*1000
        self.loop_ms = gap*1000
        numbers_ok = all(math.isfinite(v) for v in (captured_at, completed_at, inference_seconds, gap))
        if not numbers_ok or inference_seconds < 0 or gap < 0:
            self.reason = "Invalid frame timing: not ready"
        elif not frame_is_fresh(captured_at, self.settings, completed_at):
            self.reason = "Frame too old: remain disarmed"
        elif gap > self.settings.maximum_loop_gap_seconds:
            self.reason = "Processing loop too slow: remain disarmed"
        elif not valid_hand:
            self.reason = "Show one whole reliable hand"
        else:
            self.reason = "Fresh hand frames"
        self.last_frame_ok = self.reason == "Fresh hand frames"
        self.consecutive = min(self.REQUIRED_FRAMES, self.consecutive+1) if self.last_frame_ok else 0
        self.last_completed = completed_at
        return self.ready

    def summary(self):
        label = "TIMING READY" if self.ready else "WARMING UP / NOT READY"
        return (f"{label} {self.consecutive}/{self.REQUIRED_FRAMES} | "
                f"age {self.frame_age_ms:.0f}ms infer {self.inference_ms:.0f}ms loop {self.loop_ms:.0f}ms")


def motion_observation_ok(observation, armed_hand):
    if not observation.valid_hand:
        raise SafetyFault(f"Hand tracking lost/ambiguous ({observation.reason}): paused.")
    if not observation.hand_label or observation.hand_label != armed_hand:
        raise SafetyFault("Tracked hand changed: pause until fresh stable hand frames.")


def draw_overlay(cv2, frame, points, observation, gate, gesture, settings, motion, armed, winding=None, readiness=None, recovery_message=None, motion_fault=None, unloaded_test=False):
    height, width = frame.shape[:2]
    if points:
        xy = [(int(x*width), int(y*height)) for x, y, z in points]
        for a, b in ((0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (0, 17)):
            cv2.line(frame, xy[a], xy[b], (0, 220, 255), 2)
        for i in (0, 5, 6, 7, 8):
            cv2.circle(frame, xy[i], 5, (0, 255, 0), -1)
        cv2.arrowedLine(frame, xy[5], xy[8], (0, 140, 255), 3)
    angle = "--" if observation.angle_degrees is None else f"{observation.angle_degrees:.1f} deg"
    state = "ARMED - READY authorization active" if armed else ("MOTION DISABLED" if not motion else "READY authorized - waiting for fresh hand")
    if motion_fault:
        state = "PAUSED - auto-resume when tracking/communication recover"
    elif recovery_message:
        state = "TRACKING PAUSED - READY authorization retained"
    speeds = speeds_for(gesture, settings)
    if winding is not None:
        speeds = winding.coordinate(speeds)
    lines = [state, f"ARROW angle: {angle}  direction: {observation.gesture}",
             f"{'Planned' if armed else 'Preview'}: {gesture}   S1/S2/S3: {speeds[0]:+.2f} {speeds[1]:+.2f} {speeds[2]:+.2f}",
             gate.status, observation.reason,
             "READY authorizes this session | X/Q/Esc: stop/exit",
             "Left=S2 | Right=S3 | Up=ALL REVERSE | STOP=all stationary",
             f"Servo 1 forward mode: {settings.away_mode}",
             "NO travel/session/pulse cutoff. STOP pose stops all motors."]
    if winding is not None:
        lines.append("EST winding: " + " ".join(f"{v:.3f}" for v in winding.amounts))
    if readiness is not None:
        lines.extend((readiness.summary(), readiness.reason))
    if motion_fault or recovery_message:
        lines.append(motion_fault or recovery_message)
    for row, text in enumerate(lines):
        pos = (8, 22+22*row)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)


def stop_drive_on_exit(drive, exc_type, exc_value, traceback):
    """Keep the original fault visible if cleanup also detects a stalled loop."""
    try:
        drive.stop()
    except Exception as error:
        if exc_type is None:
            raise
        print(f"Additional STOP warning: {error}", file=sys.stderr)
    return False


def run_camera(settings, base, enable_motion=False, unloaded_test=False):
    import cv2
    if unloaded_test and not enable_motion:
        raise ValueError("Unloaded test requires explicit motion enable.")
    if enable_motion:
        if unloaded_test:
            print("Legacy disconnected test: travel/time cutoffs are removed in ALL modes.")
            print("Disconnect ALL tendons/loads. Do NOT use this mode with the tentacle attached.")
            print("UP reverses ALL motors. STOP pose stops. Automatic arming/resumption is enabled.")
            expected = "UNLOADED"
            prompt = "Type UNLOADED after confirming all tendons are disconnected: "
        else:
            print("CONTINUOUS MOTOR CONTROL: no travel, cumulative-motion or session cutoff.")
            print("LEFT/RIGHT/BACK run while held; UP reverses ALL motors. STOP pose stops.")
            print("READY authorizes automatic arming and resumption. No A key or neutral hold required.")
            print("No position/tension feedback; check neutral, rotation signs and your mechanical protection.")
            expected = "READY"
            prompt = "After checking the mechanism, type READY to continue: "
        print("Keep a physical servo-power cutoff ready. Restarting resets time estimates.")
        if input(prompt).strip() != expected:
            print("Cancelled; no serial port opened.")
            return
    model = Path(settings.hand_model)
    model = model if model.is_absolute() else base/model
    with ExitStack() as stack:
        tracker = HandTracker(model, settings)
        stack.callback(tracker.close)
        camera = Camera(cv2, settings.camera)
        stack.callback(camera.close)
        # Obtain a real startup frame before opening a serial connection.
        initial = camera.read()
        link = drive = None
        if enable_motion:
            import serial
            link = ESP32Link(serial, settings.serial_port, settings.baudrate)
            stack.callback(link.close)
            drive = GuardedDrive(link, settings, unloaded_test=unloaded_test)
            stack.push(lambda *exc: stop_drive_on_exit(drive, *exc))
            initial = None  # Opening serial took time: discard the pre-handshake frame.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        stack.callback(cv2.destroyAllWindows)
        gate = PulseGate(settings)
        readiness = FrameReadiness(settings)
        print("Warming up: READY authorization stays active. Motors arm automatically on fresh hand frames."
              if enable_motion else "Camera-only preview: no serial port or motor motion.")
        preview_winding = WindingEstimate(settings)
        preview_previous, preview_at = ZERO, time.monotonic()
        armed_hand = None
        motion_fault = None
        next_arm_attempt_at = 0.0

        def pause(reason, retry_delay=0.0):
            nonlocal gate, readiness, armed_hand, motion_fault, next_arm_attempt_at
            try:
                drive.pause()
            except Exception as stop_error:
                # A failed STOP is visible, but not a permanent software latch.
                # Each automatic arm attempt must verify STOP and PING again.
                print(f"STOP warning: {stop_error}; cut servo power if moving.", file=sys.stderr)
            motion_fault = f"Paused: {reason}. Auto-resume after fresh tracking/communication."
            print(motion_fault)
            gate = PulseGate(settings)
            readiness = FrameReadiness(settings)
            armed_hand = None
            next_arm_attempt_at = time.monotonic()+retry_delay

        while True:
            captured_at, frame = initial if initial is not None else camera.read()
            initial = None
            if settings.mirror:
                frame = cv2.flip(frame, 1)
            inference_started = time.monotonic()
            observation, points = tracker.process(cv2, frame)
            now = time.monotonic()
            inference_seconds = now-inference_started
            armed = drive is not None and drive.armed
            if armed:
                try:
                    motion_observation_ok(observation, armed_hand)
                    assert_fresh(captured_at, settings, now)
                except SafetyFault as error:
                    pause(error)
                    armed = False
            preview_winding.update(preview_previous, now-preview_at)
            preview_at = now
            if frame_is_fresh(captured_at, settings, now):
                gesture = gate.update(observation, now)
            else:
                gate = PulseGate(settings)
                gate.status = "Warm-up frame too old: motors remain DISARMED"
                gesture = STOP
            displayed_gesture = gesture if armed or readiness.ready else STOP
            draw_overlay(cv2, frame, points, observation, gate, displayed_gesture, settings,
                         enable_motion, armed,
                         drive.winding if drive is not None else preview_winding, readiness,
                         None, motion_fault, unloaded_test)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q'), ord('x'), ord('X')):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            completed_at = time.monotonic()
            if armed:
                try:
                    assert_fresh(captured_at, settings, completed_at)
                except SafetyFault as error:
                    pause(error)
                    armed = False
                    gesture = STOP
            readiness.finish(captured_at, completed_at, inference_seconds,
                             observation.valid_hand and bool(observation.hand_label))
            if not armed and not readiness.last_frame_ok:
                gate = PulseGate(settings)
                gesture = STOP
            preview_previous = (preview_winding.coordinate(speeds_for(gesture, settings))
                                if readiness.ready else ZERO)
            if drive is not None:
                if not drive.armed and readiness.ready and completed_at >= next_arm_attempt_at:
                    try:
                        assert_fresh(captured_at, settings)
                        drive.arm()
                    except Exception as error:
                        pause(error, retry_delay=1.0)
                        continue
                    armed_hand = observation.hand_label
                    motion_fault = None
                    # ARM always sends ZERO first. Never replay the pre-handshake
                    # target: read a new frame before any nonzero command.
                    continue
                if drive.armed:
                    try:
                        command = gesture if readiness.ready else STOP
                        drive.send_speeds(speeds_for(command, settings))
                    except Exception as error:
                        pause(error, retry_delay=1.0)


def serial_mode(settings, neutral=False):
    if neutral:
        print("Disconnect ALL THREE tendons before this stop-pulse test.")
        print("Zero-speed PWM is not power removal. Cut servo power if anything creeps.")
        if input("Type UNLOADED after disconnecting tendons: ").strip() != "UNLOADED":
            print("Cancelled; no serial port opened.")
            return
    import serial
    link = ESP32Link(serial, settings.serial_port, settings.baudrate)
    try:
        print("STOP and PING acknowledged: serial protocol compatible.")
        if neutral:
            drive = GuardedDrive(link, settings)
            try:
                drive.arm()
                end = time.monotonic()+10.0
                while time.monotonic() < end:
                    drive.send_speeds(ZERO)
                    time.sleep(0.04)
            finally:
                drive.stop()
            print("Neutral test complete. Confirm visually that every spool stayed still.")
    finally:
        link.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preview", choices=("preview", "run", "probe", "neutral"))
    parser.add_argument("--config", type=Path, default=ROOT/"config_gestures.json")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--port")
    parser.add_argument("--away-mode", choices=AWAY_MODES)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--unloaded-test", action="store_true",
                        help="Legacy disconnected-test confirmation; travel limits are removed in ALL modes.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.enable_motion and args.mode != "run":
        parser.error("--enable-motion is allowed only with run.")
    if args.unloaded_test and (args.mode != "run" or not args.enable_motion):
        parser.error("--unloaded-test requires run --enable-motion and disconnected tendons.")
    settings = load_settings(args.config, camera=args.camera, serial_port=args.port,
                             away_mode=args.away_mode)
    base = args.config.resolve().parent
    if args.mode in ("probe", "neutral"):
        serial_mode(settings, neutral=args.mode == "neutral")
    else:
        run_camera(settings, base, enable_motion=args.enable_motion, unloaded_test=args.unloaded_test)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; stopping and closing resources.")
        raise SystemExit(130)
    except Exception as error:
        print(f"Stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
