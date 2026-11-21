"""Independent ESP32 speeds, recoverable stops, and command-history estimates."""
from __future__ import annotations

import math
import time

from gesture_logic import Settings, ZERO


class SafetyFault(RuntimeError):
    pass


def checked_speeds(values):
    values = tuple(values)
    if len(values) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) <= 0.35 for v in values):
        raise ValueError("Exactly three finite speeds in [-0.35,+0.35] are required.")
    return values


class ESP32Link:
    """Requires the existing ARM/STOP/PING/V sequence-numbered firmware."""
    def __init__(self, serial_module, port, baudrate=115200):
        self.connection = serial_module.Serial()
        self.connection.port = port
        self.connection.baudrate = baudrate
        self.connection.timeout = 0
        self.connection.write_timeout = 0.08
        self.connection.rtscts = False
        self.connection.dsrdtr = False
        self.connection.dtr = False
        self.connection.rts = False
        self.sequence = 0
        self.armed = False
        self.receive_buffer = b""
        try:
            self.connection.open()
            time.sleep(1.0)  # Opening USB may reset an ESP32; no ARM has been sent.
            self.connection.reset_input_buffer()
            self.exchange("STOP", "OK,STOP", 1.0)
            self.exchange("PING", "PONG", 1.0)
        except BaseException:
            self.connection.close()
            raise

    def drain(self):
        size = self.connection.in_waiting
        if size:
            self.receive_buffer += self.connection.read(size)
        if len(self.receive_buffer) > 8192:
            raise SafetyFault("ESP32 response buffer overflow.")
        chunks = self.receive_buffer.split(b"\n")
        self.receive_buffer = chunks.pop()
        lines = [c.decode("ascii", errors="replace").strip() for c in chunks]
        errors = [line for line in lines if line.startswith("ERR,")]
        if errors:
            self.armed = False
            raise SafetyFault("ESP32 reported: "+"; ".join(errors))
        return lines

    def exchange(self, command, expected, timeout=0.08):
        self.drain()  # Never accept a buffered acknowledgement from an earlier command.
        payload = (command+"\n").encode("ascii")
        if self.connection.write(payload) != len(payload):
            self.armed = False
            raise SafetyFault("Incomplete serial write.")
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if expected in self.drain():
                return
            time.sleep(0.001)
        self.armed = False
        raise SafetyFault(f"Missing ESP32 acknowledgement: {expected}")

    def arm(self):
        if self.armed:
            raise SafetyFault("Already armed.")
        self.exchange("ARM", "OK,ARM", 1.0)
        self.armed = True

    def ping(self):
        self.exchange("PING", "PONG", 1.0)

    def send_speeds(self, values):
        if not self.armed:
            raise SafetyFault("Cannot send speeds until explicitly armed.")
        values = checked_speeds(values)
        self.sequence = (self.sequence+1) & 0xFFFFFFFF
        command = f"V,{self.sequence},{values[0]:.4f},{values[1]:.4f},{values[2]:.4f}"
        self.exchange(command, f"OK,V,{self.sequence}")

    def stop(self):
        self.armed = False
        if self.connection.is_open:
            # A previously reported watchdog fault must not prevent a STOP attempt.
            try:
                self.drain()
            except SafetyFault:
                self.receive_buffer = b""
            self.exchange("STOP", "OK,STOP", 1.0)

    def close(self):
        if self.connection.is_open:
            try:
                self.stop()
            finally:
                self.connection.close()


class MotionHistory:
    """Telemetry only: normalized speed-seconds never limit motor travel."""
    def __init__(self):
        self.signed = [0.0]*3
        self.cumulative = [0.0]*3

    def _future(self, speeds, seconds):
        speeds = checked_speeds(speeds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid elapsed time.")
        delta = [v*seconds for v in speeds]
        signed = [a+b for a, b in zip(self.signed, delta)]
        cumulative = [a+abs(b) for a, b in zip(self.cumulative, delta)]
        return signed, cumulative

    def update(self, speeds, seconds):
        self.signed, self.cumulative = self._future(speeds, seconds)


class WindingEstimate:
    """Approximate forward winding since manual home; never actual position.

    All accounting uses physical commands transformed by per-servo forward signs.
    The small residual and watchdog horizon limit estimated return overshoot.
    """
    RETURN_RATIO = 0.5
    RESIDUAL = 0.002
    RESERVE_SECONDS = 0.35

    def __init__(self, settings):
        self.settings = settings
        self.amounts = [0.0, 0.0, 0.0]

    def update(self, speeds, elapsed):
        speeds = checked_speeds(speeds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid winding-estimate time.")
        self.amounts = [max(0.0, amount+speed*sign*elapsed)
                        for amount, speed, sign in zip(self.amounts, speeds, self.settings.servo_forward_signs)]

    def coordinate(self, speeds):
        speeds = checked_speeds(speeds)
        signs = self.settings.servo_forward_signs
        logical = [speed*sign for speed, sign in zip(speeds, signs)]
        selected = [i for i, speed in enumerate(logical) if speed > 0]
        # Zero/neutral and non-forward commands never trigger automatic returns.
        if len(selected) != 1 or any(speed < 0 for speed in logical):
            return speeds
        selected = selected[0]
        half_speed = logical[selected]*self.RETURN_RATIO
        for i in range(3):
            if i == selected:
                continue
            available = max(0.0, self.amounts[i]-self.RESIDUAL)
            release_speed = min(half_speed, available/self.RESERVE_SECONDS)
            # Serial has four decimals: round DOWN so wire rounding cannot exceed cap.
            release_speed = math.floor(release_speed*10000.0)/10000.0
            logical[i] = -release_speed
        return tuple(speed*sign for speed, sign in zip(logical, signs))


class GuardedDrive:
    """Fresh-command transport guards only; no travel/time limit or rearm latch."""
    def __init__(self, link, settings: Settings, clock=time.monotonic, unloaded_test=False):
        if type(unloaded_test) is not bool:
            raise ValueError("unloaded_test must be a boolean.")
        self.link, self.settings, self.clock = link, settings, clock
        self.unloaded_test = unloaded_test
        self.history = MotionHistory()
        self.previous = ZERO
        self.accounted_at = clock()
        self.winding = WindingEstimate(settings)
        self.winding_at = self.accounted_at
        self.started_at = None
        self.armed = False

    def arm(self):
        if self.armed:
            raise SafetyFault("Already armed.")
        # Every READY-authorized arm/retry verifies a stopped, responsive controller.
        # No permanent latch and no reset of the command/winding history.
        self.link.stop()
        self.link.ping()
        self.link.arm()
        self.armed = True
        self.accounted_at = self.clock()
        self.winding_at = self.accounted_at
        if self.started_at is None:
            self.started_at = self.accounted_at
        self.send_speeds(ZERO)  # First command after ARM is always zero.

    def _account(self):
        now = self.clock()
        elapsed = now-self.accounted_at
        self.history.update(self.previous, elapsed)
        self.winding.update(self.previous, now-self.winding_at)
        self.winding_at = now
        self.accounted_at = now
        if self.armed and elapsed > self.settings.maximum_loop_gap_seconds:
            raise SafetyFault("Control loop stalled; stopped. Do not relax the watchdog.")

    def send_speeds(self, speeds):
        if not self.armed:
            raise SafetyFault("Drive is not armed.")
        speeds = checked_speeds(speeds)
        if any(abs(v) > self.settings.speed+1e-8 for v in speeds):
            raise ValueError("Speed exceeds the configured limit.")
        self._account()
        speeds = self.winding.coordinate(speeds)
        # Conservatively count serial transmission/ACK time as commanded motion.
        old = self.previous
        self.previous = speeds
        self.link.send_speeds(speeds)
        # During a write/ACK transition the switch time is unknown. Count the
        # more-releasing of old/new logical commands, avoiding optimistic winding.
        now = self.clock()
        transition = tuple(min(a*sign, b*sign)*sign for a,b,sign in
                           zip(old, speeds, self.settings.servo_forward_signs))
        self.winding.update(transition, now-self.winding_at)
        self.winding_at = now

    def _halt(self):
        old = self.previous
        try:
            if self.armed or old != ZERO:
                self._account()
        finally:
            try:
                self.link.stop()
            finally:
                now = self.clock()
                try:
                    # STOP acknowledgement is not instantaneous. Charge the full
                    # interval as the previous command for telemetry.
                    self.history.update(old, now-self.accounted_at)
                    transition = tuple(min(v*sign, 0.0)*sign for v,sign in
                                       zip(old, self.settings.servo_forward_signs))
                    self.winding.update(transition, now-self.winding_at)
                finally:
                    self.armed = False
                    self.previous = ZERO
                    self.accounted_at = self.winding_at = now

    def pause(self):
        """Stop/disarm without a permanent lock; READY-authorized control can retry."""
        self._halt()

    def stop(self):
        self._halt()
