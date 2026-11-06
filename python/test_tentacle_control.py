"""Hardware-free checks for the vision controller's math and serial protocol."""

import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import tentacle_control as control


class CalibrationTests(unittest.TestCase):
    def make_trainer(self, clock, marker_visible=True):
        class FakeLink:
            def __init__(self):
                self.commands = []
                self.stopped = False

            def arm(self):
                pass

            def send_speeds(self, speeds):
                self.commands.append(np.asarray(speeds).copy())

            def stop(self):
                self.stopped = True

        class FakeCamera:
            def read(self):
                clock[0] += 0.005
                return np.zeros((4, 4, 3))

        cv = types.SimpleNamespace(imshow=lambda *a: None, waitKey=lambda *a: -1)
        observation = types.SimpleNamespace(state=np.array([0., 1.]))
        tracker = types.SimpleNamespace(observe=lambda *a: observation if marker_visible else None)
        link = FakeLink()
        guarded = control.GuardedDrive(link, dict(control.DEFAULT_CONFIG))
        guarded.arm()
        trainer = control.CalibrationTrainer(cv, FakeCamera(), tracker, guarded,
                                             dict(control.DEFAULT_CONFIG))
        trainer.reference_length = 180
        return trainer, guarded, link

    def test_settling_sends_zero_heartbeats(self):
        clock = [10.]
        def advance(seconds):
            clock[0] += seconds
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(control.time, "sleep", side_effect=advance), \
                patch.object(control, "put_lines"):
            trainer, _, link = self.make_trainer(clock)
            state = trainer.sample_state(0.35)
        self.assertGreater(len(link.commands), 3)
        self.assertTrue(all(np.allclose(command, 0) for command in link.commands))
        np.testing.assert_allclose(state, [0, 1])

    def test_pulse_accounts_observed_command_time_and_finishes_neutral(self):
        clock = [10.]
        def advance(seconds):
            clock[0] += seconds
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(control.time, "sleep", side_effect=advance), \
                patch.object(control, "put_lines"):
            trainer, guarded, link = self.make_trainer(clock)
            exposure = trainer.pulse(0, 0.18, 0.18)
        self.assertGreater(exposure, 0.03)
        self.assertLess(exposure, 0.05)
        self.assertAlmostEqual(exposure, guarded.budget.signed[0])
        np.testing.assert_allclose(link.commands[-1], np.zeros(3))

    def test_lost_marker_stops_calibration(self):
        clock = [10.]
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]):
            trainer, _, link = self.make_trainer(clock, marker_visible=False)
            with self.assertRaisesRegex(RuntimeError, "marker lost"):
                trainer.sample_state(0.35)
        self.assertTrue(link.stopped)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, -1.0]])
        self.model = control.VisualJacobianModel(self.matrix, 180.0)

    def test_full_rank_model(self):
        rank, condition = self.model.rank_and_condition()
        self.assertEqual(rank, 2)
        self.assertTrue(np.isfinite(condition))

    def test_rank_one_model_is_usable(self):
        model = control.VisualJacobianModel([[1, -1, 0], [0, 0, 0]], 180)
        self.assertEqual(model.rank_and_condition()[0], 1)
        np.testing.assert_allclose(model.projected_error([0.2, 0.8]), [0.2, 0])

    def test_unobservable_error_does_not_create_motor_motion(self):
        model = control.VisualJacobianModel([[1, -1, 0], [0, 0, 0]], 180)
        command = model.command([0, 0.8], np.zeros(3), 0.8, 0.08, 0, 0.28)
        np.testing.assert_allclose(command, np.zeros(3), atol=1e-12)

    def test_task_motion_reduces_error(self):
        error = np.array([0.08, -0.05])
        command = self.model.command(error, np.zeros(3), 0.8, 0.08, 0, 0.28)
        self.assertGreater(float(error @ (self.matrix @ command)), 0)

    def test_command_speed_is_bounded(self):
        command = self.model.command([50, -80], np.zeros(3), 0.8, 0.08, 0, 0.28)
        self.assertLessEqual(float(np.max(np.abs(command))), 0.280000001)

    def test_nullspace_recovery_does_not_change_task(self):
        command = self.model.command([0, 0], np.ones(3), 0.8, 0.08, 0.25, 0.28)
        np.testing.assert_allclose(self.matrix @ command, [0, 0], atol=1e-12)
        self.assertLess(float(np.sum(command)), 0)

    def test_zero_or_noise_only_model_is_rejected(self):
        for matrix in (np.zeros((2, 3)), np.ones((2, 3)) * 0.001):
            with self.subTest(matrix=matrix.tolist()):
                with self.assertRaises(ValueError):
                    control.VisualJacobianModel(matrix, 180)

    def test_invalid_model_states_are_rejected(self):
        cases = [
            (np.zeros((3, 3)), 180, None),
            (np.full((2, 3), np.nan), 180, None),
            (self.matrix, float("inf"), None),
            (self.matrix, 0, None),
            (self.matrix, 180, [np.nan, 1]),
            (self.matrix, 180, [0, 1, 2]),
        ]
        for matrix, length, home in cases:
            with self.subTest(length=length, home=home):
                with self.assertRaises(ValueError):
                    control.VisualJacobianModel(matrix, length, home)

    def test_saved_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = control.VisualJacobianModel.load(path)
        np.testing.assert_allclose(loaded.jacobian, self.matrix)
        np.testing.assert_allclose(loaded.home_state, [0, 1])
        self.assertEqual(loaded.straight_length_pixels, 180)


class BudgetTests(unittest.TestCase):
    def test_signed_and_cumulative_have_different_meanings(self):
        budget = control.MotionBudget(1, 2)
        budget.update(np.array([0.2, -0.1, 0]), 0.5)
        budget.update(np.array([-0.2, 0.1, 0]), 0.5)
        np.testing.assert_allclose(budget.signed, [0, 0, 0])
        np.testing.assert_allclose(budget.cumulative, [0.2, 0.1, 0])

    def test_true_elapsed_time_is_not_clipped(self):
        budget = control.MotionBudget(1, 2)
        budget.update(np.array([0.2, 0, 0]), 1.5)
        np.testing.assert_allclose(budget.signed, [0.3, 0, 0])

    def test_signed_limit_faults(self):
        with self.assertRaisesRegex(RuntimeError, "spool-travel"):
            control.MotionBudget(0.1, 2).update(np.array([0.2, 0, 0]), 1)

    def test_cumulative_limit_faults_after_reversal(self):
        budget = control.MotionBudget(1, 0.15)
        budget.update(np.array([0.2, 0, 0]), 0.5)
        with self.assertRaisesRegex(RuntimeError, "Cumulative"):
            budget.update(np.array([-0.2, 0, 0]), 0.5)

    def test_future_reserve_does_not_mutate_budget(self):
        budget = control.MotionBudget(0.1, 0.2)
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            budget.check_future(np.array([0.3, 0, 0]), 0.5)
        np.testing.assert_allclose(budget.signed, [0, 0, 0])
        np.testing.assert_allclose(budget.cumulative, [0, 0, 0])

    def test_invalid_limits_are_rejected(self):
        for limits in ((0, 1), (1, -1), (np.nan, 1), (1, np.inf)):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    control.MotionBudget(*limits)

    def test_invalid_updates_are_rejected(self):
        cases = [([0, 0], 0.1), ([np.nan, 0, 0], 0.1), ([0, 0, 0], -1),
                 ([0, 0, 0], np.inf)]
        for command, elapsed in cases:
            with self.subTest(command=command, elapsed=elapsed):
                with self.assertRaises(ValueError):
                    control.MotionBudget(1, 2).update(np.array(command), elapsed)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        control.validate_config(copy.deepcopy(control.DEFAULT_CONFIG))

    def test_unsafe_or_inconsistent_settings_are_rejected(self):
        cases = {
            "maximum_speed": 2,
            "damping": 0,
            "gain": np.nan,
            "control_hz": 0,
            "calibration_pulse_seconds": 5,
            "calibration_repeats": 2.5,
            "camera": -1,
            "baudrate": 9600,
            "tip_marker_id": 0,
        }
        for key, value in cases.items():
            settings = copy.deepcopy(control.DEFAULT_CONFIG)
            settings[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    control.validate_config(settings)


class FakeSerial:
    def __init__(self, chunk_size=8192):
        self.is_open = False
        self.received = bytearray()
        self.commands = []
        self.chunk_size = chunk_size
        self.respond = True
        self.short_write = False

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        self.received.clear()

    @property
    def in_waiting(self):
        return min(len(self.received), self.chunk_size)

    def read(self, count):
        result = bytes(self.received[:count])
        del self.received[:count]
        return result

    def write(self, payload):
        self.commands.append(payload.decode("ascii").strip())
        if self.respond:
            command = self.commands[-1]
            if command == "PING":
                response = "PONG"
            elif command == "ARM":
                response = "OK,ARM"
            elif command == "STOP":
                response = "OK,STOP"
            elif command.startswith("V,"):
                response = "OK,V," + command.split(",")[1]
            else:
                response = "ERR,BAD_COMMAND"
            self.received.extend((response + "\r\n").encode("ascii"))
        return len(payload) - int(self.short_write)

    def flush(self):
        pass


class SerialTests(unittest.TestCase):
    def make_link(self, chunk_size=8192):
        connection = FakeSerial(chunk_size)
        module = types.SimpleNamespace(Serial=lambda: connection)
        with patch.object(control.time, "sleep", return_value=None):
            link = control.ESP32Link(module, "COM_TEST", 115200)
        return link, connection

    def test_handshake_and_fragmented_response(self):
        link, serial = self.make_link(chunk_size=2)
        self.assertEqual(serial.commands[:2], ["STOP", "PING"])
        link.arm()
        self.assertTrue(link.armed)
        link.send_speeds(np.array([0.1, -0.2, 0]))
        self.assertEqual(serial.commands[-1], "V,1,0.1000,-0.2000,0.0000")
        link.close()
        self.assertFalse(serial.is_open)

    def test_unarmed_velocity_is_rejected(self):
        link, _ = self.make_link()
        with self.assertRaisesRegex(RuntimeError, "until"):
            link.send_speeds(np.zeros(3))

    def test_invalid_velocity_is_rejected(self):
        link, _ = self.make_link()
        link.arm()
        for command in ([0, 0], [np.nan, 0, 0], [0.5, 0, 0]):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    link.send_speeds(np.array(command))

    def test_partial_lines_are_retained(self):
        link, serial = self.make_link()
        serial.received.extend(b"OK,V,")
        self.assertEqual(link.drain_responses(), [])
        serial.received.extend(b"42\r\n")
        self.assertEqual(link.drain_responses(), ["OK,V,42"])

    def test_firmware_error_disarms_host(self):
        link, serial = self.make_link()
        link.arm()
        serial.received.extend(b"ERR,WATCHDOG\r\n")
        with self.assertRaisesRegex(RuntimeError, "WATCHDOG"):
            link.drain_responses()
        self.assertFalse(link.armed)

    def test_missing_ack_disarms_host(self):
        link, serial = self.make_link()
        link.arm()
        serial.respond = False
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            link._exchange("PING", "PONG", timeout=0.002)
        self.assertFalse(link.armed)

    def test_short_write_is_not_silently_accepted(self):
        link, serial = self.make_link()
        serial.short_write = True
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            link._write("PING")

    def test_unterminated_response_overflow_faults(self):
        link, serial = self.make_link()
        serial.chunk_size = 10000
        serial.received.extend(b"X" * 8193)
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            link.drain_responses()


class GuardedDriveTests(unittest.TestCase):
    def make_drive(self, clock):
        self.link = types.SimpleNamespace(
            arm=lambda: None, send_speeds=lambda speeds: None,
            stop=lambda: None,
        )
        with patch.object(control.time, "monotonic", return_value=clock[0]):
            drive = control.GuardedDrive(self.link, copy.deepcopy(control.DEFAULT_CONFIG))
            drive.arm()
        return drive

    def test_accounts_previous_command_not_new_command(self):
        clock = [0.0]
        drive = self.make_drive(clock)
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]):
            clock[0] = 0.05
            drive.send_speeds(np.array([0.2, 0, 0]))
            clock[0] = 0.10
            drive.send_speeds(np.zeros(3))
        np.testing.assert_allclose(drive.budget.signed, [0.01, 0, 0])

    def test_long_loop_gap_faults(self):
        clock = [0.0]
        drive = self.make_drive(clock)
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]):
            clock[0] = 0.5
            with self.assertRaisesRegex(RuntimeError, "stalled"):
                drive.send_speeds(np.zeros(3))

    def test_stop_still_sends_stop_when_accounting_faults(self):
        clock = [0.0]
        drive = self.make_drive(clock)
        calls = []
        self.link.stop = lambda: calls.append("STOP")
        with patch.object(control.time, "monotonic", side_effect=lambda: clock[0]):
            clock[0] = 0.5
            with self.assertRaises(RuntimeError):
                drive.stop()
        self.assertEqual(calls, ["STOP"])
        self.assertFalse(drive.armed)


if __name__ == "__main__":
    unittest.main()
