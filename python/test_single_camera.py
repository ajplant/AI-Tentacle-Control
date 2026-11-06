"""Hardware-free regression checks for one-webcam tracking and startup safety."""

import copy
from pathlib import Path
import queue
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import tentacle_control as control


class SharedFrameTests(unittest.TestCase):
    def test_trackers_receive_isolated_copies_of_the_same_raw_pixels(self):
        raw = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        original = raw.copy()
        received = {}
        target = np.array([0.1, 0.9])
        observation = object()

        def hand_process(cv, frame):
            received["hand"] = frame.copy()
            frame[:] = 25
            return target

        def tentacle_process(frame, reference_length):
            received["tentacle"] = frame.copy()
            self.assertEqual(reference_length, 180)
            frame[:] = 75
            return observation

        cv = types.SimpleNamespace(flip=Mock(side_effect=AssertionError("No mirroring allowed")))
        tracker = control.SharedFrameTracker(
            cv, types.SimpleNamespace(process=hand_process),
            types.SimpleNamespace(observe=tentacle_process)
        )
        result_target, result_observation, hand_frame, tentacle_frame = tracker.process(raw, 180)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(received["hand"], original)
        np.testing.assert_array_equal(received["tentacle"], original)
        self.assertFalse(np.shares_memory(hand_frame, raw))
        self.assertFalse(np.shares_memory(tentacle_frame, raw))
        self.assertFalse(np.shares_memory(hand_frame, tentacle_frame))
        self.assertTrue(np.all(hand_frame == 25))
        self.assertTrue(np.all(tentacle_frame == 75))
        np.testing.assert_allclose(result_target, target)
        self.assertIs(result_observation, observation)
        cv.flip.assert_not_called()

    def test_missing_detection_is_propagated_without_a_fake_measurement(self):
        hand = types.SimpleNamespace(process=lambda *args: None)
        tentacle = types.SimpleNamespace(observe=lambda *args: None)
        result = control.SharedFrameTracker(object(), hand, tentacle).process(np.zeros((3, 3, 3)), None)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])


class CameraStartupTests(unittest.TestCase):
    def make_camera(self):
        camera = control.Camera.__new__(control.Camera)
        camera.name = "shared webcam"
        camera.failure = None
        camera.first_read = True
        camera.has_frame = threading.Event()
        camera.closed = threading.Event()
        camera.frames = queue.Queue(maxsize=1)
        camera.startup_deadline = 105.0
        return camera

    def test_only_the_first_read_uses_the_long_startup_wait(self):
        camera = self.make_camera()
        frame = np.zeros((3, 3, 3))
        camera.frames = Mock()
        camera.frames.get.return_value = (100.0, frame)
        with patch.object(control.time, "monotonic", return_value=100.0):
            camera.read()
            camera.read()
        self.assertEqual([call.kwargs["timeout"] for call in camera.frames.get.call_args_list], [5.0, 0.15])

    def test_initial_failed_reads_are_retried_but_runtime_failure_stops_capture(self):
        camera = self.make_camera()
        frame = np.ones((3, 3, 3))
        camera.capture = types.SimpleNamespace(
            read=Mock(side_effect=[(False, None), (False, None), (True, frame), (False, None)])
        )
        camera.closed = Mock()
        camera.closed.is_set.return_value = False
        with patch.object(control.time, "monotonic", return_value=100.0):
            camera._capture()
        self.assertEqual(camera.capture.read.call_count, 4)
        self.assertEqual(camera.closed.wait.call_count, 2)
        self.assertTrue(camera.has_frame.is_set())
        self.assertIn("stopped", camera.failure)

    def test_startup_retry_has_a_finite_deadline(self):
        camera = self.make_camera()
        camera.capture = types.SimpleNamespace(read=Mock(return_value=(False, None)))
        with patch.object(control.time, "monotonic", return_value=106.0):
            camera._capture()
        self.assertEqual(camera.capture.read.call_count, 1)
        self.assertIsNotNone(camera.failure)

    def test_stale_frame_is_still_rejected_during_startup(self):
        camera = self.make_camera()
        camera.frames.put((99.0, np.zeros((3, 3, 3))))
        with patch.object(control.time, "monotonic", return_value=100.0):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                camera.read()

    def test_runtime_empty_queue_remains_a_fault(self):
        camera = self.make_camera()
        camera.first_read = False
        camera.frames = Mock()
        camera.frames.get.side_effect = queue.Empty
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            camera.read()
        camera.frames.get.assert_called_once_with(timeout=0.15)


class SingleCameraConfigTests(unittest.TestCase):
    def test_one_camera_index_is_valid(self):
        settings = copy.deepcopy(control.DEFAULT_CONFIG)
        settings["camera"] = 0
        control.validate_config(settings)

    def test_camera_index_must_be_a_nonnegative_integer(self):
        for value in (-1, 0.5, True):
            with self.subTest(value=value):
                settings = copy.deepcopy(control.DEFAULT_CONFIG)
                settings["camera"] = value
                with self.assertRaises(ValueError):
                    control.validate_config(settings)

    def test_legacy_two_camera_config_requires_explicit_migration(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"hand_camera": 0, "tentacle_camera": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                control.load_config(path)


class RuntimeSafetyTests(unittest.TestCase):
    def check_loss_is_latched(self, missing_hand=False):
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        camera = types.SimpleNamespace(read=Mock(return_value=frame))
        cv = types.SimpleNamespace(
            imshow=Mock(), waitKey=Mock(side_effect=[-1, ord("a"), ord("q")]),
            getWindowProperty=Mock(return_value=1), WND_PROP_VISIBLE=0,
            flip=Mock(side_effect=AssertionError("Do not mirror tracking input")),
        )
        link = Mock()
        link.armed = True
        link.stop.side_effect = lambda: setattr(link, "armed", False)
        settings = copy.deepcopy(control.DEFAULT_CONFIG)
        settings["stable_frames_before_arm"] = 1
        model = control.VisualJacobianModel([[1, -1, 0], [0, 0, 0]], 180)
        app = control.TentacleApplication(cv, camera, Mock(), Mock(), model, link, settings)
        app.armed = True
        target = np.array([0., 1.])
        observation = types.SimpleNamespace(state=target.copy())
        absent = (None, observation) if missing_hand else (target, None)
        app.tracking = types.SimpleNamespace(process=Mock(side_effect=[
            (*absent, frame.copy(), frame.copy()),
            (target, observation, frame.copy(), frame.copy()),
            (target, observation, frame.copy(), frame.copy()),
        ]))
        with patch.object(control, "put_lines"), \
                patch.object(control, "resized", side_effect=lambda cv, value: value), \
                patch.object(control.time, "sleep", return_value=None):
            app.run()
        self.assertFalse(app.armed)
        self.assertTrue(app.hard_fault)
        link.stop.assert_called_once()
        link.arm.assert_not_called()
        link.send_speeds.assert_not_called()
        self.assertEqual(camera.read.call_count, 3)
        self.assertEqual(app.tracking.process.call_count, 3)
        cv.flip.assert_not_called()

    def test_marker_occlusion_latches_stop_and_reacquisition_cannot_rearm(self):
        self.check_loss_is_latched(missing_hand=False)

    def test_hand_loss_latches_stop_and_reacquisition_cannot_rearm(self):
        self.check_loss_is_latched(missing_hand=True)


class SetupTests(unittest.TestCase):
    def test_initial_preview_opens_one_camera_and_reads_one_shared_frame(self):
        settings = copy.deepcopy(control.DEFAULT_CONFIG)
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        camera = Mock()
        camera.read.return_value = frame
        hand = Mock()
        hand.process.return_value = np.array([0., 1.])
        tentacle = Mock()
        tentacle.observe.return_value = None
        cv = types.SimpleNamespace(
            destroyAllWindows=Mock(), imshow=Mock(), waitKey=Mock(return_value=ord("q"))
        )
        with patch.object(control, "Camera", return_value=camera) as factory, \
                patch.object(control, "HandTargetTracker", return_value=hand), \
                patch.object(control, "ArucoTentacleTracker", return_value=tentacle), \
                patch.object(control, "put_lines"), \
                patch.object(control, "resized", side_effect=lambda cv, value: value), \
                patch.object(control, "ESP32Link", side_effect=AssertionError("No serial in preview")):
            control.preview_cameras(cv, settings, Path("."))
        factory.assert_called_once()
        camera.read.assert_called_once()
        camera.close.assert_called_once()
        hand.process.assert_called_once()
        tentacle.observe.assert_called_once()
        np.testing.assert_array_equal(hand.process.call_args.args[1], frame)
        np.testing.assert_array_equal(tentacle.observe.call_args.args[0], frame)

    def test_run_preview_opens_exactly_one_capture_and_no_serial_link(self):
        settings = copy.deepcopy(control.DEFAULT_CONFIG)
        camera = Mock()
        cv = types.SimpleNamespace(destroyAllWindows=Mock())
        hand = Mock()
        model = control.VisualJacobianModel([[1, -1, 0], [0, 0, 0]], 180,
                                            camera_index=settings["camera"], frame_size=(4, 3))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / settings["visual_model"]).touch()
            with patch.object(control.VisualJacobianModel, "load", return_value=model), \
                    patch.object(control, "Camera", return_value=camera) as factory, \
                    patch.object(control, "HandTargetTracker", return_value=hand), \
                    patch.object(control, "ArucoTentacleTracker", return_value=Mock()), \
                    patch.object(control, "TentacleApplication") as application, \
                    patch.object(control, "ESP32Link", side_effect=AssertionError("Preview must not open serial")):
                control.run_controller(cv, Mock(), settings, base, enable_motion=False)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.args[1], settings["camera"])
                self.assertIs(application.call_args.args[1], camera)
                application.return_value.run.assert_called_once()
        camera.close.assert_called_once()
        hand.close.assert_called_once()


class CalibrationMetadataTests(unittest.TestCase):
    def test_old_two_camera_model_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps({
                "model_type": "local_visual_jacobian_2x3",
                "jacobian": [[1, -1, 0], [0, 0, 0]],
                "straight_length_pixels": 180,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retrain"):
                control.VisualJacobianModel.load(path)

    def test_shared_camera_metadata_round_trips(self):
        model = control.VisualJacobianModel([[1, -1, 0], [0, 0, 0]], 180,
                                            camera_index=2, frame_size=(640, 480))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = control.VisualJacobianModel.load(path)
        self.assertEqual(loaded.camera_index, 2)
        self.assertEqual(loaded.frame_size, (640, 480))


if __name__ == "__main__":
    unittest.main()
