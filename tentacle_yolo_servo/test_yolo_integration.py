"""CPU-only logic tests. No PyTorch, webcam, serial port, or motor is used."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tentacle_control_yolo as app
from yolo_pose_tracker import YoloTentacleTracker, TentacleObservation, tracker_signature


POINTS = np.array([[300.,420.],[300.,380.],[300.,290.],[300.,200.],[300.,100.]])


class CV2:
    FONT_HERSHEY_SIMPLEX=0
    LINE_AA=0
    WND_PROP_VISIBLE=0
    def __init__(self, keys=None): self.keys=iter(keys or [ord('q')])
    def polylines(self,*a): pass
    def circle(self,*a): pass
    def putText(self,*a): pass
    def imshow(self,*a): pass
    def waitKey(self,*a): return next(self.keys,ord('q'))
    def getWindowProperty(self,*a): return 1
    def resize(self,frame,size): return np.zeros((size[1],size[0],3),dtype=np.uint8)


class FakeYolo:
    task='pose'
    names={0:'tentacle'}
    def __init__(self):
        self.result=NS(boxes=NS(conf=np.array([.9]),cls=np.array([0.])),
                       keypoints=NS(xy=POINTS[None].copy(),conf=np.ones((1,5))*.95))
        self.calls=[]
    def predict(self,**kwargs):
        self.calls.append(kwargs)
        return [self.result]


class FakeDrive:
    def __init__(self):
        self.armed=True
        self.stops=0
        self.commands=[]
        self.budget=app.MotionBudget(.3,1.2)
    def stop(self): self.stops+=1; self.armed=False
    def send_speeds(self,x): self.commands.append(np.asarray(x).copy())


class Camera:
    def __init__(self,stale=False): self.stale=stale; self.checks=0
    def read(self): return np.zeros((480,640,3),dtype=np.uint8)
    def assert_fresh(self,settings):
        self.checks+=1
        if self.stale: raise RuntimeError('stale camera')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        (self.base/'best.pt').write_bytes(b'FAKE TEST WEIGHTS - not executable')
        self.settings=dict(app.DEFAULT_CONFIG)
        self.model=FakeYolo()
        self.tracker=YoloTentacleTracker(CV2(),self.settings,self.base,self.model)
    def tearDown(self): self.tmp.cleanup()
    def observation(self,points=POINTS,conf=None,length=280.):
        return self.tracker.validate_pose(points,np.ones(5)*.95 if conf is None else conf,
                                          (480,640,3),length)
    def test_straight_state_and_root(self):
        o=self.observation()
        np.testing.assert_allclose(o.state,[0,1])
        np.testing.assert_allclose(o.base_center,POINTS[1])
        self.assertAlmostEqual(o.straightness,1.)
    def test_left_and_right_state(self):
        p=POINTS.copy(); p[4,0]+=20
        self.assertGreater(self.observation(p).state[0],0)
    def test_missing_keypoint(self): self.assertIsNone(self.observation(POINTS[:4]))
    def test_low_confidence(self):
        c=np.ones(5)*.9;c[2]=.2
        self.assertIsNone(self.observation(conf=c))
    def test_nan(self):
        p=POINTS.copy();p[2,0]=np.nan
        self.assertIsNone(self.observation(p))
    def test_border(self):
        p=POINTS.copy();p[4,1]=1
        self.assertIsNone(self.observation(p))
    def test_base_axis_collapse(self):
        p=POINTS.copy();p[1]=p[0]+1
        self.assertIsNone(self.observation(p))
    def test_joint_overlap(self):
        p=POINTS.copy();p[2]=p[1]
        self.assertIsNone(self.observation(p))
    def test_tip_root_collapse(self):
        p=POINTS.copy();p[4]=p[1]+1
        self.assertIsNone(self.observation(p))
    def test_jump(self):
        self.observation();p=POINTS.copy();p[2,0]+=50
        self.assertIsNone(self.observation(p))
    def test_axis_rotation(self):
        self.observation();p=POINTS.copy();p[0,0]-=20
        self.assertIsNone(self.observation(p))
    def test_invalid_reference(self): self.assertIsNone(self.observation(length=float('nan')))
    def test_observe_valid(self): self.assertIsNotNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_multiple_instances_rejected(self):
        self.model.result.boxes=NS(conf=np.array([.9,.8]),cls=np.zeros(2))
        self.model.result.keypoints.xy=np.stack([POINTS,POINTS])
        self.model.result.keypoints.conf=np.ones((2,5))
        self.assertIsNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_no_keypoint_confidence(self):
        self.model.result.keypoints.conf=None
        self.assertIsNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_no_detection(self):
        self.model.result.boxes=None
        self.assertIsNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_low_box_confidence(self):
        self.model.result.boxes.conf=np.array([.1])
        self.assertIsNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_wrong_class(self):
        self.model.result.boxes.cls=np.array([1])
        self.assertIsNone(self.tracker.observe(np.zeros((480,640,3)),280.))
    def test_wrong_model_names(self):
        self.model.names={0:'person'}
        with self.assertRaises(ValueError): YoloTentacleTracker(CV2(),self.settings,self.base,self.model)
    def test_warmup_and_prediction_options(self):
        call=self.model.calls[0]
        self.assertEqual(call['max_det'],2)
        self.assertEqual(call['classes'],[0])
        self.assertFalse(call['save'])
        self.assertEqual(call['device'],'cpu')
    def test_signature_changes_with_weights(self):
        before=self.tracker.signature
        (self.base/'best.pt').write_bytes(b'OTHER TEST BYTES')
        self.assertNotEqual(before,tracker_signature(self.settings,self.base))
    def test_signature_changes_with_settings(self):
        altered=dict(self.settings,yolo_imgsz=320)
        self.assertNotEqual(self.tracker.signature,tracker_signature(altered,self.base))
    def test_calibration_roundtrip_and_mismatch(self):
        model=app.VisualJacobianModel(np.array([[1.,-1.,0.],[0.,0.,1.]]),280.,
             camera_index=0,frame_size=(640,480),tracker_signature=self.tracker.signature)
        path=self.base/'cal.json';model.save(path)
        loaded=app.VisualJacobianModel.load(path,self.tracker.signature)
        np.testing.assert_array_equal(model.jacobian,loaded.jacobian)
        with self.assertRaises(ValueError):app.VisualJacobianModel.load(path,'wrong')
    def test_aruco_calibration_rejected(self):
        p=self.base/'old.json';p.write_text(json.dumps({'model_type':'local_visual_jacobian_single_camera_v1'}))
        with self.assertRaises(ValueError):app.VisualJacobianModel.load(p,self.tracker.signature)
    def test_unsafe_config_rejected(self):
        for key,value in [('maximum_frame_age_seconds',.5),('maximum_loop_gap_seconds',.3),
                          ('maximum_speed',1.),('yolo_imgsz',100),('yolo_keypoint_confidence',.1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                app.validate_config(dict(self.settings,**{key:value}))
    def test_actual_config_valid(self):
        settings,base=app.load_config(Path(__file__).with_name('config_yolo.json'))
        self.assertEqual(settings['camera'],0)
        self.assertEqual(settings['serial_port'],'COM3')
    def test_shared_frame_copies(self):
        class Hand:
            def process(self,cv,frame): frame[:]=99;return np.array([0.,1.])
        class Pose:
            def observe(self,frame,ref):
                np.testing.assert_array_equal(frame,np.full((8,8,3),42));frame[:]=55
                return None
        raw=np.full((8,8,3),42)
        tracking=app.SharedFrameTracker(None,Hand(),Pose())
        target,obs,h,p=tracking.process(raw,None)
        self.assertTrue((raw==42).all());self.assertTrue((h==99).all());self.assertTrue((p==55).all())
    def make_application(self,camera=None):
        model=app.VisualJacobianModel(np.array([[1.,-1.,0.],[0.,0.,1.]]),280.,frame_size=(640,480))
        drive=FakeDrive()
        application=app.TentacleApplication(CV2(),camera or Camera(),None,None,model,drive,self.settings)
        application.armed=True
        class Tracking:
            def process(self,raw,reference): return None,None,raw.copy(),raw.copy()
        application.tracking=Tracking()
        return application,drive
    def test_missing_detection_stops_drive(self):
        application,drive=self.make_application()
        application.run()
        self.assertFalse(application.armed);self.assertTrue(application.hard_fault)
        self.assertGreater(drive.stops,0);self.assertEqual(drive.commands,[])
    def test_stale_after_inference_stops_drive(self):
        application,drive=self.make_application(Camera(stale=True))
        with self.assertRaises(RuntimeError):application.run()
        self.assertGreater(drive.stops,0);self.assertEqual(drive.commands,[])
    def test_guard_rejects_slow_loop(self):
        drive=FakeDrive()
        guard=app.GuardedDrive(drive,self.settings)
        guard.armed=True;guard.accounted_at=app.time.monotonic()-1.
        with self.assertRaises(RuntimeError):guard.send_speeds(np.zeros(3))
        self.assertEqual(drive.commands,[])
    def test_stop_even_if_accounting_fails(self):
        drive=FakeDrive();guard=app.GuardedDrive(drive,self.settings)
        guard.armed=True;guard.accounted_at=app.time.monotonic()-1.
        with self.assertRaises(RuntimeError):guard.stop()
        self.assertGreater(drive.stops,0);self.assertFalse(guard.armed)
    def test_budget_reserve(self):
        budget=app.MotionBudget(.05,.1)
        with self.assertRaises(RuntimeError):budget.check_future(np.array([.2,0,0]),.35)
    def test_calibration_lost_pose_stops(self):
        drive=FakeDrive()
        tracker=NS(observe=lambda *a:None)
        trainer=app.CalibrationTrainer(CV2(),Camera(),tracker,drive,self.settings)
        with self.assertRaises(RuntimeError):trainer.pulse(0,.1,.1)
        self.assertEqual(drive.stops,1);self.assertEqual(drive.commands,[])
    def test_calibration_home_and_arm_do_not_block_terminal(self):
        cv=CV2([-1]*14+[ord('s')]+[-1]*14+[ord('a')])
        observation=TentacleObservation(np.array([0.,1.]),POINTS[1],POINTS[4],280.,1.)
        tracker=NS(observe=lambda *a:observation,last_rejection='None')
        camera=Camera();drive=FakeDrive();drive.armed=False
        trainer=app.CalibrationTrainer(cv,camera,tracker,drive,self.settings)
        with patch('builtins.input',side_effect=AssertionError('Unexpected terminal input')):
            trainer.wait_for_straight_home();trainer.wait_for_visual_arm()
        self.assertEqual(trainer.reference_length,280.)
        self.assertEqual(drive.commands,[]);self.assertFalse(drive.armed)
    def test_parser_preview_has_no_motion_flag(self):
        self.assertEqual(app.build_parser().parse_args(['preview']).mode,'preview')
        self.assertFalse(app.build_parser().parse_args(['run']).enable_motion)
        self.assertEqual(app.build_parser().parse_args(['calibrate']).mode,'calibrate')
    def test_acknowledgement_timeouts_preserved(self):
        from unittest.mock import Mock
        link=app.ESP32Link.__new__(app.ESP32Link)
        link.connection=NS(is_open=True)
        link._exchange=Mock()
        link.arm()
        link._exchange.assert_called_with('ARM','OK,ARM',timeout=1.0)
        self.assertTrue(link.armed)
        link.stop()
        link._exchange.assert_called_with('STOP','OK,STOP',timeout=1.0)
        self.assertFalse(link.armed)
    def test_velocity_protocol_and_bounds(self):
        from unittest.mock import Mock
        link=app.ESP32Link.__new__(app.ESP32Link)
        link.armed=True;link.sequence=0;link._exchange=Mock()
        link.send_speeds(np.array([.1,-.1,0.]))
        link._exchange.assert_called_once_with('V,1,0.1000,-0.1000,0.0000','OK,V,1')
        with self.assertRaises(ValueError):link.send_speeds(np.array([1.,0.,0.]))
        link.armed=False
        with self.assertRaises(RuntimeError):link.send_speeds(np.zeros(3))


if __name__=='__main__':unittest.main()
