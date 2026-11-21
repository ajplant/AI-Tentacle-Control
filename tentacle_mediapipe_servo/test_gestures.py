"""Hardware-free tests: python -m unittest discover -s <this folder> -v."""
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gesture_logic as logic
import finger_servo_control as control
import servo_link as transport


def finger(dx=0.24, dy=0, dz=0):
    points = [[0.5, 0.5, 0.0] for _ in range(21)]
    points[0] = [0.5, 0.75, 0]
    points[9] = [0.5, 0.55, 0]
    for i, t in zip((5, 6, 7, 8), (0, 1/3, 2/3, 1)):
        points[i] = [0.5+dx*t, 0.5+dy*t, dz*t]
    return points


def observation(gesture=logic.STOP, arrow_active=False, valid=True, label="Right"):
    return logic.Observation(valid, arrow_active, gesture, hand_label=label)


class SettingsTests(unittest.TestCase):
    def test_defaults_and_actual_config(self):
        self.assertEqual(logic.Settings().validate().away_mode, "toward-camera")
        config = Path(__file__).parent/'config_gestures.json'
        self.assertEqual(logic.load_settings(config), logic.Settings())

    def test_nonfinite_or_unsafe_settings_rejected(self):
        for change in ({'speed':math.nan}, {'speed':0.36}, {'speed':True},
                       {'maximum_frame_age_seconds':0.5}, {'maximum_loop_gap_seconds':0.5},
                       {'camera':True}, {'mirror':'yes'}, {'baudrate':9600},
                       {'away_mode':'guess'}, {'servo_forward_signs':(1,0,1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(logic.Settings(), **change).validate()

    def test_cli_override_is_explicit(self):
        config = Path(__file__).parent/'config_gestures.json'
        settings = logic.load_settings(config, away_mode='down', serial_port='COM8')
        self.assertEqual(settings.away_mode, 'down')
        self.assertEqual(settings.serial_port, 'COM8')


class DirectionTests(unittest.TestCase):
    def classify(self, points, **settings):
        return logic.classify_hand(points,640,480,replace(logic.Settings(),**settings),'Right')

    def test_right(self):
        result = self.classify(finger())
        self.assertEqual(result.gesture,logic.RIGHT)
        self.assertAlmostEqual(result.angle_degrees,0)

    def test_left(self):
        result = self.classify(finger(dx=-0.24))
        self.assertEqual(result.gesture,logic.LEFT)
        self.assertAlmostEqual(result.angle_degrees,180)

    def test_up(self):
        result = self.classify(finger(dx=0,dy=-0.24))
        self.assertEqual(result.gesture,logic.UP)
        self.assertAlmostEqual(result.angle_degrees,90)

    def test_down_not_guessed_as_away(self):
        self.assertEqual(self.classify(finger(dx=0,dy=0.24)).gesture,logic.STOP)

    def test_explicit_down_mapping(self):
        self.assertEqual(self.classify(finger(dx=0,dy=0.24),away_mode='down').gesture,logic.AWAY)

    def test_from_camera_depth_positive(self):
        self.assertEqual(self.classify(finger(dx=0.01,dz=0.20),away_mode='from-camera').gesture,logic.AWAY)

    def test_toward_camera_depth_negative(self):
        self.assertEqual(self.classify(finger(dx=0.01,dz=-0.20),away_mode='toward-camera').gesture,logic.AWAY)

    def test_opposite_depth_does_not_trigger(self):
        self.assertEqual(self.classify(finger(dx=0.01,dz=-0.20),away_mode='from-camera').gesture,logic.STOP)

    def test_depth_can_be_explicitly_disabled(self):
        self.assertEqual(self.classify(finger(dx=0.01,dz=-0.20),away_mode='disabled').gesture,logic.STOP)

    def test_diagonal_is_dead_zone(self):
        self.assertEqual(self.classify(finger(dx=0.18,dy=-0.24)).gesture,logic.STOP)

    def test_pixel_aspect_ratio_used(self):
        result = self.classify(finger(dx=0.18,dy=-0.24))
        self.assertAlmostEqual(result.angle_degrees,45)

    def test_invalid_landmarks_stop(self):
        for points in ([],finger()[:20],[[math.nan,0,0]]*21,[[0.5,0.5]]*21):
            with self.subTest(points=points):
                self.assertFalse(self.classify(points).valid_hand)

    def test_edge_and_small_hand_stop(self):
        points = finger(); points[8][0] = 1.1
        self.assertFalse(self.classify(points).valid_hand)
        points = finger(); points[0] = points[9]
        self.assertFalse(self.classify(points).valid_hand)

    def test_short_arrow_is_neutral(self):
        points=finger()
        points[6]=[0.5,0.4,0];points[7]=[0.55,0.43,0];points[8]=[0.52,0.48,0]
        result=self.classify(points)
        self.assertTrue(result.valid_hand)
        self.assertFalse(result.arrow_active)
        self.assertEqual(result.gesture,logic.STOP)

    def test_exact_servo_mapping(self):
        s=logic.Settings()
        self.assertEqual(logic.speeds_for(logic.LEFT,s),(0,0.18,0))
        self.assertEqual(logic.speeds_for(logic.RIGHT,s),(0,0,0.18))
        self.assertEqual(logic.speeds_for(logic.UP,s),(-0.18,-0.18,-0.18))
        self.assertEqual(logic.speeds_for(logic.AWAY,s),(0.18,0,0))
        self.assertEqual(logic.speeds_for(logic.AWAY,replace(s,away_mode='down')),(0.18,0,0))

    def test_per_servo_direction_signs(self):
        s=replace(logic.Settings(),servo_forward_signs=(1,-1,1))
        self.assertEqual(logic.speeds_for(logic.UP,s),(-0.18,0.18,-0.18))
        self.assertEqual(logic.speeds_for(logic.LEFT,s),(0,-0.18,0))


class ArrowTests(unittest.TestCase):
    def test_bent_finger_like_example_uses_arrow_angle(self):
        points=finger()
        points[5]=[0.65,0.65,0]
        points[6]=[0.52,0.35,0]
        points[7]=[0.30,0.30,0]
        points[8]=[0.20,0.45,0]
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertTrue(result.arrow_active)
        self.assertEqual(result.gesture,logic.LEFT)
        self.assertAlmostEqual(result.angle_degrees,
                               math.degrees(math.atan2(0.20*480, -0.45*640)))

    def test_bent_right_finger_is_accepted(self):
        points=finger();points[6]=[0.48,0.3,0];points[7]=[0.60,0.35,0]
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertEqual(result.gesture,logic.RIGHT)

    def test_bent_up_finger_is_accepted(self):
        points=finger(dx=0,dy=-0.24)
        points[6]=[0.3,0.4,0];points[7]=[0.3,0.25,0]
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertEqual(result.gesture,logic.UP)

    def test_bent_depth_finger_is_accepted(self):
        points=finger(dx=0.01,dz=-0.20)
        points[6]=[0.3,0.35,0];points[7]=[0.4,0.30,0]
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertEqual(result.gesture,logic.AWAY)

    def test_changing_middle_joints_does_not_change_arrow(self):
        points=finger();expected=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        points[6]=points[5];points[7]=points[5]
        actual=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertEqual(actual,expected)

    def test_world_joint_geometry_does_not_gate_arrow(self):
        points=finger()
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right',[[0,0,0]]*21)
        self.assertEqual(result.gesture,logic.RIGHT)

    def test_coincident_tip_is_valid_neutral_not_tracking_failure(self):
        points=finger();points[8]=points[5]
        result=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        self.assertTrue(result.valid_hand)
        self.assertFalse(result.arrow_active)
        self.assertEqual(result.gesture,logic.STOP)

    def test_neutral_length_scales_with_palm(self):
        settings=logic.Settings()
        small=finger(dx=0.04)
        neutral=logic.classify_hand(small,640,480,settings,'Right')
        self.assertFalse(neutral.arrow_active)
        small[0]=[0.5,0.65,0]
        active=logic.classify_hand(small,640,480,settings,'Right')
        self.assertTrue(active.arrow_active)

    def test_bent_long_arrow_needs_no_release_gesture(self):
        points=finger();points[6]=[0.3,0.3,0];points[7]=[0.6,0.3,0]
        active=logic.classify_hand(points,640,480,logic.Settings(),'Right')
        gate=logic.PulseGate(logic.Settings())
        self.assertEqual(gate.update(active,0),logic.RIGHT)
        self.assertEqual(gate.update(active,1),logic.RIGHT)

    def test_short_arrow_stop_then_bent_direction_and_up_reverse(self):
        settings=logic.Settings();points=finger();points[8]=points[5]
        neutral=logic.classify_hand(points,640,480,settings,'Right')
        points=finger();points[6]=[0.3,0.3,0];points[7]=[0.6,0.3,0]
        active=logic.classify_hand(points,640,480,settings,'Right')
        gate=logic.PulseGate(settings)
        gate.update(neutral,0);gate.update(neutral,0.41)
        gate.update(active,0.5)
        self.assertEqual(gate.update(active,0.76),logic.RIGHT)
        self.assertEqual(gate.update(active,1.07),logic.RIGHT)
        self.assertEqual(gate.update(active,2),logic.RIGHT)
        self.assertEqual(gate.update(observation(logic.UP,True),2.01),logic.UP)

    def test_new_arrow_settings_reject_unsafe_or_nonfinite_values(self):
        for changes in ({'minimum_arrow_pixels':0},{'arrow_neutral_palm_ratio':math.nan}):
            with self.assertRaises(ValueError):replace(logic.Settings(),**changes).validate()

    def test_tracking_fault_reports_exact_rejection_reason(self):
        obs=logic.Observation(reason='Hand cropped or near image edge')
        with self.assertRaisesRegex(transport.SafetyFault,'Hand cropped or near image edge'):
            control.motion_observation_ok(obs,'Right')


class DirectGestureTests(unittest.TestCase):
    def test_up_is_reverse_not_neutral(self):
        result=logic.classify_hand(finger(dx=0,dy=-0.24),640,480,logic.Settings(),'Right')
        self.assertFalse(logic.is_neutral(result))
        self.assertEqual(result.gesture,logic.UP)
        self.assertIn('reverse ALL',result.reason)
        self.assertEqual(logic.speeds_for(result.gesture,logic.Settings()),(-0.18,-0.18,-0.18))

    def test_each_direction_is_immediate(self):
        for gesture in (logic.LEFT,logic.RIGHT,logic.AWAY,logic.UP):
            with self.subTest(gesture=gesture):
                gate=logic.PulseGate(logic.Settings())
                self.assertEqual(gate.update(observation(gesture,True),0),gesture)

    def test_directions_switch_without_stop_or_release_hold(self):
        gate=logic.PulseGate(logic.Settings())
        for n,gesture in enumerate((logic.LEFT,logic.RIGHT,logic.UP,logic.AWAY,logic.UP,logic.LEFT)):
            self.assertEqual(gate.update(observation(gesture,True),n*0.01),gesture)

    def test_only_stop_is_stationary_for_supported_directions(self):
        settings=logic.Settings()
        for gesture in (logic.LEFT,logic.RIGHT,logic.AWAY,logic.UP,logic.STOP):
            self.assertEqual(logic.speeds_for(gesture,settings)==logic.ZERO,gesture==logic.STOP)

    def test_stop_is_immediate_after_every_direction(self):
        gate=logic.PulseGate(logic.Settings())
        for gesture in (logic.LEFT,logic.RIGHT,logic.AWAY,logic.UP):
            gate.update(observation(gesture,True),0)
            self.assertEqual(gate.update(observation(),0.001),logic.STOP)
            self.assertEqual(logic.speeds_for(gate.active,logic.Settings()),logic.ZERO)

    def test_every_direction_remains_active_without_expiry(self):
        for gesture in (logic.LEFT,logic.RIGHT,logic.AWAY,logic.UP):
            gate=logic.PulseGate(logic.Settings())
            self.assertEqual(gate.update(observation(gesture,True),0),gesture)
            self.assertEqual(gate.update(observation(gesture,True),36000),gesture)

    def test_invalid_hand_clears_current_direction(self):
        gate=logic.PulseGate(logic.Settings());gate.update(observation(logic.LEFT,True),0)
        self.assertEqual(gate.update(observation(valid=False),1),logic.STOP)
        self.assertEqual(gate.update(observation(logic.RIGHT,True),2),logic.RIGHT)

    def test_invalid_timestamp_rejected(self):
        for value in (math.nan,math.inf):
            with self.assertRaises(ValueError):logic.PulseGate(logic.Settings()).update(observation(),value)

    def test_unknown_gesture_rejected(self):
        with self.assertRaises(ValueError):logic.PulseGate(logic.Settings()).update(observation('unknown',True),0)

    def test_inactive_arrow_never_moves(self):
        gate=logic.PulseGate(logic.Settings())
        self.assertEqual(gate.update(observation(logic.UP,False),0),logic.STOP)

    def test_disabled_back_is_stop(self):
        settings=replace(logic.Settings(),away_mode='disabled')
        self.assertEqual(logic.PulseGate(settings).update(observation(logic.AWAY,True),0),logic.STOP)
        self.assertEqual(logic.speeds_for(logic.AWAY,settings),logic.ZERO)

    def test_up_reverse_does_not_depend_on_positive_winding(self):
        estimate=transport.WindingEstimate(logic.Settings())
        command=estimate.coordinate(logic.speeds_for(logic.UP,logic.Settings()))
        self.assertEqual(command,(-0.18,-0.18,-0.18))
        estimate.update(command,100)
        self.assertEqual(estimate.amounts,[0,0,0])
        self.assertEqual(estimate.coordinate(command),command)

    def test_up_reverse_reduces_known_winding_without_reset(self):
        estimate=transport.WindingEstimate(logic.Settings());estimate.amounts=[0.1,0.2,0.3]
        estimate.update(logic.speeds_for(logic.UP,logic.Settings()),0.5)
        for actual,expected in zip(estimate.amounts,[0.01,0.11,0.21]):self.assertAlmostEqual(actual,expected)

    def test_up_respects_all_direction_inversions(self):
        import itertools
        for signs in itertools.product((-1,1),repeat=3):
            settings=replace(logic.Settings(),servo_forward_signs=signs)
            command=transport.WindingEstimate(settings).coordinate(logic.speeds_for(logic.UP,settings))
            self.assertEqual(tuple(value*sign for value,sign in zip(command,signs)),(-0.18,-0.18,-0.18))

    def test_legacy_holds_and_motion_limits_are_ignored(self):
        old='{"gesture_hold_seconds":99,"neutral_hold_seconds":99,"maximum_session_seconds":0,"maximum_pulse_seconds":0,"maximum_signed_travel_seconds":0,"maximum_cumulative_motion_seconds":0,"serial_port":"COM8"}'
        with patch.object(Path,'read_text',return_value=old):settings=logic.load_settings(Path('old.json'))
        self.assertEqual(settings.serial_port,'COM8')
        self.assertEqual(logic.PulseGate(settings).update(observation(logic.UP,True),0),logic.UP)
        for name in ('gesture_hold_seconds','neutral_hold_seconds','maximum_signed_travel_seconds','maximum_pulse_seconds'):
            self.assertFalse(hasattr(settings,name))


class FakeLink:
    def __init__(self):
        self.commands=[];self.arms=0;self.stops=0;self.closes=0;self.pings=0
    def arm(self):self.arms+=1
    def ping(self):self.pings+=1
    def send_speeds(self,speeds):self.commands.append(tuple(speeds))
    def stop(self):self.stops+=1
    def close(self):self.closes+=1


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.now=0.0
        self.link=FakeLink()
        self.drive=transport.GuardedDrive(self.link,logic.Settings(),clock=lambda:self.now)

    def test_unarmed_command_rejected(self):
        with self.assertRaises(transport.SafetyFault):self.drive.send_speeds((0,0.18,0))

    def test_arm_sends_zero_first(self):
        self.drive.arm()
        self.assertEqual(self.link.commands,[logic.ZERO])

    def test_commands_remain_independent(self):
        self.drive.arm();self.now=0.05;self.drive.send_speeds((0,0.18,0))
        self.assertEqual(self.link.commands[-1],(0,0.18,0))

    def test_loop_stall_blocks_next_motion(self):
        self.drive.arm();self.now=0.19
        with self.assertRaisesRegex(transport.SafetyFault,'stalled'):
            self.drive.send_speeds((0,0.18,0))
        self.assertEqual(self.link.commands,[logic.ZERO])

    def test_stop_attempt_even_after_stall(self):
        self.drive.arm();self.now=1
        with self.assertRaises(transport.SafetyFault):self.drive.stop()
        self.assertEqual(self.link.stops,2)
        self.assertFalse(self.drive.armed)

    def test_stop_has_no_permanent_rearm_lock(self):
        self.drive.arm();self.drive.stop()
        self.drive.arm()
        self.assertEqual(self.link.arms,2)
        self.assertEqual(self.link.pings,2)
        self.assertEqual(self.link.commands[-1],logic.ZERO)

    def test_speed_limit(self):
        self.drive.arm()
        for speeds in ((0.3,0,0),(math.nan,0,0),(0,0),(0,0,math.inf)):
            with self.subTest(speeds=speeds),self.assertRaises(ValueError):self.drive.send_speeds(speeds)

    def test_large_motion_history_does_not_lock_motion(self):
        self.drive.arm();self.drive.history.signed[1]=100000
        self.drive.history.cumulative[1]=1000000
        self.drive.send_speeds((0,0.18,0))
        self.assertEqual(self.link.commands[-1],(0,0.18,0))

    def test_reverse_does_not_erase_cumulative_motion(self):
        history=transport.MotionHistory()
        history.update((0.18,0,0),1);history.update((-0.18,0,0),1)
        self.assertAlmostEqual(history.signed[0],0)
        self.assertAlmostEqual(history.cumulative[0],0.36)

    def test_stale_post_inference_frame(self):
        with self.assertRaises(transport.SafetyFault):control.assert_fresh(1,logic.Settings(),1.16)
        with self.assertRaises(transport.SafetyFault):control.assert_fresh(math.nan,logic.Settings(),1.16)
        control.assert_fresh(1,logic.Settings(),1.1)

    def test_tracking_loss_and_hand_change_stop(self):
        for obs in (observation(valid=False),observation(label='Left'),observation(label='')):
            with self.assertRaises(transport.SafetyFault):control.motion_observation_ok(obs,'Right')


class FakeConnection:
    def __init__(self,respond=True,short_write=False):
        self.is_open=True;self.buffer=b'';self.writes=[]
        self.respond=respond;self.short_write=short_write
    @property
    def in_waiting(self):return len(self.buffer)
    def read(self,size):
        value=self.buffer[:size];self.buffer=self.buffer[size:];return value
    def write(self,payload):
        self.writes.append(payload)
        if self.short_write:return 0
        if self.respond:
            command=payload.decode().strip()
            ack={'ARM':'OK,ARM','STOP':'OK,STOP','PING':'PONG'}.get(command)
            if command.startswith('V,'):ack='OK,V,'+command.split(',')[1]
            if ack:self.buffer+=(ack+'\n').encode()
        return len(payload)
    def close(self):self.is_open=False


class SerialTests(unittest.TestCase):
    def make_link(self,**kwargs):
        link=transport.ESP32Link.__new__(transport.ESP32Link)
        link.connection=FakeConnection(**kwargs)
        link.sequence=0;link.armed=False;link.receive_buffer=b''
        return link

    def test_protocol_and_sequence(self):
        link=self.make_link();link.arm();link.send_speeds((0,0.18,0));link.stop()
        self.assertEqual(link.connection.writes,[b'ARM\n',b'V,1,0.0000,0.1800,0.0000\n',b'STOP\n'])
        self.assertFalse(link.armed)

    def test_never_send_without_arm(self):
        link=self.make_link()
        with self.assertRaises(transport.SafetyFault):link.send_speeds(logic.ZERO)
        self.assertEqual(link.connection.writes,[])

    def test_missing_ack_stops_protocol(self):
        link=self.make_link(respond=False)
        with self.assertRaises(transport.SafetyFault):link.exchange('ARM','OK,ARM',0.001)
        self.assertFalse(link.armed)

    def test_buffered_stale_ack_not_accepted(self):
        link=self.make_link(respond=False);link.connection.buffer=b'OK,ARM\n'
        with self.assertRaises(transport.SafetyFault):link.exchange('ARM','OK,ARM',0.001)

    def test_firmware_error_stops(self):
        link=self.make_link();link.armed=True;link.connection.buffer=b'ERR,WATCHDOG\n'
        with self.assertRaises(transport.SafetyFault):link.drain()
        self.assertFalse(link.armed)

    def test_stop_is_attempted_after_firmware_error(self):
        link=self.make_link();link.armed=True;link.connection.buffer=b'ERR,WATCHDOG\n'
        link.stop()
        self.assertIn(b'STOP\n',link.connection.writes)

    def test_short_write_stops(self):
        link=self.make_link(short_write=True)
        with self.assertRaises(transport.SafetyFault):link.arm()

    def test_overflow_stops(self):
        link=self.make_link();link.connection.buffer=b'a'*8193
        with self.assertRaises(transport.SafetyFault):link.drain()

    def test_close_stops_and_releases(self):
        link=self.make_link();link.close()
        self.assertIn(b'STOP\n',link.connection.writes)
        self.assertFalse(link.connection.is_open)


class ReturnTests(unittest.TestCase):
    def setUp(self):
        self.settings=logic.Settings()
        self.winding=transport.WindingEstimate(self.settings)

    def test_zero_start_does_not_reverse_other_motors(self):
        self.assertEqual(self.winding.coordinate((0.18,0,0)),(0.18,0,0))

    def test_only_previously_wound_motor_unwinds(self):
        self.winding.update((0,0.18,0),0.5)
        self.assertEqual(self.winding.coordinate((0.18,0,0)),(0.18,-0.09,0))

    def test_both_wound_motors_unwind_at_half_speed(self):
        self.winding.update((0,0.18,0.18),0.5)
        self.assertEqual(self.winding.coordinate((0.18,0,0)),(0.18,-0.09,-0.09))

    def test_left_can_release_one_and_three(self):
        self.winding.update((0.18,0,0.18),0.5)
        self.assertEqual(self.winding.coordinate((0,0.18,0)),(-0.09,0.18,-0.09))

    def test_right_can_release_one_and_two(self):
        self.winding.update((0.18,0.18,0),0.5)
        self.assertEqual(self.winding.coordinate((0,0,0.18)),(-0.09,-0.09,0.18))

    def test_stop_does_not_trigger_unwinding(self):
        self.winding.update((0.18,0.18,0.18),0.5)
        self.assertEqual(self.winding.coordinate(logic.ZERO),logic.ZERO)

    def test_up_reverses_all_wound_motors(self):
        self.winding.amounts=[0.09,0.09,0.09]
        self.assertEqual(self.winding.coordinate(logic.speeds_for(logic.UP,self.settings)),(-0.18,-0.18,-0.18))

    def test_near_home_release_is_capped_for_watchdog_reserve(self):
        self.winding.amounts[1]=0.005
        command=self.winding.coordinate((0.18,0,0))
        self.assertLess(abs(command[1]),0.09)
        self.assertLessEqual(abs(command[1])*0.35,0.003)
        self.winding.update(command,0.35)
        self.assertGreaterEqual(self.winding.amounts[1],0.002-1e-10)

    def test_residual_stops_release(self):
        self.winding.amounts[1]=0.002
        self.assertEqual(self.winding.coordinate((0.18,0,0)),(0.18,0,0))

    def test_estimate_never_negative(self):
        self.winding.update((-0.18,-0.18,-0.18),0.3)
        self.assertEqual(self.winding.amounts,[0,0,0])

    def test_direction_inversion_applied_to_estimate_and_output(self):
        winding=transport.WindingEstimate(replace(self.settings,servo_forward_signs=(1,-1,1)))
        winding.update((0,-0.18,0),0.5)
        self.assertEqual(winding.coordinate((0.18,0,0)),(0.18,0.09,0))

    def test_actual_drive_sends_coordinated_commands(self):
        clock=[0.0];link=FakeLink()
        drive=transport.GuardedDrive(link,self.settings,clock=lambda:clock[0])
        drive.arm();drive.winding.amounts[1]=0.09
        clock[0]=0.05;drive.send_speeds((0.18,0,0))
        self.assertEqual(link.commands[-1],(0.18,-0.09,0))
        clock[0]=0.15;drive.send_speeds(logic.ZERO)
        self.assertAlmostEqual(drive.winding.amounts[1],0.081)
        self.assertGreater(drive.history.cumulative[1],0)

    def test_return_not_blocked_by_motion_history(self):
        clock=[0.0];link=FakeLink()
        drive=transport.GuardedDrive(link,self.settings,clock=lambda:clock[0])
        drive.arm();drive.winding.amounts[1]=0.09;drive.history.signed[1]=-1000
        drive.send_speeds((0.18,0,0))
        self.assertEqual(link.commands[-1],(0.18,-0.09,0))

    def test_forward_ack_delay_not_counted_as_known_winding(self):
        clock=[0.0];link=FakeLink()
        drive=transport.GuardedDrive(link,self.settings,clock=lambda:clock[0]);drive.arm()
        original=link.send_speeds
        def delayed(values):
            original(values);clock[0]+=0.05
        link.send_speeds=delayed
        drive.send_speeds((0.18,0,0))
        self.assertEqual(drive.winding.amounts,[0,0,0])

    def test_new_default_backward_gesture_runs_servo_one(self):
        obs=logic.classify_hand(finger(dx=0.01,dz=-0.20),640,480,self.settings,'Right')
        self.assertEqual(obs.gesture,logic.AWAY)
        self.assertEqual(logic.speeds_for(obs.gesture,self.settings),(0.18,0,0))


class ApplicationTests(unittest.TestCase):
    def test_default_is_preview_without_motion(self):
        args=control.build_parser().parse_args([])
        self.assertEqual(args.mode,'preview');self.assertFalse(args.enable_motion)

    def test_run_without_flag_is_preview(self):
        with patch.object(control,'run_camera') as run:
            control.main(['run'])
        self.assertFalse(run.call_args.kwargs['enable_motion'])

    def test_preview_does_not_request_serial(self):
        with patch.object(control,'run_camera') as run,patch.object(control,'serial_mode') as serial_mode:
            control.main([])
        serial_mode.assert_not_called()
        self.assertFalse(run.call_args.kwargs['enable_motion'])

    def test_motion_flag_only_with_run(self):
        with self.assertRaises(SystemExit):control.main(['preview','--enable-motion'])

    def test_motion_requires_confirmation_before_resources(self):
        with patch.dict(sys.modules,{'cv2':SimpleNamespace()}),patch('builtins.input',return_value='NO'),\
             patch.object(control,'ESP32Link') as link,patch.object(control,'Camera') as camera:
            control.run_camera(logic.Settings(),Path(__file__).parent,True)
        link.assert_not_called();camera.assert_not_called()


class TrackingPauseTests(unittest.TestCase):
    def setUp(self):
        self.clock=[0.0];self.link=FakeLink()
        self.drive=transport.GuardedDrive(self.link,logic.Settings(),clock=lambda:self.clock[0])
        self.drive.arm()

    def test_confirmed_pause_preserves_estimates_and_permits_explicit_rearm(self):
        self.drive.send_speeds((0,0.18,0));self.clock[0]=0.1;self.drive.pause()
        cumulative=list(self.drive.history.cumulative);winding=list(self.drive.winding.amounts)
        self.assertFalse(self.drive.armed)
        self.clock[0]=0.2;self.drive.arm()
        self.assertEqual(self.link.arms,2)
        self.assertEqual(self.link.commands[-1],logic.ZERO)
        self.assertEqual(self.drive.started_at,0)
        self.assertEqual(self.drive.history.cumulative,cumulative)
        self.assertEqual(self.drive.winding.amounts,winding)

    def test_pause_accounts_motion_during_stop_ack(self):
        self.drive.send_speeds((0.18,0,0));self.clock[0]=0.05
        original=self.link.stop
        def delayed_stop():original();self.clock[0]+=0.05
        self.link.stop=delayed_stop
        self.drive.pause()
        self.assertAlmostEqual(self.drive.history.cumulative[0],0.018)
        self.assertAlmostEqual(self.drive.winding.amounts[0],0.009)

    def test_failed_stop_never_authorizes_rearm(self):
        def fail():raise RuntimeError('STOP acknowledgement missing')
        self.link.stop=fail
        with self.assertRaises(RuntimeError):self.drive.pause()
        self.assertFalse(self.drive.armed)
        with self.assertRaises(RuntimeError):self.drive.arm()
        self.assertEqual(self.link.arms,1)

    def test_stalled_pause_stops_but_allows_explicit_retry(self):
        self.clock[0]=0.2
        with self.assertRaises(transport.SafetyFault):self.drive.pause()
        self.assertGreater(self.link.stops,0)
        self.assertFalse(self.drive.armed)
        self.drive.arm()
        self.assertEqual(self.link.arms,2)

    def test_long_pause_does_not_expire_by_wall_clock(self):
        self.clock[0]=0.1;self.drive.pause();self.clock[0]=31
        self.drive.arm()
        self.assertEqual(self.link.arms,2)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.gate=control.FrameReadiness(logic.Settings())

    def finish(self, now, age=0.02, inference=0.01, valid=True):
        return self.gate.finish(now-age,now,inference,valid)

    def test_eight_good_frames_are_required(self):
        for i in range(7):self.assertFalse(self.finish(1+i*0.05))
        self.assertTrue(self.finish(1.35))

    def test_slow_cold_frame_can_recover_without_relaxing_deadline(self):
        self.assertFalse(self.finish(3,age=1,inference=1))
        self.assertEqual(self.gate.consecutive,0)
        for i in range(7):self.assertFalse(self.finish(3.05+i*0.05))
        self.assertTrue(self.finish(3.40))
        self.assertEqual(self.gate.settings.maximum_frame_age_seconds,0.15)

    def test_persistent_slow_frames_never_ready(self):
        for i in range(20):self.assertFalse(self.finish(1+i*0.2,age=0.16,inference=0.16))

    def test_late_gui_completion_is_counted_in_frame_age(self):
        self.assertFalse(self.finish(1,age=0.16,inference=0.01))
        self.assertIn('too old',self.gate.reason)

    def test_loop_gap_resets_readiness_even_if_new_frame_fresh(self):
        for i in range(8):self.finish(1+i*0.05)
        self.assertTrue(self.gate.ready)
        self.assertFalse(self.finish(2,age=0.01))
        self.assertIn('loop too slow',self.gate.reason)

    def test_invalid_hand_resets_count(self):
        for i in range(8):self.finish(1+i*0.05)
        self.assertFalse(self.finish(1.40,valid=False))
        self.assertEqual(self.gate.consecutive,0)

    def test_stale_frame_invalidates_previously_ready_status(self):
        for i in range(8):self.finish(1+i*0.05)
        self.assertFalse(self.finish(1.55,age=0.16))

    def test_bad_clock_is_not_ready(self):
        for value in (math.nan,math.inf,-1):
            gate=control.FrameReadiness(logic.Settings())
            self.assertFalse(gate.finish(1,value,0.01,True))

    def test_summary_exposes_processing_timings(self):
        self.finish(1,age=0.123,inference=0.101)
        self.assertIn('123ms',self.gate.summary())
        self.assertIn('101ms',self.gate.summary())


class FullLoopTests(unittest.TestCase):
    def run_loop(self,powered=False,lose_tracking=False,inference_delay=0.0,active_gesture=None,neutral_up=False,
                 gui_delay=0.0,arm_at=(),quit_at=None,active_start=20,read_delay=0.05,hand_sequence=None,
                 stop_failure_at=None,unloaded_test=False,send_failure_at=None,ping_failure_at=None,handshake_delay=0.0,exit_key=ord('q'),close_at=None):
        clock=[10.0]
        counts={'processed':0,'camera_closed':False,'tracker_closed':False}
        link=FakeLink()
        events=[];counts['events']=events
        for name in ('arm','send_speeds','stop','ping'):
            original=getattr(link,name)
            def record(*args,_name=name,_original=original):
                events.append((counts['processed'],_name,args))
                for method,condition,message in (('stop',stop_failure_at,'STOP acknowledgement missing'),
                        ('send_speeds',send_failure_at,'USB write failed'),('ping',ping_failure_at,'PING failed')):
                    failed=condition(counts['processed']) if callable(condition) else counts['processed']==condition
                    if _name==method and failed:raise OSError(message)
                result=_original(*args)
                if _name=='arm':clock[0]+=handshake_delay
                return result
            setattr(link,name,record)
        frame=SimpleNamespace(shape=(480,640,3))
        class FakeCamera:
            def __init__(self,*args):pass
            def read(self):
                clock[0]+=read_delay(counts['processed']+1) if callable(read_delay) else read_delay
                return clock[0],frame
            def close(self):counts['camera_closed']=True
        class FakeTracker:
            def __init__(self,*args):pass
            def process(self,*args):
                counts['processed']+=1
                clock[0]+=inference_delay(counts['processed']) if callable(inference_delay) else inference_delay
                if hand_sequence is not None:
                    return hand_sequence(counts['processed']),None
                if lose_tracking and counts['processed']>=12:
                    return observation(valid=False),None
                if active_gesture is not None and counts['processed']>=active_start:
                    return observation(active_gesture,True),None
                if neutral_up:
                    return observation(logic.UP,True),None
                return observation(),None
            def close(self):counts['tracker_closed']=True
        def key(*args):
            clock[0]+=gui_delay(counts['processed']) if callable(gui_delay) else gui_delay
            if powered and counts['processed'] in arm_at:return ord('a')
            if counts['processed']>=(quit_at or (38 if active_gesture is not None else 14)):return exit_key
            return -1
        cv=SimpleNamespace(VideoCapture=None,namedWindow=lambda *a:None,
             destroyAllWindows=lambda:None,flip=lambda image,*a:image,
             imshow=lambda *a:None,waitKey=key,
             getWindowProperty=lambda *a:0 if close_at is not None and counts['processed']>=close_at else 1,WINDOW_NORMAL=0,WND_PROP_VISIBLE=1)
        with patch.dict(sys.modules,{'cv2':cv,'serial':SimpleNamespace()}),\
             patch.object(control,'Camera',FakeCamera),\
             patch.object(control,'HandTracker',FakeTracker),\
             patch.object(control,'draw_overlay'),\
             patch.object(control,'ESP32Link',return_value=link) as serial_factory,\
             patch.object(control,'GuardedDrive',side_effect=lambda l,s,**kw:transport.GuardedDrive(l,s,clock=lambda:clock[0],**kw)),\
             patch.object(control.time,'monotonic',side_effect=lambda:clock[0]),\
             patch('builtins.input',return_value='UNLOADED' if unloaded_test else 'READY'):
            try:
                control.run_camera(logic.Settings(),Path(__file__).parent,powered,unloaded_test=unloaded_test)
            except BaseException as error:
                return link,counts,serial_factory.call_count,error
        return link,counts,serial_factory.call_count,None

    def events_of(self,counts,kind):
        return [(n,args) for n,event,args in counts['events'] if event==kind]

    def test_preview_never_opens_serial_or_arms(self):
        link,counts,calls,error=self.run_loop(active_gesture=logic.UP,active_start=1)
        self.assertIsNone(error);self.assertEqual(calls,0);self.assertEqual(link.arms,0)
        self.assertTrue(counts['camera_closed'] and counts['tracker_closed'])

    def test_ready_arms_automatically_without_any_key(self):
        link,counts,calls,error=self.run_loop(powered=True)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertEqual(self.events_of(counts,'arm')[0][0],8)
        self.assertEqual(link.commands[0],logic.ZERO)

    def test_auto_arm_requires_eight_fresh_frames(self):
        link,counts,calls,error=self.run_loop(powered=True,quit_at=7)
        self.assertIsNone(error);self.assertEqual(link.arms,0);self.assertEqual(link.commands,[])

    def test_up_at_start_reverses_without_neutral_or_a(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        moving=[v for v in link.commands if v!=logic.ZERO]
        self.assertTrue(moving);self.assertTrue(all(v==(-0.18,-0.18,-0.18) for v in moving))

    def test_each_direction_at_start_moves_without_stop_pose(self):
        for gesture,index in ((logic.LEFT,1),(logic.RIGHT,2),(logic.AWAY,0)):
            with self.subTest(gesture=gesture):
                link,counts,calls,error=self.run_loop(powered=True,active_gesture=gesture,active_start=1)
                self.assertIsNone(error);self.assertEqual(link.arms,1)
                self.assertTrue(any(v[index]==0.18 for v in link.commands))

    def test_first_nonzero_uses_new_post_handshake_measurement(self):
        def sequence(n):return observation(logic.UP if n<=8 else logic.RIGHT,True)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=20)
        self.assertIsNone(error);self.assertEqual(link.commands[0],logic.ZERO)
        self.assertTrue(any(v[2]==0.18 for v in link.commands[1:]))
        self.assertFalse(any(v[0]<0 for v in link.commands))
        self.assertEqual(self.events_of(counts,'send_speeds')[0][0],8)

    def test_stop_keeps_session_armed_and_sends_zero(self):
        link,counts,calls,error=self.run_loop(powered=True,quit_at=80)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(link.commands);self.assertTrue(all(v==logic.ZERO for v in link.commands))
        self.assertFalse(any(8<n<80 for n,args in self.events_of(counts,'stop')))

    def test_stop_pose_immediately_stops_all_servos(self):
        def sequence(n):return observation(logic.UP,True) if n<20 else observation()
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=40)
        self.assertIsNone(error)
        later=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>=20]
        self.assertTrue(later);self.assertTrue(all(v==logic.ZERO for v in later))
        self.assertEqual(link.arms,1)

    def test_up_to_left_to_right_to_back_has_no_release_hold(self):
        directions={15:logic.UP,16:logic.LEFT,17:logic.RIGHT,18:logic.AWAY,19:logic.STOP}
        def sequence(n):
            gesture=directions.get(n,logic.LEFT)
            return observation(gesture,gesture!=logic.STOP)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=25)
        self.assertIsNone(error)
        commands={n:args[0] for n,args in self.events_of(counts,'send_speeds')}
        self.assertEqual(commands[15],(-0.18,-0.18,-0.18))
        self.assertEqual(commands[16][1],0.18)
        self.assertEqual(commands[17][2],0.18)
        self.assertEqual(commands[18][0],0.18)
        self.assertEqual(commands[19],logic.ZERO)

    def test_all_directions_run_beyond_old_time_and_travel_limits(self):
        for gesture in (logic.UP,logic.LEFT,logic.RIGHT,logic.AWAY):
            with self.subTest(gesture=gesture):
                link,counts,calls,error=self.run_loop(powered=True,active_gesture=gesture,active_start=1,quit_at=760)
                self.assertIsNone(error);self.assertEqual(counts['processed'],760);self.assertEqual(link.arms,1)
                self.assertFalse(any(8<n<760 for n,args in self.events_of(counts,'stop')))
                late=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>700]
                self.assertTrue(late);self.assertTrue(all(v!=logic.ZERO for v in late))

    def test_missing_hand_stops_and_keeps_window_open(self):
        def sequence(n):return observation(logic.LEFT,True) if n<20 else observation(valid=False)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=70)
        self.assertIsNone(error);self.assertEqual(counts['processed'],70);self.assertEqual(link.arms,1)
        self.assertTrue(any(n==20 for n,args in self.events_of(counts,'stop')))
        self.assertFalse(any(n>=20 for n,args in self.events_of(counts,'send_speeds')))

    def test_returned_hand_resumes_automatically_from_current_direction(self):
        def sequence(n):
            if n<20:return observation(logic.LEFT,True)
            if n<35:return observation(valid=False)
            return observation(logic.RIGHT,True,label='Left')
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=70)
        self.assertIsNone(error);self.assertEqual(link.arms,2)
        later=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>=35]
        self.assertEqual(later[0],logic.ZERO)
        self.assertTrue(any(v[2]==0.18 for v in later))
        self.assertFalse(any(v[1]>0 for v in later))
        self.assertFalse(any(20<=n<42 for n,args in self.events_of(counts,'send_speeds')))

    def test_returned_up_hand_auto_resumes_all_reverse(self):
        def sequence(n):
            if 20<=n<35:return observation(valid=False)
            return observation(logic.LEFT if n<20 else logic.UP,True)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=70)
        self.assertIsNone(error);self.assertEqual(link.arms,2)
        late=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>=43]
        self.assertTrue(late);self.assertTrue(all(v==(-0.18,-0.18,-0.18) for v in late))

    def test_stop_on_reacquisition_does_not_replay_old_command(self):
        def sequence(n):
            if n<20:return observation(logic.RIGHT,True)
            return observation(valid=False) if n<35 else observation()
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=70)
        self.assertIsNone(error);self.assertEqual(link.arms,2)
        late=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>=35]
        self.assertTrue(late);self.assertTrue(all(v==logic.ZERO for v in late))

    def test_long_hand_absence_does_not_expire_ready_authorization(self):
        def sequence(n):
            if 20<=n<700:return observation(valid=False)
            return observation(logic.RIGHT,True)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=760)
        self.assertIsNone(error);self.assertEqual(counts['processed'],760);self.assertEqual(link.arms,2)
        self.assertTrue(any(n>710 and args[0][2]==0.18 for n,args in self.events_of(counts,'send_speeds')))

    def test_changed_hand_pauses_then_auto_arms_new_measurements(self):
        def sequence(n):return observation(logic.LEFT if n<20 else logic.RIGHT,True,label='Right' if n<20 else 'Left')
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=60)
        self.assertIsNone(error);self.assertEqual(link.arms,2)
        self.assertTrue(any(n==20 for n,args in self.events_of(counts,'stop')))
        self.assertFalse(any(20<=n<27 for n,args in self.events_of(counts,'send_speeds')))
        self.assertTrue(any(n>28 and args[0][2]==0.18 for n,args in self.events_of(counts,'send_speeds')))

    def test_persistent_invalid_identity_never_arms(self):
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=lambda n:observation(logic.UP,True,label=''),quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,0);self.assertEqual(link.commands,[])

    def test_no_hand_at_start_never_arms(self):
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=lambda n:observation(valid=False),quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,0)

    def test_unreliable_tracking_repeatedly_resets_startup_readiness(self):
        def sequence(n):return observation(logic.LEFT,True,valid=n%7!=0)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=60)
        self.assertIsNone(error);self.assertEqual(link.arms,0)

    def test_cold_inference_recovers_and_auto_arms_without_keys(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1,
                                           inference_delay=lambda n:0.5 if n==1 else 0,quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(any(v==(-0.18,-0.18,-0.18) for v in link.commands))

    def test_persistent_stale_frames_keep_preview_and_never_arm(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1,inference_delay=0.16,quit_at=40)
        self.assertIsNone(error);self.assertEqual(counts['processed'],40);self.assertEqual(link.arms,0)

    def test_persistent_slow_loop_never_arms(self):
        link,counts,calls,error=self.run_loop(powered=True,read_delay=0.19,quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,0)

    def test_slow_gui_at_start_recovers_without_a_key(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.RIGHT,active_start=1,
                                           gui_delay=lambda n:0.2 if n==1 else 0,quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(any(v[2]==0.18 for v in link.commands))

    def assert_fault_recovers(self,**kwargs):
        def sequence(n):return observation(logic.LEFT if n<20 else logic.RIGHT,True)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=90,**kwargs)
        self.assertIsNone(error);self.assertEqual(counts['processed'],90);self.assertEqual(link.arms,2)
        self.assertTrue(any(n==20 for n,args in self.events_of(counts,'stop')))
        late=[args[0] for n,args in self.events_of(counts,'send_speeds') if n>45]
        self.assertTrue(late);self.assertTrue(all(v[2]==0.18 for v in late))
        self.assertFalse(any(v[1]>0 for v in late))

    def test_stale_inference_stops_and_auto_recovers(self):
        self.assert_fault_recovers(inference_delay=lambda n:0.16 if n==20 else 0)

    def test_stale_gui_stops_and_auto_recovers(self):
        self.assert_fault_recovers(gui_delay=lambda n:0.16 if n==20 else 0)

    def test_loop_stall_stops_and_auto_recovers(self):
        self.assert_fault_recovers(read_delay=lambda n:0.19 if n==20 else 0.05)

    def test_usb_write_failure_stops_and_auto_recovers(self):
        self.assert_fault_recovers(send_failure_at=20)

    def test_failed_stop_does_not_permanently_lock_recovery(self):
        self.assert_fault_recovers(send_failure_at=20,stop_failure_at=20)

    def test_failed_initial_ping_can_auto_retry(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1,ping_failure_at=8,quit_at=80)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(any(v==(-0.18,-0.18,-0.18) for v in link.commands))

    def test_persistent_ping_failure_keeps_window_open_without_arming(self):
        link,counts,calls,error=self.run_loop(powered=True,ping_failure_at=lambda n:True,quit_at=100)
        self.assertIsNone(error);self.assertEqual(counts['processed'],100);self.assertEqual(link.arms,0)
        self.assertEqual(link.commands,[])
        attempts=[n for n,args in self.events_of(counts,'ping')]
        self.assertGreater(len(attempts),1)
        self.assertTrue(all(b-a>=19 for a,b in zip(attempts,attempts[1:])))

    def test_failed_recovery_ping_never_replays_motion(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.LEFT,active_start=1,
                        send_failure_at=20,ping_failure_at=lambda n:n>=20,quit_at=100)
        self.assertIsNone(error);self.assertEqual(counts['processed'],100);self.assertEqual(link.arms,1)
        self.assertFalse(any(n>20 for n,args in self.events_of(counts,'send_speeds')))

    def test_long_arm_handshake_never_replays_stale_up_target(self):
        def sequence(n):return observation(logic.UP,True) if n<=8 else observation()
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,handshake_delay=0.6,quit_at=40)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(all(v==logic.ZERO for v in link.commands))

    def test_stop_then_same_direction_restarts_without_hold(self):
        def sequence(n):
            if n==20:return observation()
            return observation(logic.LEFT,True)
        link,counts,calls,error=self.run_loop(powered=True,hand_sequence=sequence,quit_at=30)
        self.assertIsNone(error)
        commands={n:args[0] for n,args in self.events_of(counts,'send_speeds')}
        self.assertEqual(commands[20],logic.ZERO);self.assertEqual(commands[21][1],0.18)
        self.assertEqual(link.arms,1)

    def test_a_key_is_not_required_and_does_not_change_arm_state(self):
        link,counts,calls,error=self.run_loop(powered=True,arm_at=(10,15,20),quit_at=35)
        self.assertIsNone(error);self.assertEqual(link.arms,1)

    def test_all_exit_keys_stop_and_release_resources(self):
        for key in (ord('q'),ord('x'),27):
            with self.subTest(key=key):
                link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1,exit_key=key,quit_at=25)
                self.assertIsNone(error);self.assertEqual(link.closes,1)
                self.assertTrue(any(n==25 for n,args in self.events_of(counts,'stop')))
                self.assertTrue(counts['camera_closed'] and counts['tracker_closed'])

    def test_window_close_stops_motors(self):
        link,counts,calls,error=self.run_loop(powered=True,active_gesture=logic.UP,active_start=1,close_at=25,quit_at=40)
        self.assertIsNone(error);self.assertEqual(counts['processed'],25)
        self.assertTrue(any(n==25 for n,args in self.events_of(counts,'stop')))

    def test_legacy_unloaded_flag_uses_same_automatic_control(self):
        link,counts,calls,error=self.run_loop(powered=True,unloaded_test=True,active_gesture=logic.UP,active_start=1,quit_at=60)
        self.assertIsNone(error);self.assertEqual(link.arms,1)
        self.assertTrue(any(v==(-0.18,-0.18,-0.18) for v in link.commands))


class ReadyPolicyTests(unittest.TestCase):
    def test_mode_flags_still_require_explicit_motion_enable(self):
        for args in (['run','--unloaded-test'],['probe','--enable-motion'],['probe','--unloaded-test']):
            with self.subTest(args=args),self.assertRaises(SystemExit):control.main(args)

    def test_cli_plain_run_has_no_unloaded_requirement(self):
        with patch.object(control,'run_camera') as run:control.main(['run','--enable-motion'])
        self.assertTrue(run.call_args.kwargs['enable_motion'])
        self.assertFalse(run.call_args.kwargs['unloaded_test'])

    def test_wrong_confirmation_opens_no_resources(self):
        for answer in ('NO','HOMED','ready',''):
            with patch.dict(sys.modules,{'cv2':SimpleNamespace()}),patch('builtins.input',return_value=answer),\
                 patch.object(control,'ESP32Link') as link,patch.object(control,'Camera') as camera,patch.object(control,'HandTracker') as tracker:
                control.run_camera(logic.Settings(),Path(__file__).parent,True)
            link.assert_not_called();camera.assert_not_called();tracker.assert_not_called()

    def test_overlay_explains_up_reverse_and_session_authorization(self):
        messages=[]
        cv=SimpleNamespace(putText=lambda frame,text,*a:messages.append(text),FONT_HERSHEY_SIMPLEX=0,LINE_AA=0)
        control.draw_overlay(cv,SimpleNamespace(shape=(480,640,3)),None,observation(logic.UP,True),
                             logic.PulseGate(logic.Settings()),logic.UP,logic.Settings(),True,True)
        self.assertTrue(any('Up=ALL REVERSE' in text for text in messages))
        self.assertTrue(any('STOP=all stationary' in text for text in messages))
        self.assertFalse(any('press A' in text or 'A: arm' in text or 'UP stops' in text for text in messages))

    def test_overlay_pause_explains_automatic_recovery(self):
        messages=[]
        cv=SimpleNamespace(putText=lambda frame,text,*a:messages.append(text),FONT_HERSHEY_SIMPLEX=0,LINE_AA=0)
        control.draw_overlay(cv,SimpleNamespace(shape=(480,640,3)),None,observation(),logic.PulseGate(logic.Settings()),
                             logic.STOP,logic.Settings(),True,False,motion_fault='Paused: USB')
        self.assertTrue(any('auto-resume' in text for text in messages))
        self.assertFalse(any('LOCKED' in text for text in messages))

    def test_unlimited_reverse_is_not_blocked_at_zero_estimate(self):
        now=[0.0];link=FakeLink()
        drive=transport.GuardedDrive(link,logic.Settings(),clock=lambda:now[0]);drive.arm()
        for _ in range(2400):
            now[0]+=0.05;drive.send_speeds(logic.speeds_for(logic.UP,logic.Settings()))
        self.assertTrue(drive.armed);self.assertEqual(link.commands[-1],(-0.18,-0.18,-0.18))
        self.assertEqual(drive.winding.amounts,[0,0,0])
        self.assertTrue(all(v>20 for v in drive.history.cumulative))

    def test_telemetry_does_not_limit_any_forward_motor(self):
        for selected in range(3):
            now=[0.0];link=FakeLink();drive=transport.GuardedDrive(link,logic.Settings(),clock=lambda:now[0]);drive.arm()
            speeds=tuple(0.18 if i==selected else 0 for i in range(3))
            for _ in range(2400):now[0]+=0.05;drive.send_speeds(speeds)
            self.assertTrue(drive.armed);self.assertGreater(drive.history.cumulative[selected],20)
            self.assertEqual(link.commands[-1],speeds)

    def test_zero_command_and_pause_do_not_reset_history(self):
        now=[0.0];link=FakeLink();drive=transport.GuardedDrive(link,logic.Settings(),clock=lambda:now[0]);drive.arm()
        drive.send_speeds((0.18,0,0));now[0]=0.1;drive.send_speeds(logic.ZERO);drive.pause()
        previous=list(drive.history.cumulative);winding=list(drive.winding.amounts)
        now[0]=100;drive.arm()
        self.assertEqual(drive.history.cumulative,previous);self.assertEqual(drive.winding.amounts,winding)

    def test_half_speed_returns_still_taper_without_limiting_selected_motor(self):
        winding=transport.WindingEstimate(logic.Settings());winding.amounts[0]=0.09
        for _ in range(250):
            speeds=winding.coordinate((0,0.18,0));winding.update(speeds,0.05)
        self.assertEqual(winding.coordinate((0,0.18,0)),(0,0.18,0))
        self.assertGreaterEqual(winding.amounts[0],0.002-1e-8)

    def test_stop_ping_arm_zero_transport_order(self):
        link=FakeLink();calls=[]
        for name in ('stop','ping','arm','send_speeds'):
            original=getattr(link,name)
            def record(*args,_name=name,_original=original):
                calls.append((_name,args));return _original(*args)
            setattr(link,name,record)
        drive=transport.GuardedDrive(link,logic.Settings());drive.arm()
        self.assertEqual([name for name,args in calls],['stop','ping','arm','send_speeds'])
        self.assertEqual(calls[-1][1],(logic.ZERO,))

    def test_ping_failure_never_allows_arm_or_motion(self):
        link=FakeLink();drive=transport.GuardedDrive(link,logic.Settings())
        with patch.object(link,'ping',side_effect=OSError('USB')):
            with self.assertRaises(OSError):drive.arm()
        self.assertFalse(drive.armed);self.assertEqual(link.arms,0);self.assertEqual(link.commands,[])
        drive.arm();self.assertTrue(drive.armed)

    def test_stop_failure_never_allows_arm_or_motion(self):
        link=FakeLink();drive=transport.GuardedDrive(link,logic.Settings())
        with patch.object(link,'stop',side_effect=OSError('USB')):
            with self.assertRaises(OSError):drive.arm()
        self.assertFalse(drive.armed);self.assertEqual(link.arms,0);self.assertEqual(link.pings,0)
        drive.arm();self.assertTrue(drive.armed)


if __name__=='__main__':
    unittest.main(verbosity=2)
