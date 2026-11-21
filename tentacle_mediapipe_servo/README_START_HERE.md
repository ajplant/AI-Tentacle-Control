# MediaPipe continuous finger-angle servo controller

## Updated control policy

Type READY once in powered mode to authorize the entire session. The controller
automatically arms when fresh hand tracking is ready. No A key, neutral pose,
gesture dwell, or release hold is required. This authorization remains active
until you exit, including through tracking pauses.

**UP NOW MOVES ALL THREE MOTORS IN REVERSE. UP IS NOT A STOP GESTURE.**

| Recognized arrow direction | Motor command |
| --- | --- |
| LEFT | Servo 2 forward continuously |
| RIGHT | Servo 3 forward continuously |
| BACK toward camera (default) | Servo 1 forward continuously |
| UP | All three servos reverse continuously at the configured speed |
| STOP | All three servos at zero speed |

STOP is the only stationary recognized pose. A short orange arrow, an angle
between assigned direction sectors, or a disabled/unassigned direction is
classified as STOP. Missing/invalid tracking and communication/timing faults
also stop motion; those are fault handling, not extra gesture positions.

There are no signed-travel, cumulative-travel, pulse-duration or session-time
cutoffs. Change directly between LEFT, RIGHT, BACK and UP without going through
STOP. Each direction is used immediately once tracking is ready; no 0.25-second
dwell or 0.4-second neutral hold remains.

At default speed 0.18 and forward signs [1,1,1], UP sends
S1=-0.18, S2=-0.18, S3=-0.18. Each configured servo_forward_signs entry is
respected, so reverse is relative to that servo's configured forward direction.
UP keeps reversing even when winding estimates are zero.

When LEFT/RIGHT/BACK selects one forward motor, the other two retain their
estimate-based half-speed return, tapering near estimated home. This does not
limit the selected motor's travel. UP overrides that behavior with all-motor
reverse. STOP always sends zero to all three and does not disarm the healthy
session.

## Upgrade the current installation

Stop the old program. Back up your working controller folder, then replace
these three files together from this download:

- finger_servo_control.py
- gesture_logic.py
- servo_link.py

Keep your calibrated config_gestures.json, especially servo_forward_signs,
speed, camera and serial_port. Old maximum_session_seconds,
maximum_pulse_seconds, maximum_signed_travel_seconds,
maximum_cumulative_motion_seconds, gesture_hold_seconds and
neutral_hold_seconds keys are ignored. The new example config omits them.

Keep hand_landmarker.task next to the Python files. You can also copy the
updated tests and this README. No new dependency or firmware upload is needed.
Do not overwrite calibrated firmware neutral/inversion values with the
unchanged firmware_reference_only copy.

## Run

In Windows PowerShell from your existing project folder:

~~~powershell
cd "C:\Users\josan\source\repos\tentacle_ai_controller\tentacle_ai_controller"
.\.venv-gesture\Scripts\python.exe .\tentacle_mediapipe_servo\finger_servo_control.py run --enable-motion --port COM3
~~~

1. Read the console warning and type READY.
2. Show one complete hand. Once eight consecutive fresh valid frames are
   available, the controller automatically sends STOP/PING/ARM and a first
   zero-speed command. It then uses a NEW frame for motion.
3. Point LEFT, RIGHT, BACK or UP to move. No keyboard arming step is needed.
   Motion can start automatically as soon as tracking is ready, including if
   a direction is already held at startup.
4. Use a STOP pose to stop rotation while keeping control authorized.
5. X, Q, Esc, closing the camera window or Ctrl+C stops and exits.

A camera view labeled ARMED indicates an active hardware session. Waiting or
PAUSED can appear during startup, missing tracking, or a device fault. READY
authorization remains active through these conditions; it is not a promise
to keep hardware armed with missing/stale input.

The legacy --unloaded-test flag is still accepted for disconnected testing and
uses UNLOADED instead of READY. It applies the same automatic control policy.
No such flag is required for normal operation.

## Hand tracking and automatic recovery

If the hand is lost, ambiguous, cropped, too small, duplicated, or handedness
changes, motion stops/disarms but the camera window keeps running. When fresh
reliable single-hand frames return, the controller automatically re-arms and
uses that hand's CURRENT direction. There is no A press and no replay of the
old target. A returned UP pose therefore starts all-motor reverse.

Stale frames, loop stalls, missing acknowledgements and runtime USB-write
errors also pause motor motion. Under the existing READY authorization,
automatic recovery attempts verify STOP, PING and ARM acknowledgements before
commanding motion. Communication-fault retries are spaced by at least one
second. A device that remains unhealthy is not authorized to receive motion.
A physically failed camera/USB device may require restoring/reopening the
device or restarting the program; automatic hardware reconnection is not
implemented.

The 150ms delivered-frame-age, 180ms command-loop-gap checks, speed bounds and
existing 250ms ESP32 watchdog remain. They reject missing/stale commands, not
healthy continuous rotation. If tracking never becomes ready, inspect the
displayed age/inference/loop timings and improve lighting or reduce CPU load.

## Arrow angles

The orange arrow goes from INDEX MCP (landmark 5) to INDEX TIP (landmark 8).
The finger may be bent; joint straightness is not tested. Angles account for
pixel width/height. Displayed right=0 degrees, UP=90, LEFT=180 and down=270.
The camera view is mirrored by default, so left/right match the displayed view.

Default sectors are LEFT/RIGHT +/-25 degrees and UP +/-20 degrees. Other
image-plane angles are STOP unless explicitly assigned. Short arrow means
tip close to the index knuckle: less than max(12 pixels, 0.35 times the projected
palm length), using approximate 3D arrow magnitude. It is not simply any bent
finger. The config retains the historical arrow_neutral_palm_ratio name,
but that setting now determines the short-arrow STOP region.

BACK uses inferred MediaPipe z, not a depth camera. away_mode can be
toward-camera (default), from-camera, down, or disabled. Verify this in preview,
and use the same --away-mode flag for powered control.

~~~powershell
.\.venv-gesture\Scripts\python.exe .\tentacle_mediapipe_servo\finger_servo_control.py preview
~~~

Preview (also the default mode) and run without --enable-motion never open a
serial port. Add --camera 1 if necessary. To map BACK to the displayed downward
arrow, append --away-mode down to preview and run, or update the config.

## Hardware and feedback

- Three continuous-rotation servos receive speed/direction, not shaft angles.
- Existing ESP32 velocity firmware: 115200 baud, COM3, GPIO18/19/23 by default.
- Existing external servo supply, common ground and wiring are unchanged.
- STOP -> OK,STOP; PING -> PONG; ARM -> OK,ARM;
  V,<sequence>,<s1>,<s2>,<s3> -> OK,V,<sequence>.
- Neutral calibration and firmware speed bounds are unchanged.

The software does not measure or verify mechanical protection, spool position,
tension, temperature or current. Zero speed means neutral PWM, not electrical
power removal. Keep a physical cutoff available. Check physical neutral and
direction signs before using motion.

Signed/cumulative command-time history is telemetry only and never locks
motion. Winding estimates control the half-speed returns, persist through
pauses, and reset at process restart. They are not encoders: slip, mechanical
limits or manual adjustment can make them differ from physical bending.

With servo power disconnected, protocol compatibility can be checked with:

~~~powershell
.\.venv-gesture\Scripts\python.exe .\tentacle_mediapipe_servo\finger_servo_control.py probe --port COM3
~~~

For the separate zero-speed diagnostic, disconnect all tendons/loads and run:

~~~powershell
.\.venv-gesture\Scripts\python.exe .\tentacle_mediapipe_servo\finger_servo_control.py neutral --port COM3
~~~

Type UNLOADED. This diagnostic sends zero for 10 seconds and exits; it does
not enable gesture control or change the unlimited powered-session policy.
If a servo creeps at zero, cut power and adjust its firmware neutralUs.

## Fresh installation only

Keep the provided files together without an extra nested controller folder.
Use Python 3.12 in a separate environment:

~~~powershell
py -3.12 -m venv .venv-gesture
.\.venv-gesture\Scripts\python.exe -m pip install -r .\tentacle_mediapipe_servo\requirements.txt
~~~

This is MediaPipe-only: no YOLO, Ultralytics, PyTorch, custom tentacle detector,
dataset or training step is used. The included pretrained hand model is
unchanged. Do not uninstall your old working environments or delete datasets.

## Verification

~~~powershell
.\.venv-gesture\Scripts\python.exe -m unittest discover -s .\tentacle_mediapipe_servo -p test_gestures.py -v
~~~

Hardware-free tests cover gesture math, direct direction changes, UP reverse
and per-servo inversion, STOP zero, READY-authorized automatic startup and
recovery, serial acknowledgements, fresh-target handling and unlimited holds.
Obsolete UP-neutral/manual-A/release-hold expectations were replaced.

Tests use synthetic landmarks and mocked camera/serial interfaces. No real
camera or servo operation was tested here. Verification does not establish
tracking accuracy, physical calibration, mechanical protection or safe tension.
verification.json records package checks; firmware reference is unchanged.

MediaPipe reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
