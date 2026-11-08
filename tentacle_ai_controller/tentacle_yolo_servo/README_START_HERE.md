# YOLO + MediaPipe + ESP32 tentacle controller

This is a NEW integration folder. Keep your old python folder, config, firmware,
and ArUco calibration as backups; do not overwrite them. The uploaded best.pt is
included unchanged. No retraining of the YOLO detector is needed for installation.

## What this version does

- One webcam frame feeds MediaPipe hand tracking and your custom YOLO26n Pose model.
- Point 1: rigid base reference. Point 2: flexible root. Points 3/4: intermediate
  landmarks. Point 5: tip. All five are required on every accepted frame.
- The base direction is point 1 to point 2. The controlled 2D displacement is
  point 2 to point 5, normalized by the straight reference length.
- Hand targeting remains the INDEX finger, referenced to its palm. This is a
  local, projected TIP-POSITION imitation, NOT complete shape matching, full-hand
  articulation, independent control of each segment, or 3D reconstruction.
- Three continuous-rotation motors receive SPEED/DIRECTION, not angles.
- Your STOP/PING/ARM/V serial protocol, one-second ARM/STOP ACK waits, motion
  budgets, 180ms loop limit, and firmware watchdog assumptions are preserved.
- No ArUco detection or opencv-contrib dependency is required.

## 1. Install in your Windows project

Extract this ZIP into:

    C:\Users\josan\source\repos\tentacle_ai_controller\tentacle_ai_controller

There should be ONE folder named tentacle_yolo_servo containing this README,
tentacle_control_yolo.py, yolo_pose_tracker.py, config_yolo.json, best.pt, and
hand_landmarker.task. If Windows creates an extra nesting level, move the inner
tentacle_yolo_servo folder to the project root. The commands below assume that layout.

In Visual Studio 2026, open View -> Terminal and select PowerShell.
Run each command below as ONE COMPLETE LINE. Do not paste argument fragments alone.

```powershell
cd "C:\Users\josan\source\repos\tentacle_ai_controller\tentacle_ai_controller"
```

Use the training environment that already ran your YOLO model successfully:

```powershell
.\.venv-yolo\Scripts\python.exe -m pip install mediapipe pyserial
```

Do not install into or delete your original .venv environment. Do not reinstall
multiple OpenCV variants to obtain ArUco support; this version does not use it.
requirements.txt is supplied for a completely fresh environment if needed.

## 2. FIRST test: vision only, servo power OFF

Close all other camera previews, including the standalone YOLO command, before
running this. The Python program opens the webcam only once.

```powershell
.\.venv-yolo\Scripts\python.exe .\tentacle_yolo_servo\tentacle_control_yolo.py preview
```

This mode NEVER opens the serial port or sends a motor command.
It does not need a servo calibration file. It uses config_yolo.json beside the
script, so no --config argument is needed when following these instructions.

Both panels show the SAME physical camera frame: hand target on the left,
five-point tentacle feedback on the right. Q or closing the window quits.
Both the index finger and entire tentacle must be visible and distinguishable.
Keep the camera fixed; place the hand beside the tentacle, not covering it.

The right panel shows why a pose is rejected and the processing/frame-age timing.
Check points 1 and 2 especially: their short baseline defines the direction of
the base, so localization noise can change the apparent bend. The base must be
large and clear enough that their separation exceeds 12 pixels.

If the frame-age status says TOO SLOW FOR MOTION, DO NOT enable motion. Close
other heavy applications and test again. A supported accelerator or a smaller
yolo_imgsz (e.g. 512) can be evaluated in PREVIEW ONLY; verify accuracy again and
calibrate afterward. Changing tracker settings invalidates the calibration.
Do not extend the safety deadlines/watchdog or reduce confidence thresholds just
to make unsafe tracking appear valid.

## 3. Check neutral, with ALL tendons detached

Your existing ESP32 firmware must implement:

    PING -> PONG
    ARM -> OK,ARM
    V,<sequence>,<speed1>,<speed2>,<speed3> -> OK,V,<sequence>
    STOP -> OK,STOP

It must stop/disarm on loss of valid commands within the existing 250ms
watchdog interval. No firmware changes are included or required for that
compatible version. Signal pins remain GPIO18/GPIO19/GPIO23. Keep ESP32 USB
attached to WINDOWS COM3, not WSL. Close serial monitors that hold COM3.
If COM3 changes, edit serial_port in this folder's config_yolo.json.

Power OFF before changing wiring or attaching/removing tendons. Check common
ground, signal polarity, and external servo power. The PCA9685 remains only
power distribution in this setup; do not tie its PWM output to an ESP32 output.
Use adequate power for all three motors and keep a physical cutoff within reach.

With tendons DETACHED, power the motors and run:

```powershell
.\.venv-yolo\Scripts\python.exe .\tentacle_yolo_servo\tentacle_control_yolo.py neutral
```

Type UNLOADED only after physically detaching the tendons. This sends zero speed
for ten seconds. All three motors must remain stationary. If any creeps, stop
and correct that motor's neutral pulse in the ESP32 firmware before proceeding.
STOP is a neutral PWM command, NOT electrical power removal or a guaranteed
mechanical brake. Use the physical power cutoff for an emergency.

## 4. Recalibrate servo response with YOLO

THIS STEP WILL MOVE SERVOS after you explicitly arm it. Do not run it unattended.
Power off, straighten the tentacle, reset spools, and reconnect lines with light
tension and room to travel in BOTH directions. Restore power only when clear.
Do not permit loose fishing line to wind onto a stalled motor or over-pull a joint.
Physical travel stops/slip clutches and a power cutoff are still necessary.

```powershell
.\.venv-yolo\Scripts\python.exe .\tentacle_yolo_servo\tentacle_control_yolo.py calibrate
```

1. Keep the tentacle physically straight. Wait for 15 stable valid frames.
2. Press S INSIDE the calibration window to record the straight reference.
3. The camera keeps updating. Wait for another 15 stable straight frames.
4. Press A INSIDE the window only when ready for SMALL AUTOMATIC MOTOR PULSES.
5. Keep hands clear and watch the motion. Q or closing the window aborts.

Calibration sends small positive and negative pulses to each motor and measures
the local response. It is not neural-network training. It saves:

    tentacle_yolo_servo\visual_model_yolo.json

The old ArUco visual_model_single_camera.json is intentionally rejected.
The new calibration stores the YOLO weights/settings signature and camera size.
Changing the weights or tracker settings requires recalibration.

Losing a required pose during sampling/motion STOPS and ABORTS calibration.
It does NOT automatically resume a motor pulse after occlusion. If it aborts,
power off, inspect the mechanism, re-home physically, and restart. Do not keep
increasing speeds, limits, or deadlines to force calibration to pass.

## 5. Inspect the controller WITHOUT serial output

After calibration succeeds:

```powershell
.\.venv-yolo\Scripts\python.exe .\tentacle_yolo_servo\tentacle_control_yolo.py run
```

This mode does not open the serial port. Check the hand target and tentacle
feedback together, including in new poses. It cannot arm or move the motors.
The hand target is restricted to a small neighborhood of the calibrated home;
large hand movements are intentionally not followed in this prototype.

## 6. First powered closed-loop trial

Only after the above passes: physically re-home with power off, lightly tension
lines, and keep a power cutoff ready. Do not change camera placement or tendon routing.

```powershell
.\.venv-yolo\Scripts\python.exe .\tentacle_yolo_servo\tentacle_control_yolo.py run --enable-motion
```

Type HOMED in the terminal only after that physical reset. Motor control remains
STOPPED until BOTH hand/pose are stable and you press A in the camera window.

- A: arm once home and stable; it does not reset budgets while already moving.
- X: stop and latch a hard fault. Inspect and re-home before restarting.
- Q/Esc or closing the window: stop and exit.
- Lost hand/pose, stale frame, slow loop, excessive jump, no visible progress,
  local-workspace exit, serial errors, and budget/session limits prevent motion
  or stop the drive. Hard faults require a restart and manual re-home.

Start with small finger changes near home, not large bends. If motion is wrong
or increases the error, cut servo power immediately. Do not compensate by
guessing direction signs in Python; inspect wiring/routing and recalibrate.

## Verification and limitations

37 simulated logic tests passed on the supplied code. Run them locally with:

```powershell
.\.venv-yolo\Scripts\python.exe -m unittest discover -s .\tentacle_yolo_servo -p "test_*.py" -v
```

Tests cover five-point geometry, confidence/cropping/jump rejection, ambiguity,
shared-frame isolation, calibration signatures, motion budgets, protocol format,
stale-frame stop paths, and the nonblocking calibration UI.
They do not use an actual webcam, PyTorch inference, USB serial device, or motors.

The uploaded checkpoint was statically inspected (without unpickling): it is a
five-keypoint, class-0 tentacle pose model trained on pose_dataset/data.yaml.
Actual model loading/inference and combined MediaPipe/YOLO latency could not be
tested in the build environment. Actual Windows/hardware behavior must be
verified by the steps above. This is a prototype, not a safety-certified system.

Camera-only feedback still cannot measure absolute spool rotation, line tension,
hidden slack, or out-of-plane shape. One camera cannot resolve all 3D motion.
The 2x3 local response model may have only one usable visible direction; if it
reports rank 1, orthogonal target errors are deliberately ignored. It is not a
global model of the whole tentacle. Use more labeled footage and independent
evaluation before expanding its operating range.

Your original motion settings were retained: max speed .28, calibration speed
.18 with .18-second pulses, 30-second session, .30 signed and 1.20 cumulative
normalized command-seconds per motor, and .18-second loop limit. These are
command-time estimates, NOT degrees, millimeters, measured tension, or guarantees
against damage. Re-home every powered session; do not increase limits blindly.
