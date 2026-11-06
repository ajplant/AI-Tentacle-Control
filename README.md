# AI Tentacle Control

An experimental computer-vision control system designed to make a 3D-printed, tendon-driven tentacle follow human index-finger movements.

The system combines hand tracking, tentacle pose estimation, and an ESP32-based servo controller. Camera feedback allows the controller to compare the tentacle’s observed position with the target derived from the user’s hand and apply small corrective movements.

## Hardware Components

- 1 × ESP32 V2 development board
- 3 × MG90S continuous-rotation micro servos
- 1 × 3D-printed tentacle
- Fishing line for tendon actuation
- Webcam positioned to capture both the hand and tentacle
- External servo power supply
- Computer running the Python vision and control software

## Computer Vision Models

### YOLO26 Nano Pose

A custom-trained YOLO26 Nano Pose model tracks five keypoints along the tentacle: the base reference, flexible root, two intermediate landmarks, and tip.

These keypoints provide visual feedback for estimating the tentacle’s position and bending direction. The controller uses this feedback—not the detection model alone—to calculate motor corrections and enforce configured movement limits.

### MediaPipe Hand Landmarker

MediaPipe Hand Landmarker tracks the hand and finger landmarks. The controller uses the wrist and index-finger landmarks to calculate a target position relative to the palm.

Both models process images from the same webcam.

## Servo Control

The three servos pull separate fishing-line tendons to bend the tentacle. Because this project uses continuous-rotation servos, commands specify **speed and direction**, rather than an absolute shaft angle.

A calibration procedure measures how each servo affects the tentacle’s visible position. The controller then uses this local response model to reduce the difference between the hand-derived target and the observed tentacle position.

## Limitations and Safety

This project is a prototype. Single-camera feedback estimates image-plane motion; it does not directly measure absolute spool rotation, tendon tension, or full three-dimensional shape.

Software limits and camera feedback are not substitutes for mechanical protection. Begin with vision-only testing, verify servo neutral, and calibrate before enabling closed-loop motion. Keep a physical power cutoff accessible during powered testing.
