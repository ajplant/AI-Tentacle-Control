"""Five-point YOLO feedback dependency; contains no serial or motor code.

Point 1 is the rigid base reference, 2 the flexible root, 3/4 intermediate
landmarks, and 5 the tip. All coordinates refer to one unmirrored image.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SIGNATURE_KEYS = (
    "yolo_imgsz", "yolo_device", "yolo_box_confidence", "yolo_keypoint_confidence",
    "minimum_base_axis_pixels", "keypoint_border_pixels",
    "maximum_keypoint_jump_fraction", "maximum_axis_jump_degrees",
    "maximum_frame_age_seconds",
)
GEOMETRY = "point2_root_axis_point1_to_point2_tip_point5_v1"


def tracker_signature(settings: dict, base: Path) -> str:
    path = Path(settings["yolo_model"]).expanduser()
    path = path if path.is_absolute() else base/path
    if not path.is_file():
        raise RuntimeError(f"YOLO weights not found: {path}")
    with path.open("rb") as source:
        weights_hash = hashlib.file_digest(source, "sha256").hexdigest()
    payload = {key: settings[key] for key in SIGNATURE_KEYS}
    payload.update(weights_sha256=weights_hash, geometry=GEOMETRY, keypoint_order=[1,2,3,4,5])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class TentacleObservation:
    state: np.ndarray
    base_center: np.ndarray
    tip_center: np.ndarray
    base_to_tip_pixels: float
    straightness: float = 1.0


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


class YoloTentacleTracker:
    def __init__(self, cv2_module: Any, settings: dict, base: Path, model: Any = None):
        self.cv2 = cv2_module
        self.settings = settings
        self.signature = tracker_signature(settings, base)
        path = Path(settings["yolo_model"]).expanduser()
        path = path if path.is_absolute() else base/path
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(path), task="pose")
        if getattr(model, "task", None) != "pose":
            raise ValueError("Use the custom five-point tentacle POSE weights, not a detector.")
        if getattr(model, "names", None) != {0: "tentacle"}:
            raise ValueError("Expected exactly class 0: tentacle. Do not use COCO person weights.")
        self.model = model
        self.previous_points = None
        self.previous_axis = None
        self.last_rejection = "No pose processed yet"
        # Compile/allocate/warm up BEFORE any serial port can be armed.
        self._predict(np.zeros((480,640,3), dtype=np.uint8))

    def _predict(self, frame):
        return self.model.predict(
            source=frame, imgsz=int(self.settings["yolo_imgsz"]),
            device=self.settings["yolo_device"],
            conf=float(self.settings["yolo_box_confidence"]),
            classes=[0], max_det=2, verbose=False, save=False,
        )

    def reject(self, reason):
        self.last_rejection = reason
        self.previous_points = None
        self.previous_axis = None
        return None

    def validate_pose(self, points, confidence, frame_shape, reference_length_pixels):
        """Pure-array gate, also exercised without PyTorch or a real webcam."""
        points = np.asarray(points, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if points.shape != (5,2) or confidence.shape != (5,):
            return self.reject("Expected exactly five keypoints with confidence values")
        if (not np.isfinite(points).all() or not np.isfinite(confidence).all() or
                np.any(confidence < 0.) or np.any(confidence > 1.)):
            return self.reject("Nonfinite YOLO keypoints")
        if np.any(confidence < float(self.settings["yolo_keypoint_confidence"])):
            return self.reject("A required keypoint has low confidence")
        height, width = frame_shape[:2]
        border = float(self.settings["keypoint_border_pixels"])
        if (np.any(points[:,0] < border) or np.any(points[:,0] > width-1-border)
                or np.any(points[:,1] < border) or np.any(points[:,1] > height-1-border)):
            return self.reject("A required keypoint is cropped or near the image edge")
        axis = points[1]-points[0]
        axis_length = float(np.linalg.norm(axis))
        if axis_length < float(self.settings["minimum_base_axis_pixels"]):
            return self.reject("Points 1 and 2 are too close to define a reliable base direction")
        axis /= axis_length
        root, tip = points[1], points[4]
        displacement = tip-root
        visible_length = float(np.linalg.norm(displacement))
        if visible_length < 20.:
            return self.reject("Root and tip are too close in this camera view")
        scale = visible_length if reference_length_pixels is None else float(reference_length_pixels)
        if not math.isfinite(scale) or scale < 20.:
            return self.reject("Invalid reference length; recalibrate")
        lengths = np.linalg.norm(np.diff(points[1:],axis=0),axis=1)
        if np.any(lengths < 3.):
            return self.reject("Intermediate landmarks overlap")
        if self.previous_points is not None:
            jump = np.linalg.norm(points-self.previous_points,axis=1).max()/scale
            if jump > float(self.settings["maximum_keypoint_jump_fraction"]):
                return self.reject("Implausible keypoint jump")
            angle = math.degrees(math.acos(float(np.clip(np.dot(axis,self.previous_axis),-1.,1.))))
            if angle > float(self.settings["maximum_axis_jump_degrees"]):
                return self.reject("Base direction jumped; inspect points 1 and 2")
        self.previous_points = points.copy()
        self.previous_axis = axis.copy()
        lateral = np.array([-axis[1],axis[0]])
        state = np.array([np.dot(displacement,lateral),np.dot(displacement,axis)])/scale
        self.last_rejection = "None"
        return TentacleObservation(state,root.copy(),tip.copy(),visible_length,
                                   visible_length/float(lengths.sum()))

    def observe(self, frame, reference_length_pixels):
        results = self._predict(frame)
        if len(results) != 1:
            return self.reject("Expected one prediction result for one captured frame")
        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return self.reject("Tentacle not detected")
        boxes = result.boxes
        scores = numpy_value(boxes.conf).reshape(-1)
        classes = numpy_value(boxes.cls).reshape(-1)
        xy = numpy_value(result.keypoints.xy)
        conf = result.keypoints.conf
        if scores.size != 1 or classes.size != 1 or xy.shape != (1,5,2):
            return self.reject("Need exactly ONE tentacle with FIVE keypoints; ambiguous detections rejected")
        if (not np.isfinite(scores).all() or not np.isfinite(classes).all() or classes[0] != 0
                or scores[0] > 1.
                or scores[0] < float(self.settings["yolo_box_confidence"])):
            return self.reject("Wrong class or low-confidence tentacle detection")
        if conf is None:
            return self.reject("Pose model returned no keypoint confidence values")
        conf = numpy_value(conf)
        if conf.shape != (1,5):
            return self.reject("Unexpected keypoint confidence shape")
        observation = self.validate_pose(xy[0],conf[0],frame.shape,reference_length_pixels)
        if observation is None:
            return None
        points = np.rint(xy[0]).astype(np.int32)
        self.cv2.polylines(frame,[points],False,(0,220,0),2)
        for index, point in enumerate(points,1):
            location = tuple(int(v) for v in point)
            self.cv2.circle(frame,location,5,(0,220,255),-1)
            self.cv2.putText(frame,str(index),(location[0]+7,location[1]-7),
                             self.cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2,self.cv2.LINE_AA)
        return observation
