"""
Pose estimation engine — dual backend support.

Backend 1: MediaPipe Pose (preferred, most accurate)
Backend 2: OpenCV DNN with MoveNet/OpenPose (fallback, no extra install)

Extracts body landmarks per frame from video input.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class PoseFrame:
    """Holds pose data for a single video frame."""
    frame_index: int
    timestamp_sec: float
    landmarks: Optional[np.ndarray]  # shape (N, 3) — x, y, z (normalized 0..1)
    visibility: Optional[np.ndarray]  # shape (N,)
    raw_frame: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def has_pose(self) -> bool:
        return self.landmarks is not None


# Canonical landmark indices (shared across backends)
LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


class PoseBackend(ABC):
    """Abstract base for pose estimation backends."""

    @abstractmethod
    def detect(self, rgb_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose in a single RGB frame.
        Returns: (landmarks (N,3), visibility (N,)) or (None, None)
        """
        ...

    @abstractmethod
    def close(self):
        ...


# ── Backend 1: MediaPipe ────────────────────────────────────────────────────

class MediaPipeBackend(PoseBackend):
    """Full 33-landmark pose via Google MediaPipe."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self._pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
        )

    def detect(self, rgb_frame):
        results = self._pose.process(rgb_frame)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks = np.array([[l.x, l.y, l.z] for l in lm])
            visibility = np.array([l.visibility for l in lm])
            return landmarks, visibility
        return None, None

    def close(self):
        self._pose.close()


# ── Backend 2: OpenCV DNN (OpenPose COCO model) ─────────────────────────────

# OpenPose COCO body parts → our canonical indices mapping
_OPENPOSE_COCO_PARTS = {
    0: 0,    # Nose → nose
    2: 12,   # RShoulder → right_shoulder
    3: 14,   # RElbow → right_elbow
    4: 16,   # RWrist → right_wrist
    5: 11,   # LShoulder → left_shoulder
    6: 13,   # LElbow → left_elbow
    7: 15,   # LWrist → left_wrist
    8: 24,   # RHip → right_hip
    9: 26,   # RKnee → right_knee
    10: 28,  # RAnkle → right_ankle
    11: 23,  # LHip → left_hip
    12: 25,  # LKnee → left_knee
    13: 27,  # LAnkle → left_ankle
}

# MoveNet Lightning keypoints → our canonical indices
_MOVENET_PARTS = {
    0: 0,    # nose
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

NUM_CANONICAL_LANDMARKS = 33  # match MediaPipe's count for compatibility


class OpenCVDNNBackend(PoseBackend):
    """
    Lightweight fallback using OpenCV's DNN module.
    Supports either a TFLite MoveNet model or Caffe OpenPose model.
    Falls back to simple heatmap-based detection if no model file available.
    """

    def __init__(self, model_path: str = None, model_type: str = "auto"):
        """
        Args:
            model_path: Path to a model file. If None, will try to find one.
            model_type: "movenet", "openpose", or "auto"
        """
        self.net = None
        self.model_type = None
        self._input_size = (256, 256)

        if model_path and model_type == "openpose":
            self._load_openpose(model_path)
        elif model_path and model_type == "movenet":
            self._load_movenet(model_path)
        else:
            # Auto-detect: try common paths
            self._try_auto_load()

    def _load_openpose(self, proto_path: str, model_path: str = None):
        """Load OpenPose Caffe model."""
        try:
            if model_path:
                self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            else:
                self.net = cv2.dnn.readNet(proto_path)
            self.model_type = "openpose"
            self._input_size = (368, 368)
        except Exception:
            self.net = None

    def _load_movenet(self, path: str):
        """Load MoveNet TFLite-converted ONNX model."""
        try:
            self.net = cv2.dnn.readNetFromONNX(path)
            self.model_type = "movenet"
            self._input_size = (192, 192)
        except Exception:
            self.net = None

    def _try_auto_load(self):
        """Try to find and load a model from common locations."""
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            (os.path.join(base, "models", "movenet_lightning.onnx"), "movenet"),
            (os.path.join(base, "models", "pose_estimation.onnx"), "movenet"),
        ]
        for path, mtype in candidates:
            if os.path.exists(path):
                if mtype == "movenet":
                    self._load_movenet(path)
                if self.net is not None:
                    return
        # No model found — that's ok, will use fallback simple detection

    def detect(self, rgb_frame):
        if self.net is not None:
            return self._detect_dnn(rgb_frame)
        else:
            return self._detect_simple(rgb_frame)

    def _detect_dnn(self, rgb_frame):
        """Run DNN inference and map keypoints to canonical indices."""
        h, w = rgb_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            rgb_frame, 1.0 / 255.0, self._input_size, (0, 0, 0),
            swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()

        landmarks = np.zeros((NUM_CANONICAL_LANDMARKS, 3), dtype=float)
        visibility = np.zeros(NUM_CANONICAL_LANDMARKS, dtype=float)

        if self.model_type == "openpose":
            n_parts = output.shape[1]
            for op_idx, canon_idx in _OPENPOSE_COCO_PARTS.items():
                if op_idx >= n_parts:
                    continue
                heatmap = output[0, op_idx, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatmap)
                if conf > 0.1:
                    x = point[0] / heatmap.shape[1]
                    y = point[1] / heatmap.shape[0]
                    landmarks[canon_idx] = [x, y, 0.0]
                    visibility[canon_idx] = conf

        elif self.model_type == "movenet":
            # MoveNet output shape: (1, 1, 17, 3) — y, x, score
            kpts = output.reshape(-1, 3) if output.ndim > 2 else output[0]
            for mn_idx, canon_idx in _MOVENET_PARTS.items():
                if mn_idx >= len(kpts):
                    continue
                y, x, score = kpts[mn_idx]
                if score > 0.1:
                    landmarks[canon_idx] = [float(x), float(y), 0.0]
                    visibility[canon_idx] = float(score)

        if np.sum(visibility > 0.1) < 4:
            return None, None

        return landmarks, visibility

    def _detect_simple(self, rgb_frame):
        """
        Minimal skin-color based detection as ultimate fallback.
        Returns None — this is a placeholder that signals to the pipeline
        that no real pose estimation is available.
        """
        return None, None

    def close(self):
        self.net = None


# ── Main PoseEngine ─────────────────────────────────────────────────────────

def _create_backend(backend: str = "auto", **kwargs) -> PoseBackend:
    """
    Create the best available pose estimation backend.

    Priority order (auto):
      1. RTMPose-l  — best accuracy (76.3 COCO AP), ONNX Runtime
      2. MediaPipe  — good accuracy (~65-68 AP), zero extra deps
      3. OpenCV DNN — lightweight fallback
    """
    rtmpose_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("model_complexity",
                                  "min_detection_confidence",
                                  "min_tracking_confidence")}

    if backend == "rtmpose":
        from .rtmpose_backend import RTMPoseBackend
        return RTMPoseBackend(**rtmpose_kwargs)

    if backend in ("auto",):
        # Try RTMPose first (best accuracy)
        try:
            from .rtmpose_backend import RTMPoseBackend
            return RTMPoseBackend(**rtmpose_kwargs)
        except Exception as e:
            print(f"[PoseEngine] RTMPose unavailable ({e}), falling back to MediaPipe.")

    if backend in ("mediapipe", "auto"):
        mediapipe_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("model_complexity",
                                    "min_detection_confidence",
                                    "min_tracking_confidence")}
        try:
            return MediaPipeBackend(**mediapipe_kwargs)
        except ImportError:
            if backend == "mediapipe":
                raise ImportError(
                    "MediaPipe not installed. Install with: pip install mediapipe\n"
                    "Or use backend='opencv' for the OpenCV DNN fallback."
                )

    if backend in ("opencv", "auto"):
        return OpenCVDNNBackend()

    raise ValueError(f"Unknown backend: '{backend}'. "
                     f"Choose from: 'auto', 'rtmpose', 'mediapipe', 'opencv'")


class PoseEngine:
    """Processes video frames to extract pose landmarks."""

    def __init__(self, backend: str = "auto", model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.backend_type = backend
        self._backend_kwargs = {
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        self._backend: Optional[PoseBackend] = None

    def process_video(self, video_path: str, progress_callback=None) -> Tuple[List[PoseFrame], dict]:
        """
        Process a full video file and return pose frames + video metadata.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": duration,
            "video_path": video_path,
        }

        # Create backend
        backend = _create_backend(self.backend_type, **self._backend_kwargs)
        metadata["backend"] = type(backend).__name__

        pose_frames = []
        frame_idx = 0

        from .kalman_smoother import PoseLandmarkSmoother
        smoother = PoseLandmarkSmoother()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps if fps > 0 else frame_idx
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks, visibility = backend.detect(rgb_frame)

                # Apply per-landmark Kalman filter + visibility EMA smoothing.
                # This removes detection jitter and interpolates through low-
                # confidence frames, especially for distal landmarks (toe, heel).
                landmarks, visibility = smoother.smooth(landmarks, visibility)

                pose_frames.append(PoseFrame(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp,
                    landmarks=landmarks,
                    visibility=visibility,
                    raw_frame=None,  # Not stored — pipeline re-reads video for annotation
                ))

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                frame_idx += 1
        finally:
            backend.close()
            cap.release()

        return pose_frames, metadata

    def get_landmark_coords(self, frame: PoseFrame, name: str,
                            image_w: int, image_h: int) -> Optional[Tuple[int, int]]:
        """Return pixel (x, y) for a named landmark, or None if not visible."""
        if not frame.has_pose:
            return None
        idx = LANDMARK_INDICES.get(name)
        if idx is None:
            return None
        if idx >= len(frame.visibility):
            return None
        vis = frame.visibility[idx]
        if vis < 0.3:
            return None
        lm = frame.landmarks[idx]
        return (int(lm[0] * image_w), int(lm[1] * image_h))
