"""
RTMPose-l ONNX backend for pose estimation.

RTMPose-l achieves 76.3 COCO AP vs MediaPipe's ~65-68, with significantly
better robustness on side-view and occluded poses.

Model: rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192
  - Trained on 7 body datasets (COCO, AIC, MPII, CrowdPose, PoseTrack, ...)
  - SimCC head: two 1-D distributions for x and y coordinates per keypoint
  - 26 Halpe keypoints (COCO-17 + head/neck/hip-center + 6 foot landmarks)
  - Input: (1, 3, 256, 192) — RGB, ImageNet normalisation
  - Outputs: simcc_x (1, 26, 384), simcc_y (1, 26, 512)

Halpe-26 foot keypoints (indices 20-25):
  20: left_big_toe   → canonical 31 (left_foot_index)
  21: right_big_toe  → canonical 32 (right_foot_index)
  22: left_small_toe → (no canonical slot, skipped)
  23: right_small_toe→ (no canonical slot, skipped)
  24: left_heel      → canonical 29 (left_heel)
  25: right_heel     → canonical 30 (right_heel)

ONNX model download (auto-handled on first use):
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/
      rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip

CoreML export (Apple Silicon / iOS deployment):
  Use MMDeploy with the official config:
    configs/mmpose/pose-detection_coreml-fp16_static-256x192.py
  Reference: MMDeploy PR #1902

References:
  - RTMPose paper: https://arxiv.org/abs/2303.07399
  - MMPose model zoo: https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html
  - Halpe-26 keypoint schema: https://github.com/Fang-Haoshu/Halpe-FullBody
"""

import os
import zipfile
import urllib.request
import numpy as np
from typing import Optional, Tuple

# ── Model metadata ────────────────────────────────────────────────────────────

MODEL_FILENAME = "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.onnx"
MODEL_ZIP      = "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip"
MODEL_URL      = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    + MODEL_ZIP
)

# Default location: <repo_root>/models/
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(_HERE, "models")

# RTMPose input geometry
INPUT_H = 256
INPUT_W = 192
SIMCC_SPLIT_RATIO = 2.0  # Each input pixel → 2 SimCC bins

# ImageNet normalisation (RGB)
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

# ── Keypoint mapping ──────────────────────────────────────────────────────────
# RTMPose Halpe-26 outputs 26 keypoints.
# We map them into our canonical 33-landmark array (MediaPipe-compatible indices).
# Landmarks that RTMPose does not produce are left at zero visibility.
#
# Halpe-26 schema:
#   0  nose           17 head_top
#   1  left_eye       18 neck
#   2  right_eye      19 hip (center)
#   3  left_ear       20 left_big_toe
#   4  right_ear      21 right_big_toe
#   5  left_shoulder  22 left_small_toe
#   6  right_shoulder 23 right_small_toe
#   7  left_elbow     24 left_heel
#   8  right_elbow    25 right_heel
#   9  left_wrist
#   10 right_wrist
#   11 left_hip
#   12 right_hip
#   13 left_knee
#   14 right_knee
#   15 left_ankle
#   16 right_ankle

NUM_CANONICAL = 33  # keep parity with MediaPipe for downstream compatibility

# rtmpose_idx → canonical_idx
_RTMPOSE_TO_CANONICAL = {
    # ── COCO-17 body keypoints ────────────────────────────────────────────────
    0:  0,   # nose
    1:  1,   # left_eye  (not in LANDMARK_INDICES but harmless to fill)
    2:  2,   # right_eye
    3:  3,   # left_ear
    4:  4,   # right_ear
    5:  11,  # left_shoulder
    6:  12,  # right_shoulder
    7:  13,  # left_elbow
    8:  14,  # right_elbow
    9:  15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
    # ── Halpe-26 extra keypoints (foot landmarks) ─────────────────────────────
    # Indices 17-19 (head_top, neck, hip_center) have no canonical slot → skipped
    20: 31,  # left_big_toe   → left_foot_index  (canonical 31)
    21: 32,  # right_big_toe  → right_foot_index (canonical 32)
    # 22: left_small_toe  → no canonical slot, skipped
    # 23: right_small_toe → no canonical slot, skipped
    24: 29,  # left_heel      → left_heel         (canonical 29)
    25: 30,  # right_heel     → right_heel        (canonical 30)
}


# ── Helper: letterbox resize ──────────────────────────────────────────────────

def _letterbox(img_rgb: np.ndarray, target_h: int, target_w: int
               ) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image to target_h × target_w with aspect-ratio-preserving padding.

    Returns:
        resized_img  : (target_h, target_w, 3) uint8 RGB
        scale        : scale factor applied (used for inverse mapping)
        pad_top      : pixels of top padding
        pad_left     : pixels of left padding
    """
    h, w = img_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    import cv2
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, pad_top, pad_left


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray,
                  scale: float, pad_top: int, pad_left: int,
                  orig_h: int, orig_w: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode SimCC distributions → normalised (x, y) coordinates + visibility scores.

    simcc_x : (1, K, W*split) — x logit distributions  [K=26 for Halpe-26]
    simcc_y : (1, K, H*split) — y logit distributions  [K=26 for Halpe-26]

    Decoding follows the MMPose canonical SimCC decode:
      - location  : argmax over bins  / split_ratio  → pixel in INPUT space
      - confidence: min(max_x_logit, max_y_logit) passed through sigmoid → [0, 1]

    Softmax-based confidence (used previously) is WRONG for SimCC — it maps
    a strong logit of 0.9 to ~0.003 over 384 bins, hiding all detections.
    The raw max logit is the intended score; sigmoid brings it to [0, 1] on a
    scale comparable to MediaPipe's visibility values.

    Returns:
        landmarks   : (NUM_CANONICAL, 3) float  — (x_norm, y_norm, 0)
        visibility  : (NUM_CANONICAL,) float     — confidence [0, 1]
    """
    landmarks  = np.zeros((NUM_CANONICAL, 3), dtype=np.float32)
    visibility = np.zeros(NUM_CANONICAL,       dtype=np.float32)

    sx = simcc_x[0]  # (K, W*split)
    sy = simcc_y[0]  # (K, H*split)

    # Argmax over SimCC bins → coordinate in INPUT space
    x_idx = np.argmax(sx, axis=-1).astype(np.float32)  # (K,)
    y_idx = np.argmax(sy, axis=-1).astype(np.float32)  # (K,)
    x_in_input = x_idx / SIMCC_SPLIT_RATIO  # pixel x in [0, INPUT_W)
    y_in_input = y_idx / SIMCC_SPLIT_RATIO  # pixel y in [0, INPUT_H)

    # Confidence: min(max_x_logit, max_y_logit) → sigmoid → [0, 1]
    # This is the MMPose canonical formula. A logit of 0.0 maps to 0.5,
    # >0 means the model placed a clear peak, <0 means uncertain.
    max_x = np.max(sx, axis=-1)  # (K,)
    max_y = np.max(sy, axis=-1)  # (K,)
    raw_scores = np.minimum(max_x, max_y)     # (K,)
    scores = _sigmoid(raw_scores).astype(np.float32)  # (K,) in [0, 1]

    # Map from padded INPUT coords → original frame pixel coords → normalised
    for rtm_idx, can_idx in _RTMPOSE_TO_CANONICAL.items():
        x_px_orig = (x_in_input[rtm_idx] - pad_left) / scale
        y_px_orig = (y_in_input[rtm_idx] - pad_top)  / scale

        # Clamp to valid frame range
        x_norm = float(np.clip(x_px_orig / orig_w, 0.0, 1.0))
        y_norm = float(np.clip(y_px_orig / orig_h, 0.0, 1.0))

        landmarks[can_idx]  = [x_norm, y_norm, 0.0]
        visibility[can_idx] = float(scores[rtm_idx])

    return landmarks, visibility


# ── Model download ────────────────────────────────────────────────────────────

def _find_model(model_dir: Optional[str] = None) -> Optional[str]:
    """Return path to the RTMPose-l ONNX file if it exists, else None."""
    search_dirs = [d for d in [model_dir, DEFAULT_DIR] if d]
    for d in search_dirs:
        candidate = os.path.join(d, MODEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def download_model(model_dir: Optional[str] = None, verbose: bool = True) -> str:
    """
    Download the RTMPose-l ONNX model if not already present.

    Returns the path to the ONNX file.
    """
    dest_dir = model_dir or DEFAULT_DIR
    os.makedirs(dest_dir, exist_ok=True)

    onnx_path = os.path.join(dest_dir, MODEL_FILENAME)
    if os.path.isfile(onnx_path):
        return onnx_path

    zip_path = os.path.join(dest_dir, MODEL_ZIP)

    if verbose:
        print(f"[RTMPose] Downloading model from OpenMMLab (~150 MB)...")
        print(f"          {MODEL_URL}")

    _last_reported = [-1]

    def _progress(block_num, block_size, total_size):
        if total_size > 0 and verbose:
            pct = int(min(block_num * block_size / total_size * 100, 100))
            # Print every 10% to avoid flooding non-interactive terminals (Docker)
            if pct >= _last_reported[0] + 10:
                _last_reported[0] = pct
                print(f"          {pct}%", flush=True)

    urllib.request.urlretrieve(MODEL_URL, zip_path, reporthook=_progress)

    if verbose:
        print("[RTMPose] Extracting...")

    extracted_onnx = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".onnx"):
                zf.extract(member, dest_dir)
                raw_path = os.path.join(dest_dir, member)
                # Flatten nested directory structure → dest_dir root
                flat_path = os.path.join(dest_dir, os.path.basename(raw_path))
                if raw_path != flat_path:
                    os.makedirs(os.path.dirname(flat_path), exist_ok=True)
                    os.rename(raw_path, flat_path)
                extracted_onnx = flat_path
                break  # only need the first (and only) ONNX file

    # Clean up zip
    if os.path.isfile(zip_path):
        os.remove(zip_path)

    if extracted_onnx is None or not os.path.isfile(extracted_onnx):
        raise FileNotFoundError(
            "[RTMPose] No .onnx file found in the downloaded zip. "
            "Please download manually from:\n"
            f"  {MODEL_URL}"
        )

    # Rename to our canonical filename if needed
    if extracted_onnx != onnx_path:
        os.rename(extracted_onnx, onnx_path)

    if verbose:
        print(f"[RTMPose] Model ready: {onnx_path}")
    return onnx_path


# ── RTMPose ONNX Backend ──────────────────────────────────────────────────────

class RTMPoseBackend:
    """
    RTMPose-l ONNX Runtime backend (Halpe-26 keypoints).

    Provides the same detect(rgb_frame) → (landmarks, visibility) interface
    as MediaPipeBackend and OpenCVDNNBackend, so it can be used as a drop-in
    replacement in PoseEngine.

    Key improvements over MediaPipe Pose:
    - +8–11 COCO AP points (76.3 vs ~65-68)
    - 26 Halpe keypoints: full body + heel and toe landmarks for ankle analysis
    - Trained on 7 diverse body datasets — better generalisation
    - Top-down architecture with letterbox crop — more stable under equipment
      occlusion (bike frame, handlebars)
    - Deterministic, fully offline — no network calls after model download

    CoreML note:
    For Apple Silicon / iOS deployment, export via MMDeploy:
      python tools/deploy.py \\
        configs/mmpose/pose-detection_coreml-fp16_static-256x192.py \\
        [model_cfg] [checkpoint] [demo_video]
    Reference: MMDeploy PR #1902
    """

    def __init__(self, model_path: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 auto_download: bool = True):
        """
        Args:
            model_path    : Explicit path to the .onnx file. If None, searches
                            model_dir and the default models/ folder.
            model_dir     : Directory to search / download into.
            auto_download : Automatically download the model if not found.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for the RTMPose backend.\n"
                "Install with:  pip install onnxruntime"
            )

        # Resolve model path
        onnx_path = model_path or _find_model(model_dir)

        if onnx_path is None:
            if auto_download:
                onnx_path = download_model(model_dir=model_dir)
            else:
                raise FileNotFoundError(
                    f"RTMPose-l ONNX model not found. "
                    f"Expected: {os.path.join(model_dir or DEFAULT_DIR, MODEL_FILENAME)}\n"
                    f"Run:  python -c \"from core.rtmpose_backend import download_model; download_model()\""
                )

        # Create ONNX Runtime session
        providers = self._get_providers()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_options=opts,
                                             providers=providers)

        self._input_name = self._session.get_inputs()[0].name
        self._out_names  = [o.name for o in self._session.get_outputs()]

        # Validate expected output count (simcc_x, simcc_y)
        if len(self._out_names) < 2:
            raise RuntimeError(
                f"RTMPose ONNX model should have 2 outputs (simcc_x, simcc_y), "
                f"got {len(self._out_names)}: {self._out_names}"
            )

        active_provider = self._session.get_providers()[0]
        print(f"[RTMPose] Loaded {os.path.basename(onnx_path)} "
              f"({active_provider})")

    @staticmethod
    def _get_providers() -> list:
        """Return best available ONNX Runtime execution providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

        # Preference order: CoreML (macOS) → CUDA → DirectML → CPU
        preference = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "DirectMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preference if p in available] or ["CPUExecutionProvider"]

    def detect(self, rgb_frame: np.ndarray
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose in a single RGB frame.

        Args:
            rgb_frame : (H, W, 3) uint8 numpy array in RGB order.

        Returns:
            landmarks  : (NUM_CANONICAL, 3) float32 — normalised (x, y, z=0)
                         or None if no pose detected.
            visibility : (NUM_CANONICAL,) float32 — confidence scores [0, 1]
                         or None if no pose detected.
        """
        orig_h, orig_w = rgb_frame.shape[:2]

        # 1. Letterbox to INPUT_H × INPUT_W
        padded, scale, pad_top, pad_left = _letterbox(rgb_frame, INPUT_H, INPUT_W)

        # 2. Normalise and convert to NCHW float32
        img = padded.astype(np.float32)
        img = (img - _MEAN) / _STD
        inp = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 192)

        # 3. ONNX inference
        outputs = self._session.run(self._out_names, {self._input_name: inp})
        simcc_x, simcc_y = outputs[0], outputs[1]

        # 4. Sanity check — reject if all scores are very low (no person detected)
        x_conf = np.max(simcc_x[0], axis=-1)
        if np.mean(x_conf) < 1e-3:
            return None, None

        # 5. Decode SimCC → normalised coordinates
        landmarks, visibility = _decode_simcc(
            simcc_x, simcc_y,
            scale=scale, pad_top=pad_top, pad_left=pad_left,
            orig_h=orig_h, orig_w=orig_w,
        )

        # Require at least 4 confident keypoints before declaring a valid pose
        if np.sum(visibility > 0.3) < 4:
            return None, None

        return landmarks, visibility

    def close(self):
        """Release ONNX session resources."""
        self._session = None
