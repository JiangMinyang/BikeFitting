"""
Kalman filter smoother for pose landmarks.

Applies an independent constant-velocity Kalman filter to each landmark
across video frames. This eliminates frame-to-frame jitter and interpolates
smoothly through low-confidence or briefly-occluded detections — particularly
important for distal landmarks like the toe and heel.

Design
------
State space  : [x, y, vx, vy]  per landmark (normalised image coords)
Motion model : constant velocity with small process noise
Measurement  : [x, y] from the detector, weighted by detection confidence²

Confidence weighting
    High confidence  → low measurement noise  → filter trusts detector
    Low confidence   → high measurement noise → filter trusts prediction
    Very low (<0.10) → measurement skipped    → filter predicts forward

Visibility smoothing
    Raw confidence scores that oscillate around a draw/compute threshold
    (the root cause of toe flickering) are smoothed with an exponential
    moving average before being stored in the output PoseFrame.
"""

import numpy as np
from typing import Optional, Tuple


# ── Per-landmark filter ───────────────────────────────────────────────────────

class _LandmarkKF:
    """2-D constant-velocity Kalman filter for a single pose landmark."""

    # F: state transition (dt = 1 frame)
    _F = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    # H: measurement model  z = H x
    _H = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float64,
    )

    def __init__(self, pos_process_std: float, vel_process_std: float,
                 measure_std_at_full_conf: float):
        self._Q = np.diag([
            pos_process_std ** 2, pos_process_std ** 2,
            vel_process_std ** 2, vel_process_std ** 2,
        ])
        self._R_base = measure_std_at_full_conf ** 2  # scalar, scaled by 1/conf²
        self._x: Optional[np.ndarray] = None  # not initialised until first detection
        self._P: Optional[np.ndarray] = None

    @property
    def initialised(self) -> bool:
        return self._x is not None

    def _init(self, x: float, y: float) -> None:
        self._x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

    def step(self, z: Optional[np.ndarray], confidence: float) -> np.ndarray:
        """
        Run one predict + (optional) update step.

        Args:
            z          : [x_norm, y_norm] measurement, or None to predict only.
            confidence : detector confidence for this landmark [0, 1].

        Returns:
            Smoothed [x_norm, y_norm] as float64 array.
        """
        # ── Initialise on first reliable detection ────────────────────────────
        if not self.initialised:
            if z is not None and confidence >= 0.10:
                self._init(float(z[0]), float(z[1]))
            # Return raw measurement (or zeros) until filter is running
            return z.copy() if z is not None else np.zeros(2, dtype=np.float64)

        # ── Predict ───────────────────────────────────────────────────────────
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        if z is None or confidence < 0.10:
            # No reliable measurement — return prediction
            return self._x[:2].copy()

        # ── Update ────────────────────────────────────────────────────────────
        # Scale measurement noise by 1/confidence² so low-confidence readings
        # are treated with proportionally higher uncertainty.
        r = self._R_base / (confidence ** 2)
        R = np.eye(2) * r

        y_innov = z.astype(np.float64) - self._H @ self._x
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y_innov
        self._P = (np.eye(4) - K @ self._H) @ self._P

        return self._x[:2].copy()


# ── Multi-landmark smoother ───────────────────────────────────────────────────

class PoseLandmarkSmoother:
    """
    Wraps one _LandmarkKF per canonical landmark slot and applies an EMA to
    visibility scores, then exposes a single ``smooth()`` call for use inside
    the pose-engine frame loop.

    Parameters
    ----------
    n_landmarks : int
        Number of canonical landmark slots (33 for MediaPipe-compatible layout).
    process_std : float
        1-sigma process noise for position (normalised coords per frame).
        Default 0.003 ≈ ±0.3 % of frame width per frame — appropriate for
        cycling where limbs move slowly relative to frame width.
    vel_process_std : float
        Process noise for velocity state. Set ~3× pos to allow tracking fast
        direction changes without adding too much position jitter.
    measure_std : float
        1-sigma measurement noise at confidence = 1.0 (normalised coords).
        RTMPose-l is accurate to ~3-5 px on 256×192 input; 0.008 normalised
        units is a conservative estimate.
    vis_alpha : float
        EMA smoothing factor for visibility (0 = infinite smoothing, 1 = none).
        0.35 gives ~2-3 frame half-life, enough to smooth per-frame confidence
        oscillations without adding perceptible lag.
    """

    def __init__(
        self,
        n_landmarks: int = 33,
        process_std: float = 0.003,
        vel_process_std: float = 0.010,
        measure_std: float = 0.008,
        vis_alpha: float = 0.35,
    ):
        self._filters = [
            _LandmarkKF(process_std, vel_process_std, measure_std)
            for _ in range(n_landmarks)
        ]
        self._vis_ema = np.zeros(n_landmarks, dtype=np.float32)
        self._vis_alpha = vis_alpha

    def smooth(
        self,
        landmarks: Optional[np.ndarray],   # (N, 3) x/y/z in [0,1], or None
        visibility: Optional[np.ndarray],  # (N,) confidence, or None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Smooth one frame of pose detections.

        When the detector returns (None, None) — e.g. person not found — the
        method advances all filters forward (predict-only) and decays visibility,
        then returns (None, None) so downstream code skips the frame as usual.

        Returns
        -------
        (smoothed_landmarks, smoothed_visibility)  or  (None, None)
        """
        if landmarks is None or visibility is None:
            # Advance filters and decay visibility; keep frame as "no pose"
            for kf in self._filters:
                kf.step(None, 0.0)
            self._vis_ema *= (1.0 - self._vis_alpha)
            return None, None

        out_lm  = landmarks.copy().astype(np.float32)
        out_vis = visibility.copy().astype(np.float32)

        for i, kf in enumerate(self._filters):
            conf = float(visibility[i])
            z    = landmarks[i, :2]   # [x_norm, y_norm]

            # Skip filter for canonical slots that the backend never fills
            # (visibility always 0.0 for unmapped keypoints)
            if conf == 0.0 and not kf.initialised:
                self._vis_ema[i] = 0.0
                continue

            smoothed_xy = kf.step(z, conf)
            out_lm[i, 0] = float(smoothed_xy[0])
            out_lm[i, 1] = float(smoothed_xy[1])

            # EMA on visibility — smooths threshold-crossing oscillations
            self._vis_ema[i] = (
                self._vis_alpha * conf
                + (1.0 - self._vis_alpha) * self._vis_ema[i]
            )
            out_vis[i] = self._vis_ema[i]

        return out_lm, out_vis
