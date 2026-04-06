"""
Joint angle calculation module.
Computes knee, hip, ankle, elbow, and back angles from pose landmarks.

Research-backed angle definitions and pedal stroke phase detection.

References:
- Holmes et al.: Knee flexion 25-35 deg static at BDC
- Dynamic studies: 33-43 deg knee flexion at BDC (low intensity),
  30-40 deg at high intensity
- Max knee extension (MKE): 140-150 deg for road bikes
- Hip angle (shoulder-hip-knee): open angle 95-115 deg typical
- Ankle ROM during pedaling: ~50 deg total
  (dorsiflexion ~13 deg at 90 deg crank, plantarflexion ~37 deg at 285 deg)
- Trunk forward lean: 30-45 deg for road, varies by discipline
- Elbow angle (shoulder-elbow-wrist): 150-165 deg ideal for road cycling
- Upper arm angle (hip-shoulder-elbow): 85-105 deg typical road position
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .pose_engine import PoseFrame, LANDMARK_INDICES


# ── Research-based ideal ranges ─────────────────────────────────────────────
# These are evidence-based thresholds from cycling biomechanics literature.

IDEAL_RANGES = {
    # Knee angle at maximum extension (bottom dead center / BDC)
    # Road bikes: 140-150 deg (Holmes method static: 145-155 deg,
    # Dynamic correction: subtract ~8 deg → 137-147)
    "knee_extension_min": 135,   # below this → saddle too low
    "knee_extension_ideal_low": 140,
    "knee_extension_ideal_high": 150,
    "knee_extension_max": 155,   # above this → saddle too high, hyper-extension risk

    # Knee angle at maximum flexion (top dead center / TDC)
    # Typically 65-75 deg; below 60 → excessive compression
    "knee_flexion_min": 60,      # below → patellofemoral stress
    "knee_flexion_ideal_low": 65,
    "knee_flexion_ideal_high": 80,
    "knee_flexion_max": 90,      # above → saddle might be too high for this metric

    # Dynamic knee flexion at BDC (measured as 180 - extension angle)
    # Research recommends 33-43 deg at BDC (low intensity)
    "knee_flexion_bdc_min": 30,
    "knee_flexion_bdc_ideal_low": 33,
    "knee_flexion_bdc_ideal_high": 43,
    "knee_flexion_bdc_max": 48,

    # Hip angle (shoulder-hip-knee open angle)
    # TT/triathlon: 95-107 deg; road: slightly more open
    "hip_angle_min": 55,         # below → excessive hip closure, discomfort
    "hip_angle_ideal_low": 80,
    "hip_angle_ideal_high": 110,
    "hip_angle_max": 120,

    # Ankle angle (knee-ankle-foot when available; else shin-from-vertical).
    # With foot_index: 75-115 deg range (neutral ~90 deg).
    # Without foot_index (RTMPose): shin-from-vertical, typical 5-30 deg;
    # ROM through pedal stroke ~20-40 deg is normal.
    "ankle_angle_min": 75,       # extreme dorsiflexion (foot_index mode)
    "ankle_angle_ideal_low": 85,
    "ankle_angle_ideal_high": 110,
    "ankle_angle_max": 120,      # extreme plantarflexion — ankling too much
    # Shin-from-vertical ROM thresholds (used when foot_index unavailable)
    "ankle_shin_rom_low": 10,    # very rigid ankle
    "ankle_shin_rom_high": 40,   # excessive ankling

    # Elbow angle (shoulder → elbow → wrist)
    # Road cycling: 150-165 deg (soft bend, not locked out)
    # More flexed = reach too long / bars too low; straighter = too upright
    "elbow_angle_ideal_low": 150,
    "elbow_angle_ideal_high": 165,
    "elbow_angle_min": 130,      # very bent — reach likely too long
    "elbow_angle_max": 175,      # nearly straight — reach too short

    # Upper arm angle relative to torso (hip → shoulder → elbow, 3-point at shoulder).
    # Measures how far the upper arm extends forward of the torso direction.
    # 0 deg = arm along body toward hip; 90 deg = arm perpendicular to torso;
    # >90 deg = arm reaches forward of perpendicular.
    # Road/endurance: ~80-105 deg (arm roughly perpendicular, slightly forward).
    # Lower values = arm close to body (bars too high/close).
    # Higher values = arm reaching forward (bars too far/low).
    "shoulder_arm_ideal_low": 75,
    "shoulder_arm_ideal_high": 105,

    # Trunk lean from vertical (shoulder-hip line vs vertical)
    # Recreational: 40-55 deg; Road/endurance: 30-45 deg; TT/aero: 15-30 deg
    "trunk_recreational_low": 40,
    "trunk_recreational_high": 55,
    "trunk_road_low": 30,
    "trunk_road_high": 45,
    "trunk_aero_low": 15,
    "trunk_aero_high": 30,
}


@dataclass
class JointAngles:
    """Joint angles for a single frame (degrees)."""
    frame_index: int
    timestamp_sec: float

    # Knee angles (hip-knee-ankle angle; 180=full extension, lower=more flexion)
    left_knee: Optional[float] = None
    right_knee: Optional[float] = None

    # Hip angles (shoulder-hip-knee open angle)
    left_hip: Optional[float] = None
    right_hip: Optional[float] = None

    # Ankle angles.
    # With foot_index available: knee-ankle-foot three-point angle (~75-120 deg).
    # Without foot_index (RTMPose 17-kpt): shin-from-vertical angle (~5-30 deg).
    # The 'ankle_mode' flag on the calculator tells downstream which definition
    # is in use so recommendations can apply the right ideal ranges.
    left_ankle: Optional[float] = None
    right_ankle: Optional[float] = None

    # Elbow angles (shoulder → elbow → wrist; 180=fully extended)
    left_elbow: Optional[float] = None
    right_elbow: Optional[float] = None

    # Upper arm angles relative to torso (hip → shoulder → elbow, 3-point at shoulder).
    # 0 deg = arm along body toward hip; 90 deg = arm perpendicular to torso;
    # >90 deg = arm reaches forward. Torso-relative, so aero vs upright is accounted for.
    left_shoulder_arm: Optional[float] = None
    right_shoulder_arm: Optional[float] = None

    # Back/trunk angle from vertical (degrees)
    trunk_angle: Optional[float] = None

    # Forward lean using shoulder/hip midpoints (degrees from vertical)


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle in degrees between two 2D vectors."""
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _three_point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point B formed by rays B->A and B->C.
    Returns angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    return _angle_between_vectors(ba, bc)


def _get_lm(frame: PoseFrame, name: str,
            min_vis: float = 0.05) -> Optional[np.ndarray]:
    """Get 2D (x, y) normalized coordinates for a landmark (visibility-gated).

    The threshold is intentionally low (0.05) — the Kalman smoother has already
    stabilised positions, so we only want to exclude landmarks that are truly
    absent (confidence ≈ 0).  Per-frame None values are handled gracefully by
    summarize(), which computes statistics over whatever frames ARE valid.
    Call sites that need a stricter gate (e.g. foot_index for toe stability)
    pass an explicit min_vis override.
    """
    if not frame.has_pose:
        return None
    idx = LANDMARK_INDICES.get(name)
    if idx is None:
        return None
    if idx >= len(frame.visibility):
        return None
    if frame.visibility[idx] < min_vis:
        return None
    return frame.landmarks[idx, :2].copy()


def _hampel_filter(vals: np.ndarray, window: int = 9, k: float = 3.0) -> np.ndarray:
    """Hampel identifier: replace outliers in-place with the local window median.

    An observation is an outlier when:
        |x_i - median(window)| > k * 1.4826 * MAD(window)

    The 1.4826 factor makes MAD a consistent estimator of σ for Gaussian data,
    so k=3 corresponds roughly to 3-sigma.  None/NaN values are skipped and
    left unchanged so partially-missing angles are not disturbed.
    """
    result = vals.copy()
    n = len(vals)
    half = window // 2
    for i in range(n):
        if np.isnan(result[i]):
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = result[lo:hi]
        valid = win[~np.isnan(win)]
        if len(valid) < 3:
            continue
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        sigma = 1.4826 * mad
        if sigma > 0 and abs(result[i] - med) > k * sigma:
            result[i] = med
    return result


# Per-field Hampel filter parameters (window frames, k threshold).
# Ankle is tightest because bad toe-landmark detections cause large spikes.
_OUTLIER_PARAMS: Dict[str, tuple] = {
    "left_ankle":        (9, 2.0),
    "right_ankle":       (9, 2.0),
    "left_knee":         (9, 3.0),
    "right_knee":        (9, 3.0),
    "left_hip":          (9, 3.0),
    "right_hip":         (9, 3.0),
    "left_elbow":        (9, 3.0),
    "right_elbow":       (9, 3.0),
    "left_shoulder_arm": (9, 3.0),
    "right_shoulder_arm":(9, 3.0),
    "trunk_angle":       (9, 3.0),
}


def _filter_angle_outliers(angle_list: List["JointAngles"]) -> List["JointAngles"]:
    """Apply per-field Hampel filter to a list of JointAngles in-place."""
    if not angle_list:
        return angle_list
    for field, (window, k) in _OUTLIER_PARAMS.items():
        raw = np.array(
            [getattr(a, field) if getattr(a, field) is not None else np.nan
             for a in angle_list],
            dtype=float,
        )
        if np.all(np.isnan(raw)):
            continue
        filtered = _hampel_filter(raw, window=window, k=k)
        for i, a in enumerate(angle_list):
            if not np.isnan(filtered[i]):
                setattr(a, field, float(filtered[i]))
    return angle_list


class AngleCalculator:
    """Calculates joint angles from pose frames using biomechanics conventions."""

    def calculate_frame(self, frame: PoseFrame, near_side: Optional[str] = None) -> JointAngles:
        """Compute all joint angles for a single frame.

        Args:
            frame: Pose frame with landmark coordinates.
            near_side: If 'left' or 'right', only compute that side's angles and
                       skip the far side. Use for side-view videos where far-side
                       pose detection is unreliable.
        """
        angles = JointAngles(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
        )

        if not frame.has_pose:
            return angles

        compute_left = near_side != "right"
        compute_right = near_side != "left"

        # --- Knee angles (hip → knee → ankle) ---
        l_hip   = _get_lm(frame, "left_hip")   if compute_left  else None
        l_knee  = _get_lm(frame, "left_knee")  if compute_left  else None
        l_ankle = _get_lm(frame, "left_ankle") if compute_left  else None
        if all(v is not None for v in [l_hip, l_knee, l_ankle]):
            angles.left_knee = _three_point_angle(l_hip, l_knee, l_ankle)

        r_hip   = _get_lm(frame, "right_hip")   if compute_right else None
        r_knee  = _get_lm(frame, "right_knee")  if compute_right else None
        r_ankle = _get_lm(frame, "right_ankle") if compute_right else None
        if all(v is not None for v in [r_hip, r_knee, r_ankle]):
            angles.right_knee = _three_point_angle(r_hip, r_knee, r_ankle)

        # --- Hip angles (shoulder → hip → knee) ---
        l_shoulder = _get_lm(frame, "left_shoulder")  if compute_left  else None
        if all(v is not None for v in [l_shoulder, l_hip, l_knee]):
            angles.left_hip = _three_point_angle(l_shoulder, l_hip, l_knee)

        r_shoulder = _get_lm(frame, "right_shoulder") if compute_right else None
        if all(v is not None for v in [r_shoulder, r_hip, r_knee]):
            angles.right_hip = _three_point_angle(r_shoulder, r_hip, r_knee)

        # --- Ankle angles ---
        # Primary: knee → ankle → foot_index (MediaPipe / full 33-kpt models).
        # Fallback: shin angle from vertical (RTMPose 17-kpt, no foot keypoints).
        # Both capture ankle ROM through the pedal stroke; ideal ranges differ —
        # see IDEAL_RANGES keys prefixed with 'ankle_' vs 'ankle_shin_'.
        if compute_left and l_knee is not None and l_ankle is not None:
            l_foot = _get_lm(frame, "left_foot_index", min_vis=0.45)
            if l_foot is None:
                l_foot = _get_lm(frame, "left_heel")
            if l_foot is not None:
                angles.left_ankle = _three_point_angle(l_knee, l_ankle, l_foot)
            else:
                # Shin-from-vertical fallback
                shin = l_ankle - l_knee
                angles.left_ankle = _angle_between_vectors(shin, np.array([0.0, 1.0]))

        if compute_right and r_knee is not None and r_ankle is not None:
            r_foot = _get_lm(frame, "right_foot_index", min_vis=0.45)
            if r_foot is None:
                r_foot = _get_lm(frame, "right_heel")
            if r_foot is not None:
                angles.right_ankle = _three_point_angle(r_knee, r_ankle, r_foot)
            else:
                shin = r_ankle - r_knee
                angles.right_ankle = _angle_between_vectors(shin, np.array([0.0, 1.0]))

        # --- Elbow angles (shoulder → elbow → wrist) ---
        l_elbow = _get_lm(frame, "left_elbow") if compute_left  else None
        l_wrist = _get_lm(frame, "left_wrist") if compute_left  else None
        if all(v is not None for v in [l_shoulder, l_elbow, l_wrist]):
            angles.left_elbow = _three_point_angle(l_shoulder, l_elbow, l_wrist)

        r_elbow = _get_lm(frame, "right_elbow") if compute_right else None
        r_wrist = _get_lm(frame, "right_wrist") if compute_right else None
        if all(v is not None for v in [r_shoulder, r_elbow, r_wrist]):
            angles.right_elbow = _three_point_angle(r_shoulder, r_elbow, r_wrist)

        # --- Upper arm angles (hip → shoulder → elbow, 3-point at shoulder) ---
        # Torso-relative: angle between the upper-trunk extension (hip→shoulder
        # direction continued upward) and the upper arm (shoulder→elbow).
        # Computed as the SUPPLEMENT of the raw 3-point angle so that the result
        # matches what you see visually in the frame:
        #   < 90° = arm reaches forward of perpendicular (typical road/aero)
        #     0° = arm parallel to torso, pointing fully forward/upward
        #    90° = arm perpendicular to torso
        #  > 90° = arm behind perpendicular (very upright / bars too close)
        # Being torso-relative it is the same value whether the rider is aero
        # or upright, so aero + arms-forward and upright + arms-forward give the
        # same angle if the reach geometry is identical.
        if all(v is not None for v in [l_hip, l_shoulder, l_elbow]):
            angles.left_shoulder_arm = 180.0 - _three_point_angle(l_hip, l_shoulder, l_elbow)
        if all(v is not None for v in [r_hip, r_shoulder, r_elbow]):
            angles.right_shoulder_arm = 180.0 - _three_point_angle(r_hip, r_shoulder, r_elbow)

        # --- Trunk angle (shoulder-hip line vs vertical) ---
        # Use midpoints of both sides when available (more accurate); fall back
        # to whichever single side is visible so side-view videos always get a value.
        if (l_shoulder is not None and r_shoulder is not None
                and l_hip is not None and r_hip is not None):
            shoulder_pt = (l_shoulder + r_shoulder) / 2
            hip_pt      = (l_hip      + r_hip)      / 2
        elif l_shoulder is not None and l_hip is not None:
            shoulder_pt, hip_pt = l_shoulder, l_hip
        elif r_shoulder is not None and r_hip is not None:
            shoulder_pt, hip_pt = r_shoulder, r_hip
        else:
            shoulder_pt = hip_pt = None
        if shoulder_pt is not None:
            angles.trunk_angle = _angle_between_vectors(
                shoulder_pt - hip_pt, np.array([0.0, -1.0]))

        return angles

    def calculate_all(self, pose_frames: List[PoseFrame], near_side: Optional[str] = None) -> List[JointAngles]:
        """Calculate joint angles for all frames, then remove per-field outliers.

        Args:
            pose_frames: List of pose frames from the video.
            near_side: If 'left' or 'right', skip the far-side landmarks (side-view mode).
        """
        angle_list = [self.calculate_frame(f, near_side=near_side) for f in pose_frames]
        return _filter_angle_outliers(angle_list)

    def summarize(self, angle_list: List[JointAngles]) -> Dict:
        """
        Compute per-joint statistics: min, max, mean, range, std, percentiles.
        Returns a nested dict with comprehensive statistics.
        """
        joint_keys = [
            "left_knee", "right_knee",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle",
            "left_elbow", "right_elbow",
            "left_shoulder_arm", "right_shoulder_arm",
            "trunk_angle",
        ]
        summary = {}
        timestamps = [a.timestamp_sec for a in angle_list]

        for key in joint_keys:
            values = [getattr(a, key) for a in angle_list]
            valid = [(t, v) for t, v in zip(timestamps, values) if v is not None]
            if not valid:
                summary[key] = None
                continue
            ts_arr = np.array([x[0] for x in valid])
            v_arr = np.array([x[1] for x in valid])
            summary[key] = {
                "min": float(np.min(v_arr)),
                "max": float(np.max(v_arr)),
                "mean": float(np.mean(v_arr)),
                "std": float(np.std(v_arr)),
                "range": float(np.max(v_arr) - np.min(v_arr)),
                "p5": float(np.percentile(v_arr, 5)),
                "p25": float(np.percentile(v_arr, 25)),
                "p50": float(np.percentile(v_arr, 50)),
                "p75": float(np.percentile(v_arr, 75)),
                "p95": float(np.percentile(v_arr, 95)),
                "timestamps": ts_arr.tolist(),
                "data": v_arr.tolist(),
            }

        return summary

    def detect_pedal_phases(self, angle_list: List[JointAngles], side: str = "left") -> Dict:
        """
        Detect pedal stroke phases from knee angle oscillations.

        TDC and BDC are located at zero-crossings of the angular velocity
        (the signed first derivative of the smoothed knee angle):
          BDC: velocity crosses positive → negative  (angle stops increasing = max extension)
          TDC: velocity crosses negative → positive  (angle stops decreasing = max flexion)

        This is more precise than finding local maxima/minima on the angle signal,
        which are biased by smoothing and sample-rate discretisation.

        Returns dict with phase timing, angles at BDC/TDC, and per-stroke data.
        """
        knee_key = f"{side}_knee"
        angles = [getattr(a, knee_key) for a in angle_list]
        timestamps = [a.timestamp_sec for a in angle_list]

        valid = [(t, v) for t, v in zip(timestamps, angles) if v is not None]
        if len(valid) < 10:
            return {"phases_detected": False}

        ts   = np.array([x[0] for x in valid])
        vals = np.array([x[1] for x in valid])

        # Smooth with a Hann-windowed moving average to suppress noise before
        # differentiation without introducing the phase shift of a plain boxcar.
        win = min(11, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
        win = max(3, win)
        hann = np.hanning(win);  hann /= hann.sum()
        smoothed = np.convolve(vals, hann, mode='same')

        # Signed angular velocity (deg/s)
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / 30.0
        signed_vel = np.gradient(smoothed, dt)

        # Minimum gap between transitions: 0.3 s to reject sub-stroke noise
        fps_est = 1.0 / dt
        min_gap = max(3, int(fps_est * 0.3))

        def _zero_crossings(direction: str) -> List[int]:
            result = []
            for i in range(1, len(signed_vel)):
                if direction == 'down' and signed_vel[i - 1] > 0 >= signed_vel[i]:
                    result.append(i)
                elif direction == 'up' and signed_vel[i - 1] < 0 <= signed_vel[i]:
                    result.append(i)
            # Minimum-distance filter: keep first crossing in each gap
            if min_gap > 1 and result:
                filtered = [result[0]]
                for p in result[1:]:
                    if p - filtered[-1] >= min_gap:
                        filtered.append(p)
                result = filtered
            return result

        # BDC = pos→neg crossing (knee angle was increasing, now decreasing → max extension)
        # TDC = neg→pos crossing (knee angle was decreasing, now increasing → max flexion)
        bdc_indices = _zero_crossings('down')
        tdc_indices = _zero_crossings('up')

        if len(bdc_indices) < 2 or len(tdc_indices) < 1:
            return {"phases_detected": False}

        # Calculate per-stroke metrics
        strokes = []
        for s in range(len(bdc_indices) - 1):
            bdc_start = bdc_indices[s]
            bdc_end = bdc_indices[s + 1]
            # Find TDC between these two BDCs
            tdc_between = [t for t in tdc_indices if bdc_start < t < bdc_end]
            if not tdc_between:
                continue
            tdc_idx = tdc_between[0]

            stroke = {
                "bdc_angle": float(vals[bdc_start]),   # max extension
                "tdc_angle": float(vals[tdc_idx]),      # max flexion
                "rom": float(vals[bdc_start] - vals[tdc_idx]),  # range of motion
                "knee_flexion_at_bdc": float(180 - vals[bdc_start]),  # flexion angle
                "power_phase_duration": float(ts[bdc_start] - ts[tdc_idx]) if tdc_idx < bdc_start else float(ts[tdc_idx] - ts[bdc_start]),
                "stroke_duration": float(ts[bdc_end] - ts[bdc_start]),
                "bdc_time": float(ts[bdc_start]),
                "tdc_time": float(ts[tdc_idx]),
            }
            strokes.append(stroke)

        if not strokes:
            return {"phases_detected": False}

        bdc_angles = [s["bdc_angle"] for s in strokes]
        tdc_angles = [s["tdc_angle"] for s in strokes]
        roms = [s["rom"] for s in strokes]
        flexions_at_bdc = [s["knee_flexion_at_bdc"] for s in strokes]
        stroke_durations = [s["stroke_duration"] for s in strokes]

        return {
            "phases_detected": True,
            "side": side,
            "num_strokes": len(strokes),
            "strokes": strokes,
            # BDC (max extension) stats
            "bdc_angle_mean": float(np.mean(bdc_angles)),
            "bdc_angle_std": float(np.std(bdc_angles)),
            # TDC (max flexion) stats
            "tdc_angle_mean": float(np.mean(tdc_angles)),
            "tdc_angle_std": float(np.std(tdc_angles)),
            # Range of motion per stroke
            "rom_mean": float(np.mean(roms)),
            "rom_std": float(np.std(roms)),
            # Knee flexion at BDC (the key bike fit metric: 180 - extension angle)
            "knee_flexion_bdc_mean": float(np.mean(flexions_at_bdc)),
            "knee_flexion_bdc_std": float(np.std(flexions_at_bdc)),
            # Stroke duration consistency (coefficient of variation)
            "stroke_duration_mean": float(np.mean(stroke_durations)),
            "stroke_duration_cv": float(np.std(stroke_durations) / np.mean(stroke_durations) * 100)
            if np.mean(stroke_durations) > 0 else 0.0,
        }
