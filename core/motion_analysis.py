"""
Motion smoothness and pedaling quality analysis module.

Implements research-backed cycling biomechanics metrics:

1. Pedal Smoothness (PS): Ratio of average to peak angular velocity per stroke.
   Analogous to power-meter PS = Pavg/Pmax. Range 0-100%, higher = smoother.
   (Garmin normative: 10-40%; elite cyclists: 20-30%)

2. Torque Effectiveness proxy (TE): Estimated from angular velocity consistency.
   Measures how much of each stroke contributes positive forward motion.
   (Garmin normative: 60-100%)

3. Dead Spot Score: Quantifies hesitation at TDC and BDC transitions
   where rotational torque approaches zero.

4. Stroke-to-stroke consistency: Coefficient of variation across pedal strokes.
   Lower CV = more consistent pedaling pattern.

5. SPARC-based jerk smoothness: Spectral arc length of angular velocity,
   a validated smoothness metric from motor control research.

References:
- Cycling Analytics: TE = positive power / total power per stroke
- Garmin Cycling Dynamics: PS = avg_power / max_power per revolution
- Bini et al. (2013): Pedal force effectiveness in cycling
- Hogan & Sternad (2009): SPARC smoothness metric
"""

import numpy as np
from typing import List, Dict, Optional
from .angle_calculator import JointAngles, IDEAL_RANGES


# ── Pure numpy signal processing ────────────────────────────────────────────

def _savgol_filter(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay filter (pure numpy, no scipy)."""
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    if window_length < 3:
        return data.copy()
    if polyorder >= window_length:
        polyorder = window_length - 1

    half_w = window_length // 2
    x = np.arange(-half_w, half_w + 1, dtype=float)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    coeffs = np.linalg.lstsq(A, np.eye(window_length), rcond=None)[0]
    weights = coeffs[0]

    padded = np.concatenate([
        data[half_w - 1::-1],
        data,
        data[-1:-half_w - 1:-1],
    ])
    result = np.convolve(padded, weights[::-1], mode='valid')
    if len(result) > len(data):
        excess = len(result) - len(data)
        result = result[excess // 2: excess // 2 + len(data)]
    elif len(result) < len(data):
        result = np.resize(result, len(data))
    return result


def _find_peaks(signal: np.ndarray, distance: int = 1) -> np.ndarray:
    """Simple peak finder (pure numpy). Returns indices of local maxima."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    if not peaks:
        return np.array([], dtype=int)
    if distance > 1:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
        peaks = filtered
    return np.array(peaks, dtype=int)


def _find_valleys(signal: np.ndarray, distance: int = 1) -> np.ndarray:
    """Find local minima."""
    return _find_peaks(-signal, distance)


class MotionAnalyzer:
    """Analyzes pedaling quality using research-backed biomechanics metrics."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def smooth_signal(self, data: List[float], window: int = 11, poly: int = 3) -> np.ndarray:
        arr = np.array(data, dtype=float)
        if len(arr) < window:
            window = max(3, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        return _savgol_filter(arr, window_length=window, polyorder=poly)

    def compute_angular_velocity(self, angles: np.ndarray) -> np.ndarray:
        """Degrees per second."""
        return np.gradient(angles, 1.0 / self.fps)

    def compute_jerk(self, angles: np.ndarray) -> np.ndarray:
        """Third derivative of angle (deg/s^3). Lower = smoother."""
        vel = self.compute_angular_velocity(angles)
        acc = np.gradient(vel, 1.0 / self.fps)
        return np.gradient(acc, 1.0 / self.fps)

    # ── Pedal Smoothness (PS) ───────────────────────────────────────────────

    def pedal_smoothness(self, knee_angles: np.ndarray) -> Dict:
        """
        Compute pedal smoothness as the ratio of average to peak angular
        velocity per pedal stroke, analogous to power-meter PS = Pavg/Pmax.

        A perfectly smooth circular motion would yield PS = 100%.
        Typical values: recreational 10-25%, trained 20-35%.

        Returns dict with per-stroke PS values and overall stats.
        """
        if len(knee_angles) < 10:
            return {"pedal_smoothness_pct": None}

        smoothed = self.smooth_signal(knee_angles.tolist())
        angular_vel = np.abs(self.compute_angular_velocity(smoothed))

        # Split into strokes using peaks in angle (BDC)
        peaks = _find_peaks(smoothed, distance=max(3, int(self.fps * 0.3)))
        if len(peaks) < 2:
            # Fall back to overall
            avg_v = np.mean(angular_vel)
            max_v = np.max(angular_vel)
            ps = (avg_v / max_v * 100) if max_v > 0 else 0
            return {"pedal_smoothness_pct": round(float(ps), 1), "per_stroke": []}

        stroke_ps = []
        for i in range(len(peaks) - 1):
            seg = angular_vel[peaks[i]:peaks[i + 1]]
            if len(seg) < 3:
                continue
            avg_v = np.mean(seg)
            max_v = np.max(seg)
            ps = (avg_v / max_v * 100) if max_v > 0 else 0
            stroke_ps.append(float(ps))

        if not stroke_ps:
            return {"pedal_smoothness_pct": None}

        return {
            "pedal_smoothness_pct": round(float(np.mean(stroke_ps)), 1),
            "pedal_smoothness_std": round(float(np.std(stroke_ps)), 1),
            "per_stroke": stroke_ps,
        }

    # ── Dead Spot Detection ─────────────────────────────────────────────────

    def dead_spot_score(self, knee_angles: np.ndarray) -> Dict:
        """
        Quantify dead spots at TDC and BDC transitions with per-transition detail.

        Dead spots are detected as zero-crossings of the signed angular velocity:
          BDC: pos→neg crossing  (max extension — knee stops extending)
          TDC: neg→pos crossing  (max flexion  — knee stops flexing)

        Per-transition measurement — dead zone duration
        ------------------------------------------------
        At each crossing, we measure how long (ms) the absolute angular velocity
        stays below a dead-zone threshold (20% of that stroke's peak velocity).
        This window is the contiguous region of frames around the crossing where
        the rider is effectively not accelerating the crank.

        A skilled rider passes through TDC/BDC quickly (short dwell time).
        A dead-spot rider lingers: the knee pauses, velocity stays near zero
        for several frames, and power delivery is interrupted.

        Severity thresholds (empirical, 30 fps baseline):
          Smooth   : < 40 ms  — elite/trained cyclists
          Minor    : 40–80 ms — recreational cyclists, normal
          Moderate : 80–140 ms — noticeable dead spot
          Severe   : > 140 ms — significant technique issue

        Score 0–100: 100 = zero dwell time at every transition (ideal);
                     0   = entire stroke spent in dead zone.
        """
        if len(knee_angles) < 10:
            return {"dead_spot_score": None}

        dt = 1.0 / self.fps
        smoothed    = self.smooth_signal(knee_angles.tolist())
        signed_vel  = self.compute_angular_velocity(smoothed)
        angular_vel = np.abs(signed_vel)

        min_gap = max(3, int(self.fps * 0.3))

        def _zero_crossings(direction: str) -> np.ndarray:
            result = []
            for i in range(1, len(signed_vel)):
                if direction == 'down' and signed_vel[i - 1] > 0 >= signed_vel[i]:
                    result.append(i)
                elif direction == 'up' and signed_vel[i - 1] < 0 <= signed_vel[i]:
                    result.append(i)
            if min_gap > 1 and result:
                filtered = [result[0]]
                for p in result[1:]:
                    if p - filtered[-1] >= min_gap:
                        filtered.append(p)
                result = filtered
            return np.array(result, dtype=int)

        bdc_indices = _zero_crossings('down')
        tdc_indices = _zero_crossings('up')

        if len(bdc_indices) == 0 or len(tdc_indices) == 0:
            return {"dead_spot_score": None}

        bdc_list = list(bdc_indices)
        tdc_list = list(tdc_indices)
        n        = len(angular_vel)

        # Angular acceleration (deg/s²) — used for continuous dwell interpolation
        angular_acc = np.gradient(signed_vel, dt)

        DEAD_ZONE_RATIO = 0.20  # threshold = 20% of stroke peak velocity

        def _stroke_bounds(centre_idx: int, boundary_indices: list):
            left  = max((b for b in boundary_indices if b <  centre_idx), default=0)
            right = min((b for b in boundary_indices if b >  centre_idx), default=n - 1)
            return left, right

        def _interpolated_dwell_ms(centre_idx: int, threshold: float) -> float:
            """
            Continuous dwell time estimate — frame-rate independent.

            Near the zero crossing, signed velocity is approximately linear:
                v(t) ≈ a · (t − t₀)   where a = dv/dt at the crossing.

            The absolute velocity |v| is V-shaped with slope |a|. The time spent
            below 'threshold' on each side is threshold/|a|, giving total dwell:
                dwell = 2 · threshold / |a|

            This is an interpolated estimate requiring only the local acceleration,
            so it has sub-frame precision at any fps.

            Falls back to a linear regression over ±3 frames around the crossing
            if the single-frame gradient is noisy.
            """
            # Fit a line to signed_vel over ±3 frames for a robust slope estimate
            half = min(3, centre_idx, n - 1 - centre_idx)
            lo, hi = centre_idx - half, centre_idx + half
            if hi > lo:
                t_seg = np.arange(lo, hi + 1, dtype=float) * dt
                v_seg = signed_vel[lo:hi + 1]
                # Linear fit: v ≈ slope * t + intercept
                slope = float(np.polyfit(t_seg - t_seg.mean(), v_seg, 1)[0])
            else:
                slope = float(angular_acc[centre_idx])

            if abs(slope) < 1e-6:
                return 0.0
            return (2.0 * threshold / abs(slope)) * 1000.0  # convert s → ms

        def _severity(dwell_ms: float) -> str:
            if dwell_ms <  40: return "smooth"
            if dwell_ms <  80: return "minor"
            if dwell_ms < 140: return "moderate"
            return "severe"

        def _build_transitions(crossing_list: list, boundary_list: list,
                               label: str) -> list:
            transitions = []
            for i, idx in enumerate(crossing_list):
                left, right  = _stroke_bounds(idx, boundary_list)
                stroke_len   = right - left
                stroke_peak  = float(np.max(angular_vel[left:right + 1])) if stroke_len > 0 else 0.0
                threshold    = stroke_peak * DEAD_ZONE_RATIO
                dwell_ms     = _interpolated_dwell_ms(idx, threshold)
                stroke_dur_ms = stroke_len * dt * 1000.0
                dwell_pct    = round(dwell_ms / stroke_dur_ms * 100.0, 1) if stroke_dur_ms > 0 else 0.0

                # Angular acceleration at the crossing (deg/s²): the direct measure
                # of how aggressively the rider drives through the dead zone.
                # Computed as the magnitude of the slope used for interpolation.
                half = min(3, idx, n - 1 - idx)
                lo2, hi2 = idx - half, idx + half
                if hi2 > lo2:
                    t_seg = np.arange(lo2, hi2 + 1, dtype=float) * dt
                    v_seg = signed_vel[lo2:hi2 + 1]
                    accel_at_crossing = abs(float(np.polyfit(t_seg - t_seg.mean(), v_seg, 1)[0]))
                else:
                    accel_at_crossing = abs(float(angular_acc[idx]))

                transitions.append({
                    "transition":          i + 1,
                    "type":                label,
                    "stroke_peak_deg_s":   round(stroke_peak, 1),
                    "threshold_deg_s":     round(threshold, 1),
                    "accel_deg_s2":        round(accel_at_crossing, 1),
                    "dwell_ms":            round(dwell_ms, 1),
                    "stroke_dur_ms":       round(stroke_dur_ms, 1),
                    "dwell_pct":           dwell_pct,
                    "severity":            _severity(dwell_ms),
                })
            return transitions

        bdc_transitions = _build_transitions(bdc_list, tdc_list, "BDC")
        tdc_transitions = _build_transitions(tdc_list, bdc_list, "TDC")

        all_transitions = bdc_transitions + tdc_transitions
        if all_transitions:
            mean_dwell_ms  = float(np.mean([t["dwell_ms"]  for t in all_transitions]))
            mean_dwell_pct = float(np.mean([t["dwell_pct"] for t in all_transitions]))
            mean_accel     = float(np.mean([t["accel_deg_s2"] for t in all_transitions]))
            score = max(0.0, 100.0 - mean_dwell_pct * 2.0)
        else:
            mean_dwell_ms = mean_dwell_pct = mean_accel = None
            score = None

        vel_peaks = _find_peaks(angular_vel, distance=max(3, int(self.fps * 0.15)))
        mean_peak_vel = float(np.mean(angular_vel[vel_peaks])) if len(vel_peaks) else None

        return {
            "dead_spot_score":     round(score, 1)          if score          is not None else None,
            "mean_peak_velocity":  round(mean_peak_vel, 1)  if mean_peak_vel  is not None else None,
            "mean_dwell_ms":       round(mean_dwell_ms, 1)  if mean_dwell_ms  is not None else None,
            "mean_dwell_pct":      round(mean_dwell_pct, 1) if mean_dwell_pct is not None else None,
            "mean_accel_deg_s2":   round(mean_accel, 1)     if mean_accel     is not None else None,
            "num_dead_spots":      len(all_transitions),
            "bdc_transitions":     bdc_transitions,
            "tdc_transitions":     tdc_transitions,
        }

    # ── Stroke Consistency ──────────────────────────────────────────────────

    def stroke_consistency(self, knee_angles: np.ndarray, timestamps: np.ndarray) -> Dict:
        """
        Measure stroke-to-stroke consistency using coefficient of variation
        of stroke duration, ROM, and peak velocity.

        Lower CV = more consistent. Elite: CV < 5%, recreational: 5-15%.
        """
        if len(knee_angles) < 10:
            return {"consistency_score": None}

        smoothed = self.smooth_signal(knee_angles.tolist())
        peaks = _find_peaks(smoothed, distance=max(3, int(self.fps * 0.3)))

        if len(peaks) < 3:
            return {"consistency_score": None}

        durations = []
        roms = []
        for i in range(len(peaks) - 1):
            seg = smoothed[peaks[i]:peaks[i + 1]]
            dur = timestamps[peaks[i + 1]] - timestamps[peaks[i]]
            rom = np.max(seg) - np.min(seg)
            durations.append(dur)
            roms.append(rom)

        dur_arr = np.array(durations)
        rom_arr = np.array(roms)

        dur_cv = float(np.std(dur_arr) / np.mean(dur_arr) * 100) if np.mean(dur_arr) > 0 else 0
        rom_cv = float(np.std(rom_arr) / np.mean(rom_arr) * 100) if np.mean(rom_arr) > 0 else 0

        # Overall consistency: average of duration and ROM CVs, inverted to a 0-100 score
        avg_cv = (dur_cv + rom_cv) / 2
        score = max(0, 100.0 - avg_cv * 5)  # 5% CV → 75, 10% CV → 50, 20% CV → 0

        return {
            "consistency_score": round(float(score), 1),
            "duration_cv_pct": round(dur_cv, 2),
            "rom_cv_pct": round(rom_cv, 2),
            "stroke_durations": [round(d, 3) for d in durations],
            "stroke_roms": [round(r, 1) for r in roms],
        }

    # ── SPARC Smoothness ────────────────────────────────────────────────────

    def sparc_smoothness(self, angles: np.ndarray) -> float:
        """
        Spectral Arc Length (SPARC) smoothness metric.

        A validated, dimensionless smoothness measure from motor control
        research. More negative = less smooth. Normalized to 0-100 scale.
        """
        if len(angles) < 10:
            return 0.0

        smoothed = self.smooth_signal(angles.tolist())
        vel = self.compute_angular_velocity(smoothed)

        # Compute power spectrum
        n = len(vel)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        spectrum = np.abs(np.fft.rfft(vel))

        # Normalize spectrum
        max_spec = np.max(spectrum)
        if max_spec < 1e-8:
            return 0.0
        norm_spectrum = spectrum / max_spec

        # Compute arc length of normalized spectrum (up to a frequency cutoff)
        freq_cutoff = min(10.0, freqs[-1])  # 10 Hz cutoff
        mask = freqs <= freq_cutoff
        f_sel = freqs[mask]
        s_sel = norm_spectrum[mask]

        if len(f_sel) < 2:
            return 0.0

        # Arc length
        df = np.diff(f_sel)
        ds = np.diff(s_sel)
        arc_length = -float(np.sum(np.sqrt(df ** 2 + ds ** 2)))

        # Map to 0-100 (empirically: -1 = very smooth, -20 = very rough)
        score = max(0.0, min(100.0, 100.0 + arc_length * 5))
        return round(score, 1)

    # ── Classic smoothness (backward-compatible) ────────────────────────────

    def smoothness_score(self, angles: np.ndarray) -> float:
        """Jerk-based smoothness score 0-100. Higher = smoother."""
        if len(angles) < 5:
            return 0.0
        smoothed = self.smooth_signal(angles.tolist())
        jerk = self.compute_jerk(smoothed)
        rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
        score = max(0.0, 100.0 - min(rms_jerk / 50.0, 100.0))
        return round(score, 1)

    # ── Cadence estimation ──────────────────────────────────────────────────

    def estimate_cadence(self, knee_angles: np.ndarray, timestamps: np.ndarray) -> Optional[float]:
        """
        Estimate pedaling cadence in RPM from knee angle oscillations.

        Typical values: recreational 60-80, trained 80-100, pro 90-110 RPM.
        """
        if len(knee_angles) < 10:
            return None
        smoothed = self.smooth_signal(knee_angles.tolist())
        signal = smoothed - np.mean(smoothed)
        peaks = _find_peaks(signal, distance=max(3, int(self.fps * 0.3)))
        if len(peaks) < 2:
            peaks = _find_peaks(-signal, distance=max(3, int(self.fps * 0.3)))
        if len(peaks) < 2:
            return None
        total_time = timestamps[-1] - timestamps[0]
        if total_time <= 0:
            return None
        cadence_rps = (len(peaks) - 1) / total_time
        return round(cadence_rps * 60, 1)

    # ── Symmetry ────────────────────────────────────────────────────────────

    def symmetry_score(self, left_data: np.ndarray, right_data: np.ndarray) -> float:
        """
        Compare left vs right limb angles. 100 = perfect symmetry.
        Uses normalized cross-correlation + mean/range comparison.
        """
        if len(left_data) == 0 or len(right_data) == 0:
            return 0.0
        mean_diff = abs(np.mean(left_data) - np.mean(right_data))
        range_diff = abs((np.max(left_data) - np.min(left_data)) -
                         (np.max(right_data) - np.min(right_data)))
        # Normalized cross-correlation for shape similarity
        min_len = min(len(left_data), len(right_data))
        l = left_data[:min_len] - np.mean(left_data[:min_len])
        r = right_data[:min_len] - np.mean(right_data[:min_len])
        ncc = float(np.correlate(l, r)[0]) / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-8)
        ncc_score = max(0, ncc) * 100  # 0-100

        # Combined score
        offset_penalty = mean_diff * 0.3 + range_diff * 0.2
        score = ncc_score * 0.5 + max(0, 100 - offset_penalty) * 0.5
        return round(float(min(100, score)), 1)

    # ── Ankle ROM Analysis ──────────────────────────────────────────────────

    def ankle_rom_analysis(self, ankle_summary: Optional[Dict]) -> Optional[Dict]:
        """
        Analyze ankle range of motion during pedaling.
        Research shows total ankle ROM ~50 deg is typical.
        Excessive ankling (>60 deg ROM) may indicate instability.
        """
        if not ankle_summary:
            return None
        rom = ankle_summary["range"]
        mean_angle = ankle_summary["mean"]

        status = "normal"
        if rom > 60:
            status = "excessive_ankling"
        elif rom < 15:
            status = "rigid_ankle"
        elif rom > 50:
            status = "high_ankling"

        return {
            "rom": round(rom, 1),
            "mean_angle": round(mean_angle, 1),
            "status": status,
            "note": {
                "normal": "Ankle ROM is within typical range (~50 deg).",
                "excessive_ankling": "Excessive ankle movement may indicate instability or incorrect cleat position.",
                "rigid_ankle": "Very limited ankle motion — may restrict pedaling efficiency.",
                "high_ankling": "Slightly high ankle ROM — monitor for fatigue-related instability.",
            }.get(status, ""),
        }

    # ── Full analysis ───────────────────────────────────────────────────────

    def analyze(self, angle_list: List[JointAngles], angle_summary: Dict) -> Dict:
        """Run full motion analysis with research-backed metrics."""
        timestamps = np.array([a.timestamp_sec for a in angle_list])
        results = {}

        # --- Per-joint smoothness ---
        joint_pairs = {
            "knee":         ("left_knee",         "right_knee"),
            "hip":          ("left_hip",           "right_hip"),
            "ankle":        ("left_ankle",         "right_ankle"),
            "elbow":        ("left_elbow",         "right_elbow"),
            "shoulder_arm": ("left_shoulder_arm",  "right_shoulder_arm"),
        }

        for joint_name, (left_key, right_key) in joint_pairs.items():
            left_summary = angle_summary.get(left_key)
            right_summary = angle_summary.get(right_key)
            left_data = np.array(left_summary["data"]) if left_summary else np.array([])
            right_data = np.array(right_summary["data"]) if right_summary else np.array([])

            results[f"{joint_name}_smoothness"] = {
                "left": self.smoothness_score(left_data) if len(left_data) > 4 else None,
                "right": self.smoothness_score(right_data) if len(right_data) > 4 else None,
            }
            results[f"{joint_name}_symmetry"] = (
                self.symmetry_score(left_data, right_data)
                if len(left_data) > 0 and len(right_data) > 0 else None
            )

        # --- Cadence ---
        knee_left = angle_summary.get("left_knee")
        knee_right = angle_summary.get("right_knee")
        cadence = None
        if knee_left:
            cadence = self.estimate_cadence(
                np.array(knee_left["data"]), np.array(knee_left["timestamps"]))
        if cadence is None and knee_right:
            cadence = self.estimate_cadence(
                np.array(knee_right["data"]), np.array(knee_right["timestamps"]))
        results["estimated_cadence_rpm"] = cadence

        # --- Pedal Smoothness (new, research-based) ---
        primary_knee = knee_left or knee_right
        if primary_knee:
            kd = np.array(primary_knee["data"])
            kt = np.array(primary_knee["timestamps"])
            results["pedal_smoothness"] = self.pedal_smoothness(kd)
            results["dead_spot"] = self.dead_spot_score(kd)
            results["stroke_consistency"] = self.stroke_consistency(kd, kt)
            results["sparc_smoothness"] = self.sparc_smoothness(kd)
        else:
            results["pedal_smoothness"] = {"pedal_smoothness_pct": None}
            results["dead_spot"] = {"dead_spot_score": None}
            results["stroke_consistency"] = {"consistency_score": None}
            results["sparc_smoothness"] = None

        # --- Ankle ROM ---
        for side in ["left", "right"]:
            ankle_s = angle_summary.get(f"{side}_ankle")
            results[f"{side}_ankle_rom"] = self.ankle_rom_analysis(ankle_s)

        # --- Elbow & arm summary stats ---
        for side in ["left", "right"]:
            elbow_s = angle_summary.get(f"{side}_elbow")
            if elbow_s:
                results[f"{side}_elbow_mean"]  = round(float(elbow_s["mean"]), 1)
                results[f"{side}_elbow_min"]   = round(float(elbow_s["min"]),  1)
                results[f"{side}_elbow_max"]   = round(float(elbow_s["max"]),  1)
                results[f"{side}_elbow_range"] = round(float(elbow_s["range"]), 1)
            else:
                results[f"{side}_elbow_mean"] = None
                results[f"{side}_elbow_min"]  = None
                results[f"{side}_elbow_max"]  = None
                results[f"{side}_elbow_range"] = None

            arm_s = angle_summary.get(f"{side}_shoulder_arm")
            if arm_s:
                results[f"{side}_shoulder_arm_mean"] = round(float(arm_s["mean"]), 1)
            else:
                results[f"{side}_shoulder_arm_mean"] = None

        # --- Trunk stability ---
        trunk_summary = angle_summary.get("trunk_angle")
        if trunk_summary:
            trunk_data = np.array(trunk_summary["data"])
            trunk_std = float(np.std(trunk_data))
            results["trunk_stability_score"] = round(max(0.0, 100.0 - trunk_std * 5), 1)
            results["trunk_angle_std"] = round(trunk_std, 2)
            results["mean_trunk_angle"] = round(float(np.mean(trunk_data)), 1)
        else:
            results["trunk_stability_score"] = None
            results["trunk_angle_std"] = None
            results["mean_trunk_angle"] = None

        # --- Overall composite score ---
        scores = []
        weights = []

        # Jerk-based smoothness (per joint)
        joint_weights = {"knee": 2.0, "hip": 1.0, "ankle": 1.0, "elbow": 0.5, "shoulder_arm": 0.5}
        for joint_name, w in joint_weights.items():
            s = results.get(f"{joint_name}_smoothness", {})
            for side_val in [s.get("left"), s.get("right")]:
                if side_val is not None:
                    scores.append(side_val)
                    weights.append(w)

        # Pedal smoothness
        ps = results["pedal_smoothness"].get("pedal_smoothness_pct")
        if ps is not None:
            scores.append(ps)
            weights.append(2.0)

        # Dead spot score
        ds = results["dead_spot"].get("dead_spot_score")
        if ds is not None:
            scores.append(ds)
            weights.append(1.5)

        # Stroke consistency
        cs = results["stroke_consistency"].get("consistency_score")
        if cs is not None:
            scores.append(cs)
            weights.append(1.5)

        # Trunk stability
        if results["trunk_stability_score"] is not None:
            scores.append(results["trunk_stability_score"])
            weights.append(1.0)

        if scores and weights:
            results["overall_motion_score"] = round(
                float(np.average(scores, weights=weights)), 1)
        else:
            results["overall_motion_score"] = None

        # --- Recommendations ---
        results["recommendations"] = self._generate_recommendations(angle_summary, results)

        return results

    def _generate_recommendations(self, angle_summary: Dict, motion: Dict) -> List[Dict]:
        """Generate evidence-based bike fit recommendations."""
        recs = []
        IR = IDEAL_RANGES

        for side in ["left", "right"]:
            side_cap = side.capitalize()

            # ── Knee at max extension (BDC) ──
            knee = angle_summary.get(f"{side}_knee")
            if knee:
                max_ext = knee["max"]  # max knee angle = most extended
                flexion_at_bdc = 180 - max_ext

                if max_ext > IR["knee_extension_max"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Knee",
                        "metric": "Hyper-extension risk at BDC",
                        "value": f"{max_ext:.1f} deg (flexion: {flexion_at_bdc:.1f} deg)",
                        "suggestion": "Saddle may be too high. Lower saddle 2-5mm. "
                                      "Research recommends 140-150 deg max extension for road bikes.",
                        "reference": "Holmes et al.; BikeDynamics guidelines",
                    })
                elif max_ext < IR["knee_extension_min"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Knee",
                        "metric": "Insufficient extension at BDC",
                        "value": f"{max_ext:.1f} deg (flexion: {flexion_at_bdc:.1f} deg)",
                        "suggestion": "Saddle is too low. Raise saddle 3-5mm. Research recommends "
                                      "33-43 deg flexion at BDC (= 137-147 deg extension).",
                        "reference": "Dynamic knee studies; Bini et al.",
                    })
                elif IR["knee_extension_ideal_low"] <= max_ext <= IR["knee_extension_ideal_high"]:
                    recs.append({
                        "type": "success", "joint": f"{side_cap} Knee Extension",
                        "metric": "Good extension at BDC",
                        "value": f"{max_ext:.1f} deg",
                        "suggestion": "Within the ideal 140-150 deg range for road bikes.",
                        "reference": "Holmes method; dynamic studies",
                    })

                # Knee at max flexion (TDC)
                min_flex = knee["min"]
                if min_flex < IR["knee_flexion_min"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Knee",
                        "metric": "Excessive flexion at TDC (patellofemoral stress risk)",
                        "value": f"{min_flex:.1f} deg",
                        "suggestion": "Knee closes too much at top of stroke. Raise saddle or "
                                      "move it back. Target >65 deg minimum knee angle.",
                        "reference": "Patellofemoral compression studies",
                    })

            # ── Hip angle ──
            hip = angle_summary.get(f"{side}_hip")
            if hip:
                if hip["min"] < IR["hip_angle_min"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Hip",
                        "metric": "Excessive hip closure at TDC",
                        "value": f"{hip['min']:.1f} deg (target: >{IR['hip_angle_min']} deg)",
                        "suggestion": "Hip angle closes too much at top of stroke, which can "
                                      "restrict breathing and power. Raise handlebars, shorten "
                                      "stem, or move saddle back.",
                        "reference": "Hip angle guidelines for TT/road",
                    })

            # ── Ankle ──
            ankle = angle_summary.get(f"{side}_ankle")
            if ankle:
                rom = ankle["range"]
                if rom > 60:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Ankle",
                        "metric": "Excessive ankling (ROM too large)",
                        "value": f"{rom:.1f}° ROM (typical: ~20-40° shin, ~50° foot)",
                        "suggestion": "Too much ankle movement may indicate instability. "
                                      "Check cleat position (move cleats rearward) and "
                                      "consider stiffer-soled shoes.",
                        "reference": "Ankle ROM research; cleat position studies",
                    })
                elif rom < 10:
                    recs.append({
                        "type": "info", "joint": f"{side_cap} Ankle",
                        "metric": "Rigid ankle pattern",
                        "value": f"{rom:.1f}° ROM",
                        "suggestion": "Very limited ankle motion. Some ankle movement helps "
                                      "smooth transitions at TDC/BDC. Consider ankle mobility work.",
                        "reference": "Pedaling biomechanics literature",
                    })

            # ── Elbow ──
            elbow = angle_summary.get(f"{side}_elbow")
            if elbow:
                mean_elbow = elbow["mean"]
                if mean_elbow < IR["elbow_angle_min"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Elbow",
                        "metric": "Excessive elbow bend",
                        "value": f"{mean_elbow:.1f}° mean (ideal: {IR['elbow_angle_ideal_low']}-{IR['elbow_angle_ideal_high']}°)",
                        "suggestion": "Arm is too bent. Likely causes: reach too long, "
                                      "bars too low, or stem too long. Try raising handlebars "
                                      "or fitting a shorter stem.",
                        "reference": "Elbow angle guidelines; Burke (2002)",
                    })
                elif mean_elbow > IR["elbow_angle_max"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Elbow",
                        "metric": "Arm nearly straight (locked out)",
                        "value": f"{mean_elbow:.1f}° mean (ideal: {IR['elbow_angle_ideal_low']}-{IR['elbow_angle_ideal_high']}°)",
                        "suggestion": "Arm is almost fully extended. This transfers road vibration "
                                      "directly to your body and reduces steering control. "
                                      "Try lowering handlebars or using a longer stem.",
                        "reference": "Elbow angle guidelines; vibration dampening research",
                    })
                elif IR["elbow_angle_ideal_low"] <= mean_elbow <= IR["elbow_angle_ideal_high"]:
                    recs.append({
                        "type": "success", "joint": f"{side_cap} Elbow",
                        "metric": "Good elbow angle",
                        "value": f"{mean_elbow:.1f}°",
                        "suggestion": f"Elbow bend is within the ideal {IR['elbow_angle_ideal_low']}-{IR['elbow_angle_ideal_high']}° range for road cycling.",
                        "reference": "Elbow angle guidelines",
                    })

            # ── Upper arm (torso-relative: hip → shoulder → elbow) ──
            arm = angle_summary.get(f"{side}_shoulder_arm")
            if arm:
                mean_arm = arm["mean"]
                if mean_arm < IR["shoulder_arm_ideal_low"]:
                    recs.append({
                        "type": "info", "joint": f"{side_cap} Upper Arm",
                        "metric": "Upper arm close to body",
                        "value": f"{mean_arm:.1f}° from torso (ideal: {IR['shoulder_arm_ideal_low']}-{IR['shoulder_arm_ideal_high']}°)",
                        "suggestion": "The upper arm stays close to the body relative to the torso. "
                                      "Bars may be too high or too close. "
                                      "Consider a longer stem or lowering the bar height.",
                        "reference": "Upper arm torso-relative angle; reach analysis",
                    })
                elif mean_arm > IR["shoulder_arm_ideal_high"]:
                    recs.append({
                        "type": "warning", "joint": f"{side_cap} Upper Arm",
                        "metric": "Upper arm reaching too far forward",
                        "value": f"{mean_arm:.1f}° from torso (ideal: {IR['shoulder_arm_ideal_low']}-{IR['shoulder_arm_ideal_high']}°)",
                        "suggestion": "The upper arm extends well forward of the torso. "
                                      "This indicates excessive reach — bars may be too far or too low. "
                                      "Consider shortening the stem or raising the handlebars.",
                        "reference": "Upper arm torso-relative angle; reach analysis",
                    })

        # ── Trunk angle ──
        trunk = angle_summary.get("trunk_angle")
        if trunk:
            mean_trunk = trunk["mean"]
            if mean_trunk < IR["trunk_aero_low"]:
                recs.append({
                    "type": "info", "joint": "Trunk",
                    "metric": "Extremely aggressive position",
                    "value": f"{mean_trunk:.1f} deg forward lean",
                    "suggestion": "Very aerodynamic but may compromise breathing and comfort. "
                                  "Sustainable mainly for short TT efforts. Ensure adequate "
                                  "hip flexibility.",
                    "reference": "Trunk angle research; aero position studies",
                })
            elif IR["trunk_aero_low"] <= mean_trunk <= IR["trunk_aero_high"]:
                recs.append({
                    "type": "success", "joint": "Trunk",
                    "metric": "Good aero/TT position",
                    "value": f"{mean_trunk:.1f} deg",
                    "suggestion": "Within typical range for time trial / aggressive road position.",
                    "reference": "Aero position guidelines",
                })
            elif IR["trunk_road_low"] <= mean_trunk <= IR["trunk_road_high"]:
                recs.append({
                    "type": "success", "joint": "Trunk",
                    "metric": "Good road riding position",
                    "value": f"{mean_trunk:.1f} deg",
                    "suggestion": "Within the ideal 30-45 deg range balancing aero and comfort.",
                    "reference": "Road cycling position research",
                })
            elif IR["trunk_recreational_low"] <= mean_trunk <= IR["trunk_recreational_high"]:
                recs.append({
                    "type": "info", "joint": "Trunk",
                    "metric": "Upright/recreational position",
                    "value": f"{mean_trunk:.1f} deg",
                    "suggestion": "Comfortable position for recreational riding. For more speed, "
                                  "gradually lower handlebars to approach 30-45 deg lean.",
                    "reference": "Comfort vs aerodynamics research",
                })
            elif mean_trunk > IR["trunk_recreational_high"]:
                recs.append({
                    "type": "info", "joint": "Trunk",
                    "metric": "Very upright position",
                    "value": f"{mean_trunk:.1f} deg",
                    "suggestion": "Consider lowering handlebars for better aerodynamics if "
                                  "comfort allows.",
                    "reference": "Position optimization studies",
                })

            # Trunk stability
            if trunk.get("std") and trunk["std"] > 5:
                recs.append({
                    "type": "warning", "joint": "Trunk Stability",
                    "metric": "Excessive upper body movement",
                    "value": f"{trunk['std']:.1f} deg SD",
                    "suggestion": "High trunk angle variation suggests core instability or "
                                  "saddle discomfort. Focus on core strength and ensure saddle "
                                  "is level and at correct height.",
                    "reference": "Core stability in cycling",
                })

        # ── Pedaling quality metrics ──
        ps = motion.get("pedal_smoothness", {}).get("pedal_smoothness_pct")
        if ps is not None and ps < 15:
            recs.append({
                "type": "warning", "joint": "Pedal Stroke",
                "metric": "Low pedal smoothness",
                "value": f"{ps:.1f}% (typical: 15-35%)",
                "suggestion": "Pedal stroke is choppy. Focus on 'scraping mud' at BDC and "
                              "'pulling back' through the recovery phase. Single-leg drills help.",
                "reference": "Pedal smoothness research; Garmin normative data",
            })

        dead_spot = motion.get("dead_spot", {})
        ds = dead_spot.get("dead_spot_score")

        # ── Per-transition type recommendations ───────────────────────────────
        # Separate BDC and TDC so suggestions can be targeted to root cause.
        bdc_transitions = dead_spot.get("bdc_transitions", []) or []
        tdc_transitions = dead_spot.get("tdc_transitions", []) or []

        def _mean_dwell(transitions):
            dwells = [t["dwell_ms"]  for t in transitions if t.get("dwell_ms")  is not None]
            pcts   = [t["dwell_pct"] for t in transitions if t.get("dwell_pct") is not None]
            return (float(np.mean(dwells)) if dwells else None,
                    float(np.mean(pcts))   if pcts   else None)

        bdc_dwell_ms, bdc_dwell_pct = _mean_dwell(bdc_transitions)
        tdc_dwell_ms, tdc_dwell_pct = _mean_dwell(tdc_transitions)

        # BDC recommendation
        if bdc_dwell_ms is not None:
            if bdc_dwell_ms >= 140 or (bdc_dwell_pct is not None and bdc_dwell_pct >= 15):
                rec_type = "warning"
                suggestion = (
                    "The knee dwells near zero velocity at the bottom of the stroke for "
                    f"{bdc_dwell_ms:.0f} ms ({bdc_dwell_pct:.1f}% of stroke). "
                    "This indicates a passive push-only stroke with no drive through BDC. "
                    "Focus on toe-down drive and ankle plantarflexion through 6 o'clock, "
                    "or actively pull up with the hamstring as the opposite leg pushes down."
                )
            elif bdc_dwell_ms >= 80 or (bdc_dwell_pct is not None and bdc_dwell_pct >= 8):
                rec_type = "info"
                suggestion = (
                    f"Moderate BDC dwell of {bdc_dwell_ms:.0f} ms ({bdc_dwell_pct:.1f}% of stroke). "
                    "There is a noticeable pause at the bottom of the stroke. "
                    "Practice ankling through BDC — the foot should transition from "
                    "heel-down to toe-down as the crank passes 6 o'clock."
                )
            else:
                rec_type = None

            if rec_type:
                recs.append({
                    "type": rec_type,
                    "joint": "Pedal Stroke — BDC",
                    "metric": "Dead zone at bottom dead centre",
                    "value": f"{bdc_dwell_ms:.0f} ms ({bdc_dwell_pct:.1f}% of stroke) — ideal < 40 ms / < 5%",
                    "suggestion": suggestion,
                    "reference": "TDC/BDC transition studies; Sanderson 1991; Coyle et al.",
                })

        # TDC recommendation
        if tdc_dwell_ms is not None:
            if tdc_dwell_ms >= 140 or (tdc_dwell_pct is not None and tdc_dwell_pct >= 15):
                rec_type = "warning"
                suggestion = (
                    "The knee dwells near zero velocity at the top of the stroke for "
                    f"{tdc_dwell_ms:.0f} ms ({tdc_dwell_pct:.1f}% of stroke). "
                    "This is a significant dead zone at TDC, typically caused by weak or "
                    "disengaged hip flexors failing to pull the pedal through 12 o'clock. "
                    "Focus on lifting the knee actively at the top of the stroke "
                    "and engage core stability to maintain power transfer."
                )
            elif tdc_dwell_ms >= 80 or (tdc_dwell_pct is not None and tdc_dwell_pct >= 8):
                rec_type = "info"
                suggestion = (
                    f"Moderate TDC dwell of {tdc_dwell_ms:.0f} ms ({tdc_dwell_pct:.1f}% of stroke). "
                    "There is a noticeable pause at the top of the stroke. "
                    "Cue yourself to 'scrape mud off the bottom of your shoe' at TDC, "
                    "actively pulling the pedal forward and upward through 12 o'clock."
                )
            else:
                rec_type = None

            if rec_type:
                recs.append({
                    "type": rec_type,
                    "joint": "Pedal Stroke — TDC",
                    "metric": "Dead zone at top dead centre",
                    "value": f"{tdc_dwell_ms:.0f} ms ({tdc_dwell_pct:.1f}% of stroke) — ideal < 40 ms / < 5%",
                    "suggestion": suggestion,
                    "reference": "TDC/BDC transition studies; Sanderson 1991; Coyle et al.",
                })

        # Fallback overall dead spot recommendation if per-transition data unavailable
        if not bdc_transitions and not tdc_transitions and ds is not None and ds < 30:
            recs.append({
                "type": "warning", "joint": "Pedal Stroke",
                "metric": "Significant dead spots at TDC/BDC",
                "value": f"Score: {ds:.0f}/100",
                "suggestion": "Large velocity drops at top and bottom of stroke. Practice "
                              "'pedaling circles' — focus on smooth transitions through "
                              "12 o'clock and 6 o'clock positions.",
                "reference": "Dead spot research; TDC/BDC transition studies",
            })

        sym = motion.get("knee_symmetry")
        if sym is not None and sym < 80:
            recs.append({
                "type": "warning", "joint": "Left/Right Balance",
                "metric": "Pedaling asymmetry detected",
                "value": f"{sym:.1f}%",
                "suggestion": "Significant difference between left and right legs. Check "
                              "cleat alignment, saddle tilt, and leg length discrepancy. "
                              "Consider single-leg drills to balance strength.",
                "reference": "Bilateral symmetry in cycling research",
            })

        cs = motion.get("stroke_consistency", {}).get("consistency_score")
        if cs is not None and cs < 60:
            recs.append({
                "type": "info", "joint": "Pedal Stroke",
                "metric": "Inconsistent stroke pattern",
                "value": f"Score: {cs:.0f}/100",
                "suggestion": "Stroke-to-stroke variation is high. This may indicate fatigue, "
                              "gear selection issues, or developing pedaling technique.",
                "reference": "Stroke consistency studies",
            })

        if not recs:
            recs.append({
                "type": "success", "joint": "Overall",
                "metric": "Position and pedaling look good",
                "value": "-",
                "suggestion": "No major fitting issues detected. Keep monitoring for "
                              "fatigue-related changes during longer rides.",
                "reference": "General bike fit guidelines",
            })

        return recs
