"""
Frontal (front-view) knee dynamics analyzer.

Analyzes a front-facing video of a cyclist to detect:
1. Knee valgus/varus — medial or lateral deviation of the knee
2. Knee tracking path — lateral movement pattern through the pedal stroke
3. Lateral knee displacement — how far the knee moves side-to-side
4. Hip-knee-ankle alignment — Q-angle proxy from frontal plane
5. Left/right symmetry in frontal plane

Research basis:
- Cycling is primarily sagittal but slight medial knee movement at TDC
  reaching peak around 90 deg crank angle is normal (PMC5950749)
- Excessive valgus/varus indicates cleat, stance width, or orthotic issues
- Vertical alignment of hip-knee-ankle reduces injury risk
- Q-factor (pedal stance width) affects frontal knee loading
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .pose_engine import PoseFrame, LANDMARK_INDICES


@dataclass
class FrontalFrameData:
    """Frontal plane knee tracking data for a single frame."""
    frame_index: int
    timestamp_sec: float

    # Knee lateral position relative to hip-ankle line (pixels)
    # Positive = medial (valgus), Negative = lateral (varus)
    left_knee_deviation: Optional[float] = None
    right_knee_deviation: Optional[float] = None

    # Normalized deviation as percentage of thigh length
    left_knee_deviation_pct: Optional[float] = None
    right_knee_deviation_pct: Optional[float] = None

    # Hip-knee-ankle angle in frontal plane (180 = perfectly aligned)
    left_frontal_angle: Optional[float] = None
    right_frontal_angle: Optional[float] = None

    # Knee x-position (normalized) for tracking path
    left_knee_x: Optional[float] = None
    right_knee_x: Optional[float] = None


def _lateral_deviation(hip_x: float, knee_x: float, ankle_x: float,
                       hip_y: float, knee_y: float, ankle_y: float) -> float:
    """
    Compute lateral deviation of the knee from the hip-ankle line.

    Uses point-to-line distance in the frontal plane.
    Positive = knee is medial (toward midline / valgus direction)
    Negative = knee is lateral (away from midline / varus direction)

    Convention: For left leg, medial = toward right (positive x).
                For right leg, medial = toward left (negative x).
    """
    # Vector from hip to ankle
    dx = ankle_x - hip_x
    dy = ankle_y - hip_y
    line_len = np.sqrt(dx ** 2 + dy ** 2)
    if line_len < 1e-6:
        return 0.0

    # Signed distance of knee from hip-ankle line
    # Using cross product: (ankle-hip) x (knee-hip) / |ankle-hip|
    cross = dx * (knee_y - hip_y) - dy * (knee_x - hip_x)
    return float(cross / line_len)


def _frontal_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    Compute the hip-knee-ankle angle in the frontal plane.
    180 deg = perfectly straight alignment.
    < 180 = valgus (knee medial to line)
    > 180 = varus (knee lateral to line)
    """
    v1 = hip - knee
    v2 = ankle - knee
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))

    # Determine sign using cross product (which side the knee deviates)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        angle = 360 - angle

    return angle


def _get_px(frame: PoseFrame, name: str) -> Optional[np.ndarray]:
    """Get normalized (x, y) for a landmark, visibility-gated."""
    if not frame.has_pose:
        return None
    idx = LANDMARK_INDICES.get(name)
    if idx is None or idx >= len(frame.visibility):
        return None
    if frame.visibility[idx] < 0.3:
        return None
    return frame.landmarks[idx, :2].copy()


class FrontalAnalyzer:
    """
    Analyzes frontal-view video for knee dynamics.

    Key metrics:
    - Knee valgus/varus angle and deviation
    - Lateral knee tracking pattern through pedal stroke
    - Hip-knee-ankle alignment quality
    - Left/right frontal symmetry
    """

    def analyze_frame(self, frame: PoseFrame) -> FrontalFrameData:
        """Extract frontal knee dynamics from a single frame."""
        data = FrontalFrameData(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
        )

        if not frame.has_pose:
            return data

        for side, sign in [("left", 1.0), ("right", -1.0)]:
            hip = _get_px(frame, f"{side}_hip")
            knee = _get_px(frame, f"{side}_knee")
            ankle = _get_px(frame, f"{side}_ankle")

            if hip is None or knee is None or ankle is None:
                continue

            # Lateral deviation (signed pixel distance from hip-ankle line)
            dev = _lateral_deviation(hip[0], knee[0], ankle[0],
                                     hip[1], knee[1], ankle[1])
            # Apply sign convention: positive = medial for both legs
            dev_signed = dev * sign

            # Normalize by thigh length
            thigh_len = np.sqrt((knee[0] - hip[0]) ** 2 + (knee[1] - hip[1]) ** 2)
            dev_pct = (dev_signed / thigh_len * 100) if thigh_len > 0.01 else 0.0

            # Frontal angle
            f_angle = _frontal_angle(hip, knee, ankle)

            if side == "left":
                data.left_knee_deviation = dev_signed
                data.left_knee_deviation_pct = dev_pct
                data.left_frontal_angle = f_angle
                data.left_knee_x = float(knee[0])
            else:
                data.right_knee_deviation = dev_signed
                data.right_knee_deviation_pct = dev_pct
                data.right_frontal_angle = f_angle
                data.right_knee_x = float(knee[0])

        return data

    def analyze_all(self, pose_frames: List[PoseFrame]) -> List[FrontalFrameData]:
        """Analyze all frames from frontal video."""
        return [self.analyze_frame(f) for f in pose_frames]

    def summarize(self, frame_data: List[FrontalFrameData]) -> Dict:
        """
        Compute comprehensive frontal plane metrics.

        Returns dict with:
        - Per-side deviation stats (mean, max, std, range)
        - Knee tracking path data
        - Valgus/varus classification
        - Frontal alignment quality score
        - Symmetry assessment
        - Recommendations
        """
        results = {}

        for side in ["left", "right"]:
            dev_key = f"{side}_knee_deviation"
            dev_pct_key = f"{side}_knee_deviation_pct"
            angle_key = f"{side}_frontal_angle"
            x_key = f"{side}_knee_x"

            devs = [getattr(f, dev_key) for f in frame_data if getattr(f, dev_key) is not None]
            dev_pcts = [getattr(f, dev_pct_key) for f in frame_data if getattr(f, dev_pct_key) is not None]
            angles = [getattr(f, angle_key) for f in frame_data if getattr(f, angle_key) is not None]
            x_positions = [getattr(f, x_key) for f in frame_data if getattr(f, x_key) is not None]
            timestamps = [f.timestamp_sec for f in frame_data if getattr(f, dev_key) is not None]

            if not devs:
                results[side] = None
                continue

            dev_arr = np.array(devs)
            dev_pct_arr = np.array(dev_pcts)
            angle_arr = np.array(angles)
            x_arr = np.array(x_positions)

            # Classification based on mean deviation percentage
            mean_dev_pct = float(np.mean(dev_pct_arr))
            if mean_dev_pct > 8:
                classification = "valgus"  # knee moves medially (knock-kneed)
            elif mean_dev_pct < -8:
                classification = "varus"   # knee moves laterally (bow-legged)
            else:
                classification = "neutral"

            # Tracking quality: how much lateral wobble?
            x_range = float(np.max(x_arr) - np.min(x_arr)) if len(x_arr) > 1 else 0
            x_std = float(np.std(x_arr)) if len(x_arr) > 1 else 0

            # Tracking score: 100 = perfectly straight line, 0 = lots of wobble
            # Normalize by typical knee width in normalized coords (~0.05-0.15 range)
            tracking_score = max(0.0, 100.0 - x_std * 2000)

            results[side] = {
                "deviation_mean": round(float(np.mean(dev_arr)), 3),
                "deviation_max": round(float(np.max(np.abs(dev_arr))), 3),
                "deviation_std": round(float(np.std(dev_arr)), 3),
                "deviation_pct_mean": round(mean_dev_pct, 1),
                "deviation_pct_max": round(float(np.max(np.abs(dev_pct_arr))), 1),
                "deviation_pct_std": round(float(np.std(dev_pct_arr)), 1),
                "frontal_angle_mean": round(float(np.mean(angle_arr)), 1),
                "frontal_angle_std": round(float(np.std(angle_arr)), 1),
                "classification": classification,
                "tracking_score": round(tracking_score, 1),
                "lateral_range": round(x_range, 4),
                "lateral_std": round(x_std, 4),
                "timestamps": [round(t, 3) for t in timestamps],
                "deviation_data": [round(d, 3) for d in devs],
                "deviation_pct_data": [round(d, 1) for d in dev_pcts],
                "x_position_data": [round(x, 4) for x in x_positions],
            }

        # --- Bilateral symmetry ---
        if results.get("left") and results.get("right"):
            l = results["left"]
            r = results["right"]
            dev_diff = abs(l["deviation_pct_mean"] - r["deviation_pct_mean"])
            tracking_diff = abs(l["tracking_score"] - r["tracking_score"])
            # Symmetry: 100 = identical, penalized by deviation and tracking diffs
            symmetry = max(0.0, 100.0 - dev_diff * 2 - tracking_diff * 0.5)
            results["frontal_symmetry"] = round(symmetry, 1)
        else:
            results["frontal_symmetry"] = None

        # --- Overall frontal score ---
        scores = []
        for side in ["left", "right"]:
            if results.get(side):
                scores.append(results[side]["tracking_score"])
                # Bonus for neutral classification
                if results[side]["classification"] == "neutral":
                    scores.append(90.0)
                elif results[side]["classification"] in ("valgus", "varus"):
                    # Penalize based on severity
                    severity = abs(results[side]["deviation_pct_mean"])
                    scores.append(max(0, 80 - severity * 2))
        if results.get("frontal_symmetry"):
            scores.append(results["frontal_symmetry"])

        results["frontal_score"] = round(float(np.mean(scores)), 1) if scores else None

        # --- Recommendations ---
        results["frontal_recommendations"] = self._generate_recommendations(results)

        return results

    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate frontal-plane bike fit recommendations."""
        recs = []

        for side in ["left", "right"]:
            side_cap = side.capitalize()
            data = results.get(side)
            if not data:
                continue

            cls = data["classification"]
            dev_pct = data["deviation_pct_mean"]
            tracking = data["tracking_score"]

            if cls == "valgus":
                severity = "moderate" if abs(dev_pct) < 15 else "significant"
                recs.append({
                    "type": "warning",
                    "joint": f"{side_cap} Knee (Frontal)",
                    "metric": f"Knee valgus ({severity} medial deviation)",
                    "value": f"{dev_pct:.1f}% medial",
                    "suggestion": (
                        "Knee tracks medially (inward) during pedaling. "
                        "Check: (1) Cleat rotation — may need more external rotation. "
                        "(2) Stance width / Q-factor — try wider pedal spacers. "
                        "(3) Varus wedges under cleats or insoles. "
                        "(4) Arch support — flat feet contribute to valgus."
                    ),
                    "reference": "PMC5950749; frontal plane knee biomechanics",
                })
            elif cls == "varus":
                severity = "moderate" if abs(dev_pct) < 15 else "significant"
                recs.append({
                    "type": "warning",
                    "joint": f"{side_cap} Knee (Frontal)",
                    "metric": f"Knee varus ({severity} lateral deviation)",
                    "value": f"{dev_pct:.1f}% lateral",
                    "suggestion": (
                        "Knee tracks laterally (outward) during pedaling. "
                        "Check: (1) Cleat rotation — may need more internal rotation. "
                        "(2) Stance width — try narrower Q-factor. "
                        "(3) Valgus wedges may help. "
                        "(4) Saddle height — too high can cause lateral tracking."
                    ),
                    "reference": "Frontal plane knee studies",
                })
            else:
                recs.append({
                    "type": "success",
                    "joint": f"{side_cap} Knee (Frontal)",
                    "metric": "Good frontal alignment",
                    "value": f"{dev_pct:.1f}% deviation",
                    "suggestion": "Knee tracks well within the hip-ankle corridor.",
                    "reference": "Hip-knee-ankle alignment guidelines",
                })

            if tracking < 60:
                recs.append({
                    "type": "warning",
                    "joint": f"{side_cap} Knee Tracking",
                    "metric": "Excessive lateral wobble",
                    "value": f"Score: {tracking:.0f}/100",
                    "suggestion": (
                        "Knee moves excessively side-to-side during pedaling. "
                        "This can indicate: (1) Weak hip stabilizers (glute medius). "
                        "(2) Loose cleats. (3) Incorrect float setting. "
                        "(4) Saddle too high causing rocking."
                    ),
                    "reference": "Knee tracking and patellar stability",
                })

        if results.get("frontal_symmetry") is not None and results["frontal_symmetry"] < 75:
            recs.append({
                "type": "warning",
                "joint": "Frontal Symmetry",
                "metric": "Left/right frontal plane imbalance",
                "value": f"{results['frontal_symmetry']:.1f}%",
                "suggestion": (
                    "Significant difference in frontal knee motion between legs. "
                    "Check for: leg length discrepancy, unilateral muscle weakness, "
                    "asymmetric cleat setup, or saddle tilt."
                ),
                "reference": "Bilateral symmetry in cycling",
            })

        if not recs:
            recs.append({
                "type": "success",
                "joint": "Frontal Plane",
                "metric": "Good knee alignment",
                "value": "-",
                "suggestion": "No frontal plane issues detected.",
                "reference": "General bike fit guidelines",
            })

        return recs
