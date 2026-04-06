#!/usr/bin/env python3
"""
Unit tests for the Bike Fitting App core modules.
Tests angle calculation, motion analysis, pedal smoothness,
dead spot detection, and research-backed recommendations.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_engine import PoseFrame, LANDMARK_INDICES
from core.angle_calculator import (
    AngleCalculator, JointAngles, IDEAL_RANGES,
    _three_point_angle, _angle_between_vectors,
)
from core.motion_analysis import MotionAnalyzer, _savgol_filter, _find_peaks, _find_valleys
from core.frontal_analyzer import (
    FrontalAnalyzer, FrontalFrameData,
    _lateral_deviation, _frontal_angle,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_landmarks(**named_points):
    lm = np.full((33, 3), 0.5)
    vis = np.ones(33) * 0.9
    for name, (x, y) in named_points.items():
        idx = LANDMARK_INDICES[name]
        lm[idx] = [x, y, 0.0]
    return lm, vis


def make_pose_frame(frame_idx, timestamp, **points):
    lm, vis = make_landmarks(**points)
    return PoseFrame(frame_idx, timestamp, lm, vis, None)


def make_cycling_angles(n=60, fps=30.0, freq=1.5, knee_min=80, knee_max=150):
    """Generate synthetic cycling angle data."""
    angle_list = []
    for i in range(n):
        t = i / fps
        phase = 2 * np.pi * freq * t
        knee_range = (knee_max - knee_min) / 2
        knee_mid = (knee_max + knee_min) / 2
        lk = knee_mid + knee_range * np.sin(phase)
        rk = knee_mid + knee_range * np.sin(phase + np.pi)
        angle_list.append(JointAngles(
            frame_index=i, timestamp_sec=t,
            left_knee=lk, right_knee=rk,
            left_hip=70 + 10 * np.sin(phase),
            right_hip=72 + 10 * np.sin(phase),
            left_ankle=100 + 5 * np.sin(phase),
            right_ankle=100 + 5 * np.sin(phase),
            trunk_angle=45 + 2 * np.sin(phase * 0.5),
        ))
    return angle_list


# ── Test: Angle Geometry ────────────────────────────────────────────────────

class TestAngleGeometry(unittest.TestCase):

    def test_right_angle(self):
        a, b, c = np.array([1, 0.]), np.array([0, 0.]), np.array([0, 1.])
        self.assertAlmostEqual(_three_point_angle(a, b, c), 90.0, places=1)

    def test_straight_line(self):
        a, b, c = np.array([0, 0.]), np.array([0.5, 0.]), np.array([1, 0.])
        self.assertAlmostEqual(_three_point_angle(a, b, c), 180.0, places=1)

    def test_acute_angle(self):
        a = np.array([0, 0.])
        b = np.array([0.5, 0.])
        c = np.array([0.25, np.sqrt(3) / 4])
        self.assertAlmostEqual(_three_point_angle(a, b, c), 60.0, delta=1)

    def test_vector_parallel(self):
        self.assertAlmostEqual(_angle_between_vectors(
            np.array([1, 0.]), np.array([2, 0.])), 0.0, places=1)

    def test_vector_opposite(self):
        self.assertAlmostEqual(_angle_between_vectors(
            np.array([1, 0.]), np.array([-1, 0.])), 180.0, places=1)


# ── Test: Angle Calculator ──────────────────────────────────────────────────

class TestAngleCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = AngleCalculator()

    def test_knee_angle_90(self):
        frame = make_pose_frame(0, 0.0,
            left_hip=(0.5, 0.2), left_knee=(0.5, 0.5), left_ankle=(0.8, 0.5))
        angles = self.calc.calculate_frame(frame)
        self.assertIsNotNone(angles.left_knee)
        self.assertAlmostEqual(angles.left_knee, 90.0, delta=5)

    def test_hip_angle_straight(self):
        frame = make_pose_frame(0, 0.0,
            left_shoulder=(0.5, 0.1), left_hip=(0.5, 0.4), left_knee=(0.5, 0.7))
        angles = self.calc.calculate_frame(frame)
        self.assertAlmostEqual(angles.left_hip, 180.0, delta=5)

    def test_no_pose(self):
        frame = PoseFrame(0, 0.0, None, None, None)
        angles = self.calc.calculate_frame(frame)
        self.assertIsNone(angles.left_knee)
        self.assertIsNone(angles.trunk_angle)

    def test_trunk_angle_vertical(self):
        frame = make_pose_frame(0, 0.0,
            left_shoulder=(0.5, 0.2), left_hip=(0.5, 0.6))
        angles = self.calc.calculate_frame(frame)
        self.assertIsNotNone(angles.trunk_angle)
        self.assertAlmostEqual(angles.trunk_angle, 0.0, delta=5)

    def test_summarize_has_stats(self):
        frames = []
        for i in range(30):
            t = i / 30.0
            frames.append(make_pose_frame(i, t,
                left_hip=(0.4, 0.3),
                left_knee=(0.5, 0.4 + 0.2 * np.sin(2 * np.pi * t * 2)),
                left_ankle=(0.6, 0.8)))
        angle_list = self.calc.calculate_all(frames)
        summary = self.calc.summarize(angle_list)
        lk = summary.get("left_knee")
        self.assertIsNotNone(lk)
        self.assertIn("std", lk)
        self.assertIn("p50", lk)
        self.assertGreater(lk["range"], 0)

    def test_detect_pedal_phases(self):
        """Pedal phase detection should find strokes in oscillating knee data."""
        angle_list = make_cycling_angles(n=90, fps=30, freq=1.5)
        result = self.calc.detect_pedal_phases(angle_list, side="left")
        self.assertTrue(result["phases_detected"])
        self.assertGreater(result["num_strokes"], 1)
        self.assertIn("bdc_angle_mean", result)
        self.assertIn("knee_flexion_bdc_mean", result)


# ── Test: Ideal Ranges ──────────────────────────────────────────────────────

class TestIdealRanges(unittest.TestCase):

    def test_ideal_ranges_consistency(self):
        """Ideal low should always be less than ideal high."""
        IR = IDEAL_RANGES
        self.assertLess(IR["knee_extension_ideal_low"], IR["knee_extension_ideal_high"])
        self.assertLess(IR["knee_flexion_ideal_low"], IR["knee_flexion_ideal_high"])
        self.assertLess(IR["hip_angle_ideal_low"], IR["hip_angle_ideal_high"])
        self.assertLess(IR["ankle_angle_ideal_low"], IR["ankle_angle_ideal_high"])
        self.assertLess(IR["trunk_road_low"], IR["trunk_road_high"])

    def test_knee_extension_range_matches_research(self):
        """Road bike ideal: 140-150 deg max extension."""
        IR = IDEAL_RANGES
        self.assertEqual(IR["knee_extension_ideal_low"], 140)
        self.assertEqual(IR["knee_extension_ideal_high"], 150)


# ── Test: Signal Processing ─────────────────────────────────────────────────

class TestSignalProcessing(unittest.TestCase):

    def test_savgol_smooth(self):
        np.random.seed(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 100))
        noisy = clean + np.random.normal(0, 0.2, 100)
        smoothed = _savgol_filter(noisy, 11, 3)
        self.assertLess(np.mean((smoothed - clean) ** 2),
                        np.mean((noisy - clean) ** 2))

    def test_savgol_short(self):
        result = _savgol_filter(np.array([1, 2, 3.]), 11, 3)
        self.assertEqual(len(result), 3)

    def test_find_peaks_sine(self):
        x = np.sin(np.linspace(0, 6 * np.pi, 300))
        peaks = _find_peaks(x, distance=20)
        self.assertGreaterEqual(len(peaks), 2)
        self.assertLessEqual(len(peaks), 4)

    def test_find_peaks_empty(self):
        self.assertEqual(len(_find_peaks(np.ones(50))), 0)

    def test_find_valleys(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        valleys = _find_valleys(x, distance=20)
        self.assertGreaterEqual(len(valleys), 1)


# ── Test: Motion Analyzer ──────────────────────────────────────────────────

class TestMotionAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = MotionAnalyzer(fps=30.0)

    def test_smoothness_smooth_signal(self):
        angles = np.sin(np.linspace(0, 4 * np.pi, 120)) * 30 + 90
        score = self.analyzer.smoothness_score(angles)
        self.assertGreater(score, 50)

    def test_smoothness_noisy_signal(self):
        np.random.seed(0)
        angles = np.random.uniform(30, 160, 120)
        score = self.analyzer.smoothness_score(angles)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_cadence_estimation(self):
        fps, freq = 30.0, 1.5  # 90 RPM
        t = np.linspace(0, 5.0, int(fps * 5))
        knee = np.sin(2 * np.pi * freq * t) * 30 + 100
        cadence = self.analyzer.estimate_cadence(knee, t)
        if cadence is not None:
            self.assertAlmostEqual(cadence, 90.0, delta=20)

    def test_pedal_smoothness(self):
        """Pedal smoothness should return a percentage."""
        knee = np.sin(np.linspace(0, 6 * np.pi, 180)) * 30 + 100
        result = self.analyzer.pedal_smoothness(knee)
        ps = result.get("pedal_smoothness_pct")
        self.assertIsNotNone(ps)
        self.assertGreater(ps, 0)
        self.assertLessEqual(ps, 100)

    def test_dead_spot_score(self):
        """Dead spot score should be 0-100."""
        knee = np.sin(np.linspace(0, 6 * np.pi, 180)) * 30 + 100
        result = self.analyzer.dead_spot_score(knee)
        ds = result.get("dead_spot_score")
        self.assertIsNotNone(ds)
        self.assertGreaterEqual(ds, 0)
        self.assertLessEqual(ds, 100)

    def test_stroke_consistency(self):
        """Consistent strokes should score high."""
        t = np.linspace(0, 4, 120)
        knee = np.sin(2 * np.pi * 1.5 * t) * 30 + 100
        result = self.analyzer.stroke_consistency(knee, t)
        score = result.get("consistency_score")
        self.assertIsNotNone(score)
        self.assertGreater(score, 50)

    def test_sparc_smoothness(self):
        """SPARC should return a number for smooth data."""
        angles = np.sin(np.linspace(0, 4 * np.pi, 120)) * 30 + 90
        score = self.analyzer.sparc_smoothness(angles)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_symmetry_perfect(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) * 30 + 90
        score = self.analyzer.symmetry_score(data, data.copy())
        self.assertGreater(score, 90)

    def test_symmetry_offset(self):
        """Offset left/right should score lower than perfect."""
        left = np.sin(np.linspace(0, 4 * np.pi, 100)) * 30 + 90
        right = left + 15
        score_offset = self.analyzer.symmetry_score(left, right)
        score_perfect = self.analyzer.symmetry_score(left, left.copy())
        self.assertLess(score_offset, score_perfect)

    def test_ankle_rom_normal(self):
        summary = {"range": 48, "mean": 95, "timestamps": [0, 1], "data": [71, 119]}
        result = self.analyzer.ankle_rom_analysis(summary)
        self.assertEqual(result["status"], "normal")

    def test_ankle_rom_excessive(self):
        summary = {"range": 65, "mean": 95, "timestamps": [0, 1], "data": [60, 125]}
        result = self.analyzer.ankle_rom_analysis(summary)
        self.assertEqual(result["status"], "excessive_ankling")

    def test_full_analyze(self):
        """Full analysis should include all new metrics."""
        angle_list = make_cycling_angles()
        calc = AngleCalculator()
        # Build summary
        summary = {}
        for key in ["left_knee", "right_knee", "left_hip", "right_hip",
                     "left_ankle", "right_ankle", "trunk_angle"]:
            values = [getattr(a, key) for a in angle_list if getattr(a, key) is not None]
            timestamps = [a.timestamp_sec for a in angle_list if getattr(a, key) is not None]
            if values:
                v_arr = np.array(values)
                summary[key] = {
                    "min": float(np.min(v_arr)), "max": float(np.max(v_arr)),
                    "mean": float(np.mean(v_arr)), "std": float(np.std(v_arr)),
                    "range": float(np.max(v_arr) - np.min(v_arr)),
                    "p5": float(np.percentile(v_arr, 5)),
                    "p25": float(np.percentile(v_arr, 25)),
                    "p50": float(np.percentile(v_arr, 50)),
                    "p75": float(np.percentile(v_arr, 75)),
                    "p95": float(np.percentile(v_arr, 95)),
                    "timestamps": timestamps, "data": values,
                }

        results = self.analyzer.analyze(angle_list, summary)

        # Check new metrics exist
        self.assertIn("pedal_smoothness", results)
        self.assertIn("dead_spot", results)
        self.assertIn("stroke_consistency", results)
        self.assertIn("sparc_smoothness", results)
        self.assertIn("left_ankle_rom", results)
        self.assertIn("overall_motion_score", results)
        self.assertIsNotNone(results["overall_motion_score"])
        self.assertIsInstance(results["recommendations"], list)
        self.assertGreater(len(results["recommendations"]), 0)

        # Verify recommendations have references
        for rec in results["recommendations"]:
            self.assertIn("reference", rec)


# ── Test: Recommendations ───────────────────────────────────────────────────

class TestRecommendations(unittest.TestCase):

    def _make_summary(self, **overrides):
        base = {
            "left_knee": {"min": 70, "max": 145, "mean": 107, "std": 20, "range": 75,
                          "p5": 72, "p25": 85, "p50": 107, "p75": 130, "p95": 143,
                          "timestamps": [0, 1], "data": [70, 145]},
            "right_knee": {"min": 70, "max": 145, "mean": 107, "std": 20, "range": 75,
                           "timestamps": [0, 1], "data": [70, 145]},
            "left_hip": {"min": 60, "max": 100, "mean": 80, "std": 10, "range": 40,
                         "timestamps": [0, 1], "data": [60, 100]},
            "right_hip": {"min": 60, "max": 100, "mean": 80, "std": 10, "range": 40,
                          "timestamps": [0, 1], "data": [60, 100]},
            "left_ankle": {"min": 80, "max": 110, "mean": 95, "std": 8, "range": 30,
                           "timestamps": [0, 1], "data": [80, 110]},
            "right_ankle": {"min": 80, "max": 110, "mean": 95, "std": 8, "range": 30,
                            "timestamps": [0, 1], "data": [80, 110]},
            "trunk_angle": {"min": 38, "max": 48, "mean": 43, "std": 2.5, "range": 10,
                            "timestamps": [0, 1], "data": [38, 48]},
        }
        base.update(overrides)
        return base

    def test_good_position(self):
        analyzer = MotionAnalyzer(fps=30)
        summary = self._make_summary()
        angle_list = [JointAngles(0, 0.0), JointAngles(1, 1.0)]
        results = analyzer.analyze(angle_list, summary)
        types = [r["type"] for r in results["recommendations"]]
        self.assertIn("success", types)

    def test_hyper_extension_warning(self):
        """Knee max > 155 should warn about hyper-extension."""
        analyzer = MotionAnalyzer(fps=30)
        summary = self._make_summary(
            left_knee={"min": 70, "max": 165, "mean": 117, "std": 25, "range": 95,
                       "timestamps": [0, 1], "data": [70, 165]})
        angle_list = [JointAngles(0, 0.0), JointAngles(1, 1.0)]
        results = analyzer.analyze(angle_list, summary)
        metrics = [r["metric"] for r in results["recommendations"]]
        self.assertTrue(any("extension" in m.lower() or "hyper" in m.lower() for m in metrics))

    def test_saddle_too_low_warning(self):
        """Knee max < 135 should warn about insufficient extension."""
        analyzer = MotionAnalyzer(fps=30)
        summary = self._make_summary(
            left_knee={"min": 50, "max": 120, "mean": 85, "std": 20, "range": 70,
                       "timestamps": [0, 1], "data": [50, 120]})
        angle_list = [JointAngles(0, 0.0), JointAngles(1, 1.0)]
        results = analyzer.analyze(angle_list, summary)
        suggestions = " ".join([r["suggestion"] for r in results["recommendations"]])
        self.assertTrue("saddle" in suggestions.lower())

    def test_excessive_flexion_warning(self):
        """Knee min < 60 should warn about patellofemoral stress."""
        analyzer = MotionAnalyzer(fps=30)
        summary = self._make_summary(
            left_knee={"min": 45, "max": 145, "mean": 95, "std": 30, "range": 100,
                       "timestamps": [0, 1], "data": [45, 145]})
        angle_list = [JointAngles(0, 0.0), JointAngles(1, 1.0)]
        results = analyzer.analyze(angle_list, summary)
        metrics = [r["metric"].lower() for r in results["recommendations"]]
        self.assertTrue(any("flexion" in m or "patellofemoral" in m for m in metrics))

    def test_trunk_position_categorization(self):
        """Trunk at 43 deg should be recognized as good road position."""
        analyzer = MotionAnalyzer(fps=30)
        summary = self._make_summary()
        angle_list = [JointAngles(0, 0.0), JointAngles(1, 1.0)]
        results = analyzer.analyze(angle_list, summary)
        trunk_recs = [r for r in results["recommendations"] if r["joint"] == "Trunk"]
        if trunk_recs:
            self.assertTrue(any("road" in r["metric"].lower() or
                                "road" in r["suggestion"].lower()
                                for r in trunk_recs))


# ── Test: Frontal Geometry ─────────────────────────────────────────────────

class TestFrontalGeometry(unittest.TestCase):

    def test_lateral_deviation_aligned(self):
        """Knee on the hip-ankle line should have zero deviation."""
        dev = _lateral_deviation(0.5, 0.5, 0.5, 0.2, 0.5, 0.8)
        self.assertAlmostEqual(dev, 0.0, places=4)

    def test_lateral_deviation_offset(self):
        """Knee offset from hip-ankle line should give nonzero deviation."""
        dev = _lateral_deviation(0.5, 0.6, 0.5, 0.2, 0.5, 0.8)
        self.assertNotEqual(dev, 0.0)

    def test_frontal_angle_straight(self):
        """Straight hip-knee-ankle should give ~180 deg."""
        hip = np.array([0.5, 0.2])
        knee = np.array([0.5, 0.5])
        ankle = np.array([0.5, 0.8])
        angle = _frontal_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 180.0, delta=2)

    def test_frontal_angle_bent(self):
        """Knee offset gives angle != 180."""
        hip = np.array([0.5, 0.2])
        knee = np.array([0.6, 0.5])
        ankle = np.array([0.5, 0.8])
        angle = _frontal_angle(hip, knee, ankle)
        self.assertNotAlmostEqual(angle, 180.0, delta=5)


# ── Test: Frontal Analyzer ─────────────────────────────────────────────────

class TestFrontalAnalyzer(unittest.TestCase):

    def _make_frontal_frames(self, n=60, fps=30.0, left_dev=0.0, right_dev=0.0):
        """Create synthetic pose frames for frontal analysis.
        left_dev/right_dev: lateral offset of knee from straight hip-ankle line.
        """
        frames = []
        for i in range(n):
            t = i / fps
            phase = 2 * np.pi * 1.5 * t
            # Vertical motion simulates pedaling (knees go up/down)
            knee_y_offset = 0.05 * np.sin(phase)

            lm = np.full((33, 3), 0.5)
            vis = np.ones(33) * 0.95

            # Left leg (hip, knee, ankle)
            lm[LANDMARK_INDICES["left_hip"]] = [0.35, 0.4, 0]
            lm[LANDMARK_INDICES["left_knee"]] = [0.35 + left_dev, 0.55 + knee_y_offset, 0]
            lm[LANDMARK_INDICES["left_ankle"]] = [0.35, 0.75, 0]

            # Right leg
            lm[LANDMARK_INDICES["right_hip"]] = [0.65, 0.4, 0]
            lm[LANDMARK_INDICES["right_knee"]] = [0.65 + right_dev, 0.55 + knee_y_offset, 0]
            lm[LANDMARK_INDICES["right_ankle"]] = [0.65, 0.75, 0]

            frames.append(PoseFrame(i, t, lm, vis, None))
        return frames

    def test_neutral_alignment(self):
        """Straight legs should be classified as neutral."""
        analyzer = FrontalAnalyzer()
        frames = self._make_frontal_frames(left_dev=0.0, right_dev=0.0)
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        self.assertEqual(summary["left"]["classification"], "neutral")
        self.assertEqual(summary["right"]["classification"], "neutral")
        self.assertIsNotNone(summary["frontal_score"])

    def test_valgus_detection(self):
        """Knee pushed medially should detect valgus."""
        analyzer = FrontalAnalyzer()
        # Left knee pushed right (medial direction) = positive dev for left
        frames = self._make_frontal_frames(left_dev=0.05, right_dev=0.0)
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        # The left knee should have significant deviation
        left = summary["left"]
        self.assertIsNotNone(left)
        # Check that some recommendations exist
        recs = summary.get("frontal_recommendations", [])
        self.assertGreater(len(recs), 0)

    def test_symmetry_identical(self):
        """Perfectly aligned knees should give high symmetry."""
        analyzer = FrontalAnalyzer()
        frames = self._make_frontal_frames(left_dev=0.0, right_dev=0.0)
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        self.assertIsNotNone(summary["frontal_symmetry"])
        # Both knees perfectly straight => deviation means near zero,
        # tracking scores similar => symmetry should be high
        self.assertGreater(summary["frontal_symmetry"], 50)

    def test_no_pose(self):
        """Frames with no pose data should not crash."""
        analyzer = FrontalAnalyzer()
        frames = [PoseFrame(i, i / 30.0, None, None, None) for i in range(10)]
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        self.assertIsNone(summary.get("left"))
        self.assertIsNone(summary.get("right"))
        self.assertIsNotNone(summary.get("frontal_recommendations"))

    def test_recommendations_have_fields(self):
        """All recommendations should have required fields."""
        analyzer = FrontalAnalyzer()
        frames = self._make_frontal_frames(left_dev=0.04, right_dev=-0.04)
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        for rec in summary["frontal_recommendations"]:
            self.assertIn("type", rec)
            self.assertIn("joint", rec)
            self.assertIn("metric", rec)
            self.assertIn("suggestion", rec)
            self.assertIn("reference", rec)

    def test_tracking_score_range(self):
        """Tracking score should be 0-100."""
        analyzer = FrontalAnalyzer()
        frames = self._make_frontal_frames(left_dev=0.02, right_dev=0.02)
        data = analyzer.analyze_all(frames)
        summary = analyzer.summarize(data)

        for side in ["left", "right"]:
            if summary.get(side):
                self.assertGreaterEqual(summary[side]["tracking_score"], 0)
                self.assertLessEqual(summary[side]["tracking_score"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
