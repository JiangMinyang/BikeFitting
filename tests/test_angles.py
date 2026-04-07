#!/usr/bin/env python3
"""
Precise geometric tests for every joint angle computed by AngleCalculator.

Each test constructs landmarks at known positions and asserts the computed
angle matches an analytically expected value.  All coordinates use the
normalised image coordinate system:
    x : 0 (left edge) → 1 (right edge)
    y : 0 (top edge)  → 1 (bottom edge)   ← y increases DOWNWARD
    Rider faces RIGHT in all side-view fixtures.

Angles are asserted within ±2° to tolerate floating-point rounding.
"""

import sys
import os
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_engine import PoseFrame, LANDMARK_INDICES
from core.angle_calculator import (
    AngleCalculator, JointAngles,
    _three_point_angle, _angle_between_vectors,
)

DELTA = 2.0   # acceptable tolerance in degrees


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_frame(**pts):
    """Build a PoseFrame with specified landmark positions (all others at 0.5)."""
    lm  = np.full((33, 3), 0.5, dtype=np.float32)
    vis = np.ones(33, dtype=np.float32) * 0.95
    for name, (x, y) in pts.items():
        lm[LANDMARK_INDICES[name]] = [x, y, 0.0]
    return PoseFrame(0, 0.0, lm, vis, None)


def _calc(frame, **kwargs):
    return AngleCalculator().calculate_frame(frame, **kwargs)


# ─── Primitives ───────────────────────────────────────────────────────────────

class TestPrimitives(unittest.TestCase):
    """Low-level geometry functions."""

    def test_right_angle(self):
        # B at origin, A to the right, C upward → 90°
        self.assertAlmostEqual(
            _three_point_angle(np.array([1,0.]), np.array([0,0.]), np.array([0,1.])),
            90.0, delta=DELTA)

    def test_straight_line_180(self):
        # A-B-C all collinear → 180°
        self.assertAlmostEqual(
            _three_point_angle(np.array([0,0.]), np.array([0.5,0.]), np.array([1,0.])),
            180.0, delta=DELTA)

    def test_equilateral_60(self):
        # Equilateral triangle → 60°
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.5, math.sqrt(3)/2])
        self.assertAlmostEqual(_three_point_angle(a, b, c), 60.0, delta=DELTA)

    def test_vectors_parallel_0(self):
        self.assertAlmostEqual(
            _angle_between_vectors(np.array([1,0.]), np.array([3,0.])), 0.0, delta=DELTA)

    def test_vectors_antiparallel_180(self):
        self.assertAlmostEqual(
            _angle_between_vectors(np.array([1,0.]), np.array([-2,0.])), 180.0, delta=DELTA)

    def test_vectors_perpendicular_90(self):
        self.assertAlmostEqual(
            _angle_between_vectors(np.array([1,0.]), np.array([0,1.])), 90.0, delta=DELTA)

    def test_known_45(self):
        self.assertAlmostEqual(
            _angle_between_vectors(np.array([1,0.]), np.array([1,1.])), 45.0, delta=DELTA)


# ─── Knee angle ───────────────────────────────────────────────────────────────
# Formula: _three_point_angle(hip, knee, ankle)  — angle AT the knee.
# 180° = full extension.  Lower = more flexion.

class TestKneeAngle(unittest.TestCase):

    def _knee(self, hip, knee, ankle, near_side="right"):
        frame = _make_frame(right_hip=hip, right_knee=knee, right_ankle=ankle)
        return _calc(frame, near_side=near_side).right_knee

    def test_full_extension_180(self):
        """Perfectly straight leg (hip-knee-ankle all vertical) → 180°."""
        angle = self._knee((0.5, 0.2), (0.5, 0.5), (0.5, 0.8))
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, delta=DELTA)

    def test_right_angle_90(self):
        """Hip directly above knee, ankle directly to the right → 90°."""
        # knee at (0.5,0.5), hip directly above, ankle to the right
        angle = self._knee((0.5, 0.2), (0.5, 0.5), (0.8, 0.5))
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_45_degree(self):
        """
        Hip at 22.5° from vertical above knee, ankle at 22.5° from vertical below.
        Both on the same side so the opening angle is 180 - 45 = 135°.

        Analytically:
          knee = (0.5, 0.5)
          hip  = knee + 0.3*(-sin22.5°, -cos22.5°) = (0.5-0.115, 0.5-0.277) = (0.385, 0.223)
          ankle= knee + 0.3*( sin22.5°, +cos22.5°) = (0.5+0.115, 0.5+0.277) = (0.615, 0.777)
        The vectors hip→knee and ankle→knee are mirror images about the vertical
        so the angle between BA(up-right) and BC(down-right) is not 45°.

        Simpler construction: put hip straight up and ankle at 45° from vertical.
          hip  = (0.5, 0.2)   — directly above knee
          knee = (0.5, 0.5)
          ankle= (0.5+0.3*sin45°, 0.5+0.3*cos45°) = (0.712, 0.712)
        BA = (0, -0.3) — straight up
        BC = (0.212, 0.212) — 45° from vertical (down-right)
        angle(BA,BC) = arccos(dot / (|BA||BC|))
                     = arccos((-0.3*0.212 + 0.3*0) ... wait
        BA = hip - knee = (0, -0.3)
        BC = ankle - knee = (0.212, 0.212)
        dot = 0*0.212 + (-0.3)*0.212 = -0.0636
        |BA|=0.3, |BC|=0.3
        cos = -0.0636/0.09 = -0.707  → angle = 135°
        """
        hip   = (0.5, 0.2)
        knee  = (0.5, 0.5)
        ankle = (0.5 + 0.3*math.sin(math.radians(45)),
                 0.5 + 0.3*math.cos(math.radians(45)))  # (0.712, 0.712)
        angle = self._knee(hip, knee, ankle)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 135.0, delta=DELTA)

    def test_bdc_typical_range(self):
        """
        Typical BDC cycling position — leg nearly extended.
        With hip slightly forward and ankle slightly forward of knee,
        expect ~150-170°.
        """
        # hip forward-above, knee mid, ankle forward-below (nearly straight)
        angle = self._knee((0.42, 0.30), (0.46, 0.56), (0.50, 0.80))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 140, msg="BDC should be near extension (>140°)")
        self.assertLess(angle, 180, msg="BDC should not be hyper-extended")

    def test_tdc_typical_range(self):
        """
        Typical TDC cycling position — knee flexed as foot comes up and forward.
        Expect ~60-90°.
        """
        # At TDC: knee pulled up and forward, ankle high and forward
        angle = self._knee((0.42, 0.38), (0.54, 0.52), (0.62, 0.42))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 50,  msg="TDC should have meaningful flexion")
        self.assertLess(angle, 100, msg="TDC should not be at full extension")

    def test_near_side_only_computes_near(self):
        """near_side='right' should compute right knee, not left."""
        frame = _make_frame(
            right_hip=(0.5,0.2), right_knee=(0.5,0.5), right_ankle=(0.5,0.8),
            left_hip=(0.5,0.2),  left_knee=(0.5,0.5),  left_ankle=(0.5,0.8),
        )
        angles = _calc(frame, near_side="right")
        self.assertIsNotNone(angles.right_knee)
        self.assertIsNone(angles.left_knee)


# ─── Hip angle ────────────────────────────────────────────────────────────────
# Formula: _three_point_angle(shoulder, hip, knee)  — angle AT the hip.
# 180° = trunk and thigh perfectly aligned (straight through hip).

class TestHipAngle(unittest.TestCase):

    def _hip(self, shoulder, hip, knee, near_side="right"):
        frame = _make_frame(right_shoulder=shoulder, right_hip=hip, right_knee=knee)
        return _calc(frame, near_side=near_side).right_hip

    def test_straight_180(self):
        """Shoulder-hip-knee all vertical → 180°."""
        angle = self._hip((0.5, 0.1), (0.5, 0.4), (0.5, 0.7))
        self.assertAlmostEqual(angle, 180.0, delta=DELTA)

    def test_right_angle_90(self):
        """
        Shoulder directly above hip, knee directly to the right → 90°.
        BA = shoulder - hip = (0, -0.3)   [up]
        BC = knee    - hip = (0.3, 0)     [right]
        angle = 90°
        """
        angle = self._hip((0.5, 0.2), (0.5, 0.5), (0.8, 0.5))
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_cycling_bdc_open(self):
        """
        At BDC the hip is relatively open (shoulder-hip-knee obtuse).
        Expect ~100-140°.
        """
        # Rider facing right: shoulder forward-above, hip at saddle, knee below
        angle = self._hip((0.52, 0.25), (0.40, 0.45), (0.44, 0.68))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 90)
        self.assertLess(angle, 160)

    def test_cycling_tdc_closed(self):
        """
        At TDC the knee comes up and forward — hip angle closes.
        Expect ~55-90°.
        """
        angle = self._hip((0.52, 0.25), (0.40, 0.45), (0.58, 0.42))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 40)
        self.assertLess(angle, 100)


# ─── Ankle angle ─────────────────────────────────────────────────────────────
# Formula: _three_point_angle(knee, ankle, foot_index)  — angle AT the ankle.
# 90°  = shin vertical, foot horizontal.
# >90° = plantarflexion (toe pointing down relative to shin).
# <90° = dorsiflexion  (toe pointing up relative to shin).

class TestAnkleAngle(unittest.TestCase):

    def _ankle(self, knee, ankle, foot, near_side="right"):
        frame = _make_frame(right_knee=knee, right_ankle=ankle, right_foot_index=foot)
        return _calc(frame, near_side=near_side).right_ankle

    def test_neutral_90(self):
        """
        Shin vertical (knee directly above ankle), foot horizontal → 90°.
        BA = knee  - ankle = (0, -0.3)  [up]
        BC = foot  - ankle = (0.2, 0)   [right]
        angle = 90°
        """
        angle = self._ankle((0.5, 0.5), (0.5, 0.8), (0.7, 0.8))
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_plantarflexion_greater_90(self):
        """
        Toe pointing down relative to shin → ankle angle > 90°.
        Shin vertical, foot angled downward-forward.
        BA = (0, -0.3) [up], BC = (0.2, 0.05) [forward and slightly down]
        dot = 0*0.2 + (-0.3)*0.05 = -0.015
        cos = -0.015 / (0.3 * 0.206) = -0.243  → ~104°
        """
        angle = self._ankle((0.5, 0.5), (0.5, 0.8), (0.7, 0.85))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 90.0,
            msg="Toe pointing down → plantarflexion → >90°")

    def test_dorsiflexion_less_90(self):
        """
        Toe pointing up relative to shin → ankle angle < 90°.
        BA = (0, -0.3) [up], BC = (0.2, -0.06) [forward and slightly up]
        dot = (-0.3)*(-0.06) = 0.018
        cos = 0.018 / (0.3 * 0.209) = 0.287  → ~73°
        """
        angle = self._ankle((0.5, 0.5), (0.5, 0.8), (0.7, 0.74))
        self.assertIsNotNone(angle)
        self.assertLess(angle, 90.0,
            msg="Toe pointing up → dorsiflexion → <90°")

    def test_straight_leg_neutral_foot(self):
        """Realistic mid-stroke cycling: expect angle in 75-115° range."""
        # Shin at ~10° forward of vertical, foot roughly horizontal
        angle = self._ankle((0.48, 0.52), (0.50, 0.80), (0.68, 0.82))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 75)
        self.assertLess(angle, 115)


# ─── Trunk angle ─────────────────────────────────────────────────────────────
# Formula: _angle_between_vectors(shoulder - hip, [0, -1])
# [0, -1] is the UPWARD direction in image coords (y increases downward).
# 0°  = torso perfectly upright (shoulder directly above hip).
# 45° = trunk leaning 45° forward (rider faces right → shoulder forward of hip).
# 90° = torso horizontal (shoulder level with hip).

class TestTrunkAngle(unittest.TestCase):

    def _trunk(self, shoulder, hip, near_side="right"):
        frame = _make_frame(right_shoulder=shoulder, right_hip=hip)
        return _calc(frame, near_side=near_side).trunk_angle

    def test_upright_0_degrees(self):
        """Shoulder directly above hip → 0° lean."""
        angle = self._trunk((0.5, 0.2), (0.5, 0.6))
        self.assertAlmostEqual(angle, 0.0, delta=DELTA)

    def test_45_degree_forward_lean(self):
        """
        Shoulder 45° forward of vertical above hip.
        shoulder - hip = (0.283, -0.283)  (equal x and y components)
        angle with [0,-1] = arccos(0.283 / 0.4) = arccos(0.707) = 45°
        """
        # hip at (0.3, 0.6), shoulder at (0.3 + 0.4*sin45, 0.6 - 0.4*cos45)
        #                              = (0.583, 0.317)
        hip      = (0.3,   0.6)
        shoulder = (0.583, 0.317)
        angle = self._trunk(shoulder, hip)
        self.assertAlmostEqual(angle, 45.0, delta=DELTA)

    def test_30_degree_forward_lean(self):
        """
        shoulder - hip = (0.4*sin30, -0.4*cos30) = (0.2, -0.346)
        angle with [0,-1] = arccos(0.346/0.4) = arccos(0.866) = 30°
        """
        hip      = (0.3, 0.6)
        shoulder = (0.3 + 0.4*math.sin(math.radians(30)),
                    0.6 - 0.4*math.cos(math.radians(30)))
        angle = self._trunk(shoulder, hip)
        self.assertAlmostEqual(angle, 30.0, delta=DELTA)

    def test_90_degree_horizontal(self):
        """Shoulder level with hip, directly to the right → 90°."""
        angle = self._trunk((0.8, 0.5), (0.4, 0.5))
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_both_sides_averaged(self):
        """When both sides are present, trunk angle uses their midpoints."""
        # Both sides identical → same as single-side result
        lm  = np.full((33, 3), 0.5, dtype=np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.95
        # Left and right shoulders both at y=0.25, hips at y=0.55
        for name, (x, y) in [
            ("left_shoulder",  (0.45, 0.25)), ("right_shoulder", (0.55, 0.25)),
            ("left_hip",       (0.45, 0.55)), ("right_hip",      (0.55, 0.55)),
        ]:
            lm[LANDMARK_INDICES[name]] = [x, y, 0.0]
        frame = PoseFrame(0, 0.0, lm, vis, None)
        angles = AngleCalculator().calculate_frame(frame)
        # midpoint shoulder=(0.5,0.25), hip=(0.5,0.55) → vector=(0,-0.3) → 0°
        self.assertAlmostEqual(angles.trunk_angle, 0.0, delta=DELTA)

    def test_typical_road_position(self):
        """Typical road cycling: 30-50° forward lean."""
        angle = self._trunk((0.52, 0.25), (0.38, 0.45))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 20)
        self.assertLess(angle, 70)


# ─── Elbow angle ─────────────────────────────────────────────────────────────
# Formula: _three_point_angle(shoulder, elbow, wrist)  — angle AT the elbow.
# 180° = arm fully straight.
# <180° = arm bent.
# Ideal road cycling: 150-165°.

class TestElbowAngle(unittest.TestCase):

    def _elbow(self, shoulder, elbow, wrist, near_side="right"):
        frame = _make_frame(right_shoulder=shoulder, right_elbow=elbow, right_wrist=wrist)
        return _calc(frame, near_side=near_side).right_elbow

    def test_straight_arm_180(self):
        """Shoulder-elbow-wrist collinear → 180°."""
        angle = self._elbow((0.3, 0.4), (0.5, 0.4), (0.7, 0.4))
        self.assertAlmostEqual(angle, 180.0, delta=DELTA)

    def test_right_angle_90(self):
        """
        Shoulder to the left of elbow, wrist below elbow → 90°.
        BA = shoulder - elbow = (-0.2, 0)  [left]
        BC = wrist   - elbow = (0,  0.2)  [down]
        angle = 90°
        """
        angle = self._elbow((0.3, 0.5), (0.5, 0.5), (0.5, 0.7))
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_ideal_road_150_165(self):
        """
        Realistic handlebar reach: soft elbow bend, 150-165°.
        Shoulder at (0.35, 0.28), elbow at (0.50, 0.38), wrist at (0.68, 0.30).
        BA = (-0.15, -0.10), BC = (0.18, -0.08)
        dot = (-0.15)(0.18) + (-0.10)(-0.08) = -0.027 + 0.008 = -0.019
        |BA| = 0.180, |BC| = 0.197
        cos = -0.019 / 0.0355 = -0.535  → ~122°   ← too bent, adjust
        """
        # Straighter arm: wrist further forward and less drop
        #   shoulder (0.35, 0.28), elbow (0.50, 0.33), wrist (0.68, 0.28)
        #   BA = (-0.15, -0.05), BC = (0.18, -0.05)
        #   dot = -0.027 + 0.0025 = -0.0245
        #   |BA|=0.1581, |BC|=0.1868
        #   cos = -0.0245/0.02954 = -0.830  → ~156°
        angle = self._elbow((0.35, 0.28), (0.50, 0.33), (0.68, 0.28))
        self.assertIsNotNone(angle)
        self.assertGreater(angle, 140, msg="Road elbow should be only slightly bent")
        self.assertLess(angle, 175, msg="Road elbow should not be fully straight")

    def test_excessive_bend_130(self):
        """Very bent elbow → bars too low or reach too long → <140°."""
        # BA=(-0.2,0), BC=(0,-0.2) → 90°, but let's target ~130°
        #   shoulder (0.35,0.3), elbow (0.55,0.45), wrist (0.40,0.55)
        #   BA=(-0.2,-0.15), BC=(-0.15,0.10)
        #   dot=0.03-0.015=0.015
        #   |BA|=0.25, |BC|=0.18
        #   cos=0.015/0.045=0.333 → 71°  too much
        # easier: place wrist directly below elbow
        #   shoulder(0.5,0.3), elbow(0.5,0.5), wrist(0.35,0.6)
        #   BA=(0,-0.2) [up], BC=(-0.15,0.10)
        #   dot=0+(-0.2)(0.10)=-0.02
        #   |BA|=0.2, |BC|=0.180
        #   cos=-0.02/0.036=-0.556 → 124°
        angle = self._elbow((0.5, 0.3), (0.5, 0.5), (0.35, 0.6))
        self.assertIsNotNone(angle)
        self.assertLess(angle, 140, msg="Excessive bend should be <140°")


# ─── Upper-arm angle ─────────────────────────────────────────────────────────
# Formula: _three_point_angle(hip, shoulder, elbow)  — angle AT the shoulder
#          between the hip direction and the elbow direction.
#
# This measures the opening at the shoulder between the torso line (to hip)
# and the arm line (to elbow).
#
# Geometric interpretation:
#   ~90°  = arm roughly perpendicular to the hip→shoulder line
#   <90°  = arm swings toward the hip direction (very forward reach)
#   >90°  = arm swings away from hip direction (arm more upright / bars close)
#
# NOTE: The expected value is sensitive to where the hip sits relative to the
# shoulder. When the hip is BEHIND and BELOW the shoulder (normal upright/road
# position), the angle is typically 90-120°. When the hip is forward of the
# shoulder (very aggressive aero or bad detection), the angle drops below 90°.

class TestUpperArmAngle(unittest.TestCase):

    def _arm(self, hip, shoulder, elbow, near_side="right"):
        frame = _make_frame(right_hip=hip, right_shoulder=shoulder, right_elbow=elbow)
        return _calc(frame, near_side=near_side).right_shoulder_arm

    def test_arm_perpendicular_to_torso_90(self):
        """
        Hip directly below shoulder, elbow directly to the right of shoulder → 90°.
        At shoulder:
          BA = hip      - shoulder = (0, 0.3)   [downward]
          BC = elbow    - shoulder = (0.3, 0)   [rightward]
          angle = 90°
        """
        angle = self._arm((0.5, 0.8), (0.5, 0.5), (0.8, 0.5))
        self.assertAlmostEqual(angle, 90.0, delta=DELTA)

    def test_arm_in_line_with_torso_180(self):
        """
        Hip below shoulder, elbow above shoulder (arm raised in line with torso).
        BA = hip   - shoulder = (0, 0.3)   [down]
        BC = elbow - shoulder = (0, -0.3)  [up]
        angle = 180°
        """
        angle = self._arm((0.5, 0.8), (0.5, 0.5), (0.5, 0.2))
        self.assertAlmostEqual(angle, 180.0, delta=DELTA)

    def test_typical_road_position(self):
        """
        Realistic road position: hip behind and below shoulder, elbow forward.
        hip=(0.38,0.45), shoulder=(0.52,0.25), elbow=(0.68,0.32).

        BA = hip-shoulder = (-0.14, 0.20)
        BC = elbow-shoulder = (0.16, 0.07)
        dot = (-0.14)(0.16)+(0.20)(0.07) = -0.0224+0.014 = -0.0084
        |BA| = sqrt(0.0196+0.04) = 0.2441
        |BC| = sqrt(0.0256+0.0049) = 0.1746
        cos = -0.0084/(0.2441*0.1746) = -0.0084/0.04262 = -0.197
        angle = arccos(-0.197) ≈ 101°
        """
        angle = self._arm((0.38, 0.45), (0.52, 0.25), (0.68, 0.32))
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 101.0, delta=3.0)

    def test_hip_forward_of_shoulder_gives_acute_angle(self):
        """
        When the hip appears FORWARD of the shoulder (aggressive aero pose or
        bad landmark detection), the angle becomes acute (<90°).

        hip=(0.60,0.45), shoulder=(0.52,0.25), elbow=(0.68,0.32).
        BA = hip-shoulder = (0.08, 0.20)   ← NOTE: now points forward-and-down
        BC = elbow-shoulder = (0.16, 0.07)
        Both BA and BC point in roughly the same forward direction.
        dot = 0.08*0.16 + 0.20*0.07 = 0.0128+0.014 = 0.0268
        |BA|=0.2154, |BC|=0.1746
        cos = 0.0268/0.0376 = 0.713
        angle = arccos(0.713) ≈ 44.5°  ← acute, less than 90°

        This is the geometry that caused the user to see "135°" when the
        supplement formula (180-raw) was in use.
        """
        angle = self._arm((0.60, 0.45), (0.52, 0.25), (0.68, 0.32))
        self.assertIsNotNone(angle)
        # Angle should be acute (<90°) when hip is forward of shoulder
        self.assertLess(angle, 90.0,
            msg="Hip forward of shoulder should produce an acute angle "
                "at the shoulder vertex")
        self.assertAlmostEqual(angle, 44.5, delta=3.0)

    def test_arm_alongside_torso_pointing_hip(self):
        """
        Elbow pointing toward the hip (arm alongside body, pointing down).
        BA = hip-shoulder = (0, 0.3) [down], BC = elbow-shoulder = (0, 0.3) [down].
        Both point the same direction → angle = 0°.
        """
        angle = self._arm((0.5, 0.8), (0.5, 0.5), (0.5, 0.8))
        self.assertAlmostEqual(angle, 0.0, delta=DELTA)


# ─── Full cycling pose end-to-end ─────────────────────────────────────────────

class TestCyclingPoseEndToEnd(unittest.TestCase):
    """
    Run calculate_frame on a realistic side-view cycling pose and assert
    each angle falls in the expected physiological range for that position.

    Pose: rider facing RIGHT, near side = right, approximate BDC position.

    Landmark positions (normalised image coords):
        right_shoulder : (0.52, 0.25)  — upper body, forward-leaning
        right_elbow    : (0.64, 0.30)  — handlebars, soft bend
        right_wrist    : (0.78, 0.28)  — hands on drops
        right_hip      : (0.39, 0.44)  — saddle
        right_knee     : (0.43, 0.66)  — mid-lower leg
        right_ankle    : (0.47, 0.82)  — lower leg
        right_foot_index:(0.61, 0.84)  — ball of foot / big toe
        right_heel     : (0.44, 0.85)  — heel

    Elbow angle pre-computation:
        BA = shoulder - elbow = (0.52-0.64, 0.25-0.30) = (-0.12, -0.05)
        BC = wrist    - elbow = (0.78-0.64, 0.28-0.30) = ( 0.14, -0.02)
        dot = (-0.12)(0.14) + (-0.05)(-0.02) = -0.0168 + 0.001 = -0.0158
        cos = -0.0158 / (0.1301 * 0.1414) = -0.859  → ~149°  (soft road bend)
    """

    def setUp(self):
        self.frame = _make_frame(
            right_shoulder  = (0.52, 0.25),
            right_elbow     = (0.64, 0.30),
            right_wrist     = (0.78, 0.28),
            right_hip       = (0.39, 0.44),
            right_knee      = (0.43, 0.66),
            right_ankle     = (0.47, 0.82),
            right_foot_index= (0.61, 0.84),
            right_heel      = (0.44, 0.85),
        )
        self.angles = AngleCalculator().calculate_frame(self.frame, near_side="right")
        # Print computed angles so failures are easy to diagnose
        print("\n── BDC pose computed angles ──────────────────────────")
        for attr in ("right_knee", "right_hip", "right_ankle",
                     "right_elbow", "right_shoulder_arm", "trunk_angle"):
            print(f"  {attr:25s}: {getattr(self.angles, attr)}")

    def test_knee_near_extension(self):
        """BDC knee should be 130-175° (near but not hyper-extended)."""
        v = self.angles.right_knee
        self.assertIsNotNone(v)
        self.assertGreater(v, 130, "BDC knee should be near extension")
        self.assertLess(v, 178,    "BDC knee should not be hyper-extended")

    def test_hip_open(self):
        """BDC hip (shoulder-hip-knee) should be 90-160°."""
        v = self.angles.right_hip
        self.assertIsNotNone(v)
        self.assertGreater(v, 90,  "BDC hip should be open")
        self.assertLess(v, 165,    "BDC hip should not be fully straight")

    def test_ankle_reasonable(self):
        """Ankle angle should be in 75-115° (slight plantarflexion typical)."""
        v = self.angles.right_ankle
        self.assertIsNotNone(v, "Ankle angle should be computed when foot_index visible")
        self.assertGreater(v, 70,  "Ankle should not be extreme dorsiflexion")
        self.assertLess(v, 120,    "Ankle should not be extreme plantarflexion")

    def test_elbow_soft_bend(self):
        """Road elbow should be 140-175°."""
        v = self.angles.right_elbow
        self.assertIsNotNone(v)
        self.assertGreater(v, 140, "Road elbow should not be severely bent")
        self.assertLess(v, 175,    "Road elbow should not be locked straight")

    def test_trunk_forward_lean(self):
        """Trunk lean should reflect a forward road position (~20-60°)."""
        v = self.angles.trunk_angle
        self.assertIsNotNone(v)
        self.assertGreater(v, 15, "Trunk should be leaning forward")
        self.assertLess(v, 65,   "Trunk lean should not be extreme")

    def test_upper_arm_not_none(self):
        """Upper arm angle should be computed when hip, shoulder, elbow all visible."""
        self.assertIsNotNone(self.angles.right_shoulder_arm)

    def test_left_angles_none_when_near_side_right(self):
        """Left-side angles should all be None when near_side='right'."""
        for attr in ("left_knee", "left_hip", "left_ankle",
                     "left_elbow", "left_shoulder_arm"):
            self.assertIsNone(getattr(self.angles, attr),
                              msg=f"{attr} should be None for far side")


# ─── Summarize ────────────────────────────────────────────────────────────────

class TestSummarize(unittest.TestCase):
    """Verify that summarize() produces correct statistics."""

    def _make_angle_list(self, values, attr="left_knee"):
        return [JointAngles(i, i/30.0, **{attr: v}) for i, v in enumerate(values)]

    def test_mean(self):
        angles = self._make_angle_list([100.0, 120.0, 140.0])
        summary = AngleCalculator().summarize(angles)
        self.assertAlmostEqual(summary["left_knee"]["mean"], 120.0, delta=0.5)

    def test_range(self):
        angles = self._make_angle_list([90.0, 150.0])
        summary = AngleCalculator().summarize(angles)
        self.assertAlmostEqual(summary["left_knee"]["range"], 60.0, delta=0.5)

    def test_none_frames_skipped(self):
        """Frames where the angle is None should not crash summarize."""
        angles = [
            JointAngles(0, 0.0, left_knee=None),
            JointAngles(1, 1.0, left_knee=130.0),
            JointAngles(2, 2.0, left_knee=None),
            JointAngles(3, 3.0, left_knee=140.0),
        ]
        summary = AngleCalculator().summarize(angles)
        lk = summary.get("left_knee")
        self.assertIsNotNone(lk)
        self.assertAlmostEqual(lk["mean"], 135.0, delta=1.0)

    def test_all_none_key_is_none(self):
        """If an angle is None in every frame, its summary value should be None."""
        angles = [JointAngles(i, float(i)) for i in range(5)]  # all angles None
        summary = AngleCalculator().summarize(angles)
        # summarize() includes the key but with None value when no data available
        self.assertIn("left_knee", summary)
        self.assertIsNone(summary["left_knee"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
