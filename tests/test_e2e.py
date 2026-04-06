#!/usr/bin/env python3
"""
End-to-end test using a synthetic video with drawn stick figures.
Creates a test video, runs the full pipeline, and verifies outputs.

Since MediaPipe isn't available, this test exercises:
1. Video creation with OpenCV
2. Full pipeline flow (with OpenCV DNN fallback — no model = no detections)
3. Report generation with mock data
4. Annotated video writing
"""

import sys
import os
import tempfile
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_engine import PoseFrame, PoseEngine, LANDMARK_INDICES
from core.angle_calculator import AngleCalculator, JointAngles
from core.motion_analysis import MotionAnalyzer
from core.video_annotator import VideoAnnotator, generate_angle_chart
from reports.report_generator import generate_report


def create_synthetic_video(path: str, fps: int = 30, duration: float = 3.0,
                           width: int = 640, height: int = 480) -> str:
    """Create a test video with animated stick figure cyclist."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = int(fps * duration)

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)  # dark background

        t = i / fps
        phase = 2 * np.pi * t * 1.5  # 1.5 Hz = 90 RPM

        # Draw a simple stick figure cyclist
        # Body center
        cx, cy = 320, 200

        # Torso (shoulder to hip)
        shoulder = (cx - 20, cy - 40)
        hip = (cx, cy + 30)
        cv2.line(frame, shoulder, hip, (0, 200, 100), 2)

        # Head
        cv2.circle(frame, (shoulder[0], shoulder[1] - 20), 12, (0, 200, 100), 2)

        # Pedal motion (circular)
        pedal_cx, pedal_cy = cx + 30, cy + 80
        r = 40
        # Right leg
        knee_r = (pedal_cx + int(r * 0.5 * np.cos(phase)),
                  pedal_cy + int(r * 0.5 * np.sin(phase)))
        ankle_r = (pedal_cx + int(r * np.cos(phase)),
                   pedal_cy + int(r * np.sin(phase)))
        cv2.line(frame, hip, knee_r, (100, 180, 255), 2)
        cv2.line(frame, knee_r, ankle_r, (100, 180, 255), 2)
        cv2.circle(frame, ankle_r, 4, (255, 255, 0), -1)

        # Left leg (180° offset)
        knee_l = (pedal_cx + int(r * 0.5 * np.cos(phase + np.pi)),
                  pedal_cy + int(r * 0.5 * np.sin(phase + np.pi)))
        ankle_l = (pedal_cx + int(r * np.cos(phase + np.pi)),
                   pedal_cy + int(r * np.sin(phase + np.pi)))
        cv2.line(frame, hip, knee_l, (255, 150, 100), 2)
        cv2.line(frame, knee_l, ankle_l, (255, 150, 100), 2)
        cv2.circle(frame, ankle_l, 4, (255, 255, 0), -1)

        # Arms
        wrist = (cx - 80, cy - 20)
        cv2.line(frame, shoulder, wrist, (0, 200, 100), 2)

        # Bike frame
        cv2.line(frame, hip, (pedal_cx, pedal_cy), (100, 100, 100), 1)
        cv2.line(frame, wrist, (cx - 80, cy + 60), (100, 100, 100), 1)

        # Wheel
        cv2.circle(frame, (cx - 80, cy + 90), 50, (80, 80, 80), 1)
        cv2.circle(frame, (cx + 80, cy + 90), 50, (80, 80, 80), 1)

        # Label
        cv2.putText(frame, f"Synthetic Test | Frame {i}", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        writer.write(frame)

    writer.release()
    return path


def create_synthetic_pose_frames(fps=30, duration=3.0):
    """Create synthetic PoseFrames with realistic cycling motion."""
    frames = []
    total = int(fps * duration)

    for i in range(total):
        t = i / fps
        phase = 2 * np.pi * t * 1.5  # 90 RPM

        landmarks = np.full((33, 3), 0.5)
        visibility = np.ones(33) * 0.9

        # Simulate cycling landmarks (normalized 0-1 coords)
        # Shoulder
        landmarks[LANDMARK_INDICES["left_shoulder"]] = [0.45, 0.25, 0]
        landmarks[LANDMARK_INDICES["right_shoulder"]] = [0.48, 0.25, 0]

        # Hip
        landmarks[LANDMARK_INDICES["left_hip"]] = [0.50, 0.45, 0]
        landmarks[LANDMARK_INDICES["right_hip"]] = [0.52, 0.45, 0]

        # Knees (oscillating with pedal stroke)
        lk_y = 0.55 + 0.08 * np.sin(phase)
        rk_y = 0.55 + 0.08 * np.sin(phase + np.pi)
        landmarks[LANDMARK_INDICES["left_knee"]] = [0.52, lk_y, 0]
        landmarks[LANDMARK_INDICES["right_knee"]] = [0.54, rk_y, 0]

        # Ankles (circular pedal motion)
        la_x = 0.55 + 0.06 * np.cos(phase)
        la_y = 0.70 + 0.06 * np.sin(phase)
        ra_x = 0.55 + 0.06 * np.cos(phase + np.pi)
        ra_y = 0.70 + 0.06 * np.sin(phase + np.pi)
        landmarks[LANDMARK_INDICES["left_ankle"]] = [la_x, la_y, 0]
        landmarks[LANDMARK_INDICES["right_ankle"]] = [ra_x, ra_y, 0]

        # Foot indices
        landmarks[LANDMARK_INDICES["left_foot_index"]] = [la_x + 0.03, la_y, 0]
        landmarks[LANDMARK_INDICES["right_foot_index"]] = [ra_x + 0.03, ra_y, 0]

        # Simple frame image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 40)

        frames.append(PoseFrame(
            frame_index=i,
            timestamp_sec=t,
            landmarks=landmarks,
            visibility=visibility,
            raw_frame=img,
        ))

    return frames


def test_e2e():
    """Run the full pipeline with synthetic data."""
    print("\n== End-to-End Test ==\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic video
        print("1. Creating synthetic video...")
        video_path = os.path.join(tmpdir, "test_ride.mp4")
        create_synthetic_video(video_path)
        assert os.path.exists(video_path), "Video not created"
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Cannot open video"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"   Created: {frame_count} frames")

        # 2. Create synthetic pose data (bypass actual pose detection)
        print("2. Generating synthetic pose data...")
        pose_frames = create_synthetic_pose_frames()
        print(f"   Generated {len(pose_frames)} pose frames")

        # 3. Calculate angles
        print("3. Calculating joint angles...")
        calc = AngleCalculator()
        angle_list = calc.calculate_all(pose_frames)
        angle_summary = calc.summarize(angle_list)

        # Verify angle calculation worked
        lk = angle_summary.get("left_knee")
        assert lk is not None, "Left knee angles not calculated"
        print(f"   Left knee: min={lk['min']:.1f}, max={lk['max']:.1f}, range={lk['range']:.1f}")

        rk = angle_summary.get("right_knee")
        assert rk is not None, "Right knee angles not calculated"
        print(f"   Right knee: min={rk['min']:.1f}, max={rk['max']:.1f}, range={rk['range']:.1f}")

        trunk = angle_summary.get("trunk_angle")
        if trunk:
            print(f"   Trunk: mean={trunk['mean']:.1f}")

        # 4. Motion analysis
        print("4. Analyzing motion quality...")
        analyzer = MotionAnalyzer(fps=30.0)
        motion_metrics = analyzer.analyze(angle_list, angle_summary)

        score = motion_metrics.get("overall_motion_score")
        cadence = motion_metrics.get("estimated_cadence_rpm")
        sym = motion_metrics.get("knee_symmetry")
        trunk_stab = motion_metrics.get("trunk_stability_score")

        print(f"   Overall score:    {score}")
        print(f"   Cadence:          {cadence} RPM")
        print(f"   Knee symmetry:    {sym}")
        print(f"   Trunk stability:  {trunk_stab}")

        assert score is not None, "Overall score not computed"
        assert isinstance(motion_metrics["recommendations"], list), "No recommendations"
        print(f"   Recommendations:  {len(motion_metrics['recommendations'])}")
        for r in motion_metrics["recommendations"]:
            print(f"     [{r['type']}] {r['joint']}: {r['suggestion']}")

        # 5. Generate angle chart
        print("5. Generating angle chart...")
        chart_path = os.path.join(tmpdir, "angles.png")
        generate_angle_chart(angle_summary, chart_path)
        assert os.path.exists(chart_path), "Chart not created"
        chart_size = os.path.getsize(chart_path)
        print(f"   Chart: {chart_size / 1024:.1f} KB")

        # 6. Annotate video
        print("6. Rendering annotated video...")
        annotated_path = os.path.join(tmpdir, "annotated.mp4")
        with VideoAnnotator(30, 640, 480, annotated_path) as annotator:
            for pf, ang in zip(pose_frames, angle_list):
                annotator.write_frame(pf, ang, motion_metrics)
        assert os.path.exists(annotated_path), "Annotated video not created"
        ann_size = os.path.getsize(annotated_path)
        print(f"   Annotated video: {ann_size / 1024:.1f} KB")

        # 7. Generate HTML report
        print("7. Generating HTML report...")
        video_metadata = {
            "fps": 30, "total_frames": len(pose_frames),
            "width": 640, "height": 480, "duration_sec": 3.0,
            "video_path": video_path,
        }
        report_path = os.path.join(tmpdir, "report.html")
        generate_report(
            angle_summary=angle_summary,
            motion_metrics=motion_metrics,
            video_metadata=video_metadata,
            chart_path=chart_path,
            output_path=report_path,
            video_name="test_ride",
        )
        assert os.path.exists(report_path), "Report not created"
        report_size = os.path.getsize(report_path)
        print(f"   Report: {report_size / 1024:.1f} KB")

        # Verify report content
        with open(report_path) as f:
            html = f.read()
        assert "Bike Fit Analysis Report" in html, "Report title missing"
        assert "Motion Score" in html or "Smoothness" in html, "Metrics missing from report"

        print("\n✅  All E2E tests passed!\n")

        # Return summary
        return {
            "video_frames": frame_count,
            "pose_frames": len(pose_frames),
            "overall_score": score,
            "cadence": cadence,
            "knee_symmetry": sym,
            "recommendations": len(motion_metrics["recommendations"]),
            "chart_kb": round(chart_size / 1024, 1),
            "video_kb": round(ann_size / 1024, 1),
            "report_kb": round(report_size / 1024, 1),
        }


if __name__ == "__main__":
    results = test_e2e()
    print("Summary:", json.dumps(results, indent=2))
