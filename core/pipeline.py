"""
Main analysis pipeline — supports dual-video input.

Side view:  Joint angle analysis (knee, hip, ankle, trunk) + pedaling smoothness
Front view: Knee dynamics (valgus/varus, tracking, frontal alignment)

Either view can be provided independently, or both for a complete assessment.
"""

import os
from datetime import datetime
from typing import Callable, Optional

import cv2

from .pose_engine import PoseEngine, LANDMARK_INDICES
from .angle_calculator import AngleCalculator
from .motion_analysis import MotionAnalyzer
from .frontal_analyzer import FrontalAnalyzer
from .video_annotator import VideoAnnotator, generate_angle_chart, annotate_single_frame


def _detect_near_side(frames) -> str:
    """Determine which side (left/right) faces the camera in a side-view video.

    Compares average MediaPipe visibility scores of key landmarks across all
    frames. The near side (closer to the camera) will have consistently higher
    visibility scores than the far side, which is partially occluded.

    Returns 'left' or 'right'.
    """
    left_names  = ["left_hip",  "left_knee",  "left_ankle",  "left_shoulder"]
    right_names = ["right_hip", "right_knee", "right_ankle", "right_shoulder"]

    left_total = 0.0
    right_total = 0.0

    for frame in frames:
        if not frame.has_pose or frame.visibility is None:
            continue
        for name in left_names:
            idx = LANDMARK_INDICES.get(name)
            if idx is not None and idx < len(frame.visibility):
                left_total += float(frame.visibility[idx])
        for name in right_names:
            idx = LANDMARK_INDICES.get(name)
            if idx is not None and idx < len(frame.visibility):
                right_total += float(frame.visibility[idx])

    return "left" if left_total >= right_total else "right"


class AnalysisPipeline:
    """
    High-level pipeline for bike fit video analysis.

    Supports:
    - Side view only:  run(side_video="ride_side.mp4")
    - Front view only: run(front_video="ride_front.mp4")
    - Both views:      run(side_video="side.mp4", front_video="front.mp4")

    Legacy single-video mode still works via positional arg.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        video_path: str = None,
        side_video: str = None,
        front_video: str = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Run analysis pipeline on one or both video views.

        Args:
            video_path: Legacy single-video path (treated as side view)
            side_video: Path to side-view video for joint angle analysis
            front_video: Path to front-view video for knee dynamics analysis
            progress_callback: Optional fn(current, total, stage_name)

        Returns:
            dict with all analysis results and output file paths.
        """
        # Handle legacy single-video mode
        if video_path and not side_video:
            side_video = video_path

        if not side_video and not front_video:
            raise ValueError("At least one video (side_video or front_video) must be provided.")

        def _progress(current: int, total: int, stage: str = ""):
            if progress_callback:
                progress_callback(current, total, stage)

        # Determine naming — each session gets its own subdirectory
        base_name = "analysis"
        if side_video:
            base_name = os.path.splitext(os.path.basename(side_video))[0]
        elif front_video:
            base_name = os.path.splitext(os.path.basename(front_video))[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{base_name}_{timestamp}"

        # Create a per-session subdirectory so all output files are grouped together
        session_dir = os.path.join(self.output_dir, prefix)
        os.makedirs(session_dir, exist_ok=True)

        results = {
            "has_side_view": side_video is not None,
            "has_front_view": front_video is not None,
        }

        total_steps = 0
        if side_video:
            total_steps += 60  # pose + angles + motion + chart + annotate
        if front_video:
            total_steps += 30  # pose + frontal analysis + annotate
        total_steps += 10  # report
        step_offset = 0

        # ════════════════════════════════════════════════════════════════════
        # SIDE VIEW: Joint angle + pedaling smoothness analysis
        # ════════════════════════════════════════════════════════════════════
        if side_video:
            _progress(0, 100, "Processing side view...")

            # Stage 1: Pose estimation (side)
            engine = PoseEngine(backend="auto")
            side_frames, side_meta = engine.process_video(
                side_video,
                progress_callback=lambda f, t: _progress(
                    int(f / max(t, 1) * 25), 100,
                    f"Side view: estimating pose... frame {f}/{t}"
                ),
            )
            results["side_video_metadata"] = side_meta
            results["video_metadata"] = side_meta  # backward compat

            # Stage 2: Angle calculation
            # Detect which side faces the camera so we can ignore the far-side
            # landmarks, which are partially occluded and unreliable in side view.
            _progress(25, 100, "Side view: calculating joint angles...")
            near_side = _detect_near_side(side_frames)
            calc = AngleCalculator()
            angle_list = calc.calculate_all(side_frames, near_side=near_side)
            angle_summary = calc.summarize(angle_list)
            results["near_side"] = near_side
            results["angle_summary"] = angle_summary

            # Pedal phase detection
            for side_name in ["left", "right"]:
                phase_data = calc.detect_pedal_phases(angle_list, side=side_name)
                results[f"{side_name}_pedal_phases"] = phase_data

            # Stage 3: Motion analysis
            _progress(35, 100, "Side view: analyzing pedaling quality...")
            fps = side_meta["fps"]
            analyzer = MotionAnalyzer(fps=fps)
            motion_metrics = analyzer.analyze(angle_list, angle_summary)
            results["motion_metrics"] = motion_metrics

            # Stage 4: Angle chart
            _progress(45, 100, "Side view: generating charts...")
            chart_path = os.path.join(session_dir, f"{prefix}_angles.png")
            generate_angle_chart(angle_summary, chart_path)
            results["chart_png"] = chart_path

            # Stage 4b: BDC / TDC keyframes
            _progress(47, 100, "Side view: extracting BDC/TDC keyframes...")
            knee_key = f"{near_side}_knee"
            knee_vals = [
                (i, getattr(a, knee_key))
                for i, a in enumerate(angle_list)
                if getattr(a, knee_key) is not None
            ]
            if len(knee_vals) >= 4:
                bdc_frame_i = max(knee_vals, key=lambda x: x[1])[0]  # max extension
                tdc_frame_i = min(knee_vals, key=lambda x: x[1])[0]  # max flexion

                for frame_i, label, out_key in [
                    (bdc_frame_i, "BDC - Max Extension", "bdc_frame_png"),
                    (tdc_frame_i, "TDC - Max Flexion",   "tdc_frame_png"),
                ]:
                    cap_kf = cv2.VideoCapture(side_video)
                    cap_kf.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                    ret_kf, raw_kf = cap_kf.read()
                    cap_kf.release()
                    if ret_kf:
                        pf_kf  = side_frames[frame_i]
                        ang_kf = angle_list[frame_i]
                        out_png = os.path.join(
                            session_dir, f"{prefix}_{out_key.replace('_png', '')}.png"
                        )
                        pf_kf.raw_frame = raw_kf
                        annotate_single_frame(raw_kf, pf_kf, ang_kf,
                                              motion_metrics, label, out_png,
                                              near_side=near_side)
                        pf_kf.raw_frame = None
                        results[out_key] = out_png

            # Stage 5: Annotated side video
            # Re-read the video frame by frame to avoid buffering all raw frames in RAM.
            _progress(50, 100, "Side view: rendering annotated video...")
            side_annotated = os.path.join(session_dir, f"{prefix}_side_annotated.mp4")
            width, height = side_meta["width"], side_meta["height"]
            cap_side = cv2.VideoCapture(side_video)
            with VideoAnnotator(fps, width, height, side_annotated) as annotator:
                for i, (pf, ang) in enumerate(zip(side_frames, angle_list)):
                    ret, raw = cap_side.read()
                    if not ret:
                        break
                    pf.raw_frame = raw
                    annotator.write_frame(pf, ang, motion_metrics, near_side=near_side)
                    pf.raw_frame = None  # Free immediately
                    if i % 30 == 0:
                        pct = 50 + int((i / max(len(side_frames), 1)) * 15)
                        _progress(pct, 100, f"Side: rendering frame {i}/{len(side_frames)}")
            cap_side.release()
            results["annotated_video"] = side_annotated
            results["side_annotated_video"] = side_annotated
            step_offset = 65

        # ════════════════════════════════════════════════════════════════════
        # FRONT VIEW: Knee dynamics / frontal plane analysis
        # ════════════════════════════════════════════════════════════════════
        if front_video:
            _progress(step_offset, 100, "Processing front view...")

            # Stage 1: Pose estimation (front)
            engine = PoseEngine(backend="auto")
            front_frames, front_meta = engine.process_video(
                front_video,
                progress_callback=lambda f, t: _progress(
                    step_offset + int(f / max(t, 1) * 15), 100,
                    f"Front view: estimating pose... frame {f}/{t}"
                ),
            )
            results["front_video_metadata"] = front_meta
            if "video_metadata" not in results:
                results["video_metadata"] = front_meta

            # Stage 2: Frontal analysis
            _progress(step_offset + 15, 100, "Front view: analyzing knee dynamics...")
            frontal = FrontalAnalyzer()
            frontal_data = frontal.analyze_all(front_frames)
            frontal_summary = frontal.summarize(frontal_data)
            results["frontal_analysis"] = frontal_summary

            # Stage 3: Annotated front video
            # Re-read the video frame by frame to avoid buffering all raw frames in RAM.
            _progress(step_offset + 20, 100, "Front view: rendering annotated video...")
            front_annotated = os.path.join(session_dir, f"{prefix}_front_annotated.mp4")
            f_width, f_height = front_meta["width"], front_meta["height"]
            f_fps = front_meta["fps"]
            from .frontal_video_annotator import FrontalVideoAnnotator
            cap_front = cv2.VideoCapture(front_video)
            with FrontalVideoAnnotator(f_fps, f_width, f_height, front_annotated) as f_annotator:
                for i, (pf, fd) in enumerate(zip(front_frames, frontal_data)):
                    ret, raw = cap_front.read()
                    if not ret:
                        break
                    pf.raw_frame = raw
                    f_annotator.write_frame(pf, fd, frontal_summary)
                    pf.raw_frame = None  # Free immediately
                    if i % 30 == 0:
                        pct = step_offset + 20 + int((i / max(len(front_frames), 1)) * 10)
                        _progress(pct, 100, f"Front: rendering frame {i}/{len(front_frames)}")
            cap_front.release()
            results["front_annotated_video"] = front_annotated
            if "annotated_video" not in results:
                results["annotated_video"] = front_annotated

            step_offset = step_offset + 30

        # ════════════════════════════════════════════════════════════════════
        # REPORT: Combined HTML report
        # ════════════════════════════════════════════════════════════════════
        _progress(90, 100, "Generating combined report...")
        report_path = os.path.join(session_dir, f"{prefix}_report.html")

        from reports.report_generator import generate_report
        generate_report(
            angle_summary=results.get("angle_summary", {}),
            motion_metrics=results.get("motion_metrics", {}),
            video_metadata=results.get("video_metadata", {}),
            chart_path=results.get("chart_png"),
            output_path=report_path,
            video_name=base_name,
            frontal_analysis=results.get("frontal_analysis"),
            pedal_phases={
                side: results.get(f"{side}_pedal_phases")
                for side in ("left", "right")
                if results.get(f"{side}_pedal_phases")
            } or None,
        )
        results["report_html"] = report_path

        _progress(100, 100, "Done!")
        return results
