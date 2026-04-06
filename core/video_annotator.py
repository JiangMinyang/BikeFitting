"""
Video annotation module.
Overlays skeleton, joint angles, and metric HUD onto video frames.
Produces an annotated MP4 output file.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from .pose_engine import PoseFrame, LANDMARK_INDICES
from .angle_calculator import JointAngles


# Color palette (BGR)
COLORS = {
    "skeleton":   (0, 200, 100),
    "joint_dot":  (255, 255, 0),
    "angle_text": (255, 255, 255),
    "hud_bg":     (20, 20, 20),
    "hud_border": (80, 200, 120),
    "warning":    (0, 100, 255),
    "good":       (0, 200, 80),
    "info":       (255, 200, 0),
}

# Fraction of shin length used to project an estimated toe point when
# no foot landmark is available at all.
_FOOT_ESTIMATE_RATIO = 0.45

# Skeleton connections to draw.
# Foot chain: ankle → heel → big_toe — shows full foot shape when landmarks available.
POSE_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("left_ankle",     "left_heel"),        # ankle → heel
    ("left_heel",      "left_foot_index"),  # heel  → big toe
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("right_ankle",    "right_heel"),       # ankle → heel
    ("right_heel",     "right_foot_index"), # heel  → big toe
]

# Joint triplets to annotate angles at (vertex landmark, label, JointAngles attr)
ANGLE_ANNOTATIONS = [
    ("left_knee",     "L-Knee", "left_knee"),
    ("right_knee",    "R-Knee", "right_knee"),
    ("left_hip",      "L-Hip",  "left_hip"),
    ("right_hip",     "R-Hip",  "right_hip"),
    ("left_ankle",    "L-Ank",  "left_ankle"),
    ("right_ankle",   "R-Ank",  "right_ankle"),
    ("left_elbow",    "L-Elb",  "left_elbow"),
    ("right_elbow",   "R-Elb",  "right_elbow"),
    ("left_shoulder",  "L-Arm", "left_shoulder_arm"),
    ("right_shoulder", "R-Arm", "right_shoulder_arm"),
]


def _px(frame: PoseFrame, name: str, w: int, h: int,
        min_vis: float = 0.25) -> Optional[Tuple[int, int]]:
    """Get pixel coords for a landmark, returning None if below min_vis."""
    if not frame.has_pose:
        return None
    idx = LANDMARK_INDICES.get(name)
    if idx is None:
        return None
    if frame.visibility[idx] < min_vis:
        return None
    lm = frame.landmarks[idx]
    return (int(lm[0] * w), int(lm[1] * h))


class VideoAnnotator:
    """Renders annotated frames and writes output video."""

    def __init__(self, fps: float, width: int, height: int, output_path: str):
        self.fps = fps
        self.width = width
        self.height = height
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def draw_feet(self, frame_img: np.ndarray, pose_frame: PoseFrame,
                  near_side: Optional[str] = None):
        """
        Estimation-only fallback for when no foot landmarks are detected at all.

        Heel and big-toe landmarks are normally drawn by draw_skeleton as regular
        joints (same color, same size). This method only runs to provide a visual
        estimate when the ankle is visible but both heel and toe landmarks are
        absent — e.g. with a 17-keypoint model that has no foot keypoints.
        """
        w, h = self.width, self.height
        far_side = ("right" if near_side == "left" else "left") if near_side else None

        for side in ("left", "right"):
            if side == far_side:
                continue
            # If any real foot landmark is detected, skip — draw_skeleton handles it
            if (_px(pose_frame, f"{side}_foot_index", w, h, min_vis=0.10) is not None or
                    _px(pose_frame, f"{side}_heel",       w, h, min_vis=0.10) is not None):
                continue

            knee_pt  = _px(pose_frame, f"{side}_knee",  w, h)
            ankle_pt = _px(pose_frame, f"{side}_ankle", w, h)
            if ankle_pt is None or knee_pt is None:
                continue

            # Estimate: project shin vector past the ankle
            shin_vec = np.array([ankle_pt[0] - knee_pt[0],
                                 ankle_pt[1] - knee_pt[1]], dtype=float)
            shin_len = np.linalg.norm(shin_vec)
            if shin_len < 1e-3:
                continue
            shin_unit = shin_vec / shin_len
            offset    = shin_unit * shin_len * _FOOT_ESTIMATE_RATIO
            est_pt    = (int(ankle_pt[0] + offset[0]),
                         int(ankle_pt[1] + offset[1]))

            # Dashed line to estimated point
            steps = 6
            for i in range(steps):
                if i % 2 == 0:
                    t0 = i / steps
                    t1 = (i + 1) / steps
                    p0 = (int(ankle_pt[0] + offset[0] * t0),
                          int(ankle_pt[1] + offset[1] * t0))
                    p1 = (int(ankle_pt[0] + offset[0] * t1),
                          int(ankle_pt[1] + offset[1] * t1))
                    cv2.line(frame_img, p0, p1, COLORS["skeleton"], 2, cv2.LINE_AA)

            # Hollow circle = estimated, not detected
            cv2.circle(frame_img, est_pt, 4, COLORS["joint_dot"], 1, cv2.LINE_AA)

    def draw_skeleton(self, frame_img: np.ndarray, pose_frame: PoseFrame,
                      near_side: Optional[str] = None):
        """Draw skeleton connections and joint dots.

        If near_side is 'left' or 'right', all landmarks belonging exclusively
        to the far side are hidden. Cross-body connections (shoulder-to-shoulder,
        hip-to-hip) are kept so the torso outline remains visible.
        """
        w, h = self.width, self.height
        far_pfx = ("right_" if near_side == "left" else "left_") if near_side else None

        for start_name, end_name in POSE_CONNECTIONS:
            # Skip connections where BOTH endpoints are on the far side
            if far_pfx and start_name.startswith(far_pfx) and end_name.startswith(far_pfx):
                continue
            p1 = _px(pose_frame, start_name, w, h)
            p2 = _px(pose_frame, end_name, w, h)
            if p1 and p2:
                cv2.line(frame_img, p1, p2, COLORS["skeleton"], 2, cv2.LINE_AA)

        for name in LANDMARK_INDICES:
            # Skip far-side joint dots
            if far_pfx and name.startswith(far_pfx):
                continue
            pt = _px(pose_frame, name, w, h)
            if pt:
                cv2.circle(frame_img, pt, 4, COLORS["joint_dot"], -1, cv2.LINE_AA)

    def draw_angle_labels(self, frame_img: np.ndarray, pose_frame: PoseFrame,
                          angles: JointAngles, near_side: Optional[str] = None):
        """Draw angle values next to each joint."""
        w, h = self.width, self.height
        far_pfx = ("right_" if near_side == "left" else "left_") if near_side else None

        for landmark_name, label, attr in ANGLE_ANNOTATIONS:
            if far_pfx and landmark_name.startswith(far_pfx):
                continue
            pt = _px(pose_frame, landmark_name, w, h)
            value = getattr(angles, attr, None)
            if pt and value is not None:
                text = f"{label}: {value:.1f} deg"
                tx, ty = pt[0] + 10, pt[1] - 10
                # Background pill
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                cv2.rectangle(frame_img,
                              (tx - 3, ty - th - 3),
                              (tx + tw + 3, ty + 3),
                              (0, 0, 0), -1)
                cv2.putText(frame_img, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            COLORS["angle_text"], 2, cv2.LINE_AA)

    def draw_hud(self, frame_img: np.ndarray, angles: JointAngles,
                 motion_metrics: Optional[Dict] = None, frame_idx: int = 0):
        """Draw heads-up display panel with key metrics."""
        panel_w = 270
        panel_h = 355
        margin = 10
        x0 = self.width - panel_w - margin
        y0 = margin

        # Semi-transparent background
        overlay = frame_img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame_img, 0.25, 0, frame_img)
        cv2.rectangle(frame_img, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      COLORS["hud_border"], 1)

        y = y0 + 24
        def put(text, color=(200, 200, 200), scale=0.58):
            nonlocal y
            cv2.putText(frame_img, text, (x0 + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 22

        put("BIKE FIT ANALYSIS", (80, 220, 120), 0.60)
        y += 4
        put(f"Frame: {frame_idx}", scale=0.52)

        # Lower body
        if angles.left_knee is not None:
            put(f"L-Knee:  {angles.left_knee:.1f} deg")
        if angles.right_knee is not None:
            put(f"R-Knee:  {angles.right_knee:.1f} deg")
        if angles.left_ankle is not None:
            put(f"L-Ankle: {angles.left_ankle:.1f} deg")
        if angles.right_ankle is not None:
            put(f"R-Ankle: {angles.right_ankle:.1f} deg")

        # Upper body
        if angles.left_elbow is not None:
            put(f"L-Elbow: {angles.left_elbow:.1f} deg")
        if angles.right_elbow is not None:
            put(f"R-Elbow: {angles.right_elbow:.1f} deg")
        if angles.left_shoulder_arm is not None:
            put(f"L-Arm:   {angles.left_shoulder_arm:.1f} deg")
        if angles.right_shoulder_arm is not None:
            put(f"R-Arm:   {angles.right_shoulder_arm:.1f} deg")
        if angles.trunk_angle is not None:
            put(f"Trunk:   {angles.trunk_angle:.1f} deg")

        if motion_metrics:
            y += 4
            cadence = motion_metrics.get("estimated_cadence_rpm")
            if cadence:
                put(f"Cadence: {cadence:.0f} RPM", COLORS["info"])
            score = motion_metrics.get("overall_motion_score")
            if score is not None:
                color = COLORS["good"] if score >= 75 else COLORS["warning"]
                put(f"Score:   {score:.0f}/100", color)

    def write_frame(self, pose_frame: PoseFrame, angles: JointAngles,
                    motion_metrics: Optional[Dict] = None,
                    near_side: Optional[str] = None):
        """Annotate and write a single frame."""
        if pose_frame.raw_frame is None:
            return
        frame_img = pose_frame.raw_frame.copy()

        if pose_frame.has_pose:
            self.draw_skeleton(frame_img, pose_frame, near_side)
            self.draw_feet(frame_img, pose_frame, near_side)
            self.draw_angle_labels(frame_img, pose_frame, angles, near_side)

        self.draw_hud(frame_img, angles, motion_metrics, pose_frame.frame_index)
        self.writer.write(frame_img)

    def finalize(self):
        """Release the video writer."""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finalize()


def annotate_single_frame(
    raw_frame: np.ndarray,
    pose_frame: PoseFrame,
    angles: JointAngles,
    motion_metrics: Optional[Dict] = None,
    label: str = "",
    output_path: Optional[str] = None,
    near_side: Optional[str] = None,
) -> np.ndarray:
    """
    Annotate a single BGR frame with skeleton, joint angles, and HUD.
    Optionally saves the result as a PNG and returns the annotated image.

    Used for BDC/TDC keyframe extraction — no VideoWriter is created.
    """
    h, w = raw_frame.shape[:2]

    # Build a lightweight annotator shell — no VideoWriter needed
    ann = VideoAnnotator.__new__(VideoAnnotator)
    ann.fps    = 30
    ann.width  = w
    ann.height = h
    ann.writer = None

    frame_img = raw_frame.copy()
    if pose_frame.has_pose:
        ann.draw_skeleton(frame_img, pose_frame, near_side)
        ann.draw_feet(frame_img, pose_frame, near_side)
        ann.draw_angle_labels(frame_img, pose_frame, angles, near_side)
    ann.draw_hud(frame_img, angles, motion_metrics, pose_frame.frame_index)

    # Centred label banner at the bottom
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.9, 2)
        x = max(w // 2 - tw // 2, 4)
        y = h - 16
        cv2.rectangle(frame_img, (x - 8, y - th - 8), (x + tw + 8, y + 6),
                      (0, 0, 0), -1)
        cv2.putText(frame_img, label, (x, y), font, 0.9,
                    (0, 220, 120), 2, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, frame_img)

    return frame_img


def generate_angle_chart(angle_summary: Dict, output_path: str):
    """
    Generate a static PNG chart of joint angles over time using matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle("Joint Angle Analysis", fontsize=14, fontweight="bold", color="#1a1a2e")
    fig.patch.set_facecolor("#f8f9fa")

    plot_groups = [
        ("Knee Angles",       ["left_knee",        "right_knee"],        axes[0, 0]),
        ("Hip Angles",        ["left_hip",          "right_hip"],         axes[0, 1]),
        ("Ankle Angles",      ["left_ankle",        "right_ankle"],       axes[1, 0]),
        ("Trunk Angle",       ["trunk_angle"],                            axes[1, 1]),
        ("Elbow Angles",      ["left_elbow",        "right_elbow"],       axes[2, 0]),
        ("Upper Arm Angles",  ["left_shoulder_arm", "right_shoulder_arm"], axes[2, 1]),
    ]

    colors = {
        "left_knee":         "#e74c3c", "right_knee":         "#c0392b",
        "left_hip":          "#3498db", "right_hip":          "#2980b9",
        "left_ankle":        "#2ecc71", "right_ankle":        "#27ae60",
        "trunk_angle":       "#9b59b6",
        "left_elbow":        "#e67e22", "right_elbow":        "#d35400",
        "left_shoulder_arm": "#1abc9c", "right_shoulder_arm": "#16a085",
    }

    labels = {
        "left_knee":         "Left Knee",       "right_knee":         "Right Knee",
        "left_hip":          "Left Hip",         "right_hip":          "Right Hip",
        "left_ankle":        "Left Ankle",       "right_ankle":        "Right Ankle",
        "trunk_angle":       "Trunk Angle",
        "left_elbow":        "Left Elbow",       "right_elbow":        "Right Elbow",
        "left_shoulder_arm": "Left Upper Arm",   "right_shoulder_arm": "Right Upper Arm",
    }

    for title, keys, ax in plot_groups:
        ax.set_facecolor("#ffffff")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Angle (°)", fontsize=9)
        ax.grid(True, alpha=0.3)

        any_data = False
        for key in keys:
            s = angle_summary.get(key)
            if s:
                ax.plot(s["timestamps"], s["data"],
                        color=colors.get(key, "#555"),
                        label=labels.get(key, key), linewidth=1.5, alpha=0.85)
                any_data = True

        if any_data:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
