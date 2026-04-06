"""
Frontal view video annotator.
Draws knee tracking visualization, deviation indicators, and alignment guides.
"""

import cv2
import numpy as np
from typing import Optional, Dict
from .pose_engine import PoseFrame, LANDMARK_INDICES
from .frontal_analyzer import FrontalFrameData


COLORS = {
    "alignment_good": (0, 200, 80),
    "alignment_warn": (0, 160, 255),
    "alignment_bad": (0, 80, 255),
    "skeleton": (0, 200, 100),
    "guide_line": (120, 120, 120),
    "knee_trail": (255, 200, 0),
    "text": (255, 255, 255),
    "hud_bg": (20, 20, 20),
}


def _px(frame: PoseFrame, name: str, w: int, h: int):
    if not frame.has_pose:
        return None
    idx = LANDMARK_INDICES.get(name)
    if idx is None or idx >= len(frame.visibility):
        return None
    if frame.visibility[idx] < 0.25:
        return None
    lm = frame.landmarks[idx]
    return (int(lm[0] * w), int(lm[1] * h))


class FrontalVideoAnnotator:
    """Renders frontal-view annotated frames with knee tracking visualization."""

    def __init__(self, fps: float, width: int, height: int, output_path: str):
        self.fps = fps
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        # Trail buffers for knee tracking path
        self._left_trail = []
        self._right_trail = []
        self._max_trail = 60  # ~2 seconds at 30fps

    def _draw_alignment_line(self, img, hip_pt, ankle_pt, knee_pt, dev_pct):
        """Draw hip-ankle reference line and knee deviation indicator."""
        if not hip_pt or not ankle_pt or not knee_pt:
            return

        # Determine color based on deviation
        abs_dev = abs(dev_pct) if dev_pct is not None else 0
        if abs_dev < 8:
            color = COLORS["alignment_good"]
        elif abs_dev < 15:
            color = COLORS["alignment_warn"]
        else:
            color = COLORS["alignment_bad"]

        # Draw hip-ankle reference line (dashed)
        cv2.line(img, hip_pt, ankle_pt, COLORS["guide_line"], 1, cv2.LINE_AA)

        # Draw actual hip-knee-ankle path
        cv2.line(img, hip_pt, knee_pt, color, 2, cv2.LINE_AA)
        cv2.line(img, knee_pt, ankle_pt, color, 2, cv2.LINE_AA)

        # Knee dot
        cv2.circle(img, knee_pt, 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, knee_pt, 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Deviation label
        if dev_pct is not None:
            label = f"{dev_pct:+.1f}%"
            tx = knee_pt[0] + 12
            ty = knee_pt[1] - 4
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _draw_knee_trail(self, img, trail, color):
        """Draw fading trail showing recent knee positions."""
        if len(trail) < 2:
            return
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            pt_color = tuple(int(c * alpha) for c in color)
            cv2.circle(img, trail[i], 2, pt_color, -1, cv2.LINE_AA)

    def _draw_hud(self, img, fd: FrontalFrameData, summary: Optional[Dict]):
        """Draw frontal analysis HUD."""
        panel_w = 220
        panel_h = 160
        margin = 10
        x0, y0 = margin, margin

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 200, 120), 1)

        y = y0 + 18

        def put(text, color=(200, 200, 200), scale=0.40):
            nonlocal y
            cv2.putText(img, text, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 16

        put("FRONTAL KNEE ANALYSIS", (80, 220, 120), 0.42)
        y += 2
        put(f"Frame: {fd.frame_index}")

        if fd.left_knee_deviation_pct is not None:
            dev = fd.left_knee_deviation_pct
            c = COLORS["alignment_good"] if abs(dev) < 8 else COLORS["alignment_warn"]
            put(f"L-Knee: {dev:+.1f}%", c)

        if fd.right_knee_deviation_pct is not None:
            dev = fd.right_knee_deviation_pct
            c = COLORS["alignment_good"] if abs(dev) < 8 else COLORS["alignment_warn"]
            put(f"R-Knee: {dev:+.1f}%", c)

        if summary:
            score = summary.get("frontal_score")
            if score is not None:
                c = COLORS["alignment_good"] if score >= 70 else COLORS["alignment_warn"]
                put(f"Score: {score:.0f}/100", c)

            for side in ["left", "right"]:
                sd = summary.get(side)
                if sd:
                    cls = sd["classification"].upper()
                    put(f"{side[0].upper()}: {cls}", (200, 200, 200))

    def write_frame(self, pose_frame: PoseFrame, fd: FrontalFrameData,
                    summary: Optional[Dict] = None):
        """Annotate and write one frontal frame."""
        if pose_frame.raw_frame is None:
            return
        img = pose_frame.raw_frame.copy()
        w, h = self.width, self.height

        # Draw alignment for each leg
        for side in ["left", "right"]:
            hip_pt = _px(pose_frame, f"{side}_hip", w, h)
            knee_pt = _px(pose_frame, f"{side}_knee", w, h)
            ankle_pt = _px(pose_frame, f"{side}_ankle", w, h)
            dev_pct = getattr(fd, f"{side}_knee_deviation_pct", None)

            self._draw_alignment_line(img, hip_pt, ankle_pt, knee_pt, dev_pct)

            # Update and draw knee trail
            trail = self._left_trail if side == "left" else self._right_trail
            if knee_pt:
                trail.append(knee_pt)
                if len(trail) > self._max_trail:
                    trail.pop(0)
            trail_color = (255, 180, 0) if side == "left" else (0, 180, 255)
            self._draw_knee_trail(img, trail, trail_color)

        # Draw other skeleton connections
        for start, end in [("left_shoulder", "right_shoulder"),
                           ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                           ("left_hip", "right_hip")]:
            p1 = _px(pose_frame, start, w, h)
            p2 = _px(pose_frame, end, w, h)
            if p1 and p2:
                cv2.line(img, p1, p2, COLORS["skeleton"], 1, cv2.LINE_AA)

        self._draw_hud(img, fd, summary)
        self.writer.write(img)

    def finalize(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finalize()
