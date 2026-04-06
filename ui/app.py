"""
macOS Desktop UI for the Bike Fitting App.
Built with customtkinter for a modern, native-feeling interface on macOS.
Supports dual-video input: side view (joint angles) and front view (knee dynamics).
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import AnalysisPipeline

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

APP_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT = "#2ecc71"
ACCENT2 = "#3498db"
PURPLE = "#9b59b6"
TEXT = "#ecf0f1"
MUTED = "#95a5a6"


class VideoPreviewPanel(ctk.CTkFrame):
    """Shows a thumbnail of the selected video with a label."""

    def __init__(self, parent, label_text="Video", accent_color=ACCENT2, **kwargs):
        super().__init__(parent, fg_color=CARD_BG, corner_radius=12, **kwargs)
        self._accent = accent_color

        tag = ctk.CTkLabel(self, text=label_text,
                           font=ctk.CTkFont(size=10, weight="bold"),
                           text_color=accent_color)
        tag.pack(anchor="w", padx=12, pady=(10, 0))

        self._label = ctk.CTkLabel(
            self, text="Drop or select a video",
            text_color=MUTED, font=ctk.CTkFont(size=12),
            width=180, height=100,
        )
        self._label.pack(expand=True, fill="both", padx=12, pady=(4, 12))
        self._img_ref = None

    def set_video(self, path: str):
        """Extract and display first frame as thumbnail."""
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self._label.configure(text=os.path.basename(path))
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((180, 100), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self._img_ref = tk_img
        self._label.configure(image=tk_img, text="")

    def clear(self):
        self._img_ref = None
        self._label.configure(image=None, text="Drop or select a video")


class MetricBadge(ctk.CTkFrame):
    """Small metric display card."""

    def __init__(self, parent, label: str, value: str = "\u2014", color: str = ACCENT, **kwargs):
        super().__init__(parent, fg_color=CARD_BG, corner_radius=10,
                         border_width=1, border_color="#2c3e50", **kwargs)
        self._value_lbl = ctk.CTkLabel(self, text=value,
                                        font=ctk.CTkFont(size=22, weight="bold"),
                                        text_color=color)
        self._value_lbl.pack(pady=(12, 2))
        ctk.CTkLabel(self, text=label, font=ctk.CTkFont(size=11),
                     text_color=MUTED).pack(pady=(0, 12))

    def set_value(self, value: str, color: str = None):
        self._value_lbl.configure(text=value, **({"text_color": color} if color else {}))


class BikeFitApp(ctk.CTk):
    """Main application window with dual-video support."""

    def __init__(self):
        super().__init__()

        self.title("Bike Fit Analyzer")
        self.geometry("920x750")
        self.minsize(860, 680)
        self.configure(fg_color=APP_BG)

        self._side_video_path: str = ""
        self._front_video_path: str = ""
        self._results: dict = {}
        self._output_dir = os.path.join(os.path.expanduser("~"), "BikeFitResults")

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header bar
        header = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=0, height=64)
        header.pack(fill="x", padx=0, pady=0)
        ctk.CTkLabel(header, text="\U0001f6b4  Bike Fit Analyzer",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=ACCENT).pack(side="left", padx=24, pady=18)
        ctk.CTkLabel(header, text="Powered by MediaPipe",
                     font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="right", padx=24)

        # Main content (left + right columns)
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=16)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # ── Left column ──
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(left, text="VIDEO INPUT",
                     font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", pady=(0, 6))

        # Dual video previews side by side
        preview_row = ctk.CTkFrame(left, fg_color="transparent")
        preview_row.pack(fill="x")
        preview_row.columnconfigure((0, 1), weight=1)

        self.side_preview = VideoPreviewPanel(preview_row, "SIDE VIEW", ACCENT2)
        self.side_preview.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.front_preview = VideoPreviewPanel(preview_row, "FRONT VIEW", PURPLE)
        self.front_preview.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # File labels
        label_row = ctk.CTkFrame(left, fg_color="transparent")
        label_row.pack(fill="x", pady=(4, 0))
        label_row.columnconfigure((0, 1), weight=1)

        self.side_file_label = ctk.CTkLabel(label_row, text="No side video",
                                             text_color=MUTED, font=ctk.CTkFont(size=11))
        self.side_file_label.grid(row=0, column=0, sticky="w")

        self.front_file_label = ctk.CTkLabel(label_row, text="No front video",
                                              text_color=MUTED, font=ctk.CTkFont(size=11))
        self.front_file_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Select buttons
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", pady=8)
        btn_row.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_row, text="Select Side Video\u2026",
                      command=self._pick_side_video,
                      fg_color=ACCENT2, hover_color="#2980b9", text_color="white",
                      font=ctk.CTkFont(weight="bold"), height=36).grid(
                          row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(btn_row, text="Select Front Video\u2026",
                      command=self._pick_front_video,
                      fg_color=PURPLE, hover_color="#8e44ad", text_color="white",
                      font=ctk.CTkFont(weight="bold"), height=36).grid(
                          row=0, column=1, sticky="ew", padx=(4, 0))

        # Output directory
        out_row = ctk.CTkFrame(left, fg_color="transparent")
        out_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(out_row, text="OUTPUT FOLDER",
                     font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w")
        self.out_label = ctk.CTkLabel(out_row, text=self._output_dir,
                                       text_color=MUTED, font=ctk.CTkFont(size=11),
                                       wraplength=360, justify="left")
        self.out_label.pack(anchor="w")
        ctk.CTkButton(out_row, text="Change\u2026", command=self._pick_output_dir,
                      height=28, width=90, fg_color="#2c3e50",
                      hover_color="#34495e").pack(anchor="w", pady=(4, 0))

        # Run button
        self.run_btn = ctk.CTkButton(
            left, text="\u25b6  Run Analysis", command=self._run_analysis,
            fg_color="#2980b9", hover_color="#2471a3",
            font=ctk.CTkFont(size=15, weight="bold"), height=50,
            state="disabled",
        )
        self.run_btn.pack(fill="x", pady=(8, 0))

        # Progress
        self.progress_bar = ctk.CTkProgressBar(left, height=8)
        self.progress_bar.pack(fill="x", pady=(8, 2))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(left, text="Ready",
                                          text_color=MUTED, font=ctk.CTkFont(size=12))
        self.status_label.pack(anchor="w")

        # ── Right column ──
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(right, text="ANALYSIS RESULTS",
                     font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", pady=(0, 6))

        # Side-view metrics
        ctk.CTkLabel(right, text="SIDE VIEW METRICS",
                     font=ctk.CTkFont(size=10), text_color=ACCENT2).pack(anchor="w", pady=(0, 4))

        metrics_frame = ctk.CTkFrame(right, fg_color="transparent")
        metrics_frame.pack(fill="x")
        metrics_frame.columnconfigure((0, 1), weight=1)

        self.badge_score = MetricBadge(metrics_frame, "Motion Score", "\u2014", ACCENT)
        self.badge_score.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self.badge_cadence = MetricBadge(metrics_frame, "Cadence (RPM)", "\u2014", "#9b59b6")
        self.badge_cadence.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 4))

        self.badge_sym = MetricBadge(metrics_frame, "Knee Symmetry", "\u2014", "#3498db")
        self.badge_sym.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self.badge_trunk = MetricBadge(metrics_frame, "Trunk Stability", "\u2014", "#e67e22")
        self.badge_trunk.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        # Front-view metrics
        ctk.CTkLabel(right, text="FRONT VIEW METRICS",
                     font=ctk.CTkFont(size=10), text_color=PURPLE).pack(anchor="w", pady=(10, 4))

        frontal_frame = ctk.CTkFrame(right, fg_color="transparent")
        frontal_frame.pack(fill="x")
        frontal_frame.columnconfigure((0, 1), weight=1)

        self.badge_frontal = MetricBadge(frontal_frame, "Frontal Score", "\u2014", PURPLE)
        self.badge_frontal.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self.badge_frontal_sym = MetricBadge(frontal_frame, "Frontal Symmetry", "\u2014", ACCENT2)
        self.badge_frontal_sym.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 4))

        self.badge_l_knee = MetricBadge(frontal_frame, "L-Knee Dev", "\u2014", ACCENT)
        self.badge_l_knee.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self.badge_r_knee = MetricBadge(frontal_frame, "R-Knee Dev", "\u2014", "#e67e22")
        self.badge_r_knee.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        # Recommendations panel
        ctk.CTkLabel(right, text="RECOMMENDATIONS",
                     font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", pady=(14, 6))

        self.rec_box = ctk.CTkScrollableFrame(right, fg_color=CARD_BG, corner_radius=12, height=160)
        self.rec_box.pack(fill="both", expand=True)

        self.rec_placeholder = ctk.CTkLabel(
            self.rec_box,
            text="Run analysis to see recommendations",
            text_color=MUTED, font=ctk.CTkFont(size=13),
        )
        self.rec_placeholder.pack(pady=20)

        # Action buttons (post-analysis)
        action_row = ctk.CTkFrame(right, fg_color="transparent")
        action_row.pack(fill="x", pady=(12, 0))

        self.open_report_btn = ctk.CTkButton(
            action_row, text="\U0001f4c4 Report",
            command=self._open_report, state="disabled",
            fg_color="#16213e", border_width=1, border_color=ACCENT,
            text_color=ACCENT, hover_color="#0f3460", width=90,
        )
        self.open_report_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.open_side_btn = ctk.CTkButton(
            action_row, text="\U0001f3ac Side",
            command=lambda: self._open_file("side_annotated_video"), state="disabled",
            fg_color="#16213e", border_width=1, border_color=ACCENT2,
            text_color=ACCENT2, hover_color="#0f3460", width=90,
        )
        self.open_side_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))

        self.open_front_btn = ctk.CTkButton(
            action_row, text="\U0001f3ac Front",
            command=lambda: self._open_file("front_annotated_video"), state="disabled",
            fg_color="#16213e", border_width=1, border_color=PURPLE,
            text_color=PURPLE, hover_color="#0f3460", width=90,
        )
        self.open_front_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _pick_side_video(self):
        path = filedialog.askopenfilename(
            title="Select side-view video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return
        self._side_video_path = path
        self.side_file_label.configure(text=os.path.basename(path), text_color=ACCENT2)
        self.side_preview.set_video(path)
        self._update_run_btn()
        self._set_status("Side video loaded.")

    def _pick_front_video(self):
        path = filedialog.askopenfilename(
            title="Select front-view video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return
        self._front_video_path = path
        self.front_file_label.configure(text=os.path.basename(path), text_color=PURPLE)
        self.front_preview.set_video(path)
        self._update_run_btn()
        self._set_status("Front video loaded.")

    def _update_run_btn(self):
        if self._side_video_path or self._front_video_path:
            self.run_btn.configure(state="normal")
        else:
            self.run_btn.configure(state="disabled")

    def _pick_output_dir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._output_dir = d
            self.out_label.configure(text=d)

    def _run_analysis(self):
        if not self._side_video_path and not self._front_video_path:
            messagebox.showwarning("No video", "Please select at least one video file.")
            return

        self.run_btn.configure(state="disabled")
        self.open_report_btn.configure(state="disabled")
        self.open_side_btn.configure(state="disabled")
        self.open_front_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self._clear_recommendations()

        def worker():
            try:
                pipeline = AnalysisPipeline(output_dir=self._output_dir)
                results = pipeline.run(
                    side_video=self._side_video_path or None,
                    front_video=self._front_video_path or None,
                    progress_callback=self._on_progress,
                )
                self._results = results
                self.after(0, self._on_analysis_complete)
            except Exception as e:
                self.after(0, lambda: self._on_analysis_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, current: int, total: int, stage: str = ""):
        pct = current / max(total, 1)
        self.after(0, lambda: self.progress_bar.set(pct))
        self.after(0, lambda: self._set_status(stage or f"{int(pct * 100)}%"))

    def _on_analysis_complete(self):
        # Side-view metrics
        m = self._results.get("motion_metrics", {})
        if m:
            score = m.get("overall_motion_score")
            cadence = m.get("estimated_cadence_rpm")
            sym = m.get("knee_symmetry")
            trunk = m.get("trunk_stability_score")
            score_color = ACCENT if (score or 0) >= 75 else "#e67e22" if (score or 0) >= 50 else "#e74c3c"
            self.badge_score.set_value(f"{score:.0f}" if score else "\u2014", score_color)
            self.badge_cadence.set_value(f"{cadence:.0f}" if cadence else "\u2014")
            self.badge_sym.set_value(f"{sym:.0f}%" if sym else "\u2014")
            self.badge_trunk.set_value(f"{trunk:.0f}" if trunk else "\u2014")

        # Front-view metrics
        fa = self._results.get("frontal_analysis", {})
        if fa:
            fs = fa.get("frontal_score")
            fsym = fa.get("frontal_symmetry")
            self.badge_frontal.set_value(f"{fs:.0f}" if fs else "\u2014",
                                          ACCENT if (fs or 0) >= 70 else "#e67e22")
            self.badge_frontal_sym.set_value(f"{fsym:.0f}%" if fsym else "\u2014")

            left_d = fa.get("left", {})
            right_d = fa.get("right", {})
            if left_d:
                dev = left_d.get("deviation_pct_mean", 0)
                c = ACCENT if abs(dev) < 8 else "#e67e22"
                self.badge_l_knee.set_value(f"{dev:.1f}%", c)
            if right_d:
                dev = right_d.get("deviation_pct_mean", 0)
                c = ACCENT if abs(dev) < 8 else "#e67e22"
                self.badge_r_knee.set_value(f"{dev:.1f}%", c)

        # Recommendations (combined)
        all_recs = m.get("recommendations", []) + fa.get("frontal_recommendations", [])
        self._populate_recommendations(all_recs)

        self.run_btn.configure(state="normal")
        self.open_report_btn.configure(state="normal")
        if self._results.get("side_annotated_video"):
            self.open_side_btn.configure(state="normal")
        if self._results.get("front_annotated_video"):
            self.open_front_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self._set_status("\u2705 Analysis complete!")

    def _on_analysis_error(self, error: str):
        messagebox.showerror("Analysis Error", f"An error occurred:\n{error}")
        self.run_btn.configure(state="normal")
        self.progress_bar.set(0)
        self._set_status("Error \u2014 check your video file and try again.")

    def _populate_recommendations(self, recs: list):
        self._clear_recommendations()
        icons = {"warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f", "success": "\u2705"}
        colors = {"warning": "#e67e22", "info": "#3498db", "success": ACCENT}

        for rec in recs:
            rec_type = rec.get("type", "info")
            icon = icons.get(rec_type, "\u2022")
            color = colors.get(rec_type, MUTED)

            card = ctk.CTkFrame(self.rec_box, fg_color="#0f3460", corner_radius=8)
            card.pack(fill="x", padx=6, pady=4)

            header_text = f"{icon}  {rec['joint']} \u2014 {rec['metric']}"
            ctk.CTkLabel(card, text=header_text, font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=color, wraplength=340, justify="left").pack(
                             anchor="w", padx=12, pady=(10, 2))
            ctk.CTkLabel(card, text=rec["suggestion"], font=ctk.CTkFont(size=11),
                         text_color=MUTED, wraplength=340, justify="left").pack(
                             anchor="w", padx=12, pady=(0, 10))

    def _clear_recommendations(self):
        for widget in self.rec_box.winfo_children():
            widget.destroy()

    def _open_report(self):
        path = self._results.get("report_html")
        if path and os.path.exists(path):
            subprocess.run(["open", path])

    def _open_file(self, key: str):
        path = self._results.get(key)
        if path and os.path.exists(path):
            subprocess.run(["open", path])

    def _set_status(self, text: str):
        self.status_label.configure(text=text)


def run():
    app = BikeFitApp()
    app.mainloop()


if __name__ == "__main__":
    run()
