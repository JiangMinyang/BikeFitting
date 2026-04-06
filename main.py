#!/usr/bin/env python3
"""
Bike Fit Analyzer — entry point.
Run:  python main.py                          → launches macOS desktop app
      python main.py --web                     → launches web UI at http://localhost:8080
      python main.py --cli --side <video>      → side-view analysis
      python main.py --cli --front <video>     → front-view analysis
      python main.py --cli --side s.mp4 --front f.mp4  → dual analysis
      python main.py --cli <video>             → legacy single-video mode (side)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli(side_video: str = None, front_video: str = None, output_dir: str = None):
    """Headless CLI mode for scripting / testing."""
    from core.pipeline import AnalysisPipeline

    # In Docker, default to /output; on host, default to ~/BikeFitResults
    if output_dir:
        out = output_dir
    elif os.path.isdir("/output"):
        out = "/output"
    else:
        out = os.path.join(os.path.expanduser("~"), "BikeFitResults")

    def progress(current, total, stage):
        bar = "\u2588" * int(current / max(total, 1) * 30)
        bar = bar.ljust(30)
        print(f"\r  [{bar}] {current}% {stage}   ", end="", flush=True)

    print(f"\n\U0001f6b4  Bike Fit Analyzer \u2014 CLI Mode")
    if side_video:
        print(f"   Side video:  {side_video}")
    if front_video:
        print(f"   Front video: {front_video}")
    print(f"   Output: {out}\n")

    pipeline = AnalysisPipeline(output_dir=out)
    results = pipeline.run(
        side_video=side_video,
        front_video=front_video,
        progress_callback=progress,
    )

    print(f"\n\n\u2705  Done!\n")

    if results.get("side_annotated_video"):
        print(f"   \U0001f4f9 Side annotated:  {results['side_annotated_video']}")
    if results.get("front_annotated_video"):
        print(f"   \U0001f4f9 Front annotated: {results['front_annotated_video']}")
    print(f"   \U0001f4c4 HTML report:      {results['report_html']}")
    if results.get("chart_png"):
        print(f"   \U0001f4ca Angle chart:     {results['chart_png']}")

    # Side-view metrics
    m = results.get("motion_metrics", {})
    if m:
        print(f"\n\u2500\u2500 Side View Metrics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        print(f"   Overall Motion Score: {m.get('overall_motion_score', 'N/A')}/100")
        print(f"   Estimated Cadence:    {m.get('estimated_cadence_rpm', 'N/A')} RPM")
        print(f"   Knee Symmetry:        {m.get('knee_symmetry', 'N/A')}%")
        print(f"   Trunk Stability:      {m.get('trunk_stability_score', 'N/A')}/100")

    # Frontal metrics
    fa = results.get("frontal_analysis", {})
    if fa:
        print(f"\n\u2500\u2500 Front View Metrics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        print(f"   Frontal Score:    {fa.get('frontal_score', 'N/A')}/100")
        print(f"   Frontal Symmetry: {fa.get('frontal_symmetry', 'N/A')}%")
        for side in ["left", "right"]:
            sd = fa.get(side)
            if sd:
                print(f"   {side.capitalize()} Knee: {sd['classification']} "
                      f"(dev: {sd['deviation_pct_mean']:.1f}%, "
                      f"tracking: {sd['tracking_score']:.0f}/100)")

    # Recommendations
    all_recs = m.get("recommendations", []) + fa.get("frontal_recommendations", [])
    if all_recs:
        print(f"\n\u2500\u2500 Recommendations \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        for rec in all_recs:
            icon = {"warning": "\u26a0\ufe0f ", "info": "\u2139\ufe0f ", "success": "\u2705"}.get(rec["type"], "  ")
            print(f"   {icon} [{rec['joint']}] {rec['metric']}: {rec['suggestion']}")
    print()


def run_gui():
    """Launch the macOS desktop app."""
    from ui.app import run
    run()


def run_web(port: int = 8080):
    """Launch the web UI server."""
    from server import main as server_main
    sys.argv = ["server", "--port", str(port)]
    server_main()


if __name__ == "__main__":
    if "--web" in sys.argv:
        port = 8080
        if "--port" in sys.argv:
            pi = sys.argv.index("--port")
            port = int(sys.argv[pi + 1]) if pi + 1 < len(sys.argv) else 8080
        run_web(port)
    elif "--cli" in sys.argv:
        side = None
        front = None
        out_dir = None

        args = sys.argv[1:]

        # Parse --side, --front, --output
        if "--side" in args:
            si = args.index("--side")
            side = args[si + 1] if si + 1 < len(args) else None
        if "--front" in args:
            fi = args.index("--front")
            front = args[fi + 1] if fi + 1 < len(args) else None
        if "--output" in args:
            oi = args.index("--output")
            out_dir = args[oi + 1] if oi + 1 < len(args) else None

        # Legacy: python main.py --cli <video> (positional arg after --cli)
        if not side and not front:
            ci = args.index("--cli")
            if ci + 1 < len(args) and not args[ci + 1].startswith("--"):
                side = args[ci + 1]

        if not side and not front:
            print("Usage: python main.py --cli --side <video> [--front <video>] [--output <dir>]")
            print("       python main.py --cli <video>  (legacy single-video mode)")
            sys.exit(1)

        run_cli(side_video=side, front_video=front, output_dir=out_dir)
    else:
        run_gui()
