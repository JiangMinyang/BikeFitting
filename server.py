#!/usr/bin/env python3
"""
Bike Fit Analyzer — Web Server
Lightweight HTTP server using only stdlib (no Flask/FastAPI needed).
Serves the web UI and handles video upload + analysis via JSON API.

Run:  python server.py [--port 8080]
"""

import os
import re
import sys
import json
import threading
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import AnalysisPipeline
import db as session_db

# ── Config ────────────────────────────────────────────────────────────────────

PORT       = int(os.environ.get("PORT", 8080))
HOST       = "0.0.0.0"
WEB_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
DB_PATH    = os.path.join(OUTPUT_DIR, "sessions.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
session_db.init_db(DB_PATH)

# Shared state for analysis progress
_analysis_state = {
    "running":  False,
    "progress": 0,
    "stage":    "idle",
    "results":  None,
    "error":    None,
}
_state_lock = threading.Lock()


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types (float32, int64, etc.)."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.floating):  return float(obj)
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, np.ndarray):   return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


class BikeFitHandler(SimpleHTTPRequestHandler):
    """Handles both static file serving and API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    # ── Routing ───────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/status":
            self._json_response(_analysis_state)
        elif path == "/api/sessions":
            self._list_sessions()
        elif re.fullmatch(r"/api/sessions/(\d+)", path):
            self._get_session(int(re.fullmatch(r"/api/sessions/(\d+)", path).group(1)))
        elif path.startswith("/output/"):
            self._serve_output_file(path)
        else:
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/analyze":
            self._handle_upload_and_analyze()
        else:
            self.send_error(404, "Not found")

    def do_PATCH(self):
        path = urlparse(self.path).path
        m = re.fullmatch(r"/api/sessions/(\d+)", path)
        if m:
            self._update_session(int(m.group(1)))
        else:
            self.send_error(404, "Not found")

    def do_DELETE(self):
        path = urlparse(self.path).path
        m = re.fullmatch(r"/api/sessions/(\d+)", path)
        if m:
            self._delete_session(int(m.group(1)))
        else:
            self.send_error(404, "Not found")

    # ── Session API ───────────────────────────────────────────────────────────

    def _list_sessions(self):
        qs     = parse_qs(urlparse(self.path).query)
        limit  = int(qs.get("limit",  ["50"])[0])
        offset = int(qs.get("offset", ["0"])[0])
        rows   = session_db.list_sessions(DB_PATH, limit=limit, offset=offset)
        total  = session_db.count_sessions(DB_PATH)
        # Prefix stored basenames with /output/ for the frontend
        for r in rows:
            for key in ("report_html", "side_annotated_video", "front_annotated_video",
                        "chart_png", "bdc_frame_png", "tdc_frame_png"):
                if r.get(key):
                    r[key] = "/output/" + r[key]
        self._json_response({"sessions": rows, "total": total,
                             "limit": limit, "offset": offset})

    def _get_session(self, session_id: int):
        row = session_db.get_session(DB_PATH, session_id)
        if row is None:
            self._json_response({"error": "Not found"}, 404)
            return
        # Reconstruct results dict (same shape as web_results in _run_analysis)
        results = {
            "has_side_view":         bool(row["has_side_view"]),
            "has_front_view":        bool(row["has_front_view"]),
            "motion_metrics":        row["motion_metrics"],
            "angle_summary":         row["angle_summary"],
            "frontal_analysis":      row["frontal_analysis"] or None,
        }
        for key in ("report_html", "side_annotated_video", "front_annotated_video",
                    "chart_png", "bdc_frame_png", "tdc_frame_png"):
            if row.get(key):
                results[key] = "/output/" + row[key]
        bike_params = {k: row[k] for k in session_db._BIKE_PARAM_COLS}
        bike_params["label"]     = row["label"]
        bike_params["notes"]     = row["notes"]
        bike_params["user_name"] = row.get("user_name", "")
        bike_params["bike_name"] = row.get("bike_name", "")
        self._json_response({"id": row["id"], "created_at": row["created_at"],
                             "label": row["label"], "notes": row["notes"],
                             "user_name": row.get("user_name", ""),
                             "bike_name": row.get("bike_name", ""),
                             "bike_params": bike_params, "results": results})

    def _update_session(self, session_id: int):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._json_response({"error": "Invalid JSON body"}, 400)
            return
        ok = session_db.update_session(DB_PATH, session_id, body)
        if ok:
            self._json_response({"ok": True})
        else:
            self._json_response({"error": "Not found"}, 404)

    def _delete_session(self, session_id: int):
        ok = session_db.delete_session(DB_PATH, session_id)
        if ok:
            self._json_response({"ok": True})
        else:
            self._json_response({"error": "Not found"}, 404)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _handle_upload_and_analyze(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json_response({"error": "Expected multipart/form-data"}, 400)
            return

        boundary       = content_type.split("boundary=")[-1].strip()
        content_length = int(self.headers.get("Content-Length", 0))
        body           = self.rfile.read(content_length)

        files, fields = self._parse_multipart(body, boundary)

        # Parse bike params from text field (graceful fallback)
        try:
            bike_params = json.loads(fields.get("bike_params", "{}") or "{}")
        except Exception:
            bike_params = {}

        side_path  = None
        front_path = None
        for field_name, (file_data, filename) in files.items():
            safe = "".join(c for c in filename if c.isalnum() or c in "._- ") or "upload.mp4"
            path = os.path.join(UPLOAD_DIR, safe)
            with open(path, "wb") as f:
                f.write(file_data)
            if field_name == "side_video":
                side_path = path
            elif field_name == "front_video":
                front_path = path
            elif field_name == "video":
                side_path = path

        if not side_path and not front_path:
            self._json_response({"error": "No video file found in upload"}, 400)
            return

        with _state_lock:
            if _analysis_state["running"]:
                self._json_response({"error": "Analysis already in progress"}, 409)
                return
            _analysis_state.update(running=True, progress=0,
                                   stage="Starting...", results=None, error=None)

        thread = threading.Thread(
            target=self._run_analysis,
            args=(side_path, front_path, bike_params),
            daemon=True,
        )
        thread.start()

        names = []
        if side_path:  names.append(f"side: {os.path.basename(side_path)}")
        if front_path: names.append(f"front: {os.path.basename(front_path)}")
        self._json_response({"status": "started", "files": names})

    def _parse_multipart(self, body: bytes, boundary: str):
        """
        Split multipart body into files and text fields.

        Returns:
            files  : {field_name: (bytes, filename)}
            fields : {field_name: str_value}
        """
        boundary_bytes = f"--{boundary}".encode()
        parts  = body.split(boundary_bytes)
        files  = {}
        fields = {}

        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            headers   = part[:header_end].decode("utf-8", errors="replace")
            part_body = part[header_end + 4:]
            # Strip trailing boundary markers
            for suffix in (b"\r\n", b"--\r\n", b"--"):
                if part_body.endswith(suffix):
                    part_body = part_body[: -len(suffix)]

            # Extract field name
            field_name = "unknown"
            if 'name="' in headers:
                nm_s = headers.index('name="') + 6
                nm_e = headers.index('"', nm_s)
                field_name = headers[nm_s:nm_e]

            if 'filename="' in headers:
                fn_s     = headers.index('filename="') + 10
                fn_e     = headers.index('"', fn_s)
                filename = headers[fn_s:fn_e]
                if filename and part_body:
                    files[field_name] = (part_body, filename)
            else:
                # Plain text field
                fields[field_name] = part_body.decode("utf-8", errors="replace")

        return files, fields

    @staticmethod
    def _run_analysis(side_path=None, front_path=None, bike_params=None):
        bike_params = bike_params or {}
        try:
            pipeline = AnalysisPipeline(output_dir=OUTPUT_DIR)

            def on_progress(current, total, stage):
                with _state_lock:
                    _analysis_state["progress"] = current
                    _analysis_state["stage"]    = stage

            results = pipeline.run(
                side_video=side_path,
                front_video=front_path,
                progress_callback=on_progress,
            )

            def _rel(path):
                """Return the path relative to OUTPUT_DIR for use in /output/ URLs."""
                return "/output/" + os.path.relpath(path, OUTPUT_DIR)

            web_results = {
                "has_side_view":  results.get("has_side_view", False),
                "has_front_view": results.get("has_front_view", False),
                "video_metadata": results.get("video_metadata", {}),
                "report_html":    _rel(results["report_html"]),
            }
            for key, src_key in [
                ("side_annotated_video",  "side_annotated_video"),
                ("front_annotated_video", "front_annotated_video"),
                ("chart_png",             "chart_png"),
                ("bdc_frame_png",         "bdc_frame_png"),
                ("tdc_frame_png",         "tdc_frame_png"),
                ("annotated_video",       "annotated_video"),
            ]:
                if results.get(src_key):
                    web_results[key] = _rel(results[src_key])
            for key in ("motion_metrics", "angle_summary", "frontal_analysis"):
                if results.get(key):
                    web_results[key] = results[key]

            # Persist to session DB — round-trip through _NumpyEncoder to convert
            # any numpy scalars to native Python types before SQLite storage
            clean_results = json.loads(json.dumps(web_results, cls=_NumpyEncoder))
            session_id = session_db.save_session(DB_PATH, bike_params, clean_results)
            web_results["session_id"] = session_id

            with _state_lock:
                _analysis_state.update(running=False, progress=100,
                                       stage="Done!", results=web_results)

        except Exception as e:
            import traceback; traceback.print_exc()
            with _state_lock:
                _analysis_state.update(running=False, error=str(e),
                                       stage=f"Error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _json_response(self, data, status=200):
        body = json.dumps(data, cls=_NumpyEncoder).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_output_file(self, path):
        rel       = path.replace("/output/", "", 1)
        file_path = os.path.join(OUTPUT_DIR, rel)
        if os.path.isfile(file_path):
            mime, _ = mimetypes.guess_type(file_path)
            mime = mime or "application/octet-stream"
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "File not found")

    def log_message(self, format, *args):
        path = args[0] if args else ""
        if "/api/" in str(path) or "404" in str(args) or "500" in str(args):
            super().log_message(format, *args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bike Fit Analyzer Web Server")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), BikeFitHandler)
    print(f"\n  Bike Fit Analyzer -- Web Server")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
