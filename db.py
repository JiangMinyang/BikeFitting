"""
Session store — persists bike fitting sessions to SQLite.

Each session records optional bike parameters (saddle height, stem length, etc.)
alongside the analysis results so users can track changes over time.

The DB file lives in OUTPUT_DIR (mounted Docker volume) so it persists across
container restarts together with the output files it references.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT    NOT NULL,

    -- User-supplied metadata
    label                TEXT    DEFAULT '',
    notes                TEXT    DEFAULT '',

    -- Bike parameters (all nullable)
    saddle_height_mm     REAL,
    saddle_setback_mm    REAL,
    saddle_tilt_deg      REAL,
    stem_length_mm       REAL,
    stem_angle_deg       REAL,
    handlebar_drop_mm    REAL,
    handlebar_width_mm   REAL,
    crank_length_mm      REAL,
    spacers_mm           REAL,

    -- Denormalised scalar metrics for fast list queries
    motion_score         REAL,
    cadence_rpm          REAL,
    trunk_stability      REAL,
    frontal_score        REAL,
    has_side_view        INTEGER DEFAULT 0,
    has_front_view       INTEGER DEFAULT 0,

    -- Full JSON blobs for replaying results
    motion_metrics       TEXT,
    angle_summary        TEXT,
    frontal_analysis     TEXT,

    -- Output file paths (relative to OUTPUT_DIR, e.g. "foo_report.html")
    report_html          TEXT    DEFAULT '',
    side_annotated_video TEXT    DEFAULT '',
    front_annotated_video TEXT   DEFAULT '',
    chart_png            TEXT    DEFAULT '',
    bdc_frame_png        TEXT    DEFAULT '',
    tdc_frame_png        TEXT    DEFAULT ''
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_created_at
    ON sessions (created_at DESC)
"""

# Columns returned in list queries (lightweight — no JSON blobs)
_LIST_COLS = [
    "id", "created_at", "label", "notes", "user_name", "bike_name",
    "saddle_height_mm", "saddle_setback_mm", "saddle_tilt_deg",
    "stem_length_mm", "stem_angle_deg", "handlebar_drop_mm",
    "handlebar_width_mm", "crank_length_mm", "spacers_mm",
    "motion_score", "cadence_rpm", "trunk_stability", "frontal_score",
    "has_side_view", "has_front_view",
    "report_html", "side_annotated_video", "front_annotated_video",
    "chart_png", "bdc_frame_png", "tdc_frame_png",
]

# Additional columns included in full get_session response
_BLOB_COLS = ["motion_metrics", "angle_summary", "frontal_analysis"]
_ALL_COLS  = _LIST_COLS + _BLOB_COLS

# Bike parameter field names (subset of columns, all nullable floats)
_BIKE_PARAM_COLS = [
    "saddle_height_mm", "saddle_setback_mm", "saddle_tilt_deg",
    "stem_length_mm", "stem_angle_deg", "handlebar_drop_mm",
    "handlebar_width_mm", "crank_length_mm", "spacers_mm",
]


def init_db(db_path: str) -> None:
    """Create the sessions table and index if they don't exist, then migrate."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        # Add columns introduced after initial release (idempotent)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        for col, defn in [
            ("user_name", "TEXT DEFAULT ''"),
            ("bike_name", "TEXT DEFAULT ''"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {defn}")
        conn.commit()


def save_session(db_path: str, bike_params: Dict, results: Dict) -> int:
    """
    Insert a new session row combining bike parameters and analysis results.

    Args:
        bike_params : dict with optional keys from _BIKE_PARAM_COLS plus 'label'/'notes'
        results     : the web_results dict built by server._run_analysis

    Returns:
        The new session id.
    """
    mm = results.get("motion_metrics") or {}
    fa = results.get("frontal_analysis") or {}

    def _rel(path: str) -> str:
        """Store just the basename so paths survive volume remounts."""
        return os.path.basename(path) if path else ""

    def _float(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    row = {
        "created_at":           datetime.now(timezone.utc).isoformat(),
        "label":                bike_params.get("label", ""),
        "notes":                bike_params.get("notes", ""),
        "user_name":            bike_params.get("user_name", ""),
        "bike_name":            bike_params.get("bike_name", ""),
        # Bike params
        "saddle_height_mm":     _float(bike_params, "saddle_height_mm"),
        "saddle_setback_mm":    _float(bike_params, "saddle_setback_mm"),
        "saddle_tilt_deg":      _float(bike_params, "saddle_tilt_deg"),
        "stem_length_mm":       _float(bike_params, "stem_length_mm"),
        "stem_angle_deg":       _float(bike_params, "stem_angle_deg"),
        "handlebar_drop_mm":    _float(bike_params, "handlebar_drop_mm"),
        "handlebar_width_mm":   _float(bike_params, "handlebar_width_mm"),
        "crank_length_mm":      _float(bike_params, "crank_length_mm"),
        "spacers_mm":           _float(bike_params, "spacers_mm"),
        # Scalar metrics
        "motion_score":         _float(mm, "overall_motion_score"),
        "cadence_rpm":          _float(mm, "estimated_cadence_rpm"),
        "trunk_stability":      _float(mm, "trunk_stability_score"),
        "frontal_score":        _float(fa, "frontal_score"),
        "has_side_view":        int(bool(results.get("has_side_view"))),
        "has_front_view":       int(bool(results.get("has_front_view"))),
        # JSON blobs
        "motion_metrics":       json.dumps(mm),
        "angle_summary":        json.dumps(results.get("angle_summary") or {}),
        "frontal_analysis":     json.dumps(fa),
        # File paths (basename only)
        "report_html":          _rel(results.get("report_html", "")),
        "side_annotated_video": _rel(results.get("side_annotated_video", "")),
        "front_annotated_video":_rel(results.get("front_annotated_video", "")),
        "chart_png":            _rel(results.get("chart_png", "")),
        "bdc_frame_png":        _rel(results.get("bdc_frame_png", "")),
        "tdc_frame_png":        _rel(results.get("tdc_frame_png", "")),
    }

    cols   = ", ".join(row.keys())
    params = ", ".join("?" * len(row))
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO sessions ({cols}) VALUES ({params})",
            list(row.values()),
        )
        conn.commit()
        return cur.lastrowid


def list_sessions(db_path: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    """Return lightweight session rows (no JSON blobs), newest first."""
    sql = (
        f"SELECT {', '.join(_LIST_COLS)} FROM sessions "
        f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (limit, offset)).fetchall()
    return [dict(zip(_LIST_COLS, r)) for r in rows]


def count_sessions(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


def get_session(db_path: str, session_id: int) -> Optional[Dict]:
    """Return a full session row including JSON blobs, decoded to dicts."""
    sql = f"SELECT {', '.join(_ALL_COLS)} FROM sessions WHERE id = ?"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, (session_id,)).fetchone()
    if row is None:
        return None
    d = dict(zip(_ALL_COLS, row))
    for blob_col in _BLOB_COLS:
        try:
            d[blob_col] = json.loads(d[blob_col]) if d[blob_col] else {}
        except Exception:
            d[blob_col] = {}
    return d


def update_session(db_path: str, session_id: int, fields: Dict) -> bool:
    """
    Update mutable fields on an existing session.

    Allowed keys: label, notes, plus any _BIKE_PARAM_COLS key.
    Returns True if a row was found and updated.
    """
    allowed = {"label", "notes", "user_name", "bike_name"} | set(_BIKE_PARAM_COLS)
    to_set  = {k: v for k, v in fields.items() if k in allowed}
    if not to_set:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in to_set)
    values     = list(to_set.values()) + [session_id]
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE sessions SET {set_clause} WHERE id = ?", values
        )
        conn.commit()
        return cur.rowcount > 0


def delete_session(db_path: str, session_id: int) -> bool:
    """Delete a session record (output files are NOT removed from disk)."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0
