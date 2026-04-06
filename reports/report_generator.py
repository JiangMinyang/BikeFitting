"""
HTML report generator for bike fitting analysis results.
Generates a self-contained HTML report with research-backed metrics,
pedaling quality analysis, and evidence-based recommendations.
"""

import base64
import os
from datetime import datetime
from typing import Dict, Optional


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _score_badge(v, fmt=".0f"):
    if v is None:
        return '<span class="badge grey">N/A</span>'
    color = "green" if v >= 75 else "orange" if v >= 50 else "red"
    return f'<span class="badge {color}">{v:{fmt}}</span>'


def _metric_card(label: str, value, unit: str = "", color: str = "#3498db",
                 subtitle: str = "") -> str:
    val_str = (f"{value:.1f}" if isinstance(value, float) else
               str(value) if value is not None else "N/A")
    sub = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-card" style="border-top: 4px solid {color};">
        <div class="metric-value">{val_str}<span class="metric-unit">{unit}</span></div>
        <div class="metric-label">{label}</div>{sub}
    </div>"""


def generate_report(
    angle_summary: Dict,
    motion_metrics: Dict,
    video_metadata: Dict,
    chart_path: Optional[str] = None,
    output_path: str = "bike_fit_report.html",
    video_name: str = "analysis",
    frontal_analysis: Optional[Dict] = None,
    pedal_phases: Optional[Dict] = None,   # {left: {...}, right: {...}}
) -> str:

    now = datetime.now().strftime("%B %d, %Y at %H:%M")
    duration = video_metadata.get("duration_sec", 0)
    fps = video_metadata.get("fps", 0)
    total_frames = video_metadata.get("total_frames", 0)
    backend = video_metadata.get("backend", "unknown")

    # Chart
    chart_html = ""
    if chart_path and os.path.exists(chart_path):
        img_b64 = _encode_image(chart_path)
        chart_html = f'<img src="data:image/png;base64,{img_b64}" class="chart-img" alt="Joint Angle Chart"/>'

    # ── Key Metrics cards ───────────────────────────────────────────────────
    overall = motion_metrics.get("overall_motion_score")
    cadence = motion_metrics.get("estimated_cadence_rpm")
    trunk_stable = motion_metrics.get("trunk_stability_score")
    ps = motion_metrics.get("pedal_smoothness", {}).get("pedal_smoothness_pct")
    ds = motion_metrics.get("dead_spot", {}).get("dead_spot_score")
    cons = motion_metrics.get("stroke_consistency", {}).get("consistency_score")
    sparc = motion_metrics.get("sparc_smoothness")

    score_color = ("#2ecc71" if (overall or 0) >= 75 else
                   "#e67e22" if (overall or 0) >= 50 else "#e74c3c")

    metrics_html = "".join([
        _metric_card("Overall Score", overall, "/100", score_color),
        _metric_card("Cadence", cadence, " RPM", "#9b59b6",
                     "Typical: 80-100") if cadence else "",
        _metric_card("Pedal Smoothness", ps, "%", "#3498db",
                     "Typical: 15-35%") if ps is not None else "",
        _metric_card("Dead Spot Score", ds, "/100", "#e67e22",
                     "Higher = fewer dead spots") if ds is not None else "",
        _metric_card("Stroke Consistency", cons, "/100", "#1abc9c",
                     "Higher = more consistent") if cons is not None else "",
        _metric_card("Trunk Stability", trunk_stable, "/100", "#2c3e50"),
    ])

    # ── Joint Angle Table ───────────────────────────────────────────────────
    joint_display = [
        ("Left Knee", "left_knee"), ("Right Knee", "right_knee"),
        ("Left Hip", "left_hip"), ("Right Hip", "right_hip"),
        ("Left Ankle", "left_ankle"), ("Right Ankle", "right_ankle"),
        ("Left Elbow", "left_elbow"), ("Right Elbow", "right_elbow"),
        ("Left Upper Arm", "left_shoulder_arm"), ("Right Upper Arm", "right_shoulder_arm"),
        ("Trunk Angle", "trunk_angle"),
    ]
    angle_rows = ""
    for display_name, key in joint_display:
        s = angle_summary.get(key)
        if s:
            std_val = s.get("std", 0)
            angle_rows += f"""
            <tr>
                <td><strong>{display_name}</strong></td>
                <td>{s['min']:.1f}&deg;</td>
                <td>{s['max']:.1f}&deg;</td>
                <td>{s['mean']:.1f}&deg;</td>
                <td>{s['range']:.1f}&deg;</td>
                <td>{std_val:.1f}&deg;</td>
            </tr>"""

    # ── Ideal Range Reference Table ─────────────────────────────────────────
    ref_rows = """
    <tr><th colspan="3" style="background:#e8f4fd;color:#1a5276;padding:8px 12px;">Lower Body</th></tr>
    <tr><td>Knee at BDC (max extension)</td><td>140&ndash;150&deg;</td>
        <td>Road bikes; Holmes method dynamic correction. Below 135&deg; = saddle too low; above 155&deg; = hyper-extension risk.</td></tr>
    <tr><td>Knee flexion at BDC</td><td>33&ndash;43&deg;</td>
        <td>180&deg; minus extension angle. Dynamic studies (Bini et al.); below 30&deg; risks under-loading, above 48&deg; may cause knee strain.</td></tr>
    <tr><td>Knee at TDC (max flexion)</td><td>65&ndash;80&deg;</td>
        <td>Below 60&deg; risks patellofemoral compression. Above 90&deg; may indicate saddle too high or poor hip mobility.</td></tr>
    <tr><td>Hip open angle</td><td>80&ndash;110&deg;</td>
        <td>Shoulder&ndash;hip&ndash;knee angle. Narrower for TT/triathlon (95&ndash;107&deg;). Below 55&deg; = excessive closure, discomfort &amp; power loss.</td></tr>
    <tr><td>Ankle angle (knee&ndash;ankle&ndash;foot)</td><td>85&ndash;110&deg;</td>
        <td>3-point angle at ankle. Measures actual ankle position. ROM through stroke ~20&ndash;40&deg; is normal. Above 120&deg; = excessive plantarflexion (ankling).</td></tr>
    <tr><th colspan="3" style="background:#e8f4fd;color:#1a5276;padding:8px 12px;">Upper Body</th></tr>
    <tr><td>Elbow angle</td><td>150&ndash;165&deg;</td>
        <td>Shoulder&ndash;elbow&ndash;wrist. Soft bend absorbs road vibration. Below 130&deg; = reach too long or bars too low. Above 175&deg; = nearly straight, reach too short.</td></tr>
    <tr><td>Upper arm angle (torso-relative)</td><td>75&ndash;105&deg;</td>
        <td>Supplement of the hip&ndash;shoulder&ndash;elbow angle, measured from the upper-trunk extension (toward head) to the upper arm. 90&deg; = arm perpendicular to torso; &lt;90&deg; = arm forward of perpendicular (typical road reach). Torso-independent, so valid across aero and upright positions. Below 75&deg; = excessive reach (bars too far/low); above 105&deg; = arm behind perpendicular (bars too close/high).</td></tr>
    <tr><td>Trunk lean (road/endurance)</td><td>30&ndash;45&deg;</td>
        <td>From vertical. Aero/TT: 15&ndash;30&deg;; recreational: 40&ndash;55&deg;. Consistent lean indicates good core stability.</td></tr>
    <tr><th colspan="3" style="background:#e8f4fd;color:#1a5276;padding:8px 12px;">Pedaling Quality</th></tr>
    <tr><td>Cadence</td><td>80&ndash;100 RPM</td>
        <td>Trained cyclists. Recreational: 60&ndash;80 RPM. High cadence reduces joint loading; low cadence increases muscular demand.</td></tr>
    <tr><td>Pedal Smoothness</td><td>15&ndash;35%</td>
        <td>Mean/peak angular velocity ratio. Below 15% = pronounced power dead spots. Above 35% = very smooth (elite track cyclists).</td></tr>
    <tr><td>Dead Spot Score</td><td>&ge;50/100</td>
        <td>Derived from mean dwell time across all TDC/BDC transitions, inverted (100 &minus; mean_dwell_pct &times; 2). Below 30 = significant dead spots.</td></tr>
    <tr><td>BDC dwell time</td><td>&lt;40 ms / &lt;5% of stroke</td>
        <td>Estimated from angular acceleration at the zero-crossing: dwell&nbsp;=&nbsp;2&nbsp;&times;&nbsp;threshold&nbsp;/&nbsp;|dv/dt|, where threshold = 20% of stroke peak velocity. Frame-rate independent. 40&ndash;80 ms = minor; 80&ndash;140 ms = moderate (consider ankling cue); &gt;140 ms or &gt;15% = severe (passive push-only stroke — focus on toe-down drive and hamstring pull-up).</td></tr>
    <tr><td>TDC dwell time</td><td>&lt;40 ms / &lt;5% of stroke</td>
        <td>Same method as BDC. 40&ndash;80 ms = minor; 80&ndash;140 ms = moderate (scrape-mud cue); &gt;140 ms or &gt;15% = severe (weak hip-flexor pull-through — actively lift knee at 12 o'clock). Ref: Sanderson 1991; Coyle et al.</td></tr>
    <tr><td>Accel. at crossing (deg/s&sup2;)</td><td>Higher is better</td>
        <td>Angular acceleration of the knee at the TDC/BDC zero-crossing, estimated by linear fit to signed velocity over &plusmn;3 frames. Directly measures drive aggressiveness through the dead zone; dwell time is derived from this value.</td></tr>
    <tr><td>Stroke Consistency (CV)</td><td>&lt;8% CV</td>
        <td>Coefficient of variation of stroke duration. Below 5% = very consistent; above 15% = fatigue or technique breakdown.</td></tr>"""

    # ── Ankle ROM ───────────────────────────────────────────────────────────
    ankle_html = ""
    for side in ["left", "right"]:
        ar = motion_metrics.get(f"{side}_ankle_rom")
        if ar:
            status_color = {"normal": "#2ecc71", "excessive_ankling": "#e74c3c",
                            "rigid_ankle": "#e67e22", "high_ankling": "#e67e22"}.get(
                            ar["status"], "#888")
            ankle_html += f"""
            <tr>
                <td><strong>{side.capitalize()} Ankle</strong></td>
                <td>{ar['rom']:.1f}&deg;</td>
                <td>{ar['mean_angle']:.1f}&deg;</td>
                <td><span style="color:{status_color};font-weight:600;">
                    {ar['status'].replace('_', ' ').title()}</span></td>
                <td>{ar['note']}</td>
            </tr>"""

    # ── Transition smoothness table (from dead_spot detail) ─────────────────
    def _transition_table(dead_spot: Dict) -> str:
        """Build a combined BDC+TDC per-transition dwell-duration table."""
        bdc_list = dead_spot.get("bdc_transitions", [])
        tdc_list = dead_spot.get("tdc_transitions", [])
        if not bdc_list and not tdc_list:
            return ""
        sev_color = {
            "smooth":   "#2ecc71",
            "minor":    "#f39c12",
            "moderate": "#e67e22",
            "severe":   "#e74c3c",
        }
        rows = ""
        # Interleave TDC/BDC in stroke order (TDC→BDC→TDC→BDC…)
        max_len = max(len(bdc_list), len(tdc_list))
        for i in range(max_len):
            for lst in (tdc_list, bdc_list):
                if i >= len(lst):
                    continue
                t = lst[i]
                col = sev_color.get(t["severity"], "#888")
                rows += (
                    f'<tr>'
                    f'<td><strong>{t["type"]} #{t["transition"]}</strong></td>'
                    f'<td>{t["stroke_peak_deg_s"]:.1f}</td>'
                    f'<td>{t["threshold_deg_s"]:.1f}</td>'
                    f'<td>{t["accel_deg_s2"]:.1f}</td>'
                    f'<td style="color:{col};font-weight:600">{t["dwell_ms"]:.1f} ms</td>'
                    f'<td>{t["stroke_dur_ms"]:.0f} ms</td>'
                    f'<td>{t["dwell_pct"]:.1f}%</td>'
                    f'<td style="color:{col};font-weight:600">{t["severity"].capitalize()}</td>'
                    f'</tr>'
                )
        if not rows:
            return ""
        mean_dwell_ms  = dead_spot.get("mean_dwell_ms")
        mean_dwell_pct = dead_spot.get("mean_dwell_pct")
        summary_row = ""
        mean_accel = dead_spot.get("mean_accel_deg_s2")
        if mean_dwell_ms is not None:
            sc = sev_color.get(
                "smooth"   if mean_dwell_ms <  40 else
                "minor"    if mean_dwell_ms <  80 else
                "moderate" if mean_dwell_ms < 140 else "severe", "#888")
            summary_row = (
                f'<tr style="background:#f8f9fa;font-weight:600;">'
                f'<td>Overall average</td>'
                f'<td>—</td><td>—</td>'
                f'<td>{mean_accel:.1f}</td>'
                f'<td style="color:{sc}">{mean_dwell_ms:.1f} ms</td>'
                f'<td>—</td>'
                f'<td style="color:{sc}">{mean_dwell_pct:.1f}%</td>'
                f'<td style="color:{sc}">—</td>'
                f'</tr>'
            )
        return (
            '<div class="table-wrap"><table style="margin-top:14px">'
            '<thead><tr>'
            '<th>Transition</th>'
            '<th>Stroke peak (deg/s)</th>'
            '<th>Dead-zone threshold (deg/s)</th>'
            '<th>Accel at crossing (deg/s²)</th>'
            '<th>Est. dwell time</th>'
            '<th>Stroke duration</th>'
            '<th>% of stroke</th>'
            '<th>Severity</th>'
            '</tr></thead>'
            f'<tbody>{rows}{summary_row}</tbody></table></div>'
        )

    # ── Pedal Stroke Analysis ────────────────────────────────────────────────
    pedal_html = ""
    if pedal_phases:
        phase_rows = ""
        phase_cards = ""
        for side, ph in pedal_phases.items():
            if not ph or not ph.get("phases_detected"):
                continue
            side_cap = side.capitalize()
            n = ph.get("num_strokes", 0)
            bdc_mean = ph.get("bdc_angle_mean")
            bdc_std  = ph.get("bdc_angle_std")
            tdc_mean = ph.get("tdc_angle_mean")
            tdc_std  = ph.get("tdc_angle_std")
            rom_mean = ph.get("rom_mean")
            rom_std  = ph.get("rom_std")
            flex_bdc = ph.get("knee_flexion_bdc_mean")
            flex_std = ph.get("knee_flexion_bdc_std")
            dur_mean = ph.get("stroke_duration_mean")
            dur_cv   = ph.get("stroke_duration_cv")

            # BDC/TDC status colours
            def _bdc_color(v):
                if v is None: return "#888"
                if 140 <= v <= 150: return "#2ecc71"
                if 135 <= v <= 155: return "#e67e22"
                return "#e74c3c"
            def _tdc_color(v):
                if v is None: return "#888"
                if 65 <= v <= 80: return "#2ecc71"
                if 60 <= v <= 90: return "#e67e22"
                return "#e74c3c"
            def _cv_color(v):
                if v is None: return "#888"
                if v < 8:  return "#2ecc71"
                if v < 15: return "#e67e22"
                return "#e74c3c"

            phase_cards += f"""
            <div class="metric-card" style="border-top:4px solid #3498db;">
                <div class="metric-value">{n}</div>
                <div class="metric-label">{side_cap} Strokes Detected</div>
            </div>"""
            if bdc_mean is not None:
                phase_cards += f"""
            <div class="metric-card" style="border-top:4px solid {_bdc_color(bdc_mean)};">
                <div class="metric-value">{bdc_mean:.1f}<span class="metric-unit">&deg;</span></div>
                <div class="metric-label">{side_cap} BDC (Max Extension)</div>
                <div class="metric-sub">&sigma; {bdc_std:.1f}&deg; &nbsp;|&nbsp; ideal 140&ndash;150&deg;</div>
            </div>"""
            if tdc_mean is not None:
                phase_cards += f"""
            <div class="metric-card" style="border-top:4px solid {_tdc_color(tdc_mean)};">
                <div class="metric-value">{tdc_mean:.1f}<span class="metric-unit">&deg;</span></div>
                <div class="metric-label">{side_cap} TDC (Max Flexion)</div>
                <div class="metric-sub">&sigma; {tdc_std:.1f}&deg; &nbsp;|&nbsp; ideal 65&ndash;80&deg;</div>
            </div>"""
            if rom_mean is not None:
                phase_cards += f"""
            <div class="metric-card" style="border-top:4px solid #9b59b6;">
                <div class="metric-value">{rom_mean:.1f}<span class="metric-unit">&deg;</span></div>
                <div class="metric-label">{side_cap} Stroke ROM</div>
                <div class="metric-sub">&sigma; {rom_std:.1f}&deg;</div>
            </div>"""

            # Per-stroke table rows
            strokes = ph.get("strokes", [])
            for i, s in enumerate(strokes):
                bdc = s.get('bdc_angle', 0)
                bdc_col = ("#2ecc71" if 140 <= bdc <= 150
                           else "#e67e22" if 135 <= bdc <= 155
                           else "#e74c3c")
                phase_rows += f"""
                <tr>
                    <td>{side_cap} #{i+1}</td>
                    <td style="color:{bdc_col};font-weight:600">{bdc:.1f}&deg;</td>
                    <td>{s.get('tdc_angle', 0):.1f}&deg;</td>
                    <td>{s.get('rom', 0):.1f}&deg;</td>
                    <td>{s.get('stroke_duration', 0):.2f}s</td>
                </tr>"""

            # Stroke-to-stroke consistency row
            if dur_mean is not None and dur_cv is not None:
                cadence_from_dur = round(60 / dur_mean, 1) if dur_mean > 0 else None
                phase_rows += f"""
                <tr style="background:#f8f9fa;font-style:italic;">
                    <td colspan="5"><strong>{side_cap} average</strong>: {dur_mean:.2f}s/stroke
                        &nbsp;&middot;&nbsp; CV {dur_cv:.1f}%
                        &nbsp;<span style="color:{_cv_color(dur_cv)};font-weight:700;">
                        {'Good' if dur_cv < 8 else 'Moderate' if dur_cv < 15 else 'Variable'}</span>
                        {f'&nbsp;&middot;&nbsp; estimated {cadence_from_dur} RPM' if cadence_from_dur else ''}
                    </td>
                    <td></td>
                </tr>"""

        if phase_cards or phase_rows:
            pedal_html = f"""
  <div class="section">
    <h2>Pedal Stroke Analysis</h2>
    <p>Per-stroke breakdown of knee angles at BDC (bottom dead centre, maximum extension) and TDC
    (top dead centre, maximum flexion). <strong>BDC angle</strong> is the primary saddle height indicator;
    <strong>knee flexion at BDC</strong> (180&deg; &minus; extension angle) should be 33&ndash;43&deg;
    for road cycling. Stroke-to-stroke CV below 8% indicates consistent technique.</p>
    <div class="metrics-grid">{phase_cards}</div>
    {'<div class="table-wrap"><table style="margin-top:20px"><thead><tr><th>Stroke</th><th>BDC Angle</th><th>TDC Angle</th><th>ROM</th><th>Duration</th></tr></thead><tbody>' + phase_rows + '</tbody></table></div>' if phase_rows else ''}
  </div>
  <div class="section">
    <h2>TDC / BDC Transition Smoothness</h2>
    <p>At each TDC and BDC the angular velocity of the knee is zero by definition — the crank
    is at the point where vertical leg force produces zero torque. What matters is how
    <strong>aggressively the rider drives through that zero point</strong>.
    <strong>Accel at crossing</strong> (deg/s²) is the angular acceleration of the knee at
    the exact transition — the slope of the velocity curve at zero — estimated by a
    linear fit over the surrounding frames. Higher acceleration means a quicker, more
    forceful reversal through the dead zone.
    <strong>Est. dwell time</strong> is derived from this acceleration as
    <em>2 &times; threshold / |accel|</em>, where the threshold is 20% of that stroke's peak
    velocity. This is a continuous, frame-rate-independent estimate — not a frame count —
    so it is valid at 30 fps and above.
    A skilled rider passes through each dead zone in under 40 ms; longer dwell times indicate
    passive coasting. Low dwell at TDC requires active hip-flexor pull-through; low dwell at
    BDC requires toe-down drive or ankling through the bottom of the stroke.</p>
    {_transition_table(motion_metrics.get("dead_spot", {}))}
  </div>"""

    # ── Shared icon/color maps ────────────────────────────────────────────
    rec_icons = {"warning": "&#x26A0;&#xFE0F;", "info": "&#x2139;&#xFE0F;",
                 "success": "&#x2705;"}
    rec_colors = {"warning": "#fff3cd", "info": "#d1ecf1", "success": "#d4edda"}

    # ── Frontal Analysis Section ───────────────────────────────────────────
    frontal_html = ""
    if frontal_analysis:
        frontal_score = frontal_analysis.get("frontal_score")
        frontal_sym = frontal_analysis.get("frontal_symmetry")

        # Frontal metric cards
        frontal_metrics = ""
        frontal_metrics += _metric_card("Frontal Score", frontal_score, "/100",
                                         "#2ecc71" if (frontal_score or 0) >= 70 else "#e67e22")
        if frontal_sym is not None:
            frontal_metrics += _metric_card("Frontal Symmetry", frontal_sym, "%",
                                             "#2ecc71" if frontal_sym >= 75 else "#e67e22")

        for side in ["left", "right"]:
            sd = frontal_analysis.get(side)
            if sd:
                cls = sd["classification"]
                cls_color = {"neutral": "#2ecc71", "valgus": "#e67e22", "varus": "#e67e22"}.get(cls, "#888")
                frontal_metrics += _metric_card(
                    f"{side.capitalize()} Knee", sd["deviation_pct_mean"], "%",
                    cls_color, subtitle=cls.capitalize()
                )
                frontal_metrics += _metric_card(
                    f"{side.capitalize()} Tracking", sd["tracking_score"], "/100",
                    "#2ecc71" if sd["tracking_score"] >= 60 else "#e67e22"
                )

        # Frontal detail table
        frontal_rows = ""
        for side in ["left", "right"]:
            sd = frontal_analysis.get(side)
            if sd:
                cls_badge = {"neutral": "green", "valgus": "orange", "varus": "orange"}.get(
                    sd["classification"], "grey")
                frontal_rows += f"""
                <tr>
                    <td><strong>{side.capitalize()}</strong></td>
                    <td>{sd['deviation_pct_mean']:.1f}%</td>
                    <td>{sd['deviation_pct_max']:.1f}%</td>
                    <td>{sd['deviation_pct_std']:.1f}%</td>
                    <td><span class="badge {cls_badge}">{sd['classification'].capitalize()}</span></td>
                    <td>{sd['tracking_score']:.0f}/100</td>
                    <td>{sd['frontal_angle_mean']:.1f}&deg;</td>
                </tr>"""

        # Frontal recommendations
        frontal_recs_html = ""
        for rec in frontal_analysis.get("frontal_recommendations", []):
            rec_type = rec.get("type", "info")
            icon = rec_icons.get(rec_type, "&bull;")
            bg = rec_colors.get(rec_type, "#f8f9fa")
            ref = rec.get("reference", "")
            ref_html_r = f'<div class="rec-ref">Ref: {ref}</div>' if ref else ""
            frontal_recs_html += f"""
            <div class="rec-card" style="background:{bg};">
                <div class="rec-header">{icon} <strong>{rec['joint']}</strong> &mdash; {rec['metric']}
                    <span class="rec-value">{rec['value']}</span></div>
                <div class="rec-body">{rec['suggestion']}</div>
                {ref_html_r}
            </div>"""

        frontal_html = f"""
  <div class="section">
    <h2>Frontal Knee Analysis</h2>
    <p>Knee tracking and alignment assessed from the front-view video. Evaluates valgus/varus deviation,
    lateral tracking stability, and hip-knee-ankle alignment in the frontal plane.</p>
    <div class="metrics-grid">{frontal_metrics}</div>
  </div>

  <div class="section">
    <h2>Frontal Knee Details</h2>
    <div class="table-wrap"><table>
      <thead>
        <tr><th>Side</th><th>Mean Dev</th><th>Max Dev</th><th>Std Dev</th>
            <th>Classification</th><th>Tracking</th><th>Frontal Angle</th></tr>
      </thead>
      <tbody>{frontal_rows}</tbody>
    </table></div>
  </div>

  <div class="section">
    <h2>Frontal Plane Recommendations</h2>
    <p>Specific recommendations based on frontal knee dynamics analysis.</p>
    {frontal_recs_html}
  </div>"""

    # ── Recommendations ─────────────────────────────────────────────────────
    recs = motion_metrics.get("recommendations", [])

    recs_html = ""
    for rec in recs:
        rec_type = rec.get("type", "info")
        icon = rec_icons.get(rec_type, "&bull;")
        bg = rec_colors.get(rec_type, "#f8f9fa")
        ref = rec.get("reference", "")
        ref_html = f'<div class="rec-ref">Ref: {ref}</div>' if ref else ""
        recs_html += f"""
        <div class="rec-card" style="background:{bg};">
            <div class="rec-header">{icon} <strong>{rec['joint']}</strong> &mdash; {rec['metric']}
                <span class="rec-value">{rec['value']}</span></div>
            <div class="rec-body">{rec['suggestion']}</div>
            {ref_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Bike Fit Report &mdash; {video_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #f0f2f5; color: #2c3e50; line-height: 1.5; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e);
             color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; }}
  .header .subtitle {{ opacity: 0.7; margin-top: 6px; font-size: 14px; }}
  .header .meta {{ margin-top: 12px; font-size: 13px; opacity: 0.6; }}
  .container {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
  .section {{ background: white; border-radius: 12px; padding: 24px;
              box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 24px; }}
  .section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 18px;
                 padding-bottom: 10px; border-bottom: 2px solid #f0f2f5; color: #1a1a2e; }}
  .section p {{ font-size: 13px; color: #666; margin-bottom: 14px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                   gap: 14px; }}
  .metric-card {{ background: #fafbfc; border-radius: 10px; padding: 16px 14px;
                  text-align: center; border: 1px solid #e8ecef; }}
  .metric-value {{ font-size: 28px; font-weight: 700; color: #1a1a2e; }}
  .metric-unit {{ font-size: 13px; font-weight: 400; color: #888; }}
  .metric-label {{ margin-top: 4px; font-size: 11px; color: #888; text-transform: uppercase;
                   letter-spacing: 0.5px; }}
  .metric-sub {{ font-size: 10px; color: #aaa; margin-top: 2px; }}
  .chart-img {{ width: 100%; border-radius: 8px; border: 1px solid #e8ecef; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #f0f2f5; padding: 10px 12px; text-align: left; font-size: 11px;
        color: #555; text-transform: uppercase; letter-spacing: 0.4px; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #f0f2f5; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  .rec-card {{ border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
               border: 1px solid rgba(0,0,0,0.08); }}
  .rec-header {{ font-size: 14px; display: flex; align-items: center; gap: 8px;
                 flex-wrap: wrap; }}
  .rec-value {{ margin-left: auto; font-family: monospace; color: #555; font-size: 12px; }}
  .rec-body {{ margin-top: 6px; font-size: 13px; color: #555; padding-left: 24px; }}
  .rec-ref {{ margin-top: 4px; font-size: 11px; color: #aaa; padding-left: 24px;
              font-style: italic; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 13px; font-weight: 600; color: white; }}
  .badge.green {{ background: #2ecc71; }}
  .badge.orange {{ background: #e67e22; }}
  .badge.red {{ background: #e74c3c; }}
  .badge.grey {{ background: #aaa; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .footer {{ text-align: center; padding: 24px; color: #aaa; font-size: 12px; }}

  /* ── Print button ───────────────────────────────────────────────────────── */
  .print-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 18px; background: rgba(255,255,255,0.15); color: white;
    border: 1px solid rgba(255,255,255,0.35); border-radius: 8px;
    cursor: pointer; font-size: 13px; font-weight: 600;
    text-decoration: none; margin-top: 14px; transition: background 0.15s;
  }}
  .print-btn:hover {{ background: rgba(255,255,255,0.25); }}

  /* ── Responsive tables ──────────────────────────────────────────────────── */
  .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}

  /* ── Mobile ─────────────────────────────────────────────────────────────── */
  @media (max-width: 640px) {{
    .header {{ padding: 20px 16px; }}
    .header h1 {{ font-size: 20px; }}
    .container {{ padding: 0 12px; margin: 16px auto; }}
    .section {{ padding: 16px; }}
    .metrics-grid {{ grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; }}
    .metric-value {{ font-size: 22px; }}
    .rec-value {{ display: none; }}
    table {{ font-size: 12px; }}
    th, td {{ padding: 8px; }}
  }}

  /* ── Print / PDF ─────────────────────────────────────────────────────────── */
  @media print {{
    body {{ background: white; }}
    .header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .print-btn {{ display: none; }}
    .section {{ box-shadow: none; border: 1px solid #e0e0e0; break-inside: avoid; }}
    .metrics-grid {{ grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }}
    .table-wrap {{ overflow: visible; }}
    a {{ color: inherit; text-decoration: none; }}
    .container {{ padding: 0 8px; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>&#x1F6B4; Bike Fit Analysis Report</h1>
  <div class="subtitle">{video_name}</div>
  <div class="meta">Generated {now} &nbsp;&middot;&nbsp;
    Duration: {duration:.1f}s &nbsp;&middot;&nbsp;
    {fps:.0f} FPS &nbsp;&middot;&nbsp;
    {total_frames} frames &nbsp;&middot;&nbsp;
    Backend: {backend}</div>
  <button class="print-btn" onclick="window.print()">&#x1F4E5; Download PDF</button>
</div>

<div class="container">

  <div class="section">
    <h2>Key Metrics</h2>
    <p>Composite scores combining pedal smoothness, dead spot analysis, stroke consistency,
    joint symmetry, and trunk stability. Based on cycling biomechanics research.</p>
    <div class="metrics-grid">{metrics_html}</div>
  </div>

  {f'<div class="section"><h2>Joint Angle Chart</h2>{chart_html}</div>' if chart_html else ''}

  <div class="section">
    <h2>Joint Angle Summary</h2>
    <p>Angles measured dynamically during pedaling. Note: dynamic angles differ from static
    measurements by approximately 8&deg; for knee, 5&deg; for hip, and 9&deg; for ankle.</p>
    <div class="table-wrap"><table>
      <thead>
        <tr><th>Joint</th><th>Min</th><th>Max</th><th>Mean</th><th>Range</th><th>Std Dev</th></tr>
      </thead>
      <tbody>{angle_rows}</tbody>
    </table></div>
  </div>

  <div class="section">
    <h2>Ideal Range Reference</h2>
    <p>Evidence-based target ranges from cycling biomechanics literature.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Parameter</th><th>Ideal Range</th><th>Source</th></tr></thead>
      <tbody>{ref_rows}</tbody>
    </table></div>
  </div>

  {'<div class="section"><h2>Ankle ROM Analysis</h2><p>Total ankle range of motion per stroke. Typical is ~50 deg. Excessive ankling (&gt;60 deg) may indicate cleat issues or foot instability.</p><div class="table-wrap"><table><thead><tr><th>Side</th><th>ROM</th><th>Mean Angle</th><th>Status</th><th>Note</th></tr></thead><tbody>' + ankle_html + '</tbody></table></div></div>' if ankle_html else ''}

  {pedal_html}

  {frontal_html}

  <div class="section">
    <h2>Fitting Recommendations</h2>
    <p>Evidence-based suggestions derived from your measured angles and pedaling metrics.
    Recommendations reference published bike fitting guidelines and biomechanics research.</p>
    {recs_html}
  </div>

</div>
<div class="footer">Bike Fitting App &nbsp;&middot;&nbsp; Powered by MediaPipe &amp; OpenCV
&nbsp;&middot;&nbsp; Research references: Holmes et al., Bini et al., Garmin Cycling Dynamics, BikeDynamics</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
