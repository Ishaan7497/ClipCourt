"""
HoopClip export — CSV and PDF coaching reports.

Pure functions over already-assembled job/roster dicts (no disk I/O, no
FastAPI coupling). Keeps server.py free of report-layout code and keeps
engine.py free of matplotlib/reportlab imports it doesn't otherwise need.

matplotlib and reportlab are imported lazily inside the functions that need
them (same rationale as engine._load_yolo): the CSV path works and the
server boots even if the PDF dependencies are missing.
"""

from __future__ import annotations

import csv
import io

DRILL_LABELS = {"layups": "Layups", "three_pointers": "3-Pointers",
                "midrange": "Mid-range"}

# Mirrors static/app.css :root — matplotlib can't read CSS variables, so the
# film-room palette is duplicated here. Keep in sync if the theme changes.
_COURT = "#131f1a"
_COURT_SUNKEN = "#0d1611"
_LINE = "#2e4437"
_CHALK = "#efeae0"
_CHALK_DIM = "#aab5ac"
_MAKE = "#7dd09a"
_MISS = "#e2604d"


def player_name(player_id, players_by_id) -> str:
    if not player_id:
        return "Unassigned"
    p = players_by_id.get(player_id)
    return p["name"] if p else "(removed player)"


def build_csv(events: list, players_by_id: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["player_name", "result", "time_s",
                "shot_x_ft", "shot_dist_ft", "confidence", "drill_tag"])
    for e in events:
        w.writerow([
            player_name(e.get("player_id"), players_by_id),
            e["result"],
            f"{e['time_s']:.2f}",
            "" if e.get("court_x_ft") is None else e["court_x_ft"],
            "" if e.get("court_dist_ft") is None else e["court_dist_ft"],
            e["confidence"],
            DRILL_LABELS.get(e.get("drill_tag"), ""),
        ])
    return buf.getvalue()


def _court_chart_png(events: list, players_by_id: dict) -> bytes:
    """Half-court scatter for the PDF. Uses the object-oriented Figure API,
    NOT pyplot — pyplot's global figure registry is not thread-safe and this
    server handles requests on a thread pool."""
    try:
        from matplotlib.figure import Figure
        from matplotlib.patches import Arc, Circle, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Run: pip install matplotlib"
        ) from exc

    fig = Figure(figsize=(5.0, 4.7), dpi=150, facecolor=_COURT)
    ax = fig.add_subplot(111, facecolor=_COURT_SUNKEN)

    # Half court in feet: 50 wide x 47 deep, baseline at the top (y=0),
    # hoop center 5.25 ft off the baseline. Matches the on-page SVG.
    hoop_y = 5.25
    ax.add_patch(Rectangle((0, 0), 50, 47, fill=False, edgecolor=_LINE, lw=1.5))
    ax.add_patch(Rectangle((17, 0), 16, 19, fill=False, edgecolor=_LINE, lw=1))
    ax.add_patch(Circle((25, 19), 6, fill=False, edgecolor=_LINE, lw=1))
    ax.add_patch(Circle((25, hoop_y), 0.75, fill=False, edgecolor=_CHALK_DIM, lw=1.2))
    ax.plot([22, 28], [hoop_y - 1, hoop_y - 1], color=_CHALK_DIM, lw=1.2)  # backboard
    # 3-point line: corner segments + arc (r = 23.75 around the hoop).
    ax.plot([3, 3], [0, 14.2], color=_LINE, lw=1)
    ax.plot([47, 47], [0, 14.2], color=_LINE, lw=1)
    # Endpoints meet the corner lines at (3, 14.2)/(47, 14.2): +/-22 ft
    # laterally, sqrt(23.75^2 - 22^2) ~= 8.95 ft past the hoop.
    ax.add_patch(Arc((25, hoop_y), 47.5, 47.5, theta1=22.1, theta2=157.9,
                     edgecolor=_LINE, lw=1))

    for e in events:
        x, d = e.get("court_x_ft"), e.get("court_dist_ft")
        if x is None or d is None:
            continue
        px = min(49, max(1, 25 + x))
        py = min(46, max(0.5, hoop_y + d))
        p = players_by_id.get(e.get("player_id"))
        edge = p["color"] if p and p.get("color") else _COURT
        color = _MAKE if e["result"] == "made" else _MISS
        ax.scatter([px], [py], s=80, color=color, edgecolors=edge,
                   linewidths=1.6, zorder=3)

    ax.set_xlim(-1, 51)
    ax.set_ylim(48, -1)  # y inverted: baseline at the top, like the on-page SVG
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _player_stat_rows(events: list, players_by_id: dict) -> list:
    """[(name, attempts, made, missed, pct_str), ...] — players first,
    Unassigned (if any) next, Team total last."""
    groups: dict = {}
    for e in events:
        groups.setdefault(e.get("player_id"), []).append(e)

    def row(name, evs):
        made = sum(1 for e in evs if e["result"] == "made")
        att = len(evs)
        pct = f"{round(100 * made / att)}%" if att else "—"
        return [name, str(att), str(made), str(att - made), pct]

    rows = [row(player_name(pid, players_by_id), evs)
            for pid, evs in sorted(groups.items(),
                                   key=lambda kv: (kv[0] is None, str(kv[0])))
            if pid is not None]
    if None in groups:
        rows.append(row("Unassigned", groups[None]))
    rows.append(row("Team", events))
    return rows


def build_pdf(job_snapshot: dict, players_by_id: dict,
              video_display_name: str) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is not installed. Run: pip install reportlab"
        ) from exc

    events = job_snapshot["events"]
    made = sum(1 for e in events if e["result"] == "made")
    styles = getSampleStyleSheet()

    header_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a23")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_CHALK)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_LINE)),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

    story = [
        Paragraph(f"HoopClip Report — {video_display_name}", styles["Title"]),
        Paragraph(
            f"{len(events)} shot{'s' if len(events) != 1 else ''} — "
            f"{made} made, {len(events) - made} missed. "
            "Shot locations are approximate (single camera angle, "
            "rim-width scale).", styles["Normal"]),
        Spacer(1, 12),
        Image(io.BytesIO(_court_chart_png(events, players_by_id)),
              width=4.5 * inch, height=4.23 * inch),
        Spacer(1, 16),
        Paragraph("Player stats", styles["Heading2"]),
    ]

    stat_table = Table(
        [["Player", "Shots", "Made", "Missed", "FG%"]]
        + _player_stat_rows(events, players_by_id),
        repeatRows=1, hAlign="LEFT")
    stat_table.setStyle(header_style)
    story += [stat_table, Spacer(1, 16),
              Paragraph("Shot log", styles["Heading2"])]

    log_rows = [["#", "Player", "Result", "Time", "Drill", "Confidence"]]
    for i, e in enumerate(events, 1):
        log_rows.append([
            str(i),
            player_name(e.get("player_id"), players_by_id),
            e["result"].capitalize(),
            f"{int(e['time_s'] // 60)}:{int(e['time_s'] % 60):02d}",
            DRILL_LABELS.get(e.get("drill_tag"), "—"),
            e["confidence"].capitalize(),
        ])
    log_table = Table(log_rows, repeatRows=1, hAlign="LEFT")
    log_table.setStyle(header_style)
    story.append(log_table)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                            title="HoopClip coaching report")
    doc.build(story)
    buf.seek(0)
    return buf.read()
