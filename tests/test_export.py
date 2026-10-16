"""
Tests for the CSV / PDF coaching reports in export.py.

The PDF path needs matplotlib and reportlab; if either is missing those
checks report SKIP instead of failing, so the suite stays runnable on a
machine that only installed the core requirements.

Run:  python tests/test_export.py     (or: python -m pytest tests/ -q)
"""

import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import export

PLAYERS = {
    "p_1": {"id": "p_1", "name": "Ama", "color": "#f2c14e"},
    "p_2": {"id": "p_2", "name": "Bo", "color": "#8ecbff"},
}


def events():
    return [
        {"frame": 10, "time_s": 12.5, "result": "made", "confidence": "high",
         "court_x_ft": -3.2, "court_dist_ft": 14.0,
         "player_id": "p_1", "drill_tag": "midrange"},
        {"frame": 90, "time_s": 61.0, "result": "miss", "confidence": "medium",
         "court_x_ft": 1.1, "court_dist_ft": 4.0,
         "player_id": "p_1", "drill_tag": "layups"},
        {"frame": 150, "time_s": 95.25, "result": "made", "confidence": "high",
         "court_x_ft": None, "court_dist_ft": None,
         "player_id": None, "drill_tag": None},
        {"frame": 200, "time_s": 130.0, "result": "made", "confidence": "high",
         "court_x_ft": 0.0, "court_dist_ft": 22.0,
         "player_id": "p_gone", "drill_tag": "three_pointers"},
    ]


def test_csv_has_a_row_per_shot_with_resolved_names():
    rows = list(csv.reader(io.StringIO(export.build_csv(events(), PLAYERS))))
    header, body = rows[0], rows[1:]
    assert header[0] == "player_name" and header[-1] == "drill_tag"
    assert len(body) == 4, body
    assert body[0][:2] == ["Ama", "made"]
    assert body[0][6] == "Mid-range", body[0]
    # Unassigned shots and shots with no location must not break the export.
    assert body[2][0] == "Unassigned"
    assert body[2][3] == "" and body[2][4] == "", body[2]
    # A player deleted from the roster after tagging stays traceable.
    assert body[3][0] == "(removed player)", body[3]


def test_csv_escapes_a_name_containing_a_comma():
    players = {"p_1": {"id": "p_1", "name": 'Smith, Jr "AJ"', "color": "#fff"}}
    evs = [{"frame": 1, "time_s": 1.0, "result": "made", "confidence": "high",
            "court_x_ft": 0.0, "court_dist_ft": 1.0,
            "player_id": "p_1", "drill_tag": None}]
    rows = list(csv.reader(io.StringIO(export.build_csv(evs, players))))
    assert rows[1][0] == 'Smith, Jr "AJ"', rows[1]


def test_player_stat_rows_group_and_total_correctly():
    rows = export._player_stat_rows(events(), PLAYERS)
    by_name = {r[0]: r[1:] for r in rows}
    assert by_name["Ama"] == ["2", "1", "1", "50%"], by_name["Ama"]
    assert by_name["(removed player)"] == ["1", "1", "0", "100%"]
    assert by_name["Unassigned"] == ["1", "1", "0", "100%"]
    # Team is last and counts every shot.
    assert rows[-1][0] == "Team"
    assert rows[-1][1:] == ["4", "3", "1", "75%"]


def test_player_name_fallbacks():
    assert export.player_name(None, PLAYERS) == "Unassigned"
    assert export.player_name("p_2", PLAYERS) == "Bo"
    assert export.player_name("nope", PLAYERS) == "(removed player)"


def test_pdf_is_a_pdf():
    snapshot = {"events": events()}
    try:
        pdf = export.build_pdf(snapshot, PLAYERS, "practice.mp4")
    except RuntimeError as exc:
        print(f"SKIP  test_pdf_is_a_pdf: {exc}")
        return
    assert pdf.startswith(b"%PDF-"), pdf[:8]
    assert len(pdf) > 5000, f"suspiciously small PDF: {len(pdf)} bytes"


def test_pdf_title_survives_markup_in_the_video_name():
    """reportlab parses Paragraph text as markup; a filename is not markup."""
    try:
        pdf = export.build_pdf({"events": events()}, PLAYERS, "<b>final & <br>.mp4")
    except RuntimeError as exc:
        print(f"SKIP  test_pdf_title_survives_markup_in_the_video_name: {exc}")
        return
    assert pdf.startswith(b"%PDF-")


def test_pdf_with_no_located_shots_still_renders():
    """Every shot missing coordinates must not break the court chart."""
    evs = [{"frame": 1, "time_s": 1.0, "result": "miss", "confidence": "medium",
            "court_x_ft": None, "court_dist_ft": None,
            "player_id": None, "drill_tag": None}]
    try:
        pdf = export.build_pdf({"events": evs}, PLAYERS, "empty.mp4")
    except RuntimeError as exc:
        print(f"SKIP  test_pdf_with_no_located_shots_still_renders: {exc}")
        return
    assert pdf.startswith(b"%PDF-")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
