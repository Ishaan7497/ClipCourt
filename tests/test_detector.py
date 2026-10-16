"""
Synthetic trajectory test harness for the ClipCourt detection logic.

No video, no YOLO — we synthesize ball detections frame-by-frame and assert
that BallTracker + HoopShotDetector classify them correctly. This pins down
the regressions the code review fixed:

  * a clean swish is MADE
  * a front-of-rim brick that drops below the rim is a MISS
  * an occlusion right at the rim (net blocks YOLO) still resolves as MADE
  * velocity does not explode across a detection gap
  * two quick shots don't double-trigger inside the cooldown

Run:  python -m pytest tests/ -q     (or just: python tests/test_detector.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from engine import (BallTracker, EngineConfig, HoopShotDetector,
                    _rim_candidates, approx_shot_location_ft, device_label,
                    resolve_device, vertical_crop_window)

FRAME_W, FRAME_H = 1280, 720
FPS = 30.0
HOOP = (600, 200, 80, 30)  # x, y, w, h


def run_trajectory(points, cfg=None):
    """points: list of (cx, cy) or None (occluded frame). Returns events."""
    cfg = cfg or EngineConfig()
    tracker = BallTracker(FRAME_W, cfg)
    detector = HoopShotDetector(HOOP, FPS, cfg)
    events = []
    total = len(points) + int(cfg.make_confirm_seconds * FPS) + 5
    for i in range(total):
        p = points[i] if i < len(points) else None
        dets = [(p[0], p[1], 24, 24, 0.8)] if p else []
        pt, kind = tracker.update(dets, i)
        ev = detector.update(pt, kind, i)
        if ev:
            events.append(ev)
    return events


def arc(x0, y0, x1, y1, n):
    """Simple ballistic-looking descent from (x0,y0) to (x1,y1) over n frames."""
    pts = []
    for i in range(n):
        t = i / max(1, n - 1)
        pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * (t ** 1.6)))
    return pts


def test_clean_swish_is_made():
    rim_cx = HOOP[0] + HOOP[2] // 2
    pts = arc(rim_cx - 30, 60, 640, 320, 24)  # descends through band to below rim
    events = run_trajectory(pts)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].result == "made", events[0]
    assert events[0].ball_x is not None and events[0].ball_y is not None, events[0]


def test_front_rim_brick_is_miss():
    # Ball descends, stops ABOVE the rim band's left edge, bounces up-left,
    # then falls below rim level well outside the rim — a classic brick.
    down = arc(560, 60, 590, 195, 14)              # descends to just above rim
    bounce = arc(590, 195, 470, 140, 8)            # deflects up and away
    fall = arc(470, 140, 430, 380, 10)             # drops below rim, outside band
    events = run_trajectory(down + bounce + fall)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].result == "miss", events[0]


def test_occluded_swish_still_made():
    # Ball passes through the band, is occluded by the net for a few frames,
    # then reappears below the rim. The measured in-band sightings before the
    # occlusion carry the make.
    rim_cx = 640
    down = arc(rim_cx - 20, 60, rim_cx, 225, 20)   # ends INSIDE the rim band
    occluded = [None] * 4
    below = arc(rim_cx, 260, rim_cx + 5, 400, 6)
    events = run_trajectory(down + occluded + below)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].result == "made", events[0]


def test_velocity_not_inflated_across_gap():
    cfg = EngineConfig()
    tracker = BallTracker(FRAME_W, cfg)
    tracker.update([(100, 100, 24, 24, 0.9)], 0)
    # 5-frame gap, ball moved 50px total => 10 px/frame, NOT 50.
    tracker.update([(150, 100, 24, 24, 0.9)], 5)
    assert abs(tracker.velocity[0] - 10.0) < 1e-6, tracker.velocity


def test_velocity_after_coasting_uses_the_last_real_detection():
    # Moving +10 px/frame, then two occluded (coasted) frames. When the ball
    # is seen again the velocity must still read 10, not be shrunk by
    # measuring from the coasted prediction over the full gap.
    tracker = BallTracker(FRAME_W, EngineConfig())
    for f in range(5):
        tracker.update([(100 + 10 * f, 100, 24, 24, 0.9)], f)
    assert tracker.update([], 5)[1] == "coasted"
    assert tracker.update([], 6)[1] == "coasted"
    tracker.update([(170, 100, 24, 24, 0.9)], 7)
    assert abs(tracker.velocity[0] - 10.0) < 1e-6, tracker.velocity


def test_reacquired_track_is_not_smoothed_toward_the_stale_position():
    # Lost for longer than track_lost_frames, then seen across the frame: the
    # point reported must be where the ball IS, not an EMA blend with where it
    # was — that blend invents a ball crossing the frame (and the rim).
    cfg = EngineConfig()
    tracker = BallTracker(FRAME_W, cfg)
    tracker.update([(100, 100, 24, 24, 0.9)], 0)
    gap = cfg.track_lost_frames + 4
    for f in range(1, gap):
        tracker.update([], f)
    point, kind = tracker.update([(1100, 600, 24, 24, 0.9)], gap)
    assert kind == "measured"
    assert point == (1100, 600), point
    assert tracker.velocity == (0.0, 0.0), tracker.velocity


def test_cut_clips_stops_when_cancelled():
    import tempfile
    from engine import AnalysisResult, ShotEvent, cut_clips
    analysis = AnalysisResult(fps=FPS, frame_count=300, width=FRAME_W,
                              height=FRAME_H, duration_s=10.0)
    analysis.events = [ShotEvent(frame=90, time_s=3.0, result="made",
                                 confidence="high")]
    with tempfile.TemporaryDirectory() as out_dir:
        clips = cut_clips("/nonexistent.mp4", analysis, "all", out_dir,
                          EngineConfig(), should_cancel=lambda: True)
        assert clips == [], clips
        assert not [n for n in os.listdir(out_dir) if n.endswith(".mp4")]


def test_track_lost_resolution_uses_last_known_position():
    # Ball descends into the zone, then the track is lost entirely — the
    # detector resolves on timeout via the point-is-None branch, which must
    # fall back to the last measured position rather than emit None coords.
    rim_cx = 640
    down = arc(rim_cx - 20, 60, rim_cx, 195, 16)  # armed, still above the rim
    events = run_trajectory(down)                  # harness pads with None frames
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].confidence == "medium", events[0]
    assert events[0].ball_x is not None and events[0].ball_y is not None, events[0]


def test_approx_shot_location_scale():
    # HOOP is 80 px wide = 1.5 ft rim, so 53.33 px/ft; rim center x = 640.
    dx, dist = approx_shot_location_ft(640, HOOP[1] + HOOP[3], HOOP)
    assert abs(dx) < 1e-6 and abs(dist) < 1e-6, (dx, dist)
    dx, dist = approx_shot_location_ft(640 + 80 / 1.5, 230 + 80 / 1.5, HOOP)
    assert abs(dx - 1.0) < 0.01, dx
    assert abs(dist - 1.0) < 0.01, dist
    # Degenerate rim box must not divide by zero.
    assert approx_shot_location_ft(100, 100, (0, 0, 0, 0)) == (0.0, 0.0)


def test_cooldown_suppresses_rebound_double_trigger():
    rim_cx = 640
    shot = arc(rim_cx - 30, 60, rim_cx, 320, 22)
    # Immediate rebound bounce back up through the zone and down again,
    # all inside the 2-second cooldown window.
    rebound = arc(rim_cx, 320, rim_cx - 10, 120, 10) + arc(rim_cx - 10, 120, rim_cx, 320, 10)
    events = run_trajectory(shot + rebound)
    assert len(events) == 1, f"cooldown failed: got {len(events)} events"


def test_vertical_crop_is_nine_sixteenths_and_inside_the_frame():
    # Landscape source: keep full height, crop width to 9:16.
    w, h, x, y = vertical_crop_window(1920, 1080, center_x=960)
    assert (w, h) == (608, 1080), (w, h)   # 1080 * 9/16 = 607.5, rounded even
    assert abs((w / h) - 9 / 16) < 0.01
    assert x == (1920 - w) // 2 and y == 0

    # Centering on the action, clamped so the window stays in frame.
    _w, _h, x_left, _y = vertical_crop_window(1920, 1080, center_x=0)
    _w, _h, x_right, _y = vertical_crop_window(1920, 1080, center_x=1920)
    assert x_left == 0, x_left
    assert x_right == 1920 - _w, x_right

    # Already-portrait source must not be cropped to nothing.
    w, h, x, y = vertical_crop_window(1080, 1920, center_x=540)
    assert (w, h) == (1080, 1920) and (x, y) == (0, 0)

    # Odd dimensions get rounded down to even (libx264 + yuv420p).
    w, h, _x, _y = vertical_crop_window(1921, 1081, center_x=500)
    assert w % 2 == 0 and h % 2 == 0, (w, h)


def test_rim_candidate_prefers_the_hollow_wide_rim_over_the_ball():
    # An orange ring (rim) at the top and a solid orange disc (ball) below it.
    frame = np.zeros((480, 640, 3), np.uint8)
    orange = (20, 110, 230)  # BGR
    cv2.ellipse(frame, (300, 120), (45, 12), 0, 0, 360, orange, 3)  # hollow, wide
    cv2.circle(frame, (300, 300), 22, orange, -1)                   # solid, round

    cands = _rim_candidates(frame)
    assert cands, "expected at least the rim to be found"
    _score, x, y, w, h = cands[0]
    assert w / h > 1.2, f"best candidate should be wide, got {w}x{h}"
    assert y < 200, f"best candidate should be the upper ring, got y={y}"


def test_device_resolution_is_a_known_backend():
    dev = resolve_device()
    assert dev in ("cuda", "mps", "cpu"), dev
    assert device_label(dev) in ("NVIDIA GPU", "Apple GPU", "CPU")
    assert resolve_device() == dev, "device choice should be cached"


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
