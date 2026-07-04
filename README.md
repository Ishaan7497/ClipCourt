# ClipCourt

Turn **any** basketball footage into highlight clips. Upload a video, draw one box around the rim in your browser, and get made/miss-classified clips back — with the original audio, cut frame-accurately.

The structural difference from apps like Hooper: nothing has to be recorded inside an app. Old camcorder tape, phone video, a league's tripod footage — if it plays, HoopClip can process it.

## Quick start

```bash
pip install -r requirements.txt
uvicorn server:app --reload
# open http://127.0.0.1:8000
```

The first job downloads the YOLOv8s weights (~22 MB) automatically. `ffmpeg` on the PATH is strongly recommended — clips keep their audio and cut faster with it (there is a no-audio OpenCV fallback without it).

## How it works

```
browser                          server                         engine
─────────                        ──────                         ──────
upload video ──────────────────▶ POST /api/videos ────────────▶ probe + validate
scrub & draw rim box ──────────▶ GET  /videos/{id}/frame        extract JPEG frame
"Find my highlights" ──────────▶ POST /api/jobs ──────────────▶ worker thread:
poll every second ─────────────▶ GET  /api/jobs/{id}              YOLO → BallTracker
                                                                  → HoopShotDetector
watch clips / download zip ────▶ GET  /jobs/{id}/clips[.zip] ◀─── ffmpeg clip pass
```

**Detection pipeline** (in `engine.py`):

1. **YOLO** finds sports-ball candidates each frame (low confidence threshold to catch the blurry ball at the rim; implausibly large boxes rejected).
2. **BallTracker** picks the candidate that best continues the trajectory (confidence minus distance penalty), EMA-smooths positions, and *coasts* through short occlusions with a velocity computed over the real frame gap.
3. **HoopShotDetector** arms when a descending ball enters the attempt zone, and calls **MADE** only if the ball was *measured* descending through the rim band before being seen below it — a front-of-rim brick that drops below rim level is a **MISS**. A cooldown suppresses rebound double-triggers, and direction is never inferred across a stale gap (`direction_gap_frames`).
4. **Clipping** merges overlapping shot windows and cuts each with a frame-accurate ffmpeg re-encode (audio preserved). No `cap.set()` keyframe drift in either path.

## Accessibility

- Full keyboard flow, including rim marking on the canvas (arrows move, Shift+arrows resize, Delete clears, Alt for 1-px nudge) plus plain numeric x/y/w/h fields as an equivalent.
- Live regions announce upload status, rim-box changes, progress, and each detected shot.
- Visible focus rings, WCAG-AA contrast on the dark theme, `prefers-reduced-motion` respected, semantic landmarks and a skip link.

## Honest limits

Made/miss classification from a single 2D camera angle has an accuracy ceiling — a ball passing just in front of the rim can look identical to one passing through it. Every shot event carries a `confidence` field (`high` when the through-the-rim descent was directly measured, `medium` when resolved on timeout or after occlusion) so users know which calls to double-check. Tighter rim boxes and steadier camera positions measurably help.

## Tests

```bash
python tests/test_detector.py     # or: python -m pytest tests/ -q
```

Synthetic trajectory harness — no video or YOLO needed. Covers the clean swish, the front-rim brick, the occluded swish, velocity across detection gaps, and rebound cooldown.

## Tuning

All knobs live in `EngineConfig` (`engine.py`). The sensitive ones:

| Parameter | Default | Notes |
|---|---|---|
| `max_jump_ratio` | 0.18 | Max ball movement per frame as a fraction of frame width. Lower for tripod footage, raise for handheld/zoomed footage. |
| `yolo_conf` | 0.15 | Raise if random objects get tracked; lower if the ball vanishes near the rim. |
| `coast_frames` | 6 | How long to predict through occlusion. Coasted points never count toward a make. |
| `shot_cooldown_seconds` | 2.0 | Raise if rebounds double-trigger. |

## Deploying

The engine is fully headless (no GUI, no `input()`, no display probing). For production: put uvicorn behind nginx with a matching `client_max_body_size`, swap the in-memory job registry for Redis/DB if you need multi-process workers, and add periodic cleanup of `data/videos/`.

## Roadmap

- [ ] Batch queue UI (process a whole season of games)
- [ ] Auto-suggest the rim box (rim detector model) with manual confirm
- [ ] Vertical/social reframe export
- [ ] Per-session shot chart from event coordinates
