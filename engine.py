"""
HoopClip engine — web-native basketball shot detection and clipping.

This is the headless core behind the web app. It is a refactor of the CLI
ball-tracking script into an importable, callback-driven module:

  * No GUI, no input(), no environment probing — safe inside a worker thread.
  * The rim box arrives from the browser (drawn on a canvas), never selectROI.
  * Progress and shot events stream out through callbacks so the web layer
    can report them live.
  * Clips are cut with ffmpeg in a frame-accurate re-encode pass (audio kept),
    with a single-decode OpenCV fallback when ffmpeg is missing — the old
    keyframe-seek drift bug stays fixed in both paths.

Detection carries over every fix from the reviewed CLI version:
  * Coasting velocity divides by the real frame gap (no velocity inflation).
  * Direction comparisons are gated by DIRECTION_GAP_FRAMES so stale y-values
    from before an occlusion can't corrupt made/miss classification.
  * A make requires the ball to have descended THROUGH the rim band; a
    front-of-rim brick that reappears below the rim does not count.
  * Shot cooldown suppresses rebound double-triggers.

New noise-reduction work in this version:
  * Detection picking scores confidence against jump distance instead of
    taking the nearest box blindly, so a second ball / round object on the
    sideline doesn't steal the track.
  * Implausibly large "sports ball" boxes (relative to frame size) are
    rejected before they ever reach the tracker.
  * Ball centers are EMA-smoothed to remove per-frame YOLO jitter before the
    shot detector sees them (raw positions are kept for the jump check so
    smoothing can't mask a real teleport).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import cv2
import numpy as np

# Lazy YOLO import: the server should start (and report a clear error) even if
# ultralytics isn't installed yet.
_YOLO = None


def _load_yolo(model_name: str):
    global _YOLO
    if _YOLO is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc
        _YOLO = YOLO(model_name)
    return _YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    # YOLO
    yolo_model: str = "yolov8s.pt"
    yolo_conf: float = 0.15          # low: catches the faint/blurry ball at the rim
    sports_ball_class: int = 32

    # Tracking
    max_jump_ratio: float = 0.18     # fraction of frame WIDTH the ball may move/frame
    track_lost_frames: int = 8
    coast_frames: int = 6            # bridge net/rim occlusion
    ema_alpha: float = 0.55          # 1.0 = no smoothing; lower = smoother
    max_ball_frac: float = 0.20      # reject "balls" wider than 20% of the frame

    # Shot detection
    arm_x_pad_ratio: float = 0.5
    attempt_zone_pad_ratio: float = 0.6
    make_confirm_seconds: float = 0.8
    shot_cooldown_seconds: float = 2.0
    direction_gap_frames: int = 3    # distrust stale y across long occlusions

    # Clipping
    clip_before_seconds: float = 3.0
    clip_after_seconds: float = 1.0


@dataclass
class ShotEvent:
    frame: int
    time_s: float
    result: str            # "made" | "miss"
    confidence: str        # "high" | "medium"  (2D single-camera honesty)

    def to_dict(self):
        return asdict(self)


@dataclass
class AnalysisResult:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    events: list = field(default_factory=list)   # list[ShotEvent]


ProgressCB = Callable[[float, str], None]        # (fraction 0..1, message)
EventCB = Callable[[ShotEvent], None]
CancelCheck = Callable[[], bool]


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def probe_video(path: str) -> dict:
    """Validate a video and return its metadata. Raises ValueError on junk."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video. The file may be corrupt or an unsupported format.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise ValueError("The first frame could not be read. The file may be truncated.")
    if fps <= 1 or fps > 480 or math.isnan(fps):
        raise ValueError(f"Video reports an implausible frame rate ({fps:.1f} fps).")
    if w <= 0 or h <= 0:
        raise ValueError("Video reports invalid dimensions.")
    if frames <= 0:
        # Some containers omit the count; estimate later during decode.
        frames = 0
    return {
        "fps": fps,
        "frame_count": frames,
        "width": w,
        "height": h,
        "duration_s": (frames / fps) if frames else None,
    }


def extract_frame_jpeg(path: str, time_s: float, max_width: int = 1280) -> bytes:
    """Grab one frame near time_s as JPEG bytes (for the rim-marking canvas)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        # Fall back to the first frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("Could not read a frame at that position.")
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode the preview frame.")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class BallTracker:
    """Single-ball tracker with jump rejection, EMA smoothing and coasting.

    Feed it the candidate detections for each frame; read back a point that is
    either MEASURED (a real detection) or COASTED (a short-lived prediction to
    bridge rim/net occlusion). Coasted points are flagged so downstream logic
    can weigh them accordingly.
    """

    def __init__(self, frame_w: int, cfg: EngineConfig):
        self.cfg = cfg
        self.max_jump_px = cfg.max_jump_ratio * frame_w
        self.raw_pos: Optional[tuple] = None      # last accepted RAW center
        self.smooth_pos: Optional[tuple] = None   # EMA-smoothed center
        self.last_seen_frame: Optional[int] = None
        self.velocity = (0.0, 0.0)                # px/frame, from raw positions
        self.coasted_left = 0

    def _accept(self, cx: float, cy: float, frame_idx: int):
        if self.raw_pos is not None and self.last_seen_frame is not None:
            gap = max(1, frame_idx - self.last_seen_frame)
            # Velocity per frame divides by the REAL gap — the old inflation
            # bug came from treating a multi-frame delta as one frame.
            self.velocity = (
                (cx - self.raw_pos[0]) / gap,
                (cy - self.raw_pos[1]) / gap,
            )
        self.raw_pos = (cx, cy)
        self.last_seen_frame = frame_idx
        self.coasted_left = self.cfg.coast_frames
        a = self.cfg.ema_alpha
        if self.smooth_pos is None:
            self.smooth_pos = (cx, cy)
        else:
            self.smooth_pos = (
                a * cx + (1 - a) * self.smooth_pos[0],
                a * cy + (1 - a) * self.smooth_pos[1],
            )

    def update(self, detections: list, frame_idx: int):
        """detections: list of (cx, cy, w, h, conf). Returns (point, kind) or (None, 'lost').

        kind is 'measured' or 'coasted'. point is the smoothed center.
        """
        best = None
        if detections:
            if self.raw_pos is None:
                # No track yet: take the most confident candidate.
                best = max(detections, key=lambda d: d[4])
            else:
                gap = max(1, frame_idx - (self.last_seen_frame or frame_idx))
                allowed = self.max_jump_px * gap
                # Score = confidence minus a distance penalty, instead of a
                # blind nearest-neighbor grab. A confident sideline ball far
                # away loses to a fainter detection continuing the trajectory.
                scored = []
                for d in detections:
                    dist = math.hypot(d[0] - self.raw_pos[0], d[1] - self.raw_pos[1])
                    if dist <= allowed:
                        scored.append((d[4] - 0.5 * (dist / max(allowed, 1e-6)), d))
                if scored:
                    best = max(scored, key=lambda s: s[0])[1]
                elif self.last_seen_frame is not None and \
                        frame_idx - self.last_seen_frame > self.cfg.track_lost_frames:
                    # Track is stale — re-seed on the most confident detection.
                    best = max(detections, key=lambda d: d[4])
                    self.velocity = (0.0, 0.0)
                    self.raw_pos = None

        if best is not None:
            self._accept(best[0], best[1], frame_idx)
            return self.smooth_pos, "measured"

        # No acceptable detection: coast briefly on the last velocity.
        if self.raw_pos is not None and self.coasted_left > 0:
            self.coasted_left -= 1
            self.raw_pos = (
                self.raw_pos[0] + self.velocity[0],
                self.raw_pos[1] + self.velocity[1],
            )
            self.smooth_pos = self.raw_pos
            return self.smooth_pos, "coasted"

        return None, "lost"


# ---------------------------------------------------------------------------
# Shot detection
# ---------------------------------------------------------------------------

class HoopShotDetector:
    """Online made/miss classifier around a marked rim box.

    States: IDLE -> ARMED (ball descending in the attempt zone) -> resolved.
    A MAKE requires the ball to pass DOWN THROUGH the rim band and then be
    seen below it; a ball that merely reappears below the rim after hitting
    the front iron is a MISS (the front-of-rim brick fix).
    """

    def __init__(self, hoop_box: tuple, fps: float, cfg: EngineConfig):
        x, y, w, h = hoop_box
        self.rim_left, self.rim_right = x, x + w
        self.rim_top, self.rim_bottom = y, y + h
        self.cfg = cfg
        self.fps = fps

        pad_x = w * cfg.arm_x_pad_ratio
        self.arm_left, self.arm_right = x - pad_x, x + w + pad_x
        zpad = w * cfg.attempt_zone_pad_ratio
        self.zone_left, self.zone_right = x - zpad, x + w + zpad
        self.zone_top = y - h * 3.0

        self.state = "idle"
        self.armed_frame = None
        self.passed_through_band = False
        self.last_y = None
        self.last_y_frame = None
        self.cooldown_until = -1
        self.make_confirm_frames = int(cfg.make_confirm_seconds * fps)
        self.cooldown_frames = int(cfg.shot_cooldown_seconds * fps)

    def _direction(self, y: float, frame_idx: int) -> Optional[float]:
        """dy/frame if the previous sample is recent enough, else None."""
        if self.last_y is None or self.last_y_frame is None:
            return None
        if frame_idx - self.last_y_frame > self.cfg.direction_gap_frames:
            # Stale sample across an occlusion — don't fabricate a direction.
            return None
        return (y - self.last_y) / max(1, frame_idx - self.last_y_frame)

    def update(self, point: Optional[tuple], kind: str, frame_idx: int) -> Optional[ShotEvent]:
        event = None

        if point is not None:
            x, y = point
            dy = self._direction(y, frame_idx)

            if self.state == "idle" and frame_idx >= self.cooldown_until:
                descending = dy is not None and dy > 0
                in_zone = (self.zone_left <= x <= self.zone_right and
                           self.zone_top <= y <= self.rim_top)
                if descending and in_zone:
                    self.state = "armed"
                    self.armed_frame = frame_idx
                    self.passed_through_band = False

            elif self.state == "armed":
                in_band = (self.rim_top <= y <= self.rim_bottom and
                           self.rim_left <= x <= self.rim_right)
                descending = dy is None or dy >= 0
                if in_band and descending and kind == "measured":
                    # Only a MEASURED sighting inside the band counts toward a
                    # make; a coasted prediction through the band is not proof.
                    self.passed_through_band = True

                below = y > self.rim_bottom
                lateral = x < self.arm_left or x > self.arm_right
                if below and kind == "measured":
                    result = "made" if self.passed_through_band else "miss"
                    conf = "high" if self.passed_through_band else "medium"
                    event = self._resolve(frame_idx, result, conf)
                elif lateral and y < self.rim_top:
                    # Ball bounced up and away from the rim: miss.
                    event = self._resolve(frame_idx, "miss", "high")
                elif frame_idx - self.armed_frame > self.make_confirm_frames:
                    # Timed out without a clean below-rim sighting.
                    result = "made" if self.passed_through_band else "miss"
                    event = self._resolve(frame_idx, result, "medium")

            self.last_y = y
            self.last_y_frame = frame_idx
        else:
            # Track fully lost while armed for too long: resolve on timeout.
            if self.state == "armed" and \
                    frame_idx - self.armed_frame > self.make_confirm_frames:
                result = "made" if self.passed_through_band else "miss"
                event = self._resolve(frame_idx, result, "medium")

        return event

    def _resolve(self, frame_idx: int, result: str, confidence: str) -> ShotEvent:
        self.state = "idle"
        self.cooldown_until = frame_idx + self.cooldown_frames
        return ShotEvent(
            frame=frame_idx,
            time_s=frame_idx / self.fps,
            result=result,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Analysis pass
# ---------------------------------------------------------------------------

def analyze_video(
    path: str,
    hoop_box: tuple,
    cfg: EngineConfig,
    on_progress: Optional[ProgressCB] = None,
    on_event: Optional[EventCB] = None,
    should_cancel: Optional[CancelCheck] = None,
) -> AnalysisResult:
    """Single decode pass: YOLO -> tracker -> shot detector -> events."""
    meta = probe_video(path)
    fps, total = meta["fps"], meta["frame_count"]
    model = _load_yolo(cfg.yolo_model)

    cap = cv2.VideoCapture(path)
    tracker = BallTracker(meta["width"], cfg)
    detector = HoopShotDetector(hoop_box, fps, cfg)
    result = AnalysisResult(
        fps=fps, frame_count=total, width=meta["width"], height=meta["height"],
        duration_s=meta["duration_s"] or 0.0,
    )

    max_ball_px = cfg.max_ball_frac * meta["width"]
    frame_idx = -1
    report_every = max(1, int(fps))  # ~once a second of video

    while True:
        if should_cancel and should_cancel():
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        yolo_out = model.predict(frame, conf=cfg.yolo_conf, verbose=False,
                                 classes=[cfg.sports_ball_class])
        detections = []
        for r in yolo_out:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                if w > max_ball_px or h > max_ball_px:
                    continue  # implausibly large "ball" — reject the noise
                detections.append(((x1 + x2) / 2, (y1 + y2) / 2, w, h,
                                   float(box.conf[0])))

        point, kind = tracker.update(detections, frame_idx)
        event = detector.update(point, kind, frame_idx)
        if event:
            result.events.append(event)
            if on_event:
                on_event(event)

        if on_progress and frame_idx % report_every == 0:
            frac = (frame_idx / total) if total else 0.0
            on_progress(min(frac, 0.999), f"Analyzing frame {frame_idx:,}")

    cap.release()
    if not total:
        result.frame_count = frame_idx + 1
        result.duration_s = result.frame_count / fps
    if on_progress:
        on_progress(1.0, "Analysis complete")
    return result


# ---------------------------------------------------------------------------
# Clip cutting
# ---------------------------------------------------------------------------

def _merge_windows(windows: list) -> list:
    """Merge overlapping (start, end, events) windows so back-to-back shots
    become one clip instead of two files fighting over the same seconds."""
    if not windows:
        return []
    windows.sort(key=lambda w: w[0])
    merged = [list(windows[0])]
    for s, e, evs in windows[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = merged[-1][2] + evs
        else:
            merged.append([s, e, evs])
    return [tuple(m) for m in merged]


def cut_clips(
    video_path: str,
    analysis: AnalysisResult,
    mode: str,
    out_dir: str,
    cfg: EngineConfig,
    on_progress: Optional[ProgressCB] = None,
) -> list:
    """Cut one clip per (merged) shot window. Returns clip metadata dicts."""
    os.makedirs(out_dir, exist_ok=True)
    wanted = [e for e in analysis.events if mode == "all" or e.result == mode]
    duration = analysis.duration_s or (analysis.frame_count / analysis.fps)

    windows = []
    for e in wanted:
        s = max(0.0, e.time_s - cfg.clip_before_seconds)
        end = min(duration, e.time_s + cfg.clip_after_seconds)
        windows.append((s, end, [e]))
    windows = _merge_windows(windows)

    clips = []
    have_ffmpeg = shutil.which("ffmpeg") is not None

    for i, (start, end, events) in enumerate(windows, 1):
        label = "-".join(sorted({e.result for e in events}))
        name = f"clip_{i:02d}_{label}_{_ts(start)}.mp4"
        out_path = os.path.join(out_dir, name)

        if have_ffmpeg:
            _cut_ffmpeg(video_path, start, end, out_path)
        else:
            _cut_opencv(video_path, start, end, out_path, analysis)

        clips.append({
            "file": name,
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "events": [e.to_dict() for e in events],
        })
        if on_progress:
            on_progress(i / len(windows), f"Cut clip {i} of {len(windows)}")

    with open(os.path.join(out_dir, "clips.json"), "w") as f:
        json.dump(clips, f, indent=2)
    return clips


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m{s:02d}s"


def _cut_ffmpeg(src: str, start: float, end: float, out_path: str):
    """Frame-accurate cut with audio preserved. Re-encodes video so the clip
    starts exactly where the play does — stream-copy would snap to the
    previous keyframe and reintroduce the old drift bug."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {os.path.basename(out_path)}: {proc.stderr[-400:]}")


def _cut_opencv(src: str, start: float, end: float, out_path: str,
                analysis: AnalysisResult):
    """Fallback: single sequential decode pass (no cap.set() keyframe drift).
    No audio — OpenCV can't mux it and ffmpeg isn't available here."""
    cap = cv2.VideoCapture(src)
    fps = analysis.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps,
                             (analysis.width, analysis.height))
    start_f, end_f = int(start * fps), int(end * fps)
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < start_f:
            continue
        if idx > end_f:
            break
        writer.write(frame)
    writer.release()
    cap.release()
