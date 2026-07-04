"""
HoopClip server — upload footage, mark the rim in the browser, get clips.

Run:  uvicorn server:app --reload
Then open http://127.0.0.1:8000

Design notes
------------
* Uploads are stored under data/videos/<uuid>/ with the original name kept
  only as display metadata — nothing user-controlled ever touches a path.
* Analysis runs on a small thread pool; job state lives in an in-memory
  registry guarded by a lock. The frontend polls GET /api/jobs/{id}, which
  returns progress plus any shot events found so far, so detection feels live.
* Jobs are cancellable; cancellation is checked once per decoded frame.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine
import export

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

app = FastAPI(title="HoopClip")


# ---------------------------------------------------------------------------
# In-memory registries
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_files_lock = threading.Lock()  # guards read-modify-write of roster.json / coaching.json
_videos: dict[str, dict] = {}
_jobs: dict[str, "Job"] = {}
_executor = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2))

# Auto-assigned player colors — distinct on the dark film-room theme and
# deliberately avoiding the made/miss green/red so markers stay readable.
PLAYER_PALETTE = ["#f2c14e", "#8ecbff", "#e9701e", "#c9a0ff",
                  "#6fd6c4", "#f28db2", "#efeae0", "#a0b8ff"]


@dataclass
class Job:
    id: str
    video_id: str
    mode: str
    status: str = "queued"          # queued | analyzing | clipping | done | error | cancelled
    progress: float = 0.0
    message: str = "Waiting for a worker"
    events: list = field(default_factory=list)
    clips: list = field(default_factory=list)
    error: str | None = None
    hoop_box: tuple | None = None   # (x, y, w, h) in source pixels
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "mode": self.mode,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "events": list(self.events),
            "clips": list(self.clips),
            "error": self.error,
        }


def _video_or_404(video_id: str) -> dict:
    with _lock:
        v = _videos.get(video_id)
    if not v:
        raise HTTPException(404, "Video not found. It may have been removed — upload it again.")
    return v


def _job_or_404(job_id: str) -> Job:
    with _lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found.")
    return j


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext or 'none'}'. "
            f"Upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    video_id = uuid.uuid4().hex
    vdir = os.path.join(VIDEO_DIR, video_id)
    os.makedirs(vdir, exist_ok=True)
    dest = os.path.join(vdir, f"source{ext}")

    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File is larger than the 2 GB upload limit.")
                out.write(chunk)
        meta = engine.probe_video(dest)
    except HTTPException:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(422, str(exc))

    record = {
        "id": video_id,
        "path": dest,
        "dir": vdir,
        "display_name": os.path.basename(file.filename or "video"),
        "size_bytes": size,
        **meta,
    }
    with _lock:
        _videos[video_id] = record

    public = {k: v for k, v in record.items() if k not in ("path", "dir")}
    return JSONResponse(public)


@app.get("/api/videos/{video_id}/frame")
def get_frame(video_id: str, t: float = 0.0):
    v = _video_or_404(video_id)
    try:
        jpeg = engine.extract_frame_jpeg(v["path"], t)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    v = _video_or_404(video_id)
    with _lock:
        _videos.pop(video_id, None)
    shutil.rmtree(v["dir"], ignore_errors=True)
    return {"deleted": video_id}


# ---------------------------------------------------------------------------
# Roster (per-video player list, persisted next to the video files)
# ---------------------------------------------------------------------------

class PlayerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    number: Optional[str] = Field(default=None, max_length=8)


def _roster_path(video: dict) -> str:
    return os.path.join(video["dir"], "roster.json")


def _load_roster(video: dict) -> dict:
    path = _roster_path(video)
    if not os.path.isfile(path):
        return {"video_id": video["id"], "players": []}
    with open(path) as f:
        return json.load(f)


def _save_roster(video: dict, roster: dict):
    with open(_roster_path(video), "w") as f:
        json.dump(roster, f, indent=2)


def _next_color(players: list) -> str:
    used = {p.get("color") for p in players}
    for c in PLAYER_PALETTE:
        if c not in used:
            return c
    return PLAYER_PALETTE[len(players) % len(PLAYER_PALETTE)]


@app.get("/api/videos/{video_id}/roster")
def get_roster(video_id: str):
    video = _video_or_404(video_id)
    with _files_lock:
        return _load_roster(video)


@app.post("/api/videos/{video_id}/roster", status_code=201)
def add_player(video_id: str, req: PlayerCreate):
    video = _video_or_404(video_id)
    with _files_lock:
        roster = _load_roster(video)
        player = {
            "id": f"p_{uuid.uuid4().hex[:8]}",
            "name": req.name.strip(),
            "number": req.number,
            "color": _next_color(roster["players"]),
            "created_at": time.time(),
        }
        roster["players"].append(player)
        _save_roster(video, roster)
    return player


@app.delete("/api/videos/{video_id}/roster/{player_id}")
def remove_player(video_id: str, player_id: str):
    video = _video_or_404(video_id)
    with _files_lock:
        roster = _load_roster(video)
        kept = [p for p in roster["players"] if p["id"] != player_id]
        if len(kept) == len(roster["players"]):
            raise HTTPException(404, "Player not found.")
        roster["players"] = kept
        _save_roster(video, roster)
    return {"deleted": player_id}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class HoopBox(BaseModel):
    # In SOURCE pixel coordinates (the frontend converts from canvas space).
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class JobRequest(BaseModel):
    video_id: str
    hoop_box: HoopBox
    mode: str = Field(default="all", pattern="^(made|miss|all)$")
    clip_before_seconds: float = Field(default=3.0, ge=0, le=30)
    clip_after_seconds: float = Field(default=1.0, ge=0, le=30)


@app.post("/api/jobs")
def create_job(req: JobRequest):
    v = _video_or_404(req.video_id)

    # Sanity-check the rim box against the actual frame.
    b = req.hoop_box
    if b.x + b.w > v["width"] + 2 or b.y + b.h > v["height"] + 2:
        raise HTTPException(422, "The rim box falls outside the video frame. Mark it again.")

    job = Job(id=uuid.uuid4().hex, video_id=req.video_id, mode=req.mode,
              hoop_box=(b.x, b.y, b.w, b.h))
    with _lock:
        _jobs[job.id] = job

    cfg = engine.EngineConfig(
        clip_before_seconds=req.clip_before_seconds,
        clip_after_seconds=req.clip_after_seconds,
    )
    hoop = (b.x, b.y, b.w, b.h)
    _executor.submit(_run_job, job, v, hoop, cfg)
    return job.snapshot()


def _run_job(job: Job, video: dict, hoop_box: tuple, cfg: engine.EngineConfig):
    def progress(frac: float, msg: str):
        job.progress = frac * (0.9 if job.status == "analyzing" else 1.0)
        if job.status == "clipping":
            job.progress = 0.9 + frac * 0.1
        job.message = msg

    def on_event(ev: engine.ShotEvent):
        job.events.append(ev.to_dict())

    try:
        job.status = "analyzing"
        job.message = "Loading detection model"
        analysis = engine.analyze_video(
            video["path"], hoop_box, cfg,
            on_progress=progress, on_event=on_event,
            should_cancel=job.cancel_flag.is_set,
        )
        if job.cancel_flag.is_set():
            job.status = "cancelled"
            job.message = "Cancelled"
            return

        job.status = "clipping"
        clip_dir = os.path.join(video["dir"], "clips", job.id)
        job.clips = engine.cut_clips(
            video["path"], analysis, job.mode, clip_dir, cfg,
            on_progress=progress,
        )
        _write_coaching_meta(video, job.id, hoop_box)
        job.progress = 1.0
        n = len(job.clips)
        job.message = (f"Done — {n} clip{'s' if n != 1 else ''} cut"
                       if n else "Done — no shots matched. Try mode 'all' or re-mark the rim.")
        job.status = "done"
    except Exception as exc:  # surface, never swallow
        job.status = "error"
        job.error = str(exc)
        job.message = "Processing failed"


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_snapshot_enriched(_job_or_404(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _job_or_404(job_id)
    job.cancel_flag.set()
    return {"cancelling": job_id}


# ---------------------------------------------------------------------------
# Coaching: per-shot player/drill assignments (persisted next to the clips)
# ---------------------------------------------------------------------------

DrillTag = Literal["layups", "three_pointers", "midrange"]


class ShotAssignment(BaseModel):
    # Full replace, not a partial patch — the client always sends both fields.
    player_id: Optional[str] = None
    drill_tag: Optional[DrillTag] = None


def _coaching_path(video: dict, job_id: str) -> str:
    d = os.path.join(video["dir"], "clips", job_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "coaching.json")


def _load_coaching(video: dict, job_id: str) -> dict:
    path = _coaching_path(video, job_id)
    if not os.path.isfile(path):
        return {"hoop_box": None, "rim_diameter_ft": engine.RIM_DIAMETER_FT,
                "assignments": {}}
    with open(path) as f:
        return json.load(f)


def _write_coaching_meta(video: dict, job_id: str, hoop_box: tuple):
    with _files_lock:
        data = _load_coaching(video, job_id)
        data["hoop_box"] = dict(zip(("x", "y", "w", "h"), hoop_box))
        data["rim_diameter_ft"] = engine.RIM_DIAMETER_FT
        data.setdefault("assignments", {})
        with open(_coaching_path(video, job_id), "w") as f:
            json.dump(data, f, indent=2)


def _job_snapshot_enriched(job: Job) -> dict:
    """Job snapshot plus per-event assignments and approximate court coords."""
    snap = job.snapshot()
    assignments = {}
    if job.status == "done":
        video = _video_or_404(job.video_id)
        with _files_lock:
            assignments = _load_coaching(video, job.id).get("assignments", {})
    for ev in snap["events"]:
        a = assignments.get(str(ev["frame"]), {})
        ev["player_id"] = a.get("player_id")
        ev["drill_tag"] = a.get("drill_tag")
        if job.hoop_box and ev.get("ball_x") is not None:
            ev["court_x_ft"], ev["court_dist_ft"] = engine.approx_shot_location_ft(
                ev["ball_x"], ev["ball_y"], job.hoop_box)
        else:
            ev["court_x_ft"] = ev["court_dist_ft"] = None
    return snap


@app.patch("/api/jobs/{job_id}/shots/{frame}")
def assign_shot(job_id: str, frame: int, req: ShotAssignment):
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Wait for the job to finish before assigning shots.")
    if not any(e["frame"] == frame for e in job.events):
        raise HTTPException(404, "Shot not found for this job.")
    video = _video_or_404(job.video_id)
    if req.player_id is not None:
        roster = _load_roster(video)
        if not any(p["id"] == req.player_id for p in roster["players"]):
            raise HTTPException(422, "Unknown player — add them to the roster first.")
    with _files_lock:
        data = _load_coaching(video, job_id)
        data.setdefault("assignments", {})[str(frame)] = {
            "player_id": req.player_id, "drill_tag": req.drill_tag,
        }
        with open(_coaching_path(video, job_id), "w") as f:
            json.dump(data, f, indent=2)
    return {"frame": frame, "player_id": req.player_id, "drill_tag": req.drill_tag}


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def _clip_path(job: Job, filename: str) -> str:
    v = _video_or_404(job.video_id)
    clip_dir = os.path.join(v["dir"], "clips", job.id)
    path = os.path.normpath(os.path.join(clip_dir, filename))
    if not path.startswith(clip_dir) or not os.path.isfile(path):
        raise HTTPException(404, "Clip not found.")
    return path


@app.get("/api/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str):
    job = _job_or_404(job_id)
    return FileResponse(_clip_path(job, filename), media_type="video/mp4",
                        filename=filename)


@app.get("/api/jobs/{job_id}/clips.zip")
def get_clips_zip(job_id: str):
    job = _job_or_404(job_id)
    if not job.clips:
        raise HTTPException(404, "No clips to download for this job.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for c in job.clips:
            z.write(_clip_path(job, c["file"]), arcname=c["file"])
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="hoopclip_clips.zip"'},
    )


# ---------------------------------------------------------------------------
# Exports (CSV stats + PDF report)
# ---------------------------------------------------------------------------

def _export_ready_or_raise(job_id: str) -> tuple:
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Wait for the job to finish before exporting.")
    if not job.events:
        raise HTTPException(404, "No shots to export for this job.")
    video = _video_or_404(job.video_id)
    with _files_lock:
        players_by_id = {p["id"]: p for p in _load_roster(video)["players"]}
    return job, video, players_by_id


@app.get("/api/jobs/{job_id}/export.csv")
def export_csv(job_id: str):
    job, _video, players_by_id = _export_ready_or_raise(job_id)
    csv_text = export.build_csv(_job_snapshot_enriched(job)["events"], players_by_id)
    # utf-8-sig BOM so the file opens cleanly in Excel.
    return Response(
        content=csv_text.encode("utf-8-sig"), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="hoopclip_stats.csv"'},
    )


@app.get("/api/jobs/{job_id}/export.pdf")
def export_pdf(job_id: str):
    job, video, players_by_id = _export_ready_or_raise(job_id)
    try:
        pdf_bytes = export.build_pdf(
            _job_snapshot_enriched(job), players_by_id, video["display_name"])
    except RuntimeError as exc:  # missing matplotlib/reportlab
        raise HTTPException(501, str(exc))
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="hoopclip_report.pdf"'},
    )


# Static frontend (mounted last so /api keeps priority).
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True),
          name="static")
