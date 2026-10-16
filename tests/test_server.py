"""
Tests for the server layer: persistence, roster, shot tagging, guards.

The endpoints are plain (sync) functions, so these call them directly instead
of going over HTTP — that keeps the suite dependency-free (no httpx/TestClient)
while still exercising the real request models and the real error paths.

Each test runs against a throwaway data directory; nothing touches the actual
data/videos/ tree.

Run:  python tests/test_server.py     (or: python -m pytest tests/ -q)
"""

import asyncio
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi import HTTPException

import engine
import server

_tmp_root = None


def setup():
    """Fresh temp VIDEO_DIR and empty registries before each test."""
    global _tmp_root
    teardown()
    _tmp_root = tempfile.mkdtemp(prefix="clipcourt-test-")
    server.VIDEO_DIR = os.path.join(_tmp_root, "videos")
    os.makedirs(server.VIDEO_DIR, exist_ok=True)
    server._videos.clear()
    server._jobs.clear()


def teardown():
    global _tmp_root
    if _tmp_root and os.path.isdir(_tmp_root):
        shutil.rmtree(_tmp_root, ignore_errors=True)
    _tmp_root = None


def make_video(video_id="vid1", frames=45, size=(320, 240), with_rim=False):
    """Write a tiny real .mp4 and register it the way an upload would."""
    vdir = os.path.join(server.VIDEO_DIR, video_id)
    os.makedirs(vdir, exist_ok=True)
    path = os.path.join(vdir, "source.mp4")
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(frames):
        frame = np.zeros((h, w, 3), np.uint8)
        if with_rim:
            # A hollow, wide orange ring high in the frame: a rim.
            cv2.ellipse(frame, (w // 2, h // 4), (26, 7), 0, 0, 360,
                        (20, 110, 230), 2)
        writer.write(frame)
    writer.release()
    assert os.path.isfile(path), "failed to write the synthetic video"

    record = {
        "id": video_id, "path": path, "dir": vdir,
        "display_name": f"{video_id}.mp4", "size_bytes": os.path.getsize(path),
        "uploaded_at": 1000.0, **engine.probe_video(path),
    }
    server._videos[video_id] = record
    server._persist_video(record)
    return record


def drain(response) -> bytes:
    """Collect a StreamingResponse body into bytes."""
    async def collect():
        out = []
        async for chunk in response.body_iterator:
            out.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        return b"".join(out)
    return asyncio.run(collect())


def raises_http(status, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}"
        return exc
    raise AssertionError(f"expected HTTP {status}, no exception raised")


def done_job(video, frames=(10, 40), box=(100.0, 50.0, 40.0, 12.0)):
    job = server.Job(id="job1", video_id=video["id"], mode="all", status="done",
                     progress=1.0, message="Done", hoop_box=box, device="cpu")
    job.events = [
        {"frame": frames[0], "time_s": frames[0] / 30.0, "result": "made",
         "confidence": "high", "ball_x": 120.0, "ball_y": 70.0},
        {"frame": frames[1], "time_s": frames[1] / 30.0, "result": "miss",
         "confidence": "medium", "ball_x": 90.0, "ball_y": 75.0},
    ]
    job.clips = [{"file": "clip_01_made_00m00s.mp4", "start_s": 0.0,
                  "end_s": 1.0, "events": [job.events[0]]}]
    server._jobs[job.id] = job
    return job


# --------------------------------------------------------------- persistence

def test_video_and_job_survive_a_restart():
    setup()
    video = make_video()
    job = done_job(video)
    server._persist_job(job, video)

    # Simulate a fresh process: drop the in-memory registries and rescan disk.
    server._videos.clear()
    server._jobs.clear()
    server._restore_from_disk()

    assert "vid1" in server._videos, "video was not restored"
    assert server._videos["vid1"]["path"].endswith("source.mp4")
    restored = server._jobs.get("job1")
    assert restored is not None, "job was not restored"
    assert restored.status == "done"
    assert len(restored.events) == 2
    assert restored.hoop_box == (100.0, 50.0, 40.0, 12.0), restored.hoop_box
    teardown()


def test_interrupted_job_is_restored_as_an_error():
    setup()
    video = make_video()
    job = server.Job(id="mid", video_id="vid1", mode="all", status="analyzing",
                     progress=0.4, message="Analyzing frame 300")
    server._persist_job(job, video)
    server._jobs.clear()
    server._restore_from_disk()

    restored = server._jobs["mid"]
    assert restored.status == "error", restored.status
    assert "restart" in (restored.error or "").lower(), restored.error
    teardown()


def test_legacy_directory_without_sidecars_is_adopted():
    setup()
    video = make_video(video_id="legacy")
    # Strip the sidecar to look like an upload from before persistence existed.
    os.remove(os.path.join(video["dir"], "video.json"))
    clip_dir = os.path.join(video["dir"], "clips", "oldjob")
    os.makedirs(clip_dir, exist_ok=True)
    with open(os.path.join(clip_dir, "clips.json"), "w") as f:
        json.dump([{"file": "clip_01_made_00m00s.mp4", "start_s": 0.0,
                    "end_s": 2.0,
                    "events": [{"frame": 7, "time_s": 0.23, "result": "made",
                                "confidence": "high", "ball_x": 1.0,
                                "ball_y": 2.0}]}], f)
    with open(os.path.join(clip_dir, "coaching.json"), "w") as f:
        json.dump({"hoop_box": {"x": 10, "y": 20, "w": 30, "h": 8}}, f)

    server._videos.clear()
    server._jobs.clear()
    server._restore_from_disk()

    assert "legacy" in server._videos, "legacy upload was not adopted"
    job = server._jobs.get("oldjob")
    assert job is not None and job.status == "done"
    assert len(job.events) == 1 and job.events[0]["frame"] == 7
    assert job.hoop_box == (10, 20, 30, 8), job.hoop_box
    teardown()


def test_unreadable_sidecar_does_not_break_startup():
    setup()
    video = make_video()
    with open(server._video_meta_path(video["dir"]), "w") as f:
        f.write("{ this is not json")
    server._videos.clear()
    server._restore_from_disk()   # must not raise
    assert "vid1" not in server._videos
    teardown()


def test_malformed_job_sidecar_does_not_break_startup():
    setup()
    video = make_video()
    server._persist_job(done_job(video), video)
    jobs_dir = os.path.join(video["dir"], "jobs")
    # Valid JSON, wrong shape: no id, and a top-level list.
    with open(os.path.join(jobs_dir, "noid.json"), "w") as f:
        json.dump({"status": "done"}, f)
    with open(os.path.join(jobs_dir, "list.json"), "w") as f:
        json.dump([1, 2, 3], f)
    server._videos.clear()
    server._jobs.clear()
    server._restore_from_disk()   # must not raise
    assert "vid1" in server._videos
    assert list(server._jobs) == ["job1"], list(server._jobs)
    teardown()


def test_deleting_a_video_cancels_its_job_and_stays_deleted():
    setup()
    video = make_video()
    job = server.Job(id="running", video_id="vid1", mode="all",
                     status="analyzing")
    server._jobs[job.id] = job
    server.delete_video("vid1")
    assert job.cancel_flag.is_set(), "a running job should be cancelled"
    assert not os.path.exists(video["dir"])

    # The worker finishing afterwards must not recreate the folder.
    real_analyze = engine.analyze_video
    engine.analyze_video = lambda *a, **k: engine.AnalysisResult(
        fps=30.0, frame_count=0, width=320, height=240, duration_s=0.0)
    try:
        server._run_job(job, video, (1, 1, 10, 4), engine.EngineConfig())
    finally:
        engine.analyze_video = real_analyze
    assert job.status == "cancelled", job.status
    assert not os.path.exists(video["dir"]), "deleted video's folder came back"
    teardown()


def test_cancel_during_clipping_is_not_reported_as_done():
    setup()
    video = make_video()
    job = server.Job(id="j", video_id="vid1", mode="all")
    server._jobs[job.id] = job
    real_analyze, real_cut = engine.analyze_video, engine.cut_clips
    engine.analyze_video = lambda *a, **k: engine.AnalysisResult(
        fps=30.0, frame_count=45, width=320, height=240, duration_s=1.5)

    def cut_then_cancelled(*a, **k):
        job.cancel_flag.set()     # user hits Cancel while clips are cutting
        return []
    engine.cut_clips = cut_then_cancelled
    try:
        server._run_job(job, video, (1, 1, 10, 4), engine.EngineConfig())
    finally:
        engine.analyze_video, engine.cut_clips = real_analyze, real_cut
    assert job.status == "cancelled", job.status
    teardown()


# -------------------------------------------------------------------- roster

def test_roster_add_list_remove():
    setup()
    make_video()
    created = server.add_player("vid1", server.PlayerCreate(name="  Ama  "))
    assert created["name"] == "Ama", "name should be trimmed"
    assert created["color"] in server.PLAYER_PALETTE

    second = server.add_player("vid1", server.PlayerCreate(name="Bo"))
    assert second["color"] != created["color"], "colors should be distinct"

    roster = server.get_roster("vid1")
    assert [p["name"] for p in roster["players"]] == ["Ama", "Bo"]

    server.remove_player("vid1", created["id"])
    assert [p["name"] for p in server.get_roster("vid1")["players"]] == ["Bo"]

    raises_http(404, server.remove_player, "vid1", "p_nope")
    raises_http(404, server.get_roster, "no-such-video")
    teardown()


def test_roster_persists_to_disk():
    setup()
    video = make_video()
    server.add_player("vid1", server.PlayerCreate(name="Ama"))
    with open(os.path.join(video["dir"], "roster.json")) as f:
        assert json.load(f)["players"][0]["name"] == "Ama"
    teardown()


# ------------------------------------------------------------ shot tagging

def test_assign_shot_requires_a_finished_job_a_real_frame_and_a_real_player():
    setup()
    video = make_video()
    job = done_job(video)

    job.status = "analyzing"
    raises_http(409, server.assign_shot, "job1", 10, server.ShotAssignment())
    job.status = "done"

    raises_http(404, server.assign_shot, "job1", 999, server.ShotAssignment())
    raises_http(422, server.assign_shot, "job1", 10,
                server.ShotAssignment(player_id="p_ghost"))

    player = server.add_player("vid1", server.PlayerCreate(name="Ama"))
    saved = server.assign_shot("job1", 10, server.ShotAssignment(
        player_id=player["id"], drill_tag="layups"))
    assert saved["player_id"] == player["id"] and saved["drill_tag"] == "layups"

    # Persisted next to the clips, and surfaced on the next status poll.
    with open(server._coaching_path(video, "job1")) as f:
        assert json.load(f)["assignments"]["10"]["drill_tag"] == "layups"
    event = next(e for e in server.job_status("job1")["events"] if e["frame"] == 10)
    assert event["player_id"] == player["id"]
    assert event["drill_tag"] == "layups"
    teardown()


def test_enriched_snapshot_adds_approximate_court_coordinates():
    setup()
    video = make_video()
    done_job(video)
    events = {e["frame"]: e for e in server.job_status("job1")["events"]}
    # Rim box is 40 px wide = 1.5 ft, so 26.67 px/ft; ball_x 120 vs rim center
    # 120 means dead centre, and ball_y 70 vs rim bottom 62 is 8 px below.
    assert events[10]["court_x_ft"] == 0.0, events[10]
    assert events[10]["court_dist_ft"] == 0.3, events[10]
    assert events[40]["court_x_ft"] < 0, "the miss was left of the rim"
    teardown()


def test_snapshot_without_a_hoop_box_has_no_coordinates():
    setup()
    video = make_video()
    job = done_job(video)
    job.hoop_box = None          # e.g. a job adopted from a legacy directory
    for event in server.job_status("job1")["events"]:
        assert event["court_x_ft"] is None and event["court_dist_ft"] is None
    teardown()


def test_status_polls_do_not_mutate_the_jobs_own_events():
    setup()
    video = make_video()
    job = done_job(video)
    before = [dict(e) for e in job.events]
    server.job_status("job1")
    server.job_status("job1")
    assert job.events == before, job.events[0].keys()
    teardown()


def test_jobs_need_a_nonzero_clip_length():
    setup()
    make_video()
    req = server.JobRequest(video_id="vid1",
                            hoop_box=server.HoopBox(x=10, y=10, w=40, h=12),
                            clip_before_seconds=0, clip_after_seconds=0)
    raises_http(422, server.create_job, req)
    assert not server._jobs, "no job should have been created"
    teardown()


# -------------------------------------------------------- clips and exports

def test_clip_path_rejects_traversal_and_siblings():
    setup()
    video = make_video()
    job = done_job(video)
    clip_dir = os.path.join(video["dir"], "clips", job.id)
    os.makedirs(clip_dir, exist_ok=True)
    real = os.path.join(clip_dir, "clip_01_made_00m00s.mp4")
    with open(real, "wb") as f:
        f.write(b"not really video")
    assert server._clip_path(job, "clip_01_made_00m00s.mp4") == real

    # A sibling directory whose name merely starts with this job's id.
    sibling = os.path.join(video["dir"], "clips", job.id + "_other")
    os.makedirs(sibling, exist_ok=True)
    with open(os.path.join(sibling, "secret.mp4"), "wb") as f:
        f.write(b"nope")
    raises_http(404, server._clip_path, job, "../" + job.id + "_other/secret.mp4")
    raises_http(404, server._clip_path, job, "../../../../etc/passwd")
    raises_http(404, server._clip_path, job, "missing.mp4")
    teardown()


def test_exports_refuse_unfinished_or_empty_jobs():
    setup()
    video = make_video()
    job = done_job(video)

    job.status = "analyzing"
    raises_http(409, server.export_csv, "job1")
    raises_http(409, server.export_pdf, "job1")

    job.status = "done"
    kept, job.events = job.events, []
    raises_http(404, server.export_csv, "job1")

    job.events = kept
    body = server.export_csv("job1").body
    assert body.startswith(b"\xef\xbb\xbf"), "expected a UTF-8 BOM for Excel"
    assert b"player_name" in body
    teardown()


def test_zip_includes_the_vertical_cut():
    setup()
    video = make_video()
    job = done_job(video)
    clip_dir = os.path.join(video["dir"], "clips", job.id)
    os.makedirs(clip_dir, exist_ok=True)
    for name in ("clip_01_made_00m00s.mp4", "clip_01_made_00m00s_vertical.mp4"):
        with open(os.path.join(clip_dir, name), "wb") as f:
            f.write(b"x" * 10)
    job.clips[0]["vertical_file"] = "clip_01_made_00m00s_vertical.mp4"

    payload = drain(server.get_clips_zip("job1"))
    names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
    assert names == ["clip_01_made_00m00s.mp4",
                     "clip_01_made_00m00s_vertical.mp4"], names
    teardown()


# ------------------------------------------------------------------ listing

def test_list_videos_includes_job_summaries_newest_first():
    setup()
    first = make_video(video_id="old")
    first["uploaded_at"] = 10.0
    second = make_video(video_id="new")
    second["uploaded_at"] = 99.0
    job = server.Job(id="j", video_id="old", mode="all", status="done")
    job.events = [{"frame": 1}]
    job.clips = [{"file": "a.mp4"}]
    server._jobs["j"] = job

    listed = server.list_videos()["videos"]
    assert [v["id"] for v in listed] == ["new", "old"], listed
    assert "path" not in listed[0] and "dir" not in listed[0]
    old = next(v for v in listed if v["id"] == "old")
    assert old["jobs"] == [{"id": "j", "status": "done", "mode": "all",
                            "shots": 1, "clips": 1}], old["jobs"]
    teardown()


# ---------------------------------------------------------- rim suggestion

def test_suggest_rim_finds_a_synthetic_rim_and_shrugs_at_an_empty_frame():
    setup()
    make_video(video_id="withrim", with_rim=True)
    box = server.suggest_rim("withrim")["box"]
    assert box is not None, "the synthetic rim should be found"
    assert box["w"] > box["h"], box
    assert box["frames_agreeing"] >= 2, box

    make_video(video_id="blank", with_rim=False)
    assert server.suggest_rim("blank")["box"] is None
    teardown()


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
        except Exception as exc:  # a crash is a failure too
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    teardown()
    sys.exit(1 if failed else 0)
