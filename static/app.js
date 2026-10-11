/* ClipCourt frontend — vanilla JS, no build step.
   Flow: upload -> mark rim -> run job (polled) -> results. */

"use strict";

const $ = (id) => document.getElementById(id);

/* The queue is the source of truth for "which videos are we working on".
   state.video / state.box / state.job are views onto the SELECTED item, kept
   in sync by selectVideo() so the rim canvas and results only ever describe
   one video at a time. Rim boxes are stored per item in SOURCE pixels, since
   each video can have its own preview scale. */
const state = {
  queue: [],          // [{ videoId, video, box (source px|null), job }]
  selected: null,     // videoId
  video: null,        // metadata for the selected video
  frameImg: null,     // Image() of the current preview frame
  frameScale: 1,      // source px per preview px
  box: null,          // {x,y,w,h} in PREVIEW pixel space
  job: null,          // job snapshot shown in the results step
  pollTimer: null,
  roster: [],         // players from GET /api/videos/{id}/roster
  drillFilter: "all",
};

const ACTIVE_STATES = ["queued", "analyzing", "clipping"];

function queueItem(videoId) {
  return state.queue.find((q) => q.videoId === videoId) || null;
}

function selectedItem() {
  return queueItem(state.selected);
}

/* Copy the canvas box back onto the selected queue item — but only once its
   frame has actually loaded, or an early switch would erase a stored box
   (sourceBox() is null until there's a frame to measure against). */
function stashCurrentBox() {
  const item = selectedItem();
  if (item && state.frameImg) item.box = sourceBox();
}

/* ---------------------------------------------------------------- helpers */

function fmtTime(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function fmtBytes(n) {
  if (n > 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.round(n / 1e3) + " KB";
}

function setStepEnabled(id, enabled, scroll = true) {
  const el = $(id);
  el.setAttribute("aria-disabled", String(!enabled));
  if (enabled && scroll) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function apiError(res) {
  try {
    const data = await res.json();
    return data.detail || res.statusText;
  } catch {
    return res.statusText || `Request failed (${res.status})`;
  }
}

/* ----------------------------------------------------------------- upload */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length) {
    uploadFiles([...e.dataTransfer.files]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles([...fileInput.files]);
});

/* Uploads run one at a time on purpose: several 2 GB files in parallel just
   fight each other for bandwidth and make every progress bar misleading. */
async function uploadFiles(files) {
  $("uploadError").textContent = "";
  for (let i = 0; i < files.length; i += 1) {
    const label = files.length > 1 ? ` (${i + 1} of ${files.length})` : "";
    try {
      const data = await uploadOne(files[i], label);
      addToQueue(data);
      // Select the first upload of a batch so step 2 is immediately usable.
      if (i === 0 || !state.selected) await selectVideo(data.id);
    } catch (err) {
      $("uploadError").textContent =
        `${files[i].name}: ${err.message}` +
        (files.length > 1 ? " — continuing with the rest." : "");
    }
  }
  $("uploadStatus").textContent = "";
  $("uploadTrack").hidden = true;
  $("uploadFill").style.width = "0%";
  renderQueue();
}

function uploadOne(file, label) {
  return new Promise((resolve, reject) => {
    $("uploadTrack").hidden = false;
    $("uploadStatus").textContent = `Uploading ${file.name}${label}…`;
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/videos");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        $("uploadFill").style.width = (e.loaded / e.total) * 100 + "%";
      }
    };
    xhr.onerror = () =>
      reject(new Error("upload failed — check that the server is still running"));
    xhr.onload = () => {
      let data;
      try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
      if (xhr.status >= 400) reject(new Error(data.detail || "upload failed"));
      else resolve(data);
    };
    xhr.send(form);
  });
}

function addToQueue(video) {
  if (queueItem(video.id)) return;
  state.queue.push({ videoId: video.id, video, box: null, job: null });
}

/* ------------------------------------------------------------------ queue */

function jobBadge(job) {
  if (!job) return { cls: "pending", text: "Not run yet" };
  if (job.status === "done") {
    const shots = job.events ? job.events.length : job.shots || 0;
    return { cls: "done", text: `${shots} shot${shots === 1 ? "" : "s"}` };
  }
  if (job.status === "error") return { cls: "error", text: "Failed" };
  if (job.status === "cancelled") return { cls: "pending", text: "Cancelled" };
  return { cls: "running", text: `${Math.round((job.progress || 0) * 100)}%` };
}

function renderQueue() {
  const list = $("queueList");
  list.innerHTML = "";
  if (!state.queue.length) {
    const li = document.createElement("li");
    li.className = "queue-empty";
    li.textContent = "Nothing queued yet — choose one or more videos above.";
    list.appendChild(li);
    $("runAllBtn").hidden = true;
    return;
  }

  state.queue.forEach((item) => {
    const li = document.createElement("li");
    li.className = "queue-item" + (item.videoId === state.selected ? " selected" : "");

    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "queue-pick";
    pick.setAttribute("aria-pressed", String(item.videoId === state.selected));

    const name = document.createElement("span");
    name.className = "queue-name";
    name.textContent = item.video.display_name;

    const meta = document.createElement("span");
    meta.className = "queue-meta";
    meta.textContent =
      `${item.video.width}×${item.video.height} · ${fmtTime(item.video.duration_s || 0)}` +
      (item.box ? " · rim marked" : " · rim not marked");

    const badge = jobBadge(item.job);
    const status = document.createElement("span");
    status.className = `queue-badge ${badge.cls}`;
    status.textContent = badge.text;

    pick.append(name, meta, status);
    pick.addEventListener("click", () => selectVideo(item.videoId));

    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "queue-remove";
    rm.textContent = "✕";
    rm.setAttribute("aria-label",
      `Delete ${item.video.display_name} and any clips cut from it`);
    rm.addEventListener("click", () => removeFromQueue(item));

    li.append(pick, rm);
    list.appendChild(li);
  });

  const runnable = state.queue.filter(
    (q) => q.box && !(q.job && ACTIVE_STATES.includes(q.job.status)));
  $("runAllBtn").hidden = state.queue.length < 2 || !runnable.length;
  $("runAllBtn").textContent = `Run every marked video (${runnable.length})`;
}

async function removeFromQueue(item) {
  // This deletes the uploaded footage and its clips from the server, so it
  // needs to be a deliberate act rather than a stray click on a small ✕.
  const confirmed = window.confirm(
    `Delete "${item.video.display_name}" and any clips cut from it?\n\n` +
    "This removes the uploaded footage from the server and can't be undone.");
  if (!confirmed) return;
  if (item.job && ACTIVE_STATES.includes(item.job.status)) {
    await fetch(`/api/jobs/${item.job.id}/cancel`, { method: "POST" });
  }
  await fetch(`/api/videos/${item.videoId}`, { method: "DELETE" });
  state.queue = state.queue.filter((q) => q.videoId !== item.videoId);
  $("queueStatus").textContent = `${item.video.display_name} removed.`;
  if (state.selected === item.videoId) {
    state.selected = null;
    state.video = null;
    state.job = null;
    state.box = null;
    if (state.queue.length) await selectVideo(state.queue[0].videoId);
    else {
      setStepEnabled("step-rim", false);
      setStepEnabled("step-run", false);
      setStepEnabled("step-results", false);
    }
  }
  renderQueue();
}

async function selectVideo(videoId) {
  const item = queueItem(videoId);
  if (!item) return;

  // Stash the in-progress box of the video we're leaving, in source pixels.
  if (state.selected !== videoId) stashCurrentBox();

  state.selected = videoId;
  state.video = item.video;
  state.job = null;
  state.frameImg = null;
  state.box = null;
  $("jobError").textContent = "";
  $("eventFeed").innerHTML = "";
  $("fileMeta").innerHTML =
    `<strong>${esc(item.video.display_name)}</strong> · ${item.video.width}×${item.video.height} · ` +
    `${(item.video.fps || 0).toFixed(1)} fps · ${fmtTime(item.video.duration_s || 0)} · ` +
    fmtBytes(item.video.size_bytes || 0);

  initRimStep(item);
  renderQueue();

  setStepEnabled("step-results", false);
  if (!item.job) return;

  // The queue may only hold a summary (id + status) for jobs restored from an
  // earlier session, so fetch the full snapshot before rendering anything
  // that expects events, progress and clips.
  const res = await fetch(`/api/jobs/${item.job.id}`);
  if (!res.ok) return;
  const job = await res.json();
  item.job = job;
  state.job = job;
  renderJob(job);
  if (job.status === "done") await showResults(job, { scroll: false });
  else if (job.status === "error") {
    $("jobError").textContent = job.error || "Processing failed.";
  }
  resetRunControls();
  renderQueue();
}

/* ------------------------------------------------------------- rim canvas */

const canvas = $("rimCanvas");
const ctx = canvas.getContext("2d");

function initRimStep(item) {
  const scrub = $("frameScrub");
  scrub.max = Math.max(0, (state.video.duration_s || 1) - 0.2).toFixed(1);
  scrub.value = 0;
  $("scrubTime").textContent = "0:00";
  state.box = null;
  // Restore this video's own rim box once the frame (and its scale) is loaded.
  loadFrame(0, item && item.box ? item.box : null);
  setStepEnabled("step-rim", true, false);
  setStepEnabled("step-run", true, false);
}

let frameDebounce = null;
$("frameScrub").addEventListener("input", (e) => {
  const t = parseFloat(e.target.value);
  $("scrubTime").textContent = fmtTime(t);
  clearTimeout(frameDebounce);
  frameDebounce = setTimeout(() => loadFrame(t), 180);
});

function loadFrame(t, restoreSourceBox = null) {
  const img = new Image();
  // Which video this frame belongs to: switching queue items mid-load must
  // not drop the old video's frame onto the newly selected one.
  const forVideo = state.selected;
  img.onload = () => {
    if (state.selected !== forVideo) return;
    state.frameImg = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    state.frameScale = state.video.width / img.naturalWidth;
    if (restoreSourceBox) {
      setSourceBox(restoreSourceBox, { announce: false });
    }
    draw();
    announceBox();
  };
  img.onerror = () => {
    $("rimStatus").textContent = "Couldn't load that frame — try a different position.";
  };
  img.src = `/api/videos/${state.video.id}/frame?t=${t}&_=${Date.now()}`;
}

function draw() {
  if (!state.frameImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.frameImg, 0, 0);
  if (state.box) {
    const { x, y, w, h } = state.box;
    // Telestrator stroke: tape-yellow marker with a soft shadow.
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 6;
    ctx.strokeStyle = "#f2c14e";
    ctx.lineWidth = Math.max(3, canvas.width / 300);
    ctx.strokeRect(x, y, w, h);
    // Corner grip so the resize affordance is visible.
    ctx.fillStyle = "#f2c14e";
    const g = ctx.lineWidth * 2.2;
    ctx.fillRect(x + w - g / 2, y + h - g / 2, g, g);
    ctx.restore();
  }
}

function announceBox(message) {
  if (!state.box) {
    $("rimStatus").textContent = "No rim box yet — drag on the frame to draw one.";
  } else {
    const b = sourceBox();
    $("rimStatus").textContent = (message ? message + " " : "") +
      `Rim box set: ${Math.round(b.w)}×${Math.round(b.h)} px at (${Math.round(b.x)}, ${Math.round(b.y)}).`;
  }
  // The queue item owns the box; state.box is just the canvas-space view.
  stashCurrentBox();
  syncFields();
  renderQueue();
}

function sourceBox() {
  const s = state.frameScale;
  const b = state.box;
  return b ? { x: b.x * s, y: b.y * s, w: b.w * s, h: b.h * s } : null;
}

function setSourceBox(b, { announce = true, message } = {}) {
  const s = state.frameScale || 1;
  state.box = { x: b.x / s, y: b.y / s, w: b.w / s, h: b.h / s };
  clampBox();
  draw();
  if (announce) announceBox(message);
}

function syncFields() {
  const b = sourceBox();
  $("boxX").value = b ? Math.round(b.x) : "";
  $("boxY").value = b ? Math.round(b.y) : "";
  $("boxW").value = b ? Math.round(b.w) : "";
  $("boxH").value = b ? Math.round(b.h) : "";
}

["boxX", "boxY", "boxW", "boxH"].forEach((id) =>
  $(id).addEventListener("change", () => {
    const s = state.frameScale;
    const x = +$("boxX").value, y = +$("boxY").value;
    const w = +$("boxW").value, h = +$("boxH").value;
    if (w > 0 && h > 0) {
      state.box = { x: x / s, y: y / s, w: w / s, h: h / s };
      clampBox();
      draw();
      announceBox();
    }
  })
);

/* ----- rim tools: auto-suggest, copy to other videos ----- */

$("suggestRimBtn").addEventListener("click", async () => {
  if (!state.video) return;
  const btn = $("suggestRimBtn");
  btn.disabled = true;
  $("rimStatus").textContent = "Looking for the rim…";
  try {
    const res = await fetch(`/api/videos/${state.video.id}/suggest-rim`);
    if (!res.ok) throw new Error(await apiError(res));
    const { box } = await res.json();
    if (!box) {
      $("rimStatus").textContent =
        "Couldn't find the rim automatically — draw the box by hand instead.";
      return;
    }
    setSourceBox(box, {
      message: `Suggested from ${box.frames_agreeing} of ${box.frames_sampled} sampled frames — check it and adjust.`,
    });
  } catch (err) {
    $("rimStatus").textContent = `Rim suggestion failed — ${err.message}`;
  } finally {
    btn.disabled = false;
  }
});

$("applyRimAllBtn").addEventListener("click", () => {
  const b = sourceBox();
  if (!b) {
    $("rimStatus").textContent = "Mark a rim box first, then copy it to the others.";
    return;
  }
  // Only same-size footage: a box in pixel coordinates means nothing against
  // a different frame size, and a wrong rim box silently ruins detection.
  const targets = state.queue.filter(
    (q) => q.videoId !== state.selected &&
      q.video.width === state.video.width && q.video.height === state.video.height);
  targets.forEach((q) => { q.box = { ...b }; });
  const skipped = state.queue.length - 1 - targets.length;
  $("rimStatus").textContent = targets.length
    ? `Rim box copied to ${targets.length} video${targets.length === 1 ? "" : "s"}` +
      (skipped ? `; ${skipped} skipped for a different frame size.` : ".")
    : "No other videos share this frame size, so the box wasn't copied.";
  renderQueue();
});

function clampBox() {
  const b = state.box;
  if (!b) return;
  b.w = Math.min(Math.max(4, b.w), canvas.width);
  b.h = Math.min(Math.max(4, b.h), canvas.height);
  b.x = Math.min(Math.max(0, b.x), canvas.width - b.w);
  b.y = Math.min(Math.max(0, b.y), canvas.height - b.h);
}

/* pointer interactions: draw new box, move existing, resize via corner grip */
let drag = null;

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvas.width / r.width),
    y: (e.clientY - r.top) * (canvas.height / r.height),
  };
}

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  const p = canvasPoint(e);
  const b = state.box;
  const grip = b && Math.abs(p.x - (b.x + b.w)) < 18 && Math.abs(p.y - (b.y + b.h)) < 18;
  const inside = b && p.x > b.x && p.x < b.x + b.w && p.y > b.y && p.y < b.y + b.h;
  if (grip) drag = { kind: "resize" };
  else if (inside) drag = { kind: "move", dx: p.x - b.x, dy: p.y - b.y };
  else {
    state.box = { x: p.x, y: p.y, w: 1, h: 1 };
    drag = { kind: "draw", ox: p.x, oy: p.y };
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const p = canvasPoint(e);
  const b = state.box;
  if (drag.kind === "draw") {
    b.x = Math.min(drag.ox, p.x);
    b.y = Math.min(drag.oy, p.y);
    b.w = Math.abs(p.x - drag.ox);
    b.h = Math.abs(p.y - drag.oy);
  } else if (drag.kind === "move") {
    b.x = p.x - drag.dx;
    b.y = p.y - drag.dy;
  } else if (drag.kind === "resize") {
    b.w = p.x - b.x;
    b.h = p.y - b.y;
  }
  clampBox();
  draw();
});

canvas.addEventListener("pointerup", () => {
  if (drag && state.box && (state.box.w < 4 || state.box.h < 4)) state.box = null;
  drag = null;
  draw();
  announceBox();
});

/* keyboard: arrows move, shift+arrows resize, delete clears */
canvas.addEventListener("keydown", (e) => {
  const step = e.altKey ? 1 : 8;
  const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Delete", "Backspace"];
  if (!keys.includes(e.key)) return;
  e.preventDefault();

  if (e.key === "Delete" || e.key === "Backspace") {
    state.box = null;
    draw();
    announceBox();
    return;
  }
  if (!state.box) {
    // Seed a starter box in the center so keyboard-only users can begin.
    state.box = {
      x: canvas.width * 0.45, y: canvas.height * 0.25,
      w: canvas.width * 0.1, h: canvas.width * 0.05,
    };
  }
  const b = state.box;
  const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
  const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
  if (e.shiftKey) { b.w += dx; b.h += dy; }
  else { b.x += dx; b.y += dy; }
  clampBox();
  draw();
  announceBox();
});

/* -------------------------------------------------------------------- job */

$("startBtn").addEventListener("click", async () => {
  $("jobError").textContent = "";
  const item = selectedItem();
  if (!item) {
    $("jobError").textContent = "Upload a video first.";
    return;
  }
  if (!sourceBox()) {
    $("jobError").textContent =
      "Mark the rim first (step 2) — detection needs to know where the hoop is.";
    $("step-rim").scrollIntoView({ behavior: "smooth" });
    return;
  }
  item.box = sourceBox();
  const started = await startJobFor(item);
  if (started) {
    $("eventFeed").innerHTML = "";
    $("jobTrack").hidden = false;
  }
});

$("runAllBtn").addEventListener("click", async () => {
  $("jobError").textContent = "";
  stashCurrentBox();
  const runnable = state.queue.filter(
    (q) => q.box && !(q.job && ACTIVE_STATES.includes(q.job.status)));
  if (!runnable.length) {
    $("jobError").textContent =
      "No queued video has a rim box yet. Mark one, then copy it to the others.";
    return;
  }
  $("jobTrack").hidden = false;
  for (const item of runnable) await startJobFor(item);
  $("queueStatus").textContent =
    `Started ${runnable.length} job${runnable.length === 1 ? "" : "s"}. ` +
    "They run a couple at a time; the queue shows each one's progress.";
});

$("cancelBtn").addEventListener("click", async () => {
  const item = selectedItem();
  if (item && item.job) {
    await fetch(`/api/jobs/${item.job.id}/cancel`, { method: "POST" });
  }
});

$("restartBtn").addEventListener("click", () => location.reload());

async function startJobFor(item) {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: item.videoId,
      hoop_box: item.box,
      mode,
      clip_before_seconds: +$("beforeS").value || 3,
      clip_after_seconds: +$("afterS").value || 1,
      vertical_reframe: $("verticalChk").checked,
    }),
  });
  if (!res.ok) {
    $("jobError").textContent =
      `${item.video.display_name}: ${await apiError(res)}`;
    return false;
  }
  item.job = await res.json();
  if (item.videoId === state.selected) {
    state.job = item.job;
    $("startBtn").disabled = true;
    $("cancelBtn").hidden = false;
  }
  renderQueue();
  poll();
  return true;
}

/* One ticker polls every in-flight job: a batch of ten videos shouldn't mean
   ten independent timers drifting against each other. */
function poll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    const active = state.queue.filter(
      (q) => q.job && ACTIVE_STATES.includes(q.job.status));
    if (!active.length) {
      resetRunControls();
      return;
    }
    await Promise.all(active.map(async (item) => {
      const res = await fetch(`/api/jobs/${item.job.id}`);
      if (!res.ok) {
        item.job = { ...item.job, status: "error", error: await apiError(res) };
        return;
      }
      const job = await res.json();
      const wasActive = ACTIVE_STATES.includes(item.job.status);
      item.job = job;
      if (item.videoId === state.selected) {
        state.job = job;
        renderJob(job);
        if (job.status === "done") await showResults(job);
        if (job.status === "error") {
          $("jobError").textContent = job.error || "Processing failed.";
        }
      } else if (wasActive && job.status === "done") {
        $("queueStatus").textContent =
          `${item.video.display_name} finished — ${job.events.length} ` +
          `shot${job.events.length === 1 ? "" : "s"}. Select it to see the clips.`;
      }
    }));
    renderQueue();
    poll();
  }, 1000);
}

function resetRunControls() {
  const item = selectedItem();
  const busy = item && item.job && ACTIVE_STATES.includes(item.job.status);
  $("startBtn").disabled = Boolean(busy);
  $("cancelBtn").hidden = !busy;
}

function renderJob(job) {
  const events = job.events || [];
  const pct = Math.round((job.progress || 0) * 100);
  $("jobFill").style.width = pct + "%";
  $("jobFill").setAttribute("aria-valuenow", String(pct));
  $("jobStatus").textContent = `${pct}% — ${job.message || job.status}`;

  const feed = $("eventFeed");
  // Switching videos means the feed belongs to a different job: rebuild it
  // rather than appending this job's shots onto the previous one's.
  if (feed.children.length > events.length) feed.innerHTML = "";
  while (feed.children.length < events.length) {
    const ev = events[feed.children.length];
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${fmtTime(ev.time_s)}</span>` +
      `<span class="badge ${ev.result}">${ev.result}</span>` +
      `<span class="badge confidence">${esc(ev.confidence)} confidence</span>`;
    feed.appendChild(li);
  }
}

/* ---------------------------------------------------------------- results */

async function showResults(job, { scroll = true } = {}) {
  state.job = job;
  setStepEnabled("step-results", true, scroll);
  const dur = state.video.duration_s || 1;
  const made = job.events.filter((e) => e.result === "made").length;
  const miss = job.events.length - made;
  const on = job.device ? ` Analyzed on ${DEVICE_LABELS[job.device] || job.device}.` : "";
  $("resultsSummary").textContent =
    `${job.events.length} shot${job.events.length === 1 ? "" : "s"} detected ` +
    `(${made} made, ${miss} missed) — ${job.clips.length} clip${job.clips.length === 1 ? "" : "s"} cut.` + on;

  await loadRoster();

  // Timeline strip
  const tl = $("timeline");
  tl.innerHTML = "";
  $("timelineEnd").textContent = fmtTime(dur);
  job.events.forEach((ev) => {
    const tick = document.createElement("button");
    tick.className = `tick ${ev.result}`;
    tick.style.left = (ev.time_s / dur) * 100 + "%";
    tick.setAttribute("aria-label",
      `${ev.result === "made" ? "Made basket" : "Missed shot"} at ${fmtTime(ev.time_s)}. Jump to clip.`);
    tick.addEventListener("click", () => focusShot(ev.frame, "video"));
    tl.appendChild(tick);
  });

  // Clip cards (with per-shot player/drill pickers)
  const grid = $("clipGrid");
  grid.innerHTML = "";
  job.clips.forEach((clip) => {
    const url = `/api/jobs/${job.id}/clips/${encodeURIComponent(clip.file)}`;
    const card = document.createElement("article");
    card.className = "clip-card";

    const badges = clip.events
      .map((e) => `<span class="badge ${e.result}">${e.result}</span>`)
      .join("");
    const vertical = clip.vertical_file
      ? `<a class="download" href="/api/jobs/${job.id}/clips/${encodeURIComponent(clip.vertical_file)}"
            download>Download 9:16 version</a>`
      : "";
    card.innerHTML =
      `<video controls preload="metadata" src="${url}"
              aria-label="Clip from ${fmtTime(clip.start_s)} to ${fmtTime(clip.end_s)}"></video>
       <div class="clip-body">
         <span class="clip-title">${fmtTime(clip.start_s)} – ${fmtTime(clip.end_s)}</span>
         <div class="badges">${badges}</div>
         <a class="download" href="${url}" download>Download clip</a>
         ${vertical}
       </div>`;

    const body = card.querySelector(".clip-body");
    const download = body.querySelector("a.download");
    clip.events.forEach((ce) => {
      const ev = job.events.find((e) => e.frame === ce.frame);
      if (ev) body.insertBefore(assignmentRow(ev), download);
    });
    grid.appendChild(card);
  });

  renderCourtDiagram(job);
  renderStats(job);

  const csv = $("csvLink");
  const pdf = $("pdfLink");
  csv.hidden = pdf.hidden = !job.events.length;
  if (job.events.length) {
    csv.href = `/api/jobs/${job.id}/export.csv`;
    pdf.href = `/api/jobs/${job.id}/export.pdf`;
  }

  const zip = $("zipLink");
  if (job.clips.length) {
    zip.href = `/api/jobs/${job.id}/clips.zip`;
    zip.hidden = false;
  } else {
    zip.hidden = true;
  }
  $("restartBtn").hidden = false;
}

/* --------------------------------------------------------------- coaching */

const DRILL_LABELS = {
  layups: "Layups",
  three_pointers: "3-Pointers",
  midrange: "Mid-range",
};
const DEVICE_LABELS = { cuda: "an NVIDIA GPU", mps: "the Apple GPU", cpu: "the CPU" };
const COURT_HOOP_Y = 5.25; // hoop center, feet from the baseline (matches the SVG)

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function playerById(id) {
  return state.roster.find((p) => p.id === id) || null;
}

function playerLabel(id) {
  if (!id) return "Unassigned";
  const p = playerById(id);
  return p ? p.name : "(removed player)";
}

/* ----- roster ----- */

async function loadRoster() {
  try {
    const res = await fetch(`/api/videos/${state.video.id}/roster`);
    if (!res.ok) throw new Error(await apiError(res));
    state.roster = (await res.json()).players || [];
  } catch (err) {
    $("rosterStatus").textContent = `Couldn't load the roster — ${err.message}`;
    state.roster = [];
  }
  renderRoster();
}

function renderRoster() {
  const list = $("rosterList");
  list.innerHTML = "";
  state.roster.forEach((p) => {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = p.color;
    swatch.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.className = "roster-name";
    name.textContent = p.name;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "roster-remove";
    rm.textContent = "✕";
    rm.setAttribute("aria-label", `Remove ${p.name} from the roster`);
    rm.addEventListener("click", () => removePlayer(p));
    li.append(swatch, name, rm);
    list.appendChild(li);
  });
  if (!state.roster.length) {
    const li = document.createElement("li");
    li.className = "roster-empty";
    li.textContent = "No players yet — add one above to start tagging shots.";
    list.appendChild(li);
  }
  refreshPlayerSelects();
  renderLegend();
}

async function addPlayer() {
  const input = $("playerName");
  const name = input.value.trim();
  if (!name) {
    $("rosterStatus").textContent = "Type a player name first.";
    input.focus();
    return;
  }
  const res = await fetch(`/api/videos/${state.video.id}/roster`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    $("rosterStatus").textContent = await apiError(res);
    return;
  }
  const player = await res.json();
  state.roster.push(player);
  input.value = "";
  input.focus();
  renderRoster();
  if (state.job) renderStats(state.job);
  $("rosterStatus").textContent = `${player.name} added to the roster.`;
}

async function removePlayer(player) {
  const res = await fetch(
    `/api/videos/${state.video.id}/roster/${player.id}`, { method: "DELETE" });
  if (!res.ok) {
    $("rosterStatus").textContent = await apiError(res);
    return;
  }
  state.roster = state.roster.filter((p) => p.id !== player.id);
  renderRoster();
  if (state.job) {
    renderCourtDiagram(state.job);
    renderStats(state.job);
  }
  $("rosterStatus").textContent =
    `${player.name} removed. Shots tagged to them now show as "(removed player)".`;
}

$("addPlayerBtn").addEventListener("click", addPlayer);
$("playerName").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    addPlayer();
  }
});

/* ----- per-shot assignment ----- */

function fillPlayerOptions(sel, current) {
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "Unassigned";
  sel.appendChild(none);
  state.roster.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    sel.appendChild(o);
  });
  if (current && !state.roster.some((p) => p.id === current)) {
    const o = document.createElement("option");
    o.value = current;
    o.textContent = "(removed player)";
    sel.appendChild(o);
  }
  sel.value = current || "";
}

function refreshPlayerSelects() {
  document.querySelectorAll('select[data-kind="player"]')
    .forEach((sel) => fillPlayerOptions(sel, sel.value));
}

function assignmentRow(ev) {
  const row = document.createElement("div");
  row.className = "assign-row";
  row.dataset.frame = String(ev.frame);

  const time = document.createElement("span");
  time.className = "assign-time";
  time.textContent = `${fmtTime(ev.time_s)} ${ev.result}`;

  const playerSel = document.createElement("select");
  playerSel.dataset.kind = "player";
  playerSel.setAttribute("aria-label",
    `Player for the ${ev.result} at ${fmtTime(ev.time_s)}`);
  fillPlayerOptions(playerSel, ev.player_id || "");

  const drillSel = document.createElement("select");
  drillSel.dataset.kind = "drill";
  drillSel.setAttribute("aria-label",
    `Drill type for the ${ev.result} at ${fmtTime(ev.time_s)}`);
  const noDrill = document.createElement("option");
  noDrill.value = "";
  noDrill.textContent = "No drill";
  drillSel.appendChild(noDrill);
  Object.entries(DRILL_LABELS).forEach(([value, label]) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    drillSel.appendChild(o);
  });
  drillSel.value = ev.drill_tag || "";

  const onChange = () =>
    assignShot(ev.frame, playerSel.value || null, drillSel.value || null);
  playerSel.addEventListener("change", onChange);
  drillSel.addEventListener("change", onChange);

  row.append(time, playerSel, drillSel);
  return row;
}

async function assignShot(frame, playerId, drillTag) {
  const ev = state.job.events.find((e) => e.frame === frame);
  const res = await fetch(`/api/jobs/${state.job.id}/shots/${frame}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_id: playerId, drill_tag: drillTag }),
  });
  if (!res.ok) {
    $("assignStatus").textContent =
      `Couldn't save that tag — ${await apiError(res)}`;
    const row = document.querySelector(`.assign-row[data-frame="${frame}"]`);
    if (row && ev) {
      fillPlayerOptions(row.querySelector('select[data-kind="player"]'), ev.player_id || "");
      row.querySelector('select[data-kind="drill"]').value = ev.drill_tag || "";
    }
    return;
  }
  const saved = await res.json();
  if (ev) {
    ev.player_id = saved.player_id;
    ev.drill_tag = saved.drill_tag;
  }
  renderCourtDiagram(state.job);
  renderStats(state.job);
  $("assignStatus").textContent =
    `${fmtTime(ev.time_s)} ${ev.result} tagged: ${playerLabel(saved.player_id)}` +
    (saved.drill_tag ? `, ${DRILL_LABELS[saved.drill_tag]}` : "") + ".";
}

function focusShot(frame, target) {
  const row = document.querySelector(`.assign-row[data-frame="${frame}"]`);
  const card = row && row.closest(".clip-card");
  if (!card) {
    $("courtStatus").textContent =
      "No clip was cut for this shot (it may be filtered out by the clip mode).";
    return;
  }
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  if (target === "player") {
    row.querySelector('select[data-kind="player"]').focus({ preventScroll: true });
  } else {
    card.querySelector("video").focus({ preventScroll: true });
  }
}

/* ----- court diagram ----- */

function courtPct(xFt, distFt) {
  const cx = Math.max(-24, Math.min(24, xFt));
  const cy = Math.max(0, Math.min(41, distFt));
  return {
    left: ((25 + cx) / 50) * 100,
    top: ((COURT_HOOP_Y + cy) / 47) * 100,
  };
}

function shotLabel(ev, idx) {
  let label = `Shot ${idx + 1}: ${ev.result === "made" ? "made" : "missed"} ` +
    `at ${fmtTime(ev.time_s)}, ${playerLabel(ev.player_id)}`;
  if (ev.drill_tag) label += `, ${DRILL_LABELS[ev.drill_tag]}`;
  return label;
}

function renderCourtDiagram(job) {
  const wrap = $("courtDiagram");
  wrap.querySelectorAll(".shot-marker").forEach((n) => n.remove());
  let placed = 0;
  job.events.forEach((ev, i) => {
    if (ev.court_x_ft == null || ev.court_dist_ft == null) return;
    placed += 1;
    const { left, top } = courtPct(ev.court_x_ft, ev.court_dist_ft);
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = `shot-marker ${ev.result}`;
    marker.dataset.frame = String(ev.frame);
    marker.dataset.drill = ev.drill_tag || "";
    marker.style.left = left + "%";
    marker.style.top = top + "%";
    const p = playerById(ev.player_id);
    if (p) marker.style.borderColor = p.color;
    marker.setAttribute("aria-label", shotLabel(ev, i) + ". Activate to tag this shot.");
    const announce = () => { $("courtStatus").textContent = shotLabel(ev, i); };
    marker.addEventListener("focus", announce);
    marker.addEventListener("mouseenter", announce);
    marker.addEventListener("click", () => focusShot(ev.frame, "player"));
    wrap.appendChild(marker);
  });
  if (!placed) {
    $("courtStatus").textContent = job.events.length
      ? "No location data for these shots — the chart fills in for videos analyzed from now on."
      : "No shots detected.";
  }
  applyDrillFilter();
}

function renderLegend() {
  $("courtLegend").innerHTML =
    '<li><span class="legend-swatch made" aria-hidden="true"></span> Made</li>' +
    '<li><span class="legend-swatch miss" aria-hidden="true"></span> Missed</li>' +
    state.roster.map((p) =>
      `<li><span class="legend-swatch ring" style="border-color:${esc(p.color)}" ` +
      `aria-hidden="true"></span> ${esc(p.name)}</li>`).join("");
}

/* ----- stats dashboard ----- */

function locationBucket(ev) {
  if (ev.court_x_ft == null || ev.court_dist_ft == null) return null;
  const d = Math.hypot(ev.court_x_ft, ev.court_dist_ft);
  return d < 8 ? "close" : d <= 20 ? "mid" : "deep";
}

function tally(evs) {
  const made = evs.filter((e) => e.result === "made").length;
  return {
    att: evs.length,
    made,
    pct: evs.length ? Math.round((100 * made) / evs.length) : null,
  };
}

function statCard(title, evs, color) {
  const { att, made, pct } = tally(evs);
  const card = document.createElement("article");
  card.className = "stat-card";
  const h = document.createElement("h4");
  if (color) {
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = color;
    sw.setAttribute("aria-hidden", "true");
    h.appendChild(sw);
  }
  h.appendChild(document.createTextNode(title));
  const big = document.createElement("p");
  big.className = "stat-pct";
  big.textContent = pct == null ? "—" : pct + "%";
  const detail = document.createElement("p");
  detail.className = "stat-detail";
  detail.textContent =
    `${made} made / ${att - made} missed (${att} shot${att === 1 ? "" : "s"})`;
  card.append(h, big, detail);

  const buckets = { close: [], mid: [], deep: [] };
  let anyLocated = false;
  evs.forEach((e) => {
    const b = locationBucket(e);
    if (b) {
      buckets[b].push(e);
      anyLocated = true;
    }
  });
  if (anyLocated) {
    const loc = document.createElement("p");
    loc.className = "stat-detail stat-loc";
    loc.textContent = ["close", "mid", "deep"].map((b) => {
      const t = tally(buckets[b]);
      return `${b[0].toUpperCase()}${b.slice(1)} ${t.made}/${t.att}`;
    }).join(" · ");
    card.appendChild(loc);
  }
  return card;
}

function renderStats(job) {
  const dash = $("statsDash");
  dash.innerHTML = "";
  if (!job.events.length) {
    dash.innerHTML = '<p class="hint">Stats appear once shots are detected.</p>';
    return;
  }
  const groups = new Map(); // player_id (or "") -> events
  job.events.forEach((ev) => {
    const key = ev.player_id || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(ev);
  });
  state.roster.forEach((p) => {
    const evs = groups.get(p.id);
    if (evs) dash.appendChild(statCard(p.name, evs, p.color));
  });
  groups.forEach((evs, key) => {
    if (key && !playerById(key)) dash.appendChild(statCard("(removed player)", evs, null));
  });
  if (groups.has("")) dash.appendChild(statCard("Unassigned", groups.get(""), null));
  const team = statCard("Team", job.events, null);
  team.classList.add("team");
  dash.appendChild(team);
}

/* ----- drill filter ----- */

document.querySelectorAll('input[name="drill"]').forEach((radio) =>
  radio.addEventListener("change", () => {
    state.drillFilter = radio.value;
    applyDrillFilter();
  })
);

function applyDrillFilter() {
  const tag = state.drillFilter || "all";
  document.querySelectorAll(".shot-marker").forEach((m) => {
    m.hidden = tag !== "all" && m.dataset.drill !== tag;
  });
  const events = state.job ? state.job.events : [];
  document.querySelectorAll(".clip-card").forEach((card) => {
    const frames = [...card.querySelectorAll(".assign-row")]
      .map((r) => Number(r.dataset.frame));
    const match = tag === "all" ||
      events.some((e) => frames.includes(e.frame) && e.drill_tag === tag);
    card.hidden = !match;
  });
  if (tag === "all") {
    $("drillStats").textContent = "";
    return;
  }
  const evs = events.filter((e) => e.drill_tag === tag);
  const { att, made, pct } = tally(evs);
  $("drillStats").textContent = att
    ? `${DRILL_LABELS[tag]}: ${att} shot${att === 1 ? "" : "s"}, ${made} made (${pct}%).`
    : `${DRILL_LABELS[tag]}: no shots tagged yet.`;
}

/* -------------------------------------------------------- session restore */

/* Footage and finished jobs outlive the page (and the server process), so the
   queue is rebuilt from the API instead of starting empty on every reload. */
async function restoreSession() {
  try {
    const res = await fetch("/api/videos");
    if (!res.ok) throw new Error(await apiError(res));
    const { videos } = await res.json();
    videos.forEach((v) => {
      const jobs = v.jobs || [];
      const done = [...jobs].reverse().find((j) => j.status === "done");
      state.queue.push({
        videoId: v.id,
        video: v,
        box: null,
        job: done || jobs[jobs.length - 1] || null,
      });
    });
    if (videos.length) {
      $("queueStatus").textContent =
        `${videos.length} video${videos.length === 1 ? "" : "s"} from earlier — ` +
        "select one to re-open its highlights, or upload more.";
    }
  } catch (err) {
    $("queueStatus").textContent = `Couldn't list earlier videos — ${err.message}`;
  }
  renderQueue();
}

restoreSession();
