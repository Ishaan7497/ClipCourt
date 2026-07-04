/* HoopClip frontend — vanilla JS, no build step.
   Flow: upload -> mark rim -> run job (polled) -> results. */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  video: null,        // metadata from POST /api/videos
  frameImg: null,     // Image() of the current preview frame
  frameScale: 1,      // source px per preview px
  box: null,          // {x,y,w,h} in PREVIEW pixel space
  job: null,
  pollTimer: null,
};

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

function setStepEnabled(id, enabled) {
  const el = $(id);
  el.setAttribute("aria-disabled", String(!enabled));
  if (enabled) el.scrollIntoView({ behavior: "smooth", block: "start" });
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
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

function uploadFile(file) {
  $("uploadError").textContent = "";
  $("fileMeta").textContent = "";
  $("uploadTrack").hidden = false;
  $("uploadStatus").textContent = `Uploading ${file.name}…`;

  const form = new FormData();
  form.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/videos");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      $("uploadFill").style.width = (e.loaded / e.total) * 100 + "%";
    }
  };
  xhr.onerror = () => {
    $("uploadError").textContent = "Upload failed — check that the server is running, then try again.";
    $("uploadStatus").textContent = "";
  };
  xhr.onload = () => {
    $("uploadTrack").hidden = true;
    $("uploadFill").style.width = "0%";
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
    if (xhr.status >= 400) {
      $("uploadError").textContent = data.detail || "Upload failed.";
      $("uploadStatus").textContent = "";
      return;
    }
    state.video = data;
    $("uploadStatus").textContent = "";
    $("fileMeta").innerHTML =
      `<strong>${data.display_name}</strong> · ${data.width}×${data.height} · ` +
      `${data.fps.toFixed(1)} fps · ${fmtTime(data.duration_s || 0)} · ${fmtBytes(data.size_bytes)}`;
    initRimStep();
  };
  xhr.send(form);
}

/* ------------------------------------------------------------- rim canvas */

const canvas = $("rimCanvas");
const ctx = canvas.getContext("2d");

function initRimStep() {
  const scrub = $("frameScrub");
  scrub.max = Math.max(0, (state.video.duration_s || 1) - 0.2).toFixed(1);
  scrub.value = 0;
  state.box = null;
  loadFrame(0);
  setStepEnabled("step-rim", true);
  setStepEnabled("step-run", true);
}

let frameDebounce = null;
$("frameScrub").addEventListener("input", (e) => {
  const t = parseFloat(e.target.value);
  $("scrubTime").textContent = fmtTime(t);
  clearTimeout(frameDebounce);
  frameDebounce = setTimeout(() => loadFrame(t), 180);
});

function loadFrame(t) {
  const img = new Image();
  img.onload = () => {
    state.frameImg = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    state.frameScale = state.video.width / img.naturalWidth;
    draw();
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

function announceBox() {
  if (!state.box) {
    $("rimStatus").textContent = "No rim box yet — drag on the frame to draw one.";
  } else {
    const b = sourceBox();
    $("rimStatus").textContent =
      `Rim box set: ${Math.round(b.w)}×${Math.round(b.h)} px at (${Math.round(b.x)}, ${Math.round(b.y)}).`;
  }
  syncFields();
}

function sourceBox() {
  const s = state.frameScale;
  const b = state.box;
  return b ? { x: b.x * s, y: b.y * s, w: b.w * s, h: b.h * s } : null;
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

$("startBtn").addEventListener("click", startJob);
$("cancelBtn").addEventListener("click", async () => {
  if (state.job) await fetch(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
});
$("restartBtn").addEventListener("click", () => location.reload());

async function startJob() {
  $("jobError").textContent = "";
  const b = sourceBox();
  if (!b) {
    $("jobError").textContent = "Mark the rim first (step 2) — detection needs to know where the hoop is.";
    $("step-rim").scrollIntoView({ behavior: "smooth" });
    return;
  }
  const mode = document.querySelector('input[name="mode"]:checked').value;

  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: state.video.id,
      hoop_box: b,
      mode,
      clip_before_seconds: +$("beforeS").value || 3,
      clip_after_seconds: +$("afterS").value || 1,
    }),
  });
  if (!res.ok) {
    $("jobError").textContent = await apiError(res);
    return;
  }
  state.job = await res.json();
  $("startBtn").disabled = true;
  $("cancelBtn").hidden = false;
  $("jobTrack").hidden = false;
  $("eventFeed").innerHTML = "";
  poll();
}

function poll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    const res = await fetch(`/api/jobs/${state.job.id}`);
    if (!res.ok) {
      $("jobError").textContent = await apiError(res);
      resetRunControls();
      return;
    }
    const job = await res.json();
    renderJob(job);
    if (["done", "error", "cancelled"].includes(job.status)) {
      resetRunControls();
      if (job.status === "done") showResults(job);
      if (job.status === "error") $("jobError").textContent = job.error || "Processing failed.";
    } else {
      poll();
    }
  }, 1000);
}

function resetRunControls() {
  $("startBtn").disabled = false;
  $("cancelBtn").hidden = true;
}

function renderJob(job) {
  const pct = Math.round(job.progress * 100);
  $("jobFill").style.width = pct + "%";
  $("jobFill").setAttribute("aria-valuenow", String(pct));
  $("jobStatus").textContent = `${pct}% — ${job.message}`;

  const feed = $("eventFeed");
  while (feed.children.length < job.events.length) {
    const ev = job.events[feed.children.length];
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${fmtTime(ev.time_s)}</span>` +
      `<span class="badge ${ev.result}">${ev.result}</span>` +
      `<span class="badge confidence">${ev.confidence} confidence</span>`;
    feed.appendChild(li);
  }
}

/* ---------------------------------------------------------------- results */

function showResults(job) {
  setStepEnabled("step-results", true);
  const dur = state.video.duration_s || 1;
  const made = job.events.filter((e) => e.result === "made").length;
  const miss = job.events.length - made;
  $("resultsSummary").textContent =
    `${job.events.length} shot${job.events.length === 1 ? "" : "s"} detected ` +
    `(${made} made, ${miss} missed) — ${job.clips.length} clip${job.clips.length === 1 ? "" : "s"} cut.`;

  // Timeline strip
  const tl = $("timeline");
  tl.innerHTML = "";
  $("timelineEnd").textContent = fmtTime(dur);
  job.events.forEach((ev, i) => {
    const tick = document.createElement("button");
    tick.className = `tick ${ev.result}`;
    tick.style.left = (ev.time_s / dur) * 100 + "%";
    tick.setAttribute("aria-label",
      `${ev.result === "made" ? "Made basket" : "Missed shot"} at ${fmtTime(ev.time_s)}. Jump to clip.`);
    tick.addEventListener("click", () => {
      const card = document.querySelector(`[data-clip-for="${i}"]`);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.querySelector("video").focus();
      }
    });
    tl.appendChild(tick);
  });

  // Clip cards
  const grid = $("clipGrid");
  grid.innerHTML = "";
  job.clips.forEach((clip) => {
    const url = `/api/jobs/${job.id}/clips/${encodeURIComponent(clip.file)}`;
    const card = document.createElement("article");
    card.className = "clip-card";
    const eventIdx = job.events.findIndex((e) =>
      clip.events.some((ce) => ce.frame === e.frame));
    if (eventIdx >= 0) card.dataset.clipFor = String(eventIdx);

    const badges = clip.events
      .map((e) => `<span class="badge ${e.result}">${e.result}</span>`)
      .join("");
    card.innerHTML =
      `<video controls preload="metadata" src="${url}"
              aria-label="Clip from ${fmtTime(clip.start_s)} to ${fmtTime(clip.end_s)}"></video>
       <div class="clip-body">
         <span class="clip-title">${fmtTime(clip.start_s)} – ${fmtTime(clip.end_s)}</span>
         <div class="badges">${badges}</div>
         <a class="download" href="${url}" download>Download clip</a>
       </div>`;
    grid.appendChild(card);
  });

  const zip = $("zipLink");
  if (job.clips.length) {
    zip.href = `/api/jobs/${job.id}/clips.zip`;
    zip.hidden = false;
  } else {
    zip.hidden = true;
  }
  $("restartBtn").hidden = false;
}
