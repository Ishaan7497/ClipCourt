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
  roster: [],         // players from GET /api/videos/{id}/roster
  drillFilter: "all",
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

async function showResults(job) {
  state.job = job;
  setStepEnabled("step-results", true);
  const dur = state.video.duration_s || 1;
  const made = job.events.filter((e) => e.result === "made").length;
  const miss = job.events.length - made;
  $("resultsSummary").textContent =
    `${job.events.length} shot${job.events.length === 1 ? "" : "s"} detected ` +
    `(${made} made, ${miss} missed) — ${job.clips.length} clip${job.clips.length === 1 ? "" : "s"} cut.`;

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
    card.innerHTML =
      `<video controls preload="metadata" src="${url}"
              aria-label="Clip from ${fmtTime(clip.start_s)} to ${fmtTime(clip.end_s)}"></video>
       <div class="clip-body">
         <span class="clip-title">${fmtTime(clip.start_s)} – ${fmtTime(clip.end_s)}</span>
         <div class="badges">${badges}</div>
         <a class="download" href="${url}" download>Download clip</a>
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
