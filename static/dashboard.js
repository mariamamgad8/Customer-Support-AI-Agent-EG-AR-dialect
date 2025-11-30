// static/dashboard.js - simple polling dashboard client
const callsListEl = document.getElementById("calls-list");
const detailEl = document.getElementById("call-detail");
const statusEl = document.getElementById("status");

let selectedSid = null;
let pollingInterval = 2000;

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return await res.json();
}

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function showStatus(msg) {
  statusEl.textContent = msg;
}

async function loadCalls() {
  try {
    const data = await apiGet("/api/calls");
    showStatus(`Last update: ${new Date().toLocaleTimeString()}`);
    renderCalls(data);
    // if selectedSid is gone, clear detail
    if (selectedSid && !data.find(c => c.sid === selectedSid)) {
      selectedSid = null;
      detailEl.innerHTML = `<div class="placeholder">Select a call to view details</div>`;
    }
  } catch (err) {
    showStatus("Error loading calls: " + err.message);
    console.error(err);
  }
}

function renderCalls(calls) {
  callsListEl.innerHTML = "";
  if (!calls.length) {
    callsListEl.innerHTML = `<div class="placeholder">No calls yet</div>`;
    return;
  }
  calls.sort((a,b) => (b.last_ts||0) - (a.last_ts||0));
  for (const c of calls) {
    const div = document.createElement("div");
    div.className = "call-item";
    div.dataset.sid = c.sid;
    div.innerHTML = `<strong>${c.sid}</strong>
      <div class="meta">${c.turns} turn(s) • last: ${formatTime(c.last_ts)}</div>
      <div class="meta">Last user: ${c.last_user || "—"}</div>`;
    div.onclick = () => selectCall(c.sid);
    callsListEl.appendChild(div);
  }
}

async function selectCall(sid) {
  selectedSid = sid;
  try {
    const data = await apiGet(`/api/call/${encodeURIComponent(sid)}`);
    renderCallDetail(data);
  } catch (err) {
    detailEl.innerHTML = `<div class="placeholder">Failed to load call: ${err.message}</div>`;
  }
}

function renderCallDetail(data) {
  detailEl.innerHTML = "";
  const header = document.createElement("div");
  header.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center">
    <div><strong>Call: ${data.sid}</strong><div class="meta">Created: ${formatTime(data.created_ts)}</div></div>
    <div>
      <button id="clear-btn">Clear history</button>
    </div>
  </div>`;
  detailEl.appendChild(header);
  document.getElementById("clear-btn").onclick = async () => {
    await fetch(`/api/clear/${encodeURIComponent(data.sid)}`, { method: "POST" });
    await loadCalls();
    detailEl.innerHTML = `<div class="placeholder">History cleared</div>`;
  };

  // history
  if (!data.history || !data.history.length) {
    detailEl.innerHTML += `<div class="placeholder">No history for this call</div>`;
  } else {
    for (const item of data.history) {
      const d = document.createElement("div");
      d.className = "turn " + (item.role === "user" ? "user" : "assistant");
      const meta = `<div class="meta">${item.role} • ${formatTime(item.ts)} ${item.meta && item.meta.source ? " • " + item.meta.source : ""}</div>`;
      d.innerHTML = `${meta}<div>${escapeHtml(item.text)}</div>`;
      detailEl.appendChild(d);
    }
  }

  // debug files
  detailEl.appendChild(document.createElement("hr"));
  const filesTitle = document.createElement("div");
  filesTitle.innerHTML = `<strong>Debug audio files</strong>`;
  detailEl.appendChild(filesTitle);

  if (!data.debug_files || !data.debug_files.length) {
    detailEl.innerHTML += `<div class="placeholder">No debug files found for this call</div>`;
  } else {
    const aList = document.createElement("div");
    aList.className = "audio-list";
    for (const f of data.debug_files) {
      const item = document.createElement("div");
      item.className = "debug-file";
      const play = document.createElement("audio");
      play.controls = true;
      play.src = `/api/debug/${encodeURIComponent(f)}`;
      const label = document.createElement("div");
      label.textContent = f;
      item.appendChild(label);
      item.appendChild(play);
      aList.appendChild(item);
    }
    detailEl.appendChild(aList);
  }
}

function escapeHtml(s) {
  if (!s) return "";
  return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

// polling
loadCalls();
setInterval(loadCalls, pollingInterval);