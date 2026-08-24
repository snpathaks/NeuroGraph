/* ═══════════════════════════════════════════════════════════
   dashboard.js — NeuroGraph
   Handles: theme, tabs, canvas drawing, API calls to Flask.
═══════════════════════════════════════════════════════════ */

/* ── Theme toggle ─────────────────────────────── */
const html     = document.documentElement;
const themeBtn = document.getElementById('themeToggle');

themeBtn.addEventListener('click', () => {
  const isDark = html.dataset.theme === 'dark';
  html.dataset.theme = isDark ? 'light' : 'dark';
  themeBtn.textContent = isDark ? '🌙 Dark' : '☀️ Light';
  redraw();
});

/* ── Tab switching ────────────────────────────── */
function switchTab(id, el) {
  document.querySelectorAll('.tab-view').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}

/* ── Canvas drawing ───────────────────────────── */
const canvas  = document.getElementById('drawCanvas');
const ctx     = canvas.getContext('2d');
const hint    = document.getElementById('canvasHint');
const ptCount = document.getElementById('pointCount');
let drawing   = false;
let points    = [];

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  redraw();
}

function redraw() {
  const w = canvas.width  / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  if (points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  const isDark    = html.dataset.theme === 'dark';
  ctx.strokeStyle = isDark ? '#3b82f6' : '#2563eb';
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'round';
  ctx.lineJoin    = 'round';
  ctx.stroke();
}

function getPos(e) {
  const r   = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top, t: Date.now() };
}

canvas.addEventListener('mousedown',  e => { drawing = true; points.push(getPos(e)); hint.classList.add('hidden'); });
canvas.addEventListener('mousemove',  e => { if (!drawing) return; points.push(getPos(e)); redraw(); ptCount.textContent = points.length + ' points'; });
canvas.addEventListener('mouseup',    () => { drawing = false; });
canvas.addEventListener('mouseleave', () => { drawing = false; });
canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; points.push(getPos(e)); hint.classList.add('hidden'); }, { passive: false });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!drawing) return; points.push(getPos(e)); redraw(); ptCount.textContent = points.length + ' points'; }, { passive: false });
canvas.addEventListener('touchend',   () => { drawing = false; });

function clearCanvas() {
  points = [];
  const w = canvas.width  / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  hint.classList.remove('hidden');
  ptCount.textContent = '0 points';
  resetStats();
}

new ResizeObserver(resizeCanvas).observe(canvas);
resizeCanvas();

/* ── Model selector (synced with sidebar select) ── */
function selectedModel() {
  const sel = document.getElementById('modelSelect');
  return sel ? sel.value : 'rf';
}

/* ═══════════════════════════════════════════════
   Analyse stroke — calls Flask /api/analyse/draw
═══════════════════════════════════════════════ */
async function analyseStroke() {
  if (points.length < 20) { flashBtn('Draw more first!'); return; }

  const btn = document.getElementById('analyseBtn');
  setButtonLoading(btn, true);

  try {
    const res = await fetch('/api/analyse/draw', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ points, model: selectedModel() }),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      flashBtn(data.error || 'Error — try again');
      return;
    }

    setStats(data);

  } catch (err) {
    flashBtn('Server error');
    console.error('[NeuroGraph]', err);
  } finally {
    setButtonLoading(btn, false);
  }
}

/* ── Render results into stat cards ── */
function setStats(data) {
  const pdScore  = data.pd_prob;
  const tremorHz = data.tremor_peak_hz_vx != null ? data.tremor_peak_hz_vx.toFixed(1) : '—';
  const jerkVal  = data.jerk_mean != null
    ? (Math.abs(data.jerk_mean) > 1 ? data.jerk_mean.toFixed(2) : data.jerk_mean.toFixed(4))
    : '—';

  /* PD likelihood */
  document.getElementById('statPD').innerHTML =
    pdScore + '<span class="stat-unit">%</span>';
  document.getElementById('pdBar').style.width = Math.min(pdScore, 100) + '%';
  document.getElementById('pdBar').style.background = pdScore >= 50 ? '#ef4444' : '#22c55e';
  const isPD = data.is_pd;
  document.getElementById('statPDBadge').innerHTML =
    `<span class="stat-badge ${isPD ? 'pd' : 'ctrl'}">${isPD ? '⚠ Elevated' : '✓ Normal range'}</span>`;

  /* Tremor frequency */
  if (tremorHz !== '—') {
    document.getElementById('statTremor').innerHTML =
      tremorHz + '<span class="stat-unit">Hz</span>';
    const tPct   = Math.min(parseFloat(tremorHz) / 12 * 100, 100);
    document.getElementById('tremorBar').style.width = tPct + '%';
    const inBand = parseFloat(tremorHz) >= 3 && parseFloat(tremorHz) <= 12;
    document.getElementById('statTremorBadge').innerHTML =
      `<span class="stat-badge ${inBand ? 'pd' : 'neut'}">${inBand ? 'In PD band (3–12 Hz)' : 'Outside PD band'}</span>`;
  }

  /* Jerk mean */
  document.getElementById('statJerk').textContent = jerkVal;
}

function resetStats() {
  ['statPD', 'statTremor', 'statJerk'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
  ['statPDBadge', 'statTremorBadge'].forEach(id => {
    document.getElementById(id).innerHTML = '';
  });
  document.getElementById('pdBar').style.width     = '0%';
  document.getElementById('tremorBar').style.width = '0%';
}

function flashBtn(msg) {
  const b    = document.getElementById('analyseBtn');
  const orig = b.innerHTML;
  b.textContent = msg;
  setTimeout(() => { b.innerHTML = orig; }, 2000);
}

function setButtonLoading(btn, loading) {
  if (loading) {
    btn.dataset.origHtml = btn.innerHTML;
    btn.innerHTML = '<span class="btn-spinner"></span> Analysing…';
    btn.disabled  = true;
  } else {
    btn.innerHTML = btn.dataset.origHtml ||
      '<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg> Analyse Stroke';
    btn.disabled  = false;
  }
}

/* ═══════════════════════════════════════════════════════════
   File upload — calls Flask /api/analyse/upload
═══════════════════════════════════════════════════════════ */
let uploadedFile = null;

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  uploadedFile = file;

  document.getElementById('uploadFileName').textContent = file.name;

  /* Count rows client-side for immediate display */
  const reader = new FileReader();
  reader.onload = ev => {
    const lines = ev.target.result.trim().split('\n').filter(l => l.trim());
    document.getElementById('uploadRows').textContent = lines.length.toLocaleString();
    document.getElementById('uploadPD').textContent   = '—';
    document.getElementById('uploadPDBadge').innerHTML = '';
  };
  reader.readAsText(file);

  /* Flash upload zone */
  const zone = document.getElementById('uploadZone');
  zone.style.borderColor = 'var(--accent)';
  setTimeout(() => zone.style.borderColor = '', 1200);
}

async function analyseUpload() {
  if (!uploadedFile) { alert('Upload a file first.'); return; }

  const btn = document.querySelector('#tab-upload .btn-primary');
  if (btn) { btn.textContent = 'Analysing…'; btn.disabled = true; }

  try {
    const formData = new FormData();
    formData.append('file',    uploadedFile);
    formData.append('model',   selectedModel());
    formData.append('test_id', '-1');

    const res  = await fetch('/api/analyse/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      alert(data.error || 'Analysis failed.');
      return;
    }

    /* Display results */
    document.getElementById('uploadPD').innerHTML =
      data.pd_prob + '<span class="stat-unit">%</span>';
    const isPD = data.is_pd;
    document.getElementById('uploadPDBadge').innerHTML =
      `<span class="stat-badge ${isPD ? 'pd' : 'ctrl'}">${isPD ? '⚠ Elevated' : '✓ Normal range'}</span>`;

  } catch (err) {
    alert('Server error — is the Flask server running?');
    console.error('[NeuroGraph]', err);
  } finally {
    if (btn) { btn.textContent = '▶ Analyse File'; btn.disabled = false; }
  }
}

/* ═══════════════════════════════════════════════════════════
   Dataset Explorer — fetch from /api/dataset
═══════════════════════════════════════════════════════════ */
async function loadDataset() {
  const tbody = document.getElementById('dataTableBody');
  try {
    const res  = await fetch('/api/dataset');
    const data = await res.json();

    if (data.error) {
      tbody.innerHTML = `<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:20px">${data.error}</td></tr>`;
      return;
    }

    /* Update stat tiles if present */
    const statsMap = {
      'explorerTotal':    data.stats?.total,
      'explorerPD':       data.stats?.pd,
      'explorerControl':  data.stats?.control,
      'explorerFeatures': data.stats?.features,
    };
    for (const [id, val] of Object.entries(statsMap)) {
      const el = document.getElementById(id);
      if (el && val != null) el.textContent = val;
    }

    /* Populate table */
    tbody.innerHTML = '';
    data.rows.forEach(row => {
      const isPD = row.label === 1;
      tbody.innerHTML += `<tr>
        <td style="font-weight:500">${row.subject_id ?? '—'}</td>
        <td><span class="badge ${isPD ? 'badge-pd' : 'badge-ctrl'}">${isPD ? "Parkinson's" : 'Control'}</span></td>
        <td>${row.jerk_mean != null ? row.jerk_mean.toFixed(3) : '—'}</td>
        <td>${row.tremor_ratio_combined != null ? row.tremor_ratio_combined.toFixed(5) : '—'}</td>
        <td>${row.directional_entropy != null ? row.directional_entropy.toFixed(3) : '—'}</td>
        <td>${row.pen_lift_ratio != null ? row.pen_lift_ratio.toFixed(2) : '—'}</td>
        <td>${row.pressure_speed_corr != null ? row.pressure_speed_corr.toFixed(2) : '—'}</td>
      </tr>`;
    });

  } catch {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:20px">
      Could not load dataset — is the Flask server running?
    </td></tr>`;
  }
}

/* Load dataset when the Explorer tab is opened */
document.getElementById('nav-explorer').addEventListener('click', () => {
  loadDataset();
});

/* Also load immediately if user lands on explorer tab directly */
if (document.getElementById('tab-explorer')?.classList.contains('active')) {
  loadDataset();
}
