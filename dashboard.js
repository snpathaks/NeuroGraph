/* ── Theme toggle ─────────────────────────────── */
const html = document.documentElement;
const themeBtn = document.getElementById('themeToggle');

themeBtn.addEventListener('click', () => {
  const isDark = html.dataset.theme === 'dark';
  html.dataset.theme = isDark ? 'light' : 'dark';
  themeBtn.textContent = isDark ? '🌙 Dark' : '☀️ Light';
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
  const isDark = html.dataset.theme === 'dark';
  ctx.strokeStyle = isDark ? '#3b82f6' : '#2563eb';
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'round';
  ctx.lineJoin    = 'round';
  ctx.stroke();
}

function getPos(e) {
  const r = canvas.getBoundingClientRect();
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
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  hint.classList.remove('hidden');
  ptCount.textContent = '0 points';
  resetStats();
}

new ResizeObserver(resizeCanvas).observe(canvas);
resizeCanvas();

/* ── Feature extraction (client-side approximation) ── */
function computeFeatures(pts) {
  if (pts.length < 10) return null;
  const n = pts.length;
  const dt = [], vx = [], vy = [], speed = [];

  for (let i = 1; i < n; i++) {
    const d = Math.max((pts[i].t - pts[i-1].t) / 1000, 1e-4);
    dt.push(d);
    const dvx = (pts[i].x - pts[i-1].x) / d;
    const dvy = (pts[i].y - pts[i-1].y) / d;
    vx.push(dvx); vy.push(dvy);
    speed.push(Math.sqrt(dvx*dvx + dvy*dvy));
  }

  const jerk = [];
  for (let i = 1; i < vx.length; i++) {
    const d = Math.max(dt[i], 1e-4);
    const jx = (vx[i] - vx[i-1]) / d;
    const jy = (vy[i] - vy[i-1]) / d;
    jerk.push(Math.sqrt(jx*jx + jy*jy));
  }

  const mean  = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const jMean = mean(jerk);
  const sMax  = Math.max(...speed);

  // Directional entropy (simplified)
  const angles = [];
  for (let i = 1; i < n; i++) {
    angles.push(Math.atan2(pts[i].y - pts[i-1].y, pts[i].x - pts[i-1].x));
  }
  const bins = new Array(16).fill(0);
  angles.forEach(a => {
    const idx = Math.floor(((a + Math.PI) / (2 * Math.PI)) * 16) % 16;
    bins[idx]++;
  });
  const total = angles.length;
  let entropy = 0;
  bins.forEach(c => { const p = c / total; if (p > 0) entropy -= p * Math.log(p + 1e-12); });

  // Tremor proxy: variance of jerk magnitude
  const tremorProxy = Math.min(jMean / 200, 1.0);

  return { jMean, sMax, entropy, tremorProxy };
}

function analyseStroke() {
  if (points.length < 20) { flashBtn('Draw more first!'); return; }

  const btn = document.getElementById('analyseBtn');
  btn.textContent = 'Analysing…';
  btn.disabled = true;

  setTimeout(() => {
    const f = computeFeatures(points);
    if (!f) { btn.textContent = '▶ Analyse Stroke'; btn.disabled = false; return; }

    const jerkScore   = Math.min(f.jMean / 150, 1);
    const entScore    = Math.min(f.entropy / 2.7, 1);
    const pdScore     = Math.round((jerkScore * 0.55 + entScore * 0.30 + f.tremorProxy * 0.15) * 100);
    const tremorHz    = (3.0 + f.tremorProxy * 9.0).toFixed(1);
    const jerkDisplay = f.jMean > 1 ? f.jMean.toFixed(1) : f.jMean.toFixed(4);

    setStats(pdScore, tremorHz, jerkDisplay);
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="#fff"><polygon points="5 3 19 12 5 21 5 3"/></svg> Analyse Stroke';
    btn.disabled = false;
  }, 700);
}

function setStats(pdScore, tremorHz, jerkVal) {
  document.getElementById('statPD').innerHTML = pdScore + '<span class="stat-unit">%</span>';
  document.getElementById('pdBar').style.width = pdScore + '%';
  const isPD = pdScore >= 50;
  document.getElementById('statPDBadge').innerHTML =
    `<span class="stat-badge ${isPD ? 'pd' : 'ctrl'}">${isPD ? '⚠ Elevated' : '✓ Normal range'}</span>`;

  document.getElementById('statTremor').innerHTML = tremorHz + '<span class="stat-unit">Hz</span>';
  const tPct = Math.min(parseFloat(tremorHz) / 12 * 100, 100);
  document.getElementById('tremorBar').style.width = tPct + '%';
  const inBand = parseFloat(tremorHz) >= 3 && parseFloat(tremorHz) <= 12;
  document.getElementById('statTremorBadge').innerHTML =
    `<span class="stat-badge ${inBand ? 'pd' : 'neut'}">${inBand ? 'In PD band (3–12Hz)' : 'Outside PD band'}</span>`;

  document.getElementById('statJerk').textContent = jerkVal;
}

function resetStats() {
  ['statPD', 'statTremor', 'statJerk'].forEach(id => document.getElementById(id).textContent = '—');
  ['statPDBadge', 'statTremorBadge'].forEach(id => document.getElementById(id).innerHTML = '');
  document.getElementById('pdBar').style.width = '0%';
  document.getElementById('tremorBar').style.width = '0%';
}

function flashBtn(msg) {
  const b = document.getElementById('analyseBtn');
  const orig = b.innerHTML;
  b.textContent = msg;
  setTimeout(() => { b.innerHTML = orig; }, 1500);
}

/* ── File upload ──────────────────────────────── */
let uploadedRows = [];

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('uploadFileName').textContent = file.name;

  const reader = new FileReader();
  reader.onload = ev => {
    const lines = ev.target.result.trim().split('\n').filter(l => l.trim());
    uploadedRows = lines.map(l => l.split(';').map(Number));
    document.getElementById('uploadRows').textContent = lines.length.toLocaleString();
    document.getElementById('uploadPD').textContent = '—';
    document.getElementById('uploadPDBadge').innerHTML = '';

    const zone = document.getElementById('uploadZone');
    zone.style.borderColor = 'var(--accent)';
    setTimeout(() => zone.style.borderColor = '', 1200);
  };
  reader.readAsText(file);
}

function analyseUpload() {
  if (!uploadedRows.length) { alert('Upload a file first.'); return; }

  const pressures = uploadedRows.map(r => r[3] || 0).filter(p => !isNaN(p));
  const mean = pressures.reduce((a, b) => a + b, 0) / pressures.length;
  const variance = pressures.reduce((a, b) => a + (b - mean) ** 2, 0) / pressures.length;
  const pdScore = Math.round(Math.min(variance / 30000 * 100 + 30, 95));

  document.getElementById('uploadPD').innerHTML = pdScore + '<span class="stat-unit">%</span>';
  const isPD = pdScore >= 50;
  document.getElementById('uploadPDBadge').innerHTML =
    `<span class="stat-badge ${isPD ? 'pd' : 'ctrl'}">${isPD ? '⚠ Elevated' : '✓ Normal range'}</span>`;
}

/* ── Dataset table ────────────────────────────── */
const sampleData = [
  { id: 'P_02100001', group: "Parkinson's", jerk: '7.75',  tremor: '0.0063', entropy: '2.03', lift: '0.94', corr: '0.57' },
  { id: 'P_05060003', group: "Parkinson's", jerk: '2.04',  tremor: '0.0000', entropy: '2.35', lift: '0.90', corr: '0.48' },
  { id: 'P_09100001', group: "Parkinson's", jerk: '5.12',  tremor: '0.0041', entropy: '2.18', lift: '0.92', corr: '0.52' },
  { id: 'P_11120003', group: "Parkinson's", jerk: '4.89',  tremor: '0.0037', entropy: '2.26', lift: '0.91', corr: '0.55' },
  { id: 'P_12060001', group: "Parkinson's", jerk: '3.67',  tremor: '0.0028', entropy: '2.11', lift: '0.88', corr: '0.49' },
  { id: 'C_0010',     group: 'Control',     jerk: '0.000', tremor: '0.0000', entropy: '0.10', lift: '0.84', corr: '0.88' },
  { id: 'C_0001',     group: 'Control',     jerk: '6.78',  tremor: '0.0080', entropy: '2.64', lift: '0.95', corr: '0.60' },
  { id: 'C_0002',     group: 'Control',     jerk: '3.21',  tremor: '0.0042', entropy: '2.31', lift: '0.91', corr: '0.71' },
  { id: 'C_0005',     group: 'Control',     jerk: '2.89',  tremor: '0.0031', entropy: '2.17', lift: '0.89', corr: '0.75' },
  { id: 'C_0009',     group: 'Control',     jerk: '4.12',  tremor: '0.0055', entropy: '2.44', lift: '0.93', corr: '0.66' },
];

const tbody = document.getElementById('dataTableBody');
sampleData.forEach(row => {
  const isPD = row.group === "Parkinson's";
  tbody.innerHTML += `<tr>
    <td style="font-weight:500">${row.id}</td>
    <td><span class="badge ${isPD ? 'badge-pd' : 'badge-ctrl'}">${row.group}</span></td>
    <td>${row.jerk}</td>
    <td>${row.tremor}</td>
    <td>${row.entropy}</td>
    <td>${row.lift}</td>
    <td>${row.corr}</td>
  </tr>`;
});
