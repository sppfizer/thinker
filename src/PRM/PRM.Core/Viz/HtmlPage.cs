namespace PRM.Core.Viz;

/// <summary>
/// Generates the self-contained HTML/JS visualizer page.
/// Animation is driven by a client-side setInterval clock — completely
/// decoupled from WebSocket delivery speed.
/// All frames are buffered first; the clock starts once the 'result'
/// message arrives (full trace received).
/// </summary>
internal static class HtmlPage
{
    public static string Build(int port) => $$"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PRM — Ball Visualizer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body   { background: #0f1526; color: #c8d4ff; font-family: 'Segoe UI', sans-serif;
           overflow: hidden; height: 100vh; display: flex; flex-direction: column; }
  canvas { flex: 1; display: block; }

  #topbar { display: flex; align-items: center; gap: 16px; padding: 6px 14px;
            background: rgba(15,20,45,0.95); border-bottom: 1px solid #2a2a6a; flex-shrink: 0; }
  #topbar h1 { font-size: 14px; font-weight: 600; color: #99aaff; white-space: nowrap; }
  #inputLabel { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: 2px; }
  #predLabel  { font-size: 14px; padding: 3px 10px; border-radius: 20px;
                background: rgba(80,80,200,0.35); border: 1px solid #4a4ab0; }
  #predLabel.correct { background: rgba(0,180,100,0.35); border-color: #00cc66; color: #00ff88; }
  #predLabel.wrong   { background: rgba(200,50,50,0.35);  border-color: #cc3333; color: #ff6666; }
  #status { margin-left: auto; font-size: 11px; color: #6677aa; }

  #controls { display: flex; align-items: center; gap: 10px; padding: 6px 14px;
              background: rgba(15,20,45,0.95); border-top: 1px solid #2a2a6a; flex-shrink: 0; }
  button { background: #1e2466; color: #b8ccff; border: 1px solid #4040a0; padding: 4px 12px;
           border-radius: 4px; cursor: pointer; font-size: 12px; transition: background .15s; }
  button:hover { background: #2a3280; }
  button.active { background: #3838c0; border-color: #7070ee; color: #fff; }
  label  { font-size: 11px; color: #8899bb; display: flex; align-items: center; gap: 4px; }
  input[type=range] { width: 110px; accent-color: #7070ee; }
  #rowCounter { font-size: 12px; color: #7090bb; min-width: 80px; }
  #legend { display: flex; gap: 10px; margin-left: auto; }
  .leg-item { font-size: 11px; display: flex; align-items: center; gap: 4px; }
  .leg-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
</head>
<body>

<div id="topbar">
  <h1>PRM Ball Visualizer</h1>
  <div id="inputLabel">—</div>
  <div id="predLabel">—</div>
  <div id="status">Connecting…</div>
</div>

<canvas id="c"></canvas>

<div id="controls">
  <button id="btnPlay">▶ Play</button>
  <button id="btnStep">▶| Step</button>
  <button id="btnReset">↩ Reset</button>
  <label>Speed <input id="speedSlider" type="range" min="1" max="30" value="8"/></label>
  <div id="rowCounter">Row — / —</div>
  <label><input id="chkNails" type="checkbox" checked/> Show nails</label>
  <label><input id="chkTrails" type="checkbox" checked/> Trails</label>
  <div id="legend"></div>
</div>

<script>
"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
let cfg    = null;    // config from server
let frames = [];      // all buffered row-frames
let curRow = 0;       // current display index into frames[]
let ticker = null;    // setInterval handle — THE animation clock

const ballColors = ['#ff6b6b','#4ecdc4','#f7c948','#bb8fce','#ff9a5c','#74b9ff','#a29bfe'];
const tokenColorMap = {};
function tokenColor(id) {
  if (id < 0) return '#ffffff44';
  if (tokenColorMap[id] === undefined) {
    const idx = Object.keys(tokenColorMap).length % ballColors.length;
    tokenColorMap[id] = ballColors[idx];
  }
  return tokenColorMap[id];
}

// ─── Canvas ───────────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
function resize() { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; draw(); }
window.addEventListener('resize', resize);
setTimeout(resize, 50);

// ─── Animation clock ──────────────────────────────────────────────────────────
function getInterval() { return 1000 / Math.max(1, +document.getElementById('speedSlider').value); }

function stopClock() {
  if (ticker !== null) { clearInterval(ticker); ticker = null; }
  document.getElementById('btnPlay').textContent = '▶ Play';
  document.getElementById('btnPlay').classList.remove('active');
}

function startClock() {
  stopClock();
  if (frames.length === 0) return;
  ticker = setInterval(() => {
    if (curRow < frames.length - 1) {
      curRow++;
      draw();
      updateRowCounter();
    } else {
      stopClock();
    }
  }, getInterval());
  document.getElementById('btnPlay').textContent = '⏸ Pause';
  document.getElementById('btnPlay').classList.add('active');
}

document.getElementById('speedSlider').addEventListener('input', () => {
  if (ticker !== null) startClock(); // restart with new speed
});

// ─── Controls ─────────────────────────────────────────────────────────────────
document.getElementById('btnPlay').addEventListener('click', function() {
  if (ticker !== null) { stopClock(); }
  else { if (curRow >= frames.length - 1) curRow = 0; startClock(); }
});
document.getElementById('btnStep').addEventListener('click', () => {
  stopClock();
  if (curRow < frames.length - 1) { curRow++; draw(); updateRowCounter(); }
});
document.getElementById('btnReset').addEventListener('click', () => {
  stopClock(); curRow = 0; draw(); updateRowCounter();
});
document.getElementById('chkNails').addEventListener('change',  draw);
document.getElementById('chkTrails').addEventListener('change', draw);

// ─── WebSocket ────────────────────────────────────────────────────────────────
let ws;
function connect() {
  ws = new WebSocket('ws://localhost:{{port}}/ws');
  ws.onopen    = () => { document.getElementById('status').textContent = 'Connected ✓'; };
  ws.onclose   = () => { document.getElementById('status').textContent = 'Reconnecting…'; setTimeout(connect, 1500); };
  ws.onerror   = () => ws.close();
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}

function handle(msg) {
  switch (msg.type) {
    case 'config':
      cfg = msg;
      Object.keys(tokenColorMap).forEach(k => delete tokenColorMap[k]);
      break;

    case 'clear':
      stopClock();
      frames = [];
      curRow = 0;
      document.getElementById('inputLabel').textContent = msg.tokens.join(' · ');
      document.getElementById('predLabel').textContent  = '—';
      document.getElementById('predLabel').className    = '';
      msg.tokens.forEach(t => {
        if (cfg) { const v = cfg.vocab.find(v => v.Text.trim() === t.trim()); if (v) tokenColor(v.Id); }
      });
      buildLegend(msg.tokens);
      updateRowCounter();
      draw();
      break;

    case 'frame':
      frames.push(msg);
      break;

    case 'result':
      // Show result label — do NOT touch curRow; the clock plays from row 0
      const pred = document.getElementById('predLabel');
      pred.textContent = '→ ' + msg.predicted + (msg.target ? ' (target: ' + msg.target + ')' : '');
      pred.className   = msg.correct ? 'correct' : 'wrong';
      // Start clock now that all frames are buffered
      curRow = 0;
      draw();
      updateRowCounter();
      startClock();
      break;
  }
}

// ─── Legend ───────────────────────────────────────────────────────────────────
function buildLegend(tokens) {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  tokens.forEach(t => {
    const v = cfg && cfg.vocab.find(v => v.Text.trim() === t.trim());
    const col = tokenColor(v ? v.Id : -1);
    el.innerHTML += `<div class="leg-item"><div class="leg-dot" style="background:${col}"></div>${t}</div>`;
  });
}

// ─── Row counter ──────────────────────────────────────────────────────────────
function updateRowCounter() {
  const f = frames[curRow];
  document.getElementById('rowCounter').textContent =
    f ? `Row ${f.row} / ${cfg ? cfg.totalRows : '?'}` : `Row — / —`;
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────
const MX = 60, MY = 24;
function screenW() { return canvas.width  - MX * 2; }
function screenH() { return canvas.height - MY * 2 - 50; }

function gridWidth(row) {
  if (!cfg) return 1;
  const { entryWidth, maxWidth, wideningRows, totalRows } = cfg;
  const nRows = totalRows - wideningRows;
  const cr    = (maxWidth - entryWidth) / Math.max(nRows, 1);
  return row <= wideningRows
    ? entryWidth + row * (maxWidth - entryWidth) / Math.max(wideningRows, 1)
    : maxWidth - (row - wideningRows) * cr;
}

function toScreen(gridX, row) {
  return {
    x: MX + (gridX / cfg.maxWidth) * screenW(),
    y: MY + (row   / cfg.totalRows) * screenH()
  };
}

// ─── Drawing ──────────────────────────────────────────────────────────────────
function draw() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!cfg) {
    ctx.fillStyle = '#445'; ctx.font = '16px Segoe UI';
    ctx.fillText('Waiting for connection…', 40, 40); return;
  }
  drawDiamond();
  if (document.getElementById('chkNails').checked)  drawNails();
  if (document.getElementById('chkTrails').checked) drawTrails();
  drawBalls();
  drawSlots();
  drawRowLine();
}

// ── Diamond background ────────────────────────────────────────────────────────
function drawDiamond() {
  ctx.beginPath();
  for (let r = 0; r <= cfg.totalRows; r++) {
    const gw = gridWidth(r), left = (cfg.maxWidth - gw) / 2;
    const p  = toScreen(left, r);
    r === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
  }
  for (let r = cfg.totalRows; r >= 0; r--) {
    const gw = gridWidth(r), right = (cfg.maxWidth + gw) / 2;
    ctx.lineTo(toScreen(right, r).x, toScreen(right, r).y);
  }
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0,   'rgba(60,75,190,0.18)');
  grad.addColorStop(0.5, 'rgba(80,100,220,0.22)');
  grad.addColorStop(1,   'rgba(45,60,160,0.15)');
  ctx.fillStyle = grad; ctx.fill();
  ctx.strokeStyle = '#3a40a0'; ctx.lineWidth = 2; ctx.stroke();

  // Divider
  const dy = toScreen(0, cfg.wideningRows).y;
  ctx.save(); ctx.setLineDash([5,4]);
  ctx.strokeStyle = 'rgba(100,130,255,0.30)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, dy); ctx.lineTo(canvas.width, dy); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(100,130,255,0.55)'; ctx.font = '10px Segoe UI';
  ctx.fillText('▼ THINKING', 4, dy - 5);
  ctx.fillText('▼ SUMMARISING', 4, dy + 12);
}

// ── Nail grid ─────────────────────────────────────────────────────────────────
function drawNails() {
  if (!frames.length) return;
  const curFrame = frames[curRow] || null;
  const activeRow = curFrame ? curFrame.row : -1;

  // All rows faintly — the full nail grid
  frames.forEach(f => {
    if (!f.nails) return;
    const isCur = f.row === activeRow;
    f.nails.forEach(nail => {
      const p  = toScreen(nail.x, f.row);
      if (p.x < 4 || p.x > canvas.width - 4) return;
      const nr = isCur ? Math.max(4, Math.min(14, (nail.r ?? 0.5) * 14))
                       : Math.max(2, Math.min(7,  (nail.r ?? 0.5) * 8));
      if (isCur) {
        // Bright current-row nail
        const rs   = nail.rs ?? 0.5;
        ctx.beginPath(); ctx.arc(p.x, p.y, nr, 0, Math.PI*2);
        ctx.fillStyle   = `rgba(210,225,255,${(0.45 + rs*0.45).toFixed(2)})`;
        ctx.strokeStyle = `rgba(170,195,255,${(0.50 + rs*0.35).toFixed(2)})`;
        ctx.lineWidth   = 1.5 + rs; ctx.fill(); ctx.stroke();
        // Deflection arrow
        if (Math.abs(nail.ox) > 0.01) {
          const al  = Math.min(22, 15 * Math.abs(nail.ox));
          const dir = Math.sign(nail.ox);
          const col = dir > 0 ? '#ffdd44' : '#44ccff';
          ctx.beginPath();
          ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + al*dir, p.y);
          ctx.lineTo(p.x + (al-5)*dir, p.y - 3);
          ctx.moveTo(p.x + al*dir, p.y);
          ctx.lineTo(p.x + (al-5)*dir, p.y + 3);
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
        }
      } else {
        // Faint background nail
        ctx.beginPath(); ctx.arc(p.x, p.y, nr, 0, Math.PI*2);
        ctx.fillStyle   = 'rgba(150,165,230,0.40)';
        ctx.strokeStyle = 'rgba(120,140,220,0.45)';
        ctx.lineWidth   = 0.8; ctx.fill(); ctx.stroke();
      }
    });
  });
}

// ── Trails ────────────────────────────────────────────────────────────────────
function drawTrails() {
  if (frames.length < 2 || curRow < 1) return;
  const paths = {};
  for (let ri = 0; ri <= curRow && ri < frames.length; ri++) {
    const f = frames[ri]; if (!f) continue;
    f.balls.forEach(b => {
      if (b.TokenId < 0) return;
      if (!paths[b.TokenId]) paths[b.TokenId] = [];
      paths[b.TokenId].push({ x: b.Position, row: f.row });
    });
  }
  Object.entries(paths).forEach(([id, pts]) => {
    if (pts.length < 2) return;
    ctx.beginPath();
    pts.forEach((pt, i) => {
      const s = toScreen(pt.x, pt.row);
      i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
    });
    ctx.strokeStyle = tokenColor(+id) + '66';
    ctx.lineWidth = 1.5; ctx.stroke();
  });
}

// ── Balls ─────────────────────────────────────────────────────────────────────
function drawBalls() {
  if (!frames.length) return;
  const frame = frames[curRow]; if (!frame) return;
  frame.balls.forEach(b => {
    if (b.TokenId < 0) return;
    const p   = toScreen(b.Position, frame.row);
    const r   = Math.max(10, Math.min(44, 10 + b.Mass * 34));
    const col = tokenColor(b.TokenId);

    // Glow
    [r*3.2, r*2.0, r*1.3].forEach((gr, i) => {
      const a  = ['15','30','60'][i];
      const g2 = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr);
      g2.addColorStop(0, col+a); g2.addColorStop(1, col+'00');
      ctx.beginPath(); ctx.arc(p.x, p.y, gr, 0, Math.PI*2);
      ctx.fillStyle = g2; ctx.fill();
    });

    // Solid ball
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI*2);
    const bg = ctx.createRadialGradient(p.x - r*.3, p.y - r*.3, r*.08, p.x, p.y, r);
    bg.addColorStop(0, '#ffffff'); bg.addColorStop(0.25, col); bg.addColorStop(1, col+'cc');
    ctx.fillStyle = bg; ctx.strokeStyle = '#ffffff99'; ctx.lineWidth = 2;
    ctx.fill(); ctx.stroke();

    // Label
    const label = (cfg.vocab[b.TokenId]?.Text ?? `#${b.TokenId}`).trim();
    const fs = Math.max(9, Math.min(r - 2, 15));
    ctx.font = `bold ${fs}px Segoe UI`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#000000bb'; ctx.lineWidth = 3;
    ctx.strokeText(label.length > 8 ? label.slice(0,7)+'…' : label, p.x, p.y);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label.length > 8 ? label.slice(0,7)+'…' : label, p.x, p.y);
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  });
}

// ── Output slots ──────────────────────────────────────────────────────────────
function drawSlots() {
  if (!cfg || !cfg.vocab || cfg.vocab.length === 0) return;
  const totalSpan = cfg.vocab[cfg.vocab.length - 1].SlotRight;
  if (!totalSpan || totalSpan <= 0) return;
  const botY  = canvas.height - 50;
  const labY  = canvas.height - 30;
  const slotH = 12;
  const W     = screenW();

  ctx.fillStyle = 'rgba(15,18,50,0.80)';
  ctx.fillRect(MX, botY - 2, W, slotH + 4);

  const predId = getPredId();
  cfg.vocab.forEach(v => {
    if (v.SlotRight === undefined || v.SlotLeft === undefined) return;
    const sx = MX + (v.SlotLeft  / totalSpan) * W;
    const sw = Math.max(2, (v.SlotWidth / totalSpan) * W);
    const win = v.Id === predId;
    ctx.fillStyle   = win ? '#00ff8899' : 'rgba(55,60,130,0.45)';
    ctx.strokeStyle = win ? '#00ff88'   : '#333388';
    ctx.lineWidth   = win ? 1.5 : 0.5;
    ctx.fillRect(sx, botY, sw, slotH);
    ctx.strokeRect(sx, botY, sw, slotH);
    if (win || sw > 18) {
      ctx.fillStyle = win ? '#00ff88' : '#7788aa';
      ctx.font      = win ? 'bold 11px Segoe UI' : '9px Segoe UI';
      ctx.textAlign = 'center';
      ctx.fillText((v.Text||'').trim(), sx + sw/2, labY);
      ctx.textAlign = 'left';
    }
  });
  ctx.fillStyle = '#5566aa'; ctx.font = '9px Segoe UI';
  ctx.fillText('output slots →', 4, labY);
}

function getPredId() {
  if (!frames.length || !cfg) return -1;
  const last = frames[frames.length - 1];
  if (!last?.balls?.length) return -1;
  const scores = new Array(cfg.vocab.length).fill(0);
  const ts     = cfg.vocab[cfg.vocab.length - 1].SlotRight;
  const lastRow  = cfg.totalRows - 1;
  const gw   = gridWidth(lastRow);
  const gl   = (cfg.maxWidth - gw) / 2;
  const span = gw;
  last.balls.forEach(b => {
    if (b.TokenId < 0) return;
    const norm = (b.Position - gl) / span * ts;
    for (let t = 0; t < cfg.vocab.length; t++) {
      if (norm >= cfg.vocab[t].SlotLeft && norm < cfg.vocab[t].SlotRight) { scores[t]++; break; }
    }
  });
  return scores.indexOf(Math.max(...scores));
}

// ── Active-row indicator ──────────────────────────────────────────────────────
function drawRowLine() {
  if (!frames.length) return;
  const f = frames[curRow]; if (!f) return;
  const y = toScreen(0, f.row).y;
  ctx.save(); ctx.strokeStyle = 'rgba(200,210,255,0.22)';
  ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(200,210,255,0.55)'; ctx.font = '10px Segoe UI';
  ctx.fillText(`row ${f.row}`, 4, y - 3);
}

connect();
</script>
</body>
</html>
""";
}
