namespace PRM.Viz;

/// <summary>
/// Generates the self-contained HTML page that connects to the WebSocket server
/// and renders the PRM ball simulation with animated Canvas graphics.
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
  <label>Speed <input id="speedSlider" type="range" min="1" max="60" value="20"/></label>
  <div id="rowCounter">Row 0 / 0</div>
  <label><input id="chkNails" type="checkbox" checked/> Show nails</label>
  <label><input id="chkTrails" type="checkbox" checked/> Show trails</label>
  <div id="legend"></div>
</div>

<script>
"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
let cfg     = null;   // { totalRows, wideningRows, entryWidth, maxWidth, vocab[] }
let frames  = [];     // Array of row frames: { row, balls[], nails[] }
let curRow  = 0;      // Frame index currently displayed
let playing = false;
let lastTs  = 0;
let autoPlayPending = false;  // auto-starts animation on first frame

const ballColors = ['#ff6b6b','#4ecdc4','#f7c948','#bb8fce','#ff9a5c','#74b9ff','#a29bfe'];

// Map tokenId → color (assigned on first sight)
const tokenColorMap = {};
function tokenColor(id) {
  if (id < 0) return '#ffffff44';
  if (!tokenColorMap[id]) {
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
      tokenColorMap; // clear
      Object.keys(tokenColorMap).forEach(k => delete tokenColorMap[k]);
      buildLegend([]);
      break;

    case 'clear':
      frames = [];
      curRow = 0;
      playing = false;
      autoPlayPending = true;
      document.getElementById('btnPlay').textContent = '▶ Play';
      document.getElementById('btnPlay').classList.remove('active');
      updateRowCounter();
      document.getElementById('inputLabel').textContent = msg.tokens.join(' · ');
      document.getElementById('predLabel').textContent  = '—';
      document.getElementById('predLabel').className    = '';
      msg.tokens.forEach((t, i) => {
        if (cfg) {
          const vocab = cfg.vocab.find(v => v.Text.trim() === t.trim());
          if (vocab) tokenColor(vocab.Id);
        }
      });
      buildLegend(msg.tokens);
      draw();
      break;

    case 'frame':
      frames.push(msg);
      if (autoPlayPending) {
        autoPlayPending = false;
        curRow  = 0;
        playing = true;
        lastTs  = 0;
        document.getElementById('btnPlay').textContent = '⏸ Pause';
        document.getElementById('btnPlay').classList.add('active');
        requestAnimationFrame(animLoop);
      }
      if (!playing) { curRow = frames.length - 1; draw(); }
      updateRowCounter();
      break;

    case 'result':
      const pred = document.getElementById('predLabel');
      pred.textContent = `→ ${msg.predicted}${msg.target ? ' (target: ' + msg.target + ')' : ''}`;
      pred.className   = msg.correct ? 'correct' : 'wrong';
      // Jump to final frame
      if (frames.length > 0) { curRow = frames.length - 1; draw(); }
      break;
  }
}

// ─── Legend ───────────────────────────────────────────────────────────────────
function buildLegend(tokens) {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  tokens.forEach(t => {
    if (!cfg) return;
    const vocab = cfg.vocab.find(v => v.Text.trim() === t.trim());
    const id    = vocab ? vocab.Id : -1;
    const col   = tokenColor(id);
    el.innerHTML += `<div class="leg-item"><div class="leg-dot" style="background:${col}"></div>${t}</div>`;
  });
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────
function gridWidth(row) {
  if (!cfg) return 1;
  const { entryWidth, maxWidth, wideningRows, totalRows } = cfg;
  const nRows = totalRows - wideningRows;
  const cr = (maxWidth - entryWidth) / Math.max(nRows, 1);
  return row <= wideningRows
    ? entryWidth + row * (maxWidth - entryWidth) / Math.max(wideningRows, 1)
    : maxWidth - (row - wideningRows) * cr;
}

function toScreen(gridX, row) {
  const mx = 50, my = 20;
  const W  = canvas.width  - mx * 2;
  const H  = canvas.height - my * 2 - 40;  // 40px for output slot labels
  return {
    x: mx + (gridX / cfg.maxWidth) * W,
    y: my + (row   / cfg.totalRows) * H
  };
}

// ─── Drawing ──────────────────────────────────────────────────────────────────
function draw() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!cfg) { ctx.fillStyle='#334'; ctx.font='16px Segoe UI'; ctx.fillText('Waiting for config…', 40, 40); return; }

  drawDiamondBackground();
  if (document.getElementById('chkNails').checked)  drawNails();
  if (document.getElementById('chkTrails').checked) drawTrails();
  drawCurrentBalls();
  drawOutputSlots();
  drawRowIndicator();
}

function drawDiamondBackground() {
  // Left border path
  ctx.beginPath();
  for (let r = 0; r <= cfg.totalRows; r++) {
    const gw = gridWidth(r);
    const left = (cfg.maxWidth - gw) / 2;
    const p = toScreen(left, r);
    r === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
  }
  for (let r = cfg.totalRows; r >= 0; r--) {
    const gw = gridWidth(r);
    const right = (cfg.maxWidth + gw) / 2;
    const p = toScreen(right, r);
    ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0,   'rgba(70,80,200,0.12)');
  grad.addColorStop(0.5, 'rgba(90,100,230,0.18)');
  grad.addColorStop(1,   'rgba(50,60,160,0.12)');
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = '#2a3080';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Thinking / summarising divider
  const divY = toScreen(0, cfg.wideningRows).y;
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(100,120,255,0.25)';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(W, divY); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(80,100,200,0.5)';
  ctx.font = '10px Segoe UI';
  ctx.fillText('THINKING ↕', 4, divY - 6);
  ctx.fillText('SUMMARISING ↕', 4, divY + 12);
}

function drawNails() {
  if (!frames.length || !cfg) return;
  const curFrame = frames[Math.min(curRow, frames.length - 1)];
  const curR = curFrame ? curFrame.row : -1;

  // Background grid: all rows drawn faintly so the full diamond grid is visible
  frames.forEach(f => {
    if (!f.nails || f.row === curR) return;
    f.nails.forEach(nail => {
      const p  = toScreen(nail.x, f.row);
      if (p.x < 2 || p.x > canvas.width - 2) return;
      const nr = Math.max(2, Math.min(8, (nail.r ?? 0.5) * 10));
      ctx.beginPath();
      ctx.arc(p.x, p.y, nr, 0, Math.PI * 2);
      ctx.fillStyle   = 'rgba(140,155,220,0.13)';
      ctx.strokeStyle = 'rgba(100,120,200,0.20)';
      ctx.lineWidth   = 0.5;
      ctx.fill(); ctx.stroke();
    });
  });

  // Current row: bright nails with size, resistance, and deflection arrows
  if (!curFrame || !curFrame.nails) return;
  curFrame.nails.forEach(nail => {
    const p  = toScreen(nail.x, curR);
    if (p.x < 2 || p.x > canvas.width - 2) return;

    // Nail size: Radius 0–1 → 3–14px
    const nr = Math.max(3, Math.min(14, (nail.r ?? 0.5) * 14));
    // Resistance → brightness/opacity (stiff nails look more solid)
    const rs   = nail.rs ?? 0.5;
    const fill = `rgba(200,215,255,${(0.35 + rs * 0.55).toFixed(2)})`;
    const strk = `rgba(160,185,255,${(0.40 + rs * 0.40).toFixed(2)})`;

    ctx.beginPath();
    ctx.arc(p.x, p.y, nr, 0, Math.PI * 2);
    ctx.fillStyle   = fill;
    ctx.strokeStyle = strk;
    ctx.lineWidth   = 1 + rs;
    ctx.fill(); ctx.stroke();

    // Deflection arrow
    if (Math.abs(nail.ox) > 0.01) {
      const arrowLen = Math.min(20, 14 * Math.abs(nail.ox));
      const dir      = Math.sign(nail.ox);
      const col      = dir > 0 ? '#ffdd44' : '#44ccff';
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + arrowLen * dir, p.y);
      ctx.lineTo(p.x + (arrowLen - 5) * dir, p.y - 3);
      ctx.moveTo(p.x + arrowLen * dir, p.y);
      ctx.lineTo(p.x + (arrowLen - 5) * dir, p.y + 3);
      ctx.strokeStyle = col;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }
  });
}

function drawTrails() {
  if (frames.length < 2) return;
  const end = Math.min(curRow, frames.length - 1);

  // Collect paths per tokenId
  const paths = {};
  for (let ri = 0; ri <= end; ri++) {
    const f = frames[ri];
    if (!f) continue;
    f.balls.forEach(b => {
      if (b.TokenId < 0) return;
      if (!paths[b.TokenId]) paths[b.TokenId] = [];
      paths[b.TokenId].push({ x: b.Position, row: f.row });
    });
  }

  Object.entries(paths).forEach(([id, pts]) => {
    if (pts.length < 2) return;
    const col = tokenColor(parseInt(id));
    ctx.beginPath();
    pts.forEach((pt, i) => {
      const s = toScreen(pt.x, pt.row);
      i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
    });
    ctx.strokeStyle = col + '55';  // semi-transparent
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  });
}

function drawCurrentBalls() {
  if (!frames.length) return;
  const frame = frames[Math.min(curRow, frames.length - 1)];
  if (!frame) return;

  frame.balls.forEach(b => {
    if (b.TokenId < 0) return;  // skip probe ball
    const p   = toScreen(b.Position, frame.row);
    const r   = Math.max(8, Math.min(46, 8 + b.Mass * 38));  // 8–46px: clearly different sizes
    const col = tokenColor(b.TokenId);

    // Outer glow (3 layers for strong visibility)
    [r * 3.5, r * 2.2, r * 1.4].forEach((gr, i) => {
      const alpha = ['18', '33', '66'][i];
      const g2 = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr);
      g2.addColorStop(0,   col + alpha);
      g2.addColorStop(1,   col + '00');
      ctx.beginPath(); ctx.arc(p.x, p.y, gr, 0, Math.PI * 2);
      ctx.fillStyle = g2; ctx.fill();
    });

    // Ball fill — fully opaque solid center
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    const ballGrad = ctx.createRadialGradient(p.x - r*0.3, p.y - r*0.3, r*0.1, p.x, p.y, r);
    ballGrad.addColorStop(0, '#ffffff');
    ballGrad.addColorStop(0.25, col);
    ballGrad.addColorStop(1, col + 'bb');
    ctx.fillStyle   = ballGrad;
    ctx.strokeStyle = '#ffffff88';
    ctx.lineWidth   = 2;
    ctx.fill(); ctx.stroke();

    // Token label — bold, black outline for contrast
    const label = cfg.vocab[b.TokenId]?.Text?.trim() ?? `#${b.TokenId}`;
    const fs = Math.max(10, Math.min(r - 1, 16));
    ctx.font      = `bold ${fs}px Segoe UI`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#00000099';
    ctx.lineWidth   = 3;
    ctx.strokeText(label.length > 8 ? label.slice(0, 7) + '…' : label, p.x, p.y);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label.length > 8 ? label.slice(0, 7) + '…' : label, p.x, p.y);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  });
}

function drawOutputSlots() {
  if (!cfg) return;
  const bottomY   = canvas.height - 48;
  const labelY    = canvas.height - 28;
  const slotH     = 10;
  const mx        = 50;
  const W         = canvas.width - mx * 2;

  // Slots background strip
  ctx.fillStyle = 'rgba(20,20,60,0.7)';
  ctx.fillRect(mx, bottomY - 2, W, slotH + 4);

  // Identify which slot(s) the final balls land in
  const predictedId = getPredictedTokenId();

  cfg.vocab.forEach(v => {
    const sx = mx + (v.SlotLeft  / cfg.vocab.at(-1).SlotRight) * W;
    const sw = Math.max(1, (v.SlotWidth / cfg.vocab.at(-1).SlotRight) * W);

    const isWinner = v.Id === predictedId;
    ctx.fillStyle = isWinner ? '#00ff8888' : 'rgba(60,60,120,0.4)';
    ctx.fillRect(sx, bottomY, sw, slotH);

    // Border
    ctx.strokeStyle = isWinner ? '#00ff88' : '#2a2a6a';
    ctx.lineWidth   = isWinner ? 1.5 : 0.5;
    ctx.strokeRect(sx, bottomY, sw, slotH);

    // Label for winner + wide slots or every Nth token
    if (isWinner || sw > 20) {
      ctx.fillStyle = isWinner ? '#00ff88' : '#8888aa';
      ctx.font      = isWinner ? 'bold 11px Segoe UI' : '9px Segoe UI';
      ctx.textAlign = 'center';
      ctx.fillText(v.Text.trim(), sx + sw / 2, labelY);
      ctx.textAlign = 'left';
    }
  });

  ctx.fillStyle = '#556';
  ctx.font = '9px Segoe UI';
  ctx.fillText('output tokens →', 4, labelY);
}

function getPredictedTokenId() {
  if (!frames.length || !cfg) return -1;
  const last = frames[frames.length - 1];
  if (!last?.balls.length) return -1;
  // Count votes by slot (flat)
  const scores = new Array(cfg.vocab.length).fill(0);
  const totalSlotSpan = cfg.vocab.at(-1).SlotRight;
  const lastRow   = cfg.totalRows - 1;
  const gw        = gridWidth(lastRow);
  const gridLeft  = (cfg.maxWidth - gw) / 2;
  const gridRight = gridLeft + gw;
  const gridSpan  = gridRight - gridLeft;

  last.balls.forEach(b => {
    if (b.TokenId < 0) return;
    const norm = (b.Position - gridLeft) / gridSpan * totalSlotSpan;
    for (let t = 0; t < cfg.vocab.length; t++) {
      if (norm >= cfg.vocab[t].SlotLeft && norm < cfg.vocab[t].SlotRight) {
        scores[t]++;
        break;
      }
    }
  });
  return scores.indexOf(Math.max(...scores));
}

function drawRowIndicator() {
  if (!frames.length) return;
  const f = frames[Math.min(curRow, frames.length - 1)];
  if (!f) return;
  const y = toScreen(0, f.row).y;
  ctx.save();
  ctx.strokeStyle = 'rgba(200,200,255,0.2)';
  ctx.lineWidth   = 2;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(200,200,255,0.5)';
  ctx.font = '10px Segoe UI';
  ctx.fillText(`row ${f.row}`, 4, y - 3);
}

function updateRowCounter() {
  document.getElementById('rowCounter').textContent =
    `Row ${frames.length ? frames[Math.min(curRow, frames.length-1)].row : 0} / ${cfg ? cfg.totalRows : 0}`;
}

// ─── Animation loop ───────────────────────────────────────────────────────────
let animStepsPerRow = 1;
let animSubStep     = 0;

function animLoop(ts) {
  if (!playing) return;
  const speed = parseInt(document.getElementById('speedSlider').value);
  const delay = 1000 / speed;
  if (ts - lastTs >= delay) {
    lastTs = ts;
    if (curRow < frames.length - 1) {
      curRow++;
      updateRowCounter();
      draw();
    } else {
      playing = false;
      document.getElementById('btnPlay').textContent = '▶ Play';
      document.getElementById('btnPlay').classList.remove('active');
    }
  }
  requestAnimationFrame(animLoop);
}

// ─── Controls ─────────────────────────────────────────────────────────────────
document.getElementById('btnPlay').addEventListener('click', function() {
  playing = !playing;
  this.textContent = playing ? '⏸ Pause' : '▶ Play';
  this.classList.toggle('active', playing);
  if (playing) requestAnimationFrame(animLoop);
});
document.getElementById('btnStep').addEventListener('click', () => {
  if (curRow < frames.length - 1) { curRow++; updateRowCounter(); draw(); }
});
document.getElementById('btnReset').addEventListener('click', () => {
  curRow = 0; playing = false;
  document.getElementById('btnPlay').textContent = '▶ Play';
  document.getElementById('btnPlay').classList.remove('active');
  updateRowCounter(); draw();
});
document.getElementById('chkNails').addEventListener('change',  draw);
document.getElementById('chkTrails').addEventListener('change', draw);

// ─── Start ────────────────────────────────────────────────────────────────────
connect();
</script>
</body>
</html>
""";
}
