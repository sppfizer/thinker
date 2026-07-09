namespace PRM.Core.Viz;

/// <summary>
/// PRM Ball Visualizer — Galton-board style.
///
/// Physical animation per row:
///   Phase 0 (frac 0.00 → 0.45): ball falls STRAIGHT DOWN, x constant
///   Phase 1 (frac 0.45 → 0.55): ball AT nail row — nearest nail pulses/glows
///   Phase 2 (frac 0.55 → 1.00): ball deflects LEFT or RIGHT, falls to next row
///
/// This mirrors the actual model: nails deflect balls; the deflection direction
/// (sign of ox) is learned during training.
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
  *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
  body   { background:#0d1120; color:#c8d4ff; font-family:'Segoe UI',sans-serif;
           overflow:hidden; height:100vh; display:flex; flex-direction:column; }
  canvas { flex:1; display:block; }

  #topbar { display:flex; align-items:center; gap:14px; padding:5px 14px;
            background:rgba(12,16,40,0.96); border-bottom:1px solid #252570; flex-shrink:0; }
  #topbar h1 { font-size:13px; font-weight:600; color:#8899ee; white-space:nowrap; }
  #inputLabel { font-size:17px; font-weight:700; color:#fff; letter-spacing:2px; }
  #predLabel  { font-size:13px; padding:3px 10px; border-radius:18px;
                background:rgba(70,70,190,0.35); border:1px solid #4040aa; }
  #predLabel.correct { background:rgba(0,170,90,0.38); border-color:#00bb60; color:#00ff88; }
  #predLabel.wrong   { background:rgba(190,45,45,0.38); border-color:#bb2222; color:#ff5555; }
  #status { margin-left:auto; font-size:11px; color:#556688; }

  #controls { display:flex; align-items:center; gap:10px; padding:5px 14px;
              background:rgba(12,16,40,0.96); border-top:1px solid #252570; flex-shrink:0; }
  button { background:#182060; color:#aabbff; border:1px solid #3535aa; padding:4px 12px;
           border-radius:4px; cursor:pointer; font-size:12px; transition:background .15s; }
  button:hover { background:#253080; }
  button.active { background:#3030bb; border-color:#6060ee; color:#fff; }
  label  { font-size:11px; color:#7788aa; display:flex; align-items:center; gap:4px; }
  input[type=range] { width:100px; accent-color:#6060ee; }
  #rowCounter { font-size:12px; color:#6080aa; min-width:85px; }
  #legend { display:flex; gap:8px; margin-left:auto; }
  .leg-item { font-size:11px; display:flex; align-items:center; gap:4px; }
  .leg-dot  { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
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
  <button id="btnStep">⏭ Step</button>
  <button id="btnReset">↩ Reset</button>
  <label>Speed <input id="spd" type="range" min="1" max="20" value="5"/></label>
  <div id="rowCounter">Row — / —</div>
  <label><input id="chkNails"  type="checkbox" checked/> Nails</label>
  <label><input id="chkTrails" type="checkbox" checked/> Trails</label>
  <div id="legend"></div>
</div>

<script>
"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
// Sub-steps per row. Each tick advances 1/SUBSTEPS of a row.
// At speed=5 → 200ms/tick → 6 ticks/row = 1.2s per row → full 16 rows ≈ 20s
const SUBSTEPS = 6;

// ── State ─────────────────────────────────────────────────────────────────────
let cfg    = null;
let frames = [];       // row-frames from server
let trails = {};       // tokenId → [{x,y}] for trail drawing
let curStep = 0.0;     // float: floor=row-index, frac=phase within that row
let ticker  = null;

// ── Colors ────────────────────────────────────────────────────────────────────
const BALL_PALETTE = ['#ff5f5f','#3ecfc4','#f5c542','#bb7ee0','#ff955a','#62aaff','#9a8afe'];
const colorMap = {};
function ballColor(id) {
  if (id < 0) return '#ffffff33';
  if (!colorMap[id]) colorMap[id] = BALL_PALETTE[Object.keys(colorMap).length % BALL_PALETTE.length];
  return colorMap[id];
}

// ── Canvas ────────────────────────────────────────────────────────────────────
const C = document.getElementById('c');
const X = C.getContext('2d');
function resize() { C.width = C.offsetWidth; C.height = C.offsetHeight; draw(); }
window.addEventListener('resize', resize);
setTimeout(resize, 60);

// ── Layout constants ──────────────────────────────────────────────────────────
const MX = 55, MY = 20;
function SW() { return C.width  - MX*2; }
function SH() { return C.height - MY*2 - 50; }

function gridW(rowF) {
  if (!cfg) return 1;
  const { entryWidth:ew, maxWidth:mw, wideningRows:wr, totalRows:tr } = cfg;
  const cr = (mw - ew) / Math.max(tr - wr, 1);
  return rowF <= wr
    ? ew + rowF * (mw - ew) / Math.max(wr, 1)
    : mw - (rowF - wr) * cr;
}

function toScreen(gx, rowF) {
  return {
    x: MX + (gx / cfg.maxWidth) * SW(),
    y: MY + (rowF / cfg.totalRows) * SH()
  };
}

// ── Physical ball position — Galton-board path ─────────────────────────────
// Phase 0 (frac 0.00–0.40): ball falls STRAIGHT DOWN, x = x0
// Phase 1 (frac 0.40–0.58): ball swings to the deflection side of nearest nail
//   arc peak = nail.x + sign(ox) * halfSpacing  →  clearly visible lane choice
// Phase 2 (frac 0.58–1.00): ball settles at actual model position x1
function easeOut(t) { return 1 - (1-t)*(1-t); }

function ballPos(x0, r0, x1, nail, frac) {
  const halfSp  = (cfg?.nailSpacing ?? 2.0) / 2;
  const dx      = x1 - x0;
  // Deflection direction: use nail's learned offset (ox) when significant; fall back to actual dx
  const deflDir = (nail && Math.abs(nail.ox) > 0.015)
                    ? Math.sign(nail.ox)
                    : (dx >= 0 ? 1 : -1);
  // Arc peak: swing to the gap on the deflection side of the nearest nail
  const arcX    = nail
                    ? nail.x + deflDir * halfSp * 0.82
                    : x0 + deflDir * halfSp;

  let x, rowF;
  if (frac <= 0.40) {
    x    = x0;
    rowF = r0 + (frac / 0.40) * 0.44;
  } else if (frac <= 0.58) {
    const t = (frac - 0.40) / 0.18;
    x    = x0 + (arcX - x0) * easeOut(t);
    rowF = r0 + 0.44 + t * 0.08;
  } else {
    const t = easeOut((frac - 0.58) / 0.42);
    x    = arcX + (x1 - arcX) * t;
    rowF = r0 + 0.52 + ((frac - 0.58) / 0.42) * 0.48;
  }
  return { x, rowF };
}

// ── Get current interpolated balls for drawing ────────────────────────────
function getCurBalls() {
  if (!frames.length) return [];
  const ri    = Math.min(Math.floor(curStep), frames.length - 1);
  const frac  = curStep - Math.floor(curStep);
  const f0    = frames[ri];
  const f1    = frames[Math.min(ri+1, frames.length-1)];
  const nails = f0.nails || [];

  return (f0.balls || [])
    .filter(b => b.TokenId >= 0)
    .map(b => {
      const b1 = f1?.balls?.find(b2 => b2.TokenId === b.TokenId);
      // Find nearest nail to this ball's entry position
      let nearNail = null, nearDist = Infinity;
      nails.forEach(n => { const d = Math.abs(b.Position - n.x); if (d < nearDist) { nearDist = d; nearNail = n; } });
      const x1  = b1?.Position ?? b.Position;
      const pos = ballPos(b.Position, f0.row, x1, nearNail, frac);
      return { TokenId: b.TokenId, Mass: b.Mass, x: pos.x, rowF: pos.rowF };
    });
}

// ── Trails (historical path) — stored as grid coords to survive resize ───
function updateTrails() {
  getCurBalls().forEach(b => {
    if (!trails[b.TokenId]) trails[b.TokenId] = [];
    const arr  = trails[b.TokenId];
    const last = arr[arr.length - 1];
    // Store grid coords {gx, rowF} — converted to screen at draw time
    if (!last || Math.abs(b.x - last.gx) > 0.05 || Math.abs(b.rowF - last.rowF) > 0.05)
      arr.push({ gx: b.x, rowF: b.rowF });
  });
}

// ── Clock ────────────────────────────────────────────────────────────────────
function interval() { return 1000 / Math.max(1, +document.getElementById('spd').value); }

function stopClock() {
  if (ticker) { clearInterval(ticker); ticker = null; }
  document.getElementById('btnPlay').textContent = '▶ Play';
  document.getElementById('btnPlay').classList.remove('active');
}
function startClock() {
  stopClock();
  if (frames.length < 2) return;
  ticker = setInterval(() => {
    curStep += 1/SUBSTEPS;
    if (curStep >= frames.length - 1) { curStep = frames.length - 1; stopClock(); }
    updateTrails();
    draw();
    updateRowLbl();
  }, interval());
  document.getElementById('btnPlay').textContent = '⏸ Pause';
  document.getElementById('btnPlay').classList.add('active');
}

document.getElementById('spd').addEventListener('input', () => { if (ticker) startClock(); });
document.getElementById('btnPlay').addEventListener('click', () => {
  if (ticker) stopClock();
  else { if (curStep >= frames.length-1) { curStep=0; trails={}; } startClock(); }
});
document.getElementById('btnStep').addEventListener('click', () => {
  stopClock();
  curStep = Math.min(Math.ceil(curStep + 0.001), frames.length-1);
  updateTrails(); draw(); updateRowLbl();
});
document.getElementById('btnReset').addEventListener('click', () => {
  stopClock(); curStep=0; trails={}; draw(); updateRowLbl();
});
document.getElementById('chkNails').addEventListener('change',  draw);
document.getElementById('chkTrails').addEventListener('change', draw);

function updateRowLbl() {
  const f = frames[Math.min(Math.floor(curStep), frames.length-1)];
  document.getElementById('rowCounter').textContent =
    f ? `Row ${f.row} / ${cfg?.totalRows ?? '?'}` : 'Row — / —';
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws;
function connect() {
  ws = new WebSocket(`ws://localhost:{{port}}/ws`);
  ws.onopen    = () => document.getElementById('status').textContent = 'Connected ✓';
  ws.onclose   = () => { document.getElementById('status').textContent = 'Reconnecting…'; setTimeout(connect, 1500); };
  ws.onerror   = () => ws.close();
  ws.onmessage = e  => handle(JSON.parse(e.data));
}

function handle(msg) {
  switch (msg.type) {
    case 'config':
      cfg = msg;
      Object.keys(colorMap).forEach(k => delete colorMap[k]);
      break;

    case 'clear':
      stopClock(); frames=[]; trails={}; curStep=0;
      document.getElementById('inputLabel').textContent = msg.tokens.join(' · ');
      document.getElementById('predLabel').textContent  = '—';
      document.getElementById('predLabel').className    = '';
      msg.tokens.forEach(t => {
        if (cfg) { const v = cfg.vocab.find(v=>v.Text.trim()===t.trim()); if(v) ballColor(v.Id); }
      });
      buildLegend(msg.tokens); updateRowLbl(); draw();
      break;

    case 'frame':
      frames.push(msg);
      break;

    case 'result': {
      const el = document.getElementById('predLabel');
      el.textContent = '→ ' + msg.predicted + (msg.target ? ' (target: '+msg.target+')' : '');
      el.className   = msg.correct ? 'correct' : 'wrong';
      curStep=0; trails={}; draw(); updateRowLbl();
      startClock();
      break;
    }
  }
}

function buildLegend(tokens) {
  document.getElementById('legend').innerHTML = tokens.map(t => {
    const v = cfg?.vocab?.find(v=>v.Text.trim()===t.trim());
    return `<div class="leg-item"><div class="leg-dot" style="background:${ballColor(v?.Id??-1)}"></div>${t}</div>`;
  }).join('');
}

// ── DRAWING ───────────────────────────────────────────────────────────────────
function draw() {
  X.clearRect(0, 0, C.width, C.height);
  if (!cfg) { X.fillStyle='#334'; X.font='15px Segoe UI'; X.fillText('Waiting for data…',40,40); return; }
  drawDiamond();
  if (document.getElementById('chkNails').checked)  drawNails();
  if (document.getElementById('chkTrails').checked) drawTrailLines();
  drawBalls();
  drawSlots();
}

// ── Diamond background ────────────────────────────────────────────────────────
function drawDiamond() {
  X.beginPath();
  for (let r=0; r<=cfg.totalRows; r++) {
    const gw=gridW(r), lx=(cfg.maxWidth-gw)/2;
    const p=toScreen(lx,r); r===0 ? X.moveTo(p.x,p.y) : X.lineTo(p.x,p.y);
  }
  for (let r=cfg.totalRows; r>=0; r--) {
    const gw=gridW(r), rx=(cfg.maxWidth+gw)/2;
    X.lineTo(toScreen(rx,r).x, toScreen(rx,r).y);
  }
  X.closePath();
  const g=X.createLinearGradient(0,0,0,C.height);
  g.addColorStop(0,  'rgba(50,65,175,0.17)');
  g.addColorStop(0.5,'rgba(70,90,210,0.20)');
  g.addColorStop(1,  'rgba(38,52,148,0.14)');
  X.fillStyle=g; X.fill();
  X.strokeStyle='#353598'; X.lineWidth=2; X.stroke();

  // Phase divider
  const dy=toScreen(0,cfg.wideningRows).y;
  X.save(); X.setLineDash([5,4]);
  X.strokeStyle='rgba(90,120,255,0.25)'; X.lineWidth=1;
  X.beginPath(); X.moveTo(0,dy); X.lineTo(C.width,dy); X.stroke();
  X.restore();
  X.fillStyle='rgba(90,120,255,0.48)'; X.font='10px Segoe UI';
  X.fillText('▼ THINKING',4,dy-4); X.fillText('▼ SUMMARISING',4,dy+12);
}

// ── Nail grid ─────────────────────────────────────────────────────────────────
function drawNails() {
  if (!frames.length) return;
  const ri     = Math.min(Math.floor(curStep), frames.length-1);
  const frac   = curStep - Math.floor(curStep);
  const curRow = frames[ri]?.row ?? ri;
  const atNail = frac >= 0.35 && frac <= 0.75;

  // Determine hit nails for current step
  const hitSet = new Set();
  if (atNail) {
    const f   = frames[ri];
    const nls = f?.nails ?? [];
    (f?.balls ?? []).filter(b=>b.TokenId>=0).forEach(b => {
      let best=-1, bd=Infinity;
      nls.forEach((n,i)=>{ const d=Math.abs(b.Position-n.x); if(d<bd){bd=d;best=i;} });
      if (best>=0) hitSet.add(best);
    });
  }

  frames.forEach(f => {
    if (!f.nails) return;
    const isActive = f.row === curRow;
    // Diamond boundary for this row — clip out-of-bounds padding nails
    const gw = gridW(f.row);
    const gl = (cfg.maxWidth - gw) / 2;
    const gr = gl + gw;

    f.nails.forEach((nail, ni) => {
      // Skip nails outside the diamond boundary for this row
      if (nail.x < gl - 0.15 || nail.x > gr + 0.15) return;

      const p = toScreen(nail.x, f.row);

      if (isActive) {
        const isHit  = hitSet.has(ni);
        const rs     = nail.rs ?? 0.5;
        const baseR  = Math.max(5, Math.min(14, (nail.r ?? 0.5) * 14));
        const nr     = isHit ? baseR * 1.5 : baseR;

        if (isHit) {
          const hit = ctx_radial(p.x, p.y, nr, nr*4, 'rgba(255,215,55,0.55)', 'rgba(255,215,55,0)');
          X.beginPath(); X.arc(p.x, p.y, nr*4, 0, Math.PI*2); X.fillStyle=hit; X.fill();
          X.beginPath(); X.arc(p.x, p.y, nr,   0, Math.PI*2);
          X.fillStyle='rgba(255,225,80,0.92)'; X.strokeStyle='rgba(255,255,160,1)'; X.lineWidth=2; X.fill(); X.stroke();
        } else {
          X.beginPath(); X.arc(p.x, p.y, nr, 0, Math.PI*2);
          X.fillStyle  =`rgba(205,220,255,${(0.55+rs*0.40).toFixed(2)})`;
          X.strokeStyle=`rgba(165,190,255,${(0.60+rs*0.30).toFixed(2)})`;
          X.lineWidth=1.6+rs*0.5; X.fill(); X.stroke();
        }

        // Deflection arrow
        if (Math.abs(nail.ox) > 0.008) {
          const al  = Math.min(22, 14 * Math.abs(nail.ox));
          const dir = Math.sign(nail.ox);
          const col = isHit ? '#ffe844' : (dir>0 ? '#ffcc44' : '#44ddff');
          X.beginPath();
          X.moveTo(p.x, p.y); X.lineTo(p.x+al*dir, p.y);
          X.lineTo(p.x+(al-5)*dir, p.y-3);
          X.moveTo(p.x+al*dir, p.y); X.lineTo(p.x+(al-5)*dir, p.y+3);
          X.strokeStyle=col; X.lineWidth=isHit?2.5:1.8; X.stroke();
        }
      } else {
        // Background grid nails — clearly visible dots, sized by nail.r
        const nr = Math.max(3, Math.min(8, (nail.r ?? 0.5) * 9));
        X.beginPath(); X.arc(p.x, p.y, nr, 0, Math.PI*2);
        X.fillStyle  ='rgba(160,175,240,0.55)';
        X.strokeStyle='rgba(130,150,225,0.60)';
        X.lineWidth=1.0; X.fill(); X.stroke();
      }
    });
  });
}

function ctx_radial(cx,cy,r0,r1,c0,c1) {
  const g=X.createRadialGradient(cx,cy,r0,cx,cy,r1); g.addColorStop(0,c0); g.addColorStop(1,c1); return g;
}

// ── Trails ────────────────────────────────────────────────────────────────────
function drawTrailLines() {
  Object.entries(trails).forEach(([id, pts]) => {
    if (pts.length < 2) return;
    X.beginPath();
    pts.forEach((p, i) => {
      const s = toScreen(p.gx, p.rowF);   // convert grid→screen at draw time (resize-safe)
      i === 0 ? X.moveTo(s.x, s.y) : X.lineTo(s.x, s.y);
    });
    X.strokeStyle = ballColor(+id) + '70';
    X.lineWidth   = 1.8; X.stroke();
  });
}

// ── Balls ─────────────────────────────────────────────────────────────────────
function drawBalls() {
  // Max ball radius = 42% of the nail spacing in screen pixels so balls fit between nails
  const nailSpPx = cfg ? (cfg.nailSpacing ?? 2.0) / cfg.maxWidth * SW() : 40;
  const maxR     = nailSpPx * 0.42;

  getCurBalls().forEach(b => {
    const p   = toScreen(b.x, b.rowF);
    const r   = Math.max(6, Math.min(maxR, 6 + b.Mass * maxR * 1.8));
    const col = ballColor(b.TokenId);

    // Glow
    [[r*3.0,'18'],[r*1.9,'32'],[r*1.25,'62']].forEach(([gr,a]) => {
      const g=ctx_radial(p.x,p.y,0,gr,col+a,col+'00');
      X.beginPath(); X.arc(p.x,p.y,gr,0,Math.PI*2); X.fillStyle=g; X.fill();
    });

    // Ball body
    X.beginPath(); X.arc(p.x,p.y,r,0,Math.PI*2);
    const bg=X.createRadialGradient(p.x-r*.3,p.y-r*.3,r*.07,p.x,p.y,r);
    bg.addColorStop(0,'#fff'); bg.addColorStop(0.25,col); bg.addColorStop(1,col+'cc');
    X.fillStyle=bg; X.strokeStyle='#ffffff88'; X.lineWidth=2; X.fill(); X.stroke();

    // Label
    const lbl = ((cfg?.vocab?.[b.TokenId]?.Text) ?? `#${b.TokenId}`).trim().slice(0,8);
    const fs  = Math.max(9, Math.min(r-2, 15));
    X.font=`bold ${fs}px Segoe UI`; X.textAlign='center'; X.textBaseline='middle';
    X.strokeStyle='#000000aa'; X.lineWidth=3; X.strokeText(lbl,p.x,p.y);
    X.fillStyle='#fff'; X.fillText(lbl,p.x,p.y);
    X.textAlign='left'; X.textBaseline='alphabetic';
  });
}

// ── Output slots ──────────────────────────────────────────────────────────────
function drawSlots() {
  if (!cfg?.vocab?.length) return;
  const ts = cfg.vocab[cfg.vocab.length-1].SlotRight;
  if (!ts || ts<=0) return;
  const by=C.height-50, ly=C.height-30, sh=12, W=SW();
  X.fillStyle='rgba(10,13,40,0.88)'; X.fillRect(MX,by-2,W,sh+4);
  const pid = getPredId();
  cfg.vocab.forEach(v => {
    const sx=MX+(v.SlotLeft/ts)*W, sw=Math.max(2,(v.SlotWidth/ts)*W);
    const win=v.Id===pid;
    X.fillStyle  = win ? '#00ff8888' : 'rgba(48,52,120,0.45)';
    X.strokeStyle= win ? '#00ff88'   : '#1e1e60';
    X.lineWidth  = win ? 1.5 : 0.5;
    X.fillRect(sx,by,sw,sh); X.strokeRect(sx,by,sw,sh);
    if (win||sw>16) {
      X.fillStyle=win?'#00ff88':'#5566aa'; X.font=win?'bold 11px Segoe UI':'9px Segoe UI';
      X.textAlign='center'; X.fillText((v.Text||'').trim(),sx+sw/2,ly); X.textAlign='left';
    }
  });
  X.fillStyle='#3344aa'; X.font='9px Segoe UI'; X.fillText('output slots →',4,ly);
}

function getPredId() {
  if (!frames.length||!cfg) return -1;
  const last=frames[frames.length-1]; if (!last?.balls?.length) return -1;
  const sc=new Array(cfg.vocab.length).fill(0);
  const ts=cfg.vocab[cfg.vocab.length-1].SlotRight;
  const ri=cfg.totalRows-1; const gw=gridW(ri); const gl=(cfg.maxWidth-gw)/2;
  last.balls.forEach(b=>{
    if(b.TokenId<0)return;
    const n=(b.Position-gl)/gw*ts;
    for(let t=0;t<cfg.vocab.length;t++){ if(n>=cfg.vocab[t].SlotLeft&&n<cfg.vocab[t].SlotRight){sc[t]++;break;} }
  });
  return sc.indexOf(Math.max(...sc));
}

connect();
</script>
</body>
</html>
""";
}
