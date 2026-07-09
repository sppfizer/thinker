/**
 * viz-test.mjs — Automated verification of PRM.Viz.
 * Uses vendor puppeteer-core + system Chrome (same pattern as line-info-agent connectors).
 * Starts PRM.Viz internally with --no-browser, then connects Puppeteer.
 *
 * Usage:  node viz-test.mjs
 */

import { spawn }               from 'child_process';
import { fileURLToPath }       from 'url';
import { setTimeout as sleep } from 'timers/promises';
import path                    from 'path';

const __dir = path.dirname(fileURLToPath(import.meta.url));

const PUPPETEER_URL = new URL(
  '../../Copilot/Agents/line-info-agent/skills/_vendor/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js',
  import.meta.url
).href;
const TEST_PORT = 5060 + Math.floor(Math.random() * 40);
const CHROME  = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const VIZ_DIR = path.join(__dir, 'src', 'PRM', 'PRM.Viz');

console.log('PRM Visualizer — automated test');
console.log('================================');

// ── 0. Start PRM.Viz with --no-browser ───────────────────────────────────────
console.log('\n[0] Starting PRM.Viz --no-browser...');
const vizProc = spawn('dotnet', ['run', '-c', 'Debug', '--', '--port', String(TEST_PORT), '--no-browser', 'the', 'cat', 'sat'], {
  cwd: VIZ_DIR, stdio: ['ignore', 'pipe', 'pipe']
});
let vizLog = '';
vizProc.stdout.on('data', d => { vizLog += d; process.stdout.write('  viz> ' + d); });
vizProc.stderr.on('data', d => { vizLog += d; });

let vizPort = TEST_PORT;
const portDeadline = Date.now() + 45_000;
while (Date.now() < portDeadline) {
  await sleep(500);
  const m = vizLog.match(/localhost:(\d+)/);
  if (m) { vizPort = +m[1]; break; }
}
console.log(`\n  PRM.Viz on port ${vizPort}`);
await sleep(500);

const VIZ_URL = `http://localhost:${vizPort}/`;
const puppeteer = (await import(PUPPETEER_URL)).default;
const browser = await puppeteer.launch({
  headless: false, executablePath: CHROME,
  defaultViewport: { width: 1280, height: 800 },
  args: ['--no-first-run', '--no-default-browser-check', '--window-size=1280,800'],
});

let passed = 0, failed = 0;
const ok   = msg => { console.log(`    ok  ${msg}`); passed++; };
const fail = msg => { console.error(`    FAIL ${msg}`); failed++; };

try {
  const page = await browser.newPage();

  // Intercept WS messages before page loads
  await page.evaluateOnNewDocument(() => {
    const _WS = window.WebSocket;
    window.__vizMessages = [];
    window.WebSocket = function(...a) {
      const ws = new _WS(...a);
      ws.addEventListener('message', e => {
        try { window.__vizMessages.push(JSON.parse(e.data)); } catch {}
      });
      return ws;
    };
    Object.assign(window.WebSocket, { CONNECTING:0, OPEN:1, CLOSING:2, CLOSED:3, prototype:_WS.prototype });
  });

  // 1. Load page
  console.log('\n[1] Loading page...');
  await page.goto(VIZ_URL, { waitUntil: 'domcontentloaded', timeout: 20000 });
  ok('HTTP 200, page loaded');

  // 2. Wait for WS messages
  console.log('\n[2] Waiting for WebSocket messages (up to 15s)...');
  const dl = Date.now() + 15000;
  while (Date.now() < dl) {
    await sleep(500);
    const t = await page.evaluate(() => (window.__vizMessages||[]).map(m=>m.type));
    if (t.includes('config') && t.includes('frame') && t.includes('result')) break;
  }
  const types = await page.evaluate(() => (window.__vizMessages||[]).map(m=>m.type));
  console.log(`    Messages: [${[...new Set(types)].join(', ')}] (${types.length} total)`);
  types.includes('config') ? ok('config received')   : fail('no config — WS not connected');
  types.includes('frame')  ? ok('frame(s) received') : fail('no frames — simulation not streaming');
  types.includes('result') ? ok('result received')   : fail('no result after simulation');

  // 3. Canvas content
  console.log('\n[3] Checking canvas content...');
  await sleep(1500);
  const bright = await page.evaluate(() => {
    const c = document.getElementById('c'); if (!c) return -1;
    const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
    let n = 0; for (let i=0; i<d.length; i+=4) if (d[i]>80||d[i+1]>80||d[i+2]>80) n++;
    return n;
  });
  bright > 500 ? ok(`Canvas has ${bright} bright pixels`) : fail(`Canvas only ${bright} bright pixels`);

  // 4. Screenshot
  await page.screenshot({ path: path.join(__dir, 'viz-screenshot.png') });
  ok('viz-screenshot.png saved');

} finally {
  await browser.close();
  vizProc.kill();
}

console.log('\n================================');
if (failed === 0) { console.log(`All ${passed} checks passed!`); }
else { console.log(`${passed} passed, ${failed} failed`); process.exit(1); }

