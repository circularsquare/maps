// T-1474: how long after the player's click a sound starts. Real mouse events in headless Chrome; the
// page records the click event's own time (event.timeStamp, the browser's) and the performance.now() of
// each source.start() call, so the delay is click to start() with the main thread's queueing included.
// Scenarios: track placed (second click of a segment), station placed (click on that track), construct
// (the track inspector's button), train length (the line inspector's stepper; no sound before T-1474).
//   node trainworld/checks/t1474-latency.mjs [url] [label]     (from the maps root; a running dev server)
// Writes ../artifacts/trainworld/t1474-sounds/latency-<label>.json; prints one line per scenario.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { GL, NOISE, REAL_SAVE, acrossLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
const LABEL = process.argv[3] ?? 'run';
const REAL = process.env.REAL === '1';
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t1474-sounds/', import.meta.url));
const { ctx, close } = await launchChrome({ prefix: 'tw-t1474-', size: [1400, 900], args: [...GL, '--autoplay-policy=no-user-gesture-required'] });
const errors = [];
const result = { label: LABEL, base: BASE, scenarios: {} };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => {
    const L = (window.__lat = { clicks: [], downs: [], starts: [], ctx: null });
    addEventListener('pointerdown', (e) => L.downs.push(e.timeStamp), true);
    addEventListener('click', (e) => L.clicks.push(e.timeStamp), true);
    // AudioBufferSourceNode has its own start(); an oscillator inherits AudioScheduledSourceNode's
    for (const [C, kind] of [[AudioBufferSourceNode, 'buf'], [OscillatorNode, 'osc']]) {
      const target = Object.prototype.hasOwnProperty.call(C.prototype, 'start') ? C.prototype : AudioScheduledSourceNode.prototype;
      const o = target.start;
      target.start = function (when, ...rest) {
        L.starts.push({ t: performance.now(), kind, when: when ?? 0, now: this.context.currentTime, state: this.context.state, base: this.context.baseLatency, out: this.context.outputLatency });
        return o.call(this, when, ...rest);
      };
    }
  });
  await page.goto(`${BASE}?debug=1&new=1${REAL ? '&demandWorkers=2' : '&paused=1'}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  if (REAL) {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 100, undefined, { timeout: 600000, polling: 250 });
    await demandFinal(page, { timeout: 600000, polling: 250 }).catch(() => console.log('no demand final'));
    // the arrival sounds of running trains would count as sounds
    await page.evaluate(() => window.tw.clock.setRunning(false));
  }
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.mouse.click(600, 400);
  await page.waitForTimeout(4500);

  const reset = () => page.evaluate(() => { const L = window.__lat; L.clicks = []; L.downs = []; L.starts = []; });
  /** the delay from the last click to the first start() after it, ms; null when nothing started */
  const read = () => page.evaluate(async () => {
    const wm = await import('/src/game/world.ts');
    const L = window.__lat, c = L.clicks.at(-1), d = L.downs.at(-1), s = L.starts.find((x) => x.t >= c);
    const w = window.tw.world.value;
    return { n: L.starts.length, fromClick: s ? +(s.t - c).toFixed(1) : null, fromDown: s ? +(s.t - d).toFixed(1) : null, when: s?.when, state: s?.state, base: s?.base, out: s?.out,
      edges: w.edges.length, stations: w.save.stations.length, notice: wm.notice.value?.text ?? null };
  });
  const add = (name, r) => { (result.scenarios[name] ??= []).push(r); };
  const jump = (lng, lat, zoom) => page.evaluate(([a, b, z]) => { window.tw.map.jumpTo({ center: [a, b], zoom: z }); }, [lng, lat, zoom]);
  const screen = (lng, lat) => page.evaluate(([a, b]) => { const p = window.tw.map.project([a, b]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + p.x, r.top + p.y]; }, [lng, lat]);
  const setTool = (t) => page.evaluate(async (t) => { const ui = await import('/src/game/ui.ts'); ui.keepPlacing.value = true; ui.tool.value = t; }, t);
  const clickAt = async ([x, y]) => {
    await page.mouse.move(x, y, { steps: 6 });
    await page.waitForTimeout(450);
    await reset();
    await page.mouse.down(); await page.mouse.up();
    await page.waitForTimeout(1100);
    return read();
  };

  // track: a zigzag of 7 clicks 1 km apart; the first only starts the run
  const LAT = 40.56, LNG = -74.46, STEP = 0.0115;
  await jump(LNG + 3 * STEP, LAT, 14);
  await page.waitForTimeout(1500);
  await setTool('track');
  const pts = [];
  for (let k = 0; k < 7; k++) pts.push([LNG + k * STEP, LAT + (k % 2 ? 0.004 : -0.004)]);
  for (let k = 0; k < pts.length; k++) {
    const r = await clickAt(await screen(...pts[k]));
    if (k > 0) add('track', r);
  }
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);

  // station: the middle of each of the 6 pieces
  await setTool('station');
  for (let k = 0; k < 6; k++) {
    const a = pts[k], b = pts[k + 1];
    add('station', await clickAt(await screen((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)));
  }
  await setTool('select');

  // construct: each of three pieces, from the track inspector's button
  for (let k = 0; k < 3; k++) {
    const edge = await page.evaluate(async (k) => {
      const ui = await import('/src/game/ui.ts');
      const e = window.tw.world.value.edges.filter((e) => !e.built)[0];
      if (!e) return null;
      ui.selection.value = { kind: 'track', edge: e.id };
      return e.id;
    }, k);
    if (edge === null) break;
    await page.waitForTimeout(500);
    const box = await page.locator('button:has-text("Construct for")').first().boundingBox();
    if (!box) break;
    add('construct', await clickAt([box.x + box.width / 2, box.y + box.height / 2]));
  }

  // train length: a built line, its stepper
  await jump(-74.40, 40.64, 14);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; });
  const line = await acrossLine(page, [-74.40, 40.64], 2);
  await page.waitForTimeout(1500);
  if (line !== null) {
    await page.evaluate(async (n) => { const ui = await import('/src/game/ui.ts'); ui.selection.value = { kind: 'line', line: String(n) }; }, line);
    await page.waitForTimeout(800);
    for (let k = 0; k < 4; k++) {
      const label = k % 2 ? 'Less train length' : 'More train length';
      const b = await page.locator(`button[aria-label="${label}"]`).first().boundingBox();
      if (!b) break;
      add('cars', await clickAt([b.x + b.width / 2, b.y + b.height / 2]));
    }
  }

  const stats = (a) => {
    const v = a.map((r) => r.fromClick).filter((x) => x !== null).sort((x, y) => x - y);
    return v.length ? { n: v.length, of: a.length, min: v[0], median: v[Math.floor(v.length / 2)], max: v.at(-1), all: v } : { n: 0, of: a.length };
  };
  result.summary = Object.fromEntries(Object.entries(result.scenarios).map(([k, a]) => [k, stats(a)]));
  result.errors = errors.slice(0, 5);
  for (const [k, s] of Object.entries(result.summary)) console.log(`${k.padEnd(10)} ${s.n}/${s.of} sounded  click to start(): min ${s.min} median ${s.median} max ${s.max} ms`);
  console.log('first record', JSON.stringify(Object.values(result.scenarios)[0]?.[0]));
  console.log(errors.length ? `console errors: ${errors.slice(0, 3).join(' | ')}` : 'no console errors');
  await mkdir(OUT, { recursive: true });
  await writeFile(`${OUT}latency-${LABEL}.json`, JSON.stringify(result, null, 1));
} finally {
  await close();
}
