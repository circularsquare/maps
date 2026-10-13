// T-164 browser check: per-train loads and platform crowds on New York's real network (T-007 save)
// in the real app. Needs a Vite dev server (BASE, default http://localhost:6537; with file watching
// off, restart it after editing the app, it serves cached modules) and Playwright
// (PLAYWRIGHT_MODULE, a file URL to its index.mjs). PROFILE_DIR: where the throwaway Chrome profile
// goes. Run from anywhere:
//   node trainworld/notes/t164-browser.mjs
// Screenshots: artifacts/trainworld/T-164-*.png. Results: artifacts/trainworld/notes-outputs/t164-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6537';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't164-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 600000 });
  // The morning peak at 08:20 on the save's day, paused.
  const at = await page.evaluate(() => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    const t = Math.floor(tw.clock.now() / 86400) * 86400 + 8 * 3600 + 20 * 60;
    tw.clock.set(t);
    tw.overlay.invalidate();
    return t;
  });
  await page.waitForFunction(t => window.tw.renderer.board && window.tw.renderer.board.epoch === Math.floor(t / 3600) * 3600, at, { timeout: 60000 });
  await page.waitForTimeout(1000);
  // The account's cost per hour sent: the riders messages (every line worked out again) and hour
  // turns (none).
  out.log = await page.evaluate(() => window.tw.client.boardLog.map(e => ({ ms: +e.ms.toFixed(1), redone: e.redone, stepsMs: +e.prepMs.toFixed(1), longestStepMs: +e.prepMax.toFixed(1), kB: Math.round(e.bytes / 1024) })));
  out.hour = await page.evaluate(() => {
    const b = window.tw.renderer.board;
    return { trips: b.trips, loads: b.loads.length, platforms: b.queueCount, events: b.events.length / 2 };
  });
  // Where full trains leave people behind this hour: platforms by riders left behind.
  out.worst = await page.evaluate(() => {
    const tw = window.tw, b = tw.renderer.board, w = tw.world.value;
    const rows = [];
    let total = 0, arriving = 0;
    for (let q = 0; q < b.queueCount; q++) {
      const Q = b.queues.subarray(q * 10, q * 10 + 10);
      total += Q[5];
      arriving += Q[7];
      if (Q[5] < 1) continue;
      const line = w.save.lines.find(l => l.num === Q[1]);
      rows.push({ q, node: Q[0], station: w.nodes.get(Q[0])?.name, line: line?.name, run: Q[2], left: Math.round(Q[5]), waitMin: Math.round(Q[6] / 60), arriving: Math.round(Q[7]) });
    }
    rows.sort((a, b) => b.left - a.left);
    return { platforms: rows.length, leftTotal: Math.round(total), arrivingTotal: Math.round(arriving), top: rows.slice(0, 12) };
  });
  const shoot = async (name, fn, arg, wait = 1500) => {
    const v = await page.evaluate(fn, arg);
    await page.waitForTimeout(wait);
    await page.screenshot({ path: join(ART, `T-164-${name}.png`) });
    return v;
  };
  // The busiest subway platform leaving people behind, before and after its next train.
  // the busiest subway platform where full trains leave 100 or more behind this hour
  const sub = out.worst.top.filter(r => /^subway/.test(r.line) && r.arriving >= 1000).sort((a, b) => b.left - a.left)[0];
  out.showcase = sub;
  // its train this hour that leaves the most behind: just before it, and just after
  const worstTrain = sub && await page.evaluate(q => {
    const r = window.tw.renderer, b = r.board, Q = b.queues, E = b.events, first = Q[q * 10 + 8], count = Q[q * 10 + 9];
    let best = -1;
    for (let j = first; j < first + count; j++) if (E[j * 2] >= 0 && E[j * 2] < 3600 && (best < 0 || E[j * 2 + 1] > E[best * 2 + 1])) best = j;
    return { t: E[best * 2] + r.tripsEpoch, left: Math.round(E[best * 2 + 1]) };
  }, sub.q);
  if (worstTrain) await page.evaluate(t => { const tw = window.tw; tw.clock.set(t - 25); tw.overlay.invalidate(); }, worstTrain.t);
  out.worstTrain = worstTrain;
  const crowdAt = () => {
    const r = window.tw.renderer, b = r.board, t = window.tw.clock.now() - r.tripsEpoch, q = window.__q;
    const Q = b.queues, E = b.events, first = Q[q * 10 + 8], count = Q[q * 10 + 9];
    let lo = first;
    for (let j = first; j < first + count; j++) if (E[j * 2] <= t) lo = j;
    const next = E[(lo + 1) * 2];
    return { waiting: Math.round(E[lo * 2 + 1] + (t - E[lo * 2]) * Q[q * 10 + 4]), leftByLast: Math.round(E[lo * 2 + 1]), nextTrainIn: Math.round(next - t) };
  };
  if (sub) {
    await page.evaluate(q => (window.__q = q), sub.q);
    out.sequence = [];
    out.sequence.push(await shoot('platform-before', node => {
      const tw = window.tw, n = tw.world.value.nodes.get(node);
      tw.map.jumpTo({ center: [n.lng, n.lat], zoom: 16.6 });
    }, sub.node, 2500));
    out.sequence[0] = await page.evaluate(crowdAt);
    // just after the next train leaves
    await page.evaluate(dt => { const tw = window.tw; tw.clock.set(tw.clock.now() + dt + 10); tw.overlay.invalidate(); }, out.sequence[0].nextTrainIn);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(ART, 'T-164-platform-after.png') });
    out.sequence.push(await page.evaluate(crowdAt));
    await shoot('platform-dots', () => { window.tw.display.value = { ...window.tw.display.value, platformCrowds: 'dots' }; });
    await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, platformCrowds: 'patch' }; });
    await shoot('station', node => window.tw.select({ kind: 'station', station: String(node) }), sub.node);
    out.inspector = await page.evaluate(() => document.querySelector('.rows.waiting')?.innerText ?? null);
    // the train of that line and direction nearest the station (the one that just left), selected
    out.train = await page.evaluate(q => {
      const tw = window.tw, r = tw.renderer, b = r.board, line = b.queues[q * 10 + 1], run = b.queues[q * 10 + 2], t = tw.clock.now();
      const n = r.net, s = n.stationIds.indexOf(String(b.queues[q * 10]));
      let best = null;
      const stretches = tw.world.value.save.lines.find(l => l.num === line).stops.length - 1;
      for (let i = 0; i < r.tripCount; i++) {
        const prof = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch;
        const at = Math.floor(prof / 6) === line && Math.floor(prof / 3) % 2 === run && tw.tripAt(r.net, prof, dep, t);
        if (!at) continue;
        const o = b.offs[i * 2];
        const loads = o >= 0 ? Array.from(b.loads.subarray(o, o + stretches)).map(Math.round) : null;
        const d = Math.hypot(at.x - n.stations[s * 3], at.y - n.stations[s * 3 + 1]);
        if (!best || d < best.d) best = { prof, dep, d, loads };
      }
      if (best) tw.select({ kind: 'train', line: String(line), profile: best.prof, dep: best.dep });
      return best;
    }, sub.q);
    await page.waitForTimeout(1000);
    out.trainInspector = await page.evaluate(() => document.querySelector('#dock-root')?.innerText.split('On board')[1]?.slice(0, 260) ?? null);
    await page.screenshot({ path: join(ART, 'T-164-train.png') });
    await shoot('brooklyn', () => { const tw = window.tw; tw.select(null); tw.map.jumpTo({ center: [-73.985, 40.684], zoom: 14.2 }); });
  }
  // Frame cost while playing at 960x around that platform at zoom 15: crowds on, off, on.
  const run = async crowds => page.evaluate(async c => {
    const tw = window.tw;
    tw.display.value = { ...tw.display.value, platformCrowds: c };
    tw.map.jumpTo({ center: tw.map.getCenter(), zoom: 15 });
    tw.clock.setSpeed(2);
    await new Promise(r => setTimeout(r, 1000));
    tw.reset();
    await new Promise(r => setTimeout(r, 6000));
    const s = tw.stats();
    const res = { crowds: c, draws: s.draws, drawMs: +s.drawMs.toFixed(3), gapMs: +s.gap.toFixed(1), worstGap: +tw.overlay.frameGap.max().toFixed(1), crowdMs: +tw.renderer.crowdMs.toFixed(2) };
    tw.clock.setRunning(false);
    return res;
  }, crowds);
  out.frames = [await run('patch'), await run('off'), await run('patch')];
  out.hourTurns = await page.evaluate(() => window.tw.client.boardLog.slice(-8).map(e => ({ ms: +e.ms.toFixed(1), redone: e.redone, kB: Math.round(e.bytes / 1024) })));
  out.errors = errors.filter(e => !/favicon|DevTools/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't164-browser.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
