// T-409: two networks with no track between them, a line in Boston and one in Philadelphia, both
// within the regional market's reach. LINES picks which are built (bos, phl or both) in a new game;
// the run records every line's figures and the market's per period, then (with both) times an edit
// in each network to its final. QS="&perfOff=split" gives the market as one node (before T-409);
// SAVE=real builds them onto the real New York save.
//   BASE=http://localhost:6109/ LINES=bos,phl LABEL=both OUT=<dir> node trainworld/checks/t409-split.mjs
import { writeFile, rm, readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
// Playwright as smoke.mjs finds it: PLAYWRIGHT_MODULE, else `playwright`, else the npx cache
async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE');
}
const { chromium } = await playwright();
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6109/';
const QS = process.env.QS ?? '';
const LABEL = process.env.LABEL ?? 'run';
const LINES = (process.env.LINES ?? 'bos,phl').split(',');
const REPS = Number(process.env.REPS ?? 2);
/** SAVE=real: the lines go onto the real New York save instead of a new game */
const REAL = process.env.SAVE === 'real';
const NETS = {
  bos: { name: 'Boston test', centre: [-71.08, 42.35], pts: [[-71.0552, 42.3523], [-71.0752, 42.3474], [-71.1001, 42.3480], [-71.1210, 42.3510]] },
  phl: { name: 'Philadelphia test', centre: [-75.16, 39.955], pts: [[-75.1817, 39.9557], [-75.1636, 39.9526], [-75.1530, 39.9510], [-75.1340, 39.9690]] },
};
const profile = join(OUT, `profile-${LABEL}`);
await rm(profile, { recursive: true, force: true });
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), LABEL, ...a);
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

const waitSolved = (timeout = 240000) => page.waitForFunction(() => {
  const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
  return s && v && v.version === s.version && v.final && !d.client.waiting;
}, undefined, { timeout, polling: 50 });

/** each line's figures by name, and the market's per period from the final results */
const figures = () => page.evaluate(() => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view(), lines = {};
  for (const l of window.tw.world.value.save.lines) {
    const f = v.lines.get(l.id);
    lines[l.name] = f ? JSON.parse(JSON.stringify(f, (k, x) => (x instanceof Map ? [...x] : ArrayBuffer.isView(x) ? Array.from(x) : x))) : null;
    // the line as the solve was given it (run times, trains an hour), to tell input noise from figures
    const sl = s.svc.lines.find((x) => x.id === l.id);
    if (lines[l.name]) lines[l.name].input = sl ? JSON.stringify(sl, (k, x) => (ArrayBuffer.isView(x) ? Array.from(x) : x)) : null;
  }
  const market = s.results.map((r) => r && { summary: r.intercitySummary.map((x, i) => (i === 4 ? 0 : x)), parts: r.intercityParts, corridors: r.corridors, commuteCorridors: r.commuteCorridors });
  return { lines, market, clusters: s.clusters ?? null, cities: s.cities.join('+'), rail: v.railTripsPerDay };
});

/** an edit, then ms to the final of the solve it started */
const timed = (edit) => page.evaluate(async (edit) => {
  const d = window.twDemand, n0 = d.log.length;
  const t0 = performance.now();
  const r = await window.tw.client.edit(edit);
  const end = performance.now() + 240000;
  while (performance.now() < end) {
    const s = d.log.at(-1), v = d.view();
    if (d.log.length > n0 && s && v && v.version === s.version && v.final && !d.client.waiting) {
      return { ok: r.ok, final: Math.round(s.refinedAt - t0), clusters: s.clusters ?? null };
    }
    await new Promise((res) => setTimeout(res, 20));
  }
  return { timeout: true };
}, edit);

const build = (net) => page.evaluate(async (net) => {
  const tw = window.tw, c = tw.client;
  const xy = net.pts.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
  for (let k = 2; k < xy.length; k++) {
    await new Promise((d) => setTimeout(d, 300));
    await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false });
  }
  await new Promise((d) => setTimeout(d, 500));
  for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await new Promise((d) => setTimeout(d, 200)); }
  await c.edit({ op: 'constructAll' });
  await new Promise((d) => setTimeout(d, 500));
  const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: net.name, colour: '#0a7d3b' });
  await c.edit({ op: 'setTrainCount', line: r.line, trains: [10, 8, 6] });
  return r.line;
}, net);

const out = { label: LABEL, qs: QS, lines: LINES, edits: {}, errors };
try {
  if (REAL) {
    // the real New York save first (as t326-clusters.mjs), in a fresh profile's slot
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
  } else await page.goto(`${BASE}?debug=1&new=1&paused=1${QS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
  await waitSolved();
  const built = {};
  for (const id of LINES) {
    await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 12 }), NETS[id].centre);
    await page.waitForTimeout(2500);
    await waitSolved();
    built[id] = await build(NETS[id]);
    await page.waitForTimeout(1000);
    await waitSolved();
  }
  await page.waitForTimeout(1500);
  await waitSolved();
  out.figures = await figures();
  log('figures', JSON.stringify({ clusters: out.figures.clusters, cities: out.figures.cities, rail: out.figures.rail, market: out.figures.market.map((m) => m?.summary.slice(1, 3)) }));
  for (const id of LINES) {
    out.edits[id] = [];
    await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 12 }), NETS[id].centre);
    await page.waitForTimeout(1500);
    await waitSolved();
    for (let i = 0; i < REPS; i++) {
      await page.waitForTimeout(1500);
      const t = await timed({ op: 'setTrainCount', line: built[id], trains: i % 2 ? [10, 8, 6] : [12, 9, 6] });
      out.edits[id].push(t);
      log(`${id} edit`, JSON.stringify(t));
    }
  }
  await page.waitForTimeout(1500);
  await waitSolved();
  out.after = await figures();
} finally {
  await ctx.close();
}
await writeFile(join(OUT, `t409-${LABEL}.json`), JSON.stringify(out, null, 1));
log('errors', JSON.stringify(errors.slice(0, 5)));
