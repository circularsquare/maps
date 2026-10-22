// T-301: a long pan across the Northeast and on to Toronto, then back, with the commuter bubbles on
// (so the zones beside the view open too). Samples every worker's WASM memory and the packs open,
// and times how long each zone in view takes to be open after the view arrives (a zone closed on the
// way out reopens on the way back).
//   BASE=http://localhost:6101/ LABEL=after SAVE=1 OUT=<dir> node trainworld/checks/t301-pan.mjs
// SAVE=1 imports T-007-real-nyc.save first (else a ?new=1 game). DWELL ms at each stop (5000),
// REST ms at each end (75000, longer than the close delay), QS more URL switches. Writes
// OUT/pan-LABEL.json.
import { readFile, writeFile, rm, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6101/';
const LABEL = process.env.LABEL ?? 'run';
const SAVE = process.env.SAVE === '1';
const DWELL = Number(process.env.DWELL ?? 5000);
const REST = Number(process.env.REST ?? 75000);
const QS = process.env.QS ?? '';
await mkdir(OUT, { recursive: true });
const profile = join(OUT, `profile-${LABEL}`);
await rm(profile, { recursive: true, force: true });
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), LABEL, ...a);
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

// wdc to the New England counties, then up the Mohawk and along Lake Ontario to Toronto
const OUTWARD = [
  ['wdc', -77.0365, 38.8977], ['bal', -76.6122, 39.2904], ['phl', -75.1636, 39.9526], ['nyc', -73.985, 40.758],
  ['nhv', -72.9279, 41.3083], ['hfd', -72.6851, 41.7637], ['spf', -72.5898, 42.1015], ['wor', -71.8023, 42.2626],
  ['bos', -71.0589, 42.3601], ['pvd', -71.4128, 41.824], ['nlc', -72.0759, 41.5243], ['wnd', -72.2087, 41.7107],
  ['lit', -73.1212, 41.8007], [null, -73.75, 42.65], [null, -76.15, 43.05], [null, -77.6, 43.15], ['tor', -79.3832, 43.6534],
];
const ZOOM = 10.5;

const waitSolved = (timeout = 240000) => page.waitForFunction(() => {
  const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
  return s && v && v.version === s.version && v.final && !d.client.waiting;
}, undefined, { timeout, polling: 100 });

const state = () => page.evaluate(() => {
  const d = window.twDemand, c = d.client;
  const mem = (d.start.workers ?? []).map((w) => w?.memBytes ?? 0);
  const s = d.log.at(-1);
  const solveMem = s?.memBytes ?? [];
  return {
    t: performance.now(),
    mem: mem.map((m, i) => Math.max(m, solveMem[i] ?? 0)),
    full: [...c.opened].filter((x) => !x.startsWith('market:')).sort(),
    market: [...c.opened].filter((x) => x.startsWith('market:')).length,
    bare: [...c.bareOpen.keys()].sort(),
    view: d.activeCity.peek(),
    nearby: d.nearby ? d.nearby() : null,
  };
});

const samples = [];
let peak = [];
let sampling = true;
const sampler = (async () => {
  while (sampling) {
    try {
      const s = await state();
      samples.push(s);
      peak = s.mem.map((m, i) => Math.max(m, peak[i] ?? 0));
    } catch { /* navigating */ }
    await new Promise((r) => setTimeout(r, 500));
  }
})();

/** jump to (lon, lat); ms until the zone in view (if `id`) is open, bare or in every worker */
const go = async (id, lon, lat) => {
  const t0 = Date.now();
  await page.evaluate(([lon, lat, z]) => { window.tw.map.jumpTo({ center: [lon, lat], zoom: z }); }, [lon, lat, ZOOM]);
  let r = { focusMs: null, openMs: null };
  if (id) {
    // ms until the focus is the zone, then ms from then until it is open (0: open already)
    r = await page.evaluate(async (id) => {
      const d = window.twDemand, t0 = performance.now();
      const isOpen = () => d.client.opened.has(id) || d.client.bareOpen.has(id);
      while (d.activeCity.peek() !== id && performance.now() - t0 < 60000) await new Promise((res) => setTimeout(res, 10));
      const t1 = performance.now();
      while (!isOpen() && performance.now() - t1 < 60000) await new Promise((res) => setTimeout(res, 10));
      return { focusMs: Math.round(t1 - t0), openMs: Math.round(performance.now() - t1) };
    }, id);
  }
  const rest = DWELL - (Date.now() - t0);
  if (rest > 0) await page.waitForTimeout(rest);
  return r;
};

const out = { label: LABEL, save: SAVE, base: BASE, dwell: DWELL, rest: REST, outward: [], back: [], errors };
try {
  if (SAVE) {
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
  } else {
    await page.goto(`${BASE}?new=1&debug=1${QS}&v=${Date.now()}`);
  }
  await page.waitForFunction(() => !!window.twDemand && !!window.tw?.display, undefined, { timeout: 120000 });
  await waitSolved();
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'home' }; });
  await page.waitForTimeout(2000);
  out.start = await state();
  log('start', JSON.stringify(out.start));
  for (const [id, lon, lat] of OUTWARD) {
    const ms = await go(id, lon, lat);
    out.outward.push({ id, ...ms });
    log('out', id, JSON.stringify(ms));
  }
  await page.waitForTimeout(REST);
  out.atToronto = await state();
  log('at toronto', JSON.stringify(out.atToronto));
  for (const [id, lon, lat] of OUTWARD.slice(0, -1).reverse()) {
    const ms = await go(id, lon, lat);
    out.back.push({ id, ...ms });
    log('back', id, JSON.stringify(ms));
  }
  await page.waitForTimeout(REST);
  await waitSolved().catch(() => {});
  out.end = await state();
  log('end', JSON.stringify(out.end));
  out.loads = await page.evaluate(() => window.twDemand.start.loads.map((l) => ({ item: l.item, bare: !!l.bare, req: Math.round(l.requestedAt), dl: l.downloadedAt !== undefined ? Math.round(l.downloadedAt - l.requestedAt) : null, open: l.openedAt !== undefined ? Math.round(l.openedAt - l.requestedAt) : null, closed: l.closedAt !== undefined ? Math.round(l.closedAt) : null })));
  // a load of an item loaded before is a reopening
  const seen = new Set();
  out.reopens = out.loads.filter((l) => { const k = `${l.item}:${l.bare}`, again = seen.has(k); seen.add(k); return again; });
  out.final = await page.evaluate(() => { const v = window.twDemand.view(); return v ? { rail: v.railTripsPerDay, version: v.version } : null; });
} finally {
  sampling = false;
  await sampler;
  await ctx.close();
}
out.peakMB = peak.map((m) => Math.round(m / 1048576));
out.endMB = (out.end?.mem ?? []).map((m) => Math.round(m / 1048576));
out.maxOpen = Math.max(...samples.map((s) => s.full.length + s.bare.length));
out.samples = samples.map((s) => ({ t: Math.round(s.t), mb: s.mem.map((m) => Math.round(m / 1048576)), full: s.full.length, bare: s.bare.length, market: s.market, view: s.view }));
await writeFile(join(OUT, `pan-${LABEL}.json`), JSON.stringify(out, null, 1));
log('peak MB', JSON.stringify(out.peakMB), 'end MB', JSON.stringify(out.endMB), 'max open', out.maxOpen, 'final', JSON.stringify(out.final));
log('back openMs', JSON.stringify(out.back.map((b) => [b.id, b.focusMs, b.openMs])));
log('reopens (item, download ms, open ms)', JSON.stringify((out.reopens ?? []).map((l) => [l.item, l.dl, l.open])));
log('errors', JSON.stringify(errors.slice(0, 5)));
await rm(profile, { recursive: true, force: true });
