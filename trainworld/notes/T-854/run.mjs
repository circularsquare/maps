// T-854: a real network's demand with the packs' `nocar` replaced (nocarpack.mjs) on the way in.
//   node run.mjs <city> <save file> <mode> <out.json> [url]
// mode: real (packs as they are), zero, flat, const:<s>, dens
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { modify } from './nocarpack.mjs';

const [CITY, SAVE, MODE, OUT] = process.argv.slice(2);
const BASE = (process.argv[6] ?? 'http://localhost:6177/').replace(/\/?$/, '/');
const PACKS = 'C:/Users/anita/projects/maps/trainworld/data/packs';

async function playwright() {
  try { return await import('playwright'); } catch { /* */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t854-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11'] });
const t0 = Date.now();
const log = (...a) => console.log(((Date.now() - t0) / 1000).toFixed(0) + 's', ...a);
const notes = [];
try {
  await ctx.route(/\/packs\/[a-z0-9]+\.bin(\?.*)?$/, async (route) => {
    const req = route.request();
    const id = new URL(req.url()).pathname.match(/\/packs\/([a-z0-9]+)\.bin/)[1];
    if (MODE === 'real' || req.headers()['range']) return route.continue();
    try {
      const { buf, note } = modify(PACKS, id, MODE);
      notes.push(note);
      await route.fulfill({ status: 200, contentType: 'application/octet-stream', body: buf });
    } catch (e) { notes.push(`${id}: ${e.message}`); await route.continue(); }
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&city=${CITY}&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 240000 });
  log('page up');
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  const nLines = await page.evaluate(() => window.tw.world.value.save.lines.length);
  log('save in,', nLines, 'lines');
  let last = -1, since = Date.now();
  for (;;) {
    const s = await page.evaluate(() => { const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.(); return { ok: !!(s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 0), v: v?.version ?? -1, status: String(d?.client?.status) }; });
    if (s.status.startsWith('failed')) throw new Error(s.status);
    if (!s.ok || s.v !== last) { last = s.v; since = Date.now(); }
    else if (Date.now() - since > 20000) break;
    if (Date.now() - t0 > 2400000) { log('gave up'); break; }
    await page.waitForTimeout(2000);
  }
  log('final');
  const res = await page.evaluate(() => {
    const D = window.twDemand, v = D.view(), w = window.tw.world.value;
    const fig = (x) => ({ trips: Math.round(x.tripsPerDay), rail: Math.round(x.railTripsPerDay), split: x.split });
    const cities = {};
    for (const [id, c] of D.cities()) cities[id] = fig(c);
    const lines = w.save.lines.map((l) => ({ name: l.name ?? String(l.num), riders: Math.round(v.lines.get(String(l.id))?.ridersPerDay ?? 0) }));
    const ic = D.intercity();
    return { version: v.version, round: v.round, all: fig(v), cities, intercity: ic ? { trips: Math.round(ic.tripsPerDay), rail: Math.round(ic.railTripsPerDay) } : null, lines };
  });
  log('rail', res.all.rail, JSON.stringify(res.all.split));
  await writeFile(OUT, JSON.stringify({ city: CITY, mode: MODE, notes: [...new Set(notes)], errors, ...res }, null, 1));
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
