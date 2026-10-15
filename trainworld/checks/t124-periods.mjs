// T-124: the New York save, then (a) a train-count change at the high level alone and (b) a
// station link (every level, as a new station), each timed from the edit to its final, and every final figure hashed, so a
// run with ?perfOff=periods (every period re-solved) can be compared: the `sigs` lines must match.
//   BASE=http://localhost:6124/ QS="&perfOff=periods" LABEL=off OUT=<dir> [REPS=2] [PLAYWRIGHT_MODULE=...]
//   node trainworld/checks/t124-periods.mjs   (writes OUT/t124-LABEL.json; a fresh profile there)
import { readFile, writeFile, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
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
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}
const { chromium } = await playwright();
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6124/';
const QS = process.env.QS ?? '';
const LABEL = process.env.LABEL ?? 'run';
const REPS = Number(process.env.REPS ?? 2);
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

/** every final figure the workers sent (ms, setup and memory left out), hashed (t326-clusters.mjs) */
const sig = () => page.evaluate(() => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view();
  let h = 2166136261, k = 0;
  const hk = [2166136261, 2166136261, 2166136261, 2166136261, 2166136261];
  const eat = (x) => { const b = new Uint8Array(new Float64Array([x]).buffer); for (const c of b) { h = Math.imul(h ^ c, 16777619); hk[k] = Math.imul(hk[k] ^ c, 16777619); } };
  const arr = (a) => { if (!a) { eat(-1); return; } eat(a.length); for (const x of a) eat(x); };
  const summ = (a) => arr(a ? Array.from(a).map((x, i) => (i === 4 ? 0 : x)) : null);
  for (const r of s.results) {
    if (!r) { eat(-2); continue; }
    k = 0; summ(r.summary);
    k = 1;
    for (const id of Object.keys(r.citySummary).sort()) summ(r.citySummary[id]);
    for (const id of Object.keys(r.cityParts ?? {}).sort()) arr(r.cityParts[id]);
    k = 2; summ(r.intercitySummary); arr(r.intercityParts); arr(r.corridors); arr(r.commuteCorridors);
    k = 3; arr(r.seg); arr(r.board); arr(r.alight); arr(r.loadOfCrush);
    k = 4; if (r.parts) { arr(r.parts.seg); arr(r.parts.board); arr(r.parts.alight); }
  }
  return { hash: (h >>> 0).toString(16), kinds: hk.map((x) => (x >>> 0).toString(16)).join(' '), version: v.version, rail: v.railTripsPerDay, lines: v.lines.size, clusters: s.clusters ?? null, periods: s.periods ?? null };
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
      return { ok: r.ok, final: Math.round(s.refinedAt - t0), sent: Math.round(s.sentAt - t0), setupMs: s.setupMs.map(Math.round), solves: d.log.length - n0, clusters: s.clusters ?? null, periods: s.periods ?? null };
    }
    if (d.log.length === n0 && performance.now() - t0 > 30000) return { ok: r.ok, noSolve: true };
    await new Promise((res) => setTimeout(res, 20));
  }
  return { timeout: true };
}, edit);

const out = { label: LABEL, qs: QS, stages: {}, level: [], station: [], errors };
try {
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
  const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
  const before = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
  await page.goto(`${BASE}?debug=1${QS}&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
  await waitSolved();
  out.stages.save = await sig();
  log('save', JSON.stringify(out.stages.save));
  const nyLine = await page.evaluate(() => window.tw.world.value.save.lines[0].num);
  // a known count on every level first, so the timed edits change the high level alone
  const prep = await timed({ op: 'setTrainCount', line: nyLine, trains: [20, 12, 6] });
  log('prep', JSON.stringify(prep));
  // new trains enter service over game time, each step a network of its own: settle before each edit
  const settle = async () => {
    for (let quiet = 0; quiet < 8000;) {
      const n = await page.evaluate(() => window.twDemand.log.length);
      await page.waitForTimeout(1000);
      await waitSolved();
      quiet = (await page.evaluate(() => window.twDemand.log.length)) === n ? quiet + 1000 : 0;
    }
  };
  await settle();
  out.stages.prep = await sig();
  // (a) the high level's trains on one line
  for (let i = 0; i < REPS + 1; i++) {
    await settle();
    const t = await timed({ op: 'setTrainCount', line: nyLine, trains: i % 2 ? [20, 12, 6] : [22, 12, 6] });
    out.level.push(t);
    log('high level edit', JSON.stringify(t));
    out.stages[`level${i}`] = await sig();
  }
  // (b) a change at every level
  for (let i = 0; i < REPS; i++) {
    await settle();
    // a constructed station enters the network only once built over game time, so (b) is a link
    // between two stations of a line (a new complex: access and subzones set up again, every
    // period), and then the same link removed
    const [a, b] = await page.evaluate(() => { const l = window.tw.world.value.save.lines[1]; return [Number(l.stops[2]), Number(l.stops[3])]; });
    const t = await timed({ op: 'link', a, b, on: i % 2 === 0 });
    out.station.push({ link: [a, b], ...t });
    log('every level', JSON.stringify({ link: [a, b], ...t }));
    out.stages[`every${i}`] = await sig();
  }
  log('sigs', JSON.stringify(Object.fromEntries(Object.entries(out.stages).map(([k, v]) => [k, v.hash]))));
} finally {
  await ctx.close();
}
await writeFile(join(OUT, `t124-${LABEL}.json`), JSON.stringify(out, null, 1));
log('errors', JSON.stringify(errors.slice(0, 5)));
