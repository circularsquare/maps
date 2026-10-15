// T-326: the New York save plus a Toronto line; edits in each cluster timed to their final, and
// every final figure hashed, so a run with ?perfOff=clusters (one cluster) can be compared: the
// `sigs` line of the two runs must match.
//   BASE=http://localhost:6126/ QS="&perfOff=clusters" LABEL=before OUT=<dir> PLAYWRIGHT_MODULE=...
//   node trainworld/checks/t326-clusters.mjs   (writes OUT/measure-LABEL.json; a fresh profile there)
import { readFile, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6126/';
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

/** every final figure the workers sent (ms, setup and memory left out), hashed, plus the totals */
const sig = () => page.evaluate(() => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view();
  let h = 2166136261;
  const eat = (x) => { const b = new Uint8Array(new Float64Array([x]).buffer); for (const c of b) h = Math.imul(h ^ c, 16777619); };
  const arr = (a) => { if (!a) { eat(-1); return; } eat(a.length); for (const x of a) eat(x); };
  const summ = (a) => arr(a ? Array.from(a).map((x, i) => (i === 4 ? 0 : x)) : null);
  for (const r of s.results) {
    if (!r) { eat(-2); continue; }
    summ(r.summary);
    for (const id of Object.keys(r.citySummary).sort()) summ(r.citySummary[id]);
    for (const id of Object.keys(r.cityParts ?? {}).sort()) arr(r.cityParts[id]);
    summ(r.intercitySummary); arr(r.intercityParts); arr(r.corridors); arr(r.commuteCorridors);
    arr(r.seg); arr(r.board); arr(r.alight); arr(r.loadOfCrush);
    if (r.parts) { arr(r.parts.seg); arr(r.parts.board); arr(r.parts.alight); }
  }
  return { hash: (h >>> 0).toString(16), version: v.version, rail: v.railTripsPerDay, split: [v.split.train, v.split.walk, v.split.drive], lines: v.lines.size, clusters: s.clusters ?? null, cities: s.cities.join('+') };
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
      return { ok: r.ok, final: Math.round(s.refinedAt - t0), first: s.firstAt ? Math.round(s.firstAt - t0) : null, sent: Math.round(s.sentAt - t0), setupMs: s.setupMs.map(Math.round), solves: d.log.length - n0, clusters: s.clusters ?? null };
    }
    await new Promise((res) => setTimeout(res, 20));
  }
  return { timeout: true };
}, edit);

const out = { label: LABEL, qs: QS, stages: {}, tor: [], ny: [], errors };
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
  out.stages.ny = await sig();
  log('ny', JSON.stringify(out.stages.ny));
  // to Toronto, wait for it to open, then a line up Yonge St (Union to Eglinton)
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.39, 43.67], zoom: 12 }));
  await page.waitForTimeout(2500);
  await waitSolved();
  const PTS = [[-79.3806, 43.6453], [-79.3832, 43.6560], [-79.3866, 43.6710], [-79.3904, 43.6870], [-79.3985, 43.7055]];
  const built = await page.evaluate(async (PTS) => {
    const tw = window.tw, c = tw.client;
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
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
    const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Yonge test', colour: '#0a7d3b' });
    await c.edit({ op: 'setTrainCount', line: r.line, trains: [10, 8, 6] });
    return r.line;
  }, PTS);
  await page.waitForTimeout(1000);
  await waitSolved();
  out.stages.both = await sig();
  log('ny+tor', JSON.stringify(out.stages.both));
  const nyLine = await page.evaluate(() => window.tw.world.value.save.lines[0].num);
  for (let i = 0; i < REPS; i++) {
    await page.waitForTimeout(1500);
    const t = await timed({ op: 'setTrainCount', line: built, trains: i % 2 ? [10, 8, 6] : [12, 9, 6] });
    out.tor.push(t);
    log('toronto edit', JSON.stringify(t));
    if (i === 0) out.stages.torEdit = await sig();
  }
  for (let i = 0; i < REPS; i++) {
    await page.waitForTimeout(1500);
    const t = await timed({ op: 'setTrainCount', line: nyLine, trains: i % 2 ? [20, 12, 6] : [22, 14, 7] });
    out.ny.push(t);
    log('new york edit', JSON.stringify(t));
    if (i === 0) out.stages.nyEdit = await sig();
  }
  log('sigs', JSON.stringify(Object.fromEntries(Object.entries(out.stages).map(([k, v]) => [k, v.hash]))));
} finally {
  await ctx.close();
}
await writeFile(join(OUT, `measure-${LABEL}.json`), JSON.stringify(out, null, 1));
log('errors', JSON.stringify(errors.slice(0, 5)));
