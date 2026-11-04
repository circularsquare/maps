// T-1611 scratch: load a save to the weekday final (optionally then JUMP and WAIT_REFINE), collect
// the diag build's tagged heap reports if any (BroadcastChannel "t1169"), and figures per zone,
// station and line.
//   node measure.mjs <base url> <save> <out.json> [timeout min]   env: QS, JUMP=lng,lat,zoom, WAIT_REFINE=1
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const LIB = process.env.LIB ?? 'C:/Users/anita/projects/maps/.tw-trees/t1611/trainworld/checks/lib.mjs';
const { launchChrome, collectErrors, importSave, NOISE } = await import(pathToFileURL(LIB).href);

const [BASE, SAVE, OUT, MIN = '45'] = process.argv.slice(2);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1611-' });
const errors = [];
let result = {};
let page;
const dump = (tag) => page.evaluate(async (tag) => {
  const d = window.twDemand, v = d.view(), c = d.client;
  const cities = {};
  for (const [id, cv] of d.cities()) cities[id] = { rail: cv.railTripsPerDay, trips: cv.tripsPerDay, split: cv.split, final: cv.final };
  const stations = {};
  for (const [id, s] of v?.stations ?? []) stations[id] = s.boardings;
  const lines = {};
  for (const [id, l] of v?.lines ?? []) lines[id] = l.ridersPerDay;
  const D = window.__diag;
  const diag = D ? { light: D.light, done: D.done.slice(-4), full: Object.values(D.full).filter((r) => r.kind !== 'counts') } : null;
  if (D) D.light = [];
  return { tag, at: performance.now(), rail: v?.railTripsPerDay ?? 0, trips: v?.tripsPerDay ?? 0, version: v?.version, final: v?.final, active: d.activeCity.value, cities, stations, lines, status: String(c.status), computeN: c.computeN, memory: c.memory(), lod: c.lodState?.(), diag };
}, tag);
try {
  page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    const D = (window.__diag = { light: [], full: {}, done: [] });
    const ch = new BroadcastChannel('t1169');
    ch.onmessage = (e) => {
      const m = e.data;
      const r = JSON.parse(m.report);
      D.light.push({ w: m.w, aux: m.aux, kind: m.kind, t: m.t, live: r.live, peak: r.peak, mem: r.mem, blocks: r.blocks, ncities: m.cities.length });
      const full = { w: m.w, aux: m.aux, mine: m.mine, kind: m.kind, t: m.t, cities: m.cities, bare: m.bare, version: m.version, ...r };
      if (!m.aux || m.kind === 'done') D.full[`${m.w}:${m.kind}`] = full;
      if (m.kind === 'done' && !m.aux) D.done.push(full);
      if (D.done.length > 6) D.done.shift();
    };
  });
  collectErrors(page, errors, NOISE, /glyphs|ERR_INSUFFICIENT_RESOURCES/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to|T1223/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`, { timeout: 300000, waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const jumpView = async () => {
    if (!process.env.VIEW) return;
    const [lng, lat, zoom] = process.env.VIEW.split(',').map(Number);
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
  };
  await jumpView();
  if (process.env.VIEW) await page.waitForTimeout(4000);
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`${secs()} s loaded`);
  await jumpView();
  const tick = setInterval(() => page.evaluate(() => {
    const d = window.twDemand, c = d.client, s = d.log.at(-1), v = d.view(), l = c.lodState?.();
    return `solves ${d.log.length} last v${s?.version} cities ${s?.cities?.length} markets ${s?.markets?.length} coarse ${s?.coarse?.length}; view v${v?.version} final ${v?.final}; waiting ${!!c.waiting}; lod fineOpen ${l?.fineOpen?.length} coarseOpen ${l?.coarseOpen?.length} refining ${l?.refining}; mem ${JSON.stringify(c.memory().map((m) => [m.wasm, m.heap, m.peakNow]))}`;
  }).then((t) => console.log(`${secs()} s: ${t}`), () => {}), 60000);
  page.once('close', () => clearInterval(tick));
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && s.markets?.length > 0 && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  result.finalSecs = Number(secs());
  await page.waitForTimeout(3000);
  result.final = await dump('final');
  if (process.env.JUMP) {
    const [lng, lat, zoom] = process.env.JUMP.split(',').map(Number);
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
    result.jumpSecs = Number(secs());
  }
  if (process.env.WAIT_REFINE) {
    await page.waitForTimeout(5000);
    await page.waitForFunction(() => { const s = window.twDemand.client.lodState?.(); return s && !s.refining && s.fineOpen?.includes(window.twDemand.activeCity.value) && window.twDemand.view()?.final && !window.twDemand.client.waiting; }, undefined, { timeout: 25 * 60000, polling: 1000 });
    await page.waitForTimeout(3000);
    result.refineSecs = Number(secs());
    result.refined = await dump('refined');
    console.log(`${secs()} s refined`);
  }
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
  try { result.atError = await dump('error'); } catch {}
} finally {
  result.errors = errors.slice(0, 50);
  result.nErrors = errors.length;
  await writeFile(OUT, JSON.stringify(result));
  const f = result.refined ?? result.final ?? result.atError;
  console.log(`wrote ${OUT}; ${errors.length} errors; rail ${f?.rail}; memory ${JSON.stringify(f?.memory)?.slice(0, 600)}`);
  await close();
}
