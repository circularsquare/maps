// T-1223 scratch: load a save to the weekday final; dump memory, timings, and figures per zone,
// station and line, for comparing a LOD load against an all-fine one.
//   node measure.mjs <base url> <save> <out.json> [timeout min]    (QS env: extra query string)
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const LIB = process.env.LIB ?? 'C:/Users/anita/projects/maps/.tw-trees/t1223/trainworld/checks/lib.mjs';
const { launchChrome, collectErrors, importSave, NOISE } = await import(pathToFileURL(LIB).href);

const [BASE, SAVE, OUT, MIN = '45'] = process.argv.slice(2);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1223-' });
const errors = [];
let result = {};
let page;
const dump = () => page.evaluate(async () => {
  const d = window.twDemand, v = d.view(), c = d.client;
  const cities = {};
  for (const [id, cv] of d.cities()) cities[id] = { rail: cv.railTripsPerDay, trips: cv.tripsPerDay, split: cv.split, round: cv.round, final: cv.final };
  const stations = {};
  for (const [id, s] of v?.stations ?? []) stations[id] = s.boardings;
  const where = {};
  for (const s of window.tw.world.value.save.stations) where[s.id] = [s.lng, s.lat];
  const lines = {};
  for (const [id, l] of v?.lines ?? []) lines[id] = l.ridersPerDay;
  const log = d.log.map((s) => ({ version: s.version, sentAt: s.sentAt, roundAt: s.roundAt, cities: s.cities?.length, markets: s.markets?.length, rough: s.rough, memBytes: s.memBytes }));
  return { at: performance.now(), rail: v?.railTripsPerDay ?? 0, trips: v?.tripsPerDay ?? 0, version: v?.version, final: v?.final, active: d.activeCity.value, cities, stations, where, lines, status: String(c.status), computeN: c.computeN, memory: c.memory(), lod: c.lodState?.(), log };
});
try {
  page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs|ERR_INSUFFICIENT_RESOURCES/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to|T1223/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`, { timeout: 300000, waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  if (process.env.VIEW) {
    const [lng, lat, zoom] = process.env.VIEW.split(',').map(Number);
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
    await page.waitForTimeout(4000);
  }
  const tImport = await page.evaluate(() => performance.now());
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`${secs()} s loaded`);
  const tick = setInterval(() => page.evaluate(() => {
    const d = window.twDemand, c = d.client, s = d.log.at(-1), v = d.view(), l = c.lodState?.();
    return `solves ${d.log.length} last v${s?.version} pub ${s?.published} cities ${s?.cities?.length} markets ${s?.markets?.length} coarse ${s?.coarse?.length}; view v${v?.version} final ${v?.final}; waiting ${!!c.waiting}; status ${c.status}; lod fine ${l?.fine?.length} fineOpen ${l?.fineOpen?.length} coarseOpen ${l?.coarseOpen?.length} relevel ${l?.releveling?.length} refining ${l?.refining}; active ${d.activeCity.value}; mem ${JSON.stringify(c.memory().map((m) => m.wasm))}`;
  }).then((t) => console.log(`${secs()} s: ${t}`), () => {}), 30000);
  page.once('close', () => clearInterval(tick));
  if (process.env.VIEW) {
    const [lng, lat, zoom] = process.env.VIEW.split(',').map(Number);
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
  }
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && s.markets?.length > 0 && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  result.tImport = tImport;
  result.finalSecs = Number(secs());
  await page.waitForTimeout(3000);
  result.final = await dump();
  // WAIT_REFINE: then wait until the zone in view is fine and its refine is shown (T-1223)
  if (process.env.JUMP) {
    const [lng, lat, zoom] = process.env.JUMP.split(',').map(Number);
    await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
    result.jumpSecs = Number(secs());
    await page.waitForTimeout(3000);
    result.jumped = await dump();
    if (process.env.SHOT) {
      await page.waitForTimeout(1500);
      await page.screenshot({ path: process.env.SHOT });
      // a rough zone's city view, opened from here
      if (process.env.SHOT2) {
        await page.evaluate((z) => window.tw.select({ kind: 'city', city: z }), process.env.SHOT2_ZONE);
        await page.waitForTimeout(1500);
        await page.screenshot({ path: process.env.SHOT2 });
        await page.evaluate(() => window.tw.select(null));
      }
    }
  }
  if (process.env.WAIT_REFINE) {
    await page.waitForFunction(() => { const s = window.twDemand.client.lodState?.(); return s && !s.refining && s.fineOpen?.includes(window.twDemand.activeCity.value) && window.twDemand.view()?.final && !window.twDemand.client.waiting; }, undefined, { timeout: 20 * 60000, polling: 1000 });
    await page.waitForTimeout(3000);
    result.refineSecs = Number(secs());
    result.refined = await dump();
    console.log(`${secs()} s refined`);
  }
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
  try { result.atError = await dump(); } catch {}
} finally {
  result.errors = errors;
  await writeFile(OUT, JSON.stringify(result));
  const f = result.refined ?? result.final ?? result.atError;
  console.log(`wrote ${OUT}; ${errors.length} errors; rail ${f?.rail}; memory ${JSON.stringify(f?.memory)?.slice(0, 600)}`);
  await close();
}
