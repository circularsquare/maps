// T-1734 scratch: load a save to the weekday final, then build a short line every EVERY seconds for
// MINUTES minutes around CENTER; log every final that lands (its version, when) and every view the
// page shows; once building stops wait for the last final, dump it, then solve the same network
// afresh (fresh market parts, every population from free flow) and dump that, to compare to the bit.
//   node steady.mjs <base url> <save> <out.json>
// env: QS, VIEW lng_lat_zoom, CENTER lng_lat, EVERY (s, 60), MINUTES (35), SPACING deg (0.02)
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const LIB = 'C:/Users/anita/projects/maps/.tw-trees/t1734/trainworld/checks/lib.mjs';
const { launchChrome, collectErrors, importSave, NOISE, acrossLine } = await import(pathToFileURL(LIB).href);

const [BASE, SAVE, OUT] = process.argv.slice(2);
const EVERY = Number(process.env.EVERY ?? 60), MINUTES = Number(process.env.MINUTES ?? 35), SP = Number(process.env.SPACING ?? 0.02);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(1);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1734-' });
const errors = [];
const result = { edits: [], finals: [], views: [] };
let page;
const dump = () => page.evaluate(() => {
  const d = window.twDemand, v = d.view();
  const cities = {};
  for (const [id, cv] of d.cities()) cities[id] = { rail: cv.railTripsPerDay, trips: cv.tripsPerDay, split: cv.split };
  const lines = {};
  for (const [id, l] of v?.lines ?? []) lines[id] = l.ridersPerDay;
  const stations = {};
  for (const [id, s] of v?.stations ?? []) stations[id] = s.boardings;
  return { version: v?.version, final: v?.final, rail: v?.railTripsPerDay ?? 0, cities, lines, stations, nLines: window.tw.world.value.save.lines.length };
});
const idle = (after) => page.waitForFunction((after) => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view();
  return String(d.client.status).startsWith('failed') || (!!v && s && s.version > after && v.version === s.version && v.final && !d.client.waiting);
}, after, { timeout: 90 * 60000, polling: 1000 });
try {
  page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs|ERR_INSUFFICIENT_RESOURCES/);
  page.on('console', (m) => { if (/panicked|out of memory|T1733|T1734|rror/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`, { timeout: 300000, waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const [vl, va, vz] = (process.env.VIEW ?? '139.70_35.68_11').split('_').map(Number);
  await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [vl, va, vz]);
  await page.waitForTimeout(3000);
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [vl, va, vz]);
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && s.markets?.length > 0 && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: 90 * 60000, polling: 2000 });
  console.log(`${secs()} s loaded to its final`);
  result.loadSecs = Number(secs());
  await page.waitForTimeout(5000);
  // every final that lands and every shown view, polled each second from here on
  let lastFinal = -1, lastView = '';
  const watch = setInterval(() => page.evaluate(() => {
    const d = window.twDemand, v = d.view(), s = d.log.at(-1);
    const finals = d.log.filter((x) => x.published >= 3 && !x.rough).map((x) => ({ version: x.version, sent: x.sentAt, at: x.roundAt?.[3], held: x.heldFrom, local: x.localShownAt }));
    return { view: v ? `${v.version}:${v.round}:${v.final}` : '', newest: s?.version, finals, nLines: window.tw.world.value.save.lines.length, ms: performance.now() };
  }).then((w) => {
    for (const f of w.finals) if (f.version > lastFinal) { lastFinal = f.version; result.finals.push({ ...f, secs: Number(secs()) }); console.log(`${secs()} s: final of v${f.version} (sent ${((w.ms - f.sent) / 1000).toFixed(0)} s ago; newest v${w.newest}; ${w.nLines} lines)`); }
    if (w.view !== lastView) { lastView = w.view; result.views.push({ view: w.view, secs: Number(secs()) }); }
  }, () => {}), 1000);
  const [cl, ca] = (process.env.CENTER ?? '139.74_35.69').split('_').map(Number);
  const spots = [];
  for (let k = 0; k < 60; k++) spots.push([cl + SP * 3 * ((k % 3) - 1), ca + SP * (Math.floor(k / 3) % 10 - 5) + 0.003 * Math.floor(k / 30)]);
  const before = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
  const tStart = Date.now();
  for (let k = 0; Date.now() - tStart < MINUTES * 60000; k++) {
    const tEdit = Date.now();
    let line = null;
    try { line = await acrossLine(page, spots[k % spots.length], 2); } catch (err) { console.log(`edit ${k} failed: ${err}`); }
    result.edits.push({ k, line, secs: (tEdit - t0) / 1000, built: (Date.now() - tEdit) / 1000 });
    console.log(`${secs()} s: edit ${k} built line ${line} (${((Date.now() - tEdit) / 1000).toFixed(1)} s)`);
    const wait = tEdit + EVERY * 1000 - Date.now();
    if (wait > 0) await page.waitForTimeout(wait);
  }
  console.log(`${secs()} s: building stopped`);
  if (result.edits.length) await idle(before);
  await page.waitForTimeout(3000);
  result.last = await dump();
  console.log(`${secs()} s: last final shown, v${result.last.version}`);
  const v0 = result.last.version;
  await page.evaluate(() => { const c = window.twDemand.client; c.freshParts = true; c.resubmit(); });
  await idle(v0);
  await page.waitForTimeout(3000);
  result.fresh = await dump();
  console.log(`${secs()} s: fresh solve shown, v${result.fresh.version}`);
  clearInterval(watch);
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
} finally {
  result.errors = errors;
  await writeFile(OUT, JSON.stringify(result));
  console.log(`wrote ${OUT}; ${errors.length} errors`);
  await close();
}
