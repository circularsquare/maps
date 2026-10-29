// T-1733 scratch: load a save to the weekday final, build a short line at EDIT (lng,lat), then dump
// the shown figures when the edit's local result lands and again when its whole solve lands.
//   node edit.mjs <base url> <save> <out.json> [timeout min]
// env: QS extra query, VIEW lng,lat,zoom before import, EDIT lng,lat, EDIT2 lng,lat (a second edit
// right after the local result), ZONE the zone to report
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const LIB = new URL('../../checks/lib.mjs', import.meta.url).href;
// T-1735: EDIT or EDIT2 'express', with EXPRESS=<name pattern>,<k>: a copy of the longest line whose
// name matches, stopping at every k-th of its stops (and its last), in place of a short line
const { launchChrome, collectErrors, importSave, NOISE, acrossLine } = await import(LIB);

const [BASE, SAVE, OUT, MIN = '60'] = process.argv.slice(2);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(1);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1733-' });
const errors = [];
const result = {};
let page;
const dump = () => page.evaluate(() => {
  const d = window.twDemand, v = d.view();
  const cities = {};
  for (const [id, cv] of d.cities()) cities[id] = { rail: cv.railTripsPerDay, trips: cv.tripsPerDay, split: cv.split };
  const lines = {};
  for (const [id, l] of v?.lines ?? []) lines[id] = l.ridersPerDay;
  const longer = {};
  for (const [id, l] of v?.purposes?.get('longer')?.lines ?? []) longer[id] = l.ridersPerDay;
  const stations = {};
  for (const [id, s] of v?.stations ?? []) stations[id] = s.boardings;
  const s = d.log.at(-1);
  const svcLines = Object.fromEntries((s?.svc?.lines ?? []).map((l) => [l.id, { tph: l.tph, stops: l.stops.length }]));
  return { svcLines, at: performance.now(), rail: v?.railTripsPerDay ?? 0, version: v?.version, final: v?.final, cities, lines, longer, stations, memory: d.client.memory(), solve: { version: s?.version, sentAt: s?.sentAt, local: s?.local, clusters: s?.clusters, keys: [...(s?.clusterRes?.keys() ?? [])], at: [...(s?.clusterRes?.values() ?? [])].map((e) => e.at) } };
});
const finalOf = (after) => page.waitForFunction((after) => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view();
  return String(d.client.status).startsWith('failed') || (!!v && s && s.version > after && v.version === s.version && v.final && !d.client.waiting);
}, after, { timeout: Number(MIN) * 60000, polling: 1000 });
// the new line's riders shown (a local result or the whole solve's)
const localOf = (line) => page.waitForFunction((line) => {
  const v = window.twDemand.view();
  return (v?.lines?.get(String(line))?.ridersPerDay ?? 0) > 0;
}, line, { timeout: Number(MIN) * 60000, polling: 250 });
const express = (spec) => page.evaluate(async ([pat, k, trains]) => {
  const c = window.tw.client, s = window.twDemand.log.at(-1).svc;
  const at = new Map(s.stations.map((x) => [x.id, x]));
  const km = (l) => { let d = 0; for (let i = 1; i < l.stops.length; i++) { const a = at.get(l.stops[i - 1]), b = at.get(l.stops[i]); if (a && b) d += Math.hypot(a.x - b.x, a.y - b.y); } return d / 1000; };
  const re = new RegExp(pat, 'i');
  let best = null;
  // (a pattern "km:<n>": the line nearest n km long)
  const near = /^km:(\d+)$/.exec(pat);
  if (near) { const t = Number(near[1]); for (const l of s.lines) if (l.stops.length >= 8 && (!best || Math.abs(km(l) - t) < Math.abs(km(best) - t))) best = l; }
  else for (const l of s.lines) if ((!pat || re.test(l.name ?? '')) && (!best || km(l) > km(best))) best = l;
  // (no name matched: the longest line, as T-1650's express)
  if (!best) for (const l of s.lines) if (!best || km(l) > km(best)) best = l;
  // (k 0: about seven stops)
  if (!k) k = Math.ceil((best.stops.length - 1) / 6);
  const stops = best.stops.filter((_, i) => i % k === 0);
  if (stops[stops.length - 1] !== best.stops[best.stops.length - 1]) stops.push(best.stops[best.stops.length - 1]);
  const r = await c.edit({ op: 'copyLine', line: Number(best.id) });
  const r2 = await c.edit({ op: 'setStops', line: r.line, stops: stops.map(Number) });
  // (a copy runs no trains of its own: about 6 an hour at the peak on a 3-hour run each way)
  const r3 = await c.edit({ op: 'setTrainCount', line: r.line, trains });
  console.log(`T1735 express: copy of line ${best.id} ${best.name ?? ''} (${best.stops.length} stops, ${Math.round(km(best))} km) as ${r.line}, ${stops.length} stops: ${JSON.stringify(r2)} ${JSON.stringify(r3)}`);
  return r.line;
}, [spec.split(',')[0], Number(spec.split(',')[1] ?? 5), (process.env.EXTRAINS ?? '40_30_20').split('_').map(Number)]);
const edit = async (where, tag) => {
  const before = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
  const [lng, lat] = where.split(/[,_]/).map(Number);
  const tEdit = Date.now();
  const line = where === 'express' ? await express(process.env.EXPRESS) : await acrossLine(page, [lng, lat], 2);
  const tBuilt = Date.now();
  console.log(`${secs()} s: ${tag} built line ${line} (${((tBuilt - tEdit) / 1000).toFixed(1)} s to build)`);
  // (an express may have no riders of the cities' own: wait for it in the shown figures instead)
  if (where === 'express') await page.waitForFunction((line) => !!window.twDemand.view()?.lines?.has(String(line)), line, { timeout: Number(MIN) * 60000, polling: 250 });
  else await localOf(line);
  const tLocal = Date.now();
  result[tag] = { line, editSecs: (tEdit - t0) / 1000, builtSecs: (tBuilt - tEdit) / 1000, localSecs: (tLocal - tEdit) / 1000 };
  result[tag].local = await dump();
  console.log(`${secs()} s: ${tag} local result shown ${((tLocal - tEdit) / 1000).toFixed(1)} s after the edit began; keys ${JSON.stringify(result[tag].local.solve.keys)}`);
  return before;
};
try {
  page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs|ERR_INSUFFICIENT_RESOURCES/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to|T1223|T1609|T1733|T1735|rror/.test(m.text()) && (!/T1733DBG/.test(m.text()) || !process.env.QUIETDBG)) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`, { timeout: 300000, waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const [vl, va, vz] = (process.env.VIEW ?? '139.70,35.68,11').split(/[,_]/).map(Number);
  await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [vl, va, vz]);
  await page.waitForTimeout(3000);
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [vl, va, vz]);
  console.log(`${secs()} s loaded`);
  const tick = setInterval(() => page.evaluate(() => {
    const d = window.twDemand, c = d.client, s = d.log.at(-1), v = d.view();
    return `v${s?.version} pub ${s?.published} keys ${[...(s?.clusterRes?.keys() ?? [])].length}; view v${v?.version} final ${v?.final}; waiting ${!!c.waiting}; mem ${JSON.stringify(c.memory().map((m) => m.wasm))}`;
  }).then((t) => console.log(`${secs()} s: ${t}`), () => {}), 60000);
  page.once('close', () => clearInterval(tick));
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && s.markets?.length > 0 && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  result.loadSecs = Number(secs());
  await page.waitForTimeout(5000);
  result.base = await dump();
  const before = await edit(process.env.EDIT, 'edit1');
  if (process.env.EDIT2) {
    await edit(process.env.EDIT2, 'edit2');
  }
  await writeFile(OUT.replace(/\.json$/, '.local.json'), JSON.stringify(result));
  if (process.env.NOWHOLE) throw new Error('NOWHOLE: stopped after the local result');
  await finalOf(before);
  const tWhole = Date.now();
  await page.waitForTimeout(2000);
  result.whole = await dump();
  result.wholeSecs = (tWhole - t0) / 1000;
  console.log(`${secs()} s: whole solve shown, ${(result.wholeSecs - result.edit1.editSecs).toFixed(1)} s after the first edit began`);
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
  try { result.atError = await dump(); } catch {}
} finally {
  result.errors = errors;
  await writeFile(OUT, JSON.stringify(result));
  console.log(`wrote ${OUT}; ${errors.length} errors`);
  await close();
}
