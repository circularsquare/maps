// T-197 / T-161 browser check: linking two stations into one complex in the real app, on New
// York's real network (T-007 save). Picks two busy constructed stations 380-480 m apart in
// different complexes with nothing of theirs within 320 m of each other (so before the link no
// walking transfer joins them), links them through the station inspector ("Link to another
// station", then a click on the other on the map), waits for demand to solve again, and reads the
// transfers between the two (the complex map's curves) and a trip explorer route across the link.
// Then saves the game to a file, unlinks, loads the file again (the link comes back), and unlinks.
// Needs a Vite dev server (BASE; the probes import /src/...) and Playwright (PLAYWRIGHT_MODULE, a
// file URL to its index.mjs). PROFILE_DIR: where the throwaway Chrome profile goes.
//   node trainworld/notes/t197-browser.mjs
// Screenshots: artifacts/trainworld/T-197-*.png. Results: artifacts/trainworld/notes-outputs/t197-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6431';
// PAIR=a,b: link these two station ids instead of searching; TAG=name: files T-197-name-*.png and
// t197-browser-name.json
const TAG = process.env.TAG ? process.env.TAG + '-' : '';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't197-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, acceptDownloads: true, args: ['--enable-unsafe-swiftshader'] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const importSave = async (bytes) => page.evaluate(data => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await importSave(await readFile(join(ROOT, 'T-007-real-nyc.save')));
  const solved = async (after) => page.waitForFunction((v) => { const d = window.twDemand.view(); return d?.round === 3 && d.version > v && d.lines.size > 100; }, after, { timeout: 900000 });
  await solved(-1);
  log('round 3');
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });

  // the pair to link
  const pick = await page.evaluate(async (given) => {
    const cv = await import('/src/game/complexView.ts');
    const c = cv.complexes.value, w = window.tw.world.value, dv = window.twDemand.view();
    const st = w.save.stations.filter(s => s.built);
    const byId = new Map(w.save.stations.map(s => [s.id, s]));
    const b = id => dv.stations.get(id)?.boardings ?? 0;
    const lines = id => w.save.lines.filter(l => l.stops.includes(id)).map(l => l.id);
    const members = id => c.members[c.of.get(id)];
    let best = null;
    if (given) {
      const [A, B] = given.map(id => byId.get(id));
      best = { score: 0, a: A.id, b: B.id, d: Math.round(Math.hypot(A.x - B.x, A.y - B.y)), near: NaN };
    } else for (let i = 0; i < st.length; i++) for (let j = i + 1; j < st.length; j++) {
      const A = st[i], B = st[j], d = Math.hypot(A.x - B.x, A.y - B.y);
      // small complexes, so the changes across the link are among the complex map's 8 curves
      if (d < 380 || d > 480 || c.of.get(A.id) === c.of.get(B.id) || members(A.id).length + members(B.id).length > 6) continue;
      const score = Math.min(b(A.id), b(B.id));
      if (best && score <= best.score) continue;
      const la = lines(A.id), lb = lines(B.id);
      if (!la.length || !lb.length || la.some(l => lb.includes(l))) continue;
      // nothing of A's complex within 320 m of B's: no walking transfer between them yet
      let near = Infinity;
      for (const x of members(A.id)) for (const y of members(B.id)) { const p = byId.get(x), q = byId.get(y); near = Math.min(near, Math.hypot(p.x - q.x, p.y - q.y)); }
      if (near < 320) continue;
      best = { score, a: A.id, b: B.id, d: Math.round(d), near: Math.round(near) };
    }
    const s = id => ({ id, name: byId.get(id).name, lng: byId.get(id).lng, lat: byId.get(id).lat, x: byId.get(id).x, y: byId.get(id).y, members: members(id), lines: lines(id).map(l => w.save.lines.find(x => x.id === l).name), boardings: Math.round(b(id)) });
    return { ...best, A: s(best.a), B: s(best.b) };
  }, process.env.PAIR ? process.env.PAIR.split(',') : null);
  log('pair', pick.A.name, pick.A.lines.join('/'), 'and', pick.B.name, pick.B.lines.join('/'), pick.d, 'm');
  const both = [...pick.A.members, ...pick.B.members];
  // riders changing between the A side and the B side, and the biggest pairs
  const across = () => page.evaluate(async ({ a, b }) => {
    const w = window.tw.world.value, ln = new Map(w.save.lines.map(l => [l.id, l.name])), sn = new Map(w.save.stations.map(s => [s.id, s.name]));
    const r = await window.twDemand.transfers([...a, ...b]);
    if (!r) return null;
    const A = new Set(a), B = new Set(b);
    const x = r.pairs.filter(p => (A.has(p.fromStation) && B.has(p.toStation)) || (B.has(p.fromStation) && A.has(p.toStation)));
    x.sort((p, q) => q.riders - p.riders);
    return { riders: Math.round(x.reduce((s, p) => s + p.riders, 0)), best: x[0] ?? null, top: x.slice(0, 6).map(p => `${ln.get(p.fromLine)} at ${sn.get(p.fromStation)} -> ${ln.get(p.toLine)} at ${sn.get(p.toStation)}: ${Math.round(p.riders)}`) };
  }, { a: pick.A.members, b: pick.B.members });
  const before = await across();
  log('before', before.riders);
  let ends = null;
  const tripAcross = async () => {
    await page.evaluate(async ({ from, to, q }) => { (await import('/src/game/trip.ts')).tripPeriod.value = q; window.tw.select({ kind: 'trip', from, to }); }, { from: ends.from.xy, to: ends.to.xy, q: ends.q });
    await page.waitForFunction(async () => { const t = (await import('/src/game/trip.ts')).tripView.value; return t && !t.asking && t.trip; }, undefined, { timeout: 60000 });
    return page.evaluate(async ({ a, b, q }) => {
      const t = (await import('/src/game/trip.ts')).tripView.value.trip;
      const A = new Set(a), B = new Set(b), w = window.tw.world.value, sn = new Map(w.save.stations.map(s => [s.id, s.name])), ln = new Map(w.save.lines.map(l => [l.id, l.name]));
      const p = t.periods[q];
      if (!p) return null;
      const linkShare = p.routes.filter(r => r.legs.some(l => l.kind === 'walk' && ((A.has(l.from) && B.has(l.to)) || (B.has(l.from) && A.has(l.to))))).reduce((s, r) => s + r.share, 0);
      return { period: q, pTrain: +p.pTrain.toFixed(3), linkShare: +linkShare.toFixed(3), routes: p.routes.slice(0, 4).map(r => `${(100 * r.share).toFixed(0)}%: ` + r.legs.map(l => l.kind === 'ride' ? `${ln.get(l.line)} ${sn.get(l.from)} to ${sn.get(l.to)}` : `walk ${sn.get(l.from)} to ${sn.get(l.to)} ${l.min.toFixed(1)} min`).join(', ')) };
    }, { a: pick.A.members, b: pick.B.members, q: ends.q });
  };
  // as t176-browser.mjs: so the selected station's commuter catchment does not cover the platforms
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).purposeFilter.value = 'shopping'; });

  // link through the inspector and a click on the map
  await page.evaluate(({ a, A, B }) => {
    window.tw.display.value = { ...window.tw.display.value, complexView: false, stationRiders: false, commuters: false };
    window.tw.select({ kind: 'station', station: a });
    window.tw.map.jumpTo({ center: [(A.lng + B.lng) / 2, (A.lat + B.lat) / 2], zoom: 16 });
  }, { a: pick.A.id, A: pick.A, B: pick.B });
  await page.waitForTimeout(800);
  await page.click('button:has-text("Link to another station")');
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(ART, `T-197-${TAG}linking.png`) });
  const version0 = await page.evaluate(() => window.twDemand.view().version);
  const at = await page.evaluate(({ lng, lat }) => { const p = window.tw.map.project([lng, lat]), r = window.tw.map.getCanvas().getBoundingClientRect(); return { x: r.left + p.x, y: r.top + p.y }; }, pick.B);
  await page.mouse.click(at.x, at.y);
  await page.waitForFunction(() => window.tw.world.value.save.links.length === 1, undefined, { timeout: 10000 });
  const notice = await page.evaluate(() => document.querySelector('.notice, [class*=notice]')?.textContent ?? '');
  log('linked', notice);
  await solved(version0);
  log('solved again');
  const after = await across();
  log('after', after.riders);
  // the complex view: the map with its curves and the dashed link
  await page.evaluate(({ a, A, B }) => {
    window.tw.display.value = { ...window.tw.display.value, complexView: true };
    window.tw.select({ kind: 'station', station: a });
    window.tw.map.jumpTo({ center: [(A.lng + B.lng) / 2, (A.lat + B.lat) / 2], zoom: 15.6 });
  }, { a: pick.A.id, A: pick.A, B: pick.B });
  await page.waitForFunction(() => document.querySelectorAll('.complex-map-change').length > 0, undefined, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(ART, `T-197-${TAG}complex.png`) });
  const map = await page.$('.complex-map');
  if (map) await map.screenshot({ path: join(ART, `T-197-${TAG}complex-map.png`) });
  const mapInfo = await page.evaluate(() => {
    const svg = document.querySelector('.complex-map svg');
    return svg ? {
      // T-210: the hover text is in data-tip, not an SVG <title>
      curves: [...svg.querySelectorAll('.complex-map-change')].map(t => t.getAttribute('data-tip')),
      link: [...svg.querySelectorAll('.complex-map-link')].map(t => t.getAttribute('data-tip')),
      captions: [...document.querySelectorAll('.complex-map p')].map(p => p.textContent),
      linkRows: [...document.querySelectorAll('.rows li')].filter(li => li.textContent.includes('Unlink')).map(li => li.textContent),
    } : null;
  });
  // a trip across the link, along the biggest change across it: of the trips from 2-12 stops along
  // the line riders arrive on to 2-10 stops along the line they leave on (period 0, asked of the
  // model directly), the one whose train riders cross the link most
  ends = await page.evaluate(async ({ best, a, b }) => {
    const { askTrip } = await import('/src/game/trip.ts');
    const w = window.tw.world.value, byId = new Map(w.save.stations.map(s => [s.id, s]));
    const A = new Set(a), B = new Set(b);
    const around = (line, station, lo, hi) => {
      const l = w.save.lines.find(x => x.id === line), k = l.stops.indexOf(station), out = [];
      for (let d = lo; d <= hi; d++) for (const j of [k + d, k - d]) if (l.stops[j]) out.push({ id: l.stops[j], line: l.name });
      return out;
    };
    const at = c => { const s = byId.get(c.id); return { ...c, name: s.name, xy: [s.x + 150, s.y + 150] }; };
    let top = null;
    for (const f of around(best.fromLine, best.fromStation, 2, 12)) for (const t of around(best.toLine, best.toStation, 2, 10)) {
      const from = at(f), to = at(t);
      const trip = await askTrip(from.xy, to.xy);
      (trip?.periods ?? []).forEach((p, q) => {
        if (!p) return;
        const share = p.routes.filter(r => r.legs.some(l => l.kind === 'walk' && ((A.has(l.from) && B.has(l.to)) || (B.has(l.from) && A.has(l.to))))).reduce((s, r) => s + r.share, 0);
        if (!top || share > top.share) top = { share, q, from, to };
      });
    }
    return top;
  }, { best: after.best, a: pick.A.members, b: pick.B.members });
  log('trip ends', ends.from.name, ends.to.name, ends.share);
  const tripAfter = await tripAcross();
  log('trip after', JSON.stringify(tripAfter));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(ART, `T-197-${TAG}trip.png`) });
  const versionLinked = await page.evaluate(() => window.twDemand.view().version);

  // save to a file: the link is in it; unlink; load the file: the link is back
  const dl = page.waitForEvent('download');
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const file = await (await dl).path();
  const saved = gunzipSync(await readFile(file));
  const hasBlock = saved.includes(Buffer.from('LNK1'));
  await page.evaluate(({ a }) => { window.tw.display.value = { ...window.tw.display.value, complexView: false }; window.tw.select({ kind: 'station', station: a }); }, { a: pick.A.id });
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(ART, `T-197-${TAG}station.png`) });
  await page.click('button:has-text("Unlink")');
  await page.waitForFunction(() => window.tw.world.value.save.links.length === 0, undefined, { timeout: 10000 });
  const unlinked = true;
  // the same trip without the link (demand solved again)
  await solved(versionLinked);
  const tripBefore = await tripAcross();
  log('trip without the link', JSON.stringify(tripBefore));
  await importSave(await readFile(file));
  await page.waitForFunction(() => window.tw.world.value.save.links.length === 1, undefined, { timeout: 60000 });
  const reloaded = await page.evaluate(() => window.tw.world.value.save.links);
  const complexAfterReload = await page.evaluate(async (a) => (await import('/src/game/complexView.ts')).complexOf(a).length, pick.A.id);
  // unlink again so the profile's autosave ends without it (the profile is thrown away anyway)
  await page.evaluate(({ a }) => window.tw.select({ kind: 'station', station: a }), { a: pick.A.id });
  await page.waitForTimeout(300);
  await page.click('button:has-text("Unlink")');
  await page.waitForFunction(() => window.tw.world.value.save.links.length === 0, undefined, { timeout: 10000 });
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = {
    pair: { a: pick.A, b: pick.B, metres: pick.d, nearestMembers: pick.near }, both: both.length,
    transfersBetween: { before, after }, notice, mapInfo, trip: { from: `${ends.from.name} (${ends.from.line})`, to: `${ends.to.name} (${ends.to.line})`, before: tripBefore, after: tripAfter },
    save: { hasBlock, unlinked, reloaded, complexAfterReload }, errors, gl,
  };
  console.log(JSON.stringify(result, null, 1));
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', `t197-browser${TAG ? '-' + TAG.slice(0, -1) : ''}.json`), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
