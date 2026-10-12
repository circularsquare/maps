// T-125: the world frame in a real browser. Needs `python serve.py` at the maps root
// (localhost:8800/trainworld/ serving app/dist), playwright (PLAYWRIGHT_MODULE) and the real NYC
// fixture. Isolated browser context. Run from the maps root:
//   node trainworld/notes/t125-browser.mjs
// 1. a new game is in the world frame; the legacy real NYC save loads in the legacy frame with the
//    stations, money and line riders it has always had (compared with T-125-browser-before.json when
//    present: the figures recorded by the build before the change);
// 2. a Philadelphia line added to it (the shared Northeast save) keeps the legacy frame; it is
//    exported to trainworld/T-125-northeast.save and loads back the same;
// 3. a new game gets a New York to Philadelphia line whose measured length is compared with the
//    great circle, as is the same line in the legacy frame;
// 4. in the world frame both cities' local demand and the regional market solve, and a
//    Philadelphia station's catchment cells sit where the pack says.
import { readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); if (!ok) throw new Error(`check failed: ${name} ${JSON.stringify(detail)}`); };
const gc = (a, b) => { // great circle, metres, on the sphere the game uses
  const R = 6371008.8, r = Math.PI / 180, dl = (b[0] - a[0]) * r, p1 = a[1] * r, p2 = b[1] * r;
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
};
try {
  const page = await browser.newPage({ viewport: { width: 1300, height: 900 } }), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !/404|blob:captured/.test(m.text())) errors.push(m.text()); });
  await page.addInitScript(() => { window.__blob = null; URL.createObjectURL = b => (window.__blob = b, 'blob:captured'); });
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1&demandWorkers=3');
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client);
  await page.waitForFunction(() => window.twDemand.view()?.round === 3, undefined, { timeout: 120000 });

  const settle = (minLines, afterVersion) => page.waitForFunction(([m, v]) => {
    const d = window.twDemand.view();
    return d && d.round === 3 && d.lines.size >= m && d.version > v;
  }, [minLines, afterVersion], { timeout: 300000 });
  const version = () => page.evaluate(() => window.twDemand.view().version);
  const state = () => page.evaluate(() => {
    const tw = window.tw, w = tw.world.value, d = window.twDemand.view(), c = window.twDemand.cities(), m = tw.moneyState.value;
    return {
      frame: tw.frame().kind, stations: w.save.stations.length, stationList: w.save.stations.map(s => [s.name, +s.lng.toFixed(7), +s.lat.toFixed(7)]),
      edges: w.edges.length, edgeKm: w.edges.reduce((s, e) => s + e.lengthM, 0) / 1000, nodes: w.nodes.size,
      cash: m.cash, fleet: m.fleet, ledgerDays: m.ledger.length, fares: m.fares,
      lines: d.lines.size, ridersPerLine: [...d.lines].map(([id, v]) => [id, Math.round(v.ridersPerDay), v.fullestPct]),
      tripsPerDay: d.tripsPerDay, rail: d.railTripsPerDay, split: d.split,
      cities: [...c].map(([id, v]) => [id, v.railTripsPerDay, v.split]), intercity: window.twDemand.intercity(),
    };
  });
  const sum = (s, f) => s.reduce((t, x) => t + f(x), 0);

  // ---- 1. new game, then the legacy real NYC save
  const fresh = await state();
  check('a new game is in the world frame', fresh.frame === 'lcc', fresh.frame);
  const v0 = await version();
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  const importSave = (b) => page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(b));
  await importSave(bytes);
  await settle(100, v0);
  await page.waitForTimeout(1500);
  const legacy = await state();
  check('the legacy NYC save loads in the legacy frame', legacy.frame === 'equirect', legacy.frame);
  check('legacy NYC: 109 lines, 6000 US$M, fleet and stations as saved', legacy.lines === 109 && legacy.cash === 6000 && legacy.stations > 800, { lines: legacy.lines, cash: legacy.cash, stations: legacy.stations });
  result.legacy = { ...legacy, stationList: undefined, stationsHashInput: legacy.stationList.length };
  // the page start a player has: the autosave written by that import, read back on a reload
  await page.waitForTimeout(1500);
  await page.goto('http://localhost:8800/trainworld/?debug=1&paused=1&demandWorkers=3');
  await page.waitForFunction(() => window.tw?.world.value?.save.stations.length > 800 && window.twDemand?.client);
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size >= 100, undefined, { timeout: 300000 });
  const reloaded = await state();
  check('a reload opens the legacy autosave in the legacy frame, as saved', reloaded.frame === 'equirect' && JSON.stringify(reloaded.stationList) === JSON.stringify(legacy.stationList) && reloaded.cash === legacy.cash && JSON.stringify(reloaded.ridersPerLine) === JSON.stringify(legacy.ridersPerLine), { frame: reloaded.frame, cash: reloaded.cash });

  // ---- 2. the Philadelphia line in the legacy frame: the shared Northeast save
  const v1 = await version();
  const fixture = await page.evaluate(async () => {
    const t = window.tw, run = async op => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    const R = 6371008.8, rad = Math.PI / 180, x = R * (-75.1636 + 73.985) * rad * Math.cos(40.758 * rad), y = R * (39.9526 - 40.758) * rad;
    await run({ op: 'route', from: { kind: 'free', x, y: y - 2000, level: -2 }, to: { kind: 'free', x, y: y + 2000, level: -2 }, pis: [], single: false });
    const edge = t.world.value.edges.at(-1);
    await run({ op: 'addStation', at: { kind: 'node', node: edge.a }, platform: 200, name: 'Philadelphia south' });
    await run({ op: 'addStation', at: { kind: 'node', node: edge.b }, platform: 200, name: 'Philadelphia north' });
    const line = (await run({ op: 'addLine', stops: [edge.a, edge.b], name: 'Philadelphia check', colour: '#17a488' })).line;
    await run({ op: 'setTrainCount', line, trains: [4, 4, 2] }); await run({ op: 'constructAll' });
    return { line };
  });
  await settle(110, v1);
  await page.waitForTimeout(1500);
  const shared = await state();
  check('the shared Northeast save keeps the legacy frame', shared.frame === 'equirect', shared.frame);
  check('Philadelphia riders appear', shared.lines === 110 && shared.cities.find(c => c[0] === 'phl')[1] > 0, shared.cities);
  result.shared = { ...shared, stationList: undefined };
  await page.evaluate(() => { window.__blob = null; window.tw.client.post({ kind: 'export' }); });
  await page.waitForFunction(() => window.__blob);
  const northeast = Buffer.from(await page.evaluate(async () => Array.from(new Uint8Array(await window.__blob.arrayBuffer()))));
  await writeFile('trainworld/T-125-northeast.save', northeast);
  result.northeastSaveBytes = northeast.length;

  // ---- new game, then the shared save back: everything as it was
  const v2 = await version();
  await page.evaluate(() => window.tw.client.post({ kind: 'newGame' }));
  await page.waitForFunction(() => window.tw.frame().kind === 'lcc');
  await page.waitForTimeout(800);
  const v3 = await version();
  await importSave(northeast);
  await settle(110, v3);
  await page.waitForTimeout(1500);
  const again = await state();
  const same = (a, b, what) => check(`${what} unchanged after a save and load`, JSON.stringify(a) === JSON.stringify(b), { a: JSON.stringify(a).slice(0, 200), b: JSON.stringify(b).slice(0, 200) });
  same(shared.stationList, again.stationList, 'stations');
  same([shared.cash, shared.fleet, shared.fares, shared.ledgerDays], [again.cash, again.fleet, again.fares, again.ledgerDays], 'money');
  same(shared.ridersPerLine, again.ridersPerLine, 'line riders');
  same([shared.edgeKm, shared.edges], [again.edgeKm, again.edges], 'track length');

  // ---- 3. a New York to Philadelphia line: measured length against the great circle
  const longLine = (frameName) => page.evaluate(async () => {
    const t = window.tw, run = async op => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    const ny = [-73.9935, 40.7506], ph = [-75.1819, 39.9557];
    const a = t.lngLatToPack(...ny), b = t.lngLatToPack(...ph);
    const before = t.world.value.edges.length;
    await run({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -2 }, to: { kind: 'free', x: b.x, y: b.y, level: -2 }, pis: [], single: false });
    const e = t.world.value.edges.at(-1);
    return { frame: t.frame().kind, lengthM: e.lengthM, ny, ph, edgesAdded: t.world.value.edges.length - before };
  });
  const v4 = await version();
  await page.evaluate(() => window.tw.client.post({ kind: 'newGame' }));
  await page.waitForFunction(() => window.tw.frame().kind === 'lcc');
  await page.waitForTimeout(800);
  const world = await longLine();
  const greatCircle = gc(world.ny, world.ph);
  const worldErr = (world.lengthM / greatCircle - 1) * 100;
  check('world frame: New York to Philadelphia within 0.1% of the great circle', Math.abs(worldErr) < 0.1, { lengthM: world.lengthM, greatCircle, worldErr });
  // the same line in the legacy frame (an old save)
  await importSave(bytes);
  await page.waitForFunction(() => window.tw.frame().kind === 'equirect');
  await page.waitForTimeout(800);
  const old = await longLine();
  const oldErr = (old.lengthM / greatCircle - 1) * 100;
  check('legacy frame: the same line is off by the distortion', Math.abs(oldErr) > 0.2, { lengthM: old.lengthM, oldErr });
  result.longLine = { greatCircleKm: greatCircle / 1000, worldFrameKm: world.lengthM / 1000, worldErrPercent: worldErr, legacyFrameKm: old.lengthM / 1000, legacyErrPercent: oldErr };

  // ---- 4. demand in the world frame: both cities and the regional market, catchment cells where the pack puts them
  await page.evaluate(() => window.tw.client.post({ kind: 'newGame' }));
  await page.waitForFunction(() => window.tw.frame().kind === 'lcc');
  await page.waitForTimeout(800);
  const v5 = await version();
  await page.evaluate(async () => {
    const t = window.tw, run = async op => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    const seg = async (ll, name) => {
      const p = t.lngLatToPack(...ll);
      await run({ op: 'route', from: { kind: 'free', x: p.x, y: p.y - 2000, level: -2 }, to: { kind: 'free', x: p.x, y: p.y + 2000, level: -2 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      await run({ op: 'addStation', at: { kind: 'node', node: e.a }, platform: 200, name: name + ' south' });
      await run({ op: 'addStation', at: { kind: 'node', node: e.b }, platform: 200, name: name + ' north' });
      const line = (await run({ op: 'addLine', stops: [e.a, e.b], name: name + ' local', colour: '#17a488' })).line;
      await run({ op: 'setTrainCount', line, trains: [6, 6, 3] });
    };
    await seg([-73.985, 40.758], 'Times Square');
    await seg([-75.1636, 39.9526], 'City Hall');
    await run({ op: 'constructAll' });
  });
  await settle(2, v5);
  await page.waitForTimeout(1500);
  const solved = await state();
  const phl = solved.cities.find(c => c[0] === 'phl'), nyc = solved.cities.find(c => c[0] === 'nyc');
  check('world frame: both cities carry rail riders', phl[1] > 0 && nyc[1] > 0, solved.cities);
  check('world frame: the regional market solved', solved.intercity?.round === 3 && solved.intercity.tripsPerDay > 100000, solved.intercity && { round: solved.intercity.round, trips: solved.intercity.tripsPerDay });
  await page.evaluate(() => { window.twDemand.activeCity.value = 'phl'; });
  const cells = await page.evaluate(async () => {
    const c = await window.twDemand.cityCells();
    const f = window.tw.frame();
    // first cells in world metres to lng/lat through the game's frame
    const ll = [...Array(5).keys()].map(i => window.tw.packToLngLat(c.xy[2 * i * 1000], c.xy[2 * i * 1000 + 1]));
    return { n: c.n, ll, frame: f.kind };
  });
  check('Philadelphia cells: 110,578 and all in the Philadelphia area', cells.n === 110578 && cells.ll.every(([lon, lat]) => lon > -76.2 && lon < -74.2 && lat > 39.3 && lat < 40.7), cells);
  const station = await page.evaluate(async () => {
    const st = window.tw.world.value.save.stations.find(s => s.name === 'City Hall north');
    const r = await window.twDemand.stationRiders(String(st.id));
    return r && { home: r.home.total, work: r.work.total, maxCell: Math.max(...r.home.cells) };
  });
  check('a Philadelphia station has a catchment', station && station.home > 0 && station.maxCell < 110578, station);
  result.worldFrameDemand = { cities: solved.cities, intercity: solved.intercity && { tripsPerDay: solved.intercity.tripsPerDay, railTripsPerDay: solved.intercity.railTripsPerDay }, cells: cells.n, station };
  await page.screenshot({ path: 'artifacts/trainworld/T-125-world-frame.png' });
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  check('no page errors or WebGL errors', errors.length === 0 && gl === 0, { errors, gl });
  result.errors = errors;
  await writeFile('trainworld/notes/T-125-browser.json', JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result.checks.map(c => [c.ok, c.name])));
  console.log(JSON.stringify({ longLine: result.longLine, legacy: { stations: legacy.stations, cash: legacy.cash, lines: legacy.lines, rail: legacy.rail }, shared: { stations: shared.stations, rail: shared.rail } }));
} catch (err) {
  result.failed = String(err);
  await writeFile('trainworld/notes/T-125-browser.json', JSON.stringify(result, null, 1));
  console.error(String(err));
  process.exitCode = 1;
} finally { await browser.close(); }
