// T-206 browser check: changing a line's train count moves no train. On New York's real network
// (T-007 save) in the real app, paused at midday: the trains of one line before and after an edit
// that adds trains, then one that removes them, from the GPU's own positions (`renderer.probe`);
// then the clock run on to see trains join and leave at the ends. Needs a Vite dev server (BASE,
// default http://localhost:6931; file watching off) and Playwright (PLAYWRIGHT_MODULE, a file URL
// to its index.mjs). PROFILE_DIR: where the throwaway Chrome profile goes.
//   node trainworld/notes/t206-browser.mjs
// Screenshots: artifacts/trainworld/T-206-*.png. Results: artifacts/trainworld/notes-outputs/t206-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6931';
const LINE = process.env.LINE ?? 'subway L';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't206-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value, undefined, { timeout: 240000 });
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 100, undefined, { timeout: 120000 });
  const line = await page.evaluate(name => {
    const l = window.tw.world.value.save.lines.find(l => l.name === name) ?? window.tw.world.value.save.lines.find(l => l.name.startsWith('subway'));
    return { num: l.num, name: l.name, id: l.id };
  }, LINE);
  out.line = line;
  // 11:20 on the save's day, paused; the camera on the line
  const at = await page.evaluate(num => {
    const tw = window.tw;
    tw.clock.setRunning(false);
    const t = Math.floor(tw.clock.now() / 86400) * 86400 + 11 * 3600 + 20 * 60;
    tw.clock.set(t);
    const w = tw.world.value, l = w.save.lines.find(l => l.num === num);
    const pts = l.stops.map(s => w.save.stations.find(st => st.id === s)).filter(Boolean);
    const lng = pts.map(p => p.lng), lat = pts.map(p => p.lat);
    tw.map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 60, animate: false });
    tw.overlay.invalidate();
    return t;
  }, line.num);
  await page.waitForFunction(t => window.tw.renderer.tripsEpoch === Math.floor(t / 3600) * 3600 && window.tw.renderer.tripCount > 0, at, { timeout: 60000 });
  await page.waitForTimeout(1500);
  // the line's trains as the GPU places them: key (profile, departure) and position (local units)
  const trains = () => page.evaluate(num => {
    const tw = window.tw, r = tw.renderer, t = tw.clock.now(), gpu = r.probe(t), res = [];
    for (let i = 0; i < r.tripCount; i++) {
      const prof = r.trips[i * 2];
      if (Math.floor(prof / 6) !== num || gpu[i * 2] > 1e8) continue;
      res.push({ key: prof % 6 + '@' + (r.trips[i * 2 + 1] + r.tripsEpoch).toFixed(2), x: gpu[i * 2], y: gpu[i * 2 + 1], end: r.tripEnds ? r.tripEnds[i] + r.tripsEpoch - t : null });
    }
    return { t, version: r.tripVersion ?? null, list: res };
  }, line.num);
  // metres per local unit along this line (its length over its drawn polyline's)
  const mPerUnit = await page.evaluate(num => {
    const n = window.tw.renderer.net, L = n.lineTable.subarray(num * 4, num * 4 + 4);
    let len = 0;
    for (let k = L[0]; k < L[0] + L[1] - 1; k++) len += Math.hypot(n.samples[k * 2 + 2] - n.samples[k * 2], n.samples[k * 2 + 3] - n.samples[k * 2 + 1]);
    return L[2] / len;
  }, line.num);
  const compare = (a, b) => {
    // each train before: the nearest train after the edit, and how far that is (a train kept in
    // place is 0 m away; a moved train is drawn with a new departure but the same place)
    const near = a.list.map(p => Math.min(...b.list.map(q => Math.hypot(p.x - q.x, p.y - q.y))) * mPerUnit);
    return { before: a.list.length, after: b.list.length, keptInPlace: near.filter(d => d < 1).length, worstM: +Math.max(...near).toFixed(2) };
  };
  const counts = await page.evaluate(num => window.tw.world.value.save.lines.find(l => l.num === num), line.num);
  out.schedule = { tph: counts.tph, trains: counts.trains };
  const shot = name => page.screenshot({ path: join(ART, `T-206-${name}.png`) });
  const b0 = await trains();
  await shot('1-before');
  // add trains: the line converted to a whole-line fleet with half as many again at midday
  const n0 = await page.evaluate(id => window.tw.world.value.lineStats[id]?.trainsNeeded ?? null, line.id);
  out.fleetBefore = n0;
  const add = n0 ? [n0[0], Math.round(n0[1] * 1.5) + 1, n0[2]] : [20, 15, 6];
  const edit = async trainsArr => {
    const v = await page.evaluate(() => window.tw.world.value.version);
    const ans = await page.evaluate(([num, tr]) => window.tw.client.edit({ op: 'setTrainCount', line: num, trains: tr }), [line.num, trainsArr]);
    await page.waitForFunction(v => window.tw.world.value.version > v, v, { timeout: 30000 });
    await page.waitForTimeout(1500);
    return ans;
  };
  out.addAnswer = await edit(add);
  const b1 = await trains();
  await shot('2-after-adding');
  out.adding = { fleet: add, ...compare(b0, b1) };
  // run on 2, 6 and 15 minutes: the new trains come in from the ends
  out.addingLater = [];
  for (const [m, name] of [[2, '3-adding-2min'], [6, '4-adding-6min'], [15, '5-adding-15min']]) {
    await page.evaluate(t => { window.tw.clock.set(t); window.tw.overlay.invalidate(); }, at + m * 60);
    await page.waitForTimeout(1200);
    const b = await trains();
    out.addingLater.push({ minutes: m, trains: b.list.length, standing: b.list.filter(x => x.end !== null && x.end > 0 && x.end < 3600).length });
    await shot(name);
  }
  // then back to fewer trains than at first, at 11:40
  const at2 = at + 20 * 60;
  await page.evaluate(t => { window.tw.clock.set(t); window.tw.overlay.invalidate(); }, at2);
  await page.waitForTimeout(1200);
  const c0 = await trains();
  const cut = [add[0], Math.max(2, Math.round((n0?.[1] ?? 15) * 0.6)), add[2]];
  out.cutAnswer = await edit(cut);
  const c1 = await trains();
  await shot('6-after-removing');
  out.removing = { fleet: cut, ...compare(c0, c1) };
  out.removingLater = [];
  for (const m of [5, 15, 40, 70]) {
    await page.evaluate(t => { window.tw.clock.set(t); window.tw.overlay.invalidate(); }, at2 + m * 60);
    await page.waitForTimeout(1200);
    const b = await trains();
    out.removingLater.push({ minutes: m, trains: b.list.length });
  }
  await shot('7-removing-70min');
  out.fleetAfter = await page.evaluate(id => window.tw.world.value.lineStats[id]?.trainsNeeded ?? null, line.id);
  out.errors = errors.filter(e => !/favicon|DevTools/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't206-browser.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
