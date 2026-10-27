// T-1528: the car park drawn as a lot beside the complex and the complex-wide build, on the real
// save. A lone through station with park and ride and a waiting room, at zoom 15 and 17, light and
// dark; Grand Central's complex given both from the complex view's buttons (every station gets them,
// paid once per station, one lot and one room drawn), then the same zooms; the hover tag over a lot.
// Screenshots in artifacts/trainworld/t1528/.
//   node trainworld/checks/t1528-shots.mjs [url]     (from the maps root; a running dev server)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:8417/';
const OUT = join(ARTIFACTS, 't1528');
const ZOOMS = [15, 17];
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t1528-', size: [1400, 900] });
const errors = [];
let bad = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'} ${msg}`); if (!cond) bad++; };
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { polling: 500, lines: 100 });
  const shot = async (name, zoom, center, size = [520, 360]) => {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z, bearing: 0, pitch: 0 }), [center, zoom]);
    await page.waitForTimeout(1500);
    const [cx, cy] = await page.evaluate((c) => { const p = window.tw.map.project(c), r = window.tw.map.getCanvas().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, center);
    await page.screenshot({ path: join(OUT, `${name}.png`), clip: { x: Math.max(0, cx - size[0] / 2), y: Math.max(0, cy - size[1] / 2), width: size[0], height: size[1] } });
  };
  const lots = () => page.evaluate(() => Array.from(window.tw.renderer.lots.lots));

  // a through station part way along a line, away from others
  const st = await page.evaluate(() => {
    const w = window.tw.world.value, byId = new Map(w.save.stations.map((x) => [x.id, x]));
    const far = (s) => w.save.stations.every((o) => o === s || Math.hypot((o.lng - s.lng) * Math.cos(s.lat * Math.PI / 180), o.lat - s.lat) * 111320 > 500);
    for (const l of w.save.lines) for (const id of l.stops.slice(1, -1)) {
      const s = byId.get(id);
      if (s && s.built && (w.nodes.get(s.num)?.ports ?? 0) >= 2 && far(s)) return { num: s.num, id: s.id, lng: s.lng, lat: s.lat, name: s.name };
    }
    return null;
  });
  if (!st) throw new Error('no lone through station');
  console.log(`station: ${st.name}`);
  await page.evaluate((num) => Promise.all([2, 4].map((bit) => window.tw.client.edit({ op: 'stationUpgrade', nodes: [num], bit, on: true }))), st.num);
  await page.waitForTimeout(1000);
  ok((await lots()).length === 10, 'one lot for the lone station');

  // Grand Central's complex, from the complex view's buttons
  const gc = await page.evaluate(() => {
    const ss = window.tw.world.value.save.stations.filter((s) => s.built && /Grand Central/.test(s.name));
    return ss.length ? { id: ss[0].id, lng: ss.reduce((a, s) => a + s.lng, 0) / ss.length, lat: ss.reduce((a, s) => a + s.lat, 0) / ss.length } : null;
  });
  if (!gc) throw new Error('no Grand Central');
  await page.evaluate((id) => window.tw.select({ kind: 'station', station: id }), gc.id);
  await page.waitForSelector('[aria-label="Complex upgrades"] button[aria-label="Park and ride"]', { timeout: 10000 });
  const members = await page.evaluate((id) => {
    const w = window.tw.world.value, me = w.save.stations.find((s) => s.id === id);
    // the complex as the panel shows it: its "Complex of n stations" line
    const n = Number(/Complex of (\d+)/.exec(document.querySelector('.inspector .sub, .sub')?.textContent ?? '')?.[1] ?? 0);
    return { n, name: me.name };
  }, gc.id);
  console.log(`complex of ${members.n} stations at ${members.name}`);
  const cash0 = await page.evaluate(() => window.tw.moneyState.value?.cash ?? null);
  for (const label of ['Park and ride', 'Waiting room']) {
    await page.click(`[aria-label="Complex upgrades"] button[aria-label="${label}"]`);
    await page.waitForTimeout(800);
  }
  const after = await page.evaluate((id) => {
    const w = window.tw.world.value, sel = new Set(), cx = [...document.querySelectorAll('.complex-stations li')].length;
    const st = w.save.stations.filter((s) => s.built && /Grand Central/.test(s.name));
    return { cash: window.tw.moneyState.value?.cash ?? null, pr: st.filter((s) => (s.upgrades & 2) !== 0).length, wr: st.filter((s) => (s.upgrades & 4) !== 0).length, rows: cx };
  }, gc.id);
  const gcLots = (await lots()).length / 10;
  ok(gcLots === 2, `one lot for the whole complex (lots now ${gcLots}, the lone station's and Grand Central's)`);
  if (cash0 !== null && after.cash !== null) ok(Math.abs((cash0 - after.cash) - members.n * 25) < 0.01, `paid ${(cash0 - after.cash).toFixed(1)}M for ${members.n} stations (25M each)`);
  const rooms = await page.evaluate(() => { const n = window.tw.renderer.net; let k = 0; for (let i = 0; i < n.stationCount; i++) if (n.stations[i * 3 + 2] & 32) k++; return k; });
  ok(rooms === 2, `two waiting rooms drawn, one per complex (${rooms})`);
  // the panel with its buttons on
  const box = await page.locator('[aria-label="Complex upgrades"]').boundingBox();
  if (box) {
    await page.hover('[aria-label="Complex upgrades"] button[aria-label="Park and ride"]');
    await page.waitForTimeout(700);
    await page.screenshot({ path: join(OUT, 'panel-complex.png'), clip: { x: Math.max(0, box.x - 20), y: Math.max(0, box.y - 140), width: 460, height: 330 } });
  }
  await page.evaluate(() => window.tw.select(null));

  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.waitForTimeout(1500);
    for (const z of ZOOMS) {
      await shot(`lone-z${z}-${shade}`, z, [st.lng, st.lat]);
      await shot(`grandcentral-z${z}-${shade}`, z, [gc.lng, gc.lat], [760, 520]);
    }
  }
  // the hover tag over the lone station's lot
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 17, bearing: 0, pitch: 0 }), [[st.lng, st.lat]]);
  await page.waitForTimeout(1500);
  const spot = await page.evaluate(() => {
    const r = window.tw.renderer, ov = window.tw.overlay, l = r.lots.lots, rect = window.tw.map.getCanvas().getBoundingClientRect();
    const [x, y] = ov.toScreen(l[0], l[1]);
    return [Math.round(x + rect.left), Math.round(y + rect.top)];
  });
  await page.mouse.move(spot[0], spot[1]);
  await page.waitForTimeout(900);
  const bit = await page.evaluate(([id, p]) => {
    const r = window.tw.renderer, ov = window.tw.overlay, rect = window.tw.map.getCanvas().getBoundingClientRect();
    return r.upgradeAt(r.net.stationIds.indexOf(id), p[0] - rect.left, p[1] - rect.top, (a, b) => ov.toScreen(a, b));
  }, [st.id, spot]);
  ok(bit === 2, `the hover over the lot names park and ride (bit ${bit})`);
  await page.screenshot({ path: join(OUT, 'hover-lot.png'), clip: { x: Math.max(0, spot[0] - 240), y: Math.max(0, spot[1] - 160), width: 560, height: 360 } });
  // a click on the lot opens its complex
  await page.mouse.click(spot[0], spot[1]);
  await page.waitForTimeout(500);
  const selected = await page.evaluate(() => window.tw.selection.value);
  ok(selected?.kind === 'station' && selected.station === st.id, `a click on the lot selects ${st.name}`);
  // taking it out from the complex view removes it from every station
  await page.evaluate((id) => window.tw.select({ kind: 'station', station: id }), gc.id);
  await page.waitForSelector('[aria-label="Complex upgrades"] button[aria-label="Park and ride"]');
  await page.click('[aria-label="Complex upgrades"] button[aria-label="Park and ride"]');
  await page.waitForTimeout(800);
  const left = await page.evaluate(() => window.tw.world.value.save.stations.filter((s) => /Grand Central/.test(s.name) && (s.upgrades & 2)).length);
  ok(left === 0, `removed from the whole complex (${left} left)`);
  ok(errors.length === 0, errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
} finally {
  await close();
}
if (bad) process.exit(1);
