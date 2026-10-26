// T-1416: station upgrades drawn on the box. On the real save, one through station with every
// combination of terminal, park and ride and waiting room, at zoom 14.5 and 16, light and dark;
// Grand Central with all three on each of its stations; the hover tag over each upgrade. Screenshots
// in artifacts/trainworld/t1416/.
//   node trainworld/checks/t1416-annexes.mjs [url]     (from the maps root; a running dev server)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6185/';
const OUT = join(ARTIFACTS, 't1416');
const UP = { terminal: 1, park: 2, room: 4 };
const COMBOS = [['plain', []], ['room', ['room']], ['park', ['park']], ['terminal', ['terminal']], ['room-park', ['room', 'park']], ['room-terminal', ['room', 'terminal']], ['park-terminal', ['park', 'terminal']], ['all', ['room', 'park', 'terminal']]];
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t1416-', size: [1400, 900] });
const errors = [];
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
  const set = (nums, on) => page.evaluate(async ([nums, on]) => {
    for (const [bit, yes] of on) await window.tw.client.edit({ op: 'stationUpgrade', nodes: nums, bit, on: yes });
  }, [nums, on]);
  const combo = (names) => Object.entries(UP).map(([k, bit]) => [bit, names.includes(k)]);
  // a station through which track runs on, part way along a line, away from others
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
  const shot = async (name, zoom, center, size = [420, 300]) => {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z, bearing: 0, pitch: 0 }), [center, zoom]);
    await page.waitForTimeout(1200);
    const [cx, cy] = await page.evaluate((c) => { const p = window.tw.map.project(c), r = window.tw.map.getCanvas().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, center);
    await page.screenshot({ path: join(OUT, `${name}.png`), clip: { x: Math.max(0, cx - size[0] / 2), y: Math.max(0, cy - size[1] / 2), width: size[0], height: size[1] } });
  };
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.waitForTimeout(1500);
    // ONLY=all,terminal: those combinations alone
    for (const [name, names] of COMBOS.filter(([n]) => !process.env.ONLY || process.env.ONLY.split(',').includes(n))) {
      await set([st.num], combo(names));
      await page.waitForTimeout(800);
      for (const z of [14.5, 16]) await shot(`box-${name}-z${z}-${shade}`, z, [st.lng, st.lat]);
    }
  }
  // the hover tag over each upgrade, at zoom 16 with all three
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 16, bearing: 0, pitch: 0 }), [[st.lng, st.lat]]);
  await page.waitForTimeout(1500);
  const spots = await page.evaluate((s) => {
    const r = window.tw.renderer, ov = window.tw.overlay, i = r.net.stationIds.indexOf(s.id), out = {};
    const rect = window.tw.map.getCanvas().getBoundingClientRect(), c = window.tw.map.project([s.lng, s.lat]);
    for (let y = c.y - 150; y < c.y + 150; y += 2) for (let x = c.x - 150; x < c.x + 150; x += 2) {
      const b = r.upgradeAt(i, x, y, (a, q) => ov.toScreen(a, q));
      if (b) (out[b] ??= []).push([Math.round(x + rect.left), Math.round(y + rect.top)]);
    }
    // the middle one of each, away from the edges (a terminal's: of its first end)
    for (const b in out) out[b] = out[b][out[b].length >> (b === '1' ? 2 : 1)];
    return out;
  }, st);
  for (const [bit, name] of [[4, 'room'], [2, 'park'], [1, 'terminal']]) {
    const p = spots[bit];
    console.log(`hover ${name}: ${p ? p.join(',') : 'not found'}`);
    if (!p) continue;
    await page.mouse.move(p[0], p[1]);
    await page.waitForTimeout(900);
    await page.screenshot({ path: join(OUT, `hover-${name}.png`), clip: { x: Math.max(0, p[0] - 200), y: Math.max(0, p[1] - 150), width: 520, height: 360 } });
  }
  await page.mouse.move(5, 5);
  await set([st.num], combo([]));
  // Grand Central with all three on each of its stations
  const gc = await page.evaluate(() => {
    const ss = window.tw.world.value.save.stations.filter((s) => s.built && /Grand Central/.test(s.name));
    if (!ss.length) return null;
    return { nums: ss.map((s) => s.num), lng: ss.reduce((a, s) => a + s.lng, 0) / ss.length, lat: ss.reduce((a, s) => a + s.lat, 0) / ss.length };
  });
  if (gc) {
    await set(gc.nums, combo(['room', 'park', 'terminal']));
    for (const shade of ['light', 'dark']) {
      await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
      await page.waitForTimeout(1500);
      for (const z of [14.5, 16]) await shot(`grandcentral-all-z${z}-${shade}`, z, [gc.lng, gc.lat], [700, 520]);
    }
    console.log(`Grand Central: ${gc.nums.length} stations`);
  } else console.log('no Grand Central');
  console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
} finally {
  await close();
}
