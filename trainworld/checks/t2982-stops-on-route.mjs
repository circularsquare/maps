// T-2982: a bus stop near a train station sits on the bus line's route, never on the station's centre,
// with a light dashed line from the stop to the station; Manual stop editing offers no station marks and a
// click on a station does nothing, while a click on the route beside it adds the station's stop there.
// On the real save, a bus line drawn about 90 m beside two stations of a rail line.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t2982-stops-on-route.mjs [dev url] [tag]
// About 2 minutes. Pictures in artifacts/trainworld/t2982-<tag>-*.png (tag "after" by default).
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6440/';
const TAG = args.find((a) => !/^https?:/.test(a)) ?? 'after';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t2982-${TAG}-${name}.png`) });

const chrome = await launchChrome({ prefix: 'tw-t2982-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);

/** the line's stops, each with its station's place, and how far each is off the route (plane units, and ground m) */
const state = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), b = l && w.lineStats[l.id].bus;
  if (!b) return null;
  const r = b.route;
  const off = (x, y) => {
    let best = Infinity;
    for (let i = 0; i + 3 < r.length; i += 2) {
      const ax = r[i], ay = r[i + 1], dx = r[i + 2] - ax, dy = r[i + 3] - ay, l2 = dx * dx + dy * dy;
      const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      best = Math.min(best, Math.hypot(ax + t * dx - x, ay + t * dy - y));
    }
    return best;
  };
  const m = (a, b) => Math.hypot((a.lng - b.lng) * 111320 * Math.cos((a.lat * Math.PI) / 180), (a.lat - b.lat) * 111320);
  return {
    id: l.id, hand: b.hand, street: b.street.length / 2,
    stops: b.stops.map((s) => {
      const n = s.station >= 0 ? w.save.stations.find((x) => x.id === String(s.station)) : null;
      return { kind: s.kind, station: s.station, lng: s.lng, lat: s.lat, x: s.x, y: s.y, off: off(s.x, s.y), toStation: n ? m(s, n) : null, sLng: n?.lng, sLat: n?.lat };
    }),
  };
}, num);
const until = async (num, test) => { for (let k = 0; k < 150; k++) { const s = await state(num); if (s && test(s)) return s; await wait(200); } return state(num); };
const point = (lng, lat) => page.evaluate(([lng, lat]) => { const p = window.tw.map.project([lng, lat]); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, [lng, lat]);
const click = async (lng, lat) => { const [x, y] = await point(lng, lat); await page.mouse.move(x, y); await wait(200); await page.mouse.click(x, y); };
const mode = (name) => page.locator('.h .group button').filter({ hasText: new RegExp(`^${name}$`) });

try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(3000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));

  // two neighbouring built stations of a rail line, 0.8-2.5 km apart; the bus runs 90 m to one side of them
  const made = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value;
    const m = (a, b) => Math.hypot((a.lng - b.lng) * 111320 * Math.cos((a.lat * Math.PI) / 180), (a.lat - b.lat) * 111320);
    let pair = null;
    const byId = new Map(w.save.stations.map((s) => [s.id, s]));
    for (const l of w.save.lines) {
      if (w.lineStats[l.id]?.bus || l.stops.length < 3) continue;
      for (let i = 1; i < l.stops.length && !pair; i++) {
        const a = byId.get(String(l.stops[i - 1])), b = byId.get(String(l.stops[i]));
        if (a && b && a.built && b.built && m(a, b) > 800 && m(a, b) < 2500) pair = [a, b];
      }
      if (pair) break;
    }
    if (!pair) return { ok: false, why: 'no station pair' };
    const [a, b] = pair, d = Math.hypot(b.x - a.x, b.y - a.y), k = d / m(a, b);
    const ux = (b.x - a.x) / d, uy = (b.y - a.y) / d, nx = -uy, ny = ux, off = 90 * k, ext = 600 * k;
    const route = [[a.x - ux * ext + nx * off, a.y - uy * ext + ny * off], [b.x + ux * ext + nx * off, b.y + uy * ext + ny * off]].flatMap((p) => [p[0], p[1], 1]);
    const add = await tw.client.edit({ op: 'addBusLine', route, name: 'Check bus', colour: '#c0392b' });
    const con = await tw.client.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    return { ok: add.ok && con.ok, line: add.line, a: { id: a.id, lng: a.lng, lat: a.lat }, b: { id: b.id, lng: b.lng, lat: b.lat }, issues: [...(add.issues ?? []), ...(con.issues ?? [])] };
  });
  check(made.ok, 'a bus line beside two stations', JSON.stringify(made).slice(0, 300));
  const num = made.line;
  let s = await until(num, (x) => x.stops.length >= 4);
  const at = s.stops.filter((x) => x.kind === 1);
  console.log(`stops ${s.stops.map((x) => x.kind).join('')}, station stops ${at.map((x) => `${x.station}: ${x.toStation?.toFixed(0)} m off its station`).join(', ')}`);
  check(at.length >= 2, 'both stations are stops', `${at.length}`);
  check(s.stops.every((x) => x.off < 0.5), 'every stop is on the route', s.stops.map((x) => x.off.toFixed(2)).join(' '));
  check(at.every((x) => x.toStation > 60 && x.toStation < 120), 'a station stop sits about 90 m from its station, not on it', at.map((x) => x.toStation.toFixed(0)).join(' '));

  // the dashed transfer lines
  const xfer = await page.evaluate(() => window.tw.renderer.buses?.transfers ?? -1);
  check(xfer >= at.length, 'a dashed transfer line for each station stop', `${xfer}`);

  // pictures: the line selected, zoom 16 at the first station, light and dark
  // a station stop between the ends (a route end within reach of a station stops there too)
  const a = at.find((x) => String(x.station) === made.a.id && x !== s.stops[0]) ?? at.find((x) => x !== s.stops[0] && x !== s.stops[s.stops.length - 1]) ?? at[0];
  const centre = [(a.lng + a.sLng) / 2, (a.lat + a.sLat) / 2];
  await page.evaluate(([id, c]) => { window.tw.tools.exit(); window.tw.selection.value = { kind: 'line', line: id }; window.tw.map.jumpTo({ center: c, zoom: 16.5 }); window.tw.map.panBy([200, -150], { duration: 0 }); }, [s.id, centre]);
  // the demand dialog sits in the middle of the map: the pictures wait for it
  await page.getByText('Calculating demand').waitFor({ state: 'hidden', timeout: 240000 }).catch(() => console.log('the demand dialog stayed'));
  await page.mouse.move(1390, 450);
  await wait(3000);
  await shot('light');
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await wait(2500);
  await shot('dark');
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await wait(1500);
  // unselected, a zoom out: the quiet look
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ zoom: 15 }); });
  await wait(2500);
  await shot('light-z15');

  // Manual: no station marks, a click on a station's centre adds nothing, one on the route beside it adds its stop there
  await page.evaluate(([id, c]) => { window.tw.selection.value = { kind: 'line', line: id }; window.tw.map.jumpTo({ center: c, zoom: 16 }); window.tw.map.panBy([200, -150], { duration: 0 }); }, [s.id, centre]);
  await wait(1500);
  await mode('Manual').click();
  s = await until(num, (x) => x.hand && x.stops.length === 2);
  check(s.hand && s.stops.length === 2, 'Manual leaves the two ends', `${s.stops.map((x) => x.kind)}`);
  await wait(1000);
  const marks = await page.evaluate(() => { const b = window.tw.tools.busStops; return b?.active ? b.items.filter((m) => !m.on).length : -1; });
  check(marks === 0, 'Manual draws no station marks', `${marks}`);
  await page.mouse.move(...(await point(a.sLng, a.sLat)));
  await wait(400);
  const cursor = await page.evaluate(() => window.tw.map.getCanvas().style.cursor);
  check(cursor !== 'pointer', 'a station off the route is not clickable', cursor);
  await shot('manual');
  await click(a.sLng, a.sLat);
  await wait(1500);
  s = await state(num);
  check(s.stops.length === 2, 'a click on the station adds no stop', `${s.stops.map((x) => x.kind)}`);
  // the route beside the station: the stop's place before Manual
  const [hx, hy] = await point(a.lng, a.lat);
  await page.mouse.move(hx + 6, hy + 2);
  await wait(400);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.tool-tag.stop-tag')].filter((t) => t.style.display !== 'none').map((t) => t.innerText).join(' | '));
  await shot('manual-hover');
  check(/add a stop/i.test(tag) && /\n/.test(tag), 'the route beside the station offers its stop, the tag naming it', JSON.stringify(tag));
  await click(a.lng, a.lat);
  s = await until(num, (x) => x.stops.length === 3);
  const added = s.stops[1];
  check(s.stops.length === 3 && added.kind === 1 && added.station === a.station, 'a click on the route beside the station stops at the station', `${s.stops.map((x) => x.kind)} ${added?.station} vs ${a.station}`);
  check(added && added.off < 0.5 && added.toStation > 60, 'and the stop is on the route', `${added?.off.toFixed(2)}, ${added?.toStation?.toFixed(0)} m from the station`);
  await page.mouse.move(1390, 450);
  await wait(1200);
  await shot('manual-placed');
  await page.evaluate(() => window.tw.tools.exit());
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2982 FAILED: ${fail}` : 'T-2982 PASSED');
process.exit(fail ? 1 : 0);
