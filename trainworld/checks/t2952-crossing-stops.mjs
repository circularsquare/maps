// T-2952: where two bus lines cross, both stop at one shared place (a stop of each within demand's 100 m
// merge of each other, or one station both stop at) or neither stops there. On the real save, with four
// pairs of bus lines added that cross beside a station only one of each pair reaches: every crossing of
// two constructed Auto bus lines, how many share a place, the misses, and the weekday's bus riders and
// transfers; then pictures of a crossing, light and dark.
// From the maps root; it starts its own Vite on the port given (6450 by default) and stops it:
//   node trainworld/checks/t2952-crossing-stops.mjs [port] [tag]
// Tag "before" writes artifacts/trainworld/t2952/before.json; any other tag reads it to picture the same
// crossing. About 5 minutes. Pictures in artifacts/trainworld/t2952-<tag>-*.png.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome, startVite } from './lib.mjs';

const args = process.argv.slice(2);
const PORT = Number(args.find((a) => /^\d+$/.test(a)) ?? 6450);
const TAG = args.find((a) => !/^\d+$/.test(a)) ?? 'after';
const OUT = join(ARTIFACTS, 't2952');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const vite = await startVite(PORT);
const chrome = await launchChrome({ prefix: 'tw-t2952-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t2952-${TAG}-${name}.png`) });

/** every crossing of two constructed bus lines placed by themselves, and whether they share a place there */
const crossings = () => page.evaluate(() => {
  const w = window.tw.world.value;
  const lines = w.save.lines.map((l) => ({ l, b: w.lineStats[l.id]?.bus })).filter((x) => x.b && x.b.built && !x.b.hand && !x.b.spacing && !x.b.ferry);
  const m = (a, b) => Math.hypot((a[0] - b[0]) * 111320 * Math.cos((a[1] * Math.PI) / 180), (a[1] - b[1]) * 111320);
  const boxes = (r) => { const out = []; for (let i = 0; i + 3 < r.length; i += 32) { let b = [Infinity, Infinity, -Infinity, -Infinity]; for (let j = i; j < Math.min(i + 34, r.length); j += 2) b = [Math.min(b[0], r[j]), Math.min(b[1], r[j + 1]), Math.max(b[2], r[j]), Math.max(b[3], r[j + 1])]; out.push([i, b]); } return out; };
  const meet = (a, b) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
  const places = (b) => b.stops.map((s) => ({ kind: s.kind, station: s.station, on: [s.lng, s.lat], at: s.at ?? [s.lng, s.lat] }));
  const res = [];
  for (let i = 0; i < lines.length; i++) for (let j = i + 1; j < lines.length; j++) {
    const A = lines[i], B = lines[j], ra = A.b.route, rb = B.b.route, la = A.b.routeLL;
    for (const [ia, ba] of boxes(ra)) for (const [ib, bb] of boxes(rb)) {
      if (!meet(ba, bb)) continue;
      for (let p = ia; p + 3 < Math.min(ia + 34, ra.length); p += 2) for (let q = ib; q + 3 < Math.min(ib + 34, rb.length); q += 2) {
        const [px, py, d1x, d1y] = [ra[p], ra[p + 1], ra[p + 2] - ra[p], ra[p + 3] - ra[p + 1]];
        const [rx, ry, d2x, d2y] = [rb[q], rb[q + 1], rb[q + 2] - rb[q], rb[q + 3] - rb[q + 1]];
        const den = d1x * d2y - d1y * d2x;
        if (Math.abs(den) < 1e-12) continue;
        const t = ((rx - px) * d2y - (ry - py) * d2x) / den, u = ((rx - px) * d1y - (ry - py) * d1x) / den;
        if (t < 0 || t > 1 || u < 0 || u > 1) continue;
        const P = [la[p] + t * (la[p + 2] - la[p]), la[p + 1] + t * (la[p + 3] - la[p + 1])];
        if (res.some((c) => c.a === A.l.id && c.b === B.l.id && m(c.at, P) < 1)) continue;
        const near = (b) => places(b).filter((s) => m(s.on, P) <= 400);
        const na = near(A.b), nb = near(B.b);
        const shared = na.some((x) => nb.some((y) => (x.station >= 0 && x.station === y.station) || m(x.at, y.at) <= 100));
        const desc = (n) => n.map((s) => `${s.kind}@${m(s.on, P).toFixed(0)}${s.station >= 0 ? `(st ${m(s.at, P).toFixed(0)})` : ''}`).join(' ');
        res.push({ a: A.l.id, b: B.l.id, an: A.l.name, bn: B.l.name, at: P, shared, near: `${desc(na)} | ${desc(nb)}` });
      }
    }
  }
  return { lines: lines.length, res };
});

/** the weekday's bus riders, all riders, rail trips and transfers */
const figures = () => page.evaluate(() => {
  const w = window.tw.world.value, v = window.twDemand.days()[0].view;
  let bus = 0, all = 0, xfer = 0, xferBus = 0;
  for (const l of w.save.lines) {
    const r = v.lines.get(l.id)?.ridersPerDay ?? 0;
    all += r;
    if (w.lineStats[l.id]?.bus) bus += r;
  }
  for (const [id, s] of v.stations) { xfer += s.transfers ?? 0; if (Number(id) >= 2 ** 30) xferBus += s.transfers ?? 0; }
  const lines = w.save.lines.filter((l) => /^Check /.test(l.name)).map((l) => `${l.name} ${Math.round(v.lines.get(l.id)?.ridersPerDay ?? 0)} riders, stops ${w.lineStats[l.id].bus.stops.map((s) => s.kind).join('')}`);
  return { version: v.version, trips: Math.round(v.tripsPerDay), rail: Math.round(v.railTripsPerDay), boardings: Math.round(all), bus: Math.round(bus), transfers: Math.round(xfer), transfersAtBusStops: Math.round(xferBus), lines };
});

try {
  await page.goto(`${vite.url}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 900000, lines: 100 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const v0 = await page.evaluate(() => window.twDemand.days()[0].view.version);
  // the real save has no buses: at four stations of different rail lines spread over the city, line A runs
  // 150 m beside the station along its rail line and B crosses A 150 m on, square to the way from the
  // station, so B passes it 212 m off, out of its reach (T-2949's case)
  const made = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value;
    const m = (a, b) => Math.hypot((a.lng - b.lng) * 111320 * Math.cos((a.lat * Math.PI) / 180), (a.lat - b.lat) * 111320);
    const byId = new Map(w.save.stations.map((s) => [s.id, s]));
    const picked = [];
    for (const l of w.save.lines) {
      if (w.lineStats[l.id]?.bus || l.stops.length < 3 || picked.length >= 4) continue;
      for (let i = 1; i < l.stops.length; i++) {
        const a = byId.get(String(l.stops[i - 1])), b = byId.get(String(l.stops[i]));
        if (a && b && a.built && b.built && m(a, b) > 800 && m(a, b) < 2500 && picked.every((p) => m(p[0], a) > 5000)) { picked.push([a, b]); break; }
      }
    }
    const route = (pts) => pts.flatMap((p) => [p[0], p[1], 1]);
    const lines = [], pairs = [];
    for (const [i, [a, b]] of picked.entries()) {
      const d = Math.hypot(b.x - a.x, b.y - a.y), k = d / m(a, b);
      const u = [(b.x - a.x) / d, (b.y - a.y) / d], n = [-u[1], u[0]];
      const at = (s, o) => [a.x + (u[0] * s + n[0] * o) * k, a.y + (u[1] * s + n[1] * o) * k];
      const c = at(150, 150), dd = [(u[0] - n[0]) / Math.SQRT2, (u[1] - n[1]) / Math.SQRT2];
      const A = await tw.client.edit({ op: 'addBusLine', route: route([at(-1500, 150), at(2500, 150)]), name: `Check A${i + 1}`, colour: '#c0392b' });
      const B = await tw.client.edit({ op: 'addBusLine', route: route([[c[0] - dd[0] * 2000 * k, c[1] - dd[1] * 2000 * k], [c[0] + dd[0] * 2000 * k, c[1] + dd[1] * 2000 * k]]), name: `Check B${i + 1}`, colour: '#2d7dd2' });
      if (!A.ok || !B.ok) return { ok: false, issues: [...(A.issues ?? []), ...(B.issues ?? [])] };
      lines.push(A.line, B.line);
      pairs.push({ station: a.id, name: a.name });
    }
    const con = await tw.client.edit({ op: 'construct', edges: [], stations: [], buses: lines });
    return { ok: con.ok && lines.length === 8, lines, pairs, issues: con.issues ?? [] };
  });
  check(made.ok, 'four pairs of crossing bus lines', JSON.stringify(made).slice(0, 400));
  await demandFinal(page, { timeout: 900000, lines: 100, after: v0 });
  await wait(15000);
  await demandFinal(page, { timeout: 900000, lines: 100, after: v0 });
  const f = await figures();
  const c = await crossings();
  const miss = c.res.filter((x) => !x.shared);
  console.log(`${TAG}: ${c.lines} auto bus lines, ${c.res.length} crossings, ${c.res.length - miss.length} shared, ${miss.length} missed`);
  for (const x of miss.slice(0, 20)) console.log(`  miss ${x.an} x ${x.bn} at ${x.at.map((v) => v.toFixed(5)).join(',')}: ${x.near}`);
  console.log(`${TAG} figures ${JSON.stringify(f)}`);
  await writeFile(join(OUT, `${TAG}.json`), JSON.stringify({ figures: f, lines: c.lines, crossings: c.res }, null, 1));
  if (TAG !== 'before') check(miss.length === 0, 'every crossing of two Auto bus lines shares a place', `${miss.length} missed`);

  // the picture: the first crossing missed before, else any crossing
  let spot = c.res[0];
  try { const before = JSON.parse(await readFile(join(OUT, 'before.json'), 'utf8')); spot = before.crossings.find((x) => !x.shared) ?? spot; } catch {}
  if (spot) {
    console.log(`picture at ${spot.an} x ${spot.bn}, ${spot.at.map((v) => v.toFixed(5)).join(',')}`);
    await page.evaluate((at) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: at, zoom: 16.5 }); }, spot.at);
    await page.getByText('Calculating demand').waitFor({ state: 'hidden', timeout: 240000 }).catch(() => console.log('the demand dialog stayed'));
    await page.mouse.move(1390, 450);
    await wait(3000);
    await shot('light');
    await page.evaluate(() => { window.tw.shade.value = 'dark'; });
    await wait(2500);
    await shot('dark');
    await page.evaluate(() => { window.tw.shade.value = 'light'; });
  }
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
  vite.child.kill();
}
console.log(fail ? `T-2952 FAILED: ${fail}` : 'T-2952 PASSED');
process.exit(fail ? 1 : 0);
