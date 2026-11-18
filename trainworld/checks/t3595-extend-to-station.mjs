// T-3595: Anita's Caracas Line 2 (artifacts/trainworld/t3541-caracas.save) turns back at the loose end just past
// Avenida Arismendi. Played through lib/play.mjs: tram double track drawn 300 m on from that end, a station placed on
// the new track, Construct blueprints; Line 2 must still run. Two cases, each in a fresh game:
//   mid  the station 60% along the new track, bare track beyond it (runs on master since T-3541)
//   end  the station at the new track's far end (her San Martin extension; broken on master, T-3565 to fix)
// Then her own save with the extension built (artifacts/trainworld/t3595-caracas-sanmartin-broken.save): Line 2 runs.
// Skips a case whose save is absent. An acceptance check for T-3565; fails on master today (notes/T-3595.md).
// From the tree root, on a Vite of your own:   node trainworld/checks/t3595-extend-to-station.mjs [dev url]
// About 3 minutes.
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6179/';
const SAVE = join(ARTIFACTS, 't3541-caracas.save');
const BROKEN = join(ARTIFACTS, 't3595-caracas-sanmartin-broken.save');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

/** Line 2 as the game holds it and as its panel shows it */
async function line2(g) {
  await g.page.waitForTimeout(2500);
  const s = await g.page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines.find((l) => l.name === 'Line 2'), st = w.lineStats[l.id];
    return { status: st?.status, rt: Math.round(st?.roundTripS ?? 0), stops: l.stops.length };
  });
  await g.act.openLine('Line 2');
  const p = await g.see.panel();
  await g.act.key('Escape');
  return { ...s, roundTrip: p.rows['Round trip'] ?? null, alerts: p.alerts };
}

async function extend(at) {
  const g = await openGame({ url: BASE, save: SAVE, saveVia: 'post', ui: { shade: 'light', v: 2 }, prefix: 'tw-t3595-' });
  try {
    const geo = await g.page.evaluate(() => {
      const tw = window.tw, w = tw.world.value, ar = w.save.stations.find((s) => s.name === 'Avenida Arismendi');
      const ends = new Map();
      for (const e of w.edges) for (const n of [e.a, e.b]) ends.set(n, [...(ends.get(n) ?? []), e]);
      const loose = [...w.nodes.values()].filter((n) => ends.get(n.id)?.length === 1).sort((a, b) => Math.hypot(a.lng - ar.lng, a.lat - ar.lat) - Math.hypot(b.lng - ar.lng, b.lat - ar.lat))[0];
      const e = ends.get(loose.id)[0], o = w.nodes.get(e.a === loose.id ? e.b : e.a);
      const p = tw.lngLatToPack(loose.lng, loose.lat), q = tw.lngLatToPack(o.lng, o.lat), L = Math.hypot(p.x - q.x, p.y - q.y);
      const far = tw.packToLngLat(p.x + ((p.x - q.x) / L) * 300, p.y + ((p.y - q.y) / L) * 300);
      return { loose: [loose.lng, loose.lat], far: Array.isArray(far) ? far : [far.lng, far.lat], m: Math.round(Math.hypot((loose.lng - ar.lng) * 109000, (loose.lat - ar.lat) * 111000)) };
    });
    const before = await line2(g);
    check(before.status === 'running' && geo.m < 60, `${at}: her Line 2 runs, a loose end ${geo.m} m past Avenida Arismendi`, JSON.stringify(before));
    await g.act.camera(geo.loose, 16);
    await g.act.tool('track', { type: 'tram', tracks: 2, level: 0, snap: { real: false, streets: false }, keepPlacing: false });
    const t = await g.act.drawTrack([geo.loose, geo.far]);
    check(t.ok, `${at}: tram double drawn 300 m on from that end`);
    const id = t.segments.at(-1).edges.at(-1).id;
    const sp = at === 'end' ? geo.far : await g.page.evaluate((id) => {
      const tw = window.tw, pts = tw.renderer.net.edgePts.get(id), k = Math.floor((pts.length / 3) * 0.6) * 3;
      return tw.toLngLat(pts[k], pts[k + 1]);
    }, id);
    await g.act.camera(sp, 16);
    const st = await g.act.placeStation(sp);
    check(st.ok, `${at}: a station placed on the new track`, JSON.stringify(st).slice(0, 160));
    const bp = await line2(g);
    check(bp.status === 'running', `${at}: as blueprint, Line 2 runs`, JSON.stringify(bp));
    await g.act.construct();
    const built = await line2(g);
    check(built.status === 'running' && !built.alerts.some((a) => /is gone/.test(a)), `${at}: constructed, Line 2 still runs`, JSON.stringify(built));
    check(g.pageErrors().length === 0, `${at}: no console errors`, g.pageErrors().slice(0, 3).join(' | '));
  } finally {
    await g.close();
  }
}

try {
  if (!existsSync(SAVE)) console.log('SKIP: no ' + SAVE);
  else for (const at of ['mid', 'end']) await extend(at);
  if (!existsSync(BROKEN)) console.log('SKIP: no ' + BROKEN);
  else {
    const g = await openGame({ url: BASE, save: BROKEN, saveVia: 'post', ui: { shade: 'light', v: 2 }, prefix: 'tw-t3595-' });
    try {
      const s = await line2(g);
      check(s.status === 'running', 'her save extended to San Martin: Line 2 runs', JSON.stringify(s));
    } finally {
      await g.close();
    }
  }
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
}
console.log(fail ? `T-3595 FAILED: ${fail}` : 'T-3595 PASSED');
process.exit(fail ? 1 : 0);
