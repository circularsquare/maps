// T-4075: trains, trams and buses are drawn on their track or road, curves and stations included. Before, a train's place
// between two of its line's samples 50 m apart was the straight between them, up to 13 m off a tram's street corner
// (players: "trams visually drifting off the track"). On Anita's Amsterdam save (trams on street corners, buses, a metro)
// this places every trip in view over ten game minutes where the app places it (`tw.tripAt`, the GPU's twin, its own
// track's side shift left out) and measures how far that is from the line's drawn path (its edges' points; a bus's
// route), and that the GPU draws them where the CPU does. Screenshot: artifacts/trainworld/t4075/amsterdam.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6113):
//   node trainworld/checks/t4075-on-track.mjs [dev url]
// About a minute.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6113/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
const OUT = join(ARTIFACTS, 't4075');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const g = await openGame({ url: BASE, save: SAVE, saveVia: 'post', ui: { shade: 'light', v: 2 }, prefix: 'tw-t4075-' });
try {
  await g.act.camera([4.8975, 52.372], 15);
  await g.page.waitForFunction(() => window.tw.renderer.net?.sampleBend && window.tw.renderer.tripsBy?.size > 0, undefined, { timeout: 60000 });
  const r = await g.page.evaluate(() => {
    const tw = window.tw, rnd = tw.renderer, n = rnd.net, w = tw.world.value;
    const psi0 = 2 * Math.PI * (0.5 - n.anchor.merc[1]), PPU = (2 * Math.PI) / 40075016.68557849;
    // each line table row's drawn path (local units): its edges' points along its path, or a bus's route
    const paths = new Map();
    const busRoute = new Map(w.save.lines.flatMap((l) => { const b = w.lineStats[l.id]?.bus; return b?.routeLL?.length ? [[Number(l.id), b.routeLL]] : []; }));
    const pathOf = (geo, line) => {
      if (paths.has(geo)) return paths.get(geo);
      const pts = [];
      const ll = busRoute.get(line);
      if (ll) for (let i = 0; i < ll.length; i += 2) pts.push(tw.toLocal(ll[i], ll[i + 1], n.anchor));
      else for (let j = n.pathTable[geo * 2]; j < n.pathTable[geo * 2] + n.pathTable[geo * 2 + 1]; j++) {
        const pe = n.pathEntry[j], p = n.edgePts.get(pe >> 1);
        if (!p) continue;
        const m = p.length / 3;
        for (let q = 0; q < m; q++) { const k = pe & 1 ? m - 1 - q : q; pts.push([p[k * 3], p[k * 3 + 1]]); }
      }
      paths.set(geo, pts);
      return pts;
    };
    const off = (pts, x, y) => {
      let e = Infinity;
      for (let i = 0; i + 1 < pts.length; i++) {
        const [x0, y0] = pts[i], dx = pts[i + 1][0] - x0, dy = pts[i + 1][1] - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / l2)) : 0;
        e = Math.min(e, Math.hypot(x0 + t * dx - x, y0 + t * dy - y));
      }
      return e / Math.cosh(psi0 - y * PPU);
    };
    const out = { rail: { n: 0, max: 0, over1: 0 }, bus: { n: 0, max: 0, over1: 0 } };
    const t0 = tw.clock.now();
    for (let t = t0; t < t0 + 600; t += 5) {
      const e = Math.floor(t / 3600) * 3600, src = rnd.tripsBy.get(e);
      if (!src) continue;
      for (let i = 0; i < src.data.length / 2; i++) {
        const at = tw.tripAt(n, src.data[i * 2], src.data[i * 2 + 1] + e, t);
        if (!at) continue;
        const len = n.lineTable[at.geo * 4 + 2], d = at.gr === 0 ? at.s : len - at.s;
        // (a train standing at a line's end may stand past it, along the track carried on straight)
        if (d < 30 || d > len - 30) continue;
        const o = busRoute.has(at.line) ? out.bus : out.rail, v = off(pathOf(at.geo, at.line), at.x, at.y);
        o.n++;
        o.max = Math.max(o.max, v);
        if (v > 1) o.over1++;
      }
    }
    return { ...out, bends: n.bends?.length / 2, bent: n.sampleBend.filter((c) => c > 0).length, samples: n.samples.length / 2 };
  });
  console.log(JSON.stringify(r));
  check(r.rail.n > 50 && r.bus.n > 20, 'trips of trains, trams and buses were placed', `${r.rail.n} rail, ${r.bus.n} bus`);
  check(r.rail.max < 1, 'every train and tram is drawn within 1 m of its track', `worst ${r.rail.max.toFixed(2)} m; 13 m before T-4075`);
  check(r.bus.max < 1, 'every bus is drawn within 1 m of its route', `worst ${r.bus.max.toFixed(2)} m; 12 m before T-4075`);
  const p = await g.page.evaluate(() => window.tw.checkPrecision({ t: window.tw.clock.now(), frames: 20, dt: 13 }));
  check(p.n > 0 && p.maxM < 0.05, 'the GPU draws them where the CPU does', `worst ${p.maxM?.toFixed(3)} m of ${p.n}`);
  await g.act.camera([4.8975, 52.372], 17.5);
  await g.shot(join(OUT, 'amsterdam.png'));
  check(!g.errors.length, 'no console errors', g.errors.slice(0, 3).join(' | '));
} finally {
  await g.close();
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
