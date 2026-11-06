// Causes: the same pairs as measure.mjs on variants of the L1 box, each upscaled to L3' and L10'
// by the build rule and solved from the start only (hazard off).
//   real   the L1 grid
//   uni    every land cell 3 km/h, water kept     -> water topology + geometry
//   flat   every cell 3 km/h, no water            -> geometry and cell snapping only
//   fill   water filled at the box's median speed -> speed contrast only
//   fcap   fill, speeds capped at the median      -> slow patches (ridges, swamp) only
//   ffloor fill, speeds raised to the median      -> fast corridors (roads, valleys) only
//   node variants.mjs > results/variants.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';

H.cfg.hazard = false;
const pure = JSON.parse(readFileSync(H.SCR + 'results/pure_walk.json', 'utf8'));
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const out = [];
for (const reg of pure) {
    if (only && !only.includes(reg.name)) continue;
    const { lat, lng } = reg;
    const t0 = performance.now();
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north);
    const w3 = H.win(3, box.west, box.south, box.east, box.north), w10 = H.win(10, box.west, box.south, box.east, box.north);
    const g = (await H.loadBox(w1)).grid;
    const landv = []; for (let i = 0; i < g.length; i += 7) if (g[i]) landv.push(g[i]);
    const med = H.quant(landv, 0.5);
    const V = {
        real: g,
        uni: g.map(v => v ? 120 : 0),
        flat: g.map(() => 120),
        fill: g.map(v => v || med),
        fcap: g.map(v => Math.min(v || med, med)),
        ffloor: g.map(v => Math.max(v || med, med)),
    };
    const dests = reg.dests.map(d => ({ D: d.D, lat: d.lat, lng: d.lng }));
    for (const [k, gv] of Object.entries(V)) {
        for (const [lv, w, grid] of [[1, w1, gv], [3, w3, H.upscale(gv, w1.cols, w1.rows, 3)], [10, w10, H.upscale(gv, w1.cols, w1.rows, 10)]]) {
            const L = H.level(w, grid, null);
            await H.solveFrom(L, lat, lng, false);
            for (const d of dests) d[`t${lv}${k}`] = H.timeAt(L, d.lat, d.lng).t;
        }
    }
    out.push({ name: reg.name, kind: reg.kind, lat, lng, median: med / 40, dests });
    console.error(`${reg.name}: median ${(med / 40).toFixed(2)} km/h, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + 'results/variants.json', JSON.stringify(out));
process.exit(0);
