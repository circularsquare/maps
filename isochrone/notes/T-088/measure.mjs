// Pure level comparison: same start, same destinations, each level solved on its own from the
// start only (L10 the world; L3 and L1 a box round the start), walking, hazard on and off.
//   node measure.mjs [walk|canoe] > results/pure_<mode>.json
import { writeFileSync, mkdirSync } from 'node:fs';
import * as H from './lib.mjs';
import { STARTS, DISTS, BEARINGS } from './regions.mjs';

const mode = process.argv[2] || 'walk';
const canoe = mode === 'canoe';
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
mkdirSync(H.SCR + 'results', { recursive: true });
const out = [];
for (const [name, kind, lat, lng] of STARTS) {
    if (only && !only.includes(name)) continue;
    const t0 = performance.now();
    const half = 3.3, box = H.boxAround(lat, lng, half, half / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north), w3 = H.win(3, box.west, box.south, box.east, box.north);
    const [b1, b3] = await Promise.all([H.loadBox(w1), H.loadBox(w3)]);
    // our upscale against the built L3 tiles
    const u3 = H.upscale(b1.grid, w1.cols, w1.rows, 3);
    let same = 0, land = 0, rel = 0;
    for (let i = 0; i < u3.length; i++) if (b3.grid[i] || u3[i]) { land++; if (u3[i] === b3.grid[i]) same++; else if (u3[i] && b3.grid[i]) rel += Math.abs(u3[i] - b3.grid[i]) / b3.grid[i]; }
    const L1 = H.level(w1, b1.grid, b1.hz), L3 = H.level(w3, b3.grid, b3.hz);
    const L10 = await H.world10(canoe);
    const r = H.rng(12345 + Math.round(lat * 100));
    const dests = [];
    for (const D of DISTS) for (let j = 0; j < BEARINGS; j++) {
        const [la, lo] = H.offset(lat, lng, D * (0.85 + 0.3 * r()), 360 * (j + r()) / BEARINGS);
        if (la > box.north - 0.3 || la < box.south + 0.3 || lo < box.west + 0.3 || lo > box.east - 0.3) continue;
        const c = Math.floor((lo - w1.west) / w1.cell), rr = Math.floor((w1.north - la) / w1.cell);
        if (!b1.grid[rr * w1.cols + c]) continue;   // destinations on L1 land
        dests.push({ D, lat: la, lng: lo, km: 0 });
    }
    for (const d of dests) {
        const dy = (d.lat - lat) * H.KM, dx = (d.lng - lng) * H.KM * Math.cos((lat + d.lat) / 2 * Math.PI / 180);
        d.km = Math.hypot(dx, dy);
    }
    for (const hz of process.env.HAZ === 'on' ? [true] : [true, false]) {
        H.cfg.hazard = hz;
        const tag = hz ? 'on' : 'off';
        for (const [lv, L] of [[10, L10], [3, L3], [1, L1]]) {
            await H.solveFrom(L, lat, lng, canoe);
            for (const d of dests) { const x = H.timeAt(L, d.lat, d.lng); d['t' + lv + tag] = x.t; d['n' + lv + tag] = x.near; }
        }
    }
    out.push({ name, kind, lat, lng, upscaleSame: same / land, upscaleRel: rel / land, dests });
    console.error(`${name}: ${dests.length} dests, upscale same ${(100 * same / land).toFixed(1)}%, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
    L1.cn = null; L3.cn = null;
}
writeFileSync(H.SCR + `results/pure_${mode}${process.env.TAG ? '_' + process.env.TAG : ''}${only ? '_' + only.join('_') : ''}.json`, JSON.stringify(out));
process.exit(0);
