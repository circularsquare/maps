// T-088's pure level comparison (notes/T-088/measure.mjs) on lib.mjs with directions: same start,
// same destinations, L10 (the world), L3 and L1 (a box round the start) each solved from the start
// alone, hazard on.
//   TILES=pre_t011 ACC=0.5 TAG=x OUT=<folder>/ node measure.mjs [walk|canoe]  -> results/pure_<mode>_<TAG>.json
import { writeFileSync, mkdirSync } from 'node:fs';
import * as H from './lib.mjs';
import { STARTS, DISTS, BEARINGS } from '../T-088/regions.mjs';

const mode = process.argv[2] || 'canoe';
const canoe = mode === 'canoe';
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
mkdirSync(H.SCR + 'results', { recursive: true });
console.error(`walking from ${H.BASE.slice(H.ROOT.length)}, access ${H.cfg.access}, ${mode}`);
const out = [];
H.cfg.hazard = true;
const L10 = await H.world10(canoe);
console.error(`L10 directions ${L10.sd ? 'on' : 'off'}`);
for (const [name, kind, lat, lng] of STARTS) {
    if (only && !only.includes(name)) continue;
    const t0 = performance.now();
    const half = 3.3, box = H.boxAround(lat, lng, half, half / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north), w3 = H.win(3, box.west, box.south, box.east, box.north);
    const [b1, b3] = await Promise.all([H.loadBox(w1), H.loadBox(w3)]);
    const L1 = H.level(w1, b1), L3 = H.level(w3, b3);
    const r = H.rng(12345 + Math.round(lat * 100));
    const dests = [];
    for (const D of DISTS) for (let j = 0; j < BEARINGS; j++) {
        const [la, lo] = H.offset(lat, lng, D * (0.85 + 0.3 * r()), 360 * (j + r()) / BEARINGS);
        if (la > box.north - 0.3 || la < box.south + 0.3 || lo < box.west + 0.3 || lo > box.east - 0.3) continue;
        const c = Math.floor((lo - w1.west) / w1.cell), rr = Math.floor((w1.north - la) / w1.cell);
        if (!b1.grid[rr * w1.cols + c]) continue;   // destinations on L1 land
        dests.push({ D, lat: la, lng: lo });
    }
    for (const [lv, L] of [[10, L10], [3, L3], [1, L1]]) {
        await H.solveFrom(L, lat, lng, canoe);
        for (const d of dests) d['t' + lv + 'on'] = H.timeAt(L, d.lat, d.lng).t;
    }
    out.push({ name, kind, lat, lng, dirs1: !!L1.sd, dirs3: !!L3.sd, dests });
    console.error(`${name}: ${dests.length} dests, dirs L1 ${L1.sd ? 'on' : 'off'} L3 ${L3.sd ? 'on' : 'off'}, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + `results/pure_${mode}${process.env.TAG ? '_' + process.env.TAG : ''}${only ? '_' + only.join('_') : ''}.json`, JSON.stringify(out));
process.exit(0);
