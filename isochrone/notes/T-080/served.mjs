// What the page shows: for each destination, a view centred on it at zoom 4 (L10), 6 (an L3
// window seeded from L10) and 8 (L3 then L1 windows, as js/session.js refineWindows), on a
// 1400 x 900 screen; the time read from the finest level holding the point. Walking, hazard on.
//   ONLY=a,b node served.mjs tag > results/served_<tag>.json
import { writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { STARTS } from './regions.mjs';

const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const tag = process.argv[2] || 'all';
H.cfg.hazard = true;
const { windowFor, solveWindow, MARGIN, MARGIN_L3_UNDER_L1, Z1 } = H.Wn;

function viewAt(lat, lng, z) {
    const scale = 256 * 2 ** z;
    const hw = 700 / scale * 360;
    const y = 0.5 - Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) / (2 * Math.PI);
    const latOf = (yy) => (2 * Math.atan(Math.exp((0.5 - yy) * 2 * Math.PI)) - Math.PI / 2) * 180 / Math.PI;
    return { west: lng - hw, east: lng + hw, north: latOf(y - 450 / scale), south: latOf(y + 450 / scale), zoom: z };
}
function finestAt(levels, lat, lng) {
    for (const L of levels) {
        if (!L) continue;
        if (lng < L.west || lng >= L.east || lat > L.north || lat <= L.south) continue;
        return { ...H.timeAt(L, lat, lng), f: L.win ? L.win.f : 10 };
    }
    return { t: NaN, f: 0 };
}
const out = [];
for (const [name, kind, lat, lng] of STARTS) {
    if (only && !only.includes(name)) continue;
    const t0 = performance.now();
    const L10 = await H.world10(false);
    await H.solveFrom(L10, lat, lng, false);
    const origin = { lng, lat };
    const dests = [];
    for (const D of [20, 80, 300]) for (let j = 0; j < 4; j++) {
        const [la, lo] = H.offset(lat, lng, D, 90 * j + 45 * (D === 80));
        dests.push({ D, lat: la, lng: lo });
    }
    for (const d of dests) {
        d.z4 = finestAt([L10], d.lat, d.lng);
        for (const z of [6, 8]) {
            const v = viewAt(d.lat, d.lng, z);
            const w3 = windowFor(3, v, z >= Z1 ? MARGIN_L3_UNDER_L1 : MARGIN, false);
            const L3 = await solveWindow(w3, [L10], origin, () => true, null);
            let L1 = null;
            if (z >= Z1) L1 = await solveWindow(windowFor(1, v, MARGIN, false), [L3, L10], origin, () => true, null);
            d['z' + z] = finestAt([L1, L3, L10], d.lat, d.lng);
            d['in' + z] = [L3.win, L1 && L1.win].map(w => w && lng >= w.west && lng < w.east && lat <= w.north && lat > w.south);
        }
    }
    out.push({ name, kind, lat, lng, dests });
    console.error(`${name}: ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + `results/served_${tag}.json`, JSON.stringify(out));
process.exit(0);
