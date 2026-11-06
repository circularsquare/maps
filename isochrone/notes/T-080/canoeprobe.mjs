// Why the canoe is faster at L10 in river country: hours carried and paddled at L10, L3, L1.
import { pathToFileURL } from 'node:url';
import * as H from './lib.mjs';
const { trace } = await import(pathToFileURL(H.ROOT + 'js/route.js').href);
H.cfg.hazard = false;
for (const [name, lat, lng] of [['Coari', -4.08, -63.14], ['Khulna', 22.82, 89.55], ['Kisangani', 0.52, 25.19]]) {
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const L = { 10: await H.world10(true) };
    for (const f of [3, 1]) { const w = H.win(f, box.west, box.south, box.east, box.north); const b = await H.loadBox(w); L[f] = H.level(w, b.grid, null); }
    for (const f of [10, 3, 1]) await H.solveFrom(L[f], lat, lng, true);
    for (const [km, bear] of [[160, 0], [160, 90], [160, 180], [160, 270]]) {
        const [la, lo] = H.offset(lat, lng, km, bear);
        const row = [];
        for (const f of [10, 3, 1]) {
            const r = trace([L[f]], true, lo, la, { lng, lat });
            row.push(r ? `L${f} ${r.t.toFixed(0)} h (carry ${r.parts[1].toFixed(0)}, paddle ${r.parts[2].toFixed(0)})` : `L${f} -`);
        }
        console.log(name, km, bear, row.join(' | '));
    }
    for (const f of [3, 1]) L[f].cn = null;
}
process.exit(0);
