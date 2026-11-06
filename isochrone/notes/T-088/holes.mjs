// Holes (as notes/T-085.md): canoe starts, hazard on; a hole is a cell with a node, none of its
// nodes reached, and 6 or more of its 8 neighbours reached. Also: river nodes on further slots
// left unreached while a node of their cell is reached (T-082).
//   CANOE=<folder>/ node holes.mjs
import * as H from './lib.mjs';
import { pathToFileURL } from 'node:url';
const { eachCanoeNode } = await H.imp('canoe.js');
H.cfg.hazard = true;
function holes(L) {
    const { W, H: Hh, grid, cn } = L, t = L.res.t;
    const reached = new Uint8Array(W * Hh), has = new Uint8Array(W * Hh);
    let extraOff = 0;
    for (let i = 0; i < W * Hh; i++) {
        let any = false, all = true, n = 0;
        if (grid[i]) { n++; if (t[i] <= 1024) any = true; else all = false; }
        eachCanoeNode(cn, i, (m) => { n++; if (t[m] <= 1024) any = true; else all = false; });
        has[i] = n > 0; reached[i] = any;
        // well inside the limits: the cell's land node reached in under 500 h at under 1% risk
        const inner = grid[i] && t[i] <= 500 && L.res.hz[i] < 0.01;
        if (inner && !all && cn.extraN[i]) eachCanoeNode(cn, i, (m) => { if (m >= W * Hh + cn.E1 && !(t[m] <= 1024)) { extraOff++; if (extraOff <= 3) console.error('extra off', m, 'cell', i, 'land t', t[i].toFixed(1)); } });
    }
    let h = 0, inner = 0;
    for (let r = 1; r < Hh - 1; r++) for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (!has[i] || reached[i]) continue;
        let k = 0, deep = true;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            if (!dr && !dc) continue;
            let cc = c + dc; if (cc < 0 || cc >= W) { if (!L.wrap) continue; cc = (cc + W) % W; }
            const j = (r + dr) * W + cc;
            if (reached[j]) {
                k++;
                // a neighbour near the time or survival limit makes it a limit hole
                let tb = Infinity, hb = Infinity;
                if (grid[j]) { tb = t[j]; hb = L.res.hz[j]; }
                eachCanoeNode(cn, j, (m) => { if (t[m] < tb) { tb = t[m]; hb = L.res.hz[m]; } });
                if (tb > 800 || hb > Math.log(100) - 0.7) deep = false;
            }
        }
        if (k >= 6) { h++; if (deep) inner++; }
    }
    return { holes: h, inner, extraOff };
}
const out = [];
for (const [name, lat, lng] of [['Constanta', 44.17, 28.63], ['Shanghai', 31.23, 121.47], ['Dhaka', 23.81, 90.41], ['Belem', -1.45, -48.5], ['Nukus', 42.46, 59.6], ['Nuuk', 64.18, -51.72], ['Coari', -4.08, -63.14], ['Jyvaskyla', 62.24, 25.75]]) {
    const L = await H.world10(true);
    await H.solveFrom(L, lat, lng, true);
    out.push(`L10 ${name}: ${JSON.stringify(holes(L))}`);
}
for (const [name, lat, lng] of [['Danube delta', 45.1, 29.3], ['Shanghai', 31.23, 121.47], ['Sundarbans', 22.0, 89.3], ['Nuuk', 64.18, -51.72], ['Coari', -4.08, -63.14], ['Jyvaskyla', 62.24, 25.75]]) {
    const box = H.boxAround(lat, lng, 1.2, 1.6);
    for (const f of [3, 1]) {
        const w = H.win(f, box.west, box.south, box.east, box.north);
        const b = await H.loadBox(w);
        const L = H.level(w, b.grid, b.hz);
        await H.solveFrom(L, lat, lng, true);
        out.push(`L${f} ${name}: ${JSON.stringify(holes(L))}`);
        L.cn = null;
    }
}
console.log(out.join('\n'));
process.exit(0);
