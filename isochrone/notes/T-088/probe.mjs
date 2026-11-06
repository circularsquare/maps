// Hours by kind of edge along the route to destinations at L10, L3, L1 (canoe start), with canoeFrom.
//   node probe.mjs <lat> <lng> [km,km] [bearings]
import * as H from './lib.mjs';
const { MDC, MDR, P_GRID, P_XFER, P_DOWN, P_UP } = await H.imp('search.js');
const { canoeFrom, eachCanoeNode } = await H.imp('canoe.js');
H.cfg.hazard = true;
const lat = Number(process.argv[2]), lng = Number(process.argv[3]), KMS = (process.argv[4] || '80,160').split(',').map(Number);
const BEAR = (process.argv[5] || '0,90,180,270').split(',').map(Number);
const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
const L = { 10: await H.world10(true) };
for (const f of [3, 1]) { const w = H.win(f, box.west, box.south, box.east, box.north); const b = await H.loadBox(w); L[f] = H.level(w, b.grid, b.hz); }
for (const f of [10, 3, 1]) await H.solveFrom(L[f], lat, lng, true);

function route(Lv, la, lo, verbose) {
    const { W, res, grid } = Lv, N = W * Lv.H, cn = Lv.cn;
    const cellAt = (r, c) => r * W + ((c % W) + W) % W;
    const r0 = Math.floor((Lv.north - la) / Lv.cellDeg), c0 = Math.floor((lo - Lv.west) / Lv.cellDeg);
    const cell0 = cellAt(r0, c0);
    let n = -1, bt = Infinity;
    if (grid[cell0] && res.t[cell0] < bt) { bt = res.t[cell0]; n = cell0; }
    eachCanoeNode(cn, cell0, (x) => { if (res.t[x] < bt) { bt = res.t[x]; n = x; } });
    const by = { carry: 0, xfer: 0, down: 0, up: 0, lake: 0, sea: 0, other: 0 };
    const seq = [];
    let nx = 0;
    for (let g = 0; g < 1e6 && n >= 0; g++) {
        const p = res.pred[n];
        if (p === 0 || p === 255) break;
        const cell = n < N ? n : cn.cellOf[n - N], r = (cell / W) | 0, c = cell - r * W;
        let dc, dr, other = false, kind;
        if (p >= P_UP) { const k = p - P_UP; dc = MDC[k]; dr = MDR[k]; kind = 'up'; }
        else if (p >= P_DOWN) { const k = p - P_DOWN; dc = MDC[k]; dr = MDR[k]; kind = 'down'; }
        else if (p >= P_XFER) { const j = p - P_XFER; dc = j % 3 - 1; dr = ((j / 3) | 0) - 1; other = true; kind = 'xfer'; nx++; }
        else { const k = p - P_GRID; dc = MDC[k]; dr = MDR[k]; const canoe = n >= N || !grid[n]; kind = canoe ? ((cn.info[cell] & 0xFF) >= 3 ? 'lake' : (cn.info[cell] & 0xFF) === 1 ? 'sea' : 'other') : 'carry'; }
        const pc = cellAt(r - dr, c - dc);
        const canoeHere = n >= N || !grid[n];
        const canoePrev = other ? !canoeHere : canoeHere;
        const m = canoePrev ? canoeFrom(Lv, res, n, p, pc) : pc;
        const dt = res.t[n] - res.t[m];
        by[kind] += dt;
        if (verbose) seq.push(`${kind[0]}${(Lv.north - (r + 0.5) * Lv.cellDeg).toFixed(2)},${(Lv.west + (c + 0.5) * Lv.cellDeg).toFixed(2)}:${dt.toFixed(2)}`);
        n = m;
    }
    return `${bt.toFixed(1)} h: ` + Object.entries(by).filter(([, v]) => v > 0.05).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(', ') + ` (${nx} transfers)` + (verbose ? '\n    ' + seq.reverse().join(' ') : '');
}
const DESTS = KMS.flatMap(km => BEAR.map(b => [km, b, ...H.offset(lat, lng, km, b)]));
for (const [km, b, la, lo] of DESTS) {
    console.log(`${km} km bearing ${b}: ${la.toFixed(3)},${lo.toFixed(3)}`);
    for (const f of [10, 3, 1]) console.log(`  L${f} ${route(L[f], la, lo, process.env.V === String(f))}`);
}
process.exit(0);
