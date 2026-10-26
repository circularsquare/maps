// The route to a point, traced back along the predecessor codes (docs/canoe.md 6): through the
// finest solved level holding the point, then on through the coarser ones from the seed each
// window's path came in by. Also the time split by layer (walked, carried or paddled).
import { cfg } from './config.js';
import { MDC, MDR, P_NONE, P_XFER, P_DOWN, P_UP, P_SEED } from './search.js';

// Kinds of step, also the index into parts.
export const WALK = 0, CARRY = 1, PADDLE = 2;

function cellAt(L, lng, lat) {
    let c;
    if (L.wrap) c = ((Math.floor((lng - L.west) / L.cellDeg) % L.W) + L.W) % L.W;
    else c = Math.floor((((lng - L.west) % 360) + 360) % 360 / L.cellDeg);
    const r = Math.floor((L.north - lat) / L.cellDeg);
    if (r < 0 || r >= L.H || c < 0 || c >= L.W) return -1;
    return r * L.W + c;
}

// The node of a cell in one layer, or -1.
function nodeOf(L, cell, canoe, onCanoe) {
    if (canoe && onCanoe) return L.cn.canoeOf[cell];
    return L.grid[cell] ? cell : -1;
}

// The reached node of a layer at a point: its own cell's if reached, else the least weight in
// the 3 x 3 round it (a window seed blends the four coarser cells round its point).
function nearNode(L, lng, lat, canoe, onCanoe) {
    const cell = cellAt(L, lng, lat);
    if (cell < 0) return -1;
    const { w } = L.res;
    let n = nodeOf(L, cell, canoe, onCanoe);
    if (n >= 0 && w[n] !== Infinity) return n;
    const r = (cell / L.W) | 0, c = cell - r * L.W;
    let best = -1, bw = Infinity;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const r2 = r + dr;
        let c2 = c + dc;
        if (r2 < 0 || r2 >= L.H) continue;
        if (c2 < 0 || c2 >= L.W) { if (!L.wrap) continue; c2 = (c2 + L.W) % L.W; }
        n = nodeOf(L, r2 * L.W + c2, canoe, onCanoe);
        if (n >= 0 && w[n] < bw) { bw = w[n]; best = n; }
    }
    return best;
}

// levels: solved levels, finest first (each with res, and cn on a canoe start). Returns null when
// the point is not reached, else { t, hz, parts: hours by kind, lng, lat: Float64Array from the
// start to the point, kind: Uint8Array, the kind of the step ending at each point }.
export function trace(levels, canoe, lng, lat, origin) {
    let li = levels.findIndex(L => cellAt(L, lng, lat) >= 0);
    if (li < 0) return null;
    let L = levels[li];
    // the end: the cell's node with the least time, as painted
    const cell0 = cellAt(L, lng, lat);
    let n = -1, best = Infinity;
    for (const onCanoe of canoe ? [false, true] : [false]) {
        const m = nodeOf(L, cell0, canoe, onCanoe);
        if (m >= 0 && L.res.t[m] < best) { best = L.res.t[m]; n = m; }
    }
    if (n < 0 || !(best <= cfg.maxHours)) return null;
    const t = best, hz = L.res.hz[n];
    const parts = [0, 0, 0];
    // points from the pointer back to the start; seg[i] the kind of the step from point i to i + 1
    const xs = [lng], ys = [lat], seg = [];
    let lastX = lng, pending = -1;
    for (let guard = 0; guard < 1e7; guard++) {
        const { W, grid } = L, N = W * L.H;
        const onCanoe = canoe && (n >= N || !grid[n]);
        const layer = !canoe ? WALK : onCanoe ? PADDLE : CARRY;
        if (pending < 0) pending = layer;
        const code = L.res.pred[n];
        const cell = n < N ? n : L.cn.cellOf[n - N];
        const r = (cell / W) | 0, c = cell - r * W;
        let x = L.west + (c + 0.5) * L.cellDeg;
        const y = L.north - (r + 0.5) * L.cellDeg;
        x += Math.round((lastX - x) / 360) * 360;   // no jump across 180 degrees
        if (code === P_SEED) {
            // on in the coarser level the window was seeded from, same layer, at this point
            let m = -1;
            while (++li < levels.length && (m = nearNode(levels[li], x, y, canoe, onCanoe)) < 0) { /* next coarser */ }
            if (m < 0) break;
            const C = levels[li];
            parts[layer] += L.res.t[n] - C.res.t[m];
            L = C; n = m;
            continue;
        }
        xs.push(x); ys.push(y); seg.push(pending);
        lastX = x;
        if (code === P_NONE) { pending = layer; break; }
        let dc, dr, flip = false;
        if (code < P_XFER) { dc = MDC[code - 1]; dr = MDR[code - 1]; }
        else if (code < P_DOWN) { const j = code - P_XFER; dc = j % 3 - 1; dr = ((j / 3) | 0) - 1; flip = true; }
        else { const k = code < P_UP ? code - P_DOWN : code - P_UP; dc = MDC[k]; dr = MDR[k]; }
        let pc = c - dc;
        if (L.wrap) pc = (pc + W) % W;
        const p = nodeOf(L, (r - dr) * W + pc, canoe, flip ? !onCanoe : onCanoe);
        if (p < 0) break;
        // a transfer counts as carrying: getting in and out happens on the bank
        pending = flip ? CARRY : layer;
        parts[pending] += L.res.t[n] - L.res.t[p];
        n = p;
    }
    if (origin) {
        xs.push(origin.lng + Math.round((lastX - origin.lng) / 360) * 360); ys.push(origin.lat);
        seg.push(pending < 0 ? WALK : pending);
    }
    // drop points in the middle of straight runs of one kind
    const keep = [0];
    for (let i = 1; i < xs.length - 1; i++) {
        const a = keep[keep.length - 1];
        const cross = (xs[i] - xs[a]) * (ys[i + 1] - ys[i]) - (ys[i] - ys[a]) * (xs[i + 1] - xs[i]);
        if (seg[i - 1] !== seg[i] || Math.abs(cross) > 1e-9) keep.push(i);
    }
    if (xs.length > 1) keep.push(xs.length - 1);
    // reversed: from the start to the point
    const K = keep.length;
    const ox = new Float64Array(K), oy = new Float64Array(K), ok = new Uint8Array(K);
    for (let i = 0; i < K; i++) {
        const j = keep[K - 1 - i];
        ox[i] = xs[j]; oy[i] = ys[j];
        ok[i] = seg[j] ?? seg[j - 1] ?? WALK;   // the step from point i - 1 to i is seg[j]
    }
    for (let i = 0; i < 3; i++) parts[i] = Math.max(0, parts[i]);
    return { t, hz, parts, lng: ox, lat: oy, kind: ok };
}
