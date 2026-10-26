// Corridors (T-046, docs/corridors.md): where many trips from the start share a route. Each
// reached land cell is one trip, weighted by its area; a node's count is the area whose route
// passes through it, summed up the shortest-path tree (the predecessor codes). Edges with a count
// past a threshold are sent as polylines with a strength, log of the count.
import { cfg } from './config.js';
import { MDC, MDR, P_NONE, P_GRID, P_XFER, P_DOWN, P_UP, P_SEED } from './search.js';

export const STEPS = 8;      // strength steps; a polyline is a run of one step
// Threshold: this share of all the area reached at L10, the same at every level. A lower one in
// windows draws each coarse edge's flux across the window as its own straight line.
const SHARE = 0.002;
// A window's outer cells are seeded or reached along the border, so the flux leaving the window
// goes in this many cells from its edge, never onto the border ring itself.
const EDGE = 2;

// Each predecessor code as the parent's offset (DC, DR cells back), whether it changes layer
// (FLIP), and whether there is a parent at all (PARENT: not the start, not a window seed).
const DC = new Int8Array(256), DR = new Int8Array(256), FLIP = new Uint8Array(256), PARENT = new Uint8Array(256);
for (let code = 1; code < P_SEED; code++) {
    let k = -1;
    if (code < P_XFER) k = code - P_GRID;
    else if (code < P_DOWN) { const j = code - P_XFER; DC[code] = j % 3 - 1; DR[code] = ((j / 3) | 0) - 1; FLIP[code] = 1; }
    else if (code < P_UP + 8) k = code < P_UP ? code - P_DOWN : code - P_UP;
    else continue;
    if (k >= 0) { DC[code] = MDC[k]; DR[code] = MDR[k]; }
    PARENT[code] = 1;
}

// Work arrays, grown as needed and shared by every level (counted one at a time). nch is all 0
// between calls.
let nch = new Uint8Array(0), stack = new Int32Array(0), flag = new Uint8Array(0), reach = new Int32Array(0);
function work(M) {
    if (nch.length < M) { nch = new Uint8Array(M); stack = new Int32Array(M); flag = new Uint8Array(M); reach = new Int32Array(M); }
}

// The first of the coarser counted levels that holds the whole window L.
const holding = (L, coarser) => coarser && coarser.find(C => C.corr && (C.wrap ||
    (C.west <= L.west && C.east >= L.east && C.north >= L.north && C.south <= L.south)));

const lngOf = (L, cell) => L.west + (cell % L.W + 0.5) * L.cellDeg;
const latOf = (L, cell) => L.north - (((cell / L.W) | 0) + 0.5) * L.cellDeg;

// The counts of a solved level: L.corr = { par, cnt, total, nr } (par -1 a root; par and cnt
// hold only for reached nodes, those with t <= maxHours; nr of them in reach).
// coarser: the solved coarser levels of a window, whose counts give the area beyond its edge.
export function count(L, canoe, coarser) {
    const { W, grid, wrap, dxKm, dyKm } = L;
    const N = W * L.H;
    const { t, pred } = L.res;
    const M = t.length, maxH = cfg.maxHours;
    const canoeOf = canoe ? L.cn.canoeOf : null, cellOf = canoe ? L.cn.cellOf : null;
    const old = L.corr;
    const par = old && old.par.length >= M ? old.par : new Int32Array(M);
    const cnt = old && old.cnt.length >= M ? old.cnt : new Float32Array(M);
    work(M);
    let total = 0, nr = 0;
    // reached nodes: a predecessor code and t <= maxHours, read 4 codes at a time (most are 0)
    const words = new Uint32Array(pred.buffer, pred.byteOffset, M >> 2);
    for (let i = 0; i < words.length; i++) {
        if (words[i] === 0) continue;
        for (let n = i << 2, e = n + 4; n < e; n++) if (pred[n] !== P_NONE && t[n] <= maxH) reach[nr++] = n;
    }
    for (let n = (M >> 2) << 2; n < M; n++) if (pred[n] !== P_NONE && t[n] <= maxH) reach[nr++] = n;
    // and the start, which has no code: found as a parent
    const roots = [];
    for (let i = 0; i < nr; i++) {
        const n = reach[i];
        const cell = n < N ? n : cellOf[n - N];
        const r = (cell / W) | 0;
        const land = n < N && grid[n] !== 0;
        const w = land ? dxKm[r] * dyKm : 0;     // trips end on land
        cnt[n] = w; total += w;
        // the parent, inlined
        const code = pred[n];
        let p = -1;
        if (PARENT[code]) {
            let pc = cell - r * W - DC[code];
            if (pc < 0 || pc >= W) pc = wrap ? (pc + W) % W : pc;
            const pcell = (r - DR[code]) * W + pc;
            p = (FLIP[code] ? land : !land) ? canoeOf[pcell] : pcell;
        }
        par[n] = p;
        if (p < 0) continue;
        nch[p]++;
        if (pred[p] === P_NONE && !roots.includes(p)) {
            roots.push(p);
            reach[nr++] = p;      // its own turn in this loop sets its par and cnt
        }
    }
    // A window holding the start: which nodes' routes go back to it (org 1) and which to a border
    // seed (2). Near the border the seeds often win (a coarse time a little earlier than the fine
    // one), and downstream on a river they win far in; there the coarser level's lines are kept.
    let org = null;
    if (coarser && roots.length) {
        org = new Uint8Array(M);
        for (let i = 0; i < nr; i++) {
            let a = reach[i], k = 0;
            while (!org[a] && par[a] >= 0) { stack[k++] = a; a = par[a]; }
            let f = org[a];
            if (!f) { f = pred[a] === P_NONE ? 1 : 2; org[a] = f; }
            while (k > 0) org[stack[--k]] = f;
        }
    }
    if (coarser) total += edgeFlux(L, canoe, coarser, cnt, par, org);
    // leaves first: a node is added to its parent once all its children are in
    let sp = 0;
    for (let i = 0; i < nr; i++) { const n = reach[i]; if (nch[n] === 0) stack[sp++] = n; }
    while (sp > 0) {
        const n = stack[--sp], p = par[n];
        if (p >= 0) { cnt[p] += cnt[n]; if (--nch[p] === 0) stack[sp++] = p; }
    }
    L.corr = { par, cnt, total, nr, org };
    return L.corr;
}

// A window only holds the trips that end inside it. The trips that leave it are the coarser
// level's edges from inside the window to outside: each one's count is added to the window node
// nearest where it leaves, off the border ring, not a seed, and (org) whose route goes back to the
// start when the start is inside; none near and the flux is left to the coarser level's lines.
// Returns the area added.
function edgeFlux(L, canoe, coarser, cnt, par, org) {
    const C = holding(L, coarser);
    if (!C) return 0;
    const { W, H, grid } = L;
    const { t, pred } = L.res;
    const canoeOf = canoe ? L.cn.canoeOf : null;
    const cc = C.corr, ct = C.res.t, CW = C.W, CN = C.W * C.H;
    const ccanoeOf = canoe ? C.cn.canoeOf : null, ccellOf = canoe ? C.cn.cellOf : null;
    const maxH = cfg.maxHours;
    const R = Math.ceil(C.cellDeg / L.cellDeg) + 1 + EDGE;
    const lngIn = (cell) => L.west + ((lngOf(C, cell) - L.west) % 360 + 360) % 360;
    const inside = (cell) => {
        const x = lngIn(cell), y = latOf(C, cell);
        return x < L.east && y < L.north && y > L.south;
    };
    // coarse cells in a band 3 cells wide round the window edge
    const c0 = Math.floor((L.west - C.west) / C.cellDeg) - 3, c1 = Math.ceil((L.east - C.west) / C.cellDeg) + 3;
    const r0 = Math.max(0, Math.floor((C.north - L.north) / C.cellDeg) - 3);
    const r1 = Math.min(C.H - 1, Math.ceil((C.north - L.south) / C.cellDeg) + 3);
    let added = 0;
    const cellOfC = (n) => n < CN ? n : ccellOf[n - CN];
    // the usable window node nearest coarse node n (on coarse cell `cell`), same layer first
    const nearest = (n, cell, R) => {
        let x = lngIn(cell);
        if (x - L.east > L.west + 360 - x) x -= 360;    // just west of the window
        const fc = Math.min(W - 1, Math.max(0, Math.floor((x - L.west) / L.cellDeg)));
        const fr = Math.min(H - 1, Math.max(0, Math.floor((L.north - latOf(C, cell)) / L.cellDeg)));
        const onC = n >= CN || (canoe && !C.grid[n]);
        let best = -1, bd = Infinity;
        for (const layer of canoe ? [onC, !onC] : [false]) {
            for (let dr = -R; dr <= R; dr++) {
                const r = fr + dr;
                if (r < EDGE || r >= H - EDGE) continue;
                for (let dc = -R; dc <= R; dc++) {
                    const c = fc + dc;
                    if (c < EDGE || c >= W - EDGE) continue;
                    const cell2 = r * W + c;
                    const m = layer ? canoeOf[cell2] : (grid[cell2] ? cell2 : -1);
                    if (m < 0 || !(t[m] <= maxH) || pred[m] === P_SEED || (org && org[m] !== 1)) continue;
                    const d = dr * dr + dc * dc;
                    if (d < bd) { bd = d; best = m; }
                }
            }
            if (best >= 0) break;
        }
        return best;
    };
    // flux worth following inward when the border is seed-owned
    const follow = cc.total * SHARE * 0.1, R0 = Math.ceil(C.cellDeg / L.cellDeg);
    const visit = (n, cell) => {
        const p = cc.par[n];
        if (p < 0) return;
        if (!inside(cellOfC(p))) return;
        let best = nearest(n, cell, R);
        // none routed from the start near the edge: on inward along the coarse route, to where
        // the window's routes from the start meet the seeded ones; the coarser lines draw the rest
        if (best < 0 && org && cc.cnt[n] >= follow) {
            for (let q = p, k = 0; q >= 0 && best < 0 && k < 5000; q = cc.par[q], k++) {
                const qc = cellOfC(q);
                if (!inside(qc)) break;
                best = nearest(q, qc, R0);
            }
        }
        if (best < 0) return;
        cnt[best] += cc.cnt[n];
        added += cc.cnt[n];
    };
    for (let r = r0; r <= r1; r++) {
        for (let k = c0; k <= c1; k++) {
            let c = k;
            if (C.wrap) c = ((c % CW) + CW) % CW; else if (c < 0 || c >= CW) continue;
            const cell = r * CW + c;
            if (inside(cell)) continue;
            if (C.grid[cell] && ct[cell] <= maxH) visit(cell, cell);
            if (canoe) { const m = ccanoeOf[cell]; if (m >= 0 && ct[m] <= maxH) visit(m, cell); }
        }
    }
    return added;
}

// Polylines of the edges whose count is at least the level's threshold: { xy (lng, lat pairs),
// off (start of each line in points, and the end), s (strength step 0..STEPS-1 of each line) }.
// world: the L10 total, which sets the threshold and the top of the scale. coarser: as for count;
// the lines of the one holding the window (C.drawn) fill in where the window's own routes do not
// go back to the start. The result is kept as L.drawn for a finer window.
export function lines(L, canoe, world, coarser) {
    const P = holding(L, coarser);
    const { par, cnt, nr, org } = L.corr;
    const N = L.W * L.H;
    const cellOf = canoe ? L.cn.cellOf : null;
    const T = world * SHARE;
    const span = Math.log(world / T);
    const step = (n) => Math.min(STEPS - 1, Math.max(0, Math.floor(STEPS * Math.log(cnt[n] / T) / span)));
    const kept = org ? (n) => par[n] >= 0 && cnt[n] >= T && org[n] === 1 : (n) => par[n] >= 0 && cnt[n] >= T;
    // the kept nodes (in stack), and kept children per node in nch (all 0 after counting)
    let nk = 0;
    for (let i = 0; i < nr; i++) { const n = reach[i]; if (kept(n)) stack[nk++] = n; }
    for (let i = 0; i < nk; i++) nch[par[stack[i]]]++;
    // an edge continues the line of the edge before it when that is the only kept edge on, of one step
    const cont = (n) => { const p = par[n]; return nch[p] === 1 && kept(p) && step(p) === step(n); };
    for (let i = 0; i < nk; i++) { const n = stack[i]; if (cont(n)) flag[par[n]] = 1; }
    const xy = [], off = [0], s = [];
    const cellOfNode = (n) => n < N ? n : cellOf[n - N];
    for (let i = 0; i < nk; i++) {
        const n = stack[i];
        if (flag[n]) continue;
        // from a line's tip up to where it joins
        const st = step(n);
        let cur = n, cell = cellOfNode(n);
        let x = lngOf(L, cell), y = latOf(L, cell);
        const pts = [x, y];
        for (;;) {
            const p = par[cur];
            const pc = cellOfNode(p);
            if (pc !== cell) {
                cell = pc;
                let px = lngOf(L, cell);
                px += Math.round((x - px) / 360) * 360;
                const py = latOf(L, cell);
                // drop the middle point of a straight run
                const k = pts.length;
                if (k >= 4) {
                    const ax = pts[k - 4], ay = pts[k - 3];
                    if (Math.abs((x - ax) * (py - y) - (y - ay) * (px - x)) < 1e-9) pts.length = k - 2;
                }
                pts.push(px, py);
                x = px; y = py;
            }
            if (!cont(cur)) break;
            cur = p;
        }
        if (pts.length < 4) continue;
        for (const v of pts) xy.push(v);
        off.push(xy.length / 2);
        s.push(st);
    }
    for (let i = 0; i < nk; i++) { const n = stack[i]; nch[par[n]] = 0; flag[par[n]] = 0; }
    if (org && P && P.drawn) coarserLines(L, canoe, P.drawn, P.cellDeg, xy, off, s);
    const out = { xy: new Float32Array(xy), off: new Uint32Array(off), s: new Uint8Array(s) };
    L.drawn = { xy: out.xy.slice(), off: out.off.slice(), s: out.s.slice() };
    return out;
}

// The coarser level's lines inside a window where the window's own routes do not go back to the
// start (seeded from its border, or not reached): cut in pieces about a coarse cell long, each
// kept when the window cell under its middle has no node routed from the start.
function coarserLines(L, canoe, D, step, xy, off, s) {
    const { W, H, grid } = L;
    const org = L.corr.org, t = L.res.t, maxH = cfg.maxHours;
    const canoeOf = canoe ? L.cn.canoeOf : null;
    const covered = (x, y) => {
        const c = Math.floor((((x - L.west) % 360) + 360) % 360 / L.cellDeg), r = Math.floor((L.north - y) / L.cellDeg);
        if (r < 0 || r >= H || c >= W) return null;     // outside: drawn by the coarser level itself
        const cell = r * W + c;
        if (grid[cell] && t[cell] <= maxH && org[cell] === 1) return true;
        const m = canoe ? canoeOf[cell] : -1;
        return m >= 0 && t[m] <= maxH && org[m] === 1;
    };
    for (let i = 0; i + 1 < D.off.length; i++) {
        let run = null;
        const end = () => { if (run && run.length >= 4) { for (const v of run) xy.push(v); off.push(xy.length / 2); s.push(D.s[i]); } run = null; };
        for (let j = D.off[i]; j + 1 < D.off[i + 1]; j++) {
            const x0 = D.xy[2 * j], y0 = D.xy[2 * j + 1], x1 = D.xy[2 * j + 2], y1 = D.xy[2 * j + 3];
            const k = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / step));
            for (let q = 0; q < k; q++) {
                const xa = x0 + (x1 - x0) * q / k, ya = y0 + (y1 - y0) * q / k;
                const xb = x0 + (x1 - x0) * (q + 1) / k, yb = y0 + (y1 - y0) * (q + 1) / k;
                const cov = covered((xa + xb) / 2, (ya + yb) / 2);
                if (cov === false) {
                    if (!run) run = [xa, ya];
                    run.push(xb, yb);
                } else end();
            }
        }
        end();
    }
}
