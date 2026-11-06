// T-082 prototype helpers: canoe layers of boxes, the L1 access time, and the coarse river graph
// contracted from the L1 one (a canoe node per coarse cell and piece of river joined at L1).
import { pathToFileURL } from 'node:url';
import * as H from './lib.mjs';

const { CLOSED, CLS } = await import(pathToFileURL(H.ROOT + 'js/canoedata.js').href);
const { plainMove, P_DOWN, P_UP, NEAR_CANOE, ON_RIVER } = await import(pathToFileURL(H.ROOT + 'js/search.js').href);
const OPP = [1, 0, 3, 2, 7, 6, 5, 4];

export async function canoeBox(L) {
    const W10 = await H.world10(true);
    const data = await H.loadWindowWater(L.win, L.grid, (x, y) => { const s = H.shoreAt(W10, x, y); return s === undefined ? 78 : s; });
    H.buildCanoe(L, data, W10.sea);
    return data;
}

// walking hours from each L1 land cell to the canoe network (sea, lake, open river steps)
export async function accessL1(L1, data, euclid = false) {
    const { W, H: Hh, grid } = L1, N = W * Hh, cn = L1.cn;
    const onRiver = new Uint8Array(N);
    const st = data.steps;
    for (let s = 0; s < st.n; s++) if (!CLOSED[st.cls[s]]) { onRiver[st.a[s]] = 1; onRiver[st.b[s]] = 1; }
    const net = (i) => (cn.info[i] & 0xFF) || onRiver[i];
    const seeds = { idx: [], w: [], t: [], hz: [] };
    for (let r = 0; r < Hh; r++) for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (!grid[i]) continue;
        let ok = cn.canoeOf[i] >= N && net(i);
        for (let dr = -1; dr <= 1 && !ok; dr++) for (let dc = -1; dc <= 1 && !ok; dc++) {
            const rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= Hh || cc < 0 || cc >= W) continue;
            const j = rr * W + cc;
            if (!grid[j] && net(j)) ok = true;
        }
        if (ok) { seeds.idx.push(i); seeds.w.push(0); seeds.t.push(0); seeds.hz.push(0); }
    }
    const hz = H.cfg.hazard; H.cfg.hazard = false;
    // euclid: km over any ground (a grid of 1 km/h everywhere), for an access measured as distance
    const Ls = euclid ? H.makeLevel(new Uint8Array(N).fill(40), W, Hh, L1.west, L1.north, L1.cellDeg, false) : L1;
    const cnSave = L1.cn; L1.cn = null;
    const res = await H.solve(Ls, seeds, () => true, false);
    L1.cn = cnSave; H.cfg.hazard = hz;
    return res.t;
}

// coarse access from a distance field (km): mean km x the coarse cell's hours per km
export function accFromKm(acc, Lc) {
    const out = new Float32Array(acc.length);
    for (let i = 0; i < acc.length; i++) out[i] = Lc.grid[i] ? acc[i] * 40 / Lc.grid[i] : 0;
    return out;
}

// coarse cell of L1 window cell i, or -1
function coarseMap(w1, Lc, row0c, col0c, k, wrap) {
    return (i) => {
        const r = (i / w1.cols) | 0, c = i - r * w1.cols;
        const R = Math.floor((w1.row0 + r) / k) - row0c;
        let C = Math.floor((w1.col0 + c) / k) - col0c;
        if (wrap) C = ((C % Lc.W) + Lc.W) % Lc.W;
        if (R < 0 || R >= Lc.H || C < 0 || C >= Lc.W) return -1;
        return R * Lc.W + C;
    };
}

// coarse cells whose L1 block lies wholly in the window, less their outer ring
function interiorOf(w1, Lc, row0c, col0c, k, wrap) {
    const ins = new Uint8Array(Lc.W * Lc.H);
    const R0 = Math.ceil(w1.row0 / k) - row0c + 1, R1 = Math.floor((w1.row0 + w1.rows) / k) - row0c - 2;
    const C0 = Math.ceil(w1.col0 / k) - col0c + 1, C1 = Math.floor((w1.col0 + w1.cols) / k) - col0c - 2;
    for (let R = R0; R <= R1; R++) for (let C = C0; C <= C1; C++) {
        const CC = wrap ? ((C % Lc.W) + Lc.W) % Lc.W : C;
        if (R >= 0 && R < Lc.H && CC >= 0 && CC < Lc.W) ins[R * Lc.W + CC] = 1;
    }
    return ins;
}

export function coarseAcc(w1, L1, t1, Lc, row0c, col0c, k, wrap) {
    const map = coarseMap(w1, Lc, row0c, col0c, k, wrap);
    const sum = new Float64Array(Lc.W * Lc.H), cnt = new Float64Array(Lc.W * Lc.H);
    for (let i = 0; i < L1.grid.length; i++) {
        if (!L1.grid[i] || !(t1[i] < Infinity)) continue;
        const X = map(i); if (X < 0) continue;
        sum[X] += t1[i]; cnt[X]++;
    }
    const acc = new Float32Array(Lc.W * Lc.H);
    for (let i = 0; i < acc.length; i++) acc[i] = cnt[i] ? sum[i] / cnt[i] : 0;
    return acc;
}

// Replace the coarse level's river edges inside the window by the contraction of the L1 river
// graph: one canoe node per coarse cell and connected piece of L1 river in it (pieces touching sea
// or lake share the cell's sea or lake node), an edge per pair of pieces an open L1 step joins,
// its length half of each piece's own river length plus the step's.
export function contract(Lc, row0c, col0c, k, wrap, L1, w1, d1) {
    const cn = Lc.cn, N = Lc.W * Lc.H, grid = Lc.grid;
    const map = coarseMap(w1, Lc, row0c, col0c, k, wrap);
    const inside = interiorOf(w1, Lc, row0c, col0c, k, wrap);
    const st = d1.steps, n1 = w1.cols * w1.rows;
    const parent = new Int32Array(n1).fill(-1);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const open = (s) => !CLOSED[st.cls[s]];
    const XA = new Int32Array(st.n), XB = new Int32Array(st.n);
    for (let s = 0; s < st.n; s++) {
        XA[s] = map(st.a[s]); XB[s] = map(st.b[s]);
        if (!open(s)) continue;
        for (const x of [st.a[s], st.b[s]]) if (parent[x] < 0) parent[x] = x;
    }
    for (let s = 0; s < st.n; s++) {
        if (!open(s) || XA[s] !== XB[s] || XA[s] < 0 || !inside[XA[s]]) continue;
        const ra = find(st.a[s]), rb = find(st.b[s]);
        if (ra !== rb) parent[ra] = rb;
    }
    // pieces: length, v x length, sea or lake touched
    const pLen = new Map(), pVL = new Map(), pWet = new Map(), pCell = new Map();
    for (let s = 0; s < st.n; s++) {
        if (!open(s) || XA[s] !== XB[s] || XA[s] < 0 || !inside[XA[s]]) continue;
        const p = find(st.a[s]);
        pLen.set(p, (pLen.get(p) || 0) + st.km[s]); pVL.set(p, (pVL.get(p) || 0) + st.km[s] * st.v[s]);
    }
    for (let x = 0; x < n1; x++) {
        if (parent[x] < 0) continue;
        const X = map(x); if (X < 0 || !inside[X]) continue;
        const p = find(x);
        pCell.set(p, X);
        if (!pLen.has(p)) { pLen.set(p, 0); pVL.set(p, 0); }
        if (L1.cn.info[x] & 0xFF) pWet.set(p, 1);
    }
    // nodes: per coarse cell its pieces, the wet or longest on the cell's own node
    const byCell = new Map();
    for (const [p, X] of pCell) { if (!byCell.has(X)) byCell.set(X, []); byCell.get(X).push(p); }
    let E = cn.E;
    const extraCell = [], nodeOf = new Map(), nextList = [];
    const firstExtra = new Int32Array(N).fill(-1);
    const newNode = (X) => { extraCell.push(X); return N + E++; };
    let multi = 0;
    for (const [X, ps] of byCell) {
        ps.sort((a, b) => (pWet.get(b) || 0) - (pWet.get(a) || 0) || pLen.get(b) - pLen.get(a));
        if (ps.length > 1) multi++;
        let prim = cn.canoeOf[X];
        if (prim < 0) { prim = grid[X] ? newNode(X) : X; cn.canoeOf[X] = prim; }
        let last = -1;
        for (let q = 0; q < ps.length; q++) {
            const p = ps[q];
            if (q === 0 || pWet.get(p)) { nodeOf.set(p, prim); continue; }
            const nd = newNode(X);
            nodeOf.set(p, nd);
            if (last < 0) firstExtra[X] = nd; else nextList.push([last, nd]);
            last = nd;
        }
    }
    // access per node (km): mean over the cell's L1 land pixels of the straight distance to the
    // piece's nearest pixel; 0 for a piece on the cell's sea or lake node
    const pPix = new Map();
    for (let x = 0; x < n1; x++) {
        if (parent[x] < 0) continue;
        const X = map(x); if (X < 0 || !inside[X]) continue;
        const p = find(x);
        if (!pPix.has(p)) pPix.set(p, []);
        pPix.get(p).push(x);
    }
    const accKm = new Map();
    const km1 = 111.195 * w1.cell;
    for (const [X, ps] of byCell) {
        // the cell's land pixels in the window
        const R = (X / Lc.W) | 0, C = X - R * Lc.W;
        const r0 = (R + row0c) * k - w1.row0, c0 = (C + col0c) * k - w1.col0;   // boxes here never cross 180
        const land = [];
        for (let r = r0; r < r0 + k; r++) for (let c = c0; c < c0 + k; c++) {
            if (r < 0 || r >= w1.rows || c < 0 || c >= w1.cols) continue;
            if (L1.grid[r * w1.cols + c]) land.push(r * w1.cols + c);
        }
        const lat = w1.north - (r0 + k / 2) * w1.cell, dx = km1 * Math.cos(lat * Math.PI / 180), dy = km1;
        for (const p of ps) {
            const node = nodeOf.get(p);
            if (pWet.get(p) || !land.length) { if (!accKm.has(node)) accKm.set(node, 0); continue; }
            let sum = 0;
            for (const q of land) {
                const qr = (q / w1.cols) | 0, qc = q - qr * w1.cols;
                let best = Infinity;
                for (const x of pPix.get(p)) {
                    const xr = (x / w1.cols) | 0, xc = x - xr * w1.cols;
                    const d = Math.hypot((xc - qc) * dx, (xr - qr) * dy);
                    if (d < best) best = d;
                }
                sum += best;
            }
            const a = sum / land.length;
            accKm.set(node, accKm.has(node) ? Math.min(accKm.get(node), a) : a);
        }
    }
    // edges: the old ones but river steps with both ends inside, then the contracted ones
    const M = N + E;
    const accNode = new Float32Array(M);
    for (const [node, a] of accKm) accNode[node] = a;
    const cellOf = new Int32Array(E); cellOf.set(cn.cellOf); for (let q = 0; q < extraCell.length; q++) cellOf[cn.E + q] = extraCell[q];
    const nextNode = new Int32Array(E).fill(-1);
    for (const [a, b] of nextList) nextNode[a - N] = b;
    const edges = [];
    for (let i = 0; i < N; i++) {
        for (let e = cn.rOff[i]; e < cn.rOff[i + 1]; e++) {
            const src = cn.canoeOf[i] >= 0 ? cn.canoeOf[i] : i;
            const isRiver = cn.rCode[e] >= P_DOWN;
            if (isRiver && inside[i] && inside[cn.rTo[e]]) continue;
            const to = cn.canoeOf[cn.rTo[e]] >= 0 ? cn.canoeOf[cn.rTo[e]] : cn.rTo[e];
            edges.push([src, to, cn.rDt[e], cn.rCode[e], cn.rHz[e], cn.rKm[e]]);
        }
    }
    const best = new Map();
    const p5 = H.cfg.paddle;
    for (let s = 0; s < st.n; s++) {
        if (!open(s) || XA[s] === XB[s] || XA[s] < 0 || XB[s] < 0 || !inside[XA[s]] || !inside[XB[s]]) continue;
        const ps = find(st.a[s]), pt = find(st.b[s]);
        const a = nodeOf.get(ps), b = nodeOf.get(pt);
        const km = pLen.get(ps) / 2 + st.km[s] + pLen.get(pt) / 2;
        const v = (pVL.get(ps) / 2 + st.km[s] * st.v[s] + pVL.get(pt) / 2) / km;
        const key = a * 4194304 + b;
        const o = best.get(key);
        if (!o || o.km > km) best.set(key, { a, b, km, v, XA: XA[s], XB: XB[s], rapids: st.cls[s] === CLS.rapids });
    }
    for (const o of best.values()) {
        const ra = (o.XA / Lc.W) | 0, rb = (o.XB / Lc.W) | 0;
        let dc = (o.XB - rb * Lc.W) - (o.XA - ra * Lc.W);
        if (wrap) { if (dc > 1) dc -= Lc.W; if (dc < -1) dc += Lc.W; }
        const m = plainMove(dc, rb - ra);
        if (m < 0) continue;
        const vk = o.v * 3.6, rate = o.rapids ? H.cfg.hz.rapids : H.cfg.hz.river;
        const dd = o.km / (p5 + vk), du = o.km / Math.max(p5 - H.cfg.upShare * vk, H.cfg.upMin);
        edges.push([o.a, o.b, dd, P_DOWN + m, dd * rate, o.km]);
        edges.push([o.b, o.a, du, P_UP + OPP[m], du * rate, o.km]);
    }
    edges.sort((x, y) => x[0] - y[0]);
    const nOff = new Int32Array(M + 1), rTo = new Int32Array(edges.length), rDt = new Float32Array(edges.length);
    const rCode = new Uint8Array(edges.length), rHz = new Float32Array(edges.length), rKm = new Float32Array(edges.length);
    edges.forEach((e, q) => { nOff[e[0] + 1]++; rTo[q] = e[1]; rDt[q] = e[2]; rCode[q] = e[3]; rHz[q] = e[4]; rKm[q] = e[5]; });
    for (let i = 0; i < M; i++) nOff[i + 1] += nOff[i];
    // flags
    for (const e of edges) { const c = e[0] < N ? e[0] : cellOf[e[0] - N]; cn.info[c] |= ON_RIVER; }
    for (let i = 0; i < N; i++) {
        if (cn.canoeOf[i] < 0) continue;
        const r = (i / Lc.W) | 0, c = i - r * Lc.W;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr; if (rr < 0 || rr >= Lc.H) continue;
            let cc = c + dc; if (cc < 0 || cc >= Lc.W) { if (!wrap) continue; cc = (cc + Lc.W) % Lc.W; }
            cn.info[rr * Lc.W + cc] |= NEAR_CANOE;
        }
    }
    for (let q = 0; q < extraCell.length; q++) {
        const i = extraCell[q], r = (i / Lc.W) | 0, c = i - r * Lc.W;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr; if (rr < 0 || rr >= Lc.H) continue;
            let cc = c + dc; if (cc < 0 || cc >= Lc.W) { if (!wrap) continue; cc = (cc + Lc.W) % Lc.W; }
            cn.info[rr * Lc.W + cc] |= NEAR_CANOE;
        }
    }
    Object.assign(cn, { E, cellOf, nextNode, firstExtra, nOff, rTo, rDt, rCode, rHz, rKm, accNode });
    return { pieces: pCell.size, cells: byCell.size, multi };
}
