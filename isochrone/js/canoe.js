// The canoe layer of a level (docs/canoe.md 1, 2): which cells have a canoe node, the per-cell
// word the search reads (water kind, links, flags), the river steps as out-edges, and the current.
import { cfg } from './config.js';
import { FILE_DIRS, CLOSED, CLS } from './canoedata.js';
import { plainMove, MDC, MDR, P_GRID, P_XFER, P_DOWN, P_UP, NEAR_LAND, NEAR_CANOE, FAR, ON_RIVER } from './search.js';

// file direction -> plain move, and link bits from file order to plain move order
const DIR_MOVE = FILE_DIRS.map(([dc, dr]) => plainMove(dc, dr));
const LINK_LUT = new Uint8Array(256);
for (let b = 0; b < 256; b++) {
    let o = 0;
    for (let k = 0; k < 8; k++) if (b & (1 << k)) o |= 1 << DIR_MOVE[k];
    LINK_LUT[b] = o;
}
const OPP = [1, 0, 3, 2, 7, 6, 5, 4];   // plain move -> its reverse (search.js order)

// shore code -> km (docs/canoe.md 4)
const SHORE_KM = new Float32Array(256);
for (let s = 0; s < 256; s++) SHORE_KM[s] = s <= 20 ? s : 20 + 10 * (Math.min(s, 78) - 20);

// Hazard per paddling hour on open water (cfg.hz), from distance to shore d (km), wave height
// Hs (m) and water temperature (C, the lake band on a lake): exposure x (water + waves(Hs) x min(d / waveKm, 1) x cold +
// supplies(d)), exposure being cfg.hz.exposure real hours an hour past farKm.
function waterRate(d, hs, sst) {
    const h = cfg.hz;
    let wave = 0;
    for (const [top, r] of h.waves) if (hs < top) { wave = r; break; }
    let cold = 1;
    for (const [below, m] of h.cold) if (sst < below) { cold = m; break; }
    const [s0, s1] = h.supplyKm;
    const supply = d <= s0 ? 0 : h.supply * Math.min(1, (d - s0) / (s1 - s0));
    return (d > h.farKm ? h.exposure : 1) * (h.water + wave * Math.min(1, d / h.waveKm) * cold + supply);
}

// Set where any cell of the 3 x 3 round it is set (wrapping at 180 degrees when L wraps).
function dilate3(a, W, H, wrap) {
    const h = new Uint8Array(W * H), out = new Uint8Array(W * H);
    for (let r = 0; r < H; r++) {
        const o = r * W;
        for (let c = 0; c < W; c++) {
            let v = a[o + c];
            if (c > 0) v |= a[o + c - 1]; else if (wrap) v |= a[o + W - 1];
            if (c < W - 1) v |= a[o + c + 1]; else if (wrap) v |= a[o];
            h[o + c] = v;
        }
    }
    for (let r = 0; r < H; r++) {
        const o = r * W;
        for (let c = 0; c < W; c++) {
            let v = h[o + c];
            if (r > 0) v |= h[o - W + c];
            if (r < H - 1) v |= h[o + W + c];
            out[o + c] = v;
        }
    }
    return out;
}

// The water links of a level (T-088) as directed edges: { n, from, to (nodes), cellFrom, dt, hz, km,
// code (P_GRID + plain move) }, costed as search.js costs a sea or lake grid move. An end is a listed
// water unit (its own kind and shore) or a cell's own sea or lake node; iced sea ends are left out.
function waterLinks(L, data, sea, pieces, info, nodeAt, seaRow, seaCol) {
    const { W, dxKm, dyKm } = L;
    const unit = new Map();
    for (let q = 0; q < pieces.n; q++) unit.set(pieces.cell[q] * 256 + pieces.slot[q], q);
    const p = cfg.paddle, cur = sea.unit * 0.5 * 3.6 * 0.5;
    // an end: node, kind, hazard rate, current east and north (0.01 m/s, halved within 5 km of shore)
    const end = (cell, slot, o) => {
        const q = unit.get(cell * 256 + slot);
        let k, sh;
        if (q !== undefined) { k = pieces.kind[q]; sh = pieces.shore[q]; }
        else if (!slot) { k = info[cell] & 0xFF; sh = data.shore[cell]; }
        else return false;
        if (!k) return false;
        const node = nodeAt(cell, slot);
        if (node < 0) return false;
        const r = (cell / W) | 0, s = seaRow[r] * sea.W + seaCol[cell - r * W];
        if (k === 1 && sea.ice[s] > cfg.iceMax) return false;
        const m = sh <= 5 ? 1 : 2;
        o.node = node; o.k = k;
        o.rate = k === 1 ? waterRate(SHORE_KM[sh], sea.hs[s] * 0.05, sea.sst[s]) : waterRate(SHORE_KM[sh], (k - 3) * 0.05, sea.lake[s]);
        o.cx = k === 1 ? sea.u[s] * m : 0; o.cy = k === 1 ? sea.v[s] * m : 0;
        return true;
    };
    const A = {}, B = {};
    const n2 = 2 * pieces.m;
    const out = { n: 0, from: new Int32Array(n2), to: new Int32Array(n2), cellFrom: new Int32Array(n2),
                  dt: new Float32Array(n2), hz: new Float32Array(n2), km: new Float32Array(n2), code: new Uint8Array(n2) };
    const add = (a, b, ca, len, ex, ey, mv) => {
        let dt;
        if (a.k >= 3 || b.k >= 3) dt = len / p;
        else {
            const mx = (a.cx + b.cx) * cur, my = (a.cy + b.cy) * cur;
            if (mx === 0 && my === 0) dt = len / p;
            else {
                const cross = mx * ey - my * ex, q = p * p - cross * cross;
                if (q <= 0) return;
                const s = mx * ex + my * ey + Math.sqrt(q);
                if (s <= 0) return;
                dt = len / s;
            }
        }
        const k = out.n++;
        out.from[k] = a.node; out.to[k] = b.node; out.cellFrom[k] = ca; out.dt[k] = dt;
        out.hz[k] = dt * 0.5 * (a.rate + b.rate); out.km[k] = len; out.code[k] = P_GRID + mv;
    };
    for (let q = 0; q < pieces.m; q++) {
        const ca = pieces.a[q], cb = pieces.b[q];
        if (!end(ca, pieces.sa[q], A) || !end(cb, pieces.sb[q], B)) continue;
        const mv = DIR_MOVE[pieces.dir[q]], r = (ca / W) | 0;
        const dx = MDC[mv] * (dxKm[r] + dxKm[r + MDR[mv]]) * 0.5, dy = MDR[mv] * dyKm;
        const len = Math.sqrt(dx * dx + dy * dy), ex = dx / len, ey = -dy / len;
        add(A, B, ca, len, ex, ey, mv);
        add(B, A, cb, len, -ex, -ey, OPP[mv]);
    }
    return out;
}

// A cell's `info` bit set where the cell is water of any kind, river water included, whether or not
// it has a canoe node: what paint.js mixes towards white (T-063, T-074).
export const PAINT_WATER = 1 << 20;

// data: { kind, shore, links, steps, pieces (L3, L10, or null) } from canoedata.js; sea: the 0.25
// degree sea grid. Sets and returns L.cn = { E (canoe nodes beyond the cells), E1 (of them, the
// cells' own; then the further nodes, river pieces and water units, extra0[cell] .. + extraN[cell] - 1,
// T-082, T-088), canoeOf, cellOf, acc (km, by node
// - N, or null), info, curr, curKmh, shore, hzr, rOff (by node), rTo (node), rDt, rCode, rHz, rKm
// (the step's river length, km) }.
export function buildCanoe(L, data, sea) {
    const { grid, W, H } = L;
    const N = W * H;
    const { kind, shore, steps } = data;
    const pieces = data.pieces || null;
    // sea grid row and column of each cell row and column
    const seaRow = new Int32Array(H), seaCol = new Int32Array(W);
    for (let r = 0; r < H; r++)
        seaRow[r] = Math.min(sea.H - 1, Math.max(0, Math.floor((sea.north - (L.north - (r + 0.5) * L.cellDeg)) / sea.deg)));
    for (let c = 0; c < W; c++) {
        const x = Math.floor((L.west + (c + 0.5) * L.cellDeg - sea.west) / sea.deg);
        seaCol[c] = ((x % sea.W) + sea.W) % sea.W;
    }
    // open river steps; a cell's own canoe node carries its slot-0 river piece, and pieces on other
    // slots (L3, L10: rivers that share the cell without joining, T-082) get nodes of their own
    const open = new Uint8Array(steps.n);
    const onRiver = new Uint8Array(N), extraN = new Uint8Array(N);
    let nOpen = 0, anyAcc = false;
    for (let s = 0; s < steps.n; s++) {
        if (CLOSED[steps.cls[s]] || DIR_MOVE[steps.dir[s]] < 0) continue;
        open[s] = 1; nOpen++;
        const a = steps.a[s], b = steps.b[s], sa = steps.sa[s], sb = steps.sb[s];
        if (steps.aa[s] > 0 || steps.ab[s] > 0) anyAcc = true;
        if (sa) { if (sa > extraN[a]) extraN[a] = sa; } else onRiver[a] = 1;
        if (sb) { if (sb > extraN[b]) extraN[b] = sb; } else onRiver[b] = 1;
    }
    // water units (L3, L10: separate pieces of sea or lake in one cell, T-088) on further slots, and
    // a water unit on slot 0 of a river-water cell (wet0); cells a water link touches (linked)
    const wet0 = new Uint8Array(pieces ? N : 0), linked = new Uint8Array(pieces ? N : 0);
    if (pieces) {
        for (let q = 0; q < pieces.n; q++) {
            const c = pieces.cell[q], s = pieces.slot[q];
            if (s) { if (s > extraN[c]) extraN[c] = s; } else if (kind[c] === 2) wet0[c] = 1;
            if (pieces.acc[q] > 0) anyAcc = true;
        }
        for (let q = 0; q < pieces.m; q++) { linked[pieces.a[q]] = 1; linked[pieces.b[q]] = 1; }
    }
    // A sea or lake cell with no land, no water it is linked to, no river and no water link beside
    // it (a one-pixel pond inside river water, T-075) is river water off the line (below).
    const pond = new Uint8Array(N);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const i = r * W + c, k = kind[i];
        if (grid[i] || k === 2 || onRiver[i] || extraN[i] || (pieces && (linked[i] || wet0[i]))) continue;
        let alone = true;
        for (let b = 0; b < 8; b++) {
            const r2 = r + FILE_DIRS[b][1];
            if (r2 < 0 || r2 >= H) continue;
            let c2 = c + FILE_DIRS[b][0];
            if (c2 < 0 || c2 >= W) { if (!L.wrap) continue; c2 = (c2 + W) % W; }
            const j = r2 * W + c2, kj = kind[j];
            if (grid[j] || ((data.links[i] >> b) & 1 && kj !== 2)) { alone = false; break; }
        }
        if (alone) pond[i] = 1;
    }
    // nodes, water kind and current
    const info = new Int32Array(N), curr = new Int32Array(N), canoeOf = new Int32Array(N).fill(-1);
    const hasNode = new Uint8Array(N), isLand = new Uint8Array(N);
    const hzr = new Float32Array(N);   // hazard per paddling hour of each open-water cell
    let E = 0;
    for (let r = 0; r < H; r++) {
        const so = seaRow[r] * sea.W;
        for (let c = 0; c < W; c++) {
            const i = r * W + c, land = grid[i] > 0;
            let k = pond[i] ? 2 : kind[i];
            if (!land && k === 0) k = 1;            // water by the speed grid: sea
            let ok = k === 1 || k >= 3;
            if (ok && k === 1) {
                const s = so + seaCol[c];
                if (sea.ice[s] > cfg.iceMax) ok = false;
                else {
                    // 0.01 m/s, halved within 5 km of shore (shore codes 0-20 are km)
                    const m = shore[i] <= 5 ? 1 : 2;
                    curr[i] = ((sea.u[s] * m) & 0xFFFF) | ((sea.v[s] * m) << 16);
                }
            }
            if (ok) {
                const s = so + seaCol[c];
                hzr[i] = k === 1 ? waterRate(SHORE_KM[shore[i]], sea.hs[s] * 0.05, sea.sst[s])
                                 : waterRate(SHORE_KM[shore[i]], (k - 3) * 0.05, sea.lake[s]);
            }
            if (land) isLand[i] = 1;
            if (ok || onRiver[i] || (pieces && wet0[i])) { canoeOf[i] = land ? N + E++ : i; hasNode[i] = 1; }
            info[i] = (ok ? k : 0) | (LINK_LUT[data.links[i]] << 8) | (shore[i] >= 20 ? FAR : 0) |
                      (onRiver[i] || extraN[i] ? ON_RIVER : 0) | (!land || kind[i] !== 0 ? PAINT_WATER : 0);
        }
    }
    // River water no open river step crosses (T-036): the line runs in the next cell, the river's
    // steps there are closed, or a pond HydroLAKES lacks. Each such cell is a dead end: reached by
    // getting in from the shore, from a canoe node beside it, or from such a cell nearer the edge
    // of its patch (depth, in 8-neighbour steps from the nearest land or canoe node), at still-water
    // speed; nothing leaves it, so no route runs through it and no other time changes.
    const depth = new Int32Array(N);
    const nbr = (i, k) => {
        const r = (i / W) | 0, r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) return -1;
        let c2 = i - r * W + MDC[k];
        if (c2 < 0 || c2 >= W) { if (!L.wrap) return -1; c2 = (c2 + W) % W; }
        return r2 * W + c2;
    };
    let front = [];
    const dead = [];
    for (let i = 0; i < N; i++) {
        if (grid[i] || hasNode[i] || (kind[i] !== 2 && !pond[i])) continue;
        depth[i] = -1;
        for (let k = 0; k < 8; k++) {
            const j = nbr(i, k);
            if (j >= 0 && (grid[j] || hasNode[j])) { depth[i] = 1; front.push(i); break; }
        }
    }
    for (let d = 2; front.length; d++) {
        const next = [];
        for (const i of front) {
            dead.push(i);
            for (let k = 0; k < 8; k++) {
                const j = nbr(i, k);
                if (j >= 0 && depth[j] === -1) { depth[j] = d; next.push(j); }
            }
        }
        front = next;
    }
    // edges into dead-end cells j from their neighbours i: a canoe node to depth 1, depth d to d + 1
    const into = (i, j) => depth[i] > 0 ? depth[j] === depth[i] + 1 : hasNode[i] && depth[j] === 1;
    let nInto = 0;
    for (const j of dead) {
        for (let k = 0; k < 8; k++) { const i = nbr(j, k); if (i >= 0 && into(i, j)) { nInto++; info[i] |= ON_RIVER; } }
    }
    for (const j of dead) { canoeOf[j] = j; hasNode[j] = 1; info[j] = (shore[j] >= 20 ? FAR : 0) | (info[j] & ON_RIVER) | PAINT_WATER; }
    // a cell's further river nodes: extra0[cell] .. extra0[cell] + extraN[cell] - 1 (slots 1, 2, ...)
    let extra0 = null;
    const E1 = E;
    for (let i = 0; i < N; i++) {
        if (!extraN[i]) continue;
        if (!extra0) extra0 = new Int32Array(N).fill(-1);
        extra0[i] = N + E; E += extraN[i]; hasNode[i] = 1;
    }
    const nearLand = dilate3(isLand, W, H, L.wrap), nearCanoe = dilate3(hasNode, W, H, L.wrap);
    for (let i = 0; i < N; i++) info[i] |= (nearLand[i] ? NEAR_LAND : 0) | (nearCanoe[i] ? NEAR_CANOE : 0);
    const cellOf = new Int32Array(E);
    for (let i = 0; i < N; i++) {
        if (canoeOf[i] >= N) cellOf[canoeOf[i] - N] = i;
        for (let q = 0; q < extraN[i]; q++) cellOf[extra0[i] + q - N] = i;
    }
    const nodeAt = (cell, slot) => slot ? extra0[cell] + slot - 1 : canoeOf[cell];
    // access (km) of the river piece on each canoe node of a land cell (T-082): getting in or out
    // within the cell carries it
    let acc = null;
    if (anyAcc) {
        acc = new Float32Array(E);
        for (let s = 0; s < steps.n; s++) {
            if (!open[s]) continue;
            const na = nodeAt(steps.a[s], steps.sa[s]), nb = nodeAt(steps.b[s], steps.sb[s]);
            if (na >= N) acc[na - N] = steps.aa[s];
            if (nb >= N) acc[nb - N] = steps.ab[s];
        }
        if (pieces) for (let q = 0; q < pieces.n; q++) {
            const v = nodeAt(pieces.cell[q], pieces.slot[q]);
            if (v >= N) acc[v - N] = pieces.acc[q];
        }
    }
    const p = cfg.paddle;
    // water links (T-088): each end's node, kind, hazard rate and current; both ways, as sea or lake
    // grid moves between the two units
    const wl = pieces ? waterLinks(L, data, sea, pieces, info, nodeAt, seaRow, seaCol) : null;
    // river edges, by source node: down a -> b, up b -> a; then the edges into dead-end cells and
    // the water links
    const M = N + E;
    const rOff = new Int32Array(M + 1);
    for (let s = 0; s < steps.n; s++) {
        if (!open[s]) continue;
        rOff[nodeAt(steps.a[s], steps.sa[s]) + 1]++; rOff[nodeAt(steps.b[s], steps.sb[s]) + 1]++;
    }
    for (const j of dead) {
        for (let k = 0; k < 8; k++) { const i = nbr(j, k); if (i >= 0 && into(i, j)) rOff[canoeOf[i] + 1]++; }
    }
    if (wl) for (let q = 0; q < wl.n; q++) rOff[wl.from[q] + 1]++;
    for (let i = 0; i < M; i++) rOff[i + 1] += rOff[i];
    const fill = rOff.slice(0, M);
    const nE = 2 * nOpen + nInto + (wl ? wl.n : 0);
    const rTo = new Int32Array(nE), rDt = new Float32Array(nE), rCode = new Uint8Array(nE);
    const rHz = new Float32Array(nE), rKm = new Float32Array(nE);
    if (wl) for (let q = 0; q < wl.n; q++) {
        const e = fill[wl.from[q]]++;
        rTo[e] = wl.to[q]; rDt[e] = wl.dt[q]; rCode[e] = wl.code[q]; rHz[e] = wl.hz[q]; rKm[e] = wl.km[q];
        info[wl.cellFrom[q]] |= ON_RIVER;
    }
    for (const j of dead) {
        const r = (j / W) | 0;
        for (let k = 0; k < 8; k++) {
            const i = nbr(j, k);
            if (i < 0 || !into(i, j)) continue;
            const dx = MDC[k] * (L.dxKm[r] + L.dxKm[r + MDR[k]]) * 0.5, dy = MDR[k] * L.dyKm;
            const km = Math.sqrt(dx * dx + dy * dy), e = fill[canoeOf[i]]++;
            rTo[e] = j; rDt[e] = km / p; rCode[e] = P_GRID + OPP[k]; rHz[e] = rDt[e] * cfg.hz.river; rKm[e] = km;
        }
    }
    for (let s = 0; s < steps.n; s++) {
        if (!open[s]) continue;
        const a = nodeAt(steps.a[s], steps.sa[s]), b = nodeAt(steps.b[s], steps.sb[s]), m = DIR_MOVE[steps.dir[s]];
        const v = steps.v[s] * 3.6, km = steps.km[s];
        const rapids = steps.cls[s] === CLS.rapids, rate = rapids ? cfg.hz.rapids : cfg.hz.river;
        let e = fill[a]++;
        rTo[e] = b; rDt[e] = km / (p + v); rCode[e] = P_DOWN + m; rHz[e] = rDt[e] * rate; rKm[e] = km;
        e = fill[b]++;
        // up rapids the canoe is poled, lined or waded, whatever the river's size (T-073)
        rTo[e] = a; rDt[e] = km / (rapids ? cfg.upMin : Math.max(p - cfg.upShare * v, cfg.upMin)); rCode[e] = P_UP + OPP[m]; rHz[e] = rDt[e] * rate; rKm[e] = km;
    }
    // the strongest current (in curr's units) and the quickest river edge, for the search's
    // bucket width (T-093)
    let curMax = 0, rDtMin = Infinity;
    for (let i = 0; i < N; i++) {
        const cu = curr[i];
        if (!cu) continue;
        const x = (cu << 16) >> 16, y = cu >> 16, m = Math.sqrt(x * x + y * y);
        if (m > curMax) curMax = m;
    }
    for (let e = 0; e < nE; e++) if (rDt[e] < rDtMin) rDtMin = rDt[e];
    L.cn = { E, E1, canoeOf, cellOf, extra0, extraN, acc, info, curr, curKmh: sea.unit * 0.5 * 3.6, shore, hzr,
             rOff, rTo, rDt, rCode, rHz, rKm, curMax, rDtMin };
    return L.cn;
}

// The canoe nodes of a cell: its own (canoeOf, -1 if none) and its further river nodes (T-082).
// fn(node) for each; stops early when fn returns true. Returns whether it stopped.
export function eachCanoeNode(cn, cell, fn) {
    const n0 = cn.canoeOf[cell];
    if (n0 >= 0 && fn(n0)) return true;
    const k = cn.extraN[cell];
    for (let q = 0; q < k; q++) if (fn(cn.extra0[cell] + q)) return true;
    return false;
}

// The canoe node of cell `pcell` that node n was reached from, by predecessor code `code` (a river
// step, a transfer from the canoe layer, or a canoe grid move): with several canoe nodes in the cell,
// the one the search took (least weight plus the edge's, or least weight for a transfer, whose cost
// is the same from each); else the cell's own.
export function canoeFrom(L, res, n, code, pcell) {
    const cn = L.cn, N = L.W * L.H;
    if (!cn.extraN[pcell]) return cn.canoeOf[pcell];
    if (code < P_XFER) {
        // a grid move: from the cell's own node, or from any of its nodes by a water link (T-088),
        // which is the one if its cost gives exactly the node's weight
        let best = -1, bw = Infinity;
        eachCanoeNode(cn, pcell, (m) => {
            if (!(res.w[m] < Infinity)) return false;
            for (let e = cn.rOff[m], e1 = cn.rOff[m + 1]; e < e1; e++) {
                if (cn.rTo[e] !== n || cn.rCode[e] !== code) continue;
                const x = res.w[m] + cn.rDt[e] + cfg.K * (cfg.hazard ? cn.rHz[e] : 0);
                if (x < bw) { bw = x; best = m; }
            }
            return false;
        });
        return best >= 0 && Math.fround(bw) <= res.w[n] ? best : cn.canoeOf[pcell];
    }
    const river = code >= P_DOWN;
    // getting out within the cell carries the node's access, and pays the land's hazard on it, as
    // the search costs it (search.js; without the hazard a later node could be picked, T-090)
    let out = 0;
    if (code === P_XFER + 4 && cn.acc && L.grid[pcell]) {
        const iv = cfg.inv[L.grid[pcell]];
        out = cfg.carry * cfg.access * iv;
        if (cfg.hazard && L.hq) out *= 1 + cfg.K * L.hq[pcell] / iv;
    }
    let best = -1, bw = Infinity;
    eachCanoeNode(cn, pcell, (m) => {
        if (!(res.w[m] < Infinity)) return false;
        if (!river) {
            const x = res.w[m] + (out && m >= N ? out * cn.acc[m - N] : 0);
            if (x < bw) { bw = x; best = m; }
            return false;
        }
        for (let e = cn.rOff[m], e1 = cn.rOff[m + 1]; e < e1; e++) {
            if (cn.rTo[e] !== n || cn.rCode[e] !== code) continue;
            const x = res.w[m] + cn.rDt[e] + cfg.K * (cfg.hazard ? cn.rHz[e] : 0);
            if (x < bw) { bw = x; best = m; }
        }
        return false;
    });
    return best >= 0 ? best : cn.canoeOf[pcell];
}

// The shore code at a point of a level with a canoe layer, or undefined outside it.
export function shoreAt(L, lng, lat) {
    let c = Math.floor((lng - L.west) / L.cellDeg);
    if (L.wrap) c = ((c % L.W) + L.W) % L.W;
    const r = Math.floor((L.north - lat) / L.cellDeg);
    if (r < 0 || r >= L.H || c < 0 || c >= L.W) return undefined;
    return L.cn.shore[r * L.W + c];
}
