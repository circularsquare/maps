// Corridors (T-046, docs/corridors.md): where many trips from the start share a route. Each
// reached land cell is one trip, weighted by its area; a node's count is the area whose route
// passes through it, summed up the shortest-path tree (the predecessor codes, with land parents
// re-chosen among near-equal ones, T-056). Edges with a count past a threshold are sent as
// polylines with a strength, log of the count.
import { cfg } from './config.js';
import { MDC, MDR, P_NONE, P_GRID, P_XFER, P_DOWN, P_UP, P_SEED } from './search.js';
import { eachCanoeNode, canoeFrom } from './canoe.js';
import { SLOPE_STEPS, NO_HEIGHT } from './levels.js';

export const STEPS = 8;      // strength steps; a polyline is a run of one step
// Speed classes (km/h) a line's colour stands for: below the first edge is class 0, at or above the
// last is class 7. A line's s byte is step | mode << 3 | class << 5; v is its km/h x 10.
export const SPEED_EDGES = [1.5, 2.5, 3.5, 4.5, 5.5, 7, 9];
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
let flow = new Float32Array(0), share = new Uint32Array(0), ord = new Int32Array(0), excess = new Float32Array(0);
function work(M) {
    if (nch.length < M) {
        nch = new Uint8Array(M); stack = new Int32Array(M); flag = new Uint8Array(M); reach = new Int32Array(M);
        flow = new Float32Array(M); share = new Uint32Array(M); ord = new Int32Array(M); excess = new Float32Array(M);
    }
}

// Re-choosing land parents (T-056, docs/corridors.md, notes/T-056.md). The search keeps one fastest
// parent per node; over open land many routes are within a fraction of a percent of each other, and
// its tree sends a whole sector of trips down one run of a single move, peeling them off in parallel
// strands. A land node may instead take a land parent that keeps its route through the tree within
// NEAR of its fastest time, and takes the one most trips would pass if each trip were spread over the
// near-equal routes (SPREAD).
const NEAR = 0.01;     // a route's weight through the tree is at most this share of time over the fastest's
const SPREAD = 0.03;   // a parent's share falls as (1 - slack / (SPREAD x time))², none past it
// a share q (of 63) back to its slack, as a share of SPREAD x time
const SLACK = Float32Array.from({ length: 64 }, (_, q) => 1 - Math.sqrt(q / 63));
// The two cells a knight move passes between (as in search.js): E1 the diagonal step, E0 the step
// along the long side.
const E0C = new Int8Array(16), E0R = new Int8Array(16), E1C = new Int8Array(16), E1R = new Int8Array(16);
for (let k = 8; k < 16; k++) {
    E1C[k] = Math.sign(MDC[k]); E1R[k] = Math.sign(MDR[k]);
    E0C[k] = Math.abs(MDC[k]) === 2 ? E1C[k] : 0; E0R[k] = Math.abs(MDR[k]) === 2 ? E1R[k] : 0;
}
// Near-equal parents lie in the directions next to the search's move (a route at a bearing between
// two moves mixes those two): for each move, the moves within WIDE places of it round the 16.
const WIDE = 1;
const NEXT = (() => {
    const byAngle = Array.from({ length: 16 }, (_, k) => k).sort((a, b) => Math.atan2(MDR[a], MDC[a]) - Math.atan2(MDR[b], MDC[b]));
    const out = new Int8Array(16 * (2 * WIDE + 1));
    for (let i = 0; i < 16; i++) for (let j = -WIDE; j <= WIDE; j++) out[byAngle[i] * (2 * WIDE + 1) + j + WIDE] = byAngle[(i + j + 16) % 16];
    return out;
})();

// The first of the coarser counted levels that holds the whole window L.
const holding = (L, coarser) => coarser && coarser.find(C => C.corr && (C.wrap ||
    (C.west <= L.west && C.east >= L.east && C.north >= L.north && C.south <= L.south)));

const lngOf = (L, cell) => L.west + (cell % L.W + 0.5) * L.cellDeg;
const latOf = (L, cell) => L.north - (((cell / L.W) | 0) + 0.5) * L.cellDeg;

// The counts of a solved level: L.corr = { par, cnt, total, nr } (par -1 a root; par and cnt
// hold only for reached nodes, those with t <= maxHours; nr of them in reach).
// coarser: the solved coarser levels of a window, whose counts give the area beyond its edge.
export function count(L, canoe, coarser) {
    const { t, pred } = L.res;
    const M = t.length, maxH = cfg.maxHours;
    const old = L.corr;
    const par = old && old.par.length >= M ? old.par : new Int32Array(M);
    const cnt = old && old.cnt.length >= M ? old.cnt : new Float32Array(M);
    const dtl = old && old.dtl.length >= M ? old.dtl : new Float32Array(M);
    work(M);
    let nr = 0;
    // reached nodes: a predecessor code and t <= maxHours, read 4 codes at a time (most are 0)
    const words = new Uint32Array(pred.buffer, pred.byteOffset, M >> 2);
    for (let i = 0; i < words.length; i++) {
        if (words[i] === 0) continue;
        for (let n = i << 2, e = n + 4; n < e; n++) if (pred[n] !== P_NONE && t[n] <= maxH) reach[nr++] = n;
    }
    for (let n = (M >> 2) << 2; n < M; n++) if (pred[n] !== P_NONE && t[n] <= maxH) reach[nr++] = n;
    // parents and shares; the start, which has no code, is found as a parent and added to reach
    const roots = [];
    const s = links(L, canoe, par, cnt, dtl, nr, roots);
    nr = s.nr;
    let total = s.total;
    // the parents of land nodes re-chosen (T-056); ord then holds every node after its children
    const C10 = coarser && coarser.find(C => C.wrap);
    const no = reparent(L, canoe, par, nr, dtl, C10 ? C10.corr.total : total);
    // nch is shared by every level and click: a node left unordered would leave it dirty and break
    // the counts of whatever node has its index next time (T-090)
    if (no !== nr) for (let i = 0; i < nr; i++) nch[reach[i]] = 0;
    // A window holding the start: which nodes' routes go back to it (org 1) and which to a border
    // seed (2). Near the border the seeds often win (a coarse time a little earlier than the fine
    // one), and downstream on a river they win far in; there the coarser level's lines are kept.
    let org = null;
    if (coarser && roots.length) {
        org = new Uint8Array(M);
        for (let i = no - 1; i >= 0; i--) {
            const n = ord[i], p = par[n];
            org[n] = p >= 0 ? org[p] : pred[n] === P_NONE ? 1 : 2;
        }
    }
    if (coarser) total += edgeFlux(L, canoe, coarser, cnt, par, org);
    for (let i = 0; i < no; i++) { const n = ord[i], p = par[n]; if (p >= 0) cnt[p] += cnt[n]; }
    L.corr = { par, cnt, total, nr, org, dtl };
    return L.corr;
}

// T-056: re-choosing the parents of land nodes reached by a land grid move (walked, or carried on a
// canoe start); every other node keeps the search's parent. A node's candidates are the land cells
// one move back in the directions next to the search's move (NEXT), reached and settled earlier
// (smaller weight, so the links never loop); the slack of one is the weight lost going through it,
// w[u] + move time + K x move hazard - w[v], costed as the search costs a land move (T-077; move(),
// directed and sloped as the search's, T-091), 0 for the search's parent.
// 1. links(), in cell order: each node's parent and own trips, and which candidates it has, so
//    each node knows how many nodes may pass trips to it (nch).
// 2. Leaves first, the trips spread: each node passes its own and its children's to its candidates
//    in proportion to (1 - slack / (SPREAD x t[v]))², or all to its one parent. A node with fewer
//    trips than LOW x the drawing threshold passes them all to the search's parent: it is too small
//    to move a drawn line, and most nodes are (T-077). The order they are passed in is kept (ord).
// 3. That order backwards (every node after its candidates), each node that spread takes the
//    candidate with the most trips among those whose route through the tree stays within NEAR x t[v]
//    of the fastest: excess, the weight a node's tree route loses, plus the slack (read back from
//    the share) is at most that. The search's parent always qualifies.
// dtl gets each node's link time: the move's own when the parent changed.
//
// share[v] packs, after 1, which candidates v has (bit j for place j in NEXT); after 2, for a node
// that spread, the candidates' shares (6 bits each, bits 0-17), their sum (18-25); and throughout
// the search's move (26-29) and whether v is 2 cells or more from the level's edges (30). 0: v
// keeps the search's parent.
const LOW = 0.02;
const SUM = 18, MOVE = 26, INNER = 1 << 30, MASK = (1 << SUM) - 1;
let offs = null, offsW = 0;    // per move: the cell offset back from v, and of the cells a knight move passes
function offsets(W) {
    if (offsW !== W) {
        offs = new Int32Array(48);
        for (let k = 0; k < 16; k++) {
            offs[k] = MDR[k] * W + MDC[k];
            offs[16 + k] = (MDR[k] - E0R[k]) * W + MDC[k] - E0C[k];
            offs[32 + k] = (MDR[k] - E1R[k]) * W + MDC[k] - E1C[k];
        }
        offsW = W;
    }
    return offs;
}
// the cell one move k back from v (row r, column c), -1 off the level
function back(L, v, r, c, k) {
    const ru = r - MDR[k];
    if (ru < 0 || ru >= L.H) return -1;
    let cu = c - MDC[k];
    if (cu < 0 || cu >= L.W) { if (!L.wrap) return -1; cu = (cu + L.W) % L.W; }
    return ru * L.W + cu;
}

function links(L, canoe, par, cnt, dtl, nr, roots) {
    const { W, H, grid, wrap, dxKm, dyKm } = L;
    const N = W * H, NW = 2 * WIDE + 1;
    const { t, w, pred } = L.res;
    const maxH = cfg.maxHours;
    const canoeOf = canoe ? L.cn.canoeOf : null, cellOf = canoe ? L.cn.cellOf : null;
    const extraN = canoe ? L.cn.extraN : null;
    const off = offsets(W);
    let total = 0;
    for (let i = 0; i < nr; i++) {
        const v = reach[i];
        const cell = v < N ? v : cellOf[v - N];
        const r = (cell / W) | 0;
        const gv = v < N ? grid[v] : 0;
        const own = gv ? dxKm[r] * dyKm : 0;     // trips end on land
        cnt[v] = own; flow[v] = own; total += own; share[v] = 0;
        // the parent, inlined
        const code = pred[v];
        let p = -1;
        if (PARENT[code]) {
            let pc = cell - r * W - DC[code];
            if (pc < 0 || pc >= W) pc = wrap ? (pc + W) % W : pc;
            const pcell = (r - DR[code]) * W + pc;
            if (!(FLIP[code] ? gv !== 0 : gv === 0)) p = pcell;
            else if (!(extraN && extraN[pcell])) p = canoeOf[pcell];
            else {
                // a parent is reached no later and at no more weight than its child. One read back
                // wrong among a cell's canoe nodes could be unreached or past the time limit:
                // outside the counted tree, it would leave nch dirty for every later count (T-090).
                // Then the cell's node that fits.
                p = canoeFrom(L, L.res, v, code, pcell);
                if (p >= 0 && !(t[p] <= t[v] && w[p] <= w[v])) p = earliest(L.cn, pcell, L.res, v);
            }
        }
        par[v] = p;
        if (p < 0) { dtl[v] = 0; continue; }
        dtl[v] = t[v] - t[p];
        if (pred[p] === P_NONE && !roots.includes(p)) {
            roots.push(p);
            reach[nr++] = p;      // its own turn in this loop sets its par and cnt
        }
        nch[p]++;
        if (!gv || code < P_GRID || code >= P_GRID + 16) continue;
        // candidates other than the search's parent (place WIDE in NEXT)
        const wv = w[v], c = v - r * W;
        const kv = code - P_GRID, k0 = kv * NW;
        const inner = r >= 2 && r < H - 2 && c >= 2 && c < W - 2;
        let m = 0;
        for (let j = 0; j < NW; j++) {
            if (j === WIDE) continue;
            const u = inner ? v - off[NEXT[k0 + j]] : back(L, v, r, c, NEXT[k0 + j]);
            if (u < 0 || !grid[u] || !(w[u] < wv) || !(t[u] <= maxH)) continue;
            m |= 1 << j;
            nch[u]++;
            // the start as a candidate is counted like a root (a window's start can have no children)
            if (pred[u] === P_NONE && !roots.includes(u)) { roots.push(u); reach[nr++] = u; }
        }
        if (m) share[v] = m | kv << MOVE | (inner ? INNER : 0);
    }
    return { total, nr };
}

// The canoe node of a cell with the least weight among those reached no later and at no more
// weight than node v, or -1.
function earliest(cn, cell, res, v) {
    const { t, w } = res;
    let best = -1;
    eachCanoeNode(cn, cell, (m) => {
        if (t[m] <= t[v] && w[m] <= w[v] && (best < 0 || w[m] < w[best])) best = m;
        return false;
    });
    return best;
}

// world: the area reached at L10, whose share SHARE is the drawing threshold. Returns how many
// nodes it ordered (all nr of them when the links form a tree).
function reparent(L, canoe, par, nr, dtl, world) {
    const { W } = L;
    const { t, w } = L.res;
    const f = canoe ? cfg.carry : 1, K = cfg.K;
    // hazard per km by cell, as the search reads it (none with hazard off or no band)
    const hq = cfg.hazard && L.hq ? L.hq : null;
    const NW = 2 * WIDE + 1;
    const off = offsets(W);
    const low = LOW * SHARE * world;
    const us = new Int32Array(NW);
    // 2. spread, leaves first
    let sp = 0, no = 0;
    for (let i = 0; i < nr; i++) { const n = reach[i]; if (nch[n] === 0) stack[sp++] = n; }
    while (sp > 0) {
        const v = stack[--sp], p = par[v];
        ord[no++] = v;
        if (p < 0) continue;
        const sh = share[v], fv = flow[v];
        if (sh === 0) { flow[p] += fv; if (--nch[p] === 0) stack[sp++] = p; continue; }
        const k0 = (sh >>> MOVE & 15) * NW, inner = sh & INNER;
        if (fv < low && inner) {
            flow[p] += fv;
            share[v] = 0;
            for (let j = 0; j < NW; j++) {
                if (j !== WIDE && !(sh & (1 << j))) continue;
                const u = j === WIDE ? p : v - off[NEXT[k0 + j]];
                if (--nch[u] === 0) stack[sp++] = u;
            }
            continue;
        }
        const r = (v / W) | 0, c = v - r * W;
        for (let j = 0; j < NW; j++) {
            if (j === WIDE) { us[j] = p; continue; }
            if (!(sh & (1 << j))) { us[j] = -1; continue; }
            const k = NEXT[k0 + j];
            us[j] = inner ? v - off[k] : back(L, v, r, c, k);
        }
        let qs = 63 << (6 * WIDE), sum = 63;
        if (fv >= low) {
            // the shares: each candidate's move costed as the search costs it
            const wv = w[v];
            const lim = SPREAD * t[v], isc = 1 / lim;
            for (let j = 0; j < NW; j++) {
                const u = us[j];
                if (j === WIDE || u < 0 || !move(L, u, v, NEXT[k0 + j], f, hq)) continue;
                const slack = w[u] + mvT + K * mvH - wv;
                let q = 63;
                if (slack > 0) {
                    if (slack >= lim) continue;
                    const e = 1 - slack * isc;
                    q = ((63 * e * e + 0.5) | 0) || 1;
                }
                qs |= q << (6 * j); sum += q;
            }
        }
        if (sum === 63) {
            share[v] = 0;
            flow[p] += fv;
        } else {
            share[v] = qs | sum << SUM | (sh & ~MASK);
            const q = fv / sum;
            for (let j = 0; j < NW; j++) { const s = (qs >>> (6 * j)) & 63; if (s) flow[us[j]] += q * s; }
        }
        for (let j = 0; j < NW; j++) { const u = us[j]; if (u >= 0 && --nch[u] === 0) stack[sp++] = u; }
    }
    // 3. choose, parents first
    for (let i = no - 1; i >= 0; i--) {
        const v = ord[i], p = par[v];
        if (p < 0) { excess[v] = 0; continue; }
        const sh = share[v];
        if (sh === 0) { excess[v] = excess[p]; continue; }
        const tv = t[v], k0 = (sh >>> MOVE & 15) * NW, inner = sh & INNER;
        const lim = SPREAD * tv, room = NEAR * tv;
        let best = p, bf = flow[p], bs = 0, bk = 0;
        for (let j = 0; j < NW; j++) {
            const q = (sh >>> (6 * j)) & 63;
            if (!q || j === WIDE) continue;
            const k = NEXT[k0 + j];
            let u;
            if (inner) u = v - off[k];
            else { const r = (v / W) | 0; u = back(L, v, r, v - r * W, k); }
            const slack = lim * SLACK[q];
            if (!(flow[u] > bf) || excess[u] + slack > room) continue;
            best = u; bf = flow[u]; bs = slack; bk = k;
        }
        if (best === p) { excess[v] = excess[p]; continue; }
        excess[v] = excess[best] + bs;
        par[v] = best;
        move(L, best, v, bk, f, null);
        dtl[v] = mvT;
    }
    return no;
}

// Land moves costed as js/search.js's land loops cost them (T-091): the mean of the cells' hours
// per km, directed by L.sd where the level has it (T-011, docs/resolution.md "Directions"), times
// the slope's factor at L0.1 (L.zs, docs/fine.md "Terrain"), carried at f; hazard per km stays the
// cells' own. The search inlines this in its loops; checks/corridors.mjs holds the two together
// (every node's own move, costed here, must give its weight).
// Per level and from the search: move lengths, km, and the east-west share of each move's km, by
// the row moved from; which of a cell's four directed values (E W S N) a move's parts use.
const DXK = Int8Array.from({ length: 16 }, (_, k) => MDC[k] > 0 ? 0 : 1);
const DYK = Int8Array.from({ length: 16 }, (_, k) => MDR[k] > 0 ? 2 : 3);
function moveTables(L) {
    if (L.wx16) return;
    const { H, dxKm, dyKm } = L;
    const sk = new Float32Array(H * 16), wx = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        sk[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
        const ax = Math.abs(dx), ay = Math.abs(dy);
        wx[r * 16 + k] = ax / (ax + ay);
    }
    L.stepKm16 = sk; L.wx16 = wx;
    if (L.zs) L.iStep16 = sk.map((s) => s > 0 ? SLOPE_STEPS / (1000 * s) : 0);
}
let mvT = 0, mvH = 0;     // the last move's time and hazard
// Land move k from u into v: false where the search does not make it (a knight move past water,
// too steep), else mvT and mvH set (hq: hazard per km by cell, or null for none).
function move(L, u, v, k, f, hq) {
    const { W, grid } = L, inv = cfg.inv, sd = L.sd;
    moveTables(L);
    const ru = (u / W) | 0, cu = u - ru * W, sr = ru * 16;
    let i0 = 0, i1 = 0, mean, mq = 0;
    if (k >= 8) {
        i0 = (ru + E0R[k]) * W + (cu + E0C[k] + W) % W; i1 = (ru + E1R[k]) * W + (cu + E1C[k] + W) % W;
        if (!grid[i0] || !grid[i1]) return false;
    }
    if (sd) {
        const a = L.wx16[sr + k], b = 1 - a, X = DXK[k], Y = DYK[k];
        mean = a * (sd[4 * u + X] + sd[4 * v + X]) + b * (sd[4 * u + Y] + sd[4 * v + Y]);
        if (k < 8) mean *= 0.5;
        else mean = (mean + a * (sd[4 * i0 + X] + sd[4 * i1 + X]) + b * (sd[4 * i0 + Y] + sd[4 * i1 + Y])) * 0.25;
    } else if (k < 8) mean = (inv[grid[u]] + inv[grid[v]]) * 0.5;
    else mean = (inv[grid[u]] + inv[grid[v]] + inv[grid[i0]] + inv[grid[i1]]) * 0.25;
    if (hq) mq = k < 8 ? (hq[u] + hq[v]) * 0.5 : (hq[u] + hq[v] + hq[i0] + hq[i1]) * 0.25;
    const zs = L.zs;
    if (zs && zs[u] !== NO_HEIGHT && zs[v] !== NO_HEIGHT) {
        const sG = L.slopeG, rq = L.roadQ;
        let q = (zs[v] - zs[u]) * L.iStep16[sr + k];
        if (L.zg[u] & L.zg[v]) q = q > rq ? rq : q < -rq ? -rq : q;
        else if (q > sG || q < -sG) return false;
        const sG5 = sG + 0.5;
        mean *= L.slopeT[(q + sG5) | 0] * L.zh[u] * L.zh[v];
    }
    const len = L.stepKm16[sr + k] * f;
    mvT = len * mean; mvH = len * mq;
    return true;
}

// For checks: the weight move k from u into v adds on a solved level, as reparent costs it, or NaN.
export function moveWeight(L, canoe, u, v, k) {
    const hq = cfg.hazard && L.hq ? L.hq : null;
    return move(L, u, v, k, canoe ? cfg.carry : 1, hq) ? mvT + cfg.K * mvH : NaN;
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
    const ccellOf = canoe ? C.cn.cellOf : null;
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
            if (canoe) eachCanoeNode(C.cn, cell, (m) => { if (ct[m] <= maxH) visit(m, cell); });
        }
    }
    return added;
}

// The river length of the step from node p to node n reached by `code` in time d (river edges are
// by node, js/canoe.js; the one whose time matches is the one taken).
function riverKm(cn, p, n, code, d) {
    let best = -1, bd = Infinity;
    for (let e = cn.rOff[p], e1 = cn.rOff[p + 1]; e < e1; e++) {
        if (cn.rTo[e] !== n || cn.rCode[e] !== code) continue;
        const x = Math.abs(cn.rDt[e] - d);
        if (x < bd) { bd = x; best = e; }
    }
    return best >= 0 ? cn.rKm[best] : -1;
}

// Each kept edge's speed (child n to its parent), km/h, as the search costs it (time only, no
// hazard): the length travelled over the time taken, summed over the edge and up to SMOOTH edges
// of its mode each way along the route (up towards the start, down along the child with the most
// trips). A river step travels the river's own length, meanders included, so a reach reads its
// model speed; the window evens out where a route hops between rivers sharing cells (T-064). A
// transfer to a neighbouring cell is carried over the step, its 15 minutes included; one within a
// cell travels nothing and takes the speed of the edges round it. Returns get(n) and dt(n), the
// edge's own time (0 for a transfer within a cell).
const SMOOTH = 6;
// km/h past a class edge before a line is cut for speed (lines())
const HYST = 0.4;
function edgeSpeeds(L, canoe, nk, kept, mode) {
    const { par, cnt, dtl } = L.corr;
    const { pred } = L.res;
    const W = L.W, N = W * L.H, cn = canoe ? L.cn : null;
    const idx = new Map();
    for (let i = 0; i < nk; i++) idx.set(stack[i], i);
    const km = new Float32Array(nk), dt = new Float32Array(nk), md = new Uint8Array(nk);
    const up = new Int32Array(nk).fill(-1), down = new Int32Array(nk).fill(-1);
    for (let i = 0; i < nk; i++) {
        const n = stack[i], p = par[n], code = pred[n];
        md[i] = mode(n);
        if (kept(p)) up[i] = idx.get(p);
        const d = dtl[n];
        if (!(d > 0) || (FLIP[code] && DC[code] === 0 && DR[code] === 0)) continue;
        const cell = n < N ? n : cn.cellOf[n - N], r = (cell / W) | 0;
        let len = -1;
        if (code >= P_DOWN && p >= 0) len = riverKm(cn, p, n, code, d);
        if (len < 0) {
            // a land move's offset from the cells, as its parent may not be the search's (reparent)
            let dc = DC[code], dr = DR[code];
            if (code < P_XFER && p < N) { const pr = (p / W) | 0; dr = r - pr; dc = cell - r * W - (p - pr * W); if (dc > 2) dc -= W; else if (dc < -2) dc += W; }
            len = Math.hypot(dc * (L.dxKm[r] + L.dxKm[r - dr]) * 0.5, dr * L.dyKm);
        }
        km[i] = len; dt[i] = d;
    }
    for (let i = 0; i < nk; i++) {
        const u = up[i];
        if (u >= 0 && (down[u] < 0 || cnt[stack[i]] > cnt[stack[down[u]]])) down[u] = i;
    }
    // sums over the window round edge i; past it, when it holds no time (transfers only), on until
    // some (of the same mode first)
    const v = new Float32Array(nk);
    for (let i = 0; i < nk; i++) {
        let k = km[i], d = dt[i];
        const m = md[i];
        let j = up[i], s = 0;
        for (; j >= 0 && md[j] === m && (s < SMOOTH || d === 0); j = up[j], s++) { k += km[j]; d += dt[j]; }
        for (j = down[i], s = 0; j >= 0 && md[j] === m && (s < SMOOTH || d === 0); j = down[j], s++) { k += km[j]; d += dt[j]; }
        for (j = up[i]; d === 0 && j >= 0; j = up[j]) { k += km[j]; d += dt[j]; }
        for (j = down[i]; d === 0 && j >= 0; j = down[j]) { k += km[j]; d += dt[j]; }
        v[i] = d > 0 ? k / d : 0;
    }
    return { get: (n) => v[idx.get(n)], dt: (n) => dt[idx.get(n)] };
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
    // the mode of the edge from n: walked 0, carried 1 (on the land layer of a canoe start, and
    // getting in or out, as the hover's split counts it), paddled 2 (on the canoe layer)
    // (a water cell's canoe node is the cell's own index, so n < N does not mean the land layer)
    const { pred } = L.res, grid = L.grid;
    const mode = (n) => canoe ? (!FLIP[pred[n]] && (n >= N || grid[n] === 0) ? 2 : 1) : 0;
    const cellOfNode = (n) => n < N ? n : cellOf[n - N];
    const speed = edgeSpeeds(L, canoe, nk, kept, mode);
    const clsOf = (v) => {
        let k = 0;
        while (k < SPEED_EDGES.length && v >= SPEED_EDGES[k]) k++;
        return k;
    };
    // an edge can continue the line of the edge before it when that is the only kept edge on, of
    // one strength step and one mode
    const joins = (n) => { const p = par[n]; return nch[p] === 1 && kept(p) && step(p) === step(n) && mode(p) === mode(n); };
    // where lines are cut for speed: each edge's class label, outward from the start; an edge that
    // joins keeps the label before it while its speed is within HYST km/h of that class
    const lab = new Map(), chain = [];
    const label = (n) => {
        chain.length = 0;
        for (let a = n; !lab.has(a); a = par[a]) { chain.push(a); if (!joins(a)) break; }
        for (let i = chain.length - 1; i >= 0; i--) {
            const x = chain[i], v = speed.get(x);
            let c = clsOf(v);
            if (joins(x)) {
                const q = lab.get(par[x]);
                const lo = q > 0 ? SPEED_EDGES[q - 1] : -Infinity, hi = q < SPEED_EDGES.length ? SPEED_EDGES[q] : Infinity;
                if (v >= lo - HYST && v < hi + HYST) c = q;
            }
            lab.set(x, c);
        }
        return lab.get(n);
    };
    const cont = (n) => joins(n) && label(par[n]) === label(n);
    for (let i = 0; i < nk; i++) { const n = stack[i]; if (cont(n)) flag[par[n]] = 1; }
    const xy = [], off = [0], s = [], vs = [];
    for (let i = 0; i < nk; i++) {
        const n = stack[i];
        if (flag[n]) continue;
        // from a line's tip up to where it joins; its km/h is the mean of its edges' speeds, each
        // weighted by its time, and its colour that km/h's class
        let vSum = 0, wSum = 0;
        let cur = n, cell = cellOfNode(n);
        let x = lngOf(L, cell), y = latOf(L, cell);
        const pts = [x, y];
        for (;;) {
            const p = par[cur];
            { const w = speed.dt(cur) + 1e-3; vSum += speed.get(cur) * w; wSum += w; }
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
        const v10 = Math.min(255, Math.round(vSum / wSum * 10));
        if (pts.length < 4 || !(v10 > 0)) continue;
        for (const v of pts) xy.push(v);
        off.push(xy.length / 2);
        s.push(step(n) | (mode(n) << 3) | (clsOf(v10 / 10) << 5));
        vs.push(v10);
    }
    for (let i = 0; i < nk; i++) { const n = stack[i]; nch[par[n]] = 0; flag[par[n]] = 0; }
    if (org && P && P.drawn) coarserLines(L, canoe, P.drawn, P.cellDeg, xy, off, s, vs);
    const out = { xy: new Float32Array(xy), off: new Uint32Array(off), s: new Uint8Array(s), v: new Uint8Array(vs) };
    L.drawn = { xy: out.xy.slice(), off: out.off.slice(), s: out.s.slice(), v: out.v.slice() };
    return out;
}

// The coarser level's lines inside a window where the window's own routes do not go back to the
// start (seeded from its border, or not reached): cut in pieces about a coarse cell long, each
// kept when the window cell under its middle has no node routed from the start.
function coarserLines(L, canoe, D, step, xy, off, s, vs) {
    const { W, H, grid } = L;
    const org = L.corr.org, t = L.res.t, maxH = cfg.maxHours;
    const covered = (x, y) => {
        const c = Math.floor((((x - L.west) % 360) + 360) % 360 / L.cellDeg), r = Math.floor((L.north - y) / L.cellDeg);
        if (r < 0 || r >= H || c >= W) return null;     // outside: drawn by the coarser level itself
        const cell = r * W + c;
        if (grid[cell] && t[cell] <= maxH && org[cell] === 1) return true;
        return canoe && eachCanoeNode(L.cn, cell, (m) => t[m] <= maxH && org[m] === 1);
    };
    for (let i = 0; i + 1 < D.off.length; i++) {
        let run = null;
        const end = () => { if (run && run.length >= 4) { for (const v of run) xy.push(v); off.push(xy.length / 2); s.push(D.s[i]); vs.push(D.v[i]); } run = null; };
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
