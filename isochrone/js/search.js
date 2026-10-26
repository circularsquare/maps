// Dijkstra with an indexed binary heap over the nodes of a level (docs/canoe.md 1, 5).
// Walking start: land nodes only (ids 0..N-1 = cells). Canoe start: land nodes carried at
// cfg.carry x walking time, plus canoe nodes (a water cell's id is its canoe node; a land cell's
// canoe node is canoeOf[cell] >= N), joined by transfers. Minimises weight = time + K x hazard;
// time and hazard are kept per node beside it.
import { cfg } from './config.js';

// 16 moves as flat tables: 8 neighbours, then 8 knight moves. A knight move also counts
// the speed of the two cells it passes between, which must be land too, so a one cell wide
// strait or river of water can't be hopped over. Those two cells always lie between the
// start and end rows and columns, so they need no bounds check of their own.
export const MDC = new Int32Array(16), MDR = new Int32Array(16);
const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
// for knight moves at sea: the plain moves u->e0, u->e1, e0->v, e1->v (all need a link)
const KA = new Int32Array(16), KB = new Int32Array(16), KC = new Int32Array(16), KD = new Int32Array(16);
// transfer offset j = (dc + 1) + 3 (dr + 1) -> its plain move (-1 for the cell itself)
const JK = new Int32Array(9);
export const plainMove = (dc, dr) => { for (let k = 0; k < 8; k++) if (MDC[k] === dc && MDR[k] === dr) return k; return -1; };
(function buildMoves() {
    const plain = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    let k = 0;
    for (const [dc, dr] of plain) { MDC[k] = dc; MDR[k] = dr; k++; }
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        MDC[k] = sx; MDR[k] = 2 * sy; E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        MDC[k] = 2 * sx; MDR[k] = sy; E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
    for (let k = 8; k < 16; k++) {
        KA[k] = plainMove(E0C[k], E0R[k]); KB[k] = plainMove(E1C[k], E1R[k]);
        KC[k] = plainMove(MDC[k] - E0C[k], MDR[k] - E0R[k]); KD[k] = plainMove(MDC[k] - E1C[k], MDR[k] - E1R[k]);
    }
    for (let j = 0; j < 9; j++) JK[j] = plainMove(j % 3 - 1, ((j / 3) | 0) - 1);
})();

// How each node was reached (Uint8), for tracing the route back (T-024). The predecessor cell is
// the node's cell minus the move's offset; its layer is the node's own, except for a transfer.
export const P_NONE = 0;     // the start, or not reached
export const P_GRID = 1;     // + k (0..15): a grid move of the node's own layer
export const P_XFER = 17;    // + (dc + 1) + 3 (dr + 1): a transfer from the other layer at offset (dc, dr)
export const P_DOWN = 26;    // + plain move 0..7: a river step downstream
export const P_UP = 34;      // + plain move 0..7: a river step upstream
export const P_SEED = 255;   // seeded from the coarser level at a window's border

// The canoe layer's per-cell word (cn.info): bits 0-7 water kind for grid edges (0 none, 1 sea,
// 3+ lake), 8-15 links by plain move, then these flags.
export const NEAR_LAND = 1 << 16;    // a land cell in the 3 x 3 round the cell
export const NEAR_CANOE = 1 << 17;   // a canoe node in the 3 x 3
export const FAR = 1 << 18;          // 20 km or more from shore
export const ON_RIVER = 1 << 19;     // river edges leave the cell

// Yield to the event loop so newer messages can land; much faster than setTimeout.
const chan = new MessageChannel();
const wakers = [];
chan.port1.onmessage = () => wakers.shift()();
const tick = () => new Promise(res => { wakers.push(res); chan.port2.postMessage(0); });

// Walking with no hazard: weight is time, one array, and a loop kept as small as the walking
// search always was (the general one below runs ~1.5x slower on walking alone).
async function solveWalk(L, seeds, alive) {
    const { grid, W, H, wrap, dxKm, dyKm } = L;
    const { inv, maxHours } = cfg;
    const N = W * H;
    const stepKm = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        stepKm[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
    }
    const dist = new Float32Array(N).fill(Infinity);
    const pred = new Uint8Array(N);
    const heap = new Int32Array(N);
    const pos = new Int32Array(N).fill(-1);
    let size = 0;
    function up(i) {
        const node = heap[i], d = dist[node];
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (dist[heap[p]] <= d) break;
            heap[i] = heap[p]; pos[heap[i]] = i; i = p;
        }
        heap[i] = node; pos[node] = i;
    }
    function down(i) {
        const node = heap[i], d = dist[node];
        for (;;) {
            let c = 2 * i + 1;
            if (c >= size) break;
            if (c + 1 < size && dist[heap[c + 1]] < dist[heap[c]]) c++;
            if (dist[heap[c]] >= d) break;
            heap[i] = heap[c]; pos[heap[i]] = i; i = c;
        }
        heap[i] = node; pos[node] = i;
    }
    for (let k = 0; k < seeds.idx.length; k++) {
        const v = seeds.idx[k], t = seeds.w[k];
        if (!(t < dist[v])) continue;
        dist[v] = t; pred[v] = seeds.pred ? seeds.pred[k] : P_NONE;
        if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); }
        else up(pos[v]);
    }
    let pops = 0, tLast = performance.now();
    while (size > 0) {
        if ((++pops & 0x3FFF) === 0 && performance.now() - tLast > 30) {
            await tick();
            if (!alive()) return null;
            tLast = performance.now();
        }
        const u = heap[0];
        pos[u] = -2; // settled
        size--;
        if (size > 0) { heap[0] = heap[size]; pos[heap[0]] = 0; down(0); }
        const du = dist[u];
        if (du > maxHours) break;
        const r = (u / W) | 0, c = u - r * W;
        const invU = inv[grid[u]];
        const sr = r * 16;
        for (let k = 0; k < 16; k++) {
            const r2 = r + MDR[k];
            if (r2 < 0 || r2 >= H) continue;
            let c2 = c + MDC[k];
            if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
            const v = r2 * W + c2;
            if (pos[v] === -2) continue;
            const s = grid[v];
            if (!s) continue;
            let mean;
            if (k < 8) mean = (invU + inv[s]) * 0.5;
            else {
                let c0 = c + E0C[k], c1 = c + E1C[k];
                if (wrap) { c0 = (c0 + W) % W; c1 = (c1 + W) % W; }
                const e0 = grid[(r + E0R[k]) * W + c0], e1 = grid[(r + E1R[k]) * W + c1];
                if (!e0 || !e1) continue;
                mean = (invU + inv[s] + inv[e0] + inv[e1]) * 0.25;
            }
            const nd = du + stepKm[sr + k] * mean;
            if (nd < dist[v]) {
                dist[v] = nd; pred[v] = P_GRID + k;
                if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); }
                else up(pos[v]);
            }
        }
    }
    return { w: dist, t: dist, hz: zeros(N), pred };
}
let zeroArr = new Float32Array(0);
const zeros = (n) => zeroArr.length >= n ? zeroArr.subarray(0, n) : (zeroArr = new Float32Array(n));

// Seeds: { idx, w, t, hz, pred? } (node ids and their weight, time, hazard). `canoe` searches the
// canoe layer too (L.cn must be built). Returns { w, t, hz, pred } per node, or null if alive()
// went false.
//
// Every edge's hazard is 0 until T-006: where an edge has hazard dh, its weight (nw) gains K x dh
// and the node's hazard (hu in each inlined relax) gains dh.
export async function solve(L, seeds, alive, canoe) {
    if (!canoe && !cfg.hazard) return solveWalk(L, seeds, alive);
    const { grid, W, H, wrap, dxKm, dyKm } = L;
    const cn = canoe ? L.cn : null;
    const { inv, maxHours, K, hzStop } = cfg;
    const N = W * H, M = cn ? N + cn.E : N;
    const f = cn ? cfg.carry : 1;
    // length of each move from each row, km, and its direction (east, north)
    const stepKm = new Float32Array(H * 16), ux = new Float32Array(H * 16), uy = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        const len = Math.sqrt(dx * dx + dy * dy);
        stepKm[r * 16 + k] = len; ux[r * 16 + k] = dx / len; uy[r * 16 + k] = -dy / len;
    }
    // With every hazard rate 0 (cfg.hazard off, until T-006) weight is time: t is w and hz stays 0.
    const split = cfg.hazard;
    // past this weight no node can pass both stop rules
    const wStop = split ? maxHours + K * hzStop : maxHours;
    const w = new Float32Array(M).fill(Infinity);
    const t = split ? new Float32Array(M).fill(Infinity) : w;
    const hz = new Float32Array(M);
    const pred = new Uint8Array(M);

    // Indexed heap, 4 children a node (half the depth of a binary heap, the children in one cache
    // line), each entry's weight kept beside it (hk) so sifting reads no other per-node array.
    // pos: -1 never queued, -2 settled, else the heap slot.
    const heap = new Int32Array(M), hk = new Float32Array(M);
    const pos = new Int32Array(M).fill(-1);
    let size = 0;
    function up(i, node, d) {
        while (i > 0) {
            const p = (i - 1) >> 2;
            const kp = hk[p];
            if (kp <= d) break;
            const np = heap[p];
            heap[i] = np; hk[i] = kp; pos[np] = i; i = p;
        }
        heap[i] = node; hk[i] = d; pos[node] = i;
    }
    function down(i, node, d) {
        for (;;) {
            const c0 = 4 * i + 1;
            if (c0 >= size) break;
            let c = c0, kc = hk[c0];
            const end = c0 + 4 < size ? c0 + 4 : size;
            for (let j = c0 + 1; j < end; j++) if (hk[j] < kc) { kc = hk[j]; c = j; }
            if (kc >= d) break;
            const nc = heap[c];
            heap[i] = nc; hk[i] = kc; pos[nc] = i; i = c;
        }
        heap[i] = node; hk[i] = d; pos[node] = i;
    }

    for (let k = 0; k < seeds.idx.length; k++) {
        const v = seeds.idx[k];
        if (!(seeds.w[k] < w[v])) continue;
        w[v] = seeds.w[k];
        if (split) { t[v] = seeds.t[k]; hz[v] = seeds.hz[k]; }
        pred[v] = seeds.pred ? seeds.pred[k] : P_NONE;
        up(pos[v] === -1 ? size++ : pos[v], v, w[v]);
    }

    // canoe tables
    const canoeOf = cn && cn.canoeOf, cellOf = cn && cn.cellOf, info = cn && cn.info, curr = cn && cn.curr;
    const rOff = cn && cn.rOff, rTo = cn && cn.rTo, rDt = cn && cn.rDt, rCode = cn && cn.rCode;
    const cur = cn ? cn.curKmh * 0.5 : 0;
    const p = cfg.paddle, p2 = p * p, invP = 1 / p, xfer = cfg.transfer, sea8 = cfg.sea8;

    let pops = 0, tLast = performance.now();
    while (size > 0) {
        if ((++pops & 0x3FFF) === 0 && performance.now() - tLast > 30) {
            await tick();
            if (!alive()) return null;
            tLast = performance.now();
        }
        const u = heap[0];
        pos[u] = -2;
        size--;
        if (size > 0) down(0, heap[size], hk[size]);
        const wu = w[u];
        if (wu > wStop) break;
        const tu = t[u], hu = hz[u];
        if (tu > maxHours || hu > hzStop) continue;
        const cell = u < N ? u : cellOf[u - N];
        const r = (cell / W) | 0, c = cell - r * W;
        const sr = r * 16;
        if (u < N && grid[u]) {
            // ---- land node: walk or carry ----
            const invU = inv[grid[u]];
            for (let k = 0; k < 16; k++) {
                const r2 = r + MDR[k];
                if (r2 < 0 || r2 >= H) continue;
                let c2 = c + MDC[k];
                if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
                const v = r2 * W + c2;
                if (pos[v] === -2) continue;
                const s = grid[v];
                if (!s) continue;
                let mean;
                if (k < 8) mean = (invU + inv[s]) * 0.5;
                else {
                    let c0 = c + E0C[k], c1 = c + E1C[k];
                    if (wrap) { c0 = (c0 + W) % W; c1 = (c1 + W) % W; }
                    const e0 = grid[(r + E0R[k]) * W + c0], e1 = grid[(r + E1R[k]) * W + c1];
                    if (!e0 || !e1) continue;
                    mean = (invU + inv[s] + inv[e0] + inv[e1]) * 0.25;
                }
                const dt = stepKm[sr + k] * mean * f, nw = wu + dt;
                {   // relax, inlined
                  if (nw < w[v]) {
                    w[v] = nw; pred[v] = P_GRID + k;
                    if (split) { t[v] = tu + dt; hz[v] = hu; }
                    const pv = pos[v];
                    up(pv === -1 ? size++ : pv, v, w[v]);
                  } }
            }
            if (!cn || !(info[cell] & NEAR_CANOE)) continue;
            // get in: the canoe node of this cell or a neighbour
            for (let j = 0; j < 9; j++) {
                const r2 = r + ((j / 3) | 0) - 1;
                if (r2 < 0 || r2 >= H) continue;
                let c2 = c + j % 3 - 1;
                if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
                const v = canoeOf[r2 * W + c2];
                if (v < 0 || pos[v] === -2) continue;
                const len = j === 4 ? 0 : stepKm[sr + JK[j]];
                const dt = xfer + 0.5 * len * (f * invU + invP);
                { const nw = wu + dt;   // relax, inlined
                  if (nw < w[v]) {
                    w[v] = nw; pred[v] = P_XFER + j;
                    if (split) { t[v] = tu + dt; hz[v] = hu; }
                    const pv = pos[v];
                    up(pv === -1 ? size++ : pv, v, w[v]);
                  } }
            }
            continue;
        }
        // ---- canoe node ----
        const iu = info[cell];
        if (iu & NEAR_LAND) {
            // get out: the land node of this cell or a neighbour
            for (let j = 0; j < 9; j++) {
                const r2 = r + ((j / 3) | 0) - 1;
                if (r2 < 0 || r2 >= H) continue;
                let c2 = c + j % 3 - 1;
                if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
                const v = r2 * W + c2, s = grid[v];
                if (!s || pos[v] === -2) continue;
                const len = j === 4 ? 0 : stepKm[sr + JK[j]];
                const dt = xfer + 0.5 * len * (f * inv[s] + invP);
                { const nw = wu + dt;   // relax, inlined
                  if (nw < w[v]) {
                    w[v] = nw; pred[v] = P_XFER + j;
                    if (split) { t[v] = tu + dt; hz[v] = hu; }
                    const pv = pos[v];
                    up(pv === -1 ? size++ : pv, v, w[v]);
                  } }
            }
        }
        // paddle on open water
        const k0 = iu & 0xFF;
        if (k0) {
            const lk = iu >> 8;     // link bits (and flags above them, never tested here)
            const cu = curr[cell], ucx = (cu << 16) >> 16, ucy = cu >> 16;
            const nk = sea8 && (iu & FAR) ? 8 : 16;
            for (let k = 0; k < nk; k++) {
                const r2 = r + MDR[k];
                if (r2 < 0 || r2 >= H) continue;
                let c2 = c + MDC[k];
                if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
                const cell2 = r2 * W + c2, k2 = info[cell2] & 0xFF;
                if (!k2) continue;
                const v = grid[cell2] ? canoeOf[cell2] : cell2;
                if (pos[v] === -2) continue;
                if (k < 8) { if (!(lk & (1 << k))) continue; }
                else {
                    let c0 = c + E0C[k], c1 = c + E1C[k];
                    if (wrap) { c0 = (c0 + W) % W; c1 = (c1 + W) % W; }
                    const i0 = info[(r + E0R[k]) * W + c0], i1 = info[(r + E1R[k]) * W + c1];
                    if (!(i0 & 0xFF) || !(i1 & 0xFF) || !(lk & (1 << KA[k])) || !(lk & (1 << KB[k])) ||
                        !(i0 & (256 << KC[k])) || !(i1 & (256 << KD[k]))) continue;
                }
                const len = stepKm[sr + k];
                let dt;
                if (k0 >= 3 || k2 >= 3) dt = len * invP;
                else {
                    // sea: speed along the edge with the mean current of its two cells
                    const cv = curr[cell2];
                    const mx = (ucx + ((cv << 16) >> 16)) * cur, my = (ucy + (cv >> 16)) * cur;
                    if (mx === 0 && my === 0) dt = len * invP;
                    else {
                        const ex = ux[sr + k], ey = uy[sr + k];
                        const cross = mx * ey - my * ex;
                        const q = p2 - cross * cross;
                        if (q <= 0) continue;
                        const s = mx * ex + my * ey + Math.sqrt(q);
                        if (s <= 0) continue;
                        dt = len / s;
                    }
                }
                { const nw = wu + dt;   // relax, inlined
                  if (nw < w[v]) {
                    w[v] = nw; pred[v] = P_GRID + k;
                    if (split) { t[v] = tu + dt; hz[v] = hu; }
                    const pv = pos[v];
                    up(pv === -1 ? size++ : pv, v, w[v]);
                  } }
            }
        }
        if (!(iu & ON_RIVER)) continue;
        for (let e = rOff[cell], e1 = rOff[cell + 1]; e < e1; e++) {
            const v = canoeOf[rTo[e]];
            if (pos[v] === -2) continue;
            const dt = rDt[e];
            { const nw = wu + dt;   // relax, inlined
              if (nw < w[v]) {
                w[v] = nw; pred[v] = rCode[e];
                if (split) { t[v] = tu + dt; hz[v] = hu; }
                const pv = pos[v];
                up(pv === -1 ? size++ : pv, v, w[v]);
              } }
        }
    }
    return { w, t, hz, pred };
}

// The time painted on each cell: the least of its nodes'.
export function cellTimes(L, res, canoe) {
    const N = L.W * L.H;
    if (!canoe) return res.t;
    const out = res.t.slice(0, N);
    const { cellOf, E } = L.cn;
    for (let e = 0; e < E; e++) {
        const tt = res.t[N + e], i = cellOf[e];
        if (tt < out[i]) out[i] = tt;
    }
    return out;
}
