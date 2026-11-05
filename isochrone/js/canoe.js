// The canoe layer of a level (docs/canoe.md 1, 2): which cells have a canoe node, the per-cell
// word the search reads (water kind, links, flags), the river steps as out-edges, and the current.
import { cfg } from './config.js';
import { FILE_DIRS, CLOSED, CLS } from './canoedata.js';
import { plainMove, MDC, MDR, P_GRID, P_DOWN, P_UP, NEAR_LAND, NEAR_CANOE, FAR, ON_RIVER } from './search.js';

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
// Hs (m) and sea temperature (C): exposure x (water + waves(Hs) x min(d / waveKm, 1) x cold +
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

// A cell's `info` bit set where the cell is water of any kind, river water included, whether or not
// it has a canoe node: what paint.js mixes towards white (T-063, T-074).
export const PAINT_WATER = 1 << 20;

// data: { kind, shore, links, steps } from canoedata.js; sea: the 0.25 degree sea grid.
// Sets and returns L.cn = { E, canoeOf, cellOf, info, curr, curKmh, shore, hzr, rOff, rTo, rDt,
// rCode, rHz, rKm (the step's river length, km) }.
export function buildCanoe(L, data, sea) {
    const { grid, W, H } = L;
    const N = W * H;
    const { kind, shore, steps } = data;
    // sea grid row and column of each cell row and column
    const seaRow = new Int32Array(H), seaCol = new Int32Array(W);
    for (let r = 0; r < H; r++)
        seaRow[r] = Math.min(sea.H - 1, Math.max(0, Math.floor((sea.north - (L.north - (r + 0.5) * L.cellDeg)) / sea.deg)));
    for (let c = 0; c < W; c++) {
        const x = Math.floor((L.west + (c + 0.5) * L.cellDeg - sea.west) / sea.deg);
        seaCol[c] = ((x % sea.W) + sea.W) % sea.W;
    }
    // open river steps
    const open = new Uint8Array(steps.n);
    const onRiver = new Uint8Array(N);
    let nOpen = 0;
    for (let s = 0; s < steps.n; s++) {
        if (CLOSED[steps.cls[s]] || DIR_MOVE[steps.dir[s]] < 0) continue;
        open[s] = 1; nOpen++;
        onRiver[steps.a[s]] = 1; onRiver[steps.b[s]] = 1;
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
            let k = kind[i];
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
                hzr[i] = waterRate(SHORE_KM[shore[i]], k === 1 ? sea.hs[s] * 0.05 : (k - 3) * 0.05, sea.sst[s]);
            }
            if (land) isLand[i] = 1;
            if (ok || onRiver[i]) { canoeOf[i] = land ? N + E++ : i; hasNode[i] = 1; }
            info[i] = (ok ? k : 0) | (LINK_LUT[data.links[i]] << 8) | (shore[i] >= 20 ? FAR : 0) |
                      (onRiver[i] ? ON_RIVER : 0) | (!land || kind[i] !== 0 ? PAINT_WATER : 0);
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
        if (grid[i] || hasNode[i] || kind[i] !== 2) continue;
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
    const nearLand = dilate3(isLand, W, H, L.wrap), nearCanoe = dilate3(hasNode, W, H, L.wrap);
    for (let i = 0; i < N; i++) info[i] |= (nearLand[i] ? NEAR_LAND : 0) | (nearCanoe[i] ? NEAR_CANOE : 0);
    const cellOf = new Int32Array(E);
    for (let i = 0; i < N; i++) if (canoeOf[i] >= N) cellOf[canoeOf[i] - N] = i;
    // river edges, by source cell: down a -> b, up b -> a; then the edges into dead-end cells
    const rOff = new Int32Array(N + 1);
    for (let s = 0; s < steps.n; s++) if (open[s]) { rOff[steps.a[s] + 1]++; rOff[steps.b[s] + 1]++; }
    for (const j of dead) {
        for (let k = 0; k < 8; k++) { const i = nbr(j, k); if (i >= 0 && into(i, j)) rOff[i + 1]++; }
    }
    for (let i = 0; i < N; i++) rOff[i + 1] += rOff[i];
    const fill = rOff.slice(0, N);
    const nE = 2 * nOpen + nInto;
    const rTo = new Int32Array(nE), rDt = new Float32Array(nE), rCode = new Uint8Array(nE);
    const rHz = new Float32Array(nE), rKm = new Float32Array(nE);
    const p = cfg.paddle;
    for (const j of dead) {
        const r = (j / W) | 0;
        for (let k = 0; k < 8; k++) {
            const i = nbr(j, k);
            if (i < 0 || !into(i, j)) continue;
            const dx = MDC[k] * (L.dxKm[r] + L.dxKm[r + MDR[k]]) * 0.5, dy = MDR[k] * L.dyKm;
            const km = Math.sqrt(dx * dx + dy * dy), e = fill[i]++;
            rTo[e] = j; rDt[e] = km / p; rCode[e] = P_GRID + OPP[k]; rHz[e] = rDt[e] * cfg.hz.river; rKm[e] = km;
        }
    }
    for (let s = 0; s < steps.n; s++) {
        if (!open[s]) continue;
        const a = steps.a[s], b = steps.b[s], m = DIR_MOVE[steps.dir[s]];
        const v = steps.v[s] * 3.6, km = steps.km[s];
        const rate = steps.cls[s] === CLS.rapids ? cfg.hz.rapids : cfg.hz.river;
        let e = fill[a]++;
        rTo[e] = b; rDt[e] = km / (p + v); rCode[e] = P_DOWN + m; rHz[e] = rDt[e] * rate; rKm[e] = km;
        e = fill[b]++;
        rTo[e] = a; rDt[e] = km / Math.max(p - cfg.upShare * v, cfg.upMin); rCode[e] = P_UP + OPP[m]; rHz[e] = rDt[e] * rate; rKm[e] = km;
    }
    L.cn = { E, canoeOf, cellOf, info, curr, curKmh: sea.unit * 0.5 * 3.6, shore, hzr, rOff, rTo, rDt, rCode, rHz, rKm };
    return L.cn;
}

// The shore code at a point of a level with a canoe layer, or undefined outside it.
export function shoreAt(L, lng, lat) {
    let c = Math.floor((lng - L.west) / L.cellDeg);
    if (L.wrap) c = ((c % L.W) + L.W) % L.W;
    const r = Math.floor((L.north - lat) / L.cellDeg);
    if (r < 0 || r >= L.H || c < 0 || c >= L.W) return undefined;
    return L.cn.shore[r * L.W + c];
}
