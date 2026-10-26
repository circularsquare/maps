// The canoe layer of a level (docs/canoe.md 1, 2): which cells have a canoe node, the per-cell
// word the search reads (water kind, links, flags), the river steps as out-edges, and the current.
import { cfg } from './config.js';
import { FILE_DIRS, CLOSED, CLS } from './canoedata.js';
import { plainMove, P_DOWN, P_UP, NEAR_LAND, NEAR_CANOE, FAR, ON_RIVER } from './search.js';

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

// data: { kind, shore, links, steps } from canoedata.js; sea: the 0.25 degree sea grid.
// Sets and returns L.cn = { E, canoeOf, cellOf, info, curr, curKmh, shore, hzr, rOff, rTo, rDt,
// rCode, rHz }.
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
                      (onRiver[i] ? ON_RIVER : 0);
        }
    }
    const nearLand = dilate3(isLand, W, H, L.wrap), nearCanoe = dilate3(hasNode, W, H, L.wrap);
    for (let i = 0; i < N; i++) info[i] |= (nearLand[i] ? NEAR_LAND : 0) | (nearCanoe[i] ? NEAR_CANOE : 0);
    const cellOf = new Int32Array(E);
    for (let i = 0; i < N; i++) if (canoeOf[i] >= N) cellOf[canoeOf[i] - N] = i;
    // river edges, by source cell: down a -> b, up b -> a
    const rOff = new Int32Array(N + 1);
    for (let s = 0; s < steps.n; s++) if (open[s]) { rOff[steps.a[s] + 1]++; rOff[steps.b[s] + 1]++; }
    for (let i = 0; i < N; i++) rOff[i + 1] += rOff[i];
    const fill = rOff.slice(0, N);
    const rTo = new Int32Array(2 * nOpen), rDt = new Float32Array(2 * nOpen), rCode = new Uint8Array(2 * nOpen);
    const rHz = new Float32Array(2 * nOpen);
    const p = cfg.paddle;
    for (let s = 0; s < steps.n; s++) {
        if (!open[s]) continue;
        const a = steps.a[s], b = steps.b[s], m = DIR_MOVE[steps.dir[s]];
        const v = steps.v[s] * 3.6, km = steps.km[s];
        const rate = steps.cls[s] === CLS.rapids ? cfg.hz.rapids : cfg.hz.river;
        let e = fill[a]++;
        rTo[e] = b; rDt[e] = km / (p + v); rCode[e] = P_DOWN + m; rHz[e] = rDt[e] * rate;
        e = fill[b]++;
        rTo[e] = a; rDt[e] = km / Math.max(p - cfg.upShare * v, cfg.upMin); rCode[e] = P_UP + OPP[m]; rHz[e] = rDt[e] * rate;
    }
    L.cn = { E, canoeOf, cellOf, info, curr, curKmh: sea.unit * 0.5 * 3.6, shore, hzr, rOff, rTo, rDt, rCode, rHz };
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
