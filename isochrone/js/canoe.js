// The canoe layer of a level (docs/canoe.md 1, 2): which cells have a canoe node, the per-cell
// word the search reads (water kind, links, flags), the river steps as out-edges, and the current.
import { cfg } from './config.js';
import { FILE_DIRS, CLOSED } from './canoedata.js';
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
// Sets and returns L.cn = { E, canoeOf, cellOf, info, curr, curKmh, shore, rOff, rTo, rDt, rCode }.
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
    const p = cfg.paddle;
    for (let s = 0; s < steps.n; s++) {
        if (!open[s]) continue;
        const a = steps.a[s], b = steps.b[s], m = DIR_MOVE[steps.dir[s]];
        const v = steps.v[s] * 3.6, km = steps.km[s];
        let e = fill[a]++;
        rTo[e] = b; rDt[e] = km / (p + v); rCode[e] = P_DOWN + m;
        e = fill[b]++;
        rTo[e] = a; rDt[e] = km / Math.max(p - 0.6 * v, cfg.upMin); rCode[e] = P_UP + OPP[m];
    }
    L.cn = { E, canoeOf, cellOf, info, curr, curKmh: sea.unit * 0.5 * 3.6, shore, rOff, rTo, rDt, rCode };
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
