// A level is a block of cells with its own grid and, once solved, its dist. L10 is the whole
// world and wraps at 180 degrees; a window of L3 or L1 does not.
import { cfg } from './config.js';

const KM_PER_DEG = 111.195;

// Land hazard code c (1..240) -> rate per walking hour, 10^((c - 240) / 40); 0 none (docs/hazard.md 4).
export const HZ_RATE = new Float32Array(256);
for (let c = 1; c <= 240; c++) HZ_RATE[c] = Math.pow(10, (c - 240) / 40);

// L.hq: each land cell's hazard per km walked, rate x hours per km (0 on water), from the level's
// band of codes.
export function setHazard(L, codes) {
    const { grid } = L, inv = cfg.inv, n = grid.length;
    const hq = new Float32Array(n);
    for (let i = 0; i < n; i++) { const c = codes[i], s = grid[i]; if (c && s) hq[i] = HZ_RATE[c] * inv[s]; }
    L.hq = hq;
}

// exp(-b / 32) for a direction byte b, read as int8 (tools/build_tiles.py DSCALE)
const NEG = new Float32Array(256);
for (let b = 0; b < 256; b++) NEG[b] = Math.exp(-((b << 24) >> 24) / 32);

// L.sd: each land cell's hours per km going east, west, south and north (4 a cell, interleaved),
// from its speed (roads folded in) and its direction planes: d (no roads) and e (with roads, on
// top of d), each 4 planes of the level's cells one after another, either may be null
// (docs/resolution.md "Directions"). Null when both are absent or all 0, so the search is the
// symmetric one.
export function setDirs(L, d, e) {
    L.sd = null;
    if (!d && !e) return;
    const { grid } = L, inv = cfg.inv, n = grid.length;
    const sd = new Float32Array(4 * n);
    let any = false, min = Infinity;
    for (let j = 0; j < 4; j++) {
        const dj = d ? d.subarray(j * n, (j + 1) * n) : null, ej = e ? e.subarray(j * n, (j + 1) * n) : null;
        for (let i = 0; i < n; i++) {
            const s = grid[i];
            if (!s) continue;
            const a = dj ? dj[i] : 0, b = ej ? ej[i] : 0;
            if (a | b) any = true;
            sd[4 * i + j] = inv[s] * NEG[a] * NEG[b];
            if (sd[4 * i + j] < min) min = sd[4 * i + j];
        }
    }
    // the least, for the search's bucket width (T-093)
    if (any) { L.sd = sd; L.sdMin = min; }
}

// Slope per move at L0.1 (docs/fine.md "Terrain"): the rule of tools/terrain/walk.py, its numbers
// from the fine index (walk.py SLOPE_RULE). z: the window's heights (Int16, metres, NO_HEIGHT where
// none), g: graded cells (1: urban, a walkable way or bridge). Sets L.zs, L.zg, L.zh (per cell
// exp(-ln height factor / 2): a move's height factor is at the mean of its end cells' heights, the
// product of theirs as it is exponential) and L.slopeT (the cost factor 1 / Tobler on grade x
// SLOPE_STEPS, offset by SLOPE_STEPS x gMax); js/search.js reads them.
export const SLOPE_STEPS = 1000, NO_HEIGHT = -32768;
export function setSlope(L, z, g, rule) {
    const n = z.length, zh = new Float32Array(n).fill(1);
    const la = -Math.log(rule.heightA), b = rule.heightB;
    let zMin = 1, tMin = Infinity;
    for (let i = 0; i < n; i++) {
        const h = z[i];
        if (h !== NO_HEIGHT) { zh[i] = Math.exp(0.5 * (la + b * Math.max(h, 0))); if (zh[i] < zMin) zMin = zh[i]; }
    }
    const G = Math.round(rule.gMax * SLOPE_STEPS), T = new Float32Array(2 * G + 1);
    for (let i = 0; i <= 2 * G; i++) {
        T[i] = Math.exp(rule.toblerK * (Math.abs((i - G) / SLOPE_STEPS + rule.toblerG0) - rule.toblerG0));
        if (T[i] < tMin) tMin = T[i];
    }
    L.zs = z; L.zg = g; L.zh = zh; L.slopeT = T;
    L.slopeG = G; L.roadQ = rule.roadGrade * SLOPE_STEPS;
    // the least factor a move's slope can bring, for the search's bucket width (T-093)
    L.slopeMin = Math.min(1, tMin * zMin * zMin);
}

// A level: a block of cells cellDeg wide, its top-left corner at (west, north).
export function makeLevel(grid, W, H, west, north, cellDeg, wrap) {
    const dxKm = new Float32Array(H);
    for (let r = 0; r < H; r++) {
        const lat = north - (r + 0.5) * cellDeg;
        dxKm[r] = cellDeg * KM_PER_DEG * Math.cos(lat * Math.PI / 180);
    }
    return { grid, W, H, west, north, south: north - H * cellDeg, east: west + W * cellDeg,
             cellDeg, wrap, dxKm, dyKm: cellDeg * KM_PER_DEG, dist: null, sd: null, zs: null };
}

// Nearest land cell to a point within R cells, or -1.
export function snap(L, lng, lat, R) {
    const { grid, W, H, wrap, cellDeg } = L;
    let dc0 = (lng - L.west) / cellDeg;
    if (wrap) dc0 = ((dc0 % W) + W) % W;
    const col = Math.floor(dc0), row = Math.floor((L.north - lat) / cellDeg);
    let best = -1, bestD = Infinity;
    for (let dr = -R; dr <= R; dr++) {
        const r = row + dr;
        if (r < 0 || r >= H) continue;
        for (let dc = -R; dc <= R; dc++) {
            let c = col + dc;
            if (wrap) c = (c + W) % W; else if (c < 0 || c >= W) continue;
            if (!grid[r * W + c]) continue;
            const d = dc * dc + dr * dr;
            if (d < bestD) { bestD = d; best = r * W + c; }
        }
    }
    return best;
}

// Weight, time and hazard at a point from one layer of a solved level (L.res), blending the
// reached nodes of the four cells round it into out[0..2]. undefined when the point is outside
// the level, false when nothing near is reached, else true.
export function sampleNode(L, lng, lat, canoe, out) {
    const { W, H, cellDeg, wrap, grid } = L;
    const { w: nw, t: nt, hz: nh } = L.res;
    const canoeOf = canoe ? L.cn.canoeOf : null;
    let x = (lng - L.west) / cellDeg;
    if (wrap) x = ((x % W) + W) % W;
    else {
        x = (((lng - L.west) % 360) + 360) % 360 / cellDeg;
        if (x < 0.5 || x > W - 0.5) return undefined;
    }
    let y = (L.north - lat) / cellDeg;
    if (y < 0.5 || y > H - 0.5) { if (!wrap) return undefined; }
    x -= 0.5; y -= 0.5;
    const c0 = Math.floor(x), r0 = Math.floor(y), fx = x - c0, fy = y - r0;
    let sw = 0, st = 0, sh = 0, wsum = 0;
    for (let j = 0; j < 2; j++) {
        const r = Math.min(H - 1, Math.max(0, r0 + j));
        const wy = j ? fy : 1 - fy;
        for (let i = 0; i < 2; i++) {
            let c = c0 + i;
            if (wrap) c = (c + W) % W; else c = Math.min(W - 1, Math.max(0, c));
            const cell = r * W + c;
            let n = canoe ? canoeOf[cell] : (grid[cell] ? cell : -1);
            if (canoe && L.cn.extraN[cell]) {
                // several river nodes in the cell (T-082): the reached one of least weight
                for (let q = 0; q < L.cn.extraN[cell]; q++) {
                    const m = L.cn.extra0[cell] + q;
                    if (nw[m] < (n >= 0 ? nw[n] : Infinity)) n = m;
                }
            }
            if (n < 0 || nt[n] === Infinity) continue;
            const w = wy * (i ? fx : 1 - fx) + 1e-6;
            sw += nw[n] * w; st += nt[n] * w; sh += nh[n] * w; wsum += w;
        }
    }
    if (!wsum) return false;
    out[0] = sw / wsum; out[1] = st / wsum; out[2] = sh / wsum;
    return true;
}
