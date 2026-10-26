// View windows at L3 and L1: which cells they cover, and the solve seeded from the next
// coarser level, layer by layer. See docs/resolution.md and docs/canoe.md 5.
import { cfg } from './config.js';
import { makeLevel, snap, sampleNode } from './levels.js';
import { loadWindow } from './tiles.js';
import { loadWindowWater } from './canoedata.js';
import { buildCanoe, shoreAt } from './canoe.js';
import { solve, cellTimes, P_SEED } from './search.js';

export const Z3 = 5, Z1 = 7;            // zoom from which L3, L1 windows are solved
const CAP = { 3: 3e6, 1: 2e6 };         // most cells in a window (L1 cells cost more: rougher speeds)
const CAP_CANOE = { 3: 1.5e6, 1: 2e6 }; // on a canoe start: water is solved too (~1.1-1.3 s at these)
export const MARGIN = 0.25;             // window = view plus this share of its size on each side
export const MARGIN_L3_UNDER_L1 = 1.0;  // the L3 window under an L1 one is wider, to seed it well
const ALIGN = 0.25;                     // window edges sit on multiples of this many degrees

export function windowFor(f, v, margin, canoe) {
    const cell = f / 120, cap = (canoe ? CAP_CANOE : CAP)[f];
    const cx = (v.west + v.east) / 2, cy = (v.north + v.south) / 2;
    let hw = (v.east - v.west) / 2 * (1 + 2 * margin), hh = (v.north - v.south) / 2 * (1 + 2 * margin);
    const cells = (2 * hw / cell) * (2 * hh / cell);
    if (cells > cap) { const k = Math.sqrt(cap / cells) * 0.97; hw *= k; hh *= k; }
    hw = Math.min(hw, 170);
    const west = Math.floor((cx - hw) / ALIGN) * ALIGN, east = Math.ceil((cx + hw) / ALIGN) * ALIGN;
    const north = Math.min(85, Math.ceil((cy + hh) / ALIGN) * ALIGN);
    const south = Math.max(-60, Math.floor((cy - hh) / ALIGN) * ALIGN);
    if (north - south < ALIGN) return null;
    return { f, cell, west, east, north, south,
             col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
}

export const contains = (a, b) => a && a.west <= b.west && a.east >= b.east && a.north >= b.north && a.south <= b.south;

// Seeds for a window: every node on its border ring at the coarser level's weight, time and
// hazard there, from the same layer (the first of `coarser` that covers the point), and the
// start if it is inside.
function windowSeeds(L, w, coarser, origin, canoe) {
    const { grid } = L;
    const seeds = { idx: [], w: [], t: [], hz: [], pred: [] };
    const W = w.cols, H = w.rows;
    const out = [0, 0, 0];
    const add = (node, lng, lat, layer) => {
        let got;
        for (const C of coarser) { got = sampleNode(C, lng, lat, layer, out); if (got !== undefined) break; }
        if (got && out[1] <= cfg.maxHours && out[2] <= cfg.hzStop) {
            seeds.idx.push(node); seeds.w.push(out[0]); seeds.t.push(out[1]); seeds.hz.push(out[2]);
            seeds.pred.push(P_SEED);
        }
    };
    const addSeed = (r, c) => {
        const i = r * W + c;
        const lng = w.west + (c + 0.5) * w.cell, lat = w.north - (r + 0.5) * w.cell;
        if (grid[i]) add(i, lng, lat, false);
        if (canoe) { const n = L.cn.canoeOf[i]; if (n >= 0) add(n, lng, lat, true); }
    };
    for (let c = 0; c < W; c++) { addSeed(0, c); addSeed(H - 1, c); }
    for (let r = 1; r < H - 1; r++) { addSeed(r, 0); addSeed(r, W - 1); }
    const s = snap(L, origin.lng, origin.lat, Math.ceil(10 / w.f));
    if (s >= 0) { seeds.idx.push(s); seeds.w.push(0); seeds.t.push(0); seeds.hz.push(0); seeds.pred.push(0); }
    return seeds;
}

// canoe: the sea grid of the canoe world data, on a canoe start; null on a walking one.
export async function solveWindow(w, coarser, origin, alive, canoe) {
    const t0 = performance.now();
    const grid = await loadWindow(w);
    if (!alive()) return null;
    const L = makeLevel(grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    if (canoe) {
        const data = await loadWindowWater(w, grid, (lng, lat) => {
            for (const C of coarser) { const s = shoreAt(C, lng, lat); if (s !== undefined) return s; }
            return 78;
        });
        if (!alive()) return null;
        buildCanoe(L, data, canoe);
    }
    const t1 = performance.now();
    const seeds = windowSeeds(L, w, coarser, origin, !!canoe);
    const res = await solve(L, seeds, alive, !!canoe);
    if (!res) return null;
    L.res = res;
    L.dist = cellTimes(L, res, !!canoe);
    L.fetchMs = Math.round(t1 - t0);
    L.solveMs = Math.round(performance.now() - t1);
    return L;
}
