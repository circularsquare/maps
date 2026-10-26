// View windows at L3 and L1: which cells they cover, and the solve seeded from the next
// coarser level. See docs/resolution.md.
import { cfg } from './config.js';
import { makeLevel, snap, sample } from './levels.js';
import { loadWindow } from './tiles.js';
import { solve } from './search.js';

export const Z3 = 5, Z1 = 7;            // zoom from which L3, L1 windows are solved
const CAP = { 3: 3e6, 1: 2e6 };         // most cells in a window (L1 cells cost more: rougher speeds)
export const MARGIN = 0.25;             // window = view plus this share of its size on each side
export const MARGIN_L3_UNDER_L1 = 1.0;  // the L3 window under an L1 one is wider, to seed it well
const ALIGN = 0.25;                     // window edges sit on multiples of this many degrees

export function windowFor(f, v, margin) {
    const cell = f / 120;
    const cx = (v.west + v.east) / 2, cy = (v.north + v.south) / 2;
    let hw = (v.east - v.west) / 2 * (1 + 2 * margin), hh = (v.north - v.south) / 2 * (1 + 2 * margin);
    const cells = (2 * hw / cell) * (2 * hh / cell);
    if (cells > CAP[f]) { const k = Math.sqrt(CAP[f] / cells) * 0.97; hw *= k; hh *= k; }
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

// Seeds for a window: every land cell on its border ring at the coarser level's time there
// (the first of `coarser` that reaches the point), and the start if it is inside.
function windowSeeds(L, w, coarser, origin) {
    const { grid } = L;
    const seedIdx = [], seedT = [];
    const W = w.cols, H = w.rows;
    const addSeed = (r, c) => {
        const i = r * W + c;
        if (!grid[i]) return;
        const lng = w.west + (c + 0.5) * w.cell, lat = w.north - (r + 0.5) * w.cell;
        let t;
        for (const C of coarser) { t = sample(C, lng, lat); if (t !== undefined) break; }
        if (t !== undefined && t <= cfg.maxHours) { seedIdx.push(i); seedT.push(t); }
    };
    for (let c = 0; c < W; c++) { addSeed(0, c); addSeed(H - 1, c); }
    for (let r = 1; r < H - 1; r++) { addSeed(r, 0); addSeed(r, W - 1); }
    const s = snap(L, origin.lng, origin.lat, Math.ceil(10 / w.f));
    if (s >= 0) { seedIdx.push(s); seedT.push(0); }
    return { seedIdx, seedT };
}

export async function solveWindow(w, coarser, origin, alive) {
    const t0 = performance.now();
    const grid = await loadWindow(w);
    if (!alive()) return null;
    const t1 = performance.now();
    const L = makeLevel(grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    const { seedIdx, seedT } = windowSeeds(L, w, coarser, origin);
    const dist = await solve(L, seedIdx, seedT, alive);
    if (!dist) return null;
    L.dist = dist;
    L.fetchMs = Math.round(t1 - t0);
    L.solveMs = Math.round(performance.now() - t1);
    return L;
}
