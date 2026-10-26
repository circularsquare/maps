// The state of one map session: the solved start, the last windows, and what to do on a click
// or a new view. Posts every result to the page.
import { cfg, setConfig } from './config.js';
import { makeLevel, snap } from './levels.js';
import { setTiles } from './tiles.js';
import { solve, cellTimes } from './search.js';
import { loadWorld } from './canoedata.js';
import { buildCanoe } from './canoe.js';
import { paint } from './paint.js';
import { Z3, Z1, MARGIN, MARGIN_L3_UNDER_L1, windowFor, contains, solveWindow } from './windows.js';

let coarse = null;               // the L10 level: grid, sizes, res and dist once solved
let origin = null;               // { lng, lat } the solved start, set with the coarse solve
let originId = 0;
let mode = 'walk';               // of the solved start: 'walk' or 'canoe'
let sea = null;                  // the sea grid, once the canoe world data is loaded
let clickGen = 0, viewGen = 0, clickPending = false;
let view = null;
let last = { 3: null, 1: null }; // last posted window per level, for reuse

export function init(msg) {
    setConfig(msg);
    const m = msg.meta;
    const cell = (m.north - m.south) / m.height;
    coarse = makeLevel(new Uint8Array(msg.bytes), m.width, m.height, m.west, m.north, cell, true);
    setTiles(msg.tiles);
    self.postMessage({ type: 'ready' });
}

function post(type, L, extra) {
    const t0 = performance.now();
    const { px, cw, ch } = paint(L);
    const dist = L.dist.slice();   // the worker keeps its own for seeding
    const msg = Object.assign({ type, originId, mode, px, cw, ch, dist, W: L.W, H: L.H,
        west: L.west, east: L.east, north: L.north, south: L.south, cellDeg: L.cellDeg,
        paintMs: Math.round(performance.now() - t0) }, extra);
    self.postMessage(msg, [px.buffer, dist.buffer]);
}

export function click(lng, lat, m) {
    const g = ++clickGen;
    clickPending = true;
    runClick(g, lng, lat, m === 'canoe' ? 'canoe' : 'walk').catch(e => {
        if (g !== clickGen) return;
        clickPending = false;
        self.postMessage({ type: 'error', message: String(e && e.message || e) });
    });
}

export function setView(v) {
    view = v;
    viewGen++;
    refine();
}

// The canoe layer of L10, built once.
let canoe10 = null;
function canoeWorld() {
    return canoe10 || (canoe10 = loadWorld(coarse.W, coarse.H).then(d => {
        const t0 = performance.now();
        buildCanoe(coarse, d, d.sea);
        sea = d.sea;
        return Math.round(performance.now() - t0);
    }).catch(e => { canoe10 = null; throw e; }));
}

// ---- click: the whole world at L10 ----
async function runClick(g, lng, lat, m) {
    const alive = () => g === clickGen;
    const L = coarse;
    const start = snap(L, lng, lat, 8);
    if (start < 0) { if (alive()) { clickPending = false; self.postMessage({ type: 'none' }); } return; }
    const canoe = m === 'canoe';
    let buildMs = 0;
    if (canoe) { const t = performance.now(); await canoeWorld(); buildMs = Math.round(performance.now() - t); if (!alive()) return; }
    const t0 = performance.now();
    const res = await solve(L, { idx: [start], w: [0], t: [0], hz: [0] }, alive, canoe);
    if (!res) return;
    clickPending = false;
    L.res = res;
    L.dist = cellTimes(L, res, canoe);
    mode = m;
    originId++;
    last = { 3: null, 1: null };
    // the start is where the click was if that cell is land, else the snapped cell's centre
    const r = (start / L.W) | 0, c = start - r * L.W;
    let col = Math.floor((((lng - L.west) / L.cellDeg) % L.W + L.W) % L.W);
    let row = Math.floor((L.north - lat) / L.cellDeg);
    origin = (col === c && row === r) ? { lng, lat }
        : { lng: L.west + (c + 0.5) * L.cellDeg, lat: L.north - (r + 0.5) * L.cellDeg };
    post('coarse', L, { origin, ms: Math.round(performance.now() - t0), loadMs: buildMs });
    refine();
}

// ---- view: windows at L3 and L1 ----
async function refine() {
    if (!coarse || !coarse.dist || !view || clickPending) return;
    const cg = clickGen, vg = viewGen;
    const alive = () => cg === clickGen && vg === viewGen;
    const v = view;
    const sw = mode === 'canoe' ? sea : null;
    const clear = (levels) => {
        for (const l of levels) last[l] = null;
        self.postMessage({ type: 'clear', originId, levels });
    };
    if (v.zoom < Z3) { clear([3, 1]); return; }
    const cw = !!sw;
    const w3 = windowFor(3, v, v.zoom >= Z1 ? MARGIN_L3_UNDER_L1 : MARGIN, cw);
    const w1 = v.zoom >= Z1 ? windowFor(1, v, MARGIN, cw) : null;
    if (!w3) { clear([3, 1]); return; }
    // the last L3 window will do if it holds the wanted view round (and the L1 window)
    const need3 = windowFor(3, v, w1 ? MARGIN : 0, cw) || w3;
    let L3 = last[3];
    if (!(L3 && contains(L3.win, need3) && (!w1 || contains(L3.win, w1)))) {
        L3 = await solveWindow(w3, [coarse], origin, alive, sw);
        if (!L3) return;
        last[3] = L3;
        post('win', L3, { level: 3, fetchMs: L3.fetchMs, solveMs: L3.solveMs, cells: L3.W * L3.H });
    }
    if (!w1) { clear([1]); return; }
    if (last[1] && contains(last[1].win, windowFor(1, v, 0, cw) || w1)) return;
    const L1 = await solveWindow(w1, [L3, coarse], origin, alive, sw);
    if (!L1) return;
    last[1] = L1;
    post('win', L1, { level: 1, fetchMs: L1.fetchMs, solveMs: L1.solveMs, cells: L1.W * L1.H });
}
