// Least walking time from the start to every cell. The worker owns the grids, runs every
// search and paints every overlay into an RGBA buffer; the page only shows them.
//
// L10: the coarse world grid (data/grid.bin.gz), solved whole on each click.
// L3, L1: tiles (data/tiles/), solved for a window round the view, seeded at the window
// border from the next coarser solved level. See docs/resolution.md.
//
// Dijkstra with an indexed binary heap, 16 neighbours per cell. The world grid wraps at
// 180 degrees; a window does not.

const Z3 = 5, Z1 = 7;            // zoom from which L3, L1 windows are solved
const CAP = { 3: 3e6, 1: 2e6 };  // most cells in a window (L1 cells cost more: rougher speeds)
const MARGIN = 0.25;             // window = view plus this share of its size on each side
const MARGIN_L3_UNDER_L1 = 1.0;  // the L3 window under an L1 one is wider, to seed it well
const ALIGN = 0.25;              // window edges sit on multiples of this many degrees

let bands = null, alpha = 190, maxHours = 1024;
let coarse = null;               // the L10 level: grid, sizes, dist once solved
let tiles = null;                // data/tiles/index.json, with a Set of tiles per level
let origin = null;               // { lng, lat } the solved start, set with the coarse solve
let originId = 0;
let clickGen = 0, viewGen = 0, clickPending = false;
let view = null;
let last = { 3: null, 1: null }; // last posted window per level, for reuse

// 16 moves as flat tables: 8 neighbours, then 8 knight moves. A knight move also counts
// the speed of the two cells it passes between, which must be land too, so a one cell wide
// strait or river of water can't be hopped over. Those two cells always lie between the
// start and end rows and columns, so they need no bounds check of their own.
const MDC = new Int32Array(16), MDR = new Int32Array(16);
const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
(function buildMoves() {
    const plain = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    let k = 0;
    for (const [dc, dr] of plain) { MDC[k] = dc; MDR[k] = dr; k++; }
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        MDC[k] = sx; MDR[k] = 2 * sy; E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        MDC[k] = 2 * sx; MDR[k] = sy; E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
})();

let inv = null;                  // hours per km for each stored value, -1 for water
const KM_PER_DEG = 111.195;

// A level: a block of cells cellDeg wide, its top-left corner at (west, north).
function makeLevel(grid, W, H, west, north, cellDeg, wrap) {
    const dxKm = new Float32Array(H);
    for (let r = 0; r < H; r++) {
        const lat = north - (r + 0.5) * cellDeg;
        dxKm[r] = cellDeg * KM_PER_DEG * Math.cos(lat * Math.PI / 180);
    }
    return { grid, W, H, west, north, south: north - H * cellDeg, east: west + W * cellDeg,
             cellDeg, wrap, dxKm, dyKm: cellDeg * KM_PER_DEG, dist: null };
}

// Yield to the event loop so newer messages can land; much faster than setTimeout.
const chan = new MessageChannel();
const wakers = [];
chan.port1.onmessage = () => wakers.shift()();
const tick = () => new Promise(res => { wakers.push(res); chan.port2.postMessage(0); });

// Dijkstra from seeds (cell index -> hours). Returns dist, or null if alive() went false.
async function solve(L, seedIdx, seedT, alive) {
    const { grid, W, H, wrap, dxKm, dyKm } = L;
    const N = W * H;
    // length of each move from each row, km
    const stepKm = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        stepKm[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
    }
    const dist = new Float32Array(N).fill(Infinity);
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

    for (let k = 0; k < seedIdx.length; k++) {
        const v = seedIdx[k], t = seedT[k];
        if (!(t < dist[v])) continue;
        dist[v] = t;
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
                dist[v] = nd;
                if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); }
                else up(pos[v]);
            }
        }
    }
    return dist;
}

// Nearest land cell to a point within R cells, or -1.
function snap(L, lng, lat, R) {
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

// Time at a point from a solved level, blending the reached cells round it.
// undefined when the point is outside the level, Infinity when nothing near is reached.
function sample(L, lng, lat) {
    const { W, H, cellDeg, wrap, dist } = L;
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
    let sum = 0, wsum = 0;
    for (let j = 0; j < 2; j++) {
        const r = Math.min(H - 1, Math.max(0, r0 + j));
        const wy = j ? fy : 1 - fy;
        for (let i = 0; i < 2; i++) {
            let c = c0 + i;
            if (wrap) c = (c + W) % W; else c = Math.min(W - 1, Math.max(0, c));
            const d = dist[r * W + c];
            if (d === Infinity) continue;
            const w = wy * (i ? fx : 1 - fx) + 1e-6;
            sum += d * w; wsum += w;
        }
    }
    return wsum ? sum / wsum : Infinity;
}

// RGBA pixels, linear in Web Mercator y (how MapLibre stretches an image between its
// corners), one canvas column per cell column.
const merc = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
function paint(L) {
    const { W, H, dist, north, south } = L;
    const yTop = merc(north), yBot = merc(south);
    const lonRad = W * L.cellDeg * Math.PI / 180;
    const cw = W, ch = Math.max(1, Math.min(8192, Math.round(cw * (yTop - yBot) / lonRad)));
    const px = new Uint8ClampedArray(cw * ch * 4);
    const edges = bands.map(b => b.h), cols = bands.map(b => b.color);
    const nb = edges.length;
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        const v = (north - lat) / L.cellDeg - 0.5;
        let r0 = Math.floor(v); const fy = v - r0;
        let r1 = r0 + 1;
        if (r0 < 0) r0 = 0; if (r1 > H - 1) r1 = H - 1; if (r0 > H - 1) r0 = H - 1;
        const o0 = r0 * W, o1 = r1 * W;
        // Blend the two rows when both cells were reached, else take the nearer.
        for (let x = 0; x < cw; x++) {
            const a = dist[o0 + x], c = dist[o1 + x];
            const h = (a !== Infinity && c !== Infinity) ? a * (1 - fy) + c * fy : (fy < 0.5 ? a : c);
            if (!(h <= edges[nb - 1])) continue;
            let k = 0;
            while (h > edges[k]) k++;
            const o = (y * cw + x) * 4, col = cols[k];
            px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = alpha;
        }
    }
    return { px, cw, ch };
}

function post(type, L, extra) {
    const t0 = performance.now();
    const { px, cw, ch } = paint(L);
    const dist = L.dist.slice();   // the worker keeps its own for seeding
    const msg = Object.assign({ type, originId, px, cw, ch, dist, W: L.W, H: L.H,
        west: L.west, east: L.east, north: L.north, south: L.south, cellDeg: L.cellDeg,
        paintMs: Math.round(performance.now() - t0) }, extra);
    self.postMessage(msg, [px.buffer, dist.buffer]);
}

// ---- click: the whole world at L10 ----
async function runClick(g, lng, lat) {
    const alive = () => g === clickGen;
    const L = coarse;
    const start = snap(L, lng, lat, 8);
    if (start < 0) { if (alive()) { clickPending = false; self.postMessage({ type: 'none' }); } return; }
    const t0 = performance.now();
    const dist = await solve(L, [start], [0], alive);
    if (!dist) return;
    clickPending = false;
    L.dist = dist;
    originId++;
    last = { 3: null, 1: null };
    // the start is where the click was if that cell is land, else the snapped cell's centre
    const r = (start / L.W) | 0, c = start - r * L.W;
    let col = Math.floor((((lng - L.west) / L.cellDeg) % L.W + L.W) % L.W);
    let row = Math.floor((L.north - lat) / L.cellDeg);
    origin = (col === c && row === r) ? { lng, lat }
        : { lng: L.west + (c + 0.5) * L.cellDeg, lat: L.north - (r + 0.5) * L.cellDeg };
    post('coarse', L, { origin, ms: Math.round(performance.now() - t0) });
    refine();
}

// ---- view: windows at L3 and L1 ----
function windowFor(f, v, margin) {
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

const contains = (a, b) => a && a.west <= b.west && a.east >= b.east && a.north >= b.north && a.south <= b.south;

const tileCache = new Map();     // "f/tr_tc" -> Promise<Uint8Array>, oldest first
let cacheBytes = 0;
function getTile(f, tr, tc) {
    const key = f + '/' + tr + '_' + tc;
    let p = tileCache.get(key);
    if (p) { tileCache.delete(key); tileCache.set(key, p); return p; }
    const lv = tiles.levels[f];
    p = fetch('data/tiles/' + lv.path + '/' + tr + '_' + tc + '.bin.gz')
        .then(r => { if (!r.ok) throw new Error('tile ' + key + ' ' + r.status);
                     return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(); })
        .then(b => new Uint8Array(b));
    p.catch(() => tileCache.delete(key));
    tileCache.set(key, p);
    cacheBytes += lv.tile * lv.tile;
    while (cacheBytes > 200e6 && tileCache.size > 1) {
        const k = tileCache.keys().next().value;
        tileCache.delete(k);
        const s = tiles.levels[k.split('/')[0]].tile;
        cacheBytes -= s * s;
    }
    return p;
}

async function loadWindow(w) {
    const lv = tiles.levels[w.f], T = lv.tile, TX = tiles.tilesX;
    const grid = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    for (let tr = Math.floor(w.row0 / T); tr <= Math.floor((w.row0 + w.rows - 1) / T); tr++) {
        for (let tc = Math.floor(w.col0 / T); tc <= Math.floor((w.col0 + w.cols - 1) / T); tc++) {
            const tcw = ((tc % TX) + TX) % TX;
            if (!lv.set.has(tr * TX + tcw)) continue;
            jobs.push(getTile(w.f, tr, tcw).then(t => {
                // overlap of tile (tr, tc) with the window, in window cells
                const r0 = Math.max(0, tr * T - w.row0), r1 = Math.min(w.rows, (tr + 1) * T - w.row0);
                const c0 = Math.max(0, tc * T - w.col0), c1 = Math.min(w.cols, (tc + 1) * T - w.col0);
                for (let r = r0; r < r1; r++) {
                    const src = (r + w.row0 - tr * T) * T + (c0 + w.col0 - tc * T);
                    grid.set(t.subarray(src, src + (c1 - c0)), r * w.cols + c0);
                }
            }));
        }
    }
    await Promise.all(jobs);
    return grid;
}

async function solveWindow(w, coarser, alive) {
    const t0 = performance.now();
    const grid = await loadWindow(w);
    if (!alive()) return null;
    const t1 = performance.now();
    const L = makeLevel(grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    // seeds: every land cell on the border ring, at the coarser level's time there
    const seedIdx = [], seedT = [];
    const W = w.cols, H = w.rows;
    const addSeed = (r, c) => {
        const i = r * W + c;
        if (!grid[i]) return;
        const lng = w.west + (c + 0.5) * w.cell, lat = w.north - (r + 0.5) * w.cell;
        let t;
        for (const C of coarser) { t = sample(C, lng, lat); if (t !== undefined) break; }
        if (t !== undefined && t <= maxHours) { seedIdx.push(i); seedT.push(t); }
    };
    for (let c = 0; c < W; c++) { addSeed(0, c); addSeed(H - 1, c); }
    for (let r = 1; r < H - 1; r++) { addSeed(r, 0); addSeed(r, W - 1); }
    // and the start, if it is inside
    const s = snap(L, origin.lng, origin.lat, Math.ceil(10 / w.f));
    if (s >= 0) { seedIdx.push(s); seedT.push(0); }
    const dist = await solve(L, seedIdx, seedT, alive);
    if (!dist) return null;
    L.dist = dist;
    L.fetchMs = Math.round(t1 - t0);
    L.solveMs = Math.round(performance.now() - t1);
    return L;
}

async function refine() {
    if (!coarse || !coarse.dist || !view || clickPending) return;
    const cg = clickGen, vg = viewGen;
    const alive = () => cg === clickGen && vg === viewGen;
    const v = view;
    const clear = (levels) => {
        for (const l of levels) last[l] = null;
        self.postMessage({ type: 'clear', originId, levels });
    };
    if (v.zoom < Z3) { clear([3, 1]); return; }
    const w3 = windowFor(3, v, v.zoom >= Z1 ? MARGIN_L3_UNDER_L1 : MARGIN);
    const w1 = v.zoom >= Z1 ? windowFor(1, v, MARGIN) : null;
    if (!w3) { clear([3, 1]); return; }
    // the last L3 window will do if it holds the wanted view round (and the L1 window)
    const need3 = windowFor(3, v, w1 ? MARGIN : 0) || w3;
    let L3 = last[3];
    if (!(L3 && contains(L3.win, need3) && (!w1 || contains(L3.win, w1)))) {
        L3 = await solveWindow(w3, [coarse], alive);
        if (!L3) return;
        last[3] = L3;
        post('win', L3, { level: 3, fetchMs: L3.fetchMs, solveMs: L3.solveMs, cells: L3.W * L3.H });
    }
    if (!w1) { clear([1]); return; }
    if (last[1] && contains(last[1].win, windowFor(1, v, 0) || w1)) return;
    const L1 = await solveWindow(w1, [L3, coarse], alive);
    if (!L1) return;
    last[1] = L1;
    post('win', L1, { level: 1, fetchMs: L1.fetchMs, solveMs: L1.solveMs, cells: L1.W * L1.H });
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'init') {
        bands = msg.bands; alpha = msg.alpha; maxHours = msg.maxHours;
        const m = msg.meta;
        inv = new Float32Array(256).fill(-1);
        for (let v = 1; v < 256; v++) inv[v] = m.scale / v;
        const cell = (m.north - m.south) / m.height;
        coarse = makeLevel(new Uint8Array(msg.bytes), m.width, m.height, m.west, m.north, cell, true);
        tiles = msg.tiles;
        for (const k in tiles.levels) tiles.levels[k].set = new Set(tiles.levels[k].tiles);
        self.postMessage({ type: 'ready' });
    } else if (msg.type === 'run') {
        const g = ++clickGen;
        clickPending = true;
        runClick(g, msg.lng, msg.lat);
    } else if (msg.type === 'view') {
        view = msg;
        viewGen++;
        refine();
    }
};
