// T-007 prototype: checks/levels.mjs's L1/L10 and L1/L3 (and seams) with the coarse levels patched
// from proto.py's data. From the maps root:
//   JSDIR=<js folder>/ node isochrone/notes/T-007/p7.mjs --var=base|A|A2|B|Blite [--patch=<L10 dir>] [--patch3=<L3 dir>]
//        [--only=Sogndal,Coari] [--seams] [--l3] [--noroads] [--min] [--reps=N] [--mine]
// base: the stock js/search.js on the served levels (the reference; --mine runs this file's search
// with no patch instead, which must agree). A: directions a cell is not crossed blocked (L10).
// A2: A, and cells crossed in no direction made water. B: split cells' slots as nodes with links
// (further slots as further nodes, plain moves only). Blite: B with slot 0 only (the rest dropped).
// The start's node is the slot holding the L1 cell the L1 start snaps to. A destination in a split
// cell reads its own slot's time (--min: the least of the cell's nodes, what the paint would show).
import { readFileSync, readdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url)).replace(/\\/g, '/');
const JSDIR = (process.env.JSDIR || ROOT + 'js/').replace(/\\/g, '/');
const ARGS = process.argv.slice(2);
const opt = (k, d) => { const a = ARGS.find((x) => x.startsWith('--' + k + '=')); return a ? a.slice(k.length + 3) : d; };
const BASE = ROOT + 'data/';
const ONLY = opt('only', '') ? opt('only', '').split(',') : null;
const HALF = 3.3;
const VAR = opt('var', 'base');
const PATCH = opt('patch', ''), PATCH3 = opt('patch3', '');
const USE_MIN = ARGS.includes('--min');
const REPS = +opt('reps', '1');
const MINE = VAR !== 'base' || ARGS.includes('--mine');

globalThis.fetch = async (u) => {
    let p = String(u);
    if (p.startsWith('file://')) p = p.slice(7);
    else if (!/^[A-Za-z]:\//.test(p)) p = ROOT + p;
    try { return new Response(readFileSync(p)); } catch { return new Response(null, { status: 404 }); }
};
const imp = (m) => import(pathToFileURL(JSDIR + m).href);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, setDirs } = await imp('levels.js');
const { solve, cellTimes, P_SEED, MDC, MDR } = await imp('search.js');
const T = await imp('tiles.js');

const meta = JSON.parse(readFileSync(BASE + 'grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: ROOT + 'data/canoe/', hazardBase: ROOT + 'data/hazard/' });
cfg.hazard = false;
const ROADS = !ARGS.includes('--noroads');
if (!ROADS) cfg.roads = false;
T.setTiles(JSON.parse(readFileSync(BASE + 'tiles/index.json', 'utf8')), BASE + 'tiles/');
const gz = (f) => { const b = gunzipSync(readFileSync(BASE + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
const L10 = makeLevel(gz('grid.bin.gz'), meta.width, meta.height, meta.west, meta.north, (meta.north - meta.south) / meta.height, true);
if (cfg.roads && meta.roads) T.applyRoads(L10.grid, gz(meta.roads));
setDirs(L10, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);
L10.f = 10;

// ---- move tables (as js/search.js) ----
const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
const KA = new Int32Array(16), KB = new Int32Array(16), KC = new Int32Array(16), KD = new Int32Array(16);
const DX = new Int32Array(16), DY = new Int32Array(16);
{
    const pm = (dc, dr) => { for (let k = 0; k < 8; k++) if (MDC[k] === dc && MDR[k] === dr) return k; return -1; };
    let k = 8;
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
    for (let k = 8; k < 16; k++) {
        KA[k] = pm(E0C[k], E0R[k]); KB[k] = pm(E1C[k], E1R[k]);
        KC[k] = pm(MDC[k] - E0C[k], MDR[k] - E0R[k]); KD[k] = pm(MDC[k] - E1C[k], MDR[k] - E1R[k]);
    }
    for (let k = 0; k < 16; k++) { DX[k] = MDC[k] > 0 ? 0 : 1; DY[k] = MDR[k] > 0 ? 2 : 3; }
}

// ---- patches ----
const readRecs = (dir) => readdirSync(dir).filter((f) => f.startsWith('p_' + (ROADS ? 'on' : 'off') + '_') && f.endsWith('.json'))
    .map((f) => JSON.parse(readFileSync(dir + '/' + f, 'utf8')));
const REC10 = PATCH ? readRecs(PATCH) : [], REC3 = PATCH3 ? readRecs(PATCH3) : [];
// the plain level (no patch): one node a cell
const plainP = (L) => ({ N: L.W * L.H, M: L.W * L.H, sd: L.sd, grid: L.grid, listed: null, lk: null, extraN: null, cellOf: null, e0: null, smap: new Map(), f: L.f });
// B and Blite: recs' global cell ids mapped to the level's by toLocal (-1 outside)
function makePatch(L, recs, toLocal, lite) {
    const N = L.W * L.H;
    let ex = 0;
    if (!lite) for (const r of recs) for (const s of r.split) if (toLocal(s[0]) >= 0) ex += s[1].length - 1;
    const M = N + ex;
    const sd = new Float32Array(4 * M); sd.set(L.sd.subarray(0, 4 * N));
    const listed = new Uint8Array(N), lk = new Uint8Array(8 * M), extraN = new Uint8Array(N), cellOf = new Int32Array(ex);
    const e0 = new Map(), smap = new Map();
    let e = 0, nSplit = 0, nListed = 0;
    for (const r of recs) {
        for (const [g, sds, masks, sm] of r.split) {
            const cell = toLocal(g);
            if (cell < 0 || !L.grid[cell]) continue;
            nSplit++;
            listed[cell] = 1;
            smap.set(cell, sm);
            for (let j = 0; j < 4; j++) sd[4 * cell + j] = sds[0][j];
            for (let k = 0; k < 8; k++) lk[8 * cell + k] = lite ? masks[0][k] & 1 : masks[0][k];
            if (lite) continue;
            e0.set(cell, e);
            extraN[cell] = sds.length - 1;
            for (let s = 1; s < sds.length; s++, e++) {
                const v = N + e;
                cellOf[e] = cell;
                for (let j = 0; j < 4; j++) sd[4 * v + j] = sds[s][j];
                for (let k = 0; k < 8; k++) lk[8 * v + k] = masks[s][k];
            }
        }
        for (const [g, masks] of r.listed) {
            const cell = toLocal(g);
            if (cell < 0 || !L.grid[cell]) continue;
            nListed++;
            listed[cell] = 1;
            for (let k = 0; k < 8; k++) lk[8 * cell + k] = lite ? masks[k] & 1 : masks[k];
        }
    }
    return { N, M, sd, grid: L.grid, listed, lk, extraN, cellOf, e0, smap, f: L.f, nSplit, nListed, nExtra: e };
}
// A and A2 (L10)
function makeA(L, recs, a2) {
    const P = plainP(L);
    P.sd = L.sd.slice(); P.grid = L.grid.slice(); P.nBlocked = 0;
    for (const r of recs) for (const [cell, bits] of r.blocked) {
        if (!P.grid[cell]) continue;
        P.nBlocked++;
        if (bits === 16) { if (a2) P.grid[cell] = 0; continue; }
        for (let j = 0; j < 4; j++) if (bits >> j & 1) P.sd[4 * cell + j] = 1e3;
    }
    return P;
}
const extraId = (P, cell, s) => P.N + P.e0.get(cell) + s - 1;

// ---- the search as a page could run it: a move mask a node (bits 0-15 the 16 moves to the
// neighbour's slot 0, bit 16: further edges to other slots), made once when the level loads ----
function buildMasks(L, P) {
    const { W, H, wrap } = L, { grid, M, N, listed, lk, extraN } = P;
    const mv = new Uint32Array(M), edges = new Map();
    const nb = (r, c, k) => {
        const r2 = r + MDR[k];
        let c2 = c + MDC[k];
        if (r2 < 0 || r2 >= H) return -1;
        if (c2 < 0 || c2 >= W) { if (!wrap) return -1; c2 = (c2 + W) % W; }
        return r2 * W + c2;
    };
    const l8 = (x, k) => {
        if (listed && listed[x]) return lk[8 * x + k];
        const r = (x / W) | 0, v = nb(r, x - r * W, k);
        return v >= 0 && grid[v] ? 1 : 0;
    };
    for (let u = 0; u < M; u++) {
        const cell = u < N ? u : P.cellOf[u - N];
        if (!grid[cell]) continue;
        const r = (cell / W) | 0, c = cell - r * W;
        let m = 0;
        for (let k = 0; k < 16; k++) {
            const v = nb(r, c, k);
            if (v < 0 || !grid[v]) continue;
            if (k < 8) {
                const b = u >= N || (listed && listed[u]) ? lk[8 * u + k] : 1;
                if (b & 1) m |= 1 << k;
                if (b > 1) {
                    let e = edges.get(u); if (!e) edges.set(u, e = []);
                    for (let s = 1; s <= extraN[v]; s++) if (b >> s & 1) e.push(extraId(P, v, s), k);
                }
                continue;
            }
            if (u >= N) continue;
            const i0 = nb(r, c, KA[k]), i1 = nb(r, c, KB[k]);
            if (!grid[i0] || !grid[i1]) continue;
            if (listed && (listed[u] || listed[i0] || listed[i1]) &&
                (!(l8(u, KA[k]) & 1) || !(l8(i0, KC[k]) & 1) || !(l8(u, KB[k]) & 1) || !(l8(i1, KD[k]) & 1))) continue;
            m |= 1 << k;
        }
        if (edges.has(u)) m |= 1 << 16;
        mv[u] = m;
    }
    P.mv = mv; P.edges = edges;
}
function solveY(L, P, seeds) {
    const { W, H, dxKm, dyKm } = L;
    const { sd, M, N, cellOf, mv, edges } = P;
    const maxHours = cfg.maxHours;
    const wk = new Float32Array(H * 16), stepKm = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        stepKm[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
        wk[r * 16 + k] = Math.abs(dx) / (Math.abs(dx) + Math.abs(dy));
    }
    const dist = new Float32Array(M).fill(Infinity);
    const heap = new Int32Array(M), pos = new Int32Array(M).fill(-1);
    let size = 0;
    function up(i) { const node = heap[i], d = dist[node]; while (i > 0) { const p = (i - 1) >> 1; if (dist[heap[p]] <= d) break; heap[i] = heap[p]; pos[heap[i]] = i; i = p; } heap[i] = node; pos[node] = i; }
    function down(i) { const node = heap[i], d = dist[node]; for (;;) { let c = 2 * i + 1; if (c >= size) break; if (c + 1 < size && dist[heap[c + 1]] < dist[heap[c]]) c++; if (dist[heap[c]] >= d) break; heap[i] = heap[c]; pos[heap[i]] = i; i = c; } heap[i] = node; pos[node] = i; }
    const push = (v, nd) => { if (nd < dist[v]) { dist[v] = nd; if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); } else up(pos[v]); } };
    for (let k = 0; k < seeds.idx.length; k++) push(seeds.idx[k], seeds.w[k]);
    while (size > 0) {
        const u = heap[0];
        pos[u] = -2;
        size--;
        if (size > 0) { heap[0] = heap[size]; pos[heap[0]] = 0; down(0); }
        const du = dist[u];
        if (du > maxHours) break;
        const cell = u < N ? u : cellOf[u - N];
        const r = (cell / W) | 0, c = cell - r * W, sr = r * 16;
        const m = mv[u];
        for (let k = 0; k < 16; k++) {
            if (!(m & (1 << k))) continue;
            let c2 = c + MDC[k];
            if (c2 < 0) c2 += W; else if (c2 >= W) c2 -= W;
            const v = (r + MDR[k]) * W + c2;
            if (pos[v] === -2) continue;
            const a = wk[sr + k], b = 1 - a, X = DX[k], Y = DY[k];
            let mean = a * (sd[4 * u + X] + sd[4 * v + X]) + b * (sd[4 * u + Y] + sd[4 * v + Y]);
            if (k < 8) mean *= 0.5;
            else {
                let c0 = c + E0C[k], c1 = c + E1C[k];
                if (c0 < 0) c0 += W; else if (c0 >= W) c0 -= W;
                if (c1 < 0) c1 += W; else if (c1 >= W) c1 -= W;
                const i0 = (r + E0R[k]) * W + c0, i1 = (r + E1R[k]) * W + c1;
                mean = (mean + a * (sd[4 * i0 + X] + sd[4 * i1 + X]) + b * (sd[4 * i0 + Y] + sd[4 * i1 + Y])) * 0.25;
            }
            push(v, du + stepKm[sr + k] * mean);
        }
        if (m & 0x10000) {
            const e = edges.get(u);
            for (let j = 0; j < e.length; j += 2) {
                const t = e[j], k = e[j + 1];
                if (pos[t] === -2) continue;
                const a = wk[sr + k], b = 1 - a, X = DX[k], Y = DY[k];
                push(t, du + stepKm[sr + k] * 0.5 * (a * (sd[4 * u + X] + sd[4 * t + X]) + b * (sd[4 * u + Y] + sd[4 * t + Y])));
            }
        }
    }
    return dist;
}

// ---- reading times by slot ----
function nearestSlot(sm, f, pr, pc) {
    let best = 255, bd = Infinity;
    for (let i = 0; i < f * f; i++) {
        if (sm[i] === 255) continue;
        const d = (Math.floor(i / f) - pr) ** 2 + (i % f - pc) ** 2;
        if (d < bd) { bd = d; best = sm[i]; }
    }
    return best;
}
// the L1 pixel of (lat, lng) inside cell `cell` of level L, clamped into it
function pixelIn(L, f, cell, lat, lng) {
    const r = Math.floor(cell / L.W), c = cell - r * L.W;
    let pr = Math.floor((L.north - lat) * 120) - r * f, pc = Math.floor((lng - L.west) * 120) - c * f;
    if (L.wrap) { pc = ((pc % 43200) + 43200) % 43200; if (pc > 21600) pc -= 43200; }
    return [Math.max(0, Math.min(f - 1, pr)), Math.max(0, Math.min(f - 1, pc))];
}
function slotNode(P, L, cell, lat, lng) {
    const sm = P.smap.get(cell);
    if (!sm || !P.e0) return cell;
    const [pr, pc] = pixelIn(L, P.f, cell, lat, lng);
    const s = nearestSlot(sm, P.f, pr, pc);
    return s > 0 && s !== 255 && s <= P.extraN[cell] ? extraId(P, cell, s) : cell;
}
function nodeTime(P, L, dist, cell, lat, lng, inside) {
    const nx = P.extraN ? P.extraN[cell] : 0;
    if (!nx) return dist[cell];
    const all = () => { let b = dist[cell]; for (let s = 1; s <= nx; s++) b = Math.min(b, dist[extraId(P, cell, s)]); return b; };
    if (USE_MIN && inside) return all();
    const t = dist[slotNode(P, L, cell, lat, lng)];
    return t < Infinity ? t : all();
}
function timeAtP(P, L, dist, lat, lng) {
    let c = Math.floor((lng - L.west) / L.cellDeg), r = Math.floor((L.north - lat) / L.cellDeg);
    if (L.wrap) c = ((c % L.W) + L.W) % L.W;
    if (r < 0 || r >= L.H || c < 0 || c >= L.W) return NaN;
    let best = nodeTime(P, L, dist, r * L.W + c, lat, lng, true);
    if (best < Infinity) return best;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        let cc = c + dc; const rr = r + dr;
        if (L.wrap) cc = ((cc % L.W) + L.W) % L.W;
        if (rr >= 0 && rr < L.H && cc >= 0 && cc < L.W) best = Math.min(best, nodeTime(P, L, dist, rr * L.W + cc, lat, lng, false));
    }
    return best;
}
// bilinear from the four L10 cell centres round the point, each cell's node by slot
function sample10(P, dist, lng, lat) {
    let x = (lng - L10.west) / L10.cellDeg, y = (L10.north - lat) / L10.cellDeg;
    x = ((x % L10.W) + L10.W) % L10.W - 0.5; y -= 0.5;
    const c0 = Math.floor(x), r0 = Math.floor(y), fx = x - c0, fy = y - r0;
    let st = 0, ws = 0;
    for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
        const r = r0 + j, c = ((c0 + i) % L10.W + L10.W) % L10.W, cell = r * L10.W + c;
        if (!P.grid[cell]) continue;
        const t = nodeTime(P, L10, dist, cell, lat, lng, false);
        if (!(t < Infinity)) continue;
        const w = (j ? fy : 1 - fy) * (i ? fx : 1 - fx) + 1e-6;
        st += t * w; ws += w;
    }
    return ws ? st / ws : NaN;
}
// the start's node: the slot of the L1 cell the L1 start snaps to, in that cell's coarse cell
function startNode(P, L, L1, lat, lng) {
    const fallback = snap(L, lng, lat, L.wrap ? 8 : Math.ceil(10 / L.f));
    if (!P.e0) return fallback;
    const s1 = snap(L1, lng, lat, 10);
    if (s1 < 0) return fallback;
    const la = L1.north - (Math.floor(s1 / L1.W) + 0.5) * L1.cellDeg, lo = L1.west + (s1 % L1.W + 0.5) * L1.cellDeg;
    let c = Math.floor((lo - L.west) / L.cellDeg);
    const r = Math.floor((L.north - la) / L.cellDeg);
    if (L.wrap) c = ((c % L.W) + L.W) % L.W;
    const cell = r * L.W + c;
    if (!P.grid[cell]) return fallback;
    return slotNode(P, L, cell, la, lo);
}

// ---- the L10 patch ----
let P10 = plainP(L10);
if (VAR === 'A' || VAR === 'A2') P10 = makeA(L10, REC10, VAR === 'A2');
else if ((VAR === 'B' || VAR === 'Blite') && REC10.length) P10 = makePatch(L10, REC10, (g) => g, VAR === 'Blite');
if (MINE) { const t0 = performance.now(); buildMasks(L10, P10); console.log(`L10 move masks in ${(performance.now() - t0).toFixed(0)} ms, ${P10.edges.size} nodes with further edges`); }

// ---- levels.mjs ----
const STARTS = [
    ['Poltava', 'farmland', 49.59, 34.55], ['Ames', 'farmland', 42.03, -93.62], ['Pampa', 'farmland', -34.59, -60.95],
    ['Ludhiana', 'farmland', 30.90, 75.85], ['Brig', 'mountains', 46.32, 7.99], ['Cusco', 'mountains', -13.53, -71.97],
    ['Pokhara', 'mountains', 28.21, 83.99], ['Dillon', 'mountains', 39.63, -106.04], ['Jyvaskyla', 'lakes', 62.24, 25.75],
    ['Sudbury', 'lakes', 46.49, -80.99], ['Bemidji', 'lakes', 47.47, -94.88], ['Tamanrasset', 'desert', 22.79, 5.53],
    ['Alice Springs', 'desert', -23.70, 133.88], ['Dalanzadgad', 'desert', 43.57, 104.42], ['Kisangani', 'forest', 0.52, 25.19],
    ['Coari', 'forest', -4.08, -63.14], ['Ust-Ilimsk', 'forest', 58.00, 102.66], ['Cluj', 'city', 46.77, 23.60],
    ['New York', 'city', 40.758, -73.985], ['London', 'city', 51.507, -0.128], ['Lagos', 'city', 6.52, 3.38],
    ['Delhi', 'city', 28.61, 77.21], ['Khulna', 'coast/delta', 22.82, 89.55], ['Utrecht', 'coast/delta', 52.09, 5.12],
    ['Sogndal', 'coast/delta', 61.23, 7.10],
];
const DISTS = [20, 40, 80, 160, 300], BEARINGS = 12, KM = 111.195;
const win = (f, west, south, east, north) => {
    const cell = f / 120;
    return { f, cell, west, east, north, south, col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
};
const box = (lat, lng, hLat, hLon) => ({ west: Math.floor((lng - hLon) * 4) / 4, east: Math.ceil((lng + hLon) * 4) / 4,
    south: Math.max(-60, Math.floor((lat - hLat) * 4) / 4), north: Math.min(85, Math.ceil((lat + hLat) * 4) / 4) });
async function level(w) {
    const dirs = {};
    const g = await T.loadWindow(w, dirs);
    const L = makeLevel(g, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w; L.f = w.f;
    setDirs(L, dirs.d, dirs.e);
    return L;
}
async function solveFrom(L, lat, lng) {
    const s = snap(L, lng, lat, L.win ? Math.ceil(10 / L.win.f) : 8);
    if (s < 0) return false;
    L.res = await solve(L, { idx: [s], w: [0], t: [0], hz: [0] }, () => true, false);
    L.dist = cellTimes(L, L.res, false);
    return true;
}
function timeAt(L, lat, lng) { return timeAtP(plainP(L), L, L.dist, lat, lng); }
const offset = (lat, lng, km, b) => [lat + km * Math.cos(b * Math.PI / 180) / KM, lng + km * Math.sin(b * Math.PI / 180) / (KM * Math.cos(lat * Math.PI / 180))];
const q = (a, p) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const fmt = (a) => a.length ? `${q(a, 0.5).toFixed(2)} (${q(a, 0.1).toFixed(2)}-${q(a, 0.9).toFixed(2)})` : '-';
const w10 = (a) => a.length ? (100 * a.filter((x) => x >= 0.9 && x <= 1.1).length / a.length).toFixed(0) + '%' : '-';
const mal = (a) => a.length ? (a.reduce((s, x) => s + Math.abs(Math.log(x)), 0) / a.length).toFixed(3) : '-';
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

console.log(`p7: var ${VAR}, roads ${ROADS ? 'on' : 'off'}, L10 split ${P10.nSplit || 0}, extra nodes ${P10.nExtra || 0}, listed ${P10.nListed || 0}, blocked ${P10.nBlocked || 0}${REC3.length ? ', L3 patch tiles ' + REC3.length : ''}`);
const all = { r110: [], r13: [], inside: [], step: [] };
for (const [name, kind, lat, lng] of STARTS) {
    if (ONLY && !ONLY.includes(name)) continue;
    const cos = Math.cos(lat * Math.PI / 180), b = box(lat, lng, HALF, HALF / cos);
    const w1 = win(1, b.west, b.south, b.east, b.north);
    const L1 = await level(w1);
    seed = 12345 + Math.round(lat * 100);
    const dests = [];
    for (const D of DISTS) for (let j = 0; j < BEARINGS; j++) {
        const [la, lo] = offset(lat, lng, D * (0.85 + 0.3 * rnd()), 360 * (j + rnd()) / BEARINGS);
        if (la > b.north - 0.3 || la < b.south + 0.3 || lo < b.west + 0.3 || lo > b.east - 0.3) continue;
        const c = Math.floor((lo - w1.west) / w1.cell), r = Math.floor((w1.north - la) / w1.cell);
        if (L1.grid[r * w1.cols + c]) dests.push([la, lo]);
    }
    await solveFrom(L1, lat, lng);
    // L10
    let dist;
    const tt = [];
    for (let rep = 0; rep < REPS; rep++) {
        const t0 = performance.now();
        if (!MINE) { L10.res = await solve(L10, { idx: [snap(L10, lng, lat, 8)], w: [0], t: [0], hz: [0] }, () => true, false); dist = L10.res.t; }
        else dist = solveY(L10, P10, { idx: [startNode(P10, L10, L1, lat, lng)], w: [0] });
        tt.push(performance.now() - t0);
    }
    const one = [];
    for (const [la, lo] of dests) {
        const t1 = timeAt(L1, la, lo), t10 = timeAtP(P10, L10, dist, la, lo);
        if (!(t1 < cfg.maxHours && t10 < cfg.maxHours) || t1 <= 0) continue;
        one.push(t1 / t10);
    }
    all.r110.push(...one);
    // L3
    let l3txt = '';
    if (ARGS.includes('--l3')) {
        const w3 = win(3, b.west, b.south, b.east, b.north);
        const L3 = await level(w3);
        let P3 = plainP(L3), d3;
        if ((VAR === 'B' || VAR === 'Blite') && REC3.length)
            P3 = makePatch(L3, REC3, (g) => { const r = Math.floor(g / 14400) - w3.row0, c = g % 14400 - w3.col0; return r >= 0 && r < w3.rows && c >= 0 && c < w3.cols ? r * w3.cols + c : -1; }, VAR === 'Blite');
        if (MINE) { buildMasks(L3, P3); d3 = solveY(L3, P3, { idx: [startNode(P3, L3, L1, lat, lng)], w: [0] }); }
        else { await solveFrom(L3, lat, lng); d3 = L3.dist; }
        const r13 = [];
        for (const [la, lo] of dests) { const t1 = timeAt(L1, la, lo), t3 = timeAtP(P3, L3, d3, la, lo); if (t1 < cfg.maxHours && t3 < cfg.maxHours && t1 > 0) r13.push(t1 / t3); }
        all.r13.push(...r13);
        l3txt = `  L1/L3 ${fmt(r13)} ${w10(r13)} mean|ln| ${mal(r13)}${P3.nSplit !== undefined ? ` (L3 split ${P3.nSplit}, extra ${P3.nExtra})` : ''}`;
    }
    let seamTxt = '';
    if (ARGS.includes('--seams')) {
        const ins = [], stp = [];
        for (const [dkm, bear] of [[120, 90], [120, 270], [200, 0], [200, 180]]) {
            const [cl, cg] = offset(lat, lng, dkm, bear);
            const ww = win(1, Math.floor((cg - 0.75 / cos) * 4) / 4, Math.floor((cl - 0.5) * 4) / 4, Math.ceil((cg + 0.75 / cos) * 4) / 4, Math.ceil((cl + 0.5) * 4) / 4);
            if (ww.west < b.west || ww.east > b.east || ww.south < b.south || ww.north > b.north) continue;
            const L = await level(ww);
            const seeds = { idx: [], w: [], t: [], hz: [], pred: [] };
            const add = (r, c) => {
                const i = r * ww.cols + c;
                if (!L.grid[i]) return;
                const t = sample10(P10, dist, ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell);
                if (!(t < Infinity)) return;
                seeds.idx.push(i); seeds.w.push(t); seeds.t.push(t); seeds.hz.push(0); seeds.pred.push(P_SEED);
            };
            for (let c = 0; c < ww.cols; c++) { add(0, c); add(ww.rows - 1, c); }
            for (let r = 1; r < ww.rows - 1; r++) { add(r, 0); add(r, ww.cols - 1); }
            const rr = await solve(L, seeds, () => true, false);
            const dr = ww.row0 - w1.row0, dc = ww.col0 - w1.col0, qr = Math.round(ww.rows / 4), qc = Math.round(ww.cols / 4);
            const inside = [], step = [];
            for (let r = 0; r < ww.rows; r++) for (let c = 0; c < ww.cols; c++) {
                const t = rr.t[r * ww.cols + c];
                if (!(t < Infinity)) continue;
                const tw = L1.dist[(r + dr) * w1.cols + c + dc];
                if (tw < Infinity && tw > 0) inside.push(t / tw);
                const ring = ((r === qr || r === ww.rows - 1 - qr) && c >= qc && c < ww.cols - qc) || ((c === qc || c === ww.cols - 1 - qc) && r >= qr && r < ww.rows - qr);
                if (ring) { const s = sample10(P10, dist, ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell); if (s > 0) step.push(t / s); }
            }
            if (inside.length) ins.push(q(inside, 0.5));
            if (step.length) stp.push(q(step, 0.5));
        }
        all.inside.push(...ins); all.step.push(...stp);
        seamTxt = `  seams inside ${fmt(ins)}, step ${fmt(stp)}`;
    }
    console.log(`${name.padEnd(14)} ${kind.padEnd(12)} n ${String(one.length).padStart(3)}  L1/L10 ${fmt(one)} ${w10(one)}  mean|ln| ${mal(one)}  L10 ${q(tt, 0.5).toFixed(0)} ms${l3txt}${seamTxt}`);
}
if (!ONLY || ONLY.length > 1)
    console.log(`${'all'.padEnd(27)} n ${String(all.r110.length).padStart(3)}  L1/L10 ${fmt(all.r110)} ${w10(all.r110)}  mean|ln| ${mal(all.r110)}${all.r13.length ? `  L1/L3 ${fmt(all.r13)} ${w10(all.r13)} mean|ln| ${mal(all.r13)}` : ''}${all.inside.length ? `  seams inside ${fmt(all.inside)}, step ${fmt(all.step)}` : ''}`);
process.exit(0);
