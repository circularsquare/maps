// Canoe data files (docs/canoe.md 4): fetched, parsed and cut to a level. Everything that
// knows the byte layouts is here; js/canoe.js builds the graph from what this returns.
import { cfg } from './config.js';
import { fetchGz, getCached, forTiles, cutInto } from './tiles.js';

// Directions of link bits and river steps in the files, as (dc, dr) with dr + to the south.
export const FILE_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
// River step classes in the files; the ones a canoe cannot run are dropped (portage).
export const CLS = { flat: 0, rapids: 1, unrunnable: 2, fall: 3, dam: 4, weir: 5, reservoir: 6 };
export const CLOSED = new Uint8Array(256);
for (const k of ['unrunnable', 'fall', 'dam', 'weir', 'reservoir']) CLOSED[CLS[k]] = 1;
// sea.bin.gz: bands u, v (int8, 0.02 m/s, east and north), wave height (0.05 m), ice (%), sst
// and lake temperature (int8, degrees C; without the lake band lakes read sst); its grid in
// index.json `sea`
const SEA_UNIT = 0.02;

let indexP = null, worldP = null;
const url = (p) => cfg.canoeBase + p;
const index = () => indexP || (indexP = fetch(url('index.json')).then(r => {
    if (!r.ok) throw new Error('canoe index ' + r.status);
    return r.json();
}).then(ix => {
    for (const k in ix.levels) if (ix.levels[k].tiles) ix.levels[k].set = new Set(ix.levels[k].tiles);
    for (const k in ix.rivers) ix.rivers[k].set = new Set(ix.rivers[k].tiles);
    for (const k in ix.pieces || {}) if (ix.pieces[k].tiles) ix.pieces[k].set = new Set(ix.pieces[k].tiles);
    return ix;
}));

// River steps: uint32 n, then cell uint32[n], len uint16[n] (10 m), dir, v (0.01 m/s), cls uint8[n];
// at L3 and L10 then slotA, slotB, accA, accB uint8[n] (T-082: which canoe node of its cell each end
// is on, and that river piece's access, 0.1 km). Without them every end is slot 0, access 0.
function parseRivers(b) {
    const n = new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true);
    let o = 4;
    const cell = new Uint32Array(b.slice(o, o + 4 * n).buffer); o += 4 * n;
    const len = new Uint16Array(b.slice(o, o + 2 * n).buffer); o += 2 * n;
    const dir = b.subarray(o, o + n); o += n;
    const v = b.subarray(o, o + n); o += n;
    const cls = b.subarray(o, o + n); o += n;
    const more = b.length >= o + 4 * n;
    const col = () => { const c = b.subarray(o, o + n); o += n; return c; };
    const [slotA, slotB, accA, accB] = more ? [col(), col(), col(), col()] : [null, null, null, null];
    return { n, cell, len, dir, v, cls, slotA, slotB, accA, accB };
}

// Steps as window cells, both ends inside: { a, b (cell indices), dir, km, v (m/s), cls, sa, sb
// (slots), aa, ab (access, km) }.
function stepsOut(cap) {
    return { n: 0, a: new Int32Array(cap), b: new Int32Array(cap), dir: new Uint8Array(cap),
             km: new Float32Array(cap), v: new Float32Array(cap), cls: new Uint8Array(cap),
             sa: new Uint8Array(cap), sb: new Uint8Array(cap), aa: new Float32Array(cap), ab: new Float32Array(cap) };
}
function addSteps(out, rv, toCell) {
    for (let i = 0; i < rv.n; i++) {
        const d = FILE_DIRS[rv.dir[i]];
        if (!d) continue;
        const ab = toCell(rv.cell[i], d[0], d[1]);
        if (!ab) continue;
        const k = out.n++;
        out.a[k] = ab[0]; out.b[k] = ab[1]; out.dir[k] = rv.dir[i];
        out.km[k] = rv.len[i] * 0.01; out.v[k] = rv.v[i] * 0.01; out.cls[k] = rv.cls[i];
        if (rv.slotA) {
            out.sa[k] = rv.slotA[i]; out.sb[k] = rv.slotB[i];
            out.aa[k] = rv.accA[i] * 0.1; out.ab[k] = rv.accB[i] * 0.1;
        }
    }
}

// Water units and links of a coarse level (T-088, tools/canoe/rivers.py pack_water): uint32 n, cell
// uint32[n], slot, kind, shore, access (0.1 km) uint8[n]; then uint32 m, cell uint32[m], dir, slotA,
// slotB uint8[m]. Units are the water nodes the cell bands do not give (further slots, and slot 0
// of a river-water cell); a link joins unit slotA of its cell to unit slotB of the neighbour in dir.
function parsePieces(b) {
    const dv = new DataView(b.buffer, b.byteOffset, b.length);
    const n = dv.getUint32(0, true);
    let o = 4;
    const cell = new Uint32Array(b.slice(o, o + 4 * n).buffer); o += 4 * n;
    const col = (k) => { const c = b.subarray(o, o + k); o += k; return c; };
    const slot = col(n), kind = col(n), shore = col(n), acc = col(n);
    const m = dv.getUint32(o, true); o += 4;
    const lcell = new Uint32Array(b.slice(o, o + 4 * m).buffer); o += 4 * m;
    const ldir = col(m), lsa = col(m), lsb = col(m);
    return { n, cell, slot, kind, shore, acc, m, lcell, ldir, lsa, lsb };
}
function piecesOut(nu, nl) {
    return { n: 0, cell: new Int32Array(nu), slot: new Uint8Array(nu), kind: new Uint8Array(nu), shore: new Uint8Array(nu),
             acc: new Float32Array(nu), m: 0, a: new Int32Array(nl), b: new Int32Array(nl), dir: new Uint8Array(nl),
             sa: new Uint8Array(nl), sb: new Uint8Array(nl) };
}
// toCell(c) -> level cell or -1; toPair(c, dc, dr) -> [a, b] or null, as for river steps
function addPieces(out, pv, toCell, toPair) {
    for (let i = 0; i < pv.n; i++) {
        const c = toCell(pv.cell[i]);
        if (c < 0) continue;
        const k = out.n++;
        out.cell[k] = c; out.slot[k] = pv.slot[i]; out.kind[k] = pv.kind[i]; out.shore[k] = pv.shore[i];
        out.acc[k] = pv.acc[i] * 0.1;
    }
    for (let i = 0; i < pv.m; i++) {
        const d = FILE_DIRS[pv.ldir[i]];
        if (!d) continue;
        const ab = toPair(pv.lcell[i], d[0], d[1]);
        if (!ab) continue;
        const k = out.m++;
        out.a[k] = ab[0]; out.b[k] = ab[1]; out.dir[k] = pv.ldir[i]; out.sa[k] = pv.lsa[i]; out.sb[k] = pv.lsb[i];
    }
}

function parseSea(b, g) {
    const n = g.width * g.height;
    const sst = new Int8Array(b.buffer, b.byteOffset + 4 * n, n);
    return { W: g.width, H: g.height, deg: g.step, west: g.west, north: g.north, unit: SEA_UNIT,
             u: new Int8Array(b.buffer, b.byteOffset, n), v: new Int8Array(b.buffer, b.byteOffset + n, n),
             hs: b.subarray(2 * n, 3 * n), ice: b.subarray(3 * n, 4 * n), sst,
             lake: b.length >= 6 * n ? new Int8Array(b.buffer, b.byteOffset + 5 * n, n) : sst };
}

// The world at L10: water bands, sea grid and river steps. Loaded once.
export function loadWorld(W, H) {
    return worldP || (worldP = (async () => {
        const ix = await index();
        const pf = ix.pieces && ix.pieces['10'];
        const [w, s, r, pc] = await Promise.all([fetchGz(url(ix.levels['10'].file)), fetchGz(url(ix.sea.file)),
                                                 fetchGz(url('rivers.bin.gz')), pf ? fetchGz(url(pf.file)) : null]);
        const N = W * H;
        const rv = parseRivers(r);
        const steps = stepsOut(rv.n);
        let ab = [0, 0];
        const toPair = (cell, dc, dr) => {
            const row = (cell / W) | 0, col = cell - row * W, r2 = row + dr;
            if (r2 < 0 || r2 >= H) return null;
            ab[0] = cell; ab[1] = r2 * W + (col + dc + W) % W;
            return ab;
        };
        addSteps(steps, rv, toPair);
        let pieces = null;
        if (pc) {
            const pv = parsePieces(pc);
            pieces = piecesOut(pv.n, pv.m);
            addPieces(pieces, pv, (c) => c, toPair);
        }
        return { kind: w.subarray(0, N), shore: w.subarray(N, 2 * N), links: w.subarray(2 * N, 3 * N),
                 sea: parseSea(s, ix.sea), steps, pieces };
    })().catch(e => { worldP = null; throw e; }));
}

// A window at L3 or L1: water bands (links all set at L1, where they are implicit) and the river
// steps inside it. A tile the index does not list has no land: open sea, shore from `shoreAt`.
export async function loadWindowWater(w, grid, shoreAt) {
    const ix = await index();
    const lv = ix.levels[w.f], rl = ix.rivers[w.f], TX = ix.tilesX;
    const T = lv.tile, n = w.cols * w.rows, bands = w.f === 1 ? 2 : 3;
    const kind = new Uint8Array(n), shore = new Uint8Array(n), links = new Uint8Array(n).fill(255);
    const jobs = [], missing = [];
    forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!lv.set.has(tr * TX + tcw)) { missing.push([tr, tc]); return; }
        jobs.push(getCached(url(lv.path + '/' + tr + '_' + tcw + '.bin.gz'), bands * T * T).then(t => {
            cutInto(kind, w, t, 0, T, tr, tc);
            cutInto(shore, w, t, 1, T, tr, tc);
            if (bands === 3) cutInto(links, w, t, 2, T, tr, tc);
        }));
    });
    const raw = [], rawP = [];
    if (rl) forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!rl.set.has(tr * TX + tcw)) return;
        jobs.push(getCached(url(rl.path + '/' + tr + '_' + tcw + '.bin.gz'), 0)
            .then(b => raw.push([tr, tc, parseRivers(b)])));
    });
    const pl = ix.pieces && ix.pieces[w.f];
    if (pl) forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!pl.set.has(tr * TX + tcw)) return;
        jobs.push(getCached(url(pl.path + '/' + tr + '_' + tcw + '.bin.gz'), 0)
            .then(b => rawP.push([tr, tc, parsePieces(b)])));
    });
    await Promise.all(jobs);
    for (const [tr, tc] of missing) {
        const r0 = Math.max(0, tr * T - w.row0), r1 = Math.min(w.rows, (tr + 1) * T - w.row0);
        const c0 = Math.max(0, tc * T - w.col0), c1 = Math.min(w.cols, (tc + 1) * T - w.col0);
        for (let r = r0; r < r1; r++) {
            const lat = w.north - (r + 0.5) * w.cell;
            for (let c = c0; c < c1; c++) {
                const i = r * w.cols + c;
                if (grid[i]) continue;
                kind[i] = 1;
                shore[i] = shoreAt(w.west + (c + 0.5) * w.cell, lat);
            }
        }
    }
    const steps = stepsOut(raw.reduce((s, x) => s + x[2].n, 0));
    let ab = [0, 0];
    const pairIn = (rr, cc) => (cell, dc, dr) => {
        const lr = (cell / T) | 0, r = rr + lr, c = cc + cell - lr * T;
        const r2 = r + dr, c2 = c + dc;
        if (r < 0 || r >= w.rows || c < 0 || c >= w.cols || r2 < 0 || r2 >= w.rows || c2 < 0 || c2 >= w.cols) return null;
        ab[0] = r * w.cols + c; ab[1] = r2 * w.cols + c2;
        return ab;
    };
    for (const [tr, tc, rv] of raw) addSteps(steps, rv, pairIn(tr * T - w.row0, tc * T - w.col0));
    let pieces = null;
    if (pl) {
        pieces = piecesOut(rawP.reduce((s, x) => s + x[2].n, 0), rawP.reduce((s, x) => s + x[2].m, 0));
        for (const [tr, tc, pv] of rawP) {
            const rr = tr * T - w.row0, cc = tc * T - w.col0;
            addPieces(pieces, pv, (cell) => {
                const lr = (cell / T) | 0, r = rr + lr, c = cc + cell - lr * T;
                return r < 0 || r >= w.rows || c < 0 || c >= w.cols ? -1 : r * w.cols + c;
            }, pairIn(rr, cc));
        }
    }
    return { kind, shore, links, steps, pieces };
}
