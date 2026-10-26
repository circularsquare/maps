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
// sea.bin.gz: bands u, v (int8, 0.02 m/s, east and north), wave height, ice (%), sst; its grid in
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
    return ix;
}));

// River steps: uint32 n, then cell uint32[n], len uint16[n] (10 m), dir, v (0.01 m/s), cls uint8[n].
function parseRivers(b) {
    const n = new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true);
    let o = 4;
    const cell = new Uint32Array(b.slice(o, o + 4 * n).buffer); o += 4 * n;
    const len = new Uint16Array(b.slice(o, o + 2 * n).buffer); o += 2 * n;
    const dir = b.subarray(o, o + n); o += n;
    const v = b.subarray(o, o + n); o += n;
    const cls = b.subarray(o, o + n);
    return { n, cell, len, dir, v, cls };
}

// Steps as window cells, both ends inside: { a, b (cell indices), dir, km, v (m/s), cls }.
function stepsOut(cap) {
    return { n: 0, a: new Int32Array(cap), b: new Int32Array(cap), dir: new Uint8Array(cap),
             km: new Float32Array(cap), v: new Float32Array(cap), cls: new Uint8Array(cap) };
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
    }
}

function parseSea(b, g) {
    const n = g.width * g.height;
    return { W: g.width, H: g.height, deg: g.step, west: g.west, north: g.north, unit: SEA_UNIT,
             u: new Int8Array(b.buffer, b.byteOffset, n), v: new Int8Array(b.buffer, b.byteOffset + n, n),
             ice: b.subarray(3 * n, 4 * n) };
}

// The world at L10: water bands, sea grid and river steps. Loaded once.
export function loadWorld(W, H) {
    return worldP || (worldP = (async () => {
        const ix = await index();
        const [w, s, r] = await Promise.all([fetchGz(url(ix.levels['10'].file)), fetchGz(url(ix.sea.file)),
                                            fetchGz(url('rivers.bin.gz'))]);
        const N = W * H;
        const rv = parseRivers(r);
        const steps = stepsOut(rv.n);
        let ab = [0, 0];
        addSteps(steps, rv, (cell, dc, dr) => {
            const row = (cell / W) | 0, col = cell - row * W, r2 = row + dr;
            if (r2 < 0 || r2 >= H) return null;
            ab[0] = cell; ab[1] = r2 * W + (col + dc + W) % W;
            return ab;
        });
        return { kind: w.subarray(0, N), shore: w.subarray(N, 2 * N), links: w.subarray(2 * N, 3 * N),
                 sea: parseSea(s, ix.sea), steps };
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
    const raw = [];
    if (rl) forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!rl.set.has(tr * TX + tcw)) return;
        jobs.push(getCached(url(rl.path + '/' + tr + '_' + tcw + '.bin.gz'), 0)
            .then(b => raw.push([tr, tc, parseRivers(b)])));
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
    for (const [tr, tc, rv] of raw) {
        const rr = tr * T - w.row0, cc = tc * T - w.col0;
        addSteps(steps, rv, (cell, dc, dr) => {
            const lr = (cell / T) | 0, r = rr + lr, c = cc + cell - lr * T;
            const r2 = r + dr, c2 = c + dc;
            if (r < 0 || r >= w.rows || c < 0 || c >= w.cols || r2 < 0 || r2 >= w.rows || c2 < 0 || c2 >= w.cols) return null;
            ab[0] = r * w.cols + c; ab[1] = r2 * w.cols + c2;
            return ab;
        });
    }
    return { kind, shore, links, steps };
}
