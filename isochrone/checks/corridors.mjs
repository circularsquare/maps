// Corridor tree check (T-090): the worker's modules in node over the real L10 data, from the maps root:
//   node isochrone/checks/corridors.mjs [--tiles=<name>]   (~3 min: two canoe starts and a walking one)
// For each start, after corridors.count and lines:
//   - every reached node's parent is reached within the time limit at no more weight (a parent read
//     back from the codes that is not is outside the counted tree, and leaves the shared work
//     arrays dirty);
//   - every drawn line ends at the start or on another line;
//   - every land node's own move (the search's), costed by corridors.moveWeight, gives its weight
//     (T-091: corridors costs other parents' moves the same way, directed);
// and the first start counted again after the others gives the same lines (no state carried over).
// One PASS or FAIL line each (10); exit 1 on a fail.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');
globalThis.fetch = async (u) => new Response(readFileSync(String(u).replace(/^file:\/\//, '')));
const imp = (m) => import(pathToFileURL(ROOT + 'js/' + m).href);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, setHazard, setDirs } = await imp('levels.js');
const { solve, P_NONE, P_GRID, MDC, MDR } = await imp('search.js');
const { loadWorld } = await imp('canoedata.js');
const { buildCanoe } = await imp('canoe.js');
const { loadHazardWorld, applyRoads } = await imp('tiles.js');
const corridors = await imp('corridors.js');

// --tiles=<name>: the walking levels of data/tiles_<name>/ (the page's ?tiles= switch)
const SUB = (process.argv.find((a) => a.startsWith('--tiles=')) ?? '').slice(8).replace(/[^a-z0-9_-]/gi, '');
const BASE = ROOT + 'data/' + (SUB ? 'tiles_' + SUB + '/' : '');
const meta = JSON.parse(readFileSync(BASE + 'grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: ROOT + 'data/canoe/', hazardBase: ROOT + 'data/hazard/' });
const gz = (f) => { const b = gunzipSync(readFileSync(BASE + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
const L = makeLevel(gz('grid.bin.gz'), meta.width, meta.height, meta.west, meta.north, (meta.north - meta.south) / meta.height, true);
if (cfg.roads && meta.roads) applyRoads(L.grid, gz(meta.roads));
setDirs(L, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);
setHazard(L, await loadHazardWorld());
const d = await loadWorld(L.W, L.H);
buildCanoe(L, d, d.sea);

const results = [];
const report = (name, ok, detail = '') => {
    results.push(ok);
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

// Semnan (Anita's case, 2026-11-06: lines from the Caspian north drawn in pieces), Constanta, and
// Semnan on foot
const STARTS = [['Semnan by canoe', 53.39, 35.58, true], ['Constanta by canoe', 28.63, 44.17, true], ['Semnan on foot', 53.39, 35.58, false]];
const solved = [];
for (const [name, lng, lat, canoe] of STARTS) {
    const st = snap(L, lng, lat, 8);
    solved.push({ name, canoe, st, res: await solve(L, { idx: [st], w: [0], t: [0], hz: [0] }, () => true, canoe) });
}

function countAndDraw(s) {
    L.res = s.res;
    const c = corridors.count(L, s.canoe, null);
    return { c, ln: corridors.lines(L, s.canoe, c.total, null) };
}

let first = null;
for (const s of solved) {
    const { c, ln } = countAndDraw(s);
    const { t, w, pred } = s.res;
    let bad = 0, ex = '';
    for (let n = 0; n < t.length; n++) {
        if (pred[n] === P_NONE || !(t[n] <= cfg.maxHours)) continue;
        const p = c.par[n];
        if (p < 0 || (t[p] <= cfg.maxHours && w[p] <= w[n])) continue;
        if (!bad) ex = ` e.g. node ${n} at ${t[n].toFixed(1)} h, parent ${p} at ${t[p]} h`;
        bad++;
    }
    report(`${s.name}: parents reached before their children`, bad === 0, `(${c.nr} nodes, ${bad} not${ex})`);
    // the search's own land moves as corridors costs them
    let moves = 0, off = 0, worst = 0;
    for (let v = 0; v < L.W * L.H; v++) {
        const code = pred[v];
        if (!L.grid[v] || code < P_GRID || code >= P_GRID + 16 || !(t[v] <= cfg.maxHours)) continue;
        const k = code - P_GRID, r = (v / L.W) | 0;
        const u = (r - MDR[k]) * L.W + ((v - r * L.W - MDC[k]) % L.W + L.W) % L.W;
        const e = Math.abs(w[u] + corridors.moveWeight(L, s.canoe, u, v, k) - w[v]) / Math.max(1, w[v]);
        moves++;
        if (!(e <= 2e-6)) off++;
        if (!(e <= worst)) worst = e;
    }
    report(`${s.name}: own moves cost their weight`, moves > 1000 && off === 0, `(${moves} moves, ${off} off, worst ${worst.toExponential(1)})`);
    // each line's last point (towards the start) is the start's cell or a point of another line
    const key = (x, y) => x.toFixed(4) + ',' + y.toFixed(4);
    const seen = new Map();
    const nl = ln.s.length;
    for (let i = 0; i < nl; i++) for (let j = ln.off[i]; j < ln.off[i + 1]; j++) {
        const k = key(ln.xy[2 * j], ln.xy[2 * j + 1]);
        seen.set(k, (seen.get(k) || 0) + 1);
    }
    const sx = L.west + ((s.st % L.W) + 0.5) * L.cellDeg, sy = L.north - (((s.st / L.W) | 0) + 0.5) * L.cellDeg;
    let loose = 0;
    for (let i = 0; i < nl; i++) {
        const j = ln.off[i + 1] - 1, x = ln.xy[2 * j], y = ln.xy[2 * j + 1];
        if (key(x, y) === key(sx, sy) || seen.get(key(x, y)) > 1) continue;
        loose++;
    }
    report(`${s.name}: lines lead back to the start`, nl > 100 && loose === 0, `(${nl} lines, ${loose} loose ends)`);
    if (!first) first = Array.from(ln.xy);
}
// the first start again, after the others: the same lines
const again = Array.from(countAndDraw(solved[0]).ln.xy);
const same = again.length === first.length && again.every((v, i) => v === first[i]);
report(`${solved[0].name} counted again gives the same lines`, same, `(${first.length / 2} then ${again.length / 2} points)`);

const failed = results.filter((ok) => !ok).length;
console.log(failed ? `CORRIDORS FAILED: ${failed} of ${results.length}` : `CORRIDORS PASSED: ${results.length} checks`);
process.exit(failed ? 1 : 0);
