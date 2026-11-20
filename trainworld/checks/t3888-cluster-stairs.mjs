// T-3888: the stairs flags (ferry lines, stops at grade) reach every population's network. The demand worker
// solves each population on a cluster's sub-network or a city's reach of it (demandClusters.ts `subNetwork`,
// `reachView`), each a new network that once dropped `pierLines` and `gradeStops`: `set_stairs` then saw none and
// every stop paid. No browser; Vite loads the module (about 5 s).
//   node trainworld/checks/t3888-cluster-stairs.mjs      (from the maps root or a tree's)
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = (process.env.TW_APP_DIR ?? fileURLToPath(new URL('../app', import.meta.url))).replace(/\\/g, '/');
const require = createRequire(root + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const server = await createServer({ root, configFile: root + '/vite.config.ts', logLevel: 'error', server: { middlewareMode: true, watch: null, hmr: false } });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
// (cities.ts reads the page's address and storage when it loads)
globalThis.location ??= new URL('http://localhost/');
globalThis.localStorage ??= { getItem: () => null, setItem() {}, removeItem() {} };
try {
  const C = await server.ssrLoadModule('/src/workers/demandClusters.ts');
  const { PERIODS, TIMES_PER_STOP } = await server.ssrLoadModule('/src/game/periods.ts');
  // three lines in a row, 3 km apart: a train (stations 0-2, at grade, below, at grade), a ferry (2-4), a train (4-6)
  const lineN = [3, 3, 3], stopsOf = [0, 1, 2, 2, 3, 4, 4, 5, 6];
  const nStops = stopsOf.length, S = 7;
  const grade = [1, 0, 1, 0, 0, 0, 0, 1, 0], pier = [0, 1, 0];
  const n = {
    frame: { lon: -73.93, lat: 40.67 },
    stXY: Float32Array.from({ length: 2 * S }, (_, i) => (i % 2 ? 0 : 3000 * (i / 2))),
    complex: Uint32Array.from({ length: S }, (_, i) => i),
    lineN: Uint32Array.from(lineN), stops: Uint32Array.from(stopsOf),
    times: new Float32Array(TIMES_PER_STOP * nStops).fill(60), tph: new Float32Array(PERIODS.length * lineN.length).fill(6),
    cars: new Float32Array(lineN.length).fill(4), fare: { base: 2.5, perKm: 0.2 }, hopKm: new Float32Array(2 * nStops).fill(3),
    span: new Float32Array(0), ferry: Uint8Array.from(pier), pierLines: Uint8Array.from(pier), gradeStops: Uint8Array.from(grade),
  };
  const box = [-1e9, -1e9, 1e9, 1e9];
  const [cl] = C.splitClusters(n, [{ id: 'nyc', box }], null, 1);
  check(!!cl && cl.lines.length === 3, 'one cluster of the three lines', cl ? `${cl.lines.length} lines` : 'none');
  // the cluster's stop k is line cl.lines[i]'s; its flags must be the whole network's at that stop
  const at = (net, lines, starts) => [...lines].flatMap((l, i) => Array.from({ length: net.lineN[i] }, (_, k) => starts[l] + k));
  const start = [0, 3, 6];
  const want = (net, lines, starts) => at(net, lines, starts).map((k) => grade[k]).join('');
  check(cl.net.gradeStops?.length === cl.net.stops.length && [...cl.net.gradeStops].join('') === want(cl.net, cl.lines, start), 'the cluster keeps the stops at grade', `${cl.net.gradeStops ? [...cl.net.gradeStops].join('') : 'none'}`);
  check([...(cl.net.pierLines ?? [])].join('') === [...cl.lines].map((l) => pier[l]).join(''), 'the cluster keeps the ferry lines', `${cl.net.pierLines ? [...cl.net.pierLines].join('') : 'none'}`);
  // a city reaching stations 2-3 only: line 0 trimmed to its last 2 stops (the one reached and one before), the
  // ferry whole, the last line not at all
  const reach = Uint8Array.from({ length: S }, (_, s) => (s === 2 || s === 3 ? 1 : 0));
  const v = C.reachView(n, cl, reach);
  const g = v.net.gradeStops, p = v.net.pierLines;
  check(!!g && g.length === v.net.stops.length, 'a reach view keeps one at-grade flag per stop', g ? `${g.length} flags, ${v.net.stops.length} stops` : 'none');
  // each kept stop's flag is its own: the view's stop k (line i's j-th) is run 0's route node 2o + j, the whole
  // network's node 2g + a + j of line l (g its first stop), so the whole network's stop g + a + j = node - g
  const wholeStop = (k) => {
    let o = 0, i = 0;
    while (o + v.net.lineN[i] <= k) o += v.net.lineN[i++];
    const node = v.nodeGlobal[2 * o + (k - o)];
    let l = 0;
    while (l + 1 < start.length && 2 * start[l + 1] <= node) l++;
    return node - start[l];
  };
  const got = g ? [...g].join('') : 'none', exp = Array.from({ length: v.net.stops.length }, (_, k) => grade[wholeStop(k)]).join('');
  check(got === exp, 'a reach view keeps each stop its own flag', `${got} against ${exp}`);
  check(v.net.stops.length === 5, 'the reach view is trimmed as meant', `${v.net.stops.length} stops`);
  check(p?.length === v.net.lineN.length && [...p].join('') === '01', 'a reach view keeps one ferry flag per line', p ? [...p].join('') : 'none');
} finally {
  await server.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
