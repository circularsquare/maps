// T-3903: every per-line, per-stop, per-station and per-route-node field of a DemandNetwork reaches the networks the
// demand worker solves on (demandClusters.ts `splitClusters`' cluster networks and `reachView`). A field a view
// dropped was read as none (the tram flag, T-2926; the stairs flags, T-3888). The fixture has every field set where it
// survives the trim; the check fails for a field the full network has that a view lacks or holds at the wrong length,
// and for a field in demandProtocol.ts this list doesn't know (add it here, with its shape).
// No browser; Vite loads the module (about 5 s).
//   node trainworld/checks/t3903-view-fields.mjs      (from the maps root or a tree's)
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = (process.env.TW_APP_DIR ?? fileURLToPath(new URL('../app', import.meta.url))).replace(/\\/g, '/');
const require = createRequire(root + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const server = await createServer({ root, configFile: root + '/vite.config.ts', logLevel: 'error', server: { middlewareMode: true, watch: null, hmr: false } });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
globalThis.location ??= new URL('http://localhost/');
globalThis.localStorage ??= { getItem: () => null, setItem() {}, removeItem() {} };

// each field's shape: entries per station / line / stop / route node (2 a stop), times a factor; 'view' = made for the
// view (frame, fare), 'part' = a piece of a variable-length group, 'skip' = never sent to a solve
const SHAPE = {
  frame: 'view', fare: 'view',
  stXY: ['station', 2], complex: ['station', 1], ups: ['station', 1], fz: ['station', 1],
  passages: 'part',
  lineN: ['line', 1], tph: ['line', 3], cars: ['line', 1], ferry: ['line', 1], tram: ['line', 1], pierLines: ['line', 1],
  stops: ['stop', 1], back: ['stop', 1], streetStops: ['stop', 1], catchN: ['stop', 1], catchXY: 'part', times: ['stop', 6],
  hopKm: ['node', 1], fareW: ['node', 1], baseW: ['node', 1], fzDist: ['node', 1], span: ['node', 3], fzPassOff: 'part', fzPass: 'part',
  fareS: 'skip', baseS: 'skip',
};

try {
  const C = await server.ssrLoadModule('/src/workers/demandClusters.ts');
  const { PERIODS, TIMES_PER_STOP } = await server.ssrLoadModule('/src/game/periods.ts');
  const src = readFileSync(root + '/src/workers/demandProtocol.ts', 'utf8');
  const body = src.slice(src.indexOf('export interface DemandNetwork'));
  const names = [...body.slice(0, body.indexOf('\n}')).matchAll(/^  (\w+)\??:/gm)].map((m) => m[1]);
  const unknown = names.filter((k) => !(k in SHAPE));
  check(names.length > 20 && !unknown.length, 'every DemandNetwork field has a shape in this check', unknown.length ? `unknown: ${unknown.join(', ')}` : `${names.length} fields`);
  const P = PERIODS.length;
  SHAPE.tph = ['line', P]; SHAPE.span = ['node', P]; SHAPE.times = ['stop', TIMES_PER_STOP];

  // lines of 3 stops in a row, 3 km apart: stations 0-2, 2-4 (line 1, every flag), 4-6
  const lineN = [3, 3, 3], stopsOf = [0, 1, 2, 2, 3, 4, 4, 5, 6];
  const nStops = stopsOf.length, S = 7, nodes = 2 * nStops;
  const zeros = (T, n) => new T(n);
  const set = (T, n, at, v = 1) => { const a = zeros(T, n); for (const i of at) a[i] = v; return a; };
  const n = {
    frame: { lon: -73.93, lat: 40.67 },
    stXY: Float32Array.from({ length: 2 * S }, (_, i) => (i % 2 ? 0 : 3000 * (i / 2))),
    complex: Uint32Array.from({ length: S }, (_, i) => i),
    ups: set(Uint8Array, S, [2, 3]), fz: set(Uint32Array, S, [2, 3], 7),
    passages: Uint32Array.from([2, 3]),
    lineN: Uint32Array.from(lineN), stops: Uint32Array.from(stopsOf),
    back: Uint32Array.from(stopsOf.map((s, k) => (k === 4 ? 4 : s))),
    times: new Float32Array(TIMES_PER_STOP * nStops).fill(60), tph: new Float32Array(P * 3).fill(6),
    cars: new Float32Array(3).fill(4), fare: { base: 2.5, perKm: 0.2 }, hopKm: new Float32Array(nodes).fill(3),
    span: new Float32Array(PERIODS.length * nodes).fill(0.5),
    fareW: new Float32Array(nodes).fill(1.5), baseW: new Float32Array(nodes).fill(1.25),
    catchN: set(Uint32Array, nStops, [3, 4], 2), catchXY: Float32Array.from([6000, 0, 6100, 0, 6000, 10, 6100, 10]),
    ferry: set(Uint8Array, 3, [1]), tram: set(Uint8Array, 3, [1]), pierLines: set(Uint8Array, 3, [1]), streetStops: set(Uint8Array, nStops, [3, 4]),
    fzDist: set(Uint8Array, nodes, [6, 7, 8]),
    fzPassOff: Uint32Array.from({ length: nodes + 1 }, (_, r) => (r <= 6 ? 0 : Math.min(2, r - 6))), fzPass: Uint32Array.from([7, 7]),
    fareS: new Float32Array(nodes).fill(1), baseS: new Float32Array(nodes).fill(1),
  };
  const missing = Object.keys(SHAPE).filter((k) => n[k] === undefined);
  check(!missing.length, 'the fixture sets every field', missing.join(', '));

  const box = [-1e9, -1e9, 1e9, 1e9];
  const [cl] = C.splitClusters(n, [{ id: 'nyc', box }], null, 1);
  const reach = Uint8Array.from({ length: S }, (_, s) => (s === 2 || s === 3 ? 1 : 0));
  const v = C.reachView(n, cl, reach);
  const views = { 'cluster network': cl.net, 'reach view': v.net };
  for (const [what, net] of Object.entries(views)) {
    const size = { station: net.stXY.length / 2, line: net.lineN.length, stop: net.stops.length, node: 2 * net.stops.length };
    const bad = [];
    for (const k of Object.keys(n)) {
      const shape = SHAPE[k];
      if (shape === 'skip' || shape === 'view') { if (shape === 'view' && net[k] === undefined) bad.push(`${k} missing`); continue; }
      if (net[k] === undefined || net[k].length === 0) { bad.push(`${k} missing`); continue; }
      if (Array.isArray(shape) && net[k].length !== size[shape[0]] * shape[1]) bad.push(`${k} has ${net[k].length}, wants ${size[shape[0]] * shape[1]} (${shape[1]} a ${shape[0]})`);
    }
    check(!bad.length, `a ${what} carries every field, one entry per its ${Object.keys(size).join(', ')}`, bad.join('; ') || `${size.line} lines, ${size.stop} stops`);
  }
  check(v.net.stops.length === 5 && cl.net.stops.length === 9, 'the views are cut as meant', `${cl.net.stops.length} and ${v.net.stops.length} stops`);
  check([...(cl.net.tram ?? [])].join('') === '010' && [...(v.net.tram ?? [])].join('') === '01', 'the tram flag follows its line', `${[...(cl.net.tram ?? [])]} and ${[...(v.net.tram ?? [])]}`);
} finally {
  await server.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
