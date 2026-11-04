// T-2405: a New York game with expresses on tracks of their own beside locals, built with the track model's
// edit API (WASM) in Node and written as a game file for the diagram checks (diagram-saves.mjs and
// diagram-game.mjs `xp`): Anita's line 8 case, a local along the Westchester and Connecticut shore from
// Williams Bridge to Fairfield and a high-speed line from Melrose through Williams Bridge to South End, from
// there 150 m north of the local's stations and stopping at none of them; a local and an express of one
// colour up the Hudson from Yankees, the express 80 m off the local's track and stopping at Yonkers only; a
// line through Williams Bridge to Hartsdale and one from Melrose to West Mount Vernon.
//   node trainworld/checks/t2405-express-game.mjs      (a second; writes artifacts/trainworld/t2405/express.save)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const EXPRESS_SAVE = fileURLToPath(new URL('../../artifacts/trainworld/t2405/express.save', import.meta.url));

/** the stations, [lat, lng] */
const AT = {
  'Melrose': [40.8253, -73.9154], 'Tremont': [40.8470, -73.8995], 'Fordham': [40.8615, -73.8905], 'Williams Bridge': [40.8785, -73.8712],
  'Woodlawn': [40.8958, -73.8627], 'Fleetwood': [40.9206, -73.8381], 'Bronxville': [40.9395, -73.8358], 'Crestwood Gardens': [40.9590, -73.8209], 'Hartsdale': [41.0302, -73.7862],
  'E Gun Hill Rd': [40.8695, -73.8466], 'Wakefield': [40.9032, -73.8450],
  'West Mount Vernon': [40.9025, -73.8300], 'Homestead Park': [40.9213, -73.7840], 'Rye Neck': [40.9541, -73.7361], 'Northeast Rye Neck': [40.9697, -73.7128], 'Port Chester': [41.0002, -73.6656],
  'Chickahominy': [41.0214, -73.6247], 'South End': [41.0469, -73.5418], 'South Norwalk': [41.0962, -73.4217], 'Fairfield': [41.1430, -73.2578],
  'Yankees': [40.8250, -73.9300], 'Spuyten Duyvil': [40.8784, -73.9213], 'Riverdale': [40.9040, -73.9140], 'Ludlow': [40.9246, -73.9046],
  'Yonkers': [40.9356, -73.9028], 'Hastings-on-Hudson': [40.9946, -73.8843], 'Dobbs Ferry': [41.0122, -73.8778], 'Tarrytown': [41.0763, -73.8648],
};
const LOCAL8 = ['Williams Bridge', 'West Mount Vernon', 'Homestead Park', 'Rye Neck', 'Northeast Rye Neck', 'Port Chester', 'Chickahominy', 'South End', 'South Norwalk', 'Fairfield'];
const HUDSON = ['Yankees', 'Spuyten Duyvil', 'Riverdale', 'Ludlow', 'Yonkers', 'Hastings-on-Hudson', 'Dobbs Ferry', 'Tarrytown'];
/** each line: colour, high-speed, its stops, and the places its track passes between stops, [lat, lng], keyed by
 * the stop it leaves; `side`: [east, north] metres its track and own stations keep off the stations' places */
const LINES = [
  { colour: '#7b3fbf', stops: ['Melrose', 'Tremont', 'Fordham', 'Williams Bridge', 'Woodlawn', 'Fleetwood', 'Bronxville', 'Crestwood Gardens', 'Hartsdale'] },
  { colour: '#1aa34a', stops: ['Melrose', 'E Gun Hill Rd', 'Wakefield', 'West Mount Vernon'] },
  { colour: '#b5122e', stops: LOCAL8 },
  // the high-speed line: Williams Bridge to South End on its own track 150 m north of the local's stations
  { colour: '#14a098', fast: true, stops: ['Melrose', 'Williams Bridge', 'South End'], via: { 'Williams Bridge': LOCAL8.slice(1, 7).map((s) => [AT[s][0] + 150 / 111195, AT[s][1]]) } },
  { colour: '#f28c28', stops: HUDSON },
  // the Hudson express, one colour with the local, 80 m west of its track
  { colour: '#f28c28', stops: ['Yankees', 'Yonkers', 'Tarrytown'], side: [-80, 0], via: { 'Yankees': HUDSON.slice(1, 4).map((s) => AT[s]), 'Yonkers': HUDSON.slice(5, 7).map((s) => AT[s]) } },
];

export async function buildExpress() {
  const { default: init, TrackApi } = await import(pathToFileURL(ROOT + 'app/src/wasm/track/sim.js').href);
  await init({ module_or_path: readFileSync(ROOT + 'app/src/wasm/track/sim_bg.wasm') });
  const LNG0 = -73.985, LAT0 = 40.758;
  const api = new TrackApi(LNG0, LAT0, true);
  api.set_cash(1e9);
  const kx = 111195 * Math.cos((LAT0 * Math.PI) / 180), ky = 111195;
  const xy = ([lat, lng], [sx, sy] = [0, 0]) => [(lng - LNG0) * kx + sx, (lat - LAT0) * ky + sy];
  const must = (r, what) => { if (r !== 0) throw new Error(`${what}: ${api.issue_kinds()}`); };
  const nodes = () => { const n = api.nodes_info(), out = []; for (let i = 0; i < n.length; i += 10) out.push({ id: n[i], x: n[i + 1], y: n[i + 2] }); return out; };
  const nodeNear = (x, y, within = 5) => nodes().find((m) => Math.hypot(m.x - x, m.y - y) < within);
  const end = (x, y) => { const n = nodeNear(x, y); return n ? [1, n.id, 0, 0, 0] : [0, 0, x, y, 0]; };
  // each line its own track and platforms; where lines share a stop their stations lie 250 m apart, linked
  const OFF = [[0, 0], [0, 250], [0, -250], [250, 0], [-250, 0]];
  const used = {}, at = LINES.map(({ stops, side }) => stops.map((s) => { const k = (used[s] = (used[s] ?? -1) + 1), [x, y] = xy(AT[s], side); return [x + OFF[k][0], y + OFF[k][1]]; }));
  LINES.forEach(({ fast, stops, via = {}, side }, li) => {
    for (let i = 1; i < stops.length; i++) {
      api.set_route_tangent(true);
      api.set_design_speed(fast ? 320 : 160);
      api.set_route_hsr(!!fast);
      must(api.add_route(new Float64Array([...end(...at[li][i - 1]), ...end(...at[li][i])]), new Float64Array((via[stops[i - 1]] ?? []).flatMap((p) => [...xy(p, side), 0, 0])), 2, false), `${stops[i - 1]} to ${stops[i]}`);
    }
  });
  const stops = LINES.map(({ stops }, li) => stops.map((s, i) => {
    const n = nodeNear(...at[li][i]);
    if (!n) throw new Error(`no node at ${s}`);
    // a shorter platform where a long one would cross another line's track
    if (api.add_station(1, n.id, 0, 0, 200, s) !== 0 && api.add_station(1, n.id, 0, 0, 100, s) !== 0) must(api.add_station(1, n.id, 0, 0, 50, s), `station ${s} of line ${li + 1}`);
    return n.id;
  }));
  must(api.construct_all(), 'construct');
  const first = {};
  LINES.forEach(({ stops: list }, li) => list.forEach((s, i) => {
    if (first[s] === undefined) first[s] = stops[li][i];
    else must(api.link_stations(first[s], stops[li][i]), 'link ' + s);
  }));
  LINES.forEach(({ colour, fast }, k) => {
    const id = api.new_line_id();
    must(api.set_line_with_trains(id, new Uint32Array(stops[k]), 6, 4, 3, 30, 180, 0, parseInt(colour.slice(1), 16), `Line ${k + 1}`), `line ${k + 1}`);
    if (fast) must(api.set_line_fast(id, true), `line ${k + 1} high-speed`);
  });
  return api;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const api = await buildExpress();
  // the game file: T-007's header (a New York game) with this network
  const raw = gunzipSync(readFileSync(ROOT + 'T-007-real-nyc.save'));
  const n = raw.readUInt32LE(4);
  const header = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
  const h = Buffer.from(JSON.stringify({ ...header, cash: 1e9, clock: 8 * 3600 }), 'utf8');
  const head = Buffer.alloc(8);
  raw.copy(head, 0, 0, 4);
  head.writeUInt32LE(h.length, 4);
  mkdirSync(new URL('.', pathToFileURL(EXPRESS_SAVE)), { recursive: true });
  writeFileSync(EXPRESS_SAVE, gzipSync(Buffer.concat([head, h, Buffer.from(api.save())])));
  console.log('saved', EXPRESS_SAVE);
}
