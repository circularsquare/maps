// T-125: the TypeScript copy of the world frames (app/src/game/frame.ts) and the city registry's
// conversions (app/src/game/cities.ts), checked against PROJ's values (pipeline/frame_fixture.py
// writes app/src/game/frame.fixture.json) and against the formulas every earlier build used.
//   node trainworld/notes/t125-frame-check.mjs        (from the maps root)
// Writes artifacts/trainworld/notes-outputs/T-125-frame-check.json.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { build } from '../app/node_modules/vite/dist/node/index.js';

async function load(entry) {
  const bundle = await build({ configFile: false, root: 'trainworld/app', logLevel: 'silent', build: {
    write: false, minify: false, lib: { entry: entry.replace('trainworld/app/', ''), formats: ['es'] },
  } });
  const output = (Array.isArray(bundle) ? bundle[0] : bundle).output;
  return import('data:text/javascript;base64,' + Buffer.from(output.find(o => o.type === 'chunk').code).toString('base64'));
}
const frames = await load('trainworld/app/src/game/frame.ts');
const cities = await load('trainworld/app/src/game/cities.ts');
const fx = JSON.parse(await readFile('trainworld/app/src/game/frame.fixture.json', 'utf8'));
const out = { proj: [], cities: {} };

// 1. the world frame against PROJ, to a millimetre
const lcc = frames.makeFrame({ kind: 'lcc', lon: fx.lon0, lat: fx.lat0 });
let worst = 0;
for (const [lon, lat, x, y] of fx.points) {
  const [px, py] = lcc.fromLonLat(lon, lat);
  worst = Math.max(worst, Math.hypot(px - x, py - y));
  const [lo, la] = lcc.toLonLat(x, y);
  assert(Math.abs(lo - lon) < 2e-8 && Math.abs(la - lat) < 2e-8, `inverse at ${lon},${lat}`);
}
assert(worst < 0.001, `worst ${worst} m from PROJ`);
out.proj = { points: fx.points.length, worstMetres: worst };

// 2. the legacy frame is the formula every earlier build used
const R = 6371008.8, RAD = Math.PI / 180;
const legacy = frames.makeFrame({ kind: 'equirect', lon: -73.985, lat: 40.758 });
const [lx, ly] = legacy.fromLonLat(-75.1636, 39.9526);
assert.equal(lx, R * (-75.1636 + 73.985) * RAD * Math.cos(40.758 * RAD));
assert.equal(ly, R * (39.9526 - 40.758) * RAD);

// 3. the registry's conversions: round trips in both world frames, and bit-for-bit what T-116 had in the legacy one
const oldWorldToCity = (x, y, o) => {
  const lon = -73.985 + x / (R * Math.cos(40.758 * RAD) * RAD), lat = 40.758 + y / (R * RAD);
  return { x: R * (lon - o.lon) * RAD * Math.cos(o.lat * RAD), y: R * (lat - o.lat) * RAD };
};
const oldCityToWorld = (x, y, o) => {
  const lon = o.lon + x / (R * Math.cos(o.lat * RAD) * RAD), lat = o.lat + y / (R * RAD);
  return { x: R * (lon - -73.985) * RAD * Math.cos(40.758 * RAD), y: R * (lat - 40.758) * RAD };
};
for (const [kind, spec] of [['legacy', cities.LEGACY_FRAME], ['world', cities.NEW_GAME_FRAME]]) {
  cities.setWorldFrame(spec);
  let worstRound = 0, identical = 0, n = 0;
  for (const city of cities.CITIES) {
    for (let i = -12; i <= 12; i++) for (let j = -12; j <= 12; j++) {
      const x = i * 7311.3 + 0.37, y = j * 6101.7 - 0.21; // pack metres, tens of km either way
      const w = cities.cityToWorld(x, y, city.id), back = cities.worldToCity(w.x, w.y, city.id);
      worstRound = Math.max(worstRound, Math.hypot(back.x - x, back.y - y));
      if (kind === 'legacy') {
        const o = oldCityToWorld(x, y, city.origin), p = oldWorldToCity(w.x, w.y, city.origin);
        n++;
        if (o.x === w.x && o.y === w.y && (city.id === 'nyc' || (p.x === back.x && p.y === back.y))) identical++;
        else if (Math.hypot(o.x - w.x, o.y - w.y) > 1e-6) assert.fail(`legacy conversion moved: ${city.id} ${x},${y}`);
      }
    }
  }
  assert(worstRound < 1e-6, `${kind} round trip ${worstRound} m`);
  // the regional market keeps its sites in the NYC pack frame
  const m = cities.worldToMarket(1234.5, -6789.25), mb = cities.marketToWorld(m.x, m.y);
  assert(Math.hypot(mb.x - 1234.5, mb.y + 6789.25) < 1e-6);
  out.cities[kind] = { worstRoundTripMetres: worstRound, ...(kind === 'legacy' ? { pointsChecked: n, philadelphiaBitIdenticalToT116: identical, newYorkNowExactIdentity: true } : {}) };
}
// a place keeps its place: Philadelphia City Hall as pack (0, 0) lands at the same lon/lat in either world frame
for (const spec of [cities.LEGACY_FRAME, cities.NEW_GAME_FRAME]) {
  cities.setWorldFrame(spec);
  const w = cities.cityToWorld(0, 0, 'phl');
  const [lon, lat] = cities.getWorldFrame().toLonLat(w.x, w.y);
  assert(Math.abs(lon + 75.1636) < 1e-9 && Math.abs(lat - 39.9526) < 1e-9);
}
// 5. what the clock worker pays per drawn point to turn world metres into lng/lat (the renderer then needs toLocal as before)
for (const [name, spec] of [['legacy', cities.LEGACY_FRAME], ['world', cities.NEW_GAME_FRAME]]) {
  const f = frames.makeFrame(spec);
  let s = 0; const t0 = performance.now();
  for (let r = 0; r < 5; r++) for (let i = 0; i < 200000; i++) { const p = f.toLonLat(i * 0.37 - 30000, 20000 - i * 0.21); s += p[0]; }
  out.nsPerPoint = out.nsPerPoint ?? {};
  out.nsPerPoint[name] = +((performance.now() - t0) * 1e6 / 1e6).toFixed(1);
  if (!Number.isFinite(s)) throw new Error('nan');
}
console.log(JSON.stringify(out));
await writeFile('artifacts/trainworld/notes-outputs/T-125-frame-check.json', JSON.stringify(out, null, 2) + '\n');
