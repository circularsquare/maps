// T-1025: the realm shade (boundaryGeometry.outsideZones, triangles drawn as cut by shadeLayer.ts)
// against the zones, in Node, for many sets of shown realms. Random points in the shown zones' boxes:
// a point inside a shown zone (its own ring, even-odd, in Mercator) must not be shaded unless within
// 10 m of an edge (the union is cut along lon/lat lines), and no point may be under
// two triangles; a point outside every zone and every gap is shaded by construction (the triangles
// tile the cut ring). Points outside every zone left clear are gaps the zones enclose (shown by design,
// T-1006) or loops of rings that cross themselves; counted, not failed.
//   node trainworld/checks/t1025-shade.mjs [quick]      (OUTLINES=<file> for another outlines.json)
// Output: one line per set, PASS/FAIL, and artifacts/trainworld/t1025-shade.json.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const { outsideZones } = await import(pathToFileURL(ROOT + 'app/src/map/boundaryGeometry.ts').href);
const quick = process.argv[2] === 'quick';

const registry = JSON.parse(readFileSync(ROOT + 'app/src/game/registry.json', 'utf8'));
const realms = JSON.parse(readFileSync(ROOT + 'app/src/game/realms.json', 'utf8')).realms;
const { q, zones: flat } = JSON.parse(readFileSync(process.env.OUTLINES ?? ROOT + 'data/packs/outlines.json', 'utf8'));
const byId = new Map(realms.map((r) => [r.id, r]));
const byCountry = new Map(realms.flatMap((r) => (r.countries ?? []).map((cc) => [cc, r.id])));
const realmOf = (c) => (c.realm && byId.has(c.realm) ? c.realm : byCountry.get(c.country ?? '') ?? null);
const outlines = new Map();
for (const id in flat) outlines.set(id, { type: 'MultiPolygon', coordinates: flat[id].map((f) => {
  let x = f[0], y = f[1];
  const ring = [[x / q, y / q]];
  for (let i = 2; i < f.length; i += 2) ring.push([(x += f[i]) / q, (y += f[i + 1]) / q]);
  ring.push([...ring[0]]);
  return [ring];
}) });
const zonesFor = (shown) => registry.cities.filter((c) => outlines.has(c.id) && (!realmOf(c) || shown.has(realmOf(c)))).map((c) => outlines.get(c.id));

const mx = (lon) => (lon + 180) / 360;
const my = (lat) => { const s = Math.sin((Math.max(-85.051129, Math.min(85.051129, lat)) * Math.PI) / 180); return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); };

/** a zone ring's edges by band of latitude, for even-odd tests */
class Band {
  constructor(ring) {
    const n = ring.length - 1;
    this.xs = new Float64Array(n); this.ys = new Float64Array(n);
    const box = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const [x, y] = ring[i];
      this.xs[i] = x; this.ys[i] = y;
      box[0] = Math.min(box[0], x); box[1] = Math.min(box[1], y); box[2] = Math.max(box[2], x); box[3] = Math.max(box[3], y);
    }
    this.box = box;
    const nb = Math.max(1, Math.ceil(n / 8));
    this.dy = (box[3] - box[1]) / nb || 1;
    this.bands = Array.from({ length: nb }, () => []);
    for (let i = 0, j = n - 1; i < n; j = i++) for (let k = this.band(Math.min(this.ys[i], this.ys[j])); k <= this.band(Math.max(this.ys[i], this.ys[j])); k++) this.bands[k].push(i);
  }
  band(y) { return Math.min(this.bands.length - 1, Math.max(0, Math.floor((y - this.box[1]) / this.dy))); }
  holds(x, y) {
    if (x < this.box[0] || x > this.box[2] || y < this.box[1] || y > this.box[3]) return false;
    const { xs, ys } = this, n = xs.length;
    let c = false;
    for (const i of this.bands[this.band(y)]) {
      const j = i === 0 ? n - 1 : i - 1;
      if (ys[i] > y !== ys[j] > y && x < ((xs[j] - xs[i]) * (y - ys[i])) / (ys[j] - ys[i]) + xs[i]) c = !c;
    }
    return c;
  }
  distance(x, y) {
    const { xs, ys } = this, n = xs.length;
    let best = Infinity;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const dx = xs[i] - xs[j], dy = ys[i] - ys[j], t = Math.max(0, Math.min(1, ((x - xs[j]) * dx + (y - ys[j]) * dy) / (dx * dx + dy * dy || 1)));
      best = Math.min(best, Math.hypot(x - xs[j] - t * dx, y - ys[j] - t * dy));
    }
    return best;
  }
}

/** the triangles (Mercator x, y per corner), bucketed on a 512 x 512 grid */
class Mesh {
  constructor(t) {
    this.t = t;
    this.cells = new Map();
    for (let i = 0; i < t.length; i += 6) {
      const x0 = Math.min(t[i], t[i + 2], t[i + 4]), x1 = Math.max(t[i], t[i + 2], t[i + 4]);
      const y0 = Math.min(t[i + 1], t[i + 3], t[i + 5]), y1 = Math.max(t[i + 1], t[i + 3], t[i + 5]);
      for (let x = Math.floor(x0 * 512); x <= Math.min(511, x1 * 512); x++) for (let y = Math.floor(y0 * 512); y <= Math.min(511, y1 * 512); y++) {
        const k = x * 512 + y;
        (this.cells.get(k) ?? this.cells.set(k, []).get(k)).push(i);
      }
    }
  }
  /** how many triangles hold a point */
  count(px, py) {
    const t = this.t;
    let n = 0;
    for (const i of this.cells.get(Math.floor(px * 512) * 512 + Math.floor(py * 512)) ?? []) {
      const ax = t[i], ay = t[i + 1], bx = t[i + 2], by = t[i + 3], cx = t[i + 4], cy = t[i + 5];
      const d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (!d) continue;
      const w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / d, w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / d;
      if (w0 >= 0 && w1 >= 0 && 1 - w0 - w1 >= 0) n++;
    }
    return n;
  }
}

let seed = 12345;
const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);

function check(zones, mesh, perZone) {
  // in Mercator, where the map draws straight lines (a 50 km edge bows some metres from its lon/lat line)
  const rings = zones.flatMap((z) => z.coordinates.map((p) => new Band(p[0].map(([x, y]) => [mx(x) * 360, my(y) * 360]))));
  const cells = new Map();
  for (const r of rings) for (let x = Math.floor(r.box[0]); x <= r.box[2]; x++) for (let y = Math.floor(r.box[1]); y <= r.box[3]; y++) {
    const k = x * 1000 + y;
    (cells.get(k) ?? cells.set(k, []).get(k)).push(r);
  }
  let n = 0, zoneShaded = 0, gapClear = 0, twice = 0, atEdge = 0;
  const examples = [];
  const test = (x, y) => {
    const inZone = (cells.get(Math.floor(x) * 1000 + Math.floor(y)) ?? []).some((o) => o.holds(x, y));
    const c = mesh.count(x / 360, y / 360);
    n++;
    const at = () => [+(x - 180).toFixed(4), +((Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 360))) * 180) / Math.PI).toFixed(4)];
    if (c > 1) { twice++; if (examples.length < 5) examples.push(['twice', ...at()]); }
    if (inZone && c) {
      // metres from the nearest edge: the union is cut along lon/lat lines, drawn along Mercator ones
      const near = (cells.get(Math.floor(x) * 1000 + Math.floor(y)) ?? []).reduce((m, o) => Math.min(m, o.distance(x, y)), Infinity);
      const metres = near * 111320 * Math.cos((at()[1] * Math.PI) / 180);
      if (metres < 10) atEdge++;
      else { zoneShaded++; if (examples.length < 5) examples.push(['zone shaded', ...at()]); }
    }
    if (!inZone && !c) gapClear++;
  };
  // the zones' boxes, and the world at large
  for (const r of rings) for (let k = 0; k < perZone; k++) test(r.box[0] + (r.box[2] - r.box[0]) * rnd(), r.box[1] + (r.box[3] - r.box[1]) * rnd());
  let world = 0;
  for (let k = 0; k < 20000; k++) {
    const x = 360 * rnd(), y = 360 * my(-80 + 160 * rnd());
    const inZone = (cells.get(Math.floor(x) * 1000 + Math.floor(y)) ?? []).some((o) => o.holds(x, y));
    if (!inZone && mesh.count(x / 360, y / 360) !== 1) world++;
  }
  return { n, zoneShaded, atEdge, twice, gapClear, worldClear: world, examples };
}

const ids = realms.map((r) => r.id).filter((id) => registry.cities.some((c) => realmOf(c) === id));
const sets = [
  ['korea', 'japan', 'cn-east'],
  ids.filter((r) => r.startsWith('cn-')).concat('taiwan', 'korea', 'japan'),
  ['france', 'germany', 'benelux', 'ch-at', 'italy', 'uk-ie', 'iberia'],
  ids.filter((r) => r.startsWith('us-')).concat('canada'),
  ids,
  ...(quick ? [] : ids.map((r) => [r])),
];
const results = [];
let ok = true;
for (const s of sets) {
  const zones = zonesFor(new Set(s));
  const t0 = performance.now();
  const tri = outsideZones(zones);
  const ms = Math.round(performance.now() - t0);
  const u = check(zones, new Mesh(tri), quick ? 40 : 100);
  // far from any zone, open land and sea are shaded once: a few points can fall in enclosed gaps
  const pass = u.zoneShaded === 0 && u.twice === 0 && u.worldClear < 20;
  ok &&= pass;
  const r = { realms: s.length === ids.length ? 'all' : s.join(','), cutMs: ms, triangles: tri.length / 6, ...u };
  results.push(r);
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${r.realms}: cut ${ms} ms, ${r.triangles} triangles; ${u.n} points in zone boxes: ${u.zoneShaded} in a shown zone shaded (${u.atEdge} more within 10 m of an edge), ${u.twice} shaded twice, ${u.gapClear} outside every zone clear; ${u.worldClear} of 20000 world points outside the zones not shaded once${u.examples.length ? ' e.g. ' + JSON.stringify(u.examples) : ''}`);
}
const out = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
mkdirSync(out, { recursive: true });
writeFileSync(out + 't1025-shade.json', JSON.stringify(results, null, 1));
process.exit(ok ? 0 : 1);
