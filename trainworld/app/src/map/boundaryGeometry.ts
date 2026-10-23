// Exact metropolitan footprints, including islands and excluded enclaves, with GeoJSON winding.
import type { Feature, FeatureCollection, MultiLineString, MultiPolygon, Polygon, Position } from "geojson";

function ring(points: Position[], ccw: boolean): Position[] {
  const area = points.slice(1).reduce((sum, p, i) => sum + points[i][0] * p[1] - p[0] * points[i][1], 0);
  return (area > 0) === ccw ? points : [...points].reverse();
}

export function cityGeometry(boundary: FeatureCollection<Polygon | MultiPolygon>): MultiPolygon {
  return {
    type: "MultiPolygon",
    coordinates: boundary.features.flatMap(({ geometry }) =>
      geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates
    ).map(p => p.map((r, i) => ring(r, i === 0))),
  };
}

/** A vertex as one number, on a grid of about a metre: neighbouring zones' shared borders repeat
 * the same vertices (T-799). */
const vertex = (p: Position) => Math.round((p[0] + 180) * 1e5) * 33554432 + Math.round((p[1] + 90) * 1e5);

/** The zones' outlines with each shared border drawn once (T-799): a border two zones both hold
 * belongs to the first zone added, so it is one line at one weight, like an outer edge. A border the
 * focused zone holds is left out entirely, since the focus layers draw it. */
export class OutlineEdges {
  /** each segment's holders: the first zone to hold it, or every zone, in the order they were added */
  private holders = new Map<string, string | string[]>();
  private zones = new Map<string, { ring: Position[]; keys: string[] }[]>();

  get size() { return this.zones.size; }

  has(id: string) { return this.zones.has(id); }

  add(id: string, city: MultiPolygon) {
    if (this.zones.has(id)) return;
    const rings = city.coordinates.flat().map((ring) => {
      const keys: string[] = [];
      for (let i = 1; i < ring.length; i++) {
        const a = vertex(ring[i - 1]), b = vertex(ring[i]);
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        keys.push(key);
        const held = this.holders.get(key);
        if (held === undefined) this.holders.set(key, id);
        else if (Array.isArray(held)) held.push(id);
        else this.holders.set(key, [held, id]);
      }
      return { ring, keys };
    });
    this.zones.set(id, rings);
  }

  /** Takes a zone's outline back out (T-800: its coarse one, when its own arrives). */
  remove(id: string) {
    const rings = this.zones.get(id);
    if (!rings) return;
    for (const { keys } of rings) {
      for (const key of keys) {
        const held = this.holders.get(key);
        if (held === id) this.holders.delete(key);
        else if (Array.isArray(held)) {
          const rest = held.filter((h) => h !== id);
          if (rest.length > 1) this.holders.set(key, rest);
          else if (rest.length) this.holders.set(key, rest[0]);
          else this.holders.delete(key);
        }
      }
    }
    this.zones.delete(id);
  }

  /** One feature per zone with the lines it draws (property `id`), none for `focused`; only the zones
   * in `only` when given (T-871: those near the view). */
  lines(focused: string, only?: Set<string>): FeatureCollection<MultiLineString> {
    const features: Feature<MultiLineString>[] = [];
    for (const [id, rings] of this.zones) {
      if (id === focused || (only && !only.has(id))) continue;
      const coordinates: Position[][] = [];
      for (const { ring, keys } of rings) {
        const drawn = keys.map((key) => {
          const held = this.holders.get(key)!;
          return typeof held === "string" ? held === id : held[0] === id && !held.includes(focused);
        });
        const start = drawn.indexOf(false);
        if (start < 0) { coordinates.push(ring); continue; }
        // begin just after a gap, so a run that crosses the ring's seam stays one line
        const n = keys.length;
        let line: Position[] = [];
        for (let k = 1; k <= n; k++) {
          const i = (start + k) % n;
          if (drawn[i]) {
            if (!line.length) line.push(ring[i]);
            line.push(ring[i + 1]);
          } else if (line.length) {
            coordinates.push(line);
            line = [];
          }
        }
        if (line.length) coordinates.push(line);
      }
      if (coordinates.length) features.push({ type: "Feature", properties: { id }, geometry: { type: "MultiLineString", coordinates } });
    }
    return { type: "FeatureCollection", features };
  }
}

const WORLD: Position[] = [[-180, -85.051129], [180, -85.051129], [180, 85.051129], [-180, 85.051129], [-180, -85.051129]];

const signedArea = (r: Position[]) => {
  let a = 0;
  for (let i = 1; i < r.length; i++) a += r[i - 1][0] * r[i][1] - r[i][0] * r[i - 1][1];
  return a / 2;
};

/** A ring's edges by band of latitude, so a point test walks only the edges of its band. */
class RingIndex {
  private bands: number[][];
  private xs: Float64Array;
  private ys: Float64Array;
  private y0: number;
  private dy: number;
  constructor(xs: Float64Array, ys: Float64Array, y0: number, y1: number) {
    this.xs = xs;
    this.ys = ys;
    const n = Math.max(1, Math.ceil(xs.length / 8));
    this.y0 = y0;
    this.dy = (y1 - y0) / n || 1;
    this.bands = Array.from({ length: n }, () => []);
    for (let i = 0, j = xs.length - 1; i < xs.length; j = i++) {
      const a = Math.min(ys[i], ys[j]), b = Math.max(ys[i], ys[j]);
      for (let k = this.band(a); k <= this.band(b); k++) this.bands[k].push(i);
    }
  }
  private band(y: number) { return Math.min(this.bands.length - 1, Math.max(0, Math.floor((y - this.y0) / this.dy))); }
  /** even-odd */
  holds(x: number, y: number): boolean {
    const { xs, ys } = this, n = xs.length;
    let inside = false;
    for (const i of this.bands[this.band(y)]) {
      const j = i === 0 ? n - 1 : i - 1;
      const yi = ys[i], yj = ys[j];
      if (yi > y !== yj > y && x < ((xs[j] - xs[i]) * (y - yi)) / (yj - yi) + xs[i]) inside = !inside;
    }
    return inside;
  }
}

/** T-1006: a world-sized polygon with the zones cut out as one: their shared borders cancel (the
 * coarse file repeats a shared run's vertices in both zones, T-799), and gaps the zones enclose are
 * left in the cut, so the edge is only the zones' outer outline. The zones' rings are outer rings. */
export function outsideZones(zones: Iterable<MultiPolygon>): MultiPolygon {
  const all = [...zones];
  let total = 0;
  for (const z of all) for (const poly of z.coordinates) total += poly[0].length;
  // vertices on the identity grid (T-799) as indices, through an open-addressed table
  const size = 2 ** Math.ceil(Math.log2(total * 2 + 2)), mask = size - 1;
  const slotX = new Int32Array(size), slotY = new Int32Array(size), slotId = new Int32Array(size).fill(-1);
  const px: number[] = [], py: number[] = [];
  const id = (p: Position) => {
    const gx = Math.round((p[0] + 180) * 1e5), gy = Math.round((p[1] + 90) * 1e5);
    let s = (Math.imul(gx, 73856093) ^ Math.imul(gy, 19349663)) & mask;
    for (;;) {
      const i = slotId[s];
      if (i < 0) { slotX[s] = gx; slotY[s] = gy; slotId[s] = px.length; px.push(p[0]); py.push(p[1]); return px.length - 1; }
      if (slotX[s] === gx && slotY[s] === gy) return i;
      s = (s + 1) & mask;
    }
  };
  // each vertex's outgoing edges left after cancelling, each zone ring turned anticlockwise
  const next: number[][] = [];
  for (const z of all) for (const poly of z.coordinates) {
    const r = poly[0];
    const ccw = signedArea(r) > 0;
    const n = r.length - 1;
    let a = id(ccw ? r[0] : r[n]);
    for (let k = 1; k <= n; k++) {
      const b = id(ccw ? r[k] : r[n - k]);
      if (a === b) continue;
      const back = next[b], i = back ? back.indexOf(a) : -1;
      if (i >= 0) { back[i] = back[back.length - 1]; back.pop(); }
      else (next[a] ??= []).push(b);
      a = b;
    }
  }
  // chain the edges into rings; where several leave one vertex, the sharpest right turn, which parts a
  // gap touching the outline at a point from it (a gap joined on would be shaded)
  const turn = (o: number[], from: number, via: number) => {
    if (o.length === 1) return o.pop()!;
    const dx = px[via] - px[from], dy = py[via] - py[from];
    let best = 0, ba = Infinity;
    o.forEach((w, i) => {
      const ex = px[w] - px[via], ey = py[w] - py[via];
      const a = Math.atan2(dx * ey - dy * ex, dx * ex + dy * ey);
      if (a < ba) { ba = a; best = i; }
    });
    return o.splice(best, 1)[0];
  };
  type Ring = { ids: number[]; area: number; box: [number, number, number, number]; at?: RingIndex };
  const rings: Ring[] = [];
  for (let start = 0; start < next.length; start++) {
    const outs = next[start];
    while (outs?.length) {
      const ids = [start];
      let v = outs.pop()!;
      while (v !== start) {
        const from = ids[ids.length - 1];
        ids.push(v);
        const o = next[v];
        if (!o?.length) break;
        v = turn(o, from, v);
      }
      if (ids.length < 3) continue;
      let area = 0;
      const box: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
      for (let k = 0, j = ids.length - 1; k < ids.length; j = k++) {
        const x = px[ids[k]], y = py[ids[k]];
        area += px[ids[j]] * y - x * py[ids[j]];
        if (x < box[0]) box[0] = x; if (y < box[1]) box[1] = y; if (x > box[2]) box[2] = x; if (y > box[3]) box[3] = y;
      }
      // a clockwise ring is a gap the zones enclose: left in the cut
      if (area > 0) rings.push({ ids, area, box });
    }
  }
  // an outer ring inside another (an island in such a gap, a zone over another) is already in the cut
  rings.sort((a, b) => b.area - a.area);
  const kept: Ring[] = [];
  // the kept rings by the whole degrees their boxes cover
  const cells = new Map<number, Ring[]>();
  const cell = (x: number, y: number) => Math.floor(x) + 400 * Math.floor(y);
  for (const r of rings) {
    const mx = (px[r.ids[0]] + px[r.ids[1]]) / 2, my = (py[r.ids[0]] + py[r.ids[1]]) / 2;
    const inside = (cells.get(cell(mx, my)) ?? []).some((k) => {
      if (k.box[0] > r.box[0] || k.box[1] > r.box[1] || k.box[2] < r.box[2] || k.box[3] < r.box[3]) return false;
      k.at ??= new RingIndex(Float64Array.from(k.ids, (i) => px[i]), Float64Array.from(k.ids, (i) => py[i]), k.box[1], k.box[3]);
      return k.at.holds(mx, my);
    });
    if (inside) continue;
    kept.push(r);
    for (let x = Math.floor(r.box[0]); x <= r.box[2]; x++) for (let y = Math.floor(r.box[1]); y <= r.box[3]; y++) {
      const c = cell(x, y), list = cells.get(c);
      if (list) list.push(r); else cells.set(c, [r]);
    }
  }
  // holes run clockwise
  const holes = kept.map((k) => { const ring = k.ids.map((i) => [px[i], py[i]]).reverse(); ring.push(ring[0]); return ring; });
  return { type: "MultiPolygon", coordinates: [[WORLD, ...holes]] };
}

/** A world-sized polygon with the city's outer rings cut out, plus its excluded enclaves. */
export function outsideCity(city: MultiPolygon): MultiPolygon {
  return {
    type: "MultiPolygon",
    coordinates: [
      [
        WORLD,
        ...city.coordinates.map(p => ring(p[0], false)),
      ],
      ...city.coordinates.flatMap(p => p.slice(1).map(r => [ring(r, true)])),
    ],
  };
}
