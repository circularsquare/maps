// Exact metropolitan footprints, including islands and excluded enclaves, with GeoJSON winding
// (`cityGeometry`); the zones' outlines with each shared border drawn once (`OutlineEdges`); the
// shade cut, the triangles outside the zones shown (`outsideZones`); and the shade outside the focused
// zone (`outsideCity`, `outsideCityAndSea`).
import earcut from "earcut";
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
  /** T-1026: each zone's feature as last drawn (null: none), kept while its borders' holders and the
   * focus stay, so a redraw rebuilds only what changed and the source gets the same objects back */
  private drawn = new Map<string, Feature<MultiLineString> | null>();
  private drawnFocus = "";

  get size() { return this.zones.size; }

  has(id: string) { return this.zones.has(id); }

  /** the zones sharing a border with these keys redraw */
  private touch(keys: string[]) {
    for (const key of keys) {
      const held = this.holders.get(key);
      if (Array.isArray(held)) for (const h of held) this.drawn.delete(h);
      else if (held !== undefined) this.drawn.delete(held);
    }
  }

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
        else if (Array.isArray(held)) {
          for (const h of held) this.drawn.delete(h);
          held.push(id);
        } else {
          this.drawn.delete(held);
          this.holders.set(key, [held, id]);
        }
      }
      return { ring, keys };
    });
    this.zones.set(id, rings);
    this.drawn.delete(id);
  }

  /** Takes a zone's outline back out (T-800: its coarse one, when its own arrives). */
  remove(id: string) {
    const rings = this.zones.get(id);
    if (!rings) return;
    for (const { keys } of rings) this.touch(keys);
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
    if (focused !== this.drawnFocus) {
      // the borders the focus leaves out are those it holds
      for (const id of [this.drawnFocus, focused]) for (const { keys } of this.zones.get(id) ?? []) this.touch(keys);
      this.drawnFocus = focused;
    }
    const features: Feature<MultiLineString>[] = [];
    for (const [id, rings] of this.zones) {
      if (id === focused || (only && !only.has(id))) continue;
      let f = this.drawn.get(id);
      if (f === undefined) this.drawn.set(id, (f = this.zoneLines(id, rings, focused)));
      if (f) features.push(f);
    }
    return { type: "FeatureCollection", features };
  }

  private zoneLines(id: string, rings: { ring: Position[]; keys: string[] }[], focused: string): Feature<MultiLineString> | null {
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
    return coordinates.length ? { type: "Feature", properties: { id }, geometry: { type: "MultiLineString", coordinates } } : null;
  }
}

const WORLD: Position[] = [[-180, -85.051129], [180, -85.051129], [180, 85.051129], [-180, 85.051129], [-180, -85.051129]];

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
  /** winding number, positive inside an anticlockwise ring */
  winding(x: number, y: number): number {
    const { xs, ys } = this, n = xs.length;
    let w = 0;
    for (const i of this.bands[this.band(y)]) {
      const j = i === 0 ? n - 1 : i - 1;
      const xj = xs[j], yj = ys[j], side = (xs[i] - xj) * (y - yj) - (x - xj) * (ys[i] - yj);
      if (yj <= y) { if (ys[i] > y && side > 0) w++; }
      else if (ys[i] <= y && side < 0) w--;
    }
    return w;
  }
}

/** grid units a degree: the vertex identity grid (T-799) */
const G = 1e5;

/** Directed edges in typed arrays, a linked list per vertex, each with a tag. */
class Edges {
  head: Int32Array;
  to: Int32Array;
  tag: Int32Array;
  link: Int32Array;
  n = 0;
  constructor(vertices: number, edges: number) {
    this.head = new Int32Array(vertices).fill(-1);
    this.to = new Int32Array(edges);
    this.tag = new Int32Array(edges);
    this.link = new Int32Array(edges);
  }
  push(a: number, b: number, tag: number) {
    if (a >= this.head.length) {
      const head = new Int32Array(Math.max(a + 1, this.head.length * 2)).fill(-1);
      head.set(this.head);
      this.head = head;
    }
    if (this.n === this.to.length) {
      const grow = (old: Int32Array) => { const a = new Int32Array(old.length * 2); a.set(old); return a; };
      this.to = grow(this.to); this.tag = grow(this.tag); this.link = grow(this.link);
    }
    this.to[this.n] = b; this.tag[this.n] = tag; this.link[this.n] = this.head[a]; this.head[a] = this.n++;
  }
  /** Takes away one edge a→b (with this tag, if given); whether there was one. */
  drop(a: number, b: number, tag?: number): boolean {
    if (a >= this.head.length) return false;
    for (let p = -1, e = this.head[a]; e >= 0; p = e, e = this.link[e]) {
      if (this.to[e] !== b || (tag !== undefined && this.tag[e] !== tag)) continue;
      if (p < 0) this.head[a] = this.link[e]; else this.link[p] = this.link[e];
      return true;
    }
    return false;
  }
  /** a→b, or if b→a is there, neither: a border two zones share */
  add(a: number, b: number, tag: number) {
    if (!this.drop(b, a)) this.push(a, b, tag);
  }
}

type Box = [number, number, number, number];
/** `fn` with the key of every whole-degree cell a box covers (a degree is `G` grid units; 400 columns a row). */
const eachCell = (b: Box, fn: (cell: number) => void) => {
  for (let x = Math.floor(b[0] / G); x <= b[2] / G; x++) for (let y = Math.floor(b[1] / G); y <= b[3] / G; y++) fn(x + 400 * y);
};
/** The box round two boxes. */
const unionBox = (a: Box, b: Box): Box => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
/** A zone ring anticlockwise in grid units, open; `id` names it in a cut's cache. */
export type Ring = { id: number; xs: Float64Array; ys: Float64Array; box: Box; at?: RingIndex };
type Hole = { xs: Float64Array; ys: Float64Array };
let ringIds = 0;

/** A zone ring from lon, lat pairs (flat, open or closed), on the grid and turned anticlockwise; null
 * for one that encloses nothing. */
export function gridRing(lonLat: ArrayLike<number>): Ring | null {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < lonLat.length; i += 2) {
    const x = Math.round((lonLat[i] + 180) * G), y = Math.round((lonLat[i + 1] + 90) * G), n = xs.length;
    if (n && xs[n - 1] === x && ys[n - 1] === y) continue;
    xs.push(x); ys.push(y);
  }
  if (xs.length > 1 && xs[0] === xs[xs.length - 1] && ys[0] === ys[ys.length - 1]) { xs.pop(); ys.pop(); }
  const n = xs.length;
  let area = 0;
  for (let k = 0, j = n - 1; k < n; j = k++) area += (xs[j] - xs[0]) * (ys[k] - ys[0]) - (xs[k] - xs[0]) * (ys[j] - ys[0]);
  if (n < 3 || !area) return null;
  const r: Ring = { id: ringIds++, xs: new Float64Array(n), ys: new Float64Array(n), box: [Infinity, Infinity, -Infinity, -Infinity] };
  for (let k = 0; k < n; k++) {
    const i = area > 0 ? k : n - 1 - k, x = (r.xs[k] = xs[i]), y = (r.ys[k] = ys[i]);
    if (x < r.box[0]) r.box[0] = x; if (x > r.box[2]) r.box[2] = x;
    if (y < r.box[1]) r.box[1] = y; if (y > r.box[3]) r.box[3] = y;
  }
  return r;
}

const ringCache = new WeakMap<Position[], Ring | null>();
function ringOf(points: Position[]): Ring | null {
  let r = ringCache.get(points);
  if (r === undefined) ringCache.set(points, (r = gridRing(points.flat())));
  return r;
}

/** T-1006: the world with the zones cut out as one, its edge only the outer outline of the zones'
 * union, and gaps the zones enclose left in the cut, as triangles: Mercator x, y (0-1, y down) of
 * each corner in turn. The zones' rings are outer rings.
 * T-1025: the union is exact, since coarse outlines overlap (neighbours from different sources, rings
 * simplified into themselves): shared borders cancel (the coarse file repeats a shared run's vertices
 * in both zones, T-799), the rest are split where they cross or touch, and a piece is kept where the
 * zones wind round one side and not the other. Triangles, because MapLibre's own tiling cannot draw
 * this shape: each tile rounds the vertices, parts a few metres apart come out crossing, and earcut
 * then draws long stray triangles (shadeLayer.ts draws these instead). */
export function outsideZones(zones: Iterable<MultiPolygon>): Float64Array {
  const rings: Ring[] = [];
  for (const z of zones) for (const poly of z.coordinates) { const r = ringOf(poly[0]); if (r) rings.push(r); }
  return sharedCut.cut(rings);
}

/** pieces a cut keeps for the next */
const KEEP = 32;
/** grid units round each ring inside its piece's box */
const PAD = 0.1 * G;
const WORLD_Y0 = (90 - 85.051129) * G, WORLD_Y1 = (90 + 85.051129) * G, WORLD_X1 = 360 * G;

/** T-1026: the cut in pieces, kept between cuts. Rings whose boxes (each grown by PAD) overlap make a
 * piece, and pieces whose boxes overlap merge, so pieces lie in boxes apart. Each piece is cut and
 * triangulated within its box, the world around the boxes on its own; a change of the zones shown
 * redoes only the pieces it touches. */
export class ShadeCut {
  private pieces = new Map<string, { holes: Hole[]; box: Box; tri?: Float64Array; edge?: number[] }>();

  cut(rings: Ring[]): Float64Array {
    const used = groupRings(rings).map((g) => {
      const key = g.rings.map((r) => r.id).sort((a, b) => a - b).join(",");
      const p = this.pieces.get(key) ?? { holes: unionHoles(g.rings), box: g.box };
      this.pieces.delete(key);
      this.pieces.set(key, p);
      return p;
    });
    for (const key of this.pieces.keys()) {
      if (this.pieces.size <= Math.max(KEEP, used.length)) break;
      this.pieces.delete(key);
    }
    // a piece whose box would reach the world's edge: the whole world as one ring (zones across the
    // antimeridian or near the poles)
    if (!used.every(({ box }) => box[0] > 0 && box[2] < WORLD_X1 && box[1] > WORLD_Y0 && box[3] < WORLD_Y1)) {
      return triangles(keyhole(WORLD_BOX, used.flatMap((p) => p.holes)).ring);
    }
    let n = 0;
    for (const p of used) {
      if (!p.tri) {
        const k = keyhole(boxRing(p.box), p.holes);
        p.tri = triangles(k.ring);
        p.edge = k.edge;
      }
      n += p.tri.length;
    }
    // the world with each piece's box (as that piece's cut walks it) as a hole
    const data: number[] = [], at: number[] = [];
    const add = (x: number, y: number) => data.push(x / G / 360, mercY(y));
    for (let k = 0; k < 4; k++) add(WORLD_BOX.xs[k], WORLD_BOX.ys[k]);
    for (const p of used) {
      at.push(data.length / 2);
      for (let k = 0; k < p.edge!.length; k += 2) add(p.edge![k], p.edge![k + 1]);
    }
    const world = triangles(data, at);
    const out = new Float64Array(n + world.length);
    out.set(world);
    n = world.length;
    for (const p of used) { out.set(p.tri!, n); n += p.tri!.length; }
    return out;
  }
}
const sharedCut = new ShadeCut();

const WORLD_BOX: Hole = { xs: Float64Array.of(0, WORLD_X1, WORLD_X1, 0), ys: Float64Array.of(WORLD_Y0, WORLD_Y0, WORLD_Y1, WORLD_Y1) };
const boxRing = (b: Box): Hole => ({ xs: Float64Array.of(b[0], b[2], b[2], b[0]), ys: Float64Array.of(b[1], b[1], b[3], b[3]) });
/** Mercator y (0-1, down) of a grid y */
function mercY(y: number) {
  const s = Math.sin(((y / G - 90) * Math.PI) / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}
function triangles(ring: ArrayLike<number>, holes?: number[]): Float64Array {
  const t = earcut(ring, holes), out = new Float64Array(t.length * 2);
  for (let i = 0; i < t.length; i++) { out[2 * i] = ring[2 * t[i]]; out[2 * i + 1] = ring[2 * t[i] + 1]; }
  return out;
}

/** The rings by piece: those whose grown boxes overlap, then pieces whose boxes overlap merged. */
function groupRings(rings: Ring[]): { rings: Ring[]; box: Box }[] {
  const grown = rings.map(({ box }): Box => [box[0] - PAD, box[1] - PAD, box[2] + PAD, box[3] + PAD]);
  const overlap = (a: Box, b: Box) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
  const parent = rings.map((_, i) => i);
  const find = (i: number) => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  const cells = new Map<number, number[]>();
  grown.forEach((b, i) => eachCell(b, (c) => {
    const list = cells.get(c);
    if (!list) { cells.set(c, [i]); return; }
    for (const j of list) {
      const a = find(i), o = find(j);
      if (a !== o && overlap(b, grown[j])) parent[a] = o;
    }
    list.push(i);
  }));
  const byRoot = new Map<number, { rings: Ring[]; box: Box }>();
  rings.forEach((r, i) => {
    const b = grown[i], root = find(i), g = byRoot.get(root);
    if (!g) { byRoot.set(root, { rings: [r], box: [...b] }); return; }
    g.rings.push(r);
    g.box = unionBox(g.box, b);
  });
  const groups = [...byRoot.values()];
  for (let merged = true; merged;) {
    merged = false;
    for (let i = 0; i < groups.length; i++) for (let j = groups.length - 1; j > i; j--) {
      const a = groups[i], b = groups[j];
      if (!overlap(a.box, b.box)) continue;
      a.rings.push(...b.rings);
      a.box = unionBox(a.box, b.box);
      groups.splice(j, 1);
      merged = true;
    }
  }
  return groups;
}

/** The outer outlines of the rings' union (gaps it encloses left out), clockwise: the holes of the cut. */
function unionHoles(rings: Ring[]): Hole[] {
  let total = 0;
  for (const r of rings) total += r.xs.length;
  // vertices in grid units through an open-addressed table: the zones' on whole units (so the tests
  // below are exact), then the points where edges cross
  const size = 2 ** Math.ceil(Math.log2(total * 2 + 2)), mask = size - 1;
  const slotX = new Int32Array(size), slotY = new Int32Array(size), slotId = new Int32Array(size).fill(-1);
  const X: number[] = [], Y: number[] = [];
  const gridId = (gx: number, gy: number) => {
    // mixed well: the file's vertices all fall on every tenth grid unit
    let s = Math.imul(gx ^ Math.imul(gy, 0x9e3779b1), 0x85ebca6b);
    s = (s ^ (s >>> 15)) & mask;
    for (;;) {
      const i = slotId[s];
      if (i < 0) { slotX[s] = gx; slotY[s] = gy; slotId[s] = X.length; X.push(gx); Y.push(gy); return X.length - 1; }
      if (slotX[s] === gx && slotY[s] === gy) return i;
      s = (s + 1) & mask;
    }
  };

  // each zone ring's edges, tagged with the ring
  const edges = new Edges(total, total + 1024);
  rings.forEach(({ xs, ys }, ri) => {
    const n = xs.length, first = gridId(xs[0], ys[0]);
    for (let k = 1, a = first; k <= n; k++) {
      const b = k < n ? gridId(xs[k], ys[k]) : first;
      edges.add(a, b, ri);
      a = b;
    }
  });

  // the edges left, by cell of a grid as sorted keys (cell * F + edge)
  const { head, to, tag, link } = edges;
  let ns = 0;
  for (let a = 0; a < head.length; a++) for (let e = head[a]; e >= 0; e = link[e]) ns++;
  const sa = new Int32Array(ns), sb = new Int32Array(ns), sz = new Int32Array(ns);
  let spread = 0;
  for (let a = 0, i = 0; a < head.length; a++) for (let e = head[a]; e >= 0; e = link[e], i++) {
    sa[i] = a; sb[i] = to[e]; sz[i] = tag[e];
    spread += Math.abs(X[to[e]] - X[a]) + Math.abs(Y[to[e]] - Y[a]);
  }
  const S = Math.max(1000, (2 * spread) / Math.max(1, ns)), rows = Math.ceil((180 * G) / S) + 2;
  const F = 2 ** Math.ceil(Math.log2(ns + 1));
  let keys = new Float64Array(ns * 3 + 16), nk = 0;
  for (let i = 0; i < ns; i++) {
    const x0 = X[sa[i]], y0 = Y[sa[i]], x1 = X[sb[i]], y1 = Y[sb[i]];
    const c0 = Math.floor(Math.min(x0, x1) / S), c1 = Math.floor(Math.max(x0, x1) / S);
    for (let c = c0; c <= c1; c++) {
      // the edge's span of latitude within this column, a hair wide so that a crossing on a cell's
      // side lands in a cell both edges are in
      let ya = y0, yb = y1;
      if (c0 !== c1) {
        const xa = Math.max(Math.min(x0, x1), c * S), xb = Math.min(Math.max(x0, x1), (c + 1) * S);
        ya = y0 + ((y1 - y0) * (xa - x0)) / (x1 - x0);
        yb = y0 + ((y1 - y0) * (xb - x0)) / (x1 - x0);
      }
      const r1 = Math.floor((Math.max(ya, yb) + 1e-6) / S);
      for (let r = Math.floor((Math.min(ya, yb) - 1e-6) / S); r <= r1; r++) {
        if (nk === keys.length) { const grown = new Float64Array(nk * 2); grown.set(keys); keys = grown; }
        keys[nk++] = (c * rows + r) * F + i;
      }
    }
  }
  const sorted = keys.subarray(0, nk).sort();

  /** per edge, where it splits: [t, vertex, t, vertex, ...] */
  const splits: (number[] | undefined)[] = [];
  const crossed = new Set<number>(), crossings = new Map<string, number>();
  const split = (i: number, t: number, v: number) => (splits[i] ??= []).push(t, v);
  const meet = (i: number, j: number) => {
    const a = sa[i], b = sb[i], c = sa[j], d = sb[j];
    const ax = X[a], ay = Y[a], rx = X[b] - ax, ry = Y[b] - ay;
    const cx = X[c], cy = Y[c], sx = X[d] - cx, sy = Y[d] - cy;
    if (Math.max(ax, ax + rx) < Math.min(cx, cx + sx) || Math.max(cx, cx + sx) < Math.min(ax, ax + rx) ||
      Math.max(ay, ay + ry) < Math.min(cy, cy + sy) || Math.max(cy, cy + sy) < Math.min(ay, ay + ry)) return;
    const qx = cx - ax, qy = cy - ay;
    let den = rx * sy - ry * sx;
    if (den) {
      let tn = qx * sy - qy * sx, un = qx * ry - qy * rx;
      if (den < 0) { den = -den; tn = -tn; un = -un; }
      if (tn < 0 || tn > den || un < 0 || un > den) return;
      const tIn = tn > 0 && tn < den, uIn = un > 0 && un < den;
      if (tIn && uIn) {
        const key = i < j ? i * ns + j : j * ns + i;
        if (crossed.has(key)) return;
        crossed.add(key);
        // one vertex for edges crossing at one point
        const x = ax + (rx * tn) / den, y = ay + (ry * tn) / den;
        let v: number;
        if (Number.isInteger(x) && Number.isInteger(y)) v = gridId(x, y);
        else {
          const k = `${x},${y}`;
          v = crossings.get(k) ?? X.length;
          if (v === X.length) { crossings.set(k, v); X.push(x); Y.push(y); }
        }
        split(i, tn / den, v); split(j, un / den, v);
      } else if (tIn) split(i, tn / den, un ? d : c);
      else if (uIn) split(j, un / den, tn ? b : a);
      return;
    }
    if (qx * ry - qy * rx) return;
    // collinear: each splits where the other's ends fall inside it
    const rr = rx * rx + ry * ry, ss = sx * sx + sy * sy;
    const tc = qx * rx + qy * ry, td = (X[d] - ax) * rx + (Y[d] - ay) * ry;
    if (tc > 0 && tc < rr) split(i, tc / rr, c);
    if (td > 0 && td < rr) split(i, td / rr, d);
    const ua = -qx * sx - qy * sy, ub = (X[b] - cx) * sx + (Y[b] - cy) * sy;
    if (ua > 0 && ua < ss) split(j, ua / ss, a);
    if (ub > 0 && ub < ss) split(j, ub / ss, b);
  };
  for (let s = 0; s < nk;) {
    const base = Math.floor(sorted[s] / F) * F;
    let e = s + 1;
    while (e < nk && sorted[e] < base + F) e++;
    for (let p = s; p < e; p++) for (let q = p + 1; q < e; q++) meet(sorted[p] - base, sorted[q] - base);
    s = e;
  }
  // each split edge gives way to its pieces
  for (let i = 0; i < splits.length; i++) {
    const s = splits[i];
    if (!s) continue;
    edges.drop(sa[i], sb[i], sz[i]);
    const order = Array.from({ length: s.length / 2 }, (_, k) => k).sort((p, q) => s[2 * p] - s[2 * q]);
    let u = sa[i];
    for (const k of order) {
      const v = s[2 * k + 1];
      if (v !== u) { edges.add(u, v, sz[i]); u = v; }
    }
    if (u !== sb[i]) edges.add(u, sb[i], sz[i]);
  }

  // the rings by the whole degrees their boxes cover, for the winding tests
  const ringCells = new Map<number, number[]>();
  rings.forEach((z, zi) => eachCell(z.box, (c) => {
    const list = ringCells.get(c);
    if (list) list.push(zi); else ringCells.set(c, [zi]);
  }));
  /** per ring, the others whose boxes overlap its own */
  const near: number[][] = [];
  const nearOf = (zi: number) => {
    const z = rings[zi], out = new Set<number>();
    eachCell(z.box, (c) => {
      for (const oi of ringCells.get(c)!) {
        const o = rings[oi];
        if (oi !== zi && o.box[0] <= z.box[2] && o.box[2] >= z.box[0] && o.box[1] <= z.box[3] && o.box[3] >= z.box[1]) out.add(oi);
      }
    });
    return (near[zi] = [...out]);
  };
  const at = (z: Ring) => (z.at ??= new RingIndex(z.xs, z.ys, z.box[1], z.box[3]));
  const wind = (oi: number, x: number, y: number) => {
    const o = rings[oi];
    return x >= o.box[0] && x <= o.box[2] && y >= o.box[1] && y <= o.box[3] ? at(o).winding(x, y) : 0;
  };
  // keep the edges with the zones' total winding 0 on one side and not on the other, with the nonzero
  // side on their left. The total is one field (a cancelled border adds nothing to it), so the edges
  // kept always close into rings; where a ring that crosses itself winds a loop backwards over another
  // zone the two cancel, which leaves a sliver a few metres across shaded. The sides are tested a step
  // off the edge's midpoint, in grid units: far above rounding, far below any zone's detail.
  const EPS = 1e-4;
  const kept = new Edges(X.length, ns + 1024);
  {
    const { head, to, tag, link } = edges;
    for (let a = 0; a < head.length; a++) for (let e = head[a]; e >= 0; e = link[e]) {
      const b = to[e];
      if (b < 0) continue;
      let m = 1;
      for (let f = link[e]; f >= 0; f = link[f]) if (to[f] === b) { m++; to[f] = -1; }
      const dx = X[b] - X[a], dy = Y[b] - Y[a], l = Math.hypot(dx, dy);
      // (a crossing a hair from a vertex: kept, so its ring still closes)
      if (!l) { kept.push(a, b, 0); continue; }
      const x = (X[a] + X[b]) / 2 + (EPS * dy) / l, y = (Y[a] + Y[b]) / 2 - (EPS * dx) / l;
      const zi = tag[e];
      let right = wind(zi, x, y);
      for (const oi of near[zi] ?? nearOf(zi)) right += wind(oi, x, y);
      const left = right + m;
      if (!right && left) kept.push(a, b, 0);
      else if (right && !left) kept.push(b, a, 0);
    }
  }

  // chain the edges into rings; where several leave one vertex, the sharpest right turn, which parts a
  // gap touching the outline at a point from it (a gap joined on would be shaded)
  const take = (via: number, prev: number) => {
    const { head, to, link } = kept;
    let best = -1, bp = -1, ba = Infinity;
    const dx = X[via] - X[prev], dy = Y[via] - Y[prev];
    for (let p = -1, e = head[via]; e >= 0; p = e, e = link[e]) {
      if (best < 0 && link[e] < 0 && p < 0) { best = e; break; }
      const ex = X[to[e]] - X[via], ey = Y[to[e]] - Y[via];
      const a = Math.atan2(dx * ey - dy * ex, dx * ex + dy * ey);
      if (a < ba) { ba = a; best = e; bp = p; }
    }
    if (best < 0) return -1;
    if (bp < 0) head[via] = link[best]; else link[bp] = link[best];
    return to[best];
  };
  type Out = { ids: number[]; area: number; box: [number, number, number, number]; at?: RingIndex };
  const outs: Out[] = [];
  for (let start = 0; start < kept.head.length; start++) {
    while (kept.head[start] >= 0) {
      const ids = [start];
      const first = kept.head[start];
      kept.head[start] = kept.link[first];
      let v = kept.to[first], closed = true;
      while (v !== start) {
        const prev = ids[ids.length - 1];
        ids.push(v);
        v = take(v, prev);
        if (v < 0) { closed = false; break; }
      }
      // closing a chain that stops short would draw a straight line across whatever lies between
      if (!closed || ids.length < 3) continue;
      let area = 0;
      const x0 = X[start], y0 = Y[start];
      const box: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
      for (let k = 0, j = ids.length - 1; k < ids.length; j = k++) {
        const x = X[ids[k]], y = Y[ids[k]];
        area += (X[ids[j]] - x0) * (y - y0) - (x - x0) * (Y[ids[j]] - y0);
        if (x < box[0]) box[0] = x; if (y < box[1]) box[1] = y; if (x > box[2]) box[2] = x; if (y > box[3]) box[3] = y;
      }
      // a clockwise ring is a gap the zones enclose: left in the cut
      if (area > 0) outs.push({ ids, area, box });
    }
  }
  // an outer ring inside another (an island in such a gap) is already in the cut
  outs.sort((a, b) => b.area - a.area);
  const holes: Hole[] = [];
  // the kept rings by the whole degrees their boxes cover
  const cells = new Map<number, Out[]>();
  for (const r of outs) {
    const mx = (X[r.ids[0]] + X[r.ids[1]]) / 2, my = (Y[r.ids[0]] + Y[r.ids[1]]) / 2;
    const inside = (cells.get(Math.floor(mx / G) + 400 * Math.floor(my / G)) ?? []).some((k) => {
      if (k.box[0] > r.box[0] || k.box[1] > r.box[1] || k.box[2] < r.box[2] || k.box[3] < r.box[3]) return false;
      k.at ??= new RingIndex(Float64Array.from(k.ids, (i) => X[i]), Float64Array.from(k.ids, (i) => Y[i]), k.box[1], k.box[3]);
      return k.at.winding(mx, my) !== 0;
    });
    if (inside) continue;
    // holes run clockwise
    const n = r.ids.length, xs = new Float64Array(n), ys = new Float64Array(n);
    for (let k = 0; k < n; k++) { xs[k] = X[r.ids[n - 1 - k]]; ys[k] = Y[r.ids[n - 1 - k]]; }
    holes.push({ xs, ys });
    eachCell(r.box, (c) => {
      const list = cells.get(c);
      if (list) list.push(r); else cells.set(c, [r]);
    });
  }
  return holes;
}

/** An outer ring (anticlockwise, grid units) with the holes (clockwise) joined on as one ring, each by
 * a slit from its top vertex straight up to the first ring above it: earcut joins holes itself, but
 * takes seconds over thousands of them. A slit crosses nothing: it stops at the first edge above, and a
 * ring lower down that it passed would have been hit first. Out: `ring`, Mercator x, y (0-1, y down)
 * per vertex, open, cut into triangles in Mercator, where the map draws straight lines; `edge`, the
 * outer ring's own vertices as walked, the slits' ends on it included (grid units). */
function keyhole(outer: Hole, holes: Hole[]): { ring: number[]; edge: number[] } {
  const rings = [outer, ...holes];
  // the edges by column of longitude, for the upward rays: [ring, edge, ...]
  const C = 1e4;
  const cols = new Map<number, number[]>();
  rings.forEach(({ xs }, ri) => {
    const n = xs.length;
    for (let k = 0; k < n; k++) {
      const a = xs[k], b = xs[k + 1 === n ? 0 : k + 1];
      for (let c = Math.floor(Math.min(a, b) / C); c <= Math.max(a, b) / C; c++) {
        const list = cols.get(c);
        if (list) list.push(ri, k); else cols.set(c, [ri, k]);
      }
    }
  });
  const tops = rings.map(({ xs, ys }) => {
    let best = 0;
    for (let k = 1; k < xs.length; k++) if (ys[k] > ys[best] || (ys[k] === ys[best] && xs[k] > xs[best])) best = k;
    return best;
  });
  /** per ring, the holes hung from its edges: [edge, t, hole, ...] */
  const hung: number[][] = rings.map(() => []);
  for (let h = 1; h < rings.length; h++) {
    const x = rings[h].xs[tops[h]], y = rings[h].ys[tops[h]];
    let hit = Infinity, hr = 0, hk = 0, ht = 0;
    const list = cols.get(Math.floor(x / C)) ?? [];
    for (let i = 0; i < list.length; i += 2) {
      const ri = list[i], k = list[i + 1];
      if (ri === h) continue;
      const { xs, ys } = rings[ri], k2 = k + 1 === xs.length ? 0 : k + 1;
      const ax = xs[k], ay = ys[k], bx = xs[k2], by = ys[k2];
      if (x < Math.min(ax, bx) || x > Math.max(ax, bx) || (ax === bx && ay === by)) continue;
      let t: number, yy: number;
      if (ax === bx) {
        if (Math.max(ay, by) < y) continue;
        yy = Math.max(y, Math.min(ay, by));
        t = (yy - ay) / (by - ay);
      } else {
        t = x === ax ? 0 : x === bx ? 1 : (x - ax) / (bx - ax);
        yy = t === 0 ? ay : t === 1 ? by : ay + (by - ay) * t;
      }
      if (yy < y || yy >= hit) continue;
      hit = yy; hr = ri; hk = k; ht = t;
    }
    if (ht === 1) { hk = hk + 1 === rings[hr].xs.length ? 0 : hk + 1; ht = 0; }
    hung[hr].push(hk, ht, h);
  }
  const out: number[] = [], edge: number[] = [];
  const emit = (x: number, y: number) => out.push(x / G / 360, mercY(y));
  const walk = (ri: number, start: number) => {
    const { xs, ys } = rings[ri], n = xs.length, here = hung[ri];
    // the hung holes in walking order: by edge from the start, then along the edge
    const order = Array.from({ length: here.length / 3 }, (_, i) => i)
      .sort((p, q) => ((here[3 * p] - start + n) % n) - ((here[3 * q] - start + n) % n) || here[3 * p + 1] - here[3 * q + 1]);
    let o = 0;
    for (let s = 0; s < n; s++) {
      const k = (start + s) % n;
      emit(xs[k], ys[k]);
      if (!ri) edge.push(xs[k], ys[k]);
      for (; o < order.length && here[3 * order[o]] === k; o++) {
        const t = here[3 * order[o] + 1], h = here[3 * order[o] + 2], k2 = k + 1 === n ? 0 : k + 1;
        const fx = xs[k] + (xs[k2] - xs[k]) * t, fy = ys[k] + (ys[k2] - ys[k]) * t;
        if (t) {
          emit(fx, fy);
          if (!ri && (edge[edge.length - 2] !== fx || edge[edge.length - 1] !== fy)) edge.push(fx, fy);
        }
        walk(h, tops[h]);
        emit(rings[h].xs[tops[h]], rings[h].ys[tops[h]]);
        emit(fx, fy);
      }
    }
  };
  walk(0, 0);
  return { ring: out, edge };
}

/** T-1145: `outsideCity` with the zone's sea (the coarse file's `water`, which reaches a little into its
 * land) cut out too, as one outline with the city's outer rings. */
export function outsideCityAndSea(city: MultiPolygon, sea: MultiPolygon | undefined): MultiPolygon {
  if (!sea?.coordinates.length) return outsideCity(city);
  return outsideOutline(city, unionOutline(footprintRings(city, sea)));
}

/** The outer rings of a zone's own boundary and of its sea, as flat lon, lat pairs (what
 * `unionOutline` takes, here or in workers/focusShade.worker.ts). */
export function footprintRings(city: MultiPolygon, sea: MultiPolygon): Float64Array[] {
  return [...city.coordinates, ...sea.coordinates].map((poly) => Float64Array.from(poly[0].flat()));
}

/** The outer outlines of the rings' union (flat lon, lat pairs, open), enclosed gaps left out. */
export function unionOutline(rings: Float64Array[]): Float64Array[] {
  const grid = rings.flatMap((r) => gridRing(r) ?? []);
  return unionHoles(grid).map(({ xs, ys }) => {
    const out = new Float64Array(xs.length * 2);
    for (let i = 0; i < xs.length; i++) { out[2 * i] = xs[i] / G - 180; out[2 * i + 1] = ys[i] / G - 90; }
    return out;
  });
}

/** `outsideCity` with the outer rings given (`unionOutline`'s) in place of the city's own. */
export function outsideOutline(city: MultiPolygon, outline: Float64Array[]): MultiPolygon {
  const holes = outline.map((flat) => {
    const r: Position[] = [];
    for (let i = 0; i < flat.length; i += 2) r.push([flat[i], flat[i + 1]]);
    r.push([flat[0], flat[1]]);
    return ring(r, false);
  });
  return {
    type: "MultiPolygon",
    coordinates: [[WORLD, ...holes], ...city.coordinates.flatMap(p => p.slice(1).map(r => [ring(r, true)]))],
  };
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
