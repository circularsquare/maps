// A road graph built from zoom 14 vector tiles as they arrive, and a shortest-path search over it
// (T-1465). Nodes are the road lines' vertices, named by their place in the world's zoom 14 grid
// (`WORLD` units across), so a vertex that two tiles both hold is one node. Each tile is cut to its own
// square: a road crossing the edge of two tiles ends there in each, and the two ends are joined when
// they are within `JOIN` units. Pure: no DOM, no map.

import type { RoadLines } from "./roadTiles";

export const EXTENT = 4096;
export const ZOOM = 14;
/** grid units across the world at zoom 14 */
export const WORLD = EXTENT * 2 ** ZOOM;
const EQUATOR_M = 40075016.686;
/** Two road ends on a tile edge this close (units, about 2 m) are one node. */
const JOIN = 3;
/** The other carriageway of a divided road is crossed at most this far (ground metres) and at least
 * `HOP_MIN_M`, to a road within `PARALLEL_COS` of parallel. */
const HOP_M = 40;
const HOP_MIN_M = 4;
const PARALLEL_COS = 0.9;
/** A road line of this weight or more (`RoadLines.weights`) is a motorway. */
export const MOTORWAY = 2;
const MOTOR_EXTRA_M = 30;

export const lngOf = (gx: number) => (gx / WORLD) * 360 - 180;
export const latOf = (gy: number) => (360 / Math.PI) * Math.atan(Math.exp(Math.PI * (1 - (2 * gy) / WORLD))) - 90;
export const gxOf = (lng: number) => ((lng + 180) / 360) * WORLD;
export const gyOf = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * WORLD;
};
/** ground metres in one grid unit at grid row `gy` */
export const metresPerUnit = (gy: number) => (EQUATOR_M / WORLD) * Math.cos((latOf(gy) * Math.PI) / 180);

/** A point on the nearest road: the edge, how far along it (0 at its `a` end), the point, and the
 * distance to it in grid units. */
export interface Snap {
  edge: number;
  t: number;
  x: number;
  y: number;
  d: number;
}

/** The shortest road between two snapped points: the grid points from one to the other, and the length. */
export interface RoadPath {
  pts: number[];
  m: number;
  /** where it ends: the destination or one of its alternatives */
  to: Snap;
}

export class RoadGraph {
  /** bumped whenever a tile is added: a search or snap made before is stale */
  version = 0;
  nodes = 0;
  gx = new Int32Array(1024);
  gy = new Int32Array(1024);
  private head = new Int32Array(1024).fill(-1);
  edges = 0;
  ea = new Int32Array(1024);
  eb = new Int32Array(1024);
  /** ground metres */
  elen = new Float32Array(1024);
  /** 1 for a motorway (its line's weight 2 or more), 2 for a link (`LINK`), which a bus may drive and a click does not land on, else 0 */
  emotor = new Uint8Array(1024);
  private arcNext = new Int32Array(2048);
  private inner = new Map<number, number>();
  private borderV = new Map<number, [number, number][]>();
  private borderH = new Map<number, [number, number][]>();
  /** per tile (`tileKey`), its edges [first, end) */
  private tiles = new Map<number, [number, number]>();

  static tileKey(tx: number, ty: number) {
    return tx * 16384 + ty;
  }
  has(tx: number, ty: number) {
    return this.tiles.has(RoadGraph.tileKey(tx, ty));
  }
  get tileCount() {
    return this.tiles.size;
  }

  clear() {
    this.nodes = 0;
    this.edges = 0;
    this.head.fill(-1);
    this.inner.clear();
    this.borderV.clear();
    this.borderH.clear();
    this.tiles.clear();
    this.version++;
  }

  private growNodes() {
    const n = this.gx.length * 2;
    const grow = <T extends Int32Array>(a: T): T => {
      const b = new (a.constructor as new (n: number) => T)(n);
      b.set(a);
      return b;
    };
    this.gx = grow(this.gx);
    this.gy = grow(this.gy);
    const h = new Int32Array(n).fill(-1);
    h.set(this.head);
    this.head = h;
  }

  private growEdges() {
    const n = this.ea.length * 2;
    const ea = new Int32Array(n), eb = new Int32Array(n), len = new Float32Array(n), next = new Int32Array(n * 2), mo = new Uint8Array(n);
    ea.set(this.ea);
    eb.set(this.eb);
    len.set(this.elen);
    next.set(this.arcNext);
    mo.set(this.emotor);
    this.ea = ea;
    this.eb = eb;
    this.elen = len;
    this.arcNext = next;
    this.emotor = mo;
  }

  private newNode(x: number, y: number): number {
    if (this.nodes === this.gx.length) this.growNodes();
    const i = this.nodes++;
    this.gx[i] = x;
    this.gy[i] = y;
    this.head[i] = -1;
    return i;
  }

  /** The node at a grid point; on a tile edge, the one within `JOIN` of it along that edge. */
  private node(x: number, y: number, onV: boolean, onH: boolean): number {
    if (!onV && !onH) {
      const k = x * WORLD + y;
      let i = this.inner.get(k);
      if (i === undefined) this.inner.set(k, (i = this.newNode(x, y)));
      return i;
    }
    const lists: [Map<number, [number, number][]>, number, number][] = [];
    if (onV) lists.push([this.borderV, x, y]);
    if (onH) lists.push([this.borderH, y, x]);
    let found = -1;
    for (const [m, line, pos] of lists) {
      let best = JOIN + 1;
      for (const [p, n] of m.get(line) ?? []) {
        const d = Math.abs(p - pos);
        if (d < best) (best = d), (found = n);
      }
      if (found >= 0) break;
    }
    if (found < 0) found = this.newNode(x, y);
    for (const [m, line, pos] of lists) {
      const l = m.get(line);
      if (l) l.push([pos, found]);
      else m.set(line, [[pos, found]]);
    }
    return found;
  }

  private addEdge(a: number, b: number, m: number, kind: number) {
    if (a === b) return;
    if (this.edges === this.ea.length) this.growEdges();
    const e = this.edges++;
    this.ea[e] = a;
    this.eb[e] = b;
    this.elen[e] = m;
    this.emotor[e] = kind;
    this.arcNext[2 * e] = this.head[a];
    this.head[a] = 2 * e;
    this.arcNext[2 * e + 1] = this.head[b];
    this.head[b] = 2 * e + 1;
  }

  /** Add tile (tx, ty)'s road lines (a tile already added is not added again). */
  addTile(tx: number, ty: number, rl: RoadLines) {
    const key = RoadGraph.tileKey(tx, ty);
    if (this.tiles.has(key)) return;
    const first = this.edges;
    const ext = rl.extent, x0 = tx * EXTENT, y0 = ty * EXTENT;
    const scale = EXTENT / ext;
    const mpu = metresPerUnit(y0 + EXTENT / 2);
    const seg: number[] = [0, 0, 0, 0];
    rl.lines.forEach((line, li) => {
      const w = rl.weights?.[li] ?? 1, kind = w >= MOTORWAY ? 1 : w === LINK ? 2 : 0;
      let pa = -1, pax = NaN, pay = NaN;
      for (let i = 2; i + 1 < line.length; i += 2) {
        if (!clipSegment(line[i - 2] * scale, line[i - 1] * scale, line[i] * scale, line[i + 1] * scale, seg)) continue;
        const [ax, ay, bx, by] = seg;
        const a = pa >= 0 && pax === ax && pay === ay ? pa : this.node(x0 + ax, y0 + ay, ax === 0 || ax === EXTENT, ay === 0 || ay === EXTENT);
        const b = this.node(x0 + bx, y0 + by, bx === 0 || bx === EXTENT, by === 0 || by === EXTENT);
        this.addEdge(a, b, Math.hypot(bx - ax, by - ay) * mpu, kind);
        pa = b;
        pax = bx;
        pay = by;
      }
    });
    this.tiles.set(key, [first, this.edges]);
    this.version++;
  }

  /** Every road within `maxUnits` of a grid point: the nearest point of each edge, nearest first. */
  snapsWithin(x: number, y: number, maxUnits: number): Snap[] {
    const out: Snap[] = [];
    const tx0 = Math.floor((x - maxUnits) / EXTENT), tx1 = Math.floor((x + maxUnits) / EXTENT);
    const ty0 = Math.floor((y - maxUnits) / EXTENT), ty1 = Math.floor((y + maxUnits) / EXTENT);
    for (let tx = tx0; tx <= tx1; tx++)
      for (let ty = ty0; ty <= ty1; ty++) {
        const r = this.tiles.get(RoadGraph.tileKey(tx, ty));
        if (!r) continue;
        for (let e = r[0]; e < r[1]; e++) {
          if (this.emotor[e] === 2) continue;
          const a = this.ea[e], b = this.eb[e];
          const ax = this.gx[a], ay = this.gy[a], dx = this.gx[b] - ax, dy = this.gy[b] - ay;
          const l2 = dx * dx + dy * dy;
          const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
          const px = ax + t * dx, py = ay + t * dy, d = Math.hypot(px - x, py - y);
          if (d <= maxUnits) out.push({ edge: e, t, x: px, y: py, d });
        }
      }
    return out.sort((p, q) => this.snapCost(p) - this.snapCost(q));
  }

  /** How far a click is taken as being from a road, grid units: a motorway counts 3 times as far and
   * `MOTOR_EXTRA_M` more (T-2469), so a click by a street lands on the street, not the motorway over it. */
  snapCost(s: Snap): number {
    return this.emotor[s.edge] === 1 ? 3 * s.d + MOTOR_EXTRA_M / metresPerUnit(s.y) : s.d;
  }

  /** The road a click lands on within `maxUnits` of a grid point (the least `snapCost`), or null. */
  snap(x: number, y: number, maxUnits: number): Snap | null {
    return this.snapsWithin(x, y, maxUnits)[0] ?? null;
  }

  // ---------------------------------------------------------------- search

  private g = new Float64Array(0);
  private from = new Int32Array(0);
  private seen = new Int32Array(0);
  private done = new Int32Array(0);
  private stamp = 0;
  private hf = new Float64Array(256);
  private hn = new Int32Array(256);
  private hs = 0;

  private push(f: number, n: number) {
    if (this.hs === this.hf.length) {
      const f2 = new Float64Array(this.hs * 2), n2 = new Int32Array(this.hs * 2);
      f2.set(this.hf);
      n2.set(this.hn);
      this.hf = f2;
      this.hn = n2;
    }
    let i = this.hs++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.hf[p] <= f) break;
      this.hf[i] = this.hf[p];
      this.hn[i] = this.hn[p];
      i = p;
    }
    this.hf[i] = f;
    this.hn[i] = n;
  }

  /** Remove the smallest; its key is left in `popF`. */
  private popF = 0;
  private pop(): number {
    const top = this.hn[0];
    this.popF = this.hf[0];
    const last = --this.hs;
    if (last > 0) {
      const f = this.hf[last], n = this.hn[last];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= last) break;
        if (c + 1 < last && this.hf[c + 1] < this.hf[c]) c++;
        if (this.hf[c] >= f) break;
        this.hf[i] = this.hf[c];
        this.hn[i] = this.hn[c];
        i = c;
      }
      this.hf[i] = f;
      this.hn[i] = n;
    }
    return top;
  }

  /** The roads running beside a snapped point that its click could cross to: on each side, the nearest
   * point on a roughly parallel road 4 to `HOP_M` metres away (the other carriageway of a divided
   * road). `d` of each is the hop's length in grid units. */
  private across(s: Snap): Snap[] {
    const mpu = metresPerUnit(s.y), max = HOP_M / mpu, min = HOP_MIN_M / mpu;
    const a = this.ea[s.edge], b = this.eb[s.edge];
    let dx = this.gx[b] - this.gx[a], dy = this.gy[b] - this.gy[a];
    const l = Math.hypot(dx, dy);
    if (l === 0) return [];
    dx /= l;
    dy /= l;
    const best: (Snap | null)[] = [null, null];
    const tx0 = Math.floor((s.x - max) / EXTENT), tx1 = Math.floor((s.x + max) / EXTENT);
    const ty0 = Math.floor((s.y - max) / EXTENT), ty1 = Math.floor((s.y + max) / EXTENT);
    for (let tx = tx0; tx <= tx1; tx++)
      for (let ty = ty0; ty <= ty1; ty++) {
        const r = this.tiles.get(RoadGraph.tileKey(tx, ty));
        if (!r) continue;
        for (let e = r[0]; e < r[1]; e++) {
          if (e === s.edge || this.emotor[e] === 2) continue;
          const p = this.ea[e], q = this.eb[e];
          const ex = this.gx[q] - this.gx[p], ey = this.gy[q] - this.gy[p], l2 = ex * ex + ey * ey;
          if (l2 === 0) continue;
          const el = Math.sqrt(l2);
          // roughly parallel
          if (Math.abs((ex * dx + ey * dy) / el) < PARALLEL_COS) continue;
          const t = Math.max(0, Math.min(1, ((s.x - this.gx[p]) * ex + (s.y - this.gy[p]) * ey) / l2));
          const px = this.gx[p] + t * ex, py = this.gy[p] + t * ey, hx = px - s.x, hy = py - s.y, h = Math.hypot(hx, hy);
          if (h < min || h > max) continue;
          // the hop crosses the road rather than runs along it
          if (Math.abs((hx * dx + hy * dy) / h) > 0.5) continue;
          const side = hx * -dy + hy * dx > 0 ? 0 : 1;
          if (!best[side] || h < best[side]!.d) best[side] = { edge: e, t, x: px, y: py, d: h };
        }
      }
    return best.filter((x): x is Snap => x !== null);
  }

  /** The shortest road between two snapped points, or null if none is under `capM` metres. Either
   * point may cross to the other carriageway of a divided road first (`across`), the hop counted at its
   * length, so a route that turns on a divided road does not run to the next junction and back. `alts`
   * are other places the path may end instead of `d` (T-2467: other roads near the click), each costing
   * `penM` metres more than its road; `to` is the end taken. */
  path(s: Snap, d: Snap, capM: number, alts: Snap[] = [], penM: (c: Snap) => number = () => 0): RoadPath | null {
    const mpu = metresPerUnit(d.y);
    if (s.edge === d.edge) {
      return { pts: [s.x, s.y, d.x, d.y], m: Math.abs(d.t - s.t) * this.elen[s.edge], to: d };
    }
    if (this.g.length < this.nodes) {
      const n = Math.max(this.nodes, this.g.length * 2);
      this.g = new Float64Array(n);
      this.from = new Int32Array(n);
      this.seen = new Int32Array(n);
      this.done = new Int32Array(n);
      this.stamp = 0;
    }
    const stamp = ++this.stamp;
    this.hs = 0;
    // the guide may overshoot by the hops at either end, and by the farthest other end
    let slack = (2 * HOP_M) / mpu;
    for (const c of alts) slack = Math.max(slack, (2 * HOP_M) / mpu + Math.hypot(c.x - d.x, c.y - d.y));
    const h = (n: number) => Math.max(0, Math.hypot(this.gx[n] - d.x, this.gy[n] - d.y) - slack) * mpu;
    const hopM = (c: Snap) => c.d * mpu;
    const viaStart = new Map<number, [number, number] | null>();
    const start = (n: number, off: number, via: [number, number] | null) => {
      if (this.seen[n] === stamp && this.g[n] <= off) return;
      this.seen[n] = stamp;
      this.g[n] = off;
      this.from[n] = -1;
      viaStart.set(n, via);
      this.push(off + h(n), n);
    };
    const startFrom = (c: Snap, hop: number, via: [number, number] | null) => {
      const l = this.elen[c.edge];
      start(this.ea[c.edge], hop + c.t * l, via);
      start(this.eb[c.edge], hop + (1 - c.t) * l, via);
    };
    startFrom(s, 0, null);
    const sAcross = this.across(s);
    for (const c of sAcross) startFrom(c, hopM(c), [c.x, c.y]);
    // where the path may end: a node of the destination's road or of the one across from it
    const ends = new Map<number, { off: number; via: [number, number] | null; to: Snap; pen: number }>();
    const endAt = (c: Snap, hop: number, via: [number, number] | null, to = d, pen = 0) => {
      const l = this.elen[c.edge];
      for (const [n, off] of [[this.ea[c.edge], pen + hop + c.t * l], [this.eb[c.edge], pen + hop + (1 - c.t) * l]] as const) {
        const had = ends.get(n);
        if (!had || off < had.off) ends.set(n, { off, via, to, pen });
      }
    };
    endAt(d, 0, null);
    const dAcross = this.across(d);
    for (const c of dAcross) endAt(c, hopM(c), [c.x, c.y]);
    let best = capM, end = -1;
    let direct: number[] | null = null, directTo = d, directPen = 0;
    // across the road with no junction between: the start crossing straight to the destination's road, or the reverse
    for (const c of sAcross) {
      if (c.edge !== d.edge) continue;
      const m = hopM(c) + Math.abs(c.t - d.t) * this.elen[d.edge];
      if (m < best) (best = m), (direct = [s.x, s.y, c.x, c.y, d.x, d.y]);
    }
    for (const c of dAcross) {
      if (c.edge !== s.edge) continue;
      const m = hopM(c) + Math.abs(c.t - s.t) * this.elen[s.edge];
      if (m < best) (best = m), (direct = [s.x, s.y, c.x, c.y, d.x, d.y]);
    }
    for (const c of alts) {
      const pen = penM(c);
      if (c.edge === s.edge) {
        const m = pen + Math.abs(c.t - s.t) * this.elen[s.edge];
        if (m < best) (best = m), (direct = [s.x, s.y, c.x, c.y]), (directTo = c), (directPen = pen);
      } else endAt(c, 0, null, c, pen);
    }
    while (this.hs > 0) {
      const n = this.pop();
      if (this.popF >= best) break;
      if (this.done[n] === stamp) continue;
      this.done[n] = stamp;
      const gn = this.g[n];
      const te = ends.get(n);
      if (te && gn + te.off < best) (best = gn + te.off), (end = n), (direct = null), (directTo = d), (directPen = 0);
      for (let a = this.head[n]; a >= 0; a = this.arcNext[a]) {
        const e = a >> 1, m = (a & 1) === 0 ? this.eb[e] : this.ea[e];
        const ng = gn + this.elen[e];
        if (this.done[m] === stamp || (this.seen[m] === stamp && this.g[m] <= ng)) continue;
        const f = ng + h(m);
        if (f >= best) continue;
        this.seen[m] = stamp;
        this.g[m] = ng;
        this.from[m] = n;
        this.push(f, m);
      }
    }
    if (direct) return { pts: direct, m: best - directPen, to: directTo };
    if (end < 0) return null;
    const mid: number[] = [];
    let first = end;
    for (let n = end; n >= 0; n = this.from[n]) {
      mid.push(this.gx[n], this.gy[n]);
      first = n;
    }
    const pts = [s.x, s.y];
    const sv = viaStart.get(first);
    if (sv) pts.push(sv[0], sv[1]);
    for (let i = mid.length - 2; i >= 0; i -= 2) pts.push(mid[i], mid[i + 1]);
    const e = ends.get(end)!;
    if (e.via) pts.push(e.via[0], e.via[1]);
    pts.push(e.to.x, e.to.y);
    return { pts, m: best - e.pen, to: e.to };
  }
}

/** Clip a segment to the tile square [0, EXTENT]^2 into `out` (rounded grid points, the cut ones exactly
 * on the edge); false if nothing of it is inside. */
function clipSegment(x0: number, y0: number, x1: number, y1: number, out: number[]): boolean {
  const inside = x0 >= 0 && x0 <= EXTENT && y0 >= 0 && y0 <= EXTENT && x1 >= 0 && x1 <= EXTENT && y1 >= 0 && y1 <= EXTENT;
  if (inside) {
    out[0] = Math.round(x0);
    out[1] = Math.round(y0);
    out[2] = Math.round(x1);
    out[3] = Math.round(y1);
    return out[0] !== out[2] || out[1] !== out[3];
  }
  const dx = x1 - x0, dy = y1 - y0;
  let t0 = 0, t1 = 1;
  const test = (p: number, q: number) => {
    if (p === 0) return q >= 0;
    const r = q / p;
    if (p < 0) {
      if (r > t1) return false;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return false;
      if (r < t1) t1 = r;
    }
    return true;
  };
  if (!(test(-dx, x0) && test(dx, EXTENT - x0) && test(-dy, y0) && test(dy, EXTENT - y0))) return false;
  const px = (t: number) => Math.min(EXTENT, Math.max(0, Math.round(x0 + t * dx)));
  const py = (t: number) => Math.min(EXTENT, Math.max(0, Math.round(y0 + t * dy)));
  out[0] = px(t0);
  out[1] = py(t0);
  out[2] = px(t1);
  out[3] = py(t1);
  return out[0] !== out[2] || out[1] !== out[3];
}

/** A road's vertex this near another road (tile units, about 1 m at zoom 14) with no vertex of that road
 * under it is a junction: the tiles' end-to-road gaps stop dead at 2 units (notes/T-2873.md). */
const JUNCTION_GAP = 2;
const JOIN_CELL = 128;
/** A line of this weight (`RoadLines.weights`) is kept only as a link: at most `LINK_MAX_M` long, each end
 * on a street (T-2882). Service roads as a class stay out, so a bus does not cut through car parks. */
export const LINK = 0.5;
const LINK_MAX_M = 60;

/** The road lines with a shared vertex added at every junction. OpenMapTiles simplifies each road on its
 * own, which drops the junction's vertex from a road that runs straight through it, so the roads meeting
 * there share none (T-2873). A vertex close to a road or two ground-level roads crossing is taken as a
 * junction; a motorway takes only another motorway's end, and its own ends (slip roads) join any road
 * at their level. `mpu` is ground metres per unit in this tile. */
export function joinRoads(rl: RoadLines, mpu: number): RoadLines {
  // clipped first: an added vertex on a segment crossing the tile's edge would move where it is cut,
  // and the two tiles' cut ends would no longer meet
  const lines: Int32Array[] = [], src: number[] = [];
  const scale = EXTENT / rl.extent, seg = [0, 0, 0, 0];
  rl.lines.forEach((line, li) => {
    let cur: number[] | null = null;
    const flush = () => {
      if (cur) lines.push(Int32Array.from(cur)), src.push(li);
    };
    for (let i = 2; i + 1 < line.length; i += 2) {
      if (!clipSegment(line[i - 2] * scale, line[i - 1] * scale, line[i] * scale, line[i + 1] * scale, seg)) continue;
      if (cur && cur[cur.length - 2] === seg[0] && cur[cur.length - 1] === seg[1]) cur.push(seg[2], seg[3]);
      else flush(), (cur = [...seg]);
    }
    flush();
  });
  const level = (l: number) => rl.levels?.[src[l]] ?? 0;
  const motor = (l: number) => (rl.weights?.[src[l]] ?? 1) >= MOTORWAY;
  const sLine: number[] = [], sAt: number[] = [];
  const cells = new Map<number, number[]>();
  const cellKey = (cx: number, cy: number) => (cx + 8) * 1024 + cy + 8;
  lines.forEach((l, li) => {
    for (let i = 0; i + 3 < l.length; i += 2) {
      const s = sLine.length;
      sLine.push(li);
      sAt.push(i);
      const cx1 = Math.floor((Math.max(l[i], l[i + 2]) + JUNCTION_GAP) / JOIN_CELL), cy1 = Math.floor((Math.max(l[i + 1], l[i + 3]) + JUNCTION_GAP) / JOIN_CELL);
      for (let cx = Math.floor((Math.min(l[i], l[i + 2]) - JUNCTION_GAP) / JOIN_CELL); cx <= cx1; cx++)
        for (let cy = Math.floor((Math.min(l[i + 1], l[i + 3]) - JUNCTION_GAP) / JOIN_CELL); cy <= cy1; cy++) {
          const k = cellKey(cx, cy), c = cells.get(k);
          if (c) c.push(s);
          else cells.set(k, [s]);
        }
    }
  });
  const isLink = (l: number) => rl.weights?.[src[l]] === LINK;
  const gap2 = JUNCTION_GAP * JUNCTION_GAP;
  const onStreet = (li: number, x: number, y: number) => {
    // a link cut by the tile's edge is judged by its end inside this tile
    if (x === 0 || y === 0 || x === EXTENT || y === EXTENT) return true;
    for (const s of cells.get(cellKey(Math.floor(x / JOIN_CELL), Math.floor(y / JOIN_CELL))) ?? []) {
      const lj = sLine[s];
      if (lj === li || isLink(lj) || motor(lj) || level(lj) !== 0) continue;
      const m = lines[lj], j = sAt[s], ax = m[j], ay = m[j + 1], dx = m[j + 2] - ax, dy = m[j + 3] - ay;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
      const ex = ax + t * dx - x, ey = ay + t * dy - y;
      if (ex * ex + ey * ey <= gap2) return true;
    }
    return false;
  };
  const dropped = new Set<number>();
  lines.forEach((l, li) => {
    if (!isLink(li)) return;
    let units = 0;
    for (let i = 0; i + 3 < l.length; i += 2) units += Math.hypot(l[i + 2] - l[i], l[i + 3] - l[i + 1]);
    if (units * mpu > LINK_MAX_M || !onStreet(li, l[0], l[1]) || !onStreet(li, l[l.length - 2], l[l.length - 1])) dropped.add(li);
  });
  /** per segment, the points to add to it: t along it, x, y, in threes */
  const adds = new Map<number, number[]>();
  const add = (s: number, x: number, y: number) => {
    const l = lines[sLine[s]], i = sAt[s], ax = l[i], ay = l[i + 1], bx = l[i + 2], by = l[i + 3];
    if ((x === ax && y === ay) || (x === bx && y === by)) return;
    const dx = bx - ax, dy = by - ay, t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy);
    const a = adds.get(s);
    if (a) a.push(t, x, y);
    else adds.set(s, [t, x, y]);
  };
  lines.forEach((l, li) => {
    if (dropped.has(li)) return;
    for (let i = 0; i < l.length; i += 2) {
      const x = l[i], y = l[i + 1];
      if (x === 0 || y === 0 || x === EXTENT || y === EXTENT) continue;
      const end = i === 0 || i === l.length - 2;
      if (!end && (level(li) !== 0 || motor(li))) continue;
      for (const s of cells.get(cellKey(Math.floor(x / JOIN_CELL), Math.floor(y / JOIN_CELL))) ?? []) {
        const lj = sLine[s];
        if (lj === li || dropped.has(lj) || level(lj) !== level(li) || (motor(lj) && !motor(li))) continue;
        const m = lines[lj], j = sAt[s], ax = m[j], ay = m[j + 1], dx = m[j + 2] - ax, dy = m[j + 3] - ay;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
        const ex = ax + t * dx - x, ey = ay + t * dy - y;
        if (ex * ex + ey * ey <= gap2) add(s, x, y);
      }
    }
  });
  for (const [k, c] of cells) {
    for (let p = 0; p < c.length; p++) {
      const lp = sLine[c[p]];
      if (level(lp) !== 0 || motor(lp) || dropped.has(lp)) continue;
      const A = lines[lp], i = sAt[c[p]], a0 = A[i], a1 = A[i + 1], ax = A[i + 2] - a0, ay = A[i + 3] - a1;
      for (let q = p + 1; q < c.length; q++) {
        const lq = sLine[c[q]];
        if (lq === lp || level(lq) !== 0 || motor(lq) || dropped.has(lq)) continue;
        const B = lines[lq], j = sAt[c[q]], b0 = B[j], b1 = B[j + 1], bx = B[j + 2] - b0, by = B[j + 3] - b1;
        const det = ax * by - ay * bx;
        if (det === 0) continue;
        const u = ((b0 - a0) * by - (b1 - a1) * bx) / det, v = ((b0 - a0) * ay - (b1 - a1) * ax) / det;
        if (u <= 0 || u >= 1 || v <= 0 || v >= 1) continue;
        const px = Math.round(a0 + u * ax), py = Math.round(a1 + u * ay);
        // one cell handles each crossing
        if (cellKey(Math.floor(px / JOIN_CELL), Math.floor(py / JOIN_CELL)) !== k) continue;
        add(c[p], px, py);
        add(c[q], px, py);
      }
    }
  }
  const out = lines.slice();
  const byLine = new Map<number, number[]>();
  for (const s of adds.keys()) {
    const b = byLine.get(sLine[s]);
    if (b) b.push(s);
    else byLine.set(sLine[s], [s]);
  }
  for (const [li, segs] of byLine) {
    segs.sort((p, q) => sAt[p] - sAt[q]);
    const l = lines[li], pts: number[] = [];
    let copied = 0;
    for (const s of segs) {
      const upto = sAt[s] + 2;
      for (let i = copied; i < upto; i++) pts.push(l[i]);
      copied = upto;
      const a = adds.get(s)!, order = Array.from({ length: a.length / 3 }, (_, n) => n).sort((p, q) => a[3 * p] - a[3 * q]);
      for (const n of order) {
        const x = a[3 * n + 1], y = a[3 * n + 2];
        if (pts[pts.length - 2] !== x || pts[pts.length - 1] !== y) pts.push(x, y);
      }
    }
    for (let i = copied; i < l.length; i++) pts.push(l[i]);
    out[li] = Int32Array.from(pts);
  }
  const kept = out.map((_, li) => li).filter((li) => !dropped.has(li));
  return {
    extent: EXTENT,
    lines: kept.map((li) => out[li]),
    weights: rl.weights && kept.map((li) => rl.weights![src[li]]),
    levels: rl.levels && kept.map((li) => rl.levels![src[li]]),
  };
}

/** Grid points (flat x, y) thinned to within `tol` units of the line. */
export function thin(pts: number[], tol: number): number[] {
  const n = pts.length / 2;
  if (n <= 2) return pts;
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack: [number, number][] = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const ax = pts[2 * a], ay = pts[2 * a + 1], dx = pts[2 * b] - ax, dy = pts[2 * b + 1] - ay, l2 = dx * dx + dy * dy;
    let worst = 0, at = -1;
    for (let i = a + 1; i < b; i++) {
      const px = pts[2 * i] - ax, py = pts[2 * i + 1] - ay;
      const t = l2 > 0 ? Math.max(0, Math.min(1, (px * dx + py * dy) / l2)) : 0;
      const d = Math.hypot(px - t * dx, py - t * dy);
      if (d > worst) (worst = d), (at = i);
    }
    if (worst > tol && at >= 0) {
      keep[at] = 1;
      stack.push([a, at], [at, b]);
    }
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[2 * i], pts[2 * i + 1]);
  return out;
}
