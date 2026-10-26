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
    const ea = new Int32Array(n), eb = new Int32Array(n), len = new Float32Array(n), next = new Int32Array(n * 2);
    ea.set(this.ea);
    eb.set(this.eb);
    len.set(this.elen);
    next.set(this.arcNext);
    this.ea = ea;
    this.eb = eb;
    this.elen = len;
    this.arcNext = next;
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

  private addEdge(a: number, b: number, m: number) {
    if (a === b) return;
    if (this.edges === this.ea.length) this.growEdges();
    const e = this.edges++;
    this.ea[e] = a;
    this.eb[e] = b;
    this.elen[e] = m;
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
    for (const line of rl.lines) {
      let pa = -1, pax = NaN, pay = NaN;
      for (let i = 2; i + 1 < line.length; i += 2) {
        if (!clipSegment(line[i - 2] * scale, line[i - 1] * scale, line[i] * scale, line[i + 1] * scale, seg)) continue;
        const [ax, ay, bx, by] = seg;
        const a = pa >= 0 && pax === ax && pay === ay ? pa : this.node(x0 + ax, y0 + ay, ax === 0 || ax === EXTENT, ay === 0 || ay === EXTENT);
        const b = this.node(x0 + bx, y0 + by, bx === 0 || bx === EXTENT, by === 0 || by === EXTENT);
        this.addEdge(a, b, Math.hypot(bx - ax, by - ay) * mpu);
        pa = b;
        pax = bx;
        pay = by;
      }
    }
    this.tiles.set(key, [first, this.edges]);
    this.version++;
  }

  /** The nearest point on a road within `maxUnits` of a grid point, or null. */
  snap(x: number, y: number, maxUnits: number): Snap | null {
    let best: Snap | null = null;
    const tx0 = Math.floor((x - maxUnits) / EXTENT), tx1 = Math.floor((x + maxUnits) / EXTENT);
    const ty0 = Math.floor((y - maxUnits) / EXTENT), ty1 = Math.floor((y + maxUnits) / EXTENT);
    for (let tx = tx0; tx <= tx1; tx++)
      for (let ty = ty0; ty <= ty1; ty++) {
        const r = this.tiles.get(RoadGraph.tileKey(tx, ty));
        if (!r) continue;
        for (let e = r[0]; e < r[1]; e++) {
          const a = this.ea[e], b = this.eb[e];
          const ax = this.gx[a], ay = this.gy[a], dx = this.gx[b] - ax, dy = this.gy[b] - ay;
          const l2 = dx * dx + dy * dy;
          const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
          const px = ax + t * dx, py = ay + t * dy, d = Math.hypot(px - x, py - y);
          if (d <= maxUnits && (!best || d < best.d)) best = { edge: e, t, x: px, y: py, d };
        }
      }
    return best;
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

  /** The shortest road between two snapped points, or null if none is under `capM` metres. */
  path(s: Snap, d: Snap, capM: number): RoadPath | null {
    const mpu = metresPerUnit(d.y);
    if (s.edge === d.edge) {
      return { pts: [s.x, s.y, d.x, d.y], m: Math.abs(d.t - s.t) * this.elen[s.edge] };
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
    const h = (n: number) => Math.hypot(this.gx[n] - d.x, this.gy[n] - d.y) * mpu;
    const start = (n: number, off: number) => {
      if (this.seen[n] === stamp && this.g[n] <= off) return;
      this.seen[n] = stamp;
      this.g[n] = off;
      this.from[n] = -1;
      this.push(off + h(n), n);
    };
    const sa = this.ea[s.edge], sb = this.eb[s.edge], sl = this.elen[s.edge];
    start(sa, s.t * sl);
    start(sb, (1 - s.t) * sl);
    const u = this.ea[d.edge], v = this.eb[d.edge], dl = this.elen[d.edge];
    const du = d.t * dl, dv = (1 - d.t) * dl;
    let best = capM, end = -1;
    while (this.hs > 0) {
      const n = this.pop();
      if (this.popF >= best) break;
      if (this.done[n] === stamp) continue;
      this.done[n] = stamp;
      const gn = this.g[n];
      if (n === u && gn + du < best) (best = gn + du), (end = n);
      if (n === v && gn + dv < best) (best = gn + dv), (end = n);
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
    if (end < 0) return null;
    const mid: number[] = [];
    for (let n = end; n >= 0; n = this.from[n]) mid.push(this.gx[n], this.gy[n]);
    const pts = [s.x, s.y];
    for (let i = mid.length - 2; i >= 0; i -= 2) pts.push(mid[i], mid[i + 1]);
    pts.push(d.x, d.y);
    return { pts, m: best };
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
