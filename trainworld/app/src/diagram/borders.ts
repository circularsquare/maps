// The network diagram's zone borders (T-2464; docs/ui-diagram.md "Zone borders", notes/T-2464.md), decided in the
// drawing the way the Tube map draws fare zones: a border says which drawn stations are in which zone, not where the
// ground boundary runs. On a grid over the drawing, the network's region is every cell within a margin of a line's
// casing or a pill, with narrow gaps and bays closed and enclosed gaps filled; each cell of it takes the zone of the
// pill nearest it through the region (steps of a cell, straight or diagonal). The region's boundaries are fitted
// with runs at 0, 45 and 90 degrees, kept clear of the lines and pills, and drawn under the lines with rounded
// corners, each shared border once. So every station is on its own zone's side, and a line leaving a zone crosses
// the border between its two stations.

import type { DiagramGraph } from "./graph";
import { pillSize, type Drawing } from "./draw";
import { DIRS8, hyp } from "./fmath";

/** a zone's outline in graph metres: its rings, x, y pairs, open */
export interface ZoneShape { id: string; rings: Float64Array[] }

export interface DiagramBorders {
  /** the borders as path commands (draw.ts `toPath`), layout units */
  cmds: Float64Array;
  /** the zones drawn, and per graph node its zone among them (-1 for none) */
  zones: string[];
  zoneOf: Int32Array;
  /** checks only (`zoneBorders(..., true)`): each border run as fitted (x, y pairs, before its corners are rounded),
   * with the zones on its left and right as it runs (-1 outside the region, -2 a station with no zone) */
  runs?: Run[];
}
export interface Run { p: Float64Array; left: number; right: number; closed: boolean; /** its commands in `cmds`, from, to */ c: [number, number] }

/** a station outside every outline is in the zone whose outline is nearest within this (cities.ts ZONE_REACH_M) */
const REACH_M = 40_000;
/** in pitches: the grid's cell, and how far the region reaches past the lines' casings and the pills */
const CELL = 1, MARGIN = 5;
/** gaps and bays narrower than about twice this many pitches are closed (T-2488, T-2491: wide, so outlines neither hug
 * the lines nor follow every branch) */
const CLOSE = 50;
/** a fitted border strays from the grid's by at most its clearance from the lines and pills less CLEAR, within
 * TOL_MIN and TOL_MAX pitches (TOL_EDGE on the region's edge, so its ends and corners keep their shape) */
const CLEAR = 0.3, TOL_MIN = 0.7, TOL_MAX = 12, TOL_EDGE = 3;
/** on the region's edge a fitted border may pass this many pitches further out than the grid's (T-2491) */
const TOL_OUT = 10;
/** corners are rounded with this radius, pitches */
export const BORDER_RADIUS = 3;
/** the region's margin and closing, the edge's tolerance and the corners' radius, pitches (checks vary them) */
export interface BorderOpts { margin: number; close: number; tolEdge: number; tolOut: number; radius: number }
/** the grid's tiles, cells a side */
const SHIFT = 5, T = 1 << SHIFT, TT = T * T;
/** the eight steps to a neighbouring cell (the four sides first), their lengths (5 and 7, about 1 and the root of 2)
 * and their steps in a tile's index */
const SI = [1, -1, 0, 0, 1, 1, -1, -1], SJ = [0, 0, 1, -1, 1, -1, 1, -1], SW = [5, 5, 5, 5, 7, 7, 7, 7];
const SC = SI.map((si, s) => si + SJ[s] * T);
/** per index in a tile, 1 where all eight neighbours are in the tile */
const INNER = Uint8Array.from({ length: TT }, (_, c) => +((c & (T - 1)) > 0 && (c & (T - 1)) < T - 1 && c >> SHIFT > 0 && c >> SHIFT < T - 1));
/** a cell's label: outside the region, a station with no zone, in the region but not yet reached */
const OUT = -1, NONE = -2, OPEN = -3;
const FAR = 65535;

/** The borders of the zones holding stations, in the drawing `d` of `g` laid out at `x`, `y`. */
export function zoneBorders(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, d: Drawing, shapes: ZoneShape[], detail = false, o: Partial<BorderOpts> = {}): DiagramBorders {
  const n = g.key.length;
  const MG = o.margin ?? MARGIN, CL = o.close ?? CLOSE, TE = o.tolEdge ?? TOL_EDGE, TO = o.tolOut ?? TOL_OUT, RAD = o.radius ?? BORDER_RADIUS;
  const zoneAll = stationZones(g, shapes);
  const drawnOf = new Int32Array(shapes.length).fill(-1), zones: string[] = [];
  for (let i = 0; i < n; i++) if (zoneAll[i] >= 0 && drawnOf[zoneAll[i]] < 0) (drawnOf[zoneAll[i]] = zones.length), zones.push(shapes[zoneAll[i]].id);
  const zoneOf = Int32Array.from(zoneAll, (z) => (z < 0 ? -1 : drawnOf[z]));
  if (!zones.length || !n) return { cmds: new Float64Array(0), zones, zoneOf, runs: detail ? [] : undefined };
  const pitch = d.pitch, h = CELL * pitch, margin = MG * pitch, reach = (MG + CL) * pitch;

  // the grid: cell (i, j) spans x from ox + i h, y from oy + j h; sparse, in tiles, a tile's width of nothing all
  // round. A cell is named slot * TT + its index in its tile.
  let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
  for (const r of d.ribbons) for (let k = 0; k < r.length; k += 2) (bx0 = Math.min(bx0, r[k])), (bx1 = Math.max(bx1, r[k])), (by0 = Math.min(by0, r[k + 1])), (by1 = Math.max(by1, r[k + 1]));
  for (let k = 0; k < n; k++) (bx0 = Math.min(bx0, x[k])), (bx1 = Math.max(bx1, x[k])), (by0 = Math.min(by0, y[k])), (by1 = Math.max(by1, y[k]));
  const pad = reach + (T + 2) * h, ox = bx0 - pad, oy = by0 - pad;
  const W = Math.ceil((bx1 + pad - ox) / h) + 2, H = Math.ceil((by1 + pad - oy) / h) + 2, NT = Math.ceil(W / T), NTY = Math.ceil(H / T);
  /** per cell: its label, its distance past the nearest casing or pill (Infinity past the margin, then grown), a
   * mark for the passes below, and its steps from the nearest station */
  type Tile = { lab: Int16Array; dist: Float32Array; mark: Uint8Array; sd: Uint16Array };
  const slots: Tile[] = new Array(NT * NTY);
  const newTile = (): Tile => ({ lab: new Int16Array(TT).fill(OUT), dist: new Float32Array(TT).fill(Infinity), mark: new Uint8Array(TT).fill(255), sd: new Uint16Array(TT).fill(FAR) });
  const slotOf = (i: number, j: number) => (j >> SHIFT) * NT + (i >> SHIFT);
  const at = (i: number, j: number) => ((j & (T - 1)) << SHIFT) | (i & (T - 1));
  /** the cell beside cell `c` of slot `s` in direction `k` */
  const beside = (s: number, c: number, k: number) => {
    const li = (c & (T - 1)) + SI[k], lj = (c >> SHIFT) + SJ[k];
    if (li >= 0 && lj >= 0 && li < T && lj < T) return s * TT + c + SC[k];
    return (s + (li < 0 ? -1 : li >= T ? 1 : 0) + (lj < 0 ? -NT : lj >= T ? NT : 0)) * TT + (((lj & (T - 1)) << SHIFT) | (li & (T - 1)));
  };

  /** each cell within `rho` past the segment (ax, ay)-(bx, by) drawn `rad` wide each side, with that distance */
  const cellsNear = (ax: number, ay: number, bx: number, by: number, rad: number, rho: number, f: (t: Tile, c: number, dd: number, id: number) => void) => {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, R = rad + rho;
    const j0 = Math.max(0, Math.floor((Math.min(ay, by) - R - oy) / h)), j1 = Math.min(H - 1, Math.floor((Math.max(ay, by) + R - oy) / h));
    for (let j = j0; j <= j1; j++) {
      const yc = oy + (j + 0.5) * h, s = span(ax, ay, bx, by, R, yc);
      if (!s) continue;
      const i0 = Math.max(0, Math.floor((s[0] - ox) / h)), i1 = Math.min(W - 1, Math.floor((s[1] - ox) / h));
      let tl: Tile | undefined, ts = -1;
      for (let i = i0; i <= i1; i++) {
        const xc = ox + (i + 0.5) * h;
        const t = l2 > 0 ? Math.max(0, Math.min(1, ((xc - ax) * dx + (yc - ay) * dy) / l2)) : 0;
        const dd = hyp(ax + t * dx - xc, ay + t * dy - yc) - rad;
        if (dd > rho) continue;
        if (slotOf(i, j) !== ts) (ts = slotOf(i, j)), (tl = slots[ts] ??= newTile());
        f(tl!, at(i, j), dd, ts * TT + at(i, j));
      }
    }
  };
  const nearer = (t: Tile, c: number, dd: number) => {
    if (dd < t.dist[c]) t.dist[c] = dd;
  };
  for (const r of d.ribbons) for (let k = 0; k + 3 < r.length; k += 2) cellsNear(r[k], r[k + 1], r[k + 2], r[k + 3], pitch / 2, margin, nearer);
  const pillR = (q: number) => {
    const [r, o] = pillSize(d, d.pillOf[q]);
    return r + o + d.pills[q][4];
  };
  d.pills.forEach((p, q) => cellsNear(p[0], p[1], p[2], p[3], pillR(q), margin, nearer));

  // closing: the region grown by CLOSE, then shrunk by CLOSE but never inside the margin, so gaps and bays narrower
  // than about twice CLOSE close up. `mark` holds steps from the margin while growing (0 inside it, 255 past the
  // growth), then from the cells grown past while shrinking (254 not yet reached).
  const grow = Math.round((CL / CELL) * 5);
  const buckets: number[][] = Array.from({ length: grow + 8 }, () => []);
  /** pushes each cell whose mark is `inside` with a neighbour's not so into buckets[0], or by its step (marking it so) */
  const edgeCells = (inside: (m: number) => boolean, byStep: boolean) => slots.forEach((t, s) => {
    const mk = t.mark;
    for (let c = 0; c < TT; c++) {
      if (!inside(mk[c])) continue;
      const li = c & (T - 1), lj = c >> SHIFT, inner = li > 0 && lj > 0 && li < T - 1 && lj < T - 1;
      let best = 255;
      for (let k = 0; k < 8 && best > 5; k++) {
        if (SW[k] >= best) continue;
        let m: number;
        if (inner) m = mk[c + SC[k]];
        else {
          const id = beside(s, c, k), nt = slots[Math.floor(id / TT)];
          m = nt ? nt.mark[id % TT] : 255;
        }
        if (!inside(m)) best = SW[k];
      }
      if (best === 255) continue;
      if (byStep) mk[c] = best;
      buckets[byStep ? best : 0].push(s * TT + c);
    }
  });
  slots.forEach((t) => {
    for (let c = 0; c < TT; c++) if (t.dist[c] <= margin) t.mark[c] = 0;
  });
  edgeCells((m) => m === 0, false);
  for (let b = 0; b <= grow; b++) {
    for (const id of buckets[b]) {
      const s = Math.floor(id / TT), c = id - s * TT;
      if (slots[s].mark[c] !== b) continue;
      for (let k = 0; k < 8; k++) {
        const e = b + SW[k];
        if (e > grow) continue;
        const nid = INNER[c] ? id + SC[k] : beside(s, c, k), ns = INNER[c] ? s : Math.floor(nid / TT), nc = nid - ns * TT, nt = (slots[ns] ??= newTile());
        if (e < nt.mark[nc]) (nt.mark[nc] = e), (nt.dist[nc] = margin + (e / 5) * h), buckets[e].push(nid);
      }
    }
    buckets[b].length = 0;
  }
  slots.forEach((t) => {
    for (let c = 0; c < TT; c++) t.mark[c] = t.mark[c] <= grow ? 254 : 255;
  });
  edgeCells((m) => m < 255, true);
  for (let b = 0; b < grow; b++) {
    for (const id of buckets[b]) {
      const s = Math.floor(id / TT), c = id - s * TT;
      if (slots[s].mark[c] !== b) continue;
      for (let k = 0; k < 8; k++) {
        const e = b + SW[k];
        if (e >= grow) continue;
        const nid = INNER[c] ? id + SC[k] : beside(s, c, k), ns = INNER[c] ? s : Math.floor(nid / TT), nc = nid - ns * TT, nt = slots[ns];
        if (nt && nt.mark[nc] < 255 && e < nt.mark[nc]) (nt.mark[nc] = e), buckets[e].push(nid);
      }
    }
  }
  slots.forEach((t) => {
    for (let c = 0; c < TT; c++) if (t.dist[c] <= margin || t.mark[c] === 254) t.lab[c] = OPEN;
  });
  fillHoles();

  // each cell of the region takes the zone of the pill nearest it through the region: from the cells under each pill,
  // by steps of 5 and 7 (`sd`), the nearest first (a station passed and never stopped at has no pill, and no say)
  const ring: number[][] = Array.from({ length: 8 }, () => []);
  let pending = 0;
  const seed = (t: Tile, c: number, z: number, id: number) => {
    if (t.sd[c] === 0 || t.lab[c] === OUT) return;
    (t.sd[c] = 0), (t.lab[c] = z), ring[0].push(id), pending++;
  };
  const labelOf = (k: number) => (zoneOf[k] >= 0 ? zoneOf[k] : NONE);
  d.pills.forEach((p, q) => {
    const z = labelOf(d.pillOf[q]);
    cellsNear(p[0], p[1], p[2], p[3], pillR(q), 0, (t, c, _, id) => seed(t, c, z, id));
    // a small pill may hold no cell's middle: the cells its ends are in
    for (const [qx, qy] of [[p[0], p[1]], [p[2], p[3]]]) {
      const i = Math.floor((qx - ox) / h), j = Math.floor((qy - oy) / h), s = slotOf(i, j);
      if (slots[s]) seed(slots[s], at(i, j), z, s * TT + at(i, j));
    }
  });
  for (let cur = 0; pending > 0; cur++) {
    const bucket = ring[cur & 7];
    for (const id of bucket) {
      pending--;
      const s = Math.floor(id / TT), c = id - s * TT, t = slots[s];
      if (t.sd[c] !== cur) continue;
      for (let k = 0; k < 8; k++) {
        const e = cur + SW[k], nid = INNER[c] ? id + SC[k] : beside(s, c, k), ns = INNER[c] ? s : Math.floor(nid / TT), nc = nid - ns * TT, nt = slots[ns];
        if (!nt || nt.lab[nc] === OUT || e >= nt.sd[nc]) continue;
        (nt.sd[nc] = e), (nt.lab[nc] = t.lab[c]), ring[e & 7].push(nid), pending++;
      }
    }
    bucket.length = 0;
  }

  // each cell's clearance from the lines, the pills and the region's edge, in steps of 5 and 7 up to `cap` (`mark`):
  // how far a border between two zones may stray there
  const cap = Math.min(250, Math.ceil(((TOL_MAX + CLEAR) / CELL) * 5));
  const cb: number[][] = Array.from({ length: cap + 8 }, () => []);
  slots.forEach((t) => {
    for (let c = 0; c < TT; c++) if (t.lab[c] === OPEN) t.lab[c] = OUT;
  });
  slots.forEach((t, s) => {
    const lab = t.lab, mk = t.mark;
    mk.fill(255);
    for (let c = 0; c < TT; c++) {
      if (lab[c] === OUT) continue;
      let edge = t.dist[c] <= 0;
      for (let k = 0; k < 4 && !edge; k++) {
        const nid = INNER[c] ? s * TT + c + SC[k] : beside(s, c, k), ns = INNER[c] ? s : Math.floor(nid / TT), nt = slots[ns];
        edge = !nt || nt.lab[nid - ns * TT] === OUT;
      }
      if (edge) (mk[c] = 0), cb[0].push(s * TT + c);
    }
  });
  for (let b = 0; b < cap; b++) {
    for (const id of cb[b]) {
      const s = Math.floor(id / TT), c = id - s * TT;
      if (slots[s].mark[c] !== b) continue;
      for (let k = 0; k < 8; k++) {
        const e = b + SW[k];
        if (e >= cap) continue;
        const nid = INNER[c] ? id + SC[k] : beside(s, c, k), ns = INNER[c] ? s : Math.floor(nid / TT), nc = nid - ns * TT, nt = slots[ns];
        if (nt && nt.lab[nc] !== OUT && e < nt.mark[nc]) (nt.mark[nc] = e), cb[e].push(nid);
      }
    }
  }
  // the boundaries between cells of different labels, as unit edges between grid corners (corner id j * (W + 1) + i),
  // each with the label on its left and right as it runs and its clearance: on the region's edge from the lines and
  // pills, between zones from those and the edge
  const W1 = W + 1;
  const ea: number[] = [], eb: number[] = [], el: number[] = [], er: number[] = [], clr: number[] = [];
  const cell = (i: number, j: number): [Tile | undefined, number] => [i < 0 || j < 0 || i >= W || j >= H ? undefined : slots[slotOf(i, j)], at(i, j)];
  const labAt = (i: number, j: number) => {
    const [t, c] = cell(i, j);
    return t ? t.lab[c] : OUT;
  };
  const clrAt = (i: number, j: number, edge: boolean) => {
    const [t, c] = cell(i, j);
    return !t || t.lab[c] === OUT ? Infinity : edge ? t.dist[c] : (Math.min(t.mark[c], cap) / 5) * h;
  };
  slots.forEach((t, s) => {
    const lab = t.lab, ti = (s % NT) << SHIFT, tj = Math.floor(s / NT) << SHIFT;
    for (let c = 0; c < TT; c++) {
      const li = c & (T - 1), lj = c >> SHIFT, i = ti + li, j = tj + lj, a = lab[c];
      const r = li < T - 1 ? lab[c + 1] : labAt(i + 1, j), u = lj < T - 1 ? lab[c + T] : labAt(i, j + 1);
      if (r === a && u === a) continue;
      // west of a northward edge is on its left
      if (r !== a) {
        const edge = a === OUT || r === OUT;
        ea.push(j * W1 + i + 1), eb.push((j + 1) * W1 + i + 1), el.push(a), er.push(r), clr.push(Math.min(clrAt(i, j, edge), clrAt(i + 1, j, edge)));
      }
      // north of an eastward edge
      if (u !== a) {
        const edge = a === OUT || u === OUT;
        ea.push((j + 1) * W1 + i), eb.push((j + 1) * W1 + i + 1), el.push(u), er.push(a), clr.push(Math.min(clrAt(i, j, edge), clrAt(i, j + 1, edge)));
      }
    }
  });
  const inc = new Map<number, number[]>();
  const addInc = (v: number, e: number) => {
    const l = inc.get(v);
    if (l) l.push(e);
    else inc.set(v, [e]);
  };
  ea.forEach((v, e) => (addInc(v, e), addInc(eb[e], e)));
  const cx = (v: number) => ox + (v % W1) * h, cy = (v: number) => oy + Math.floor(v / W1) * h;

  // the pills' ends and middles, by squares of PG: a fitted run may not move any of them to its other side
  const PG = 8 * pitch, pg = new Map<string, number[]>();
  for (const p of d.pills) for (const [qx, qy] of [[p[0], p[1]], [p[2], p[3]], [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2]]) {
    const key = `${Math.floor(qx / PG)},${Math.floor(qy / PG)}`, l = pg.get(key);
    if (l) l.push(qx, qy);
    else pg.set(key, [qx, qy]);
  }
  /** whether the way from point i to point j of px, py (bent at bx, by, or straight) leaves every pill on its side:
   * none inside the polygon the points and the way make */
  const keepsSides = (px: number[], py: number[]) => (i: number, j: number, bx: number, by: number) => {
    const poly = px.slice(i, j + 1).flatMap((x, k) => [x, py[i + k]]);
    if (bx === bx) poly.push(bx, by);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = 0; k < poly.length; k += 2) (x0 = Math.min(x0, poly[k])), (x1 = Math.max(x1, poly[k])), (y0 = Math.min(y0, poly[k + 1])), (y1 = Math.max(y1, poly[k + 1]));
    for (let gx = Math.floor(x0 / PG); gx <= Math.floor(x1 / PG); gx++) for (let gy = Math.floor(y0 / PG); gy <= Math.floor(y1 / PG); gy++) {
      const l = pg.get(`${gx},${gy}`);
      if (l) for (let q = 0; q < l.length; q += 2) if (l[q] >= x0 && l[q] <= x1 && l[q + 1] >= y0 && l[q + 1] <= y1 && inRings([poly], l[q], l[q + 1])) return false;
    }
    return true;
  };

  // runs: chains of edges between corners where borders meet (other than two edges), each traced once
  const seen = new Uint8Array(ea.length), runs: Run[] = [], cmds: number[] = [];
  const tolOf = (c: number, edge: boolean) => Math.max(TOL_MIN * pitch, Math.min((edge ? TE : TOL_MAX) * pitch, c - CLEAR * pitch));
  for (let e0 = 0; e0 < ea.length; e0++) {
    if (seen[e0] || (el[e0] < 0 && er[e0] < 0)) continue;
    seen[e0] = 1;
    // forward from e0's end, then back from its start
    const fwd: [number, number][] = [], back: [number, number][] = [];
    let closed = false;
    for (const [list, from] of [[fwd, eb[e0]], [back, ea[e0]]] as const) {
      let v = from;
      for (;;) {
        const l = inc.get(v)!;
        if (l.length !== 2) break;
        const e = l[0] === (list.length ? list[list.length - 1][0] : e0) ? l[1] : l[0];
        if (e === e0) {
          closed = true;
          break;
        }
        if (seen[e]) break;
        seen[e] = 1;
        const nv = ea[e] === v ? eb[e] : ea[e];
        list.push([e, nv]);
        v = nv;
      }
      if (closed) break;
    }
    // points at the edges' middles, the corners where borders meet at the ends; the labels as e0 runs, which is the
    // way the run goes
    const order: number[] = [...back.map(([e]) => e).reverse(), e0, ...fwd.map(([e]) => e)];
    const edge = el[e0] === OUT || er[e0] === OUT;
    const px: number[] = [], py: number[] = [], tol: number[] = [], tolOut: number[] = [];
    const corner = (v: number) => (px.push(cx(v)), py.push(cy(v)), tol.push(0), tolOut.push(0));
    if (!closed) corner(back.length ? back[back.length - 1][1] : ea[e0]);
    for (const e of order) px.push((cx(ea[e]) + cx(eb[e])) / 2), py.push((cy(ea[e]) + cy(eb[e])) / 2), tol.push(tolOf(clr[e], edge)), tolOut.push(TO * pitch);
    if (!closed) corner(fwd.length ? fwd[fwd.length - 1][1] : eb[e0]);
    let v: Float64Array;
    if (closed) {
      // a ring starts at its point farthest south-west, where it turns anyway
      let s = 0;
      for (let k = 1; k < px.length; k++) if (px[k] + py[k] < px[s] + py[s]) s = k;
      const rot = (a: number[]) => [...a.slice(s), ...a.slice(0, s + 1)];
      const qx = rot(px), qy = rot(py);
      v = fitOcti(qx, qy, rot(tol), keepsSides(qx, qy), edge ? { tol: rot(tolOut), left: el[e0] === OUT } : undefined);
    } else v = fitOcti(px, py, tol, keepsSides(px, py), edge ? { tol: tolOut, left: el[e0] === OUT } : undefined);
    const c0 = cmds.length;
    rounded(v, closed, RAD * pitch, cmds);
    runs.push({ p: v, left: el[e0], right: er[e0], closed, c: [c0, cmds.length] });
  }
  return { cmds: Float64Array.from(cmds), zones, zoneOf, runs: detail ? runs : undefined };

  /** Every gap the network encloses (too wide for the closing) is taken into the region, so a zone's outline has no
   * holes: what the outside, flooded from the grid's edge, does not reach. */
  function fillHoles() {
    // mark 1: reached; slots never touched are flooded whole (`-1 - slot` in the queue)
    const slotSeen = new Uint8Array(NT * NTY), queue: number[] = [-1];
    slotSeen[0] = 1;
    slots.forEach((t) => t.mark.fill(0));
    const cellIn = (s: number, c: number) => {
      const t = slots[s];
      if (!t) {
        if (!slotSeen[s]) (slotSeen[s] = 1), queue.push(-1 - s);
      } else if (t.lab[c] === OUT && !t.mark[c]) (t.mark[c] = 1), queue.push(s * TT + c);
    };
    for (let q = 0; q < queue.length; q++) {
      const v = queue[q];
      if (v >= 0) {
        const s = Math.floor(v / TT), c = v - s * TT;
        for (let k = 0; k < 4; k++) {
          const nid = beside(s, c, k), ns = Math.floor(nid / TT);
          cellIn(ns, nid - ns * TT);
        }
        continue;
      }
      // a whole slot: the slots beside it, or their cells along the side it shares
      const s = -1 - v, si = s % NT, sj = Math.floor(s / NT);
      for (let k = 0; k < 4; k++) {
        const ni = si + SI[k], nj = sj + SJ[k];
        if (ni < 0 || nj < 0 || ni >= NT || nj >= NTY) continue;
        const ns = nj * NT + ni;
        if (!slots[ns]) cellIn(ns, 0);
        else for (let m = 0; m < T; m++) cellIn(ns, SI[k] ? (m << SHIFT) | (SI[k] > 0 ? 0 : T - 1) : ((SJ[k] > 0 ? 0 : T - 1) << SHIFT) | m);
      }
    }
    for (let s = 0; s < NT * NTY; s++) if (!slots[s] && !slotSeen[s]) (slots[s] = newTile()).mark.fill(0);
    slots.forEach((t) => {
      for (let c = 0; c < TT; c++) if (t.lab[c] === OUT && !t.mark[c]) (t.lab[c] = OPEN), (t.dist[c] = reach);
    });
  }
}

/** per graph node its shape's index: the outline holding it, else the nearest within REACH_M, else -1 */
export function stationZones(g: DiagramGraph, shapes: ZoneShape[]): Int32Array {
  const n = g.key.length;
  const box = shapes.map((s) => {
    const b = [Infinity, Infinity, -Infinity, -Infinity];
    for (const r of s.rings) for (let k = 0; k < r.length; k += 2) (b[0] = Math.min(b[0], r[k])), (b[1] = Math.min(b[1], r[k + 1])), (b[2] = Math.max(b[2], r[k])), (b[3] = Math.max(b[3], r[k + 1]));
    return b;
  });
  const zoneAll = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const px = g.x[i], py = g.y[i];
    let best = -1, bd = REACH_M;
    for (let z = 0; z < shapes.length && zoneAll[i] < 0; z++) {
      const b = box[z];
      if (px < b[0] - bd || px > b[2] + bd || py < b[1] - bd || py > b[3] + bd) continue;
      if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3] && inRings(shapes[z].rings, px, py)) zoneAll[i] = z;
      else {
        const dd = distRings(shapes[z].rings, px, py);
        if (dd < bd) (bd = dd), (best = z);
      }
    }
    if (zoneAll[i] < 0) zoneAll[i] = best;
  }
  return zoneAll;
}

/** the x-span where the row at `yc` meets the capsule round (ax, ay)-(bx, by) of radius `rho`, or null */
function span(ax: number, ay: number, bx: number, by: number, rho: number, yc: number): [number, number] | null {
  let lo = Infinity, hi = -Infinity;
  for (const [px, py] of [[ax, ay], [bx, by]]) {
    const dy = yc - py;
    if (Math.abs(dy) > rho) continue;
    const w = Math.sqrt(rho * rho - dy * dy);
    (lo = Math.min(lo, px - w)), (hi = Math.max(hi, px + w));
  }
  const dx = bx - ax, dy = by - ay, L = hyp(dx, dy);
  if (L > 0) {
    const ux = dx / L, uy = dy / L;
    let xl = -Infinity, xh = Infinity;
    // along: (x - ax) ux + (yc - ay) uy in [0, L]; across: -(x - ax) uy + (yc - ay) ux in [-rho, rho]
    for (const [c, k, a, b] of [[ux, (yc - ay) * uy, 0, L], [-uy, (yc - ay) * ux, -rho, rho]]) {
      if (Math.abs(c) < 1e-12) {
        if (k < a || k > b) xl = Infinity;
        continue;
      }
      const p = (a - k) / c, q = (b - k) / c;
      (xl = Math.max(xl, Math.min(p, q))), (xh = Math.min(xh, Math.max(p, q)));
    }
    if (xl <= xh) (lo = Math.min(lo, ax + xl)), (hi = Math.max(hi, ax + xh));
  }
  return lo <= hi ? [lo, hi] : null;
}

/** A polyline through px, py fitted with runs at 0, 45 and 90 degrees, its ends kept: from each point, the farthest
 * point reachable by one run or two (one straight, one diagonal) passing within `tol` of every point between (and
 * that `keeps` allows); runs going on the same way joined. Vertices as x, y pairs. */
export function fitOcti(px: number[], py: number[], tol: number[], keeps?: (i: number, j: number, bx: number, by: number) => boolean, outer?: { tol: number[]; left: boolean }): Float64Array {
  const m = px.length, out: number[] = [px[0], py[0]];
  let dirX = 0, dirY = 0;
  /** the ways from point i to point j (one run, or a bend at bx, by), each that fits */
  const ways = (i: number, j: number): { bx: number; by: number; dev: number; fx: number; fy: number }[] => {
    const dx = px[j] - px[i], dy = py[j] - py[i], L = hyp(dx, dy);
    if (L < 1e-12) return [{ bx: NaN, by: NaN, dev: 0, fx: 0, fy: 0 }];
    // the two octilinear directions either side of (dx, dy), and its parts along them
    let k = 0, bestDot = -Infinity;
    for (let q = 0; q < 8; q++) {
      const dot = DIRS8[q][0] * dx + DIRS8[q][1] * dy;
      if (dot > bestDot) (bestDot = dot), (k = q);
    }
    const ax = DIRS8[k][0], ay = DIRS8[k][1], cross = ax * dy - ay * dx;
    const out: { bx: number; by: number; dev: number; fx: number; fy: number }[] = [];
    if (Math.abs(cross) < 1e-9 * L) {
      const dev = deviation(i, j, NaN, NaN);
      if (dev >= 0) out.push({ bx: NaN, by: NaN, dev, fx: ax, fy: ay });
      return out;
    }
    const k2 = (k + (cross > 0 ? 1 : 7)) % 8, cx = DIRS8[k2][0], cy = DIRS8[k2][1];
    // dx = a A + b C
    const det = ax * cy - ay * cx, a = (dx * cy - dy * cx) / det, b = (ax * dy - ay * dx) / det;
    for (const [bx, by, fx, fy] of [[px[i] + a * ax, py[i] + a * ay, ax, ay], [px[i] + b * cx, py[i] + b * cy, cx, cy]]) {
      const dev = deviation(i, j, bx, by);
      if (dev >= 0) out.push({ bx, by, dev, fx, fy });
    }
    return out;
  };
  /** the worst share of its tolerance any point between i and j strays from the path (and the bend from the
   * points), or -1 where one strays past it */
  const deviation = (i: number, j: number, bx: number, by: number): number => {
    const bent = bx === bx;
    let worst = 0, bendNear = Infinity, bendTol = 0;
    for (let k = i + 1; k < j; k++) {
      const e1 = segD(px[k], py[k], px[i], py[i], bent ? bx : px[j], bent ? by : py[j]), e2 = bent ? segD(px[k], py[k], bx, by, px[j], py[j]) : Infinity;
      let e = Math.min(e1, e2), t = tol[k];
      // on the region's edge the way may pass further out than in: it fills notches, never nears the lines (T-2491)
      if (outer && e > t) {
        const [ax, ay, cx, cy] = e1 <= e2 ? [px[i], py[i], bent ? bx : px[j], bent ? by : py[j]] : [bx, by, px[j], py[j]];
        if (((cx - ax) * (py[k] - ay) - (cy - ay) * (px[k] - ax) < 0) === outer.left) t = outer.tol[k];
      }
      if (e > t) return -1;
      worst = Math.max(worst, e / t);
      if (bent) {
        e = hyp(px[k] - bx, py[k] - by);
        if (e < bendNear) (bendNear = e), (bendTol = outer ? Math.max(tol[k], outer.tol[k]) : tol[k]);
      }
    }
    if (bent && j > i + 1 && bendNear > bendTol + 1e-9) return -1;
    if (keeps && !keeps(i, j, bx, by)) return -1;
    return worst;
  };
  const pick = (i: number, j: number) => {
    const w = ways(i, j);
    if (!w.length) return null;
    // a way going on as the last run went, else the nearest
    const on = (p: { fx: number; fy: number }) => +(snap(p.fx) === dirX && snap(p.fy) === dirY);
    w.sort((p, q) => on(q) - on(p) || p.dev - q.dev);
    return w[0];
  };
  for (let i = 0; i < m - 1; ) {
    // lo fits, hi does not (m: past the end): doubling, then halving (the fits are mostly monotone)
    let lo = i + 1, hi = m, step = 1;
    while (lo + step < m) {
      if (!pick(i, lo + step)) {
        hi = lo + step;
        break;
      }
      (lo += step), (step *= 2);
    }
    if (hi === m && lo < m - 1) {
      if (pick(i, m - 1)) lo = m - 1;
      else hi = m - 1;
    }
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (pick(i, mid)) lo = mid;
      else hi = mid;
    }
    const w = pick(i, lo)!;
    if (w.bx === w.bx) out.push(w.bx, w.by);
    out.push(px[lo], py[lo]);
    const lx = px[lo] - out[out.length - 4], ly = py[lo] - out[out.length - 3], ll = hyp(lx, ly);
    if (ll > 0) (dirX = snap(lx / ll)), (dirY = snap(ly / ll));
    i = lo;
  }
  return Float64Array.from(joinStraight(out));
}

const snap = (v: number) => Math.round(v * 1e6) / 1e6;

/** vertices with those where the way goes straight on (or doubles back on itself by nothing) dropped */
function joinStraight(v: number[]): number[] {
  const out: number[] = [v[0], v[1]];
  for (let k = 2; k < v.length; k += 2) {
    const n = out.length;
    if (hyp(v[k] - out[n - 2], v[k + 1] - out[n - 1]) < 1e-9) continue;
    if (n >= 4) {
      const ax = out[n - 2] - out[n - 4], ay = out[n - 1] - out[n - 3], bx = v[k] - out[n - 2], by = v[k + 1] - out[n - 1];
      if (Math.abs(ax * by - ay * bx) < 1e-9 * hyp(ax, ay) * hyp(bx, by) && ax * bx + ay * by > 0) {
        (out[n - 2] = v[k]), (out[n - 1] = v[k + 1]);
        continue;
      }
    }
    out.push(v[k], v[k + 1]);
  }
  return out;
}

/** the polyline `v` as path commands, its corners rounded with radius `r` (less where its runs are short) */
export function rounded(v: Float64Array, closed: boolean, r: number, cmd: number[]): void {
  let m = v.length / 2;
  if (closed && m > 2 && hyp(v[0] - v[m * 2 - 2], v[1] - v[m * 2 - 1]) < 1e-9) m--;
  if (m < 2) return;
  const X = (k: number) => v[((k + m) % m) * 2], Y = (k: number) => v[((k + m) % m) * 2 + 1];
  /** where the curve round corner k starts and ends, or null for no corner */
  const corner = (k: number): [number, number, number, number] | null => {
    const ax = X(k) - X(k - 1), ay = Y(k) - Y(k - 1), bx = X(k + 1) - X(k), by = Y(k + 1) - Y(k);
    const la = hyp(ax, ay), lb = hyp(bx, by);
    if (la < 1e-12 || lb < 1e-12) return null;
    const sn = (ax * by - ay * bx) / (la * lb), cs = (ax * bx + ay * by) / (la * lb);
    if (Math.abs(sn) < 1e-9 && cs > 0) return null;
    const t = Math.min(r * (cs > -1 + 1e-12 ? Math.abs(sn) / (1 + cs) : 1e16), 0.49 * la, 0.49 * lb);
    return [X(k) - (ax / la) * t, Y(k) - (ay / la) * t, X(k) + (bx / lb) * t, Y(k) + (by / lb) * t];
  };
  if (!closed) {
    cmd.push(0, v[0], v[1]);
    for (let k = 1; k < m - 1; k++) {
      const c = corner(k);
      if (!c) continue;
      cmd.push(1, c[0], c[1], 2, X(k), Y(k), c[2], c[3]);
    }
    cmd.push(1, X(m - 1), Y(m - 1));
    return;
  }
  const c0 = corner(0);
  cmd.push(0, c0 ? c0[2] : X(0), c0 ? c0[3] : Y(0));
  for (let k = 1; k <= m; k++) {
    const c = corner(k);
    if (!c) cmd.push(1, X(k), Y(k));
    else cmd.push(1, c[0], c[1], 2, X(k), Y(k), c[2], c[3]);
  }
}

export function segD(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return hyp(px - ax - dx * t, py - ay - dy * t);
}

/** whether (x, y) is inside rings (even-odd) */
export function inRings(rings: ArrayLike<number>[], x: number, y: number): boolean {
  let inside = false;
  for (const r of rings) {
    const m = r.length;
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) {
      const yi = r[i + 1], yj = r[j + 1];
      if (yi > y !== yj > y && x < ((r[j] - r[i]) * (y - yi)) / (yj - yi) + r[i]) inside = !inside;
    }
  }
  return inside;
}

/** the distance from (x, y) to the rings' edges */
function distRings(rings: Float64Array[], x: number, y: number): number {
  let best = Infinity;
  for (const r of rings) {
    const m = r.length;
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) best = Math.min(best, segD(x, y, r[j], r[j + 1], r[i], r[i + 1]));
  }
  return best;
}
