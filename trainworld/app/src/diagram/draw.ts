// The network diagram drawn (T-2160's rough view, before T-2161's): each line one continuous ribbon
// through its pieces, at its slot's offset, corners rounded so a bundle's bends are concentric; a
// white pill with an ink outline across the lines stopping at a station. Paths are in layout units
// (y north), drawn through one canvas transform. Port of the round 2 mockups' draw2.py.

import type { DiagramGraph } from "./graph";
import type { DiagramLayout, Piece } from "./layout";

/** a stroke is this share of a pitch wide; the rest is the gap between neighbours */
export const STROKE_SHARE = 5 / 6.4;
/** a corner's radius is this many stroke widths, plus the bundle's half width and the offset */
const RADIUS_W = 6;
/** a pill's half width and its outline, in stroke widths */
const PILL_R = 0.95, PILL_OUTLINE = 0.32;
/** stop points of one station this many pitches apart or nearer share a pill */
const PILL_JOIN = 1.6;

export interface Drawing {
  /** a pitch, stroke width, layout units */
  pitch: number;
  stroke: number;
  lines: { colour: string; path: Path2D }[];
  /** per pill: its two ends and how far its points lie off its axis */
  pills: [number, number, number, number, number][];
  /** joins between pills of one station: two centres */
  passages: [number, number, number, number][];
  box: [number, number, number, number];
}

/** a polyline offset left by `o`, one point per point (mitred, capped) */
function offsetKeep(v: Float64Array, o: number): Float64Array {
  const n = v.length / 2, out = new Float64Array(v.length);
  const dir = (i: number, step: number): [number, number] | null => {
    for (; i >= 0 && i < n - 1; i += step) {
      const dx = v[i * 2 + 2] - v[i * 2], dy = v[i * 2 + 3] - v[i * 2 + 1], l = Math.hypot(dx, dy);
      if (l > 1e-9) return [dx / l, dy / l];
    }
    return null;
  };
  for (let i = 0; i < n; i++) {
    let a = i > 0 ? dir(i - 1, -1) : null, b = i < n - 1 ? dir(i, 1) : null;
    a ??= b;
    b ??= a;
    if (!a || !b) {
      out[i * 2] = v[i * 2];
      out[i * 2 + 1] = v[i * 2 + 1];
      continue;
    }
    let mx = -a[1] - b[1], my = a[0] + b[0];
    const l = Math.hypot(mx, my);
    if (l < 1e-9) (mx = -a[1]), (my = a[0]);
    else (mx /= l), (my /= l);
    const c = Math.max(0.4, mx * -a[1] + my * a[0]);
    out[i * 2] = v[i * 2] + (mx * o) / c;
    out[i * 2 + 1] = v[i * 2 + 1] + (my * o) / c;
  }
  return out;
}

type V = [number, number, number, number, number]; // x, y, offset left of travel, bundle half width, node (-1 none)

export function drawing(g: DiagramGraph, lay: DiagramLayout): Drawing {
  const pitch = lay.pu, W = pitch * STROKE_SHARE, R0 = RADIUS_W * W;
  const pieceOf = new Map<number, [Piece, number, number]>();
  const n = g.key.length;
  for (const r of lay.pieces) {
    const p = r.nodes;
    for (let i = 0; i + 1 < p.length; i++) {
      pieceOf.set(p[i] * n + p[i + 1], [r, i, 1]);
      pieceOf.set(p[i + 1] * n + p[i], [r, i + 1, -1]);
    }
  }
  const offs = new Map<Piece, Map<string, [number, Float64Array, number]>>();
  for (const r of lay.pieces) {
    const m = new Map<string, [number, Float64Array, number]>(), k = r.cols.length;
    r.cols.forEach((c, j) => {
      const o = ((k - 1) / 2 - j) * pitch;
      m.set(c, [o, offsetKeep(r.verts, o), ((k - 1) / 2) * pitch]);
    });
    offs.set(r, m);
  }
  const lines: Drawing["lines"] = [];
  const stopsAt = new Map<number, [number, number][]>();
  for (const ln of g.lines) {
    const ch = ln.chain, c = ln.colour;
    const V: V[] = [];
    let i = 0;
    while (i < ch.length - 1) {
      const first = pieceOf.get(ch[i] * n + ch[i + 1]);
      if (!first) { i++; continue; }
      const [r, a, d] = first;
      let j = i;
      while (j < ch.length - 1) {
        const x = pieceOf.get(ch[j] * n + ch[j + 1]);
        if (!x || x[0] !== r || x[2] !== d) break;
        j++;
      }
      const b = a + d * (j - i);
      const [o, op, hw] = offs.get(r)!.get(c) ?? [0, r.verts, 0];
      const va = r.at[a], vb = r.at[b];
      const nodeAt = new Map<number, number>();
      for (let t = 0; t <= j - i; t++) nodeAt.set(r.at[a + d * t], i + t);
      const run: V[] = [];
      for (let t = va; d === 1 ? t <= vb : t >= vb; t += d) {
        const ci = nodeAt.get(t);
        run.push([op[t * 2], op[t * 2 + 1], o * d, hw, ci === undefined ? -1 : ci]);
      }
      if (V.length && run.length > 1) {
        // join at the shared node: the corner where the two runs' offset lines meet, or a 45 degree jog
        const p1 = V[V.length - 1], p0 = V.length > 1 ? V[V.length - 2] : null, q1 = run[0], q2 = run[1];
        let d1x = p0 ? p1[0] - p0[0] : q2[0] - q1[0], d1y = p0 ? p1[1] - p0[1] : q2[1] - q1[1];
        let d2x = q2[0] - q1[0], d2y = q2[1] - q1[1];
        const l1 = Math.hypot(d1x, d1y), l2 = Math.hypot(d2x, d2y);
        if (l1 > 1e-9 && l2 > 1e-9) {
          d1x /= l1, d1y /= l1, d2x /= l2, d2y /= l2;
          const cr = d1x * d2y - d1y * d2x;
          if (Math.abs(cr) > 0.02) {
            const t = ((q1[0] - p1[0]) * d2y - (q1[1] - p1[1]) * d2x) / cr;
            const xx = p1[0] + d1x * t, xy = p1[1] + d1y * t;
            if (Math.hypot(xx - p1[0], xy - p1[1]) < 6 * pitch + 1e-9) {
              V[V.length - 1] = [xx, xy, p1[2], p1[3], p1[4]];
              run.shift();
            }
          } else if (Math.hypot(q1[0] - p1[0], q1[1] - p1[1]) > 1e-6) {
            const jog = Math.hypot(q1[0] - p1[0], q1[1] - p1[1]), ta = Math.min(jog, 0.45 * l1), tb = Math.min(jog, 0.45 * l2);
            V[V.length - 1] = [p1[0] - d1x * ta, p1[1] - d1y * ta, p1[2], p1[3], p1[4]];
            run[0] = [q1[0] + d2x * tb, q1[1] + d2y * tb, q1[2], q1[3], q1[4]];
          } else run.shift();
        }
      }
      V.push(...run);
      i = j;
    }
    const path = filleted(V, R0);
    if (path) lines.push({ colour: c, path });
    V.forEach((v) => {
      if (v[4] < 0 || !ln.stop[v[4]]) return;
      const k = ch[v[4]];
      let s = stopsAt.get(k);
      if (!s) stopsAt.set(k, (s = []));
      s.push([v[0], v[1]]);
    });
  }
  const pills: Drawing["pills"] = [], passages: Drawing["passages"] = [];
  for (const raw of stopsAt.values()) {
    const seen = new Set<string>(), pts: [number, number][] = [];
    for (const p of raw) {
      const key = `${p[0].toFixed(3)},${p[1].toFixed(3)}`;
      if (!seen.has(key)) seen.add(key), pts.push(p);
    }
    const left = new Set(pts.map((_, i) => i)), centres: [number, number][] = [];
    while (left.size) {
      const g0 = [left.values().next().value as number];
      left.delete(g0[0]);
      for (let grew = true; grew; ) {
        grew = false;
        for (const j of [...left]) if (g0.some((i) => Math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]) < PILL_JOIN * pitch)) g0.push(j), left.delete(j), (grew = true);
      }
      const cap = capsule(g0.map((i) => pts[i]));
      pills.push(cap);
      centres.push([(cap[0] + cap[2]) / 2, (cap[1] + cap[3]) / 2]);
    }
    for (let k = 1; k < centres.length; k++) passages.push([centres[0][0], centres[0][1], centres[k][0], centres[k][1]]);
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    x0 = Math.min(x0, lay.x[i]), x1 = Math.max(x1, lay.x[i]), y0 = Math.min(y0, lay.y[i]), y1 = Math.max(y1, lay.y[i]);
  }
  return { pitch, stroke: W, lines, pills, passages, box: [x0, y0, x1, y1] };
}

/** a path through V with each corner rounded, its radius by offset so a bundle's bends are concentric */
function filleted(V: V[], R0: number): Path2D | null {
  const P: V[] = [];
  for (const v of V) if (!P.length || Math.hypot(v[0] - P[P.length - 1][0], v[1] - P[P.length - 1][1]) > 1e-9) P.push(v);
  if (P.length < 2) return null;
  const path = new Path2D();
  path.moveTo(P[0][0], P[0][1]);
  for (let i = 1; i < P.length - 1; i++) {
    let ax = P[i][0] - P[i - 1][0], ay = P[i][1] - P[i - 1][1], bx = P[i + 1][0] - P[i][0], by = P[i + 1][1] - P[i][1];
    const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
    ax /= la, ay /= la, bx /= lb, by /= lb;
    const th = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
    if (Math.abs(th) < Math.PI / 180) continue;
    const r = R0 + P[i][3] + (th > 0 ? -P[i][2] : P[i][2]);
    const t = Math.min(Math.max(0, r) * Math.tan(Math.abs(th) / 2), 0.49 * la, 0.49 * lb);
    path.lineTo(P[i][0] - ax * t, P[i][1] - ay * t);
    path.quadraticCurveTo(P[i][0], P[i][1], P[i][0] + bx * t, P[i][1] + by * t);
  }
  path.lineTo(P[P.length - 1][0], P[P.length - 1][1]);
  return path;
}

/** a pill over points: the ends of their main axis, and how far the rest lie off it */
function capsule(pts: [number, number][]): [number, number, number, number, number] {
  if (pts.length === 1) return [pts[0][0], pts[0][1], pts[0][0], pts[0][1], 0];
  let cx = 0, cy = 0;
  for (const p of pts) (cx += p[0]), (cy += p[1]);
  cx /= pts.length;
  cy /= pts.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const dx = p[0] - cx, dy = p[1] - cy;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), ux = Math.cos(ang), uy = Math.sin(ang);
  let t0 = Infinity, t1 = -Infinity, s0 = Infinity, s1 = -Infinity, sm = 0;
  for (const p of pts) {
    const dx = p[0] - cx, dy = p[1] - cy, t = dx * ux + dy * uy, s = -dx * uy + dy * ux;
    t0 = Math.min(t0, t), t1 = Math.max(t1, t), s0 = Math.min(s0, s), s1 = Math.max(s1, s);
    sm += s;
  }
  sm /= pts.length;
  const ox = -uy * sm, oy = ux * sm;
  return [cx + ux * t0 + ox, cy + uy * t0 + oy, cx + ux * t1 + ox, cy + uy * t1 + oy, pts.length > 2 ? Math.max(sm - s0, s1 - sm) : 0];
}

export interface DiagramTheme { bg: string; ink: string; pill: string }

/** the drawing on `ctx` at `scale` px per layout unit, (cx, cy) in the canvas's middle */
export function paint(ctx: CanvasRenderingContext2D, d: Drawing, w: number, h: number, cx: number, cy: number, scale: number, theme: DiagramTheme): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  ctx.setTransform(scale, 0, 0, -scale, w / 2 - cx * scale, h / 2 + cy * scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // casings first, so neighbours sit a gap apart and bundles read as one ribbon
  ctx.strokeStyle = theme.bg;
  ctx.lineWidth = d.pitch;
  for (const l of d.lines) ctx.stroke(l.path);
  ctx.lineWidth = d.stroke;
  for (const l of d.lines) {
    ctx.strokeStyle = l.colour;
    ctx.stroke(l.path);
  }
  const rp = PILL_R * d.stroke, ol = PILL_OUTLINE * d.stroke;
  const bar = (x0: number, y0: number, x1: number, y1: number, width: number, colour: string) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1 + (x0 === x1 && y0 === y1 ? 1e-6 : 0), y1);
    ctx.stroke();
  };
  for (const [x0, y0, x1, y1] of d.passages) bar(x0, y0, x1, y1, rp * 0.9 * 2 + 2 * ol, theme.ink);
  for (const [x0, y0, x1, y1] of d.passages) bar(x0, y0, x1, y1, rp * 0.9 * 2, theme.pill);
  for (const p of d.pills) bar(p[0], p[1], p[2], p[3], 2 * rp + 2 * ol + 2 * p[4], theme.ink);
  for (const p of d.pills) bar(p[0], p[1], p[2], p[3], 2 * rp + 2 * p[4], theme.pill);
}
