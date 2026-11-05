// T-2488 spike (notes/T-2488.md), not wired into the game: zone borders drawn as lines. The real boundaries between
// zones (and each zone's outer edge) are carried into the finished drawing by T-2168's warp (each place by its nearest
// stations' own maps), cross each rail hop the real boundary crosses at about the same share of the hop, and between
// those crossings are routed on a grid the way a line would be: runs at 0, 45 and 90 degrees, each bend costing, kept
// clear of the lines and pills, near the warped course. The rail drawing is not touched.

import type { DiagramGraph } from "./graph";
import { pillSize, type Drawing } from "./draw";
import { fitOcti, inRings, rounded, segD, stationZones, type DiagramBorders, type Run, type ZoneShape } from "./borders";
import { hyp } from "./fmath";

export interface LineBorderOpts {
  /** the routing grid's cell, pitches */
  cell: number;
  /** clearance kept from casings and pills, pitches; cost rises inside it */
  clear: number;
  /** how far a route may stray from its warped course, pitches */
  corridor: number;
  /** costs: straying (at the corridor's edge, per unit length), nearness (at a casing's edge), a 45 and a 90 degree
   * bend (pitches of length), going through a casing or pill (per unit length) */
  wDev: number;
  wProx: number;
  b45: number;
  b90: number;
  block: number;
  /** a course longer than this many pitches is routed in parts */
  maxGuide: number;
  /** the real boundaries sampled this often, metres */
  densifyM: number;
  /** the warp's offsets past this many pitches grow only as their log */
  reachSoft: number;
  /** corners rounded with this radius, pitches */
  radius: number;
  /** zones with no stations that share a border with one that has, drawn too */
  neighbours: boolean;
  /** and their far edges too (else only the borders they share with zones that have stations) */
  neighbourRings: boolean;
  /** each warped point kept on the side of each station within this many of its mean real hops that it is on in
   * reality, by this many pitches past the pill (0: no such terms) */
  sideHops: number;
  sideGap: number;
  /** straying from the course costs nothing up to this many pitches */
  freeDev: number;
  /** checks: the arcs (real places), pushed here */
  arcsOut?: { pts: number[]; za: number; zb: number }[];
  /** checks: the warped courses as path commands, pushed here */
  debug?: number[];
}
export const LINE_BORDERS: LineBorderOpts = { cell: 1, clear: 2.5, corridor: 12, wDev: 0.5, wProx: 4, b45: 8, b90: 16, block: 200, maxGuide: 100, densifyM: 300, reachSoft: 20, radius: 2, neighbours: true, neighbourRings: false, sideHops: 2, sideGap: 1.5, freeDev: 5 };

export interface LineBorderStats { arcs: number; crossings: number; dropped: number; sections: number; states: number; inCasing: number; noRoute: number; widened: number; outside: number; sideMoves: number; ms: { arcs: number; crossings: number; route: number; total: number } }

const SX = [1, 1, 0, -1, -1, -1, 0, 1], SY = [0, 1, 1, 1, 0, -1, -1, -1];
const R2 = Math.SQRT1_2;
const UX = [1, R2, 0, -R2, -1, -R2, 0, R2], UY = [0, R2, 1, R2, 0, -R2, -1, -R2];
const dirOf = (dx: number, dy: number) => {
  let best = 0, bd = -Infinity;
  for (let k = 0; k < 8; k++) {
    const v = UX[k] * dx + UY[k] * dy;
    if (v > bd) (bd = v), (best = k);
  }
  return best;
};

interface Arc { pts: number[]; za: number; zb: number; j0: number; j1: number; closed: boolean }

export function lineBorders(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, d: Drawing, shapes: ZoneShape[], opt: Partial<LineBorderOpts> = {}): DiagramBorders & { stats: LineBorderStats } {
  const O = { ...LINE_BORDERS, ...opt };
  const t0 = performance.now();
  const n = g.key.length, pitch = d.pitch;
  const zoneAll = stationZones(g, shapes);

  // ---- the arcs: ring edges by the zones either side, chained between the vertices where those change
  const vkey = (vx: number, vy: number) => `${vx}_${vy}`;
  type E = { ka: string; kb: string; ax: number; ay: number; bx: number; by: number; zs: number[] };
  const edges = new Map<string, E>();
  shapes.forEach((s, z) => {
    for (const r of s.rings) {
      const m = r.length;
      for (let i = 0; i < m; i += 2) {
        const j = (i + 2) % m, ax = r[i], ay = r[i + 1], bx = r[j], by = r[j + 1];
        if (ax === bx && ay === by) continue;
        const ka = vkey(ax, ay), kb = vkey(bx, by), k = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
        const e = edges.get(k);
        if (e) { if (!e.zs.includes(z)) e.zs.push(z); }
        else edges.set(k, ka < kb ? { ka, kb, ax, ay, bx, by, zs: [z] } : { ka: kb, kb: ka, ax: bx, ay: by, bx: ax, by: ay, zs: [z] });
      }
    }
  });
  const has = new Uint8Array(shapes.length);
  for (let i = 0; i < n; i++) if (zoneAll[i] >= 0) has[zoneAll[i]] = 1;
  const use = Uint8Array.from(has);
  if (O.neighbours) for (const e of edges.values()) if (e.zs.some((z) => has[z])) for (const z of e.zs) use[z] = 1;
  const drawnOf = new Int32Array(shapes.length).fill(-1), zones: string[] = [];
  for (const pass of [1, 0]) shapes.forEach((s, z) => { if (use[z] && has[z] === pass) (drawnOf[z] = zones.length), zones.push(s.id); });
  const zoneOf = Int32Array.from(zoneAll, (z) => (z < 0 ? -1 : drawnOf[z]));
  const arcEdges: { e: E; za: number; zb: number }[] = [];
  for (const e of edges.values()) {
    const zs = e.zs.filter((z) => use[z]).map((z) => drawnOf[z]).sort((a, b) => a - b);
    if (!zs.length || (!O.neighbourRings && !e.zs.some((z) => has[z]))) continue;
    arcEdges.push({ e, za: zs[0], zb: zs.length > 1 ? zs[1] : -1 });
  }
  const inc = new Map<string, number[]>();
  arcEdges.forEach(({ e }, i) => {
    for (const k of [e.ka, e.kb]) { const l = inc.get(k); if (l) l.push(i); else inc.set(k, [i]); }
  });
  const isJ = (k: string) => {
    const l = inc.get(k)!;
    return l.length !== 2 || arcEdges[l[0]].za !== arcEdges[l[1]].za || arcEdges[l[0]].zb !== arcEdges[l[1]].zb;
  };
  const jIds = new Map<string, number>(), jx: number[] = [], jy: number[] = [];
  const jOf = (k: string, px: number, py: number) => {
    let id = jIds.get(k);
    if (id === undefined) (id = jx.length), jIds.set(k, id), jx.push(px), jy.push(py);
    return id;
  };
  const seen = new Uint8Array(arcEdges.length), arcs: Arc[] = [];
  const walk = (start: string, sx: number, sy: number, first: number, closedRing: boolean) => {
    const pts = [sx, sy];
    let k = start, ei = first;
    for (;;) {
      seen[ei] = 1;
      const { e } = arcEdges[ei];
      const fwd = e.ka === k;
      k = fwd ? e.kb : e.ka;
      pts.push(fwd ? e.bx : e.ax, fwd ? e.by : e.ay);
      if (k === start || (!closedRing && isJ(k))) break;
      const l = inc.get(k)!, nx = l[0] === ei ? l[1] : l[0];
      if (nx === undefined || seen[nx]) break;
      ei = nx;
    }
    const n2 = pts.length;
    arcs.push({ pts, za: arcEdges[first].za, zb: arcEdges[first].zb, j0: jOf(start, sx, sy), j1: jOf(k, pts[n2 - 2], pts[n2 - 1]), closed: k === start });
  };
  for (const [k, l] of inc) if (isJ(k)) for (const ei of l) if (!seen[ei]) {
    const { e } = arcEdges[ei], here = e.ka === k;
    walk(k, here ? e.ax : e.bx, here ? e.ay : e.by, ei, false);
  }
  arcEdges.forEach(({ e }, ei) => { if (!seen[ei]) walk(e.ka, e.ax, e.ay, ei, true); });
  const t1 = performance.now();
  if (O.arcsOut) {
    O.arcsOut.push(...arcs);
    return { cmds: new Float64Array(0), zones, zoneOf, runs: [], stats: null as unknown as LineBorderStats };
  }

  // ---- the warp, and the drawing's obstacles (casings and pills) by squares of OB
  // a station outside every outline (in the zone nearest it) has its zone's edge drawn past it, not short of it
  const outside = Uint8Array.from(zoneAll, (z, i) => (z >= 0 && !inRings(shapes[z].rings, g.x[i], g.y[i]) ? 1 : 0));
  const warp = makeWarp(g, x, y, O.reachSoft * pitch, O.sideHops, (1 + O.sideGap) * pitch, outside);
  const segs: number[] = [];
  for (const r of d.ribbons) for (let k = 0; k + 3 < r.length; k += 2) segs.push(r[k], r[k + 1], r[k + 2], r[k + 3], pitch / 2);
  d.pills.forEach((p, q) => {
    const [r, o] = pillSize(d, d.pillOf[q]);
    segs.push(p[0], p[1], p[2], p[3], r + o + p[4]);
  });
  const OB = 8 * pitch, C = O.clear * pitch, obs = new Map<number, number[]>();
  const okey = (i: number, j: number) => (i + 32768) * 65536 + (j + 32768);
  for (let s = 0; s < segs.length; s += 5) {
    const R = segs[s + 4] + C;
    const i0 = Math.floor((Math.min(segs[s], segs[s + 2]) - R) / OB), i1 = Math.floor((Math.max(segs[s], segs[s + 2]) + R) / OB);
    const j0 = Math.floor((Math.min(segs[s + 1], segs[s + 3]) - R) / OB), j1 = Math.floor((Math.max(segs[s + 1], segs[s + 3]) + R) / OB);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const k = okey(i, j), l = obs.get(k);
      if (l) l.push(s);
      else obs.set(k, [s]);
    }
  }
  /** distance from (px, py) past the nearest casing or pill, up to C (C when none is nearer) */
  const clearAt = (px: number, py: number) => {
    let best = C;
    for (const s of obs.get(okey(Math.floor(px / OB), Math.floor(py / OB))) ?? []) best = Math.min(best, segD(px, py, segs[s], segs[s + 1], segs[s + 2], segs[s + 3]) - segs[s + 4]);
    return best;
  };
  /** a point moved off the lines: the nearest within 8 pitches with the full clearance, else the clearest */
  const free = (px: number, py: number): [number, number] => {
    if (clearAt(px, py) >= C) return [px, py];
    let best: [number, number] = [px, py], bc = clearAt(px, py), bd = Infinity;
    const st = 0.5 * pitch;
    for (let i = -16; i <= 16; i++) for (let j = -16; j <= 16; j++) {
      const qx = px + i * st, qy = py + j * st, c = clearAt(qx, qy), dd = i * i + j * j;
      if (c >= C ? bc < C || dd < bd : bc < C && c > bc) (best = [qx, qy]), (bc = c), (bd = dd);
    }
    return best;
  };

  // ---- the crossings: each rail hop a real boundary crosses an odd number of times, crossed once at the median
  const t2s = performance.now();
  const m = g.ea.length;
  let rmed = 0;
  {
    const ls = new Float64Array(m);
    for (let e = 0; e < m; e++) ls[e] = hyp(g.x[g.eb[e]] - g.x[g.ea[e]], g.y[g.eb[e]] - g.y[g.ea[e]]);
    ls.sort();
    rmed = ls[m >> 1] || 1000;
  }
  const HC = Math.max(2000, 4 * rmed), hops = new Map<number, number[]>();
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e];
    for (let i = Math.floor(Math.min(g.x[a], g.x[b]) / HC); i <= Math.floor(Math.max(g.x[a], g.x[b]) / HC); i++)
      for (let j = Math.floor(Math.min(g.y[a], g.y[b]) / HC); j <= Math.floor(Math.max(g.y[a], g.y[b]) / HC); j++) {
        const k = okey(i, j), l = hops.get(k);
        if (l) l.push(e);
        else hops.set(k, [e]);
      }
  }
  // per graph edge, its drawn stretches: per line on it, the ribbon's points between its two nodes, a to b
  const stretches = new Map<number, number[][]>();
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < m; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
  d.ribbons.forEach((r, li) => {
    const chain = g.lines[d.lineOf[li]].chain, at = d.chainAt[li];
    let p0 = 0;
    for (let p = 1; p <= at.length; p++) {
      if (p < at.length && at[p] === at[p0]) continue;
      const ci = at[p0];
      if (ci + 1 < chain.length) {
        const u = chain[ci], w = chain[ci + 1], e = edgeOf.get(Math.min(u, w) * n + Math.max(u, w));
        if (e !== undefined) {
          const pts: number[] = [];
          for (let q = p0; q <= Math.min(p, at.length - 1); q++) pts.push(r[q * 2], r[q * 2 + 1]);
          if (u !== g.ea[e]) {
            const rev: number[] = [];
            for (let q = pts.length - 2; q >= 0; q -= 2) rev.push(pts[q], pts[q + 1]);
            pts.splice(0, pts.length, ...rev);
          }
          const l = stretches.get(e);
          if (l) l.push(pts);
          else stretches.set(e, [pts]);
        }
      }
      p0 = p;
    }
  });
  type Cross = { s: number; e: number; t: number; sign: number };
  const arcCross: Cross[][] = [];
  let kept = 0, dropped = 0;
  for (const A of arcs) {
    const P = A.pts, found = new Map<number, Cross[]>();
    let s = 0;
    for (let k = 0; k + 3 < P.length; k += 2) {
      const ax = P[k], ay = P[k + 1], bx = P[k + 2], by = P[k + 3], sl = hyp(bx - ax, by - ay);
      const cand = new Set<number>();
      for (let i = Math.floor(Math.min(ax, bx) / HC); i <= Math.floor(Math.max(ax, bx) / HC); i++)
        for (let j = Math.floor(Math.min(ay, by) / HC); j <= Math.floor(Math.max(ay, by) / HC); j++) for (const e of hops.get(okey(i, j)) ?? []) cand.add(e);
      for (const e of cand) {
        const cx = g.x[g.ea[e]], cy = g.y[g.ea[e]], dx = g.x[g.eb[e]] - cx, dy = g.y[g.eb[e]] - cy;
        const ux = bx - ax, uy = by - ay, den = ux * dy - uy * dx;
        if (Math.abs(den) < 1e-12) continue;
        // a + u r = c + v t
        const r = ((cx - ax) * dy - (cy - ay) * dx) / den, t = ((cx - ax) * uy - (cy - ay) * ux) / den;
        if (r < 0 || r >= 1 || t <= 0 || t >= 1) continue;
        const c: Cross = { s: s + r * sl, e, t, sign: dx * uy - dy * ux > 0 ? 1 : -1 };
        const l = found.get(e);
        if (l) l.push(c);
        else found.set(e, [c]);
      }
      s += sl;
    }
    const list: Cross[] = [];
    for (const l of found.values()) {
      if (l.length % 2 === 0 || !stretches.has(l[0].e)) {
        dropped += l.length;
        continue;
      }
      l.sort((p, q) => p.s - q.s);
      list.push(l[l.length >> 1]);
      kept++;
      dropped += l.length - 1;
    }
    list.sort((p, q) => p.s - q.s);
    arcCross.push(list);
  }
  const t2 = performance.now();

  // ---- each crossing drawn: a straight run across the hop's bundle, square to it, from the side the boundary comes
  // from in reality to the other, its ends the clearance off the outermost casing
  type Drawn = { ex: number; ey: number; xx: number; xy: number; dir: number };
  const drawnCross = (c: Cross): Drawn | null => {
    const st = stretches.get(c.e)!, P = st[0];
    let L = 0;
    const cum = [0];
    for (let k = 2; k < P.length; k += 2) cum.push((L += hyp(P[k] - P[k - 2], P[k + 1] - P[k - 1])));
    if (L <= 0) return null;
    const want = Math.max(0.25, Math.min(0.75, c.t)) * L;
    let k = 1;
    while (k < cum.length - 1 && cum[k] < want) k++;
    const f = (want - cum[k - 1]) / Math.max(1e-12, cum[k] - cum[k - 1]);
    const px = P[k * 2 - 2] + (P[k * 2] - P[k * 2 - 2]) * f, py = P[k * 2 - 1] + (P[k * 2 + 1] - P[k * 2 - 1]) * f;
    const du = dirOf(P[k * 2] - P[k * 2 - 2], P[k * 2 + 1] - P[k * 2 - 1]), ux = UX[du], uy = UY[du], nx = -uy, ny = ux;
    let lo = 0, hi = 0;
    for (const Q of st) for (let q = 0; q < Q.length; q += 2) {
      const al = (Q[q] - px) * ux + (Q[q + 1] - py) * uy;
      if (Math.abs(al) > 3 * pitch) continue;
      const o = (Q[q] - px) * nx + (Q[q + 1] - py) * ny;
      (lo = Math.min(lo, o)), (hi = Math.max(hi, o));
    }
    const rl = hi + pitch / 2 + C, rr = -lo + pitch / 2 + C;
    // sign > 0: the boundary goes from the hop's right (a to b) to its left
    const [ex, ey, xx, xy] = c.sign > 0 ? [px - nx * rr, py - ny * rr, px + nx * rl, py + ny * rl] : [px + nx * rl, py + ny * rl, px - nx * rr, py - ny * rr];
    return { ex, ey, xx, xy, dir: dirOf(xx - ex, xy - ey) };
  };

  // ---- the junctions, warped and moved off the lines
  const wv = new Float64Array(5);
  /** A course through real places `rr` (x, y pairs, its ends at (fx, fy) and (tx, ty) as drawn) laid out as a chain:
   * each step its real offset through the local map of the stations near it, each place pulled weakly toward its
   * warped one (W_ANCHOR), and those on the wrong side of a station near them pulled across (W_SIDE), in rounds.
   * Least squares on a chain: one tridiagonal solve per axis. */
  const W_ANCHOR = 0.02, W_SIDE = 1;
  const chainFit = (rr: number[], fx: number, fy: number, tx: number, ty: number): number[] => {
    const m = rr.length / 2;
    if (m <= 2) return [fx, fy, tx, ty];
    const vx = new Float64Array(m - 1), vy = new Float64Array(m - 1), axv = new Float64Array(m), ayv = new Float64Array(m);
    const M = new Float64Array(4), o = new Float64Array(5);
    for (let k = 0; k < m; k++) warp.at(rr[k * 2], rr[k * 2 + 1], o), (axv[k] = o[0]), (ayv[k] = o[1]);
    for (let k = 0; k + 1 < m; k++) {
      warp.mapAt((rr[k * 2] + rr[k * 2 + 2]) / 2, (rr[k * 2 + 1] + rr[k * 2 + 3]) / 2, M);
      const dx = rr[k * 2 + 2] - rr[k * 2], dy = rr[k * 2 + 3] - rr[k * 2 + 1];
      (vx[k] = M[0] * dx + M[1] * dy), (vy[k] = M[2] * dx + M[3] * dy);
    }
    const px = new Float64Array(m), py = new Float64Array(m), ws = new Float64Array(m), tsx = new Float64Array(m), tsy = new Float64Array(m);
    (px[0] = fx), (py[0] = fy), (px[m - 1] = tx), (py[m - 1] = ty);
    const cp = new Float64Array(m), dp = new Float64Array(m);
    const solve = (p: Float64Array, v: Float64Array, a: Float64Array, t: Float64Array) => {
      // unknowns 1 .. m - 2: -p[k-1] + (2 + wa + ws) p[k] - p[k+1] = v[k-1] - v[k] + wa a[k] + ws t[k]
      for (let k = 1; k <= m - 2; k++) {
        const b = 2 + W_ANCHOR + ws[k];
        let d = v[k - 1] - v[k] + W_ANCHOR * a[k] + ws[k] * t[k];
        if (k === 1) d += p[0];
        if (k === m - 2) d += p[m - 1];
        const den = b + (k > 1 ? cp[k - 1] : 0);
        cp[k] = -1 / den;
        dp[k] = (d + (k > 1 ? dp[k - 1] : 0)) / den;
      }
      for (let k = m - 2; k >= 1; k--) p[k] = dp[k] - (k < m - 2 ? cp[k] * p[k + 1] : 0);
    };
    for (let round = 0; round < 4; round++) {
      solve(px, vx, axv, tsx);
      solve(py, vy, ayv, tsy);
      if (!O.sideHops) break;
      let any = false;
      for (let k = 1; k < m - 1; k++) {
        (o[0] = px[k]), (o[1] = py[k]);
        warp.side(rr[k * 2], rr[k * 2 + 1], o);
        if (hyp(o[0] - px[k], o[1] - py[k]) > 1e-6 * pitch) (ws[k] = W_SIDE), (tsx[k] = o[0]), (tsy[k] = o[1]), (any = true);
      }
      if (!any) break;
    }
    const out: number[] = [];
    for (let k = 0; k < m; k++) out.push(px[k], py[k]);
    return out;
  };
  const warped = (px: number, py: number): [number, number] => (warp.at(px, py, wv), warp.side(px, py, wv), [wv[0], wv[1]]);
  const JX = new Float64Array(jx.length), JY = new Float64Array(jx.length);
  for (let i = 0; i < jx.length; i++) [JX[i], JY[i]] = free(...warped(jx[i], jy[i]));

  // ---- routing
  const h = O.cell * pitch, Rc = O.corridor * pitch;
  const stats: LineBorderStats = { arcs: arcs.length, crossings: kept, dropped, sections: 0, states: 0, inCasing: 0, noRoute: 0, widened: 0, outside: outside.reduce((a, b) => a + b, 0), sideMoves: 0, ms: { arcs: t1 - t0, crossings: t2 - t2s, route: 0, total: 0 } };
  const t3 = performance.now();
  /** a route from (sx, sy) to (ex, ey) near `guide` (x, y pairs, from the start to the end); dS the way it must
   * leave by, dE the way it must arrive by (-1 for any). The lines and pills are not crossed: where no route
   * clear of them is found near the course, the search widens, and only then crosses them at a cost. Vertices, x, y
   * pairs. */
  const routeOne = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[]): number[] => {
    stats.sections++;
    for (const [k, cross] of [[1, false], [3, false], [3, true]] as const) {
      const r = attempt(sx, sy, dS, ex, ey, dE, guide, k * Rc, cross);
      if (r) return r;
      stats.widened++;
    }
    stats.noRoute++;
    return [sx, sy, ex, ey];
  };
  const attempt = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[], M: number, cross: boolean): number[] | null => {
    let gx0 = Infinity, gy0 = Infinity, gx1 = -Infinity, gy1 = -Infinity;
    for (let k = 0; k < guide.length; k += 2) (gx0 = Math.min(gx0, guide[k])), (gx1 = Math.max(gx1, guide[k])), (gy0 = Math.min(gy0, guide[k + 1])), (gy1 = Math.max(gy1, guide[k + 1]));
    const i0 = Math.floor((gx0 - M - sx) / h) - 1, i1 = Math.ceil((gx1 + M - sx) / h) + 1, j0 = Math.floor((gy0 - M - sy) / h) - 1, j1 = Math.ceil((gy1 + M - sy) / h) + 1;
    const W = i1 - i0 + 1, H = j1 - j0 + 1, NC = W * H;
    // distance to the course: its cells 0, then a chamfer transform (steps 1 and the root of 2, in cells)
    const gd = new Float32Array(NC).fill(1e9);
    for (let k = 0; k + 3 < guide.length; k += 2) {
      const ax = (guide[k] - sx) / h - i0, ay = (guide[k + 1] - sy) / h - j0, bx = (guide[k + 2] - sx) / h - i0, by = (guide[k + 3] - sy) / h - j0;
      const parts = Math.max(1, Math.ceil(2 * hyp(bx - ax, by - ay)));
      for (let q = 0; q <= parts; q++) {
        const ci = Math.round(ax + ((bx - ax) * q) / parts), cj = Math.round(ay + ((by - ay) * q) / parts);
        if (ci >= 0 && cj >= 0 && ci < W && cj < H) gd[cj * W + ci] = 0;
      }
    }
    for (let cj = 0; cj < H; cj++) for (let ci = 0; ci < W; ci++) {
      const c = cj * W + ci;
      let v = gd[c];
      if (ci > 0) v = Math.min(v, gd[c - 1] + 1);
      if (cj > 0) {
        v = Math.min(v, gd[c - W] + 1);
        if (ci > 0) v = Math.min(v, gd[c - W - 1] + Math.SQRT2);
        if (ci < W - 1) v = Math.min(v, gd[c - W + 1] + Math.SQRT2);
      }
      gd[c] = v;
    }
    for (let cj = H - 1; cj >= 0; cj--) for (let ci = W - 1; ci >= 0; ci--) {
      const c = cj * W + ci;
      let v = gd[c];
      if (ci < W - 1) v = Math.min(v, gd[c + 1] + 1);
      if (cj < H - 1) {
        v = Math.min(v, gd[c + W] + 1);
        if (ci < W - 1) v = Math.min(v, gd[c + W + 1] + Math.SQRT2);
        if (ci > 0) v = Math.min(v, gd[c + W - 1] + Math.SQRT2);
      }
      gd[c] = v;
    }
    const goalI = Math.round((ex - sx) / h) - i0, goalJ = Math.round((ey - sy) / h) - j0, goal = goalJ * W + goalI, start = -j0 * W - i0;
    const clr = new Float32Array(NC).fill(NaN);
    const step = new Float64Array(NC);
    const free0 = O.freeDev * pitch;
    const cost = (c: number) => {
      if (clr[c] !== clr[c]) {
        const od = clearAt(sx + ((c % W) + i0) * h, sy + (Math.floor(c / W) + j0) * h);
        clr[c] = od;
        const dv = Math.max(0, gd[c] * h - free0) / (Rc - free0), pr = Math.max(0, 1 - od / C);
        step[c] = od < 0 && c !== goal && c !== start ? (cross ? 1 + O.block : Infinity) : 1 + O.wDev * dv * dv + O.wProx * pr * pr;
      }
      return step[c];
    };
    const NS = NC * 8, dist = new Float64Array(NS).fill(Infinity), from = new Int32Array(NS).fill(-1);
    const heap = new Heap();
    const gxw = sx + (goalI + i0) * h, gyw = sy + (goalJ + j0) * h;
    const hOf = (c: number) => hyp(sx + ((c % W) + i0) * h - gxw, sy + (Math.floor(c / W) + j0) * h - gyw);
    const bendC = [0, O.b45 * pitch, O.b90 * pitch];
    for (let k = 0; k < 8; k++) if (dS < 0 || k === dS) (dist[start * 8 + k] = 0), heap.push(hOf(start), start * 8 + k);
    let best = Infinity, bestS = -1;
    while (heap.size) {
      const [f, st] = heap.pop();
      if (f >= best) break;
      const c = st >> 3, dk = st & 7, dc = dist[st];
      if (f - hOf(c) > dc + 1e-9) continue;
      stats.states++;
      if (c === goal) {
        const tn = dE < 0 ? 0 : Math.min(Math.abs(dk - dE), 8 - Math.abs(dk - dE));
        const tot = dc + (tn < 3 ? bendC[tn] : Infinity);
        if (tot < best) (best = tot), (bestS = st);
      }
      const ci = c % W, cj = (c - ci) / W;
      for (let k = 0; k < 8; k++) {
        const tn = Math.min(Math.abs(k - dk), 8 - Math.abs(k - dk));
        if (tn >= 3) continue;
        const ni = ci + SX[k], nj = cj + SY[k];
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        const nc = nj * W + ni, sc = cost(nc);
        if (sc === Infinity) continue;
        const nd = dc + (k & 1 ? h * Math.SQRT2 : h) * sc + bendC[tn], ns = nc * 8 + k;
        if (nd < dist[ns]) (dist[ns] = nd), (from[ns] = st), heap.push(nd + hOf(nc), ns);
      }
    }
    if (bestS < 0) return null;
    const cells: number[] = [];
    for (let st = bestS; st >= 0; st = from[st]) cells.push(st >> 3);
    cells.reverse();
    const px: number[] = [], py: number[] = [];
    let last = -1;
    for (const c of cells) {
      if (c === last) continue;
      last = c;
      if (clr[c] < 0 && c !== goal && c !== start) stats.inCasing++;
      px.push(sx + ((c % W) + i0) * h), py.push(sy + (Math.floor(c / W) + j0) * h);
    }
    (px[0] = sx), (py[0] = sy);
    if (px.length < 2) px.push(ex), py.push(ey);
    else (px[px.length - 1] = ex), (py[py.length - 1] = ey);
    const tol = px.map(() => 0.6 * pitch);
    return [...fitOcti(px, py, tol)];
  };

  const runs: Run[] = [], cmds: number[] = [];
  const dens = O.densifyM;
  arcs.forEach((A, ai) => {
    // the arc sampled, with each sample's distance along it in reality
    const P = A.pts, rs: number[] = [], ss: number[] = [];
    let s = 0;
    for (let k = 0; k + 3 < P.length; k += 2) {
      const sl = hyp(P[k + 2] - P[k], P[k + 3] - P[k + 1]), parts = Math.max(1, Math.ceil(sl / dens));
      for (let q = k ? 1 : 0; q <= parts; q++) {
        const f = q / parts;
        rs.push(P[k] + (P[k + 2] - P[k]) * f, P[k + 1] + (P[k + 3] - P[k + 1]) * f), ss.push(s + sl * f);
      }
      s += sl;
    }
    /** the real place at distance `at` along the arc */
    const realAt = (at: number): [number, number] => {
      let q = 1;
      while (q < ss.length - 1 && ss[q] < at) q++;
      const f = Math.max(0, Math.min(1, (at - ss[q - 1]) / Math.max(1e-9, ss[q] - ss[q - 1])));
      return [rs[q * 2 - 2] + (rs[q * 2] - rs[q * 2 - 2]) * f, rs[q * 2 - 1] + (rs[q * 2 + 1] - rs[q * 2 - 1]) * f];
    };
    const cr = arcCross[ai].map((c) => [c, drawnCross(c)] as const).filter((q) => q[1]) as [Cross, Drawn][];
    // stops: the start, each crossing's two ends, the end
    const out: number[] = [JX[A.j0], JY[A.j0]];
    let fromS = -Infinity, fx = JX[A.j0], fy = JY[A.j0], fd = -1;
    const leg = (toS: number, tx: number, ty: number, td: number) => {
      // the course: the real samples between, laid out as a chain from (fx, fy) to (tx, ty) (the layout's stage A on
      // the border alone, the stations held where they are drawn)
      const rr: number[] = [...(fromS === -Infinity ? [rs[0], rs[1]] : realAt(fromS))];
      for (let q = 0; q < ss.length; q++) if (ss[q] > fromS && ss[q] < toS) rr.push(rs[q * 2], rs[q * 2 + 1]);
      rr.push(...(toS === Infinity ? [rs[rs.length - 2], rs[rs.length - 1]] : realAt(toS)));
      const guide = simplify(chainFit(rr, fx, fy, tx, ty), 2 * pitch);
      if (O.debug) {
        O.debug.push(0, guide[0], guide[1]);
        for (let q = 2; q < guide.length; q += 2) O.debug.push(1, guide[q], guide[q + 1]);
      }
      // long courses in parts, through waypoints moved off the lines
      let gl = 0, cut = [0];
      for (let k = 2; k < guide.length; k += 2) {
        gl += hyp(guide[k] - guide[k - 2], guide[k + 1] - guide[k - 1]);
        if (gl > O.maxGuide * pitch && k < guide.length - 2) cut.push(k), (gl = 0);
      }
      cut.push(guide.length - 2);
      let ax = fx, ay = fy, ad = fd;
      for (let c = 1; c < cut.length; c++) {
        const last = c === cut.length - 1;
        const [bx, by] = last ? [tx, ty] : free(guide[cut[c]], guide[cut[c] + 1]);
        const sub = [ax, ay, ...guide.slice(cut[c - 1] + 2, cut[c]), bx, by];
        const r = routeOne(ax, ay, ad, bx, by, last ? td : -1, sub);
        for (let k = 2; k < r.length; k++) out.push(r[k]);
        (ax = bx), (ay = by), (ad = -1);
      }
    };
    for (const [c, dc] of cr) {
      leg(c.s, dc.ex, dc.ey, dc.dir);
      out.push(dc.xx, dc.xy);
      (fromS = c.s), (fx = dc.xx), (fy = dc.xy), (fd = dc.dir);
    }
    leg(Infinity, JX[A.j1], JY[A.j1], -1);
    const v = Float64Array.from(out), c0 = cmds.length;
    rounded(v, A.closed, O.radius * pitch, cmds);
    runs.push({ p: v, left: A.za, right: A.zb, closed: A.closed, c: [c0, cmds.length] });
  });
  stats.ms.route = performance.now() - t3;
  stats.sideMoves = warpMoves;
  warpMoves = 0;
  stats.ms.total = performance.now() - t0;
  return { cmds: Float64Array.from(cmds), zones, zoneOf, runs, stats };
}

/** a binary heap of (key, value), least key first */
class Heap {
  k: number[] = [];
  v: number[] = [];
  get size() { return this.k.length; }
  push(key: number, val: number) {
    const k = this.k, v = this.v;
    let i = k.length;
    k.push(key), v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      (k[i] = k[p]), (v[i] = v[p]), (i = p);
    }
    (k[i] = key), (v[i] = val);
  }
  pop(): [number, number] {
    const k = this.k, v = this.v, top: [number, number] = [k[0], v[0]], lk = k.pop()!, lv = v.pop()!;
    if (k.length) {
      let i = 0;
      for (;;) {
        const a = 2 * i + 1, b = a + 1;
        let c = i, ck = lk;
        if (a < k.length && k[a] < ck) (c = a), (ck = k[a]);
        if (b < k.length && k[b] < ck) c = b;
        if (c === i) break;
        (k[i] = k[c]), (v[i] = v[c]), (i = c);
      }
      (k[i] = lk), (v[i] = lv);
    }
    return top;
  }
}

// ---- T-2168's warp (in git history before T-2464), with the offsets past `reach` growing as their log rather than cut

/** the nearest stations blended, and a station's weight 1 / (d^2 + EPS^2)^2, metres; a blended map's scale at most
 * MAG_MAX times the stations' median */
const K = 8, EPS_M = 50, MAG_MAX = 2;

/** a polyline (x, y pairs) with the points within `tol` of the way between their kept neighbours dropped (Douglas and
 * Peucker), its ends kept */
function simplify(p: number[], tol: number): number[] {
  const m = p.length / 2;
  if (m <= 2) return p;
  const keep = new Uint8Array(m);
  (keep[0] = 1), (keep[m - 1] = 1);
  const stack = [0, m - 1];
  while (stack.length) {
    const b = stack.pop()!, a = stack.pop()!;
    let worst = -1, wi = -1;
    for (let i = a + 1; i < b; i++) {
      const dd = segD(p[i * 2], p[i * 2 + 1], p[a * 2], p[a * 2 + 1], p[b * 2], p[b * 2 + 1]);
      if (dd > worst) (worst = dd), (wi = i);
    }
    if (worst > tol) (keep[wi] = 1), stack.push(a, wi, wi, b);
  }
  const out: number[] = [];
  for (let i = 0; i < m; i++) if (keep[i]) out.push(p[i * 2], p[i * 2 + 1]);
  return out;
}

let warpMoves = 0;
function makeWarp(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, reach: number, sideHops: number, gap: number, flip: Uint8Array) {
  const J = localMaps(g, x, y), all = kdTree(g.x, g.y);
  const scales = new Float64Array(g.key.length);
  for (let i = 0; i < g.key.length; i++) scales[i] = Math.sqrt(Math.abs(J[i * 4] * J[i * 4 + 3] - J[i * 4 + 1] * J[i * 4 + 2]));
  const medScale = scales.sort()[g.key.length >> 1] || 1;
  const near = new Int32Array(K), nd = new Float64Array(K);
  // each station's reach for the side terms: sideHops of its mean real hop
  const n = g.key.length, hopR = new Float64Array(n), cnt = new Float64Array(n);
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e], b = g.eb[e], l = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a]);
    (hopR[a] += l), (hopR[b] += l), cnt[a]++, cnt[b]++;
  }
  for (let i = 0; i < n; i++) hopR[i] = cnt[i] ? (sideHops * hopR[i]) / cnt[i] : 0;
  const SK = 16, sn = new Int32Array(SK), sd = new Float64Array(SK);
  return {
    /** the warped place `out` [x, y] of real (sx, sy) moved, by sweeps of projections, to the side of each station
     * near it in reality that it is on there (along the station's own map of their offset) */
    side(sx: number, sy: number, out: Float64Array) {
      if (!sideHops) return;
      const found = all.nearest(sx, sy, SK, sn, sd);
      for (let sweep = 0; sweep < 8; sweep++) {
        let moved = 0;
        for (let q = 0; q < found; q++) {
          const i = sn[q];
          if (sd[q] > hopR[i] * hopR[i]) continue;
          const ux = sx - g.x[i], uy = sy - g.y[i];
          const ox = J[i * 4] * ux + J[i * 4 + 1] * uy, oy = J[i * 4 + 2] * ux + J[i * 4 + 3] * uy, ol = Math.sqrt(ox * ox + oy * oy);
          if (ol < 1e-9) continue;
          const f = flip[i] ? -1 : 1, nx = (f * ox) / ol, ny = (f * oy) / ol, cur = (out[0] - x[i]) * nx + (out[1] - y[i]) * ny, want = flip[i] ? gap : Math.min(ol, gap);
          if (cur < want) (out[0] += (want - cur) * nx), (out[1] += (want - cur) * ny), (moved = 1), warpMoves++;
        }
        if (!moved) break;
      }
    },

    /** the blend of the nearest stations' own maps at real (sx, sy): a, b, d, e of X = a x + b y, Y = d x + e y */
    mapAt(sx: number, sy: number, M: Float64Array) {
      const found = all.nearest(sx, sy, K, near, nd);
      let sw = 0;
      M.fill(0);
      for (let q = 0; q < found; q++) {
        const i = near[q], d2 = nd[q] + EPS_M * EPS_M, w = 1 / (d2 * d2);
        for (let c = 0; c < 4; c++) M[c] += w * J[i * 4 + c];
        sw += w;
      }
      if (sw > 0) for (let c = 0; c < 4; c++) M[c] /= sw;
      // a dense core's magnification would blow up the noise of a coarse outline passing it: at most MAG_MAX of the median
      const sc = Math.sqrt(Math.abs(M[0] * M[3] - M[1] * M[2]));
      if (sc > MAG_MAX * medScale) for (let c = 0; c < 4; c++) M[c] *= (MAG_MAX * medScale) / sc;
    },
    at(sx: number, sy: number, out: Float64Array) {
      const found = all.nearest(sx, sy, K, near, nd);
      let sw = 0, ax = 0, ay = 0;
      for (let q = 0; q < found; q++) {
        const i = near[q], d2 = nd[q] + EPS_M * EPS_M, w = 1 / (d2 * d2), ux = sx - g.x[i], uy = sy - g.y[i];
        let ox = J[i * 4] * ux + J[i * 4 + 1] * uy, oy = J[i * 4 + 2] * ux + J[i * 4 + 3] * uy;
        const ol = Math.sqrt(ox * ox + oy * oy);
        if (ol > reach) {
          const f = (reach * (1 + Math.log(ol / reach))) / ol;
          (ox *= f), (oy *= f);
        }
        (ax += w * (x[i] + ox)), (ay += w * (y[i] + oy)), (sw += w);
      }
      out[0] = found ? ax / sw : sx;
      out[1] = found ? ay / sw : sy;
    },
  };
}

function localMaps(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>): Float64Array {
  const n = g.key.length, J = new Float64Array(n * 4);
  const nbr: number[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < g.ea.length; e++) nbr[g.ea[e]].push(g.eb[e]), nbr[g.eb[e]].push(g.ea[e]);
  for (let i = 0; i < n; i++) {
    let rxx = 0, rxy = 0, ryy = 0, dxx = 0, dxy = 0, dyx = 0, dyy = 0, sr = 0, sd = 0;
    for (const j of nbr[i]) {
      const rx = g.x[j] - g.x[i], ry = g.y[j] - g.y[i], qx = x[j] - x[i], qy = y[j] - y[i];
      (rxx += rx * rx), (rxy += rx * ry), (ryy += ry * ry);
      (dxx += qx * rx), (dxy += qx * ry), (dyx += qy * rx), (dyy += qy * ry);
      (sr += Math.sqrt(rx * rx + ry * ry)), (sd += Math.sqrt(qx * qx + qy * qy));
    }
    const s = sr > 0 ? sd / sr : 0.005, mu = nbr[i].length ? (rxx + ryy) / nbr[i].length : 1;
    const m00 = rxx + mu, m01 = rxy, m11 = ryy + mu, det = m00 * m11 - m01 * m01;
    const n00 = dxx + mu * s, n01 = dxy, n10 = dyx, n11 = dyy + mu * s;
    const i00 = m11 / det, i01 = -m01 / det, i11 = m00 / det;
    J[i * 4] = n00 * i00 + n01 * i01;
    J[i * 4 + 1] = n00 * i01 + n01 * i11;
    J[i * 4 + 2] = n10 * i00 + n11 * i01;
    J[i * 4 + 3] = n10 * i01 + n11 * i11;
  }
  return J;
}

function kdTree(xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const n = xs.length, ord = Int32Array.from({ length: n }, (_, i) => i);
  const select = (lo: number, hi: number, k: number, c: ArrayLike<number>) => {
    while (hi > lo) {
      const p = c[ord[(lo + hi) >> 1]];
      let i = lo, j = hi;
      while (i <= j) {
        while (c[ord[i]] < p) i++;
        while (c[ord[j]] > p) j--;
        if (i <= j) {
          const t = ord[i];
          (ord[i] = ord[j]), (ord[j] = t), i++, j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return;
    }
  };
  const build = (lo: number, hi: number, depth: number) => {
    if (hi - lo <= 1) return;
    const mid = (lo + hi) >> 1;
    select(lo, hi - 1, mid, depth & 1 ? ys : xs);
    build(lo, mid, depth + 1);
    build(mid + 1, hi, depth + 1);
  };
  build(0, n, 0);
  return {
    nearest(px: number, py: number, k: number, idx: Int32Array, d2: Float64Array): number {
      let found = 0;
      const visit = (lo: number, hi: number, depth: number) => {
        if (hi <= lo) return;
        const mid = (lo + hi) >> 1, i = ord[mid];
        const dx = xs[i] - px, dy = ys[i] - py, dd = dx * dx + dy * dy;
        if (found < k || dd < d2[k - 1]) {
          let s = found < k ? found++ : k - 1;
          while (s > 0 && d2[s - 1] > dd) (d2[s] = d2[s - 1]), (idx[s] = idx[s - 1]), s--;
          (d2[s] = dd), (idx[s] = i);
        }
        const off = depth & 1 ? -dy : -dx;
        if (off < 0) {
          visit(lo, mid, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(mid + 1, hi, depth + 1);
        } else {
          visit(mid + 1, hi, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(lo, mid, depth + 1);
        }
      };
      visit(0, n, 0);
      return found;
    },
  };
}
