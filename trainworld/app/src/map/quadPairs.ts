// T-3693: quad track is two pairs of double track side by side (the sim's QUAD_SPACING, 8 m between their centres),
// stored as two edges with nothing joining them. This finds each double edge's other pair by its drawn points:
// another double edge running alongside, about that far off, at its middle. The track hover's tag says "Quad" there
// (T-3714, quadTwin.ts).

import { kAtY } from "./geo";

const QUAD_M = 8;
/** how far off the spacing a twin's centre line may be, m, and how far from parallel (cosine) */
const SLACK_M = 3, PARALLEL = 0.95;
/** the grid's cell and the step segments are registered at, m */
const CELL_M = 50, STEP_M = 25;

export interface PairEdge { id: number; tracks: number; crossover?: boolean; built: boolean }

/** Each double edge's other pair (edge id to edge id), from edges' points (local units, x, y, z triples). */
export function quadPairs(edges: readonly PairEdge[], ptsOf: (id: number) => ArrayLike<number> | undefined): Map<number, number> {
  const out = new Map<number, number>();
  const cand = edges.filter((e) => e.tracks === 2 && !e.crossover);
  if (cand.length < 2) return out;
  const first = ptsOf(cand[0].id);
  const k = kAtY(first && first.length >= 2 ? first[1] : 0), cell = CELL_M * k, step = STEP_M * k;
  const grid = new Map<string, number[]>();
  /** segments: edge index, start offset into its points */
  const segE: number[] = [], segK: number[] = [];
  const key = (x: number, y: number) => `${Math.floor(x / cell)},${Math.floor(y / cell)}`;
  cand.forEach((e, ei) => {
    const p = ptsOf(e.id);
    if (!p) return;
    for (let j = 0; j + 5 < p.length; j += 3) {
      const s = segE.length;
      segE.push(ei), segK.push(j);
      const dx = p[j + 3] - p[j], dy = p[j + 4] - p[j + 1], n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / step));
      const seen = new Set<string>();
      for (let t = 0; t <= n; t++) {
        const kk = key(p[j] + (dx * t) / n, p[j + 1] + (dy * t) / n);
        if (seen.has(kk)) continue;
        seen.add(kk);
        const list = grid.get(kk);
        if (list) list.push(s);
        else grid.set(kk, [s]);
      }
    }
  });
  cand.forEach((e, ei) => {
    const p = ptsOf(e.id);
    if (!p || p.length < 6) return;
    const j0 = Math.floor((p.length / 3 - 1) / 2) * 3;
    const x = (p[j0] + p[j0 + 3]) / 2, y = (p[j0 + 1] + p[j0 + 4]) / 2;
    let dx = p[j0 + 3] - p[j0], dy = p[j0 + 4] - p[j0 + 1];
    const dl = Math.hypot(dx, dy) || 1;
    dx /= dl, dy /= dl;
    const kk = kAtY(y), cx = Math.floor(x / cell), cy = Math.floor(y / cell);
    let best = -1, bestOff = Infinity;
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
      for (const s of grid.get(`${gx},${gy}`) ?? []) {
        const oi = segE[s];
        if (oi === ei || cand[oi].built !== e.built) continue;
        const q = ptsOf(cand[oi].id)!, j = segK[s];
        let ex = q[j + 3] - q[j], ey = q[j + 4] - q[j + 1];
        const el = Math.hypot(ex, ey);
        if (el <= 0) continue;
        ex /= el, ey /= el;
        if (Math.abs(ex * dx + ey * dy) < PARALLEL) continue;
        const t = (x - q[j]) * ex + (y - q[j + 1]) * ey;
        if (t < 0 || t > el) continue;
        const off = Math.abs(Math.hypot(q[j] + ex * t - x, q[j + 1] + ey * t - y) / kk - QUAD_M);
        if (off <= SLACK_M && off < bestOff) best = cand[oi].id, bestOff = off;
      }
    }
    if (best >= 0) out.set(e.id, best);
  });
  // a twin found one way only (a station splits one pair): both ways
  for (const [a, b] of [...out]) if (!out.has(b)) out.set(b, a);
  return out;
}

/** Where an edge's segments near a point are (snapIndex.ts `edgesNear`): null when the grid cannot say. */
export type NearSegments = (x: number, y: number, reach: number) => { edge: number; pts: ArrayLike<number>; ks: number[] }[] | null;

/** T-4111: the answers of `quadPairs` for one edge at a time, from the segments near it, so a network the edit changed
 * a little costs a few lookups, not a pass over every edge. Same answers, ties included: an edge's own twin is the
 * nearest segment (off the spacing) of those the full pass's grid would hold for its middle, the first in that pass's
 * order on a tie; an edge with none is the twin of the first edge, in order, whose own twin it is. */
export class QuadLookup {
  private cand: PairEdge[];
  private at = new Map<number, number>();
  private cell = 0;
  private step = 0;
  private own = new Map<number, number>();
  private whole: Map<number, number> | null = null;
  private static WIDE = -2;

  constructor(private edges: readonly PairEdge[], private ptsOf: (id: number) => ArrayLike<number> | undefined, private near: NearSegments) {
    this.cand = edges.filter((e) => e.tracks === 2 && !e.crossover);
    this.cand.forEach((e, i) => this.at.set(e.id, i));
    if (this.cand.length >= 2) {
      const first = ptsOf(this.cand[0].id);
      const k = kAtY(first && first.length >= 2 ? first[1] : 0);
      this.cell = CELL_M * k;
      this.step = STEP_M * k;
    }
  }

  /** The other pair of quad track `id` is one pair of, or -1. */
  twin(id: number): number {
    const ei = this.at.get(id);
    if (ei === undefined || this.cand.length < 2) return -1;
    if (!this.whole) {
      const own = this.best(ei);
      if (own >= 0) return own;
      if (own !== QuadLookup.WIDE) {
        const back = this.back(ei);
        if (back !== QuadLookup.WIDE) return back;
      }
      this.whole = quadPairs(this.edges, this.ptsOf);
    }
    return this.whole.get(id) ?? -1;
  }

  /** the first edge in order whose own twin is `ei`'s edge, or -1 */
  private back(ei: number): number {
    const e = this.cand[ei], p = this.ptsOf(e.id);
    if (!p) return -1;
    const found = new Set<number>();
    for (let j = 0; j + 5 < p.length; j += 3) {
      const dx = p[j + 3] - p[j], dy = p[j + 4] - p[j + 1], my = (p[j + 1] + p[j + 4]) / 2, kk = kAtY(my);
      const n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (60 * kk)));
      for (let t = 0; t < n; t++) {
        const f = (t + 0.5) / n;
        const hit = this.near(p[j] + dx * f, p[j + 1] + dy * f, (Math.hypot(dx, dy) / n / 2 + (QUAD_M + SLACK_M) * kAtY(my) + 1));
        if (!hit) return QuadLookup.WIDE;
        for (const h of hit) {
          const oi = this.at.get(h.edge);
          if (oi !== undefined && oi !== ei && this.cand[oi].built === e.built) found.add(oi);
        }
      }
    }
    let first = -1;
    for (const oi of [...found].sort((a, b) => a - b)) {
      const b = this.best(oi);
      if (b === QuadLookup.WIDE) return b;
      if (b === e.id) { first = oi; break; }
    }
    return first < 0 ? -1 : this.cand[first].id;
  }

  /** the edge `ei`'s middle has beside it at the spacing (what `quadPairs` sets for it first), -1 for none, WIDE to ask the full pass */
  private best(ei: number): number {
    const had = this.own.get(ei);
    if (had !== undefined) return had;
    const out = this.find(ei);
    this.own.set(ei, out);
    return out;
  }

  private find(ei: number): number {
    const e = this.cand[ei], p = this.ptsOf(e.id);
    if (!p || p.length < 6) return -1;
    const j0 = Math.floor((p.length / 3 - 1) / 2) * 3;
    const x = (p[j0] + p[j0 + 3]) / 2, y = (p[j0 + 1] + p[j0 + 4]) / 2;
    let dx = p[j0 + 3] - p[j0], dy = p[j0 + 4] - p[j0 + 1];
    const dl = Math.hypot(dx, dy) || 1;
    dx /= dl, dy /= dl;
    const kk = kAtY(y), cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell);
    const hits = this.near(x, y, (QUAD_M + SLACK_M) * kk + 1);
    if (!hits) return QuadLookup.WIDE;
    let best = -1, bestOff = Infinity, bestAt = [0, 0, 0];
    for (const h of hits) {
      const oi = this.at.get(h.edge);
      if (oi === undefined || oi === ei || this.cand[oi].built !== e.built) continue;
      const q = h.pts;
      for (const j of h.ks) {
        const rank = this.rank(q, j, cx, cy);
        if (rank < 0) continue;
        let ex = q[j + 3] - q[j], ey = q[j + 4] - q[j + 1];
        const el = Math.hypot(ex, ey);
        if (el <= 0) continue;
        ex /= el, ey /= el;
        if (Math.abs(ex * dx + ey * dy) < PARALLEL) continue;
        const t = (x - q[j]) * ex + (y - q[j + 1]) * ey;
        if (t < 0 || t > el) continue;
        const off = Math.abs(Math.hypot(q[j] + ex * t - x, q[j + 1] + ey * t - y) / kk - QUAD_M);
        if (off > SLACK_M) continue;
        // the full pass meets segments by cell (x, then y), then in segment order; the first of the smallest offs wins
        const at = [rank, oi, j];
        if (off < bestOff || (off === bestOff && (at[0] - bestAt[0] || at[1] - bestAt[1] || at[2] - bestAt[2]) < 0)) best = this.cand[oi].id, bestOff = off, bestAt = at;
      }
    }
    return best;
  }

  /** where the full pass's grid first lists a segment among the 3 x 3 cells round (cx, cy), in its order (0-8), or -1 when none of them holds it */
  private rank(p: ArrayLike<number>, j: number, cx: number, cy: number): number {
    const dx = p[j + 3] - p[j], dy = p[j + 4] - p[j + 1], n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / this.step));
    let rank = -1;
    for (let t = 0; t <= n; t++) {
      const gx = Math.floor((p[j] + (dx * t) / n) / this.cell) - (cx - 1), gy = Math.floor((p[j + 1] + (dy * t) / n) / this.cell) - (cy - 1);
      if (gx < 0 || gx > 2 || gy < 0 || gy > 2) continue;
      const r = gx * 3 + gy;
      if (rank < 0 || r < rank) rank = r;
    }
    return rank;
  }
}
