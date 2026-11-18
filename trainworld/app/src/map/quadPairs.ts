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
