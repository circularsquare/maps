// T-909: a selection's far end grouped for the display, so no far bubble stands for a speck of a
// trip (notes/T-909.md). Display only: demand and every figure outside the far end are untouched.
// T-1210: the demand sums worker deals a selection's far ends as it answers (`dealtCells`), and
// `primeDealt` keeps them here, so `dealFlows` on the main thread only looks them up.
import type { FarEnd, Flows } from "./demand";
import type { PurposeFilter, PurposeKey } from "./purposes";

/** Trips a day (as a far bubble's hover counts them) that a far cell under it is grouped into. */
export const FAR_CHUNK = 10;

const ranks = new WeakMap<Float32Array, Uint32Array>();

/** Each cell's place along a Hilbert curve over the city's cells (`xy` pairs), so cells near on
 * the curve are near on the map. */
export function curveRank(xy: Float32Array): Uint32Array {
  const had = ranks.get(xy);
  if (had) return had;
  const n = xy.length / 2;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let c = 0; c < n; c++) {
    x0 = Math.min(x0, xy[2 * c]), x1 = Math.max(x1, xy[2 * c]);
    y0 = Math.min(y0, xy[2 * c + 1]), y1 = Math.max(y1, xy[2 * c + 1]);
  }
  const side = 1 << 16;
  const sx = (side - 1) / Math.max(x1 - x0, 1), sy = (side - 1) / Math.max(y1 - y0, 1);
  const d = new Float64Array(n);
  for (let c = 0; c < n; c++) {
    let x = Math.floor((xy[2 * c] - x0) * sx), y = Math.floor((xy[2 * c + 1] - y0) * sy), k = 0;
    for (let s = side >> 1; s > 0; s >>= 1) {
      const rx = x & s ? 1 : 0, ry = y & s ? 1 : 0;
      k += s * s * ((3 * rx) ^ ry);
      if (!ry) {
        if (rx) (x = side - 1 - x), (y = side - 1 - y);
        const t = x;
        (x = y), (y = t);
      }
    }
    d[c] = k;
  }
  const order = Array.from({ length: n }, (_, c) => c).sort((a, b) => d[a] - d[b]);
  const rank = new Uint32Array(n);
  order.forEach((c, i) => (rank[c] = i));
  ranks.set(xy, rank);
  return rank;
}

/** Where each small far cell's trips go: the share it keeps, and the cell taking the rest. */
interface Plan {
  keep: Map<number, number>;
  to: Map<number, number>;
}

/**
 * The far cells of `fe` under `FAR_CHUNK` trips, in curve order, dealt their total in draws of
 * total / floor(total / FAR_CHUNK) (so each at least FAR_CHUNK), systematic from a fixed start:
 * a cell holding draws keeps its trips up to its last draw, and the trips after it go to the next
 * cell holding one (those after the last draw to the first). Each drawn cell so carries whole
 * draws, a cell is drawn in proportion to its trips, and trips move only along the curve.
 */
function plan(fe: FarEnd, rank: Uint32Array): Plan | null {
  const small: number[] = [];
  const t = new Map<number, number>();
  for (let k = 0; k < fe.cells.length; k++) {
    const c = fe.cells[k];
    const v = fe.farRail[k] + fe.farWalk[k] + fe.farDrive[k];
    if (v > 0 && v < FAR_CHUNK && c < rank.length) small.push(c), t.set(c, (t.get(c) ?? 0) + v);
  }
  if (small.length < 2) return null;
  small.sort((a, b) => rank[a] - rank[b]);
  let total = 0;
  for (const c of small) total += t.get(c)!;
  const draws = Math.max(1, Math.floor(total / FAR_CHUNK));
  if (draws >= small.length) return null;
  const step = total / draws;
  const AT = 0.5;
  const before = (x: number) => Math.min(draws, Math.max(0, Math.ceil(x / step - AT)));
  const keep = new Map<number, number>(), to = new Map<number, number>();
  let pending: number[] = [];
  let first = -1, lo = 0;
  for (const c of small) {
    const v = t.get(c)!, hi = lo + v;
    const n = before(hi) - before(lo);
    if (n > 0) {
      const last = (before(hi) - 1 + AT) * step;
      keep.set(c, Math.min(1, Math.max(0, (last - lo) / v)));
      for (const p of pending) to.set(p, c);
      pending = [];
      if (first < 0) first = c;
    } else keep.set(c, 0);
    pending.push(c);
    lo = hi;
  }
  for (const p of pending) to.set(p, first);
  return { keep, to };
}

/** A far end's cells and their trips by mode. */
export type FarCells = Pick<FarEnd, "cells" | "farRail" | "farWalk" | "farDrive">;

function apply(fe: FarEnd, p: Plan): FarCells {
  const sum = new Map<number, [number, number, number]>();
  const add = (c: number, f: number, k: number) => {
    if (f <= 0) return;
    const s = sum.get(c) ?? [0, 0, 0];
    s[0] += f * fe.farRail[k], s[1] += f * fe.farWalk[k], s[2] += f * fe.farDrive[k];
    sum.set(c, s);
  };
  for (let k = 0; k < fe.cells.length; k++) {
    const c = fe.cells[k], keep = p.keep.get(c);
    if (keep === undefined) add(c, 1, k);
    else add(c, keep, k), add(p.to.get(c)!, 1 - keep, k);
  }
  const cells = [...sum.keys()].filter((c) => { const s = sum.get(c)!; return s[0] + s[1] + s[2] > 0; });
  return {
    cells: Uint32Array.from(cells),
    farRail: Float32Array.from(cells, (c) => sum.get(c)![0]),
    farWalk: Float32Array.from(cells, (c) => sum.get(c)![1]),
    farDrive: Float32Array.from(cells, (c) => sum.get(c)![2]),
  };
}

/** `dealFlows`' new far ends for `filter`: the whole far end's, then each purpose's in
 * `f.byPurpose` order; null when nothing is grouped (the far end is shown as it is). */
export interface DealtCells { all: FarCells; byPurpose: FarCells[] | null }
export function dealtCells(f: FarEnd & { byPurpose?: Map<PurposeKey, FarEnd> }, filter: PurposeFilter, rank: Uint32Array): DealtCells | null {
  const shown = filter === "all" ? f : f.byPurpose?.get(filter);
  const p = shown ? plan(shown, rank) : null;
  if (!p) return null;
  return { all: apply(f, p), byPurpose: f.byPurpose ? [...f.byPurpose.values()].map((fe) => apply(fe, p)) : null };
}

const dealt = new WeakMap<Flows, Map<PurposeFilter, Flows>>();

function remember(f: Flows, filter: PurposeFilter, d: DealtCells | null): Flows {
  let out = f;
  if (d) {
    out = { ...f, ...d.all };
    if (f.byPurpose) {
      const parts = d.byPurpose!;
      out.byPurpose = new Map([...f.byPurpose].map(([k, fe], i) => [k, { ...fe, ...parts[i] }]));
    }
  }
  if (!dealt.has(f)) dealt.set(f, new Map());
  dealt.get(f)!.set(filter, out);
  return out;
}

/** T-1210: `f` dealt for `filter` elsewhere (`dealtCells`, in the sums worker), kept for `dealFlows`. */
export function primeDealt(f: Flows, filter: PurposeFilter, d: DealtCells | null) {
  remember(f, filter, d);
}

/**
 * `f` with its far end grouped (`plan`) as shown under the purpose filter: the cells are drawn
 * from that far end's trips and every purpose's far end moves the same way, so the far end under
 * each filter and its purposes still add up, and every total by mode and purpose is exact. `xy`:
 * the city's cells (`CityCells.xy`). The same `f` gives the same answer, so the bubbles and the
 * trip explorer agree (T-586).
 */
export function dealFlows(f: Flows, filter: PurposeFilter, xy: Float32Array): Flows {
  const memo = dealt.get(f)?.get(filter);
  if (memo) return memo;
  const shown = filter === "all" ? f : f.byPurpose?.get(filter);
  return remember(f, filter, shown ? dealtCells(f, filter, curveRank(xy)) : null);
}
