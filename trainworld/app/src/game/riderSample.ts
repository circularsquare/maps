// T-1436: one rider picked from a train's riders. The boarding account counts riders aboard by
// stretch, not by trip, so the rider is drawn the way the account lets people on and off: where
// they got on from what boarded at each stop and is still aboard (stayed on through every stop
// after it, by the line's alighting shares), where they get off from the shares at the stops ahead,
// and why they ride from the stretch's riders by purpose. Stops are in the train's travel order.

import type { DemandView } from "./demand";
import type { PurposeKey } from "./purposes";

export interface RiderPick {
  from: number;
  to: number;
  purpose: PurposeKey;
}

const share = (v: number | undefined) => Math.min(1, Math.max(0, v ?? 0));

/** An index drawn in proportion to `w`; none carries weight, an even draw over `fallback`. */
function draw(w: number[], fallback: number[], rand: () => number): number {
  let sum = 0;
  for (const v of w) sum += v ?? 0;
  if (!(sum > 1e-9)) return fallback[Math.floor(rand() * fallback.length)];
  let at = rand() * sum;
  for (let i = 0; i < w.length; i++) {
    at -= w[i] ?? 0;
    if (at < 0) return i;
  }
  return w.length - 1;
}

/**
 * Where a rider aboard on stretch `seg` (stop `seg` to `seg + 1`) got on and gets off. `board` is the
 * period's boardings per route node (`run * n + stop`), `shares` the share of the riders arriving at
 * each stop who get off there (`alightShares`), `last` the last stop a trip reaches (a circle's is
 * the one before its first stop again). Null where the stretch has no stop ahead.
 */
export function pickStops(o: { board: Float32Array; run: number; n: number; seg: number; last: number; shares: number[]; rand: () => number }): { from: number; to: number } | null {
  const { board, run, n, seg, last, shares, rand } = o;
  if (seg < 0 || seg >= last) return null;
  const on = new Array<number>(seg + 1).fill(0);
  let stay = 1;
  for (let k = seg; k >= 0; k--) {
    on[k] = Math.max(0, board[run * n + k] ?? 0) * stay;
    stay *= 1 - share(shares[k]);
  }
  const off = new Array<number>(last + 1).fill(0);
  stay = 1;
  for (let m = seg + 1; m <= last; m++) {
    off[m] = stay * share(shares[m]);
    stay *= 1 - share(shares[m]);
  }
  const from = draw(on, Array.from({ length: seg + 1 }, (_, k) => k), rand);
  const to = draw(off, Array.from({ length: last - seg }, (_, i) => seg + 1 + i), rand);
  return { from, to };
}

/** The purposes of a line's riders on a stretch in period `p`, with their riders (`idx`: the
 * stretch's place in `LineDemand.loads`, run 0's segments then run 1's). */
export function purposeWeights(dv: DemandView, lineId: string, p: number, idx: number): { key: PurposeKey; w: number }[] {
  const out: { key: PurposeKey; w: number }[] = [];
  for (const [key, d] of dv.purposes ?? []) {
    const w = d.lines.get(lineId)?.loads[p]?.[idx] ?? 0;
    if (w > 0) out.push({ key, w });
  }
  return out;
}

/** A purpose drawn by its riders; "work" where the figures have none. */
export function pickPurpose(weights: { key: PurposeKey; w: number }[], rand: () => number): PurposeKey {
  if (!weights.length) return "work";
  return weights[draw(weights.map((x) => x.w), [0], rand)].key;
}
