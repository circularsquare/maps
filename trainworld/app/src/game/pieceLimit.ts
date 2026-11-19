// The speed limit stepper's step for pieces of track (T-3405, T-3525), kept free of imports so a Node check can run it.

export const LIMIT_STEP = 10;
export const LIMIT_MIN = 40;

/** The speed one step `d` (+1 or -1) asks of every piece, from what each runs at now (`speeds`: its limit, else its
 * kind's top) and each one's top; null when no piece can move that way. All alike, the step is from their speed.
 * Mixed, a step down is from the slowest (every piece then gets the same limit, below all their tops), and a step up
 * from the slowest piece still under its own top, so a piece already at its top (a tram at 80 beside rail) never
 * holds the others back. */
export function nextLimit(speeds: readonly number[], tops: readonly number[], d: number): number | null {
  if (!speeds.length) return null;
  const top = Math.max(...tops);
  const same = speeds.every((v) => v === speeds[0]);
  let from: number;
  if (same) from = speeds[0];
  else if (d < 0) from = Math.min(...speeds);
  else {
    const under = speeds.filter((v, i) => v < tops[i]);
    if (!under.length) return null;
    from = Math.min(...under);
  }
  const to = Math.max(LIMIT_MIN, Math.min(top, from + d * LIMIT_STEP));
  return same && to === from ? null : to;
}
