// A line's riders per stretch carried onto a changed stop list (T-1419): while demand solves for an
// edited network, the Traffic view keeps showing the line's last figures where its stops still are.
// Pure, no signals (checks/t1419-carry.mjs runs it in node).

/** What makes two stop ids the same place: the station itself, or its complex (game/complexView.ts). */
export type StopKey = (id: string) => string;

/** At least this share of the new stops must have been on the line before; a line rebuilt beyond
 * that is shown blank. */
const KEEP_SHARE = 0.5;

/**
 * `loads[period]` as `LineDemand.loads` lays them out for the stops `was` (run 0's stretches, then
 * run 1's in its own travel order), made over for the stops `now`; null when too few of the new
 * stops were on the line. A new stretch between two stops that were on the line takes the busiest
 * old stretch it covers, so a stop added between two others leaves both halves as the old whole and
 * a stop removed leaves the busier side; a stretch next to a new end stop is 0.
 */
export function carryLoads(loads: readonly Float32Array[], was: readonly string[], now: readonly string[], key: StopKey = (s) => s): Float32Array[] | null {
  const m = was.length, n = now.length;
  if (m < 2 || n < 2 || !loads.length || loads.some((p) => p.length !== 2 * (m - 1))) return null;
  const before = was.map(key);
  // each new stop's place on the old line, in order (a loop's first stop comes twice)
  const at = new Int32Array(n).fill(-1);
  let known = 0;
  for (let i = 0, from = 0; i < n; i++) {
    const k = key(now[i]);
    for (let j = from; j < m; j++) {
      if (before[j] !== k) continue;
      at[i] = j;
      from = j + 1;
      known++;
      break;
    }
  }
  if (known < 2 || known < KEEP_SHARE * n) return null;
  // for each new stretch the old ones it covers, [lo, hi), or none
  const lo = new Int32Array(n - 1).fill(-1), hi = new Int32Array(n - 1);
  for (let k = 0; k < n - 1; k++) {
    let a = k, b = k + 1;
    while (a >= 0 && at[a] < 0) a--;
    while (b < n && at[b] < 0) b++;
    if (a < 0 || b >= n) continue;
    lo[k] = at[a];
    hi[k] = at[b];
  }
  return loads.map((p) => {
    const out = new Float32Array(2 * (n - 1));
    for (let k = 0; k < n - 1; k++) {
      if (lo[k] < 0) continue;
      let fwd = 0, back = 0;
      for (let s = lo[k]; s < hi[k]; s++) {
        fwd = Math.max(fwd, p[s]);
        // run 1's stretch j covers run 0's stretch m - 2 - j
        back = Math.max(back, p[m - 1 + (m - 2 - s)]);
      }
      out[k] = fwd;
      out[n - 1 + (n - 2 - k)] = back;
    }
    return out;
  });
}
