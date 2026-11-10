// The marks drawn on the map for grade changes and meeting tracks (T-629, T-655, T-1871), as
// numbers: a flat crossing, a junction at ground level, and a ramp mark for every change of level
// along track. Pure data, so the clock worker builds them with the rest of the network
// (network.ts) and map/gradeMarks.ts draws them.

/** Floats per mark: x, y (local units), shape, angle (rad, screen frame, ramps), tier
 * (0 built, 1 blueprint or being drawn), level (a ramp's far end, a junction or crossing's own). */
export const MARK_STRIDE = 6;

export const M_CROSSING = 0;
export const M_JUNCTION = 1;
/** a ramp whose far end is above ground (level +1 to +3) */
export const M_RAMP = 2;
/** a ramp whose far end is below ground (level -1 to -3) */
export const M_PORTAL = 3;
/** a crossing the track model refuses: heights differ by less than a level, or one is on a ramp */
export const M_BAD = 4;

/** T-3062: a flat crossing's modes ride in its level slot as a tenth per tram track: level + 0.1 for a
 * tram over rail track (the tram yields), + 0.2 for two tram tracks. Whole levels still round to
 * themselves, so every reader of the level is unaffected. */
export const packCross = (level: number, trams: number) => level + 0.1 * trams;
/** The tram tracks (0 to 2) of a crossing's packed level. */
export const crossTrams = (packed: number) => Math.round((packed - Math.round(packed)) * 10);

/** Height of one level, m (sim `LEVEL_H`). */
const LEVEL_M = 8;
/** A step between two samples that changes the height by more than this is on a ramp, m. */
const SLOPE_M = 1e-3;
/** A sample this close to the ground is on it, m. */
const GROUND_M = 0.1;

/** Appends to `out` one mark for every ramp in `pts` (x, y, height triples, local units): a stretch
 * where the height keeps rising or keeps falling, up to the level it reaches. The mark sits at the
 * middle of the stretch and points along the track toward the end that is farther from the ground;
 * its level is that end's. A stretch that crosses the ground (-1 to +2) gets one mark for each side. */
export function rampMarks(pts: ArrayLike<number>, tier: number, out: number[]) {
  const n = pts.length / 3;
  const dz = (k: number) => pts[k * 3 + 5] - pts[k * 3 + 2];
  let k = 0;
  while (k + 1 < n) {
    if (Math.abs(dz(k)) <= SLOPE_M) { k++; continue; }
    const rising = dz(k) > 0;
    let end = k + 1;
    while (end + 1 < n && (rising ? dz(end) > SLOPE_M : dz(end) < -SLOPE_M)) end++;
    const run: number[] = [];
    for (let j = k; j <= end; j++) run.push(pts[j * 3], pts[j * 3 + 1], pts[j * 3 + 2]);
    // split at the ground, so each side is a ramp to its own level
    let cut = -1, onGround = -1;
    for (let j = 0; j + 1 < run.length / 3; j++) {
      if (j > 0 && Math.abs(run[j * 3 + 2]) < GROUND_M) { onGround = j; break; }
      if (run[j * 3 + 2] * run[j * 3 + 5] < 0) { cut = j; break; }
    }
    if (onGround > 0) {
      markRun(run.slice(0, onGround * 3 + 3), tier, out);
      markRun(run.slice(onGround * 3), tier, out);
    } else if (cut < 0) markRun(run, tier, out);
    else {
      const a = run[cut * 3 + 2], b = run[cut * 3 + 5], t = a / (a - b);
      const cx = run[cut * 3] + t * (run[cut * 3 + 3] - run[cut * 3]);
      const cy = run[cut * 3 + 1] + t * (run[cut * 3 + 4] - run[cut * 3 + 1]);
      markRun([...run.slice(0, cut * 3 + 3), cx, cy, 0], tier, out);
      markRun([cx, cy, 0, ...run.slice(cut * 3 + 3)], tier, out);
    }
    k = end;
  }
}

/** The mark of one ramp stretch that stays on one side of the ground. */
function markRun(run: number[], tier: number, out: number[]) {
  const n = run.length / 3;
  const zFirst = run[2], zLast = run[(n - 1) * 3 + 2];
  const forward = Math.abs(zLast) >= Math.abs(zFirst);
  const zFar = forward ? zLast : zFirst;
  const steps = Math.round(Math.abs(zFar) / LEVEL_M);
  if (steps < 1) return;
  const level = Math.sign(zFar) * Math.min(3, steps);
  let total = 0;
  for (let j = 0; j + 1 < n; j++) total += Math.hypot(run[j * 3 + 3] - run[j * 3], run[j * 3 + 4] - run[j * 3 + 1]);
  if (total < 1e-6) return;
  let left = total / 2;
  for (let j = 0; j + 1 < n; j++) {
    const dx = run[j * 3 + 3] - run[j * 3], dy = run[j * 3 + 4] - run[j * 3 + 1], len = Math.hypot(dx, dy);
    if (left > len && j + 2 < n) { left -= len; continue; }
    const t = len > 0 ? Math.min(1, left / len) : 0;
    out.push(run[j * 3] + t * dx, run[j * 3 + 1] + t * dy, zFar > 0 ? M_RAMP : M_PORTAL, Math.atan2(forward ? dy : -dy, forward ? dx : -dx), tier, level);
    return;
  }
}
