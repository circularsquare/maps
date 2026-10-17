// The marks drawn on the map for grade changes and meeting tracks (T-629, T-655), as numbers: a
// flat crossing, a junction at ground level, a ramp where track climbs from level 0, a tunnel
// portal where it drops from level 0. Pure data, so the clock worker builds them with the rest of
// the network (network.ts) and map/gradeMarks.ts draws them.

/** Floats per mark: x, y (local units), shape, angle (rad, screen frame, ramps and portals), tier
 * (0 built, 1 blueprint or being drawn), level (a ramp's far end, a junction or crossing's own). */
export const MARK_STRIDE = 6;

export const M_CROSSING = 0;
export const M_JUNCTION = 1;
export const M_RAMP = 2;
export const M_PORTAL = 3;
/** a crossing the track model refuses: heights differ by less than a level, or one is on a ramp */
export const M_BAD = 4;

/** Height of one level, m (sim `LEVEL_H`). */
const LEVEL_M = 8;
/** A sample this close to the ground is on it, m. */
const GROUND_M = 0.1;

/** Appends to `out` a mark for every place the track in `pts` (x, y, height triples, local units)
 * leaves the ground: at the ramp's level 0 end, pointing along the ramp. Ramps between two other
 * levels (1 to 2, -1 to -2) get none. */
export function rampMarks(pts: ArrayLike<number>, tier: number, out: number[]) {
  const n = pts.length / 3;
  for (let k = 0; k + 1 < n; k++) {
    const g0 = Math.abs(pts[k * 3 + 2]) < GROUND_M, g1 = Math.abs(pts[k * 3 + 5]) < GROUND_M;
    if (g0 === g1) continue;
    const foot = g0 ? k : k + 1, step = g0 ? 1 : -1;
    const dx = pts[(foot + step) * 3] - pts[foot * 3], dy = pts[(foot + step) * 3 + 1] - pts[foot * 3 + 1];
    if (Math.hypot(dx, dy) < 1e-6) continue;
    // the level the ramp climbs or drops to: on along it while the height keeps going the same way
    let far = foot + step, z = pts[far * 3 + 2];
    while (far + step >= 0 && far + step < n && Math.abs(pts[(far + step) * 3 + 2]) > Math.abs(z) + 1e-3 && pts[(far + step) * 3 + 2] * z > 0) {
      far += step;
      z = pts[far * 3 + 2];
    }
    const level = Math.sign(z) * Math.max(1, Math.min(3, Math.round(Math.abs(z) / LEVEL_M)));
    out.push(pts[foot * 3], pts[foot * 3 + 1], z > 0 ? M_RAMP : M_PORTAL, Math.atan2(dy, dx), tier, level);
  }
}
