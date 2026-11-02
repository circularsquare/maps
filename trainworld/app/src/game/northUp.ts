// T-1894: which end of a line its diagram puts on top: the north end, or the west end for a line
// running within about 22 degrees of east-west; a loop as built. The line panel's stops and the
// group view's spine read it.

/** A line nearer east-west than this (north-south over east-west distance) puts its west end on top. */
const EAST_WEST = 0.4;

/** True when a line from `a` to `b` (lng, lat) is drawn last stop first. */
export function drawnReversed(a: { lng: number; lat: number } | undefined, b: { lng: number; lat: number } | undefined, loop = false): boolean {
  if (!a || !b || loop) return false;
  const dy = b.lat - a.lat, dx = (b.lng - a.lng) * Math.cos(((a.lat + b.lat) / 2) * (Math.PI / 180));
  if (dx === 0 && dy === 0) return false;
  return Math.abs(dy) >= EAST_WEST * Math.abs(dx) ? dy > 0 : dx < 0;
}
