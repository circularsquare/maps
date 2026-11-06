// T-1894: which end of a line its diagram puts on top: the north end, or the west end for a line
// running within about 22 degrees of east-west; a loop as built. The line panel's stops and the
// group view's spine read it. T-2661: `lineReversed` orders a line left to right instead, west end on
// the left (north end for a line nearly north-south): a strip or chart of a line's stops or stretches
// reads it.

import type { WorldView } from "./types";
import { stationsById } from "./world";

/** A line nearer east-west than this (north-south over east-west distance) puts its west end on top. */
const EAST_WEST = 0.4;

/** True when a line from `a` to `b` (lng, lat) is drawn last stop first. */
export function drawnReversed(a: { lng: number; lat: number } | undefined, b: { lng: number; lat: number } | undefined, loop = false): boolean {
  if (!a || !b || loop) return false;
  const dy = b.lat - a.lat, dx = (b.lng - a.lng) * Math.cos(((a.lat + b.lat) / 2) * (Math.PI / 180));
  if (dx === 0 && dy === 0) return false;
  return Math.abs(dy) >= EAST_WEST * Math.abs(dx) ? dy > 0 : dx < 0;
}

/** A line this much more north-south than east-west (north-south over east-west distance) reads
 * north first from left to right; any other reads west first. */
const NORTH_SOUTH = 2.5;

/** T-2661: true when a line from `a` to `b` is drawn last stop first in a left-to-right strip: its
 * west end on the left, or for a line within about 22 degrees of north-south its north end; a loop
 * as built. (Not `drawnReversed`: a diagonal line's top end in the vertical diagram is its north
 * end, which for a north-east line is the east one.) */
export function leftToRightReversed(a: { lng: number; lat: number } | undefined, b: { lng: number; lat: number } | undefined, loop = false): boolean {
  if (!a || !b || loop) return false;
  const dy = b.lat - a.lat, dx = (b.lng - a.lng) * Math.cos(((a.lat + b.lat) / 2) * (Math.PI / 180));
  if (dx === 0 && dy === 0) return false;
  return Math.abs(dy) >= NORTH_SOUTH * Math.abs(dx) ? dy > 0 : dx < 0;
}

/** T-2661: whether to draw line `l` (its stops by station id) last stop first in a left-to-right
 * strip, chart or row (`leftToRightReversed`); `loop` as `LineStats.loop`. A line whose end stops are
 * no stations (a bus line's) gives false. */
export function lineReversed(l: { stops: readonly string[] }, w: WorldView, loop: boolean): boolean {
  const by = stationsById(w);
  return leftToRightReversed(by.get(l.stops[0]), by.get(l.stops[l.stops.length - 1]), loop);
}
