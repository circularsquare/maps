// Local time per region (T-309, docs/world.md "Local time"). There is one game clock: game seconds
// of world time, counted in the registry's `clockUtcOffset` (UTC-5, New York's standard time, so
// saves from before local time read the same). Each region runs on its own local time: its demand
// periods, its lines' service periods and boarding hours, its weekends, its fares in money. The top
// bar shows the local time of the place in view.
//
// A place's offset is its zone's `utcOffset` (the registry, a whole hour of standard time, no
// daylight saving), the zone being the nearest registered centre within reach (`centreWithin`,
// the track model's `localtime.rs` rule, which has no outlines); a place in no zone takes its
// longitude's nominal offset, `round(lon / 15)`, until its region is registered. A line runs on the local time of its
// first stop (sim/src/track/localtime.rs).
//
// Plain data and one signal: the clock worker imports it too.

import { signal } from "@preact/signals";
import { CENTRE_REACH_M, CITIES, cityById, centreWithin, REGISTRY } from "./cities";

/** The UTC offset of the game clock, hours. */
export const CLOCK_UTC_OFFSET = REGISTRY.clockUtcOffset ?? -5;

/** The nominal offset of a longitude, hours: where no registered region says otherwise. */
export const lonOffset = (lon: number): number => Math.round(lon / 15);

/** Whole-hour standard UTC offset at (lon, lat), hours. */
export function utcOffsetAt(lon: number, lat: number): number {
  const id = centreWithin(lon, lat);
  return (id ? cityById(id).utcOffset : undefined) ?? lonOffset(lon);
}

/** Seconds local time at (lon, lat) runs ahead of the game clock. */
export const shiftAt = (lon: number, lat: number): number => (utcOffsetAt(lon, lat) - CLOCK_UTC_OFFSET) * 3600;

/** The track model's table (`TrackApi.set_time_zones`): the clock's UTC offset in hours, the reach
 * in metres, then per registered zone its centre's lon and lat and its UTC offset in hours. */
export function timeZoneTable(): Float64Array {
  const v = [CLOCK_UTC_OFFSET, CENTRE_REACH_M];
  for (const c of CITIES) v.push(c.center[0], c.center[1], c.utcOffset ?? lonOffset(c.center[0]));
  return new Float64Array(v);
}

/** Seconds the local time in view runs ahead of the game clock (main thread; set by
 * map/zoneFocus.ts after each pan). 0 is New York's, and every region registered today. */
export const viewShift = signal(0);
