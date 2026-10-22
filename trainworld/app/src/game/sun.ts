// The sun for the top bar's sky (T-861): its height at a place and local clock time on the game's
// calendar (game/days.ts), from the usual low-precision formulas (declination and the equation of
// time to within a few minutes). Local time is the game's whole-hour standard time (game/localTime.ts),
// so solar noon moves with the place's longitude within its zone.

import { dayOfYear } from "./days";

const RAD = Math.PI / 180;

/** Solar declination (radians) and the equation of time (hours) on day of year `doy` (0 Jan 1). */
function sunDay(doy: number): [number, number] {
  const dec = -23.44 * RAD * Math.cos((2 * Math.PI / 365) * (doy + 10));
  const b = (2 * Math.PI / 365) * (doy - 81);
  const eot = (9.87 * Math.sin(2 * b) - 7.53 * Math.cos(b) - 1.5 * Math.sin(b)) / 60;
  return [dec, eot];
}

/** Hours the sun runs ahead of a clock on `utcOffset` at longitude `lon`. */
const solarLead = (lon: number, utcOffset: number, eot: number) => lon / 15 - utcOffset + eot;

/** The sun's elevation in degrees at local time `t` (seconds of game local time) at (lon, lat). */
export function sunElevation(t: number, lon: number, lat: number, utcOffset: number): number {
  const day = Math.floor(t / 86400);
  const [dec, eot] = sunDay(dayOfYear(day));
  const solar = (t / 3600) - day * 24 + solarLead(lon, utcOffset, eot);
  const h = (solar - 12) * 15 * RAD, phi = lat * RAD;
  return Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(h)) / RAD;
}
