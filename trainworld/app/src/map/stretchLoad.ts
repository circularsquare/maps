// T-2007: a stretch's average load in the track hover, the line panel's "Average load" (T-820) for
// one stretch: the stretch's riders on the shown day over what the day's trips could hold at crush.
// The trips a day come from the boarding account (`linePeriods`), asked once per line, network
// version, account and day, and kept; the hover asks again when an answer lands.

import { signal } from "@preact/signals";
import { accountVersion, CRUSH_PER_CAR } from "../game/boarding";
import { AVERAGE_DAY, type DemandDay } from "../game/days";
import { PERIODS } from "../game/periods";
import { PERIOD_STRIDE } from "../game/today";
import { queryClock } from "../workers/clockClient";

/** moved on as an answer lands, so a resting hover says it */
export const tripsLanded = signal(0);

/** per line, network version, account version and day: the trips, null for none, "asked" meanwhile */
const trips = new Map<string, number | null | "asked">();

/** Trips a day on line `num` (both ways) on kind of day `day`, or undefined while asked. */
function tripsOn(num: number, version: number, day: number): number | null | undefined {
  const key = `${num}|${version}|${accountVersion.peek()}|${day}`;
  const had = trips.get(key);
  if (had === "asked") return undefined;
  if (had !== undefined) return had;
  if (trips.size > 200) trips.clear();
  trips.set(key, "asked");
  void queryClock({ what: "linePeriods", line: num, day }).then((v) => {
    // no answer (no clock, or a busy one): asked again on the next hover
    if (!v) return void trips.delete(key);
    let n: number | null = null;
    if (v.length >= PERIODS.length * PERIOD_STRIDE) {
      n = 0;
      for (let i = 0; i < PERIODS.length; i++) n += v[i * PERIOD_STRIDE];
    }
    trips.set(key, n);
    tripsLanded.value++;
  });
  return undefined;
}

/** The share of crush room the stretch's `riders` (both ways, on the shown `day`) fill, 0-1, or
 * undefined while the trips are asked, null with no trips. */
export function stretchLoad(num: number, version: number, cars: number, day: DemandDay, riders: number): number | null | undefined {
  let n: number | null | undefined;
  if (day === AVERAGE_DAY) {
    // 5 weekdays, a Saturday and a Sunday over 7, as the average day's riders
    const t = [0, 1, 2].map((k) => tripsOn(num, version, k));
    if (t.some((x) => x === undefined)) return undefined;
    n = t.some((x) => x === null) ? null : (5 * t[0]! + t[1]! + t[2]!) / 7;
  } else n = tripsOn(num, version, day);
  if (n === undefined) return undefined;
  const room = (n ?? 0) * CRUSH_PER_CAR * cars;
  return room > 0 ? Math.min(1, riders / room) : null;
}
