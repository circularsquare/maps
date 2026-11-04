// T-2365: zones' trips by length as the panels last asked for them (the city view's graph, the top
// bar's hover), and a place's corridors as trips at a distance, for the graph's far end.

import { signal } from "@preact/signals";
import { CITIES, type City, type CityId } from "./cities";
import { demandView, eachCorridor, zoneTripLengths, type Corridors } from "./demand";
import type { FarTrips, TripLengths } from "./tripLengths";
import { shownDay } from "./ui";

/** Each zone's newest answer; one of an older solve or day stays until the next lands. */
export const zoneLengths = signal<ReadonlyMap<CityId, TripLengths>>(new Map());
const asked = new Map<CityId, string>();

/** Ask for `zone`'s trips by length for the solve and day shown, once each. */
export function wantLengths(zone: CityId) {
  const dv = demandView.peek();
  if (!dv) return;
  const key = `${dv.version}:${dv.round}:${shownDay.peek()}`;
  if (asked.get(zone) === key) return;
  asked.set(zone, key);
  void zoneTripLengths(zone).then((t) => {
    if (asked.get(zone) !== key) return;
    if (!t) return void asked.delete(zone);
    const m = new Map(zoneLengths.peek());
    m.set(zone, t);
    zoneLengths.value = m;
  });
}

/** `zone`'s answer for the day shown, or null. */
export function lengthsOf(zone: CityId): TripLengths | null {
  const t = zoneLengths.value.get(zone);
  return t && t.day === shownDay.value ? t : null;
}

let centres: Map<number, City> | null = null;
/** A place's centre: its first zone's, as the demand client's (`centresOfPlaces`). */
const placeCity = (place: number) => {
  if (!centres) {
    centres = new Map();
    for (const c of CITIES) if (!centres.has(c.place)) centres.set(c.place, c);
  }
  return centres.get(place);
};

/** Straight-line km between two [lon, lat] points. */
function km([lon1, lat1]: [number, number], [lon2, lat2]: [number, number]): number {
  const r = Math.PI / 180, a = Math.sin(((lat2 - lat1) * r) / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 2 * 6371.0088 * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** The trips between `place` and every other place, as the city view's To other places counts
 * them, each at the distance between the two places' centres. */
export function farTrips(place: number, c: Corridors): FarTrips[] {
  const out: FarTrips[] = [];
  const here = placeCity(place)?.center;
  eachCorridor(c, (from, to) => from !== to && (from === place || to === place), (j, v) => {
    const other = c.table.from[j] === place ? c.table.to[j] : c.table.from[j];
    const there = placeCity(other)?.center;
    out.push({ km: here && there ? km(here, there) : Infinity, trips: v.trips, rail: v.rail, air: v.air });
  });
  return out;
}
