// T-332: what the Lines and Stations tabs list. A toggle between the viewed place (`viewedZone`,
// T-318) and the whole world, and a sort by distance from the centre of the screen. A station is in
// a place when `placeAt` puts it there; a line when any of its stops is. Over dead space (no viewed
// place) the lists show the whole world. The centre is kept on move end, not every frame.

import { signal } from "@preact/signals";
import { viewedZone, type CityId } from "./cities";
import { placeAt, zonesLoaded } from "./zones";
import type { LineInput, StationInput } from "./types";

export type ListScope = "here" | "world";
/** Which stations and lines the tabs list; kept while the page is open, like the sorts. */
export const listScope = signal<ListScope>("here");
/** The map's centre [lng, lat] as of the last move end (map/zoneFocus.ts). */
export const viewCentre = signal<[number, number] | null>(null);

/** The place to filter the lists to, or null for the whole world (chosen, or nothing is modelled here). */
export function scopeZone(): CityId | null {
  return listScope.value === "here" ? viewedZone.value : null;
}

let cacheAt = -1;
const places = new Map<string, { lng: number; lat: number; zone: CityId | null }>();
/** The place a station is in, remembered while the zone boundaries are the same. */
export function stationZone(s: Pick<StationInput, "id" | "lng" | "lat">): CityId | null {
  if (cacheAt !== zonesLoaded.peek()) { places.clear(); cacheAt = zonesLoaded.peek(); }
  const c = places.get(s.id);
  if (c && c.lng === s.lng && c.lat === s.lat) return c.zone;
  const zone = placeAt(s.lng, s.lat);
  places.set(s.id, { lng: s.lng, lat: s.lat, zone });
  return zone;
}

/** Ground metres from the centre to a point (flat approximation; the list only needs the order). */
export function distFromCentre(lng: number, lat: number, c: [number, number]): number {
  const k = 111_320;
  return Math.hypot((lng - c[0]) * k * Math.cos(((lat + c[1]) / 2) * Math.PI / 180), (lat - c[1]) * k);
}

export function fmtDist(m: number): string {
  return m < 1000 ? `${Math.round(m / 10) * 10} m` : m < 10_000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m / 1000).toLocaleString("en-US")} km`;
}

/** What a tab shows: the stations in the scope, each line with any stop in it, and each station's and
 * line's distance from the centre (a line's is its nearest stop's; null for a line with no stops).
 * A line with no stops, and the selected row, stay listed so nothing just made or opened vanishes. */
export function scopedLists(stations: readonly StationInput[], lines: readonly LineInput[], keep: (kind: "line" | "station", id: string) => boolean) {
  void zonesLoaded.value; // boundaries arriving re-place everything
  const zone = scopeZone();
  const c = viewCentre.value;
  const byId = new Map(stations.map((s) => [s.id, s]));
  const dist = new Map<string, number>();
  for (const s of stations) if (c) dist.set(s.id, distFromCentre(s.lng, s.lat, c));
  const inZone = (s: StationInput | undefined) => !!s && stationZone(s) === zone;
  const outStations = zone ? stations.filter((s) => inZone(s) || keep("station", s.id)) : [...stations];
  const lineDist = new Map<string, number>();
  const outLines = lines.filter((l) => {
    let best = Infinity;
    for (const id of l.stops) best = Math.min(best, dist.get(id) ?? Infinity);
    if (best < Infinity) lineDist.set(l.id, best);
    return !zone || l.stops.length === 0 || keep("line", l.id) || l.stops.some((id) => inZone(byId.get(id)));
  });
  return { zone, stations: outStations, lines: outLines, stationDist: dist, lineDist, total: { stations: stations.length, lines: lines.length } };
}
