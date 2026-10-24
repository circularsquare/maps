// The world as one place (T-1110): the sum of every city the player has a station in that has a
// demand view. The Cities tab's first row and the top bar's name when the map is zoomed far out
// both read it, so they cannot disagree.

import { signal } from "@preact/signals";
import { cityDemandViews, type DemandView } from "./demand";
import { hashView, type CityId } from "./cities";
import { stationZone } from "./listScope";
import { world } from "./world";
import type { ModeSplit } from "./types";
import type { SplitRides } from "./splitTip";

/** Below this map zoom (a few countries on screen) the top bar names the world, not a city. */
export const WORLD_ZOOM = 5;

/** The map is zoomed out past `WORLD_ZOOM`; set by map/zoneFocus.ts. */
export const worldView = signal((() => { const v = hashView(); return !!v && v.zoom < WORLD_ZOOM; })());

/** The mode split of the cities' trips added up, in rides a day: null until one has a view. `train`
 * of `rides` is also the train rides a day. */
export function sumModes(views: Iterable<DemandView | undefined>): { split: ModeSplit; rides: SplitRides } | null {
  let all = 0, train = 0, walk = 0, drive = 0, seen = false;
  for (const v of views) {
    if (!v) continue;
    const a = v.tripsPerDay;
    seen = true;
    all += a; train += v.railTripsPerDay; walk += a * v.split.walk / 100; drive += a * v.split.drive / 100;
  }
  if (!seen || all <= 0) return null;
  return { split: { train: 100 * train / all, walk: 100 * walk / all, drive: 100 * drive / all }, rides: { all, train, walk, drive } };
}

/** Every city with a built station, and how many stations it has. */
export function builtCities(): Map<CityId, number> {
  const built = new Map<CityId, number>();
  for (const s of world.value?.save.stations ?? []) {
    if (!s.built) continue;
    const z = stationZone(s);
    if (z) built.set(z, (built.get(z) ?? 0) + 1);
  }
  return built;
}

/** The world's figures for the top bar: the sums over `builtCities()` with the day's demand views,
 * and the counts the name's hover gives. */
export function worldTotals() {
  const built = builtCities(), views = cityDemandViews.value;
  let stations = 0;
  for (const n of built.values()) stations += n;
  const sum = sumModes([...built.keys()].map((id) => views.get(id)));
  return { cities: built.size, stations, split: sum?.split ?? null, rides: sum?.rides ?? null };
}
