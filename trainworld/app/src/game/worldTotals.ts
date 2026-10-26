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

/** T-1301: the world's rides a day by every mode the split counts (train, walking, driving, the
 * last holding buses and the rest), fixed, so the world's train share is of all of them: 8.2
 * billion people (UN World Population Prospects 2024, for 2025) at 3 rides a person a day
 * (household travel surveys mostly find 2.5 to 4 trips a person a day), rounded to 25 billion. */
export const WORLD_PEOPLE = 8.2e9;
export const RIDES_PER_PERSON = 3;
export const WORLD_RIDES = 25e9;

/** The world's split: train rides over `WORLD_RIDES`, the rest walking and driving in the
 * proportions of the cities' own. */
export function worldSplit(rides: SplitRides): ModeSplit {
  const train = Math.min(100, (100 * rides.train) / WORLD_RIDES);
  const rest = rides.walk + rides.drive;
  return rest > 0
    ? { train, walk: ((100 - train) * rides.walk) / rest, drive: ((100 - train) * rides.drive) / rest }
    : { train, walk: 0, drive: 100 - train };
}

/** A share in percent to two significant figures under 1% ("0.43%", "0.043%"), else one decimal. */
export function smallPct(v: number): string {
  if (v >= 1 || v <= 0) return v.toFixed(1) + "%";
  return v.toFixed(Math.min(6, 1 - Math.floor(Math.log10(v)))) + "%";
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

/** The world's figures for the top bar: the sums over `builtCities()` with the day's demand views
 * (`citySplit` their own split), the world's split from them, and the counts the name's hover gives. */
export function worldTotals() {
  const built = builtCities(), views = cityDemandViews.value;
  let stations = 0;
  for (const n of built.values()) stations += n;
  const sum = sumModes([...built.keys()].map((id) => views.get(id)));
  return { cities: built.size, stations, split: sum ? worldSplit(sum.rides) : null, citySplit: sum?.split ?? null, rides: sum?.rides ?? null };
}
