// Attractions (T-263, docs/demand-sources.md 4.5): museums, zoos, theme parks, temples, castles and
// the like from Wikidata, each with its visitors a year (attractions.json, pipeline/attractions_json.py),
// fetched as its own chunk. The demand workers hand each pack's to its everyday layer
// (`EverydayApi.beside_with`); the map marks them (map/attractionMarks.ts). Plain data: the demand
// workers import it too.

import { AVERAGE_DAY, AWAY_DAY, type DemandDay } from "./days";

export interface Attraction {
  /** the Wikidata item's number (Q-id without the Q) */
  id: number;
  name: string;
  lng: number;
  lat: number;
  /** visitors a year: Wikidata's figure, or 10,000 for each language edition writing about it */
  visitors: number;
}

type Row = [number, string, number, number, number, number];

let data: Promise<Attraction[]> | null = null;

/** Every attraction in the world, largest first; empty if the file does not load. */
export function loadAttractions(): Promise<Attraction[]> {
  return (data ??= import("./attractions.json").then(
    (m) => ((m.default as unknown as { rows: Row[] }).rows).map(([id, name, lng, lat, visitors]) => ({ id, name, lng, lat, visitors })),
    () => { data = null; return []; },
  ));
}

/** Social outings' Saturday and Sunday per weekday (sim/src/demand/days.rs `DAY_RATE`), which
 * attractions take. */
const WEEKEND = [1.54, 1.67];
/** A superholiday's away day per Sunday (days.rs `AWAY_RATE`, T-264). */
const AWAY = 1.5;

/** Visits a day to an attraction of `visitors` a year on `day`: the year's over 365 days, set on
 * weekdays so the weekend's factors bring a week back to them (as the sim's `attraction_sites`). */
export function visitsOn(visitors: number, day: DemandDay): number {
  const weekday = (visitors * 7) / 365 / (5 + WEEKEND[0] + WEEKEND[1]);
  return day === 0 ? weekday : day === AVERAGE_DAY ? visitors / 365 : day === AWAY_DAY ? weekday * WEEKEND[1] * AWAY : weekday * WEEKEND[day - 1];
}
