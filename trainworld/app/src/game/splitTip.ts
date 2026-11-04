// The hover of a mode split bar (T-550): the top bar's, and a row of the Money tab's Cities section.

import { count } from "./format";
import { modeColours } from "./palette";
import type { TripLengths } from "./tripLengths";
import type { ModeSplit } from "./types";
import { tipBullet, tipGrid, tipSpan, tipStrip } from "../ui/tip";

/** The split in rides (trip legs) a day, `all` their total (`modeRides`). */
export type SplitRides = ModeSplit & { all: number };

/** "Trips per day in New York on a weekday: Train 18.5% (8.81M), Walking ..., Driving ..., all 47.6M
 * trips". `where` is the phrase after "Trips per day" (" in New York on a weekday"), `note` what
 * follows (" (an early estimate)"). With no split yet every share is "–". */
export function splitTip(where: string, split: ModeSplit | null, rides: SplitRides | null, note: string): string {
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const pct = (v: number) => (split ? v.toFixed(1) + "%" : "–");
  // T-2215: No route only where some trips have none
  const none = (s.none ?? 0) > 0;
  const body = rides
    ? `Train ${pct(s.train)} (${count(rides.train)}), Walking ${pct(s.walk)} (${count(rides.walk)}), Driving ${pct(s.drive)} (${count(rides.drive)}), ${none ? `No route ${pct(s.none!)} (${count(rides.none ?? 0)}), ` : ""}all ${count(rides.all)} trips`
    : `Train ${pct(s.train)}, Walking ${pct(s.walk)}, Driving ${pct(s.drive)}${none ? `, No route ${pct(s.none!)}` : ""}`;
  return `Trips per day${where}: ${body}${note}`;
}

/**
 * T-2365: a zone's split as a small table (the top bar's and the city view's Train share hover):
 * "Per day in New York", the trips and, once the zone's trips by length are known, their
 * straight-line km, each a strip of the modes and its total, then the modes' shares of each. `where`
 * as `splitTip`'s.
 */
export function splitTable(where: string, split: ModeSplit | null, rides: SplitRides | null, lengths: TripLengths | null, note: string): string {
  const head = `Per day${where}${note}`;
  if (!split) return `${head}\nTrip count: –`;
  const c = modeColours();
  // T-2215's No route where some trips have none, here where its share shows as 0.1% or more
  const none = (split.none ?? 0) >= 0.05;
  const modes: { name: string; colour: string; trips: number; km: (t: TripLengths) => Float64Array }[] = [
    { name: "Train", colour: c.train, trips: split.train, km: (t) => t.km.rail },
    { name: "Walking", colour: c.walk, trips: split.walk, km: (t) => t.km.walk },
    { name: "Driving", colour: c.drive, trips: split.drive, km: (t) => t.km.drive },
    ...(none ? [{ name: "No route", colour: c.none, trips: split.none ?? 0, km: (t: TripLengths) => t.km.none }] : []),
  ];
  const sum = (a: Float64Array) => a.reduce((s, v) => s + Math.max(0, v), 0);
  const km = lengths ? modes.map((m) => sum(m.km(lengths))) : null;
  const kmAll = lengths ? sum(lengths.km.rail) + sum(lengths.km.walk) + sum(lengths.km.drive) + sum(lengths.km.none) : 0;
  const colours = modes.map((m) => m.colour);
  const pct = (v: number) => v.toFixed(1) + "%";
  const rows: string[][] = [
    ["Trip count", tipSpan(modes.length, tipStrip(colours, false, modes.map((m) => m.trips))), rides ? count(rides.all) : ""],
    ...(km && kmAll > 0 ? [["Distance", tipSpan(modes.length, tipStrip(colours, false, km)), `${count(kmAll)} km`]] : []),
    [""],
    ["", ...modes.map((m) => `${tipBullet("", m.colour)}${m.name}`), ""],
    ["Trip count", ...modes.map((m) => pct(m.trips)), ""],
    ...(km && kmAll > 0 ? [["Distance", ...km.map((v) => pct((100 * v) / kmAll)), ""]] : []),
  ];
  return `${head}\n${tipGrid(rows)}`;
}
