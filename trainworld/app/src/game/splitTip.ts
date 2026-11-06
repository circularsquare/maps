// The hover of a mode split bar (T-550): the top bar's, and a row of the Money tab's Cities section.

import { t, tNumber } from "../i18n";
import { count } from "./format";
import { modeColours } from "./palette";
import type { TripLengths } from "./tripLengths";
import type { ModeSplit } from "./types";
import { tipBullet, tipGrid, tipSpan, tipStrip } from "../ui/tip";

const percent = (v: number) => tNumber(v, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "%";

/** The split in rides (trip legs) a day, `all` their total (`modeRides`). */
export type SplitRides = ModeSplit & { all: number };

/** "Trips per day in New York on a weekday: Train 18.5% (8.81M), Walking ..., Driving ..., all 47.6M
 * trips". `where` is the phrase after "Trips per day" (" in New York on a weekday"), `note` what
 * follows (" (an early estimate)"). With no split yet every share is "–". */
export function splitTip(where: string, split: ModeSplit | null, rides: SplitRides | null, note: string): string {
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const pct = (v: number) => (split ? percent(v) : "–");
  // T-2215: No route only where some trips have none
  const none = (s.none ?? 0) > 0;
  const v = { train: pct(s.train), walk: pct(s.walk), drive: pct(s.drive), none: pct(s.none ?? 0) };
  const body = rides
    ? t(none ? "splitTip.ridesNone" : "splitTip.rides", { ...v, trainN: count(rides.train), walkN: count(rides.walk), driveN: count(rides.drive), noneN: count(rides.none ?? 0), all: count(rides.all) })
    : t(none ? "splitTip.sharesNone" : "splitTip.shares", v);
  return t("splitTip.line", { where, body, note });
}

/**
 * T-2365: a zone's split as a small table (the top bar's and the city view's Train share hover):
 * "Per day in New York", the trips and, once the zone's trips by length are known, their
 * straight-line km, each a strip of the modes and its total, then the modes' shares of each. `where`
 * as `splitTip`'s.
 */
export function splitTable(where: string, split: ModeSplit | null, rides: SplitRides | null, lengths: TripLengths | null, note: string): string {
  const head = t("splitTip.perDay", { where, note });
  if (!split) return t("splitTip.noSplit", { head });
  const c = modeColours();
  // T-2215's No route where some trips have none, here where its share shows as 0.1% or more
  const none = (split.none ?? 0) >= 0.05;
  const modes: { name: string; colour: string; trips: number; km: (t: TripLengths) => Float64Array }[] = [
    { name: t("splitTip.train"), colour: c.train, trips: split.train, km: (l) => l.km.rail },
    { name: t("splitTip.walking"), colour: c.walk, trips: split.walk, km: (l) => l.km.walk },
    { name: t("splitTip.driving"), colour: c.drive, trips: split.drive, km: (l) => l.km.drive },
    ...(none ? [{ name: t("splitTip.noRoute"), colour: c.none, trips: split.none ?? 0, km: (l: TripLengths) => l.km.none }] : []),
  ];
  const sum = (a: Float64Array) => a.reduce((s, v) => s + Math.max(0, v), 0);
  const km = lengths ? modes.map((m) => sum(m.km(lengths))) : null;
  const kmAll = lengths ? sum(lengths.km.rail) + sum(lengths.km.walk) + sum(lengths.km.drive) + sum(lengths.km.none) : 0;
  const colours = modes.map((m) => m.colour);
  const pct = percent;
  const rows: string[][] = [
    [t("splitTip.tripCount"), tipSpan(modes.length, tipStrip(colours, false, modes.map((m) => m.trips))), rides ? count(rides.all) : ""],
    ...(km && kmAll > 0 ? [[t("splitTip.distance"), tipSpan(modes.length, tipStrip(colours, false, km)), `${count(kmAll)} km`]] : []),
    [""],
    ["", ...modes.map((m) => `${tipBullet("", m.colour)}${m.name}`), ""],
    [t("splitTip.tripCount"), ...modes.map((m) => pct(m.trips)), ""],
    ...(km && kmAll > 0 ? [[t("splitTip.distance"), ...km.map((v) => pct((100 * v) / kmAll)), ""]] : []),
  ];
  return `${head}\n${tipGrid(rows)}`;
}
