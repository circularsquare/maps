// The hover of a mode split bar (T-550): the top bar's, and a row of the Money tab's Cities section.

import { t, tNumber, tSeries } from "../i18n";
import { transitParts, type TransitShares } from "./transitModes";
import { count } from "./format";
import { modeColours } from "./palette";
import type { TripLengths } from "./tripLengths";
import type { ModeSplit } from "./types";
import { tipBullet, tipGrid, tipSpan, tipStrip } from "../ui/tip";

const percent = (v: number) => tNumber(v, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "%";

/** The split in rides (trip legs) a day, `all` their total (`modeRides`). */
export type SplitRides = ModeSplit & { all: number };

/** "Trips per day in New York on a weekday: Train 18.5% (8.81M), Walking ..., Driving ..., all 47.6M
 * trips". `where` is the phrase after "Trips per day" and `note` what follows, each through `tSpaced`
 * (" in New York on a weekday", " (an early estimate)" in English). With no split yet every share is "–". */
export function splitTip(where: string, split: ModeSplit | null, rides: SplitRides | null, note: string, shares: TransitShares | null = null): string {
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const pct = (v: number) => (split ? percent(v) : "–");
  // T-2215: No route only where some trips have none
  const none = (s.none ?? 0) > 0;
  const v = { train: pct(s.train), walk: pct(s.walk), drive: pct(s.drive), none: pct(s.none ?? 0) };
  const body = rides
    ? t(none ? "splitTip.ridesNone" : "splitTip.rides", { ...v, trainN: count(rides.train), walkN: count(rides.walk), driveN: count(rides.drive), noneN: count(rides.none ?? 0), all: count(rides.all) })
    : t(none ? "splitTip.sharesNone" : "splitTip.shares", v);
  const line = t("splitTip.line", { where, body, note });
  return split && shares ? `${line}\n${transitLine(split.train, shares)}` : line;
}

/** T-3695: the transit share `train` (%) by kind of line, each a square of its colour and its share of all trips. */
export function transitLine(train: number, shares: TransitShares): string {
  // (a kind too small to show as 0.1% is in the bar but not named)
  return tSeries(transitParts(train, shares).filter((p) => p.n >= 0.05).map((p) => `${tipBullet("", p.colour)}${p.name} ${percent(p.n)}`));
}

/**
 * T-2365: a zone's split as a small table (the top bar's and the city view's Train share hover):
 * "Per day in New York", the trips and, once the zone's trips by length are known, their
 * straight-line km, each a strip of the modes and its total, then the modes' shares of each. `where`
 * as `splitTip`'s.
 */
export function splitTable(where: string, split: ModeSplit | null, rides: SplitRides | null, lengths: TripLengths | null, note: string, shares: TransitShares | null = null): string {
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
  // T-3695: the trip count's strip shows transit by kind of line, as the bar does (the distance's has no km by kind)
  const tripCells = shares ? [...transitParts(split.train, shares), ...modes.slice(1).map((m) => ({ colour: m.colour, n: m.trips }))] : modes.map((m) => ({ colour: m.colour, n: m.trips }));
  const rows: string[][] = [
    [t("splitTip.tripCount"), tipSpan(modes.length, tipStrip(tripCells.map((c) => c.colour), false, tripCells.map((c) => c.n))), rides ? count(rides.all) : ""],
    ...(km && kmAll > 0 ? [[t("splitTip.distance"), tipSpan(modes.length, tipStrip(colours, false, km)), `${count(kmAll)} km`]] : []),
    [""],
    ["", ...modes.map((m) => `${tipBullet("", m.colour)}${m.name}`), ""],
    [t("splitTip.tripCount"), ...modes.map((m) => pct(m.trips)), ""],
    ...(km && kmAll > 0 ? [[t("splitTip.distance"), ...km.map((v) => pct((100 * v) / kmAll)), ""]] : []),
  ];
  return `${head}\n${tipGrid(rows)}${shares ? `\n${transitLine(split.train, shares)}` : ""}`;
}
