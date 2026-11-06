// Small text helpers shared by the trip and demand panels.

import { capitalise } from "../../game/format";
import { t, tNumber, type MessageKey } from "../../i18n";
import { purposeWords, type PurposeFilter } from "../../game/purposes";
import type { TripLeg } from "../../game/trip";
import { wholePeople } from "../../game/trip";

/** `v` to a fixed number of decimals in the language's digits, with no thousands separator (these are
 * minutes and weights, never counts). */
const fixed = (v: number, min: number, max: number): string => tNumber(v, { minimumFractionDigits: min, maximumFractionDigits: max, useGrouping: false });

/** To one decimal place, a trailing .0 dropped. */
export const num1 = (v: number): string => fixed(v, 0, 1);

/** To one decimal place always: "3.0". */
export const fix1 = (v: number): string => fixed(v, 1, 1);

/** A dollar figure to the cent: "$2.90". */
export const usd = (v: number): string => "$" + fixed(v, 2, 2);

/** A share of 0 to 1 as a percent to one decimal: "12.3%". */
export const pct1 = (share: number): string => tNumber(share, { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });

/** A share of 0 to 1 as a whole percent: "45%". */
export const pctWhole = (share: number): string => tNumber(share, { style: "percent", maximumFractionDigits: 0 });
const underOne = (): string => "<" + pctWhole(0.01);

/** A share of 0 to 1 as a whole percent; "<1%" for one too small to round to a percent. */
export const pctShare = (v: number): string => (v > 0 && v < 0.005 ? underOne() : pctWhole(Math.round(100 * v) / 100));

/** `n` of `of` as a whole percent; "<1%" for a count too small to round to one. */
export const pctOf = (n: number, of: number): string => {
  const p = of > 0 ? (100 * n) / of : 0;
  return n >= 0.5 && p < 0.5 ? underOne() : pctWhole(Math.round(p) / 100);
};

/** Minutes, at least one: "6 min". */
export const minText = (m: number): string => t("inspector.min", { n: Math.max(1, Math.round(m)) });

/** Minutes to the nearest whole one, zero allowed: "0 min". */
export const minRound = (m: number): string => t("inspector.min", { n: Math.round(m) });

/** Minutes to one decimal, a trailing .0 dropped: "3.5 min". */
export const minNum1 = (m: number): string => t("inspector.min", { n: num1(m) });

/** Minutes to one decimal always: "3.0 min". */
export const minFix1 = (m: number): string => t("inspector.min", { n: fix1(m) });

/** T-2215: a trip between two land parts no road joins (T-2214) has no car and no walk. */
export const noRoad = (): string => t("tripText.noRoad");

const weight = (v: number) => fixed(v, 0, 2);

/** The modes the trip panels name, by id. */
const MODE_KEYS = {
  train: "tripText.modeTrain", walk: "tripText.modeWalk", walking: "tripText.modeWalking", drive: "tripText.modeDrive",
  none: "tripText.modeNone", plane: "tripText.modePlane", car: "tripText.modeCar",
} as const satisfies Record<string, MessageKey>;
export type ModeId = keyof typeof MODE_KEYS;
export const modeName = (m: ModeId): string => t(MODE_KEYS[m]);

/** The words of an effective time equation, by the id the model gives a term (game/trip.ts `routeTerms`)
 * or the panels add. */
const TERM_KEYS: Record<string, MessageKey> = {
  "starting the drive": "tripText.termStartingTheDrive",
  drive: "tripText.termDrive",
  "drop-off": "tripText.termDropOff",
  "parking at the station": "tripText.termParkingAtStation",
  "parking at home": "tripText.termParkingAtHome",
  car: "tripText.termCar",
  walk: "tripText.termWalk",
  wait: "tripText.termWait",
  "waiting for a train with room": "tripText.termWaitingForRoom",
  ride: "tripText.termRide",
  crowding: "tripText.termCrowding",
  "walk between stations": "tripText.termWalkBetween",
  changes: "tripText.termChanges",
  change: "tripText.termChange",
  fare: "tripText.termFare",
  "transfer fare": "tripText.termTransferFare",
  "choice of routes": "tripText.termChoiceOfRoutes",
  parking: "tripText.termParking",
  ownership: "tripText.termOwnership",
  running: "tripText.termRunning",
  "crossing fare": "tripText.termCrossingFare",
  "crossing time": "tripText.termCrossingTime",
  "airport drives": "tripText.termAirportDrives",
  "at the airports": "tripText.termAtAirports",
  "until the flight": "tripText.termUntilFlight",
  flight: "tripText.termFlight",
};
export const termLabel = (id: string): string => (TERM_KEYS[id] ? t(TERM_KEYS[id]) : id);

/** One term of an effective time equation: "walk 8 × 1.5", "wait 3", "fare $2.90 × 3". `label` is
 * already worded (`termLabel`). */
export interface Term {
  label: string;
  /** the real amount, already formatted ("" where the term is a count of its own minutes) */
  real: string;
  weight: number;
  perceived: number;
}

/** "Walk 8 + wait 3 + ride 6 = 17 min": the terms' text and the total. */
export const sumLine = (terms: string, total: number): string => t("tripText.equation", { terms: capitalise(terms), total: minRound(total) });

/** The effective time hover (T-699): only the equation, in the order of the rows, "Walk 8 × 1.5 +
 * wait 3 + ride 6 = 21 min". The total is the sum of the terms' unrounded minutes. */
export function equation(terms: Term[], total = terms.reduce((s, x) => s + x.perceived, 0)): string {
  const text = terms.map((x, i) => {
    const minus = x.perceived < 0;
    const body = `${x.label}${x.real ? " " + x.real : " " + num1(Math.abs(x.perceived))}${x.real && Math.abs(x.weight - 1) >= 0.005 ? ` × ${weight(x.weight)}` : ""}`;
    return i === 0 ? body : `${minus ? "−" : "+"} ${body}`;
  }).join(" ");
  return sumLine(text, total);
}

/** T-911: shares as whole people out of `n` (null or 0: the model's own shares). */
export const roundTo = (shares: number[], n: number | null, pooled = -1): number[] => (n ? wholePeople(shares, n, pooled).map((p) => p / n) : shares);

const kind = (f: PurposeFilter) => (f === "all" ? "all" : "purpose");

/** "trips", "errands trips": the purpose before the word. */
export const tripWords = (f: PurposeFilter): string => t("tripText.tripsWords", { kind: kind(f), words: f === "all" ? "" : purposeWords(f) });

/** "312k trips", "1 errands trip": a count and the word, `amount` already written. */
export const tripsAmount = (amount: string, n: number, f: PurposeFilter): string =>
  t("tripText.tripsAmount", { kind: kind(f), words: f === "all" ? "" : purposeWords(f), amount, count: Math.round(n) });

/** The hover of a ride in a route: its wait, its ride and its stops, in minutes to one decimal. `full`
 * is the wait for a train with room (T-842), `late` the extra felt wait of trips starting when no
 * train is due for hours; both 0 where the panel gives neither. */
export function rideTip(l: TripLeg, bus: boolean, full = 0, late = 0): string {
  const v = { bus, wait: minFix1(l.waitMin - full), full: full >= 0.05, fullMin: minFix1(full), ride: minFix1(l.min), stops: l.stops };
  return t("tripText.rideTip", v) + (late >= 0.05 ? t("tripText.rideTipLate", { bus, min: minFix1(late) }) : "");
}
