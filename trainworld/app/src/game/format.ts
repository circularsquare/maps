// Number and time formatting for the UI. "−" (U+2212) for negatives, as in the mock.

import type { Level } from "./types";
import { display } from "./ui";
import { language, t, tCompact, tDuration, tList, tNumber } from "../i18n";

// one formatter each: `toLocaleString` makes a new one per call, 5-10 µs, which a panel of a few
// thousand rows felt (T-1165)
const WHOLE = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const UP_TO_3 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

/** To the whole number with thousands commas: 1234567.4 -> "1,234,567". */
export const exact = (n: number): string => WHOLE.format(Math.round(n));

/** A forecast count, rounded so it does not claim more precision than a forecast has: 12,430 -> "12,400". */
export const forecast = (n: number): string => exact(n >= 1000 ? Math.round(n / 100) * 100 : n >= 100 ? Math.round(n / 10) * 10 : n);

/** "1 car", "2 cars"; `many` is `one` and an s unless given. */
export const plural = (n: number, one: string, many = one + "s"): string => `${UP_TO_3.format(n)} ${n === 1 ? one : many}`;

/** ["a", "b", "c"] -> "a, b and c" */
export const andList = (names: string[]): string => tList(names);

export const capitalise = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

/** A short figure: `digits` of `scale` with English's `unit` ("1.84", 1e6, "M"), or in another language
 * the same value to as many significant digits in its own form ("184万", T-2672). */
export function compactOf(digits: string, scale: number, unit: string): string {
  if (language.value === "en") return digits + unit;
  return tCompact(Number(digits) * scale, digits.replace(/\D/g, "").replace(/^0+/, "").length);
}

/** 312000 -> "312k", 1.84e6 -> "1.84M"; "31.2万", "184万" in Japanese */
export function count(n: number): string {
  if (n >= 1e6) return compactOf((n / 1e6).toFixed(n >= 1e7 ? 1 : 2).replace(/\.?0+$/, ""), 1e6, "M");
  if (n >= 1e3) return compactOf(String(Math.round(n / 1000)), 1e3, "k");
  return String(Math.round(n));
}

/** Three significant figures from a million up (T-722): 22430000 -> "$22.4M", 1.153e9 -> "$1.15B",
 * 233e6 -> "$233M"; thousands to the k; under $500 "$0". */
export function money(n: number): string {
  const a = Math.abs(n);
  if (a < 500) return "$0";
  return moneyShort(n);
}

/** Four significant figures, for the cash (T-722): "$11.36B", "$893.6M", trailing zeros dropped ("$6B");
 * to the dollar under a million. */
export function moneyCash(n: number): string {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  if (a < 999.95e3) return moneyExact(n);
  const sig = (v: number) => String(Number(v.toPrecision(4)));
  return a >= 999.95e6 ? s + "$" + compactOf(sig(a / 1e9), 1e9, "B") : s + "$" + compactOf(sig(a / 1e6), 1e6, "M");
}

/** A price in millions, to a tenth, as the Build tab quotes it: 90 -> "$90M", 1.5 -> "$1.5M"; "$150万" in Japanese. */
export const millions = (v: number): string =>
  "$" + (language.value === "en" ? tNumber(v, { maximumFractionDigits: 1 }) + "M" : compactOf(String(Number(v.toFixed(1))), 1e6, "M"));

/** A range of whole millions: 10, 55 -> "$10-55M"; "$1000万-5500万" in Japanese. */
export const millionsRange = (lo: number, hi: number): string =>
  language.value === "en" ? `$${lo}-${hi}M` : `$${compactOf(String(lo), 1e6, "M")}-${compactOf(String(hi), 1e6, "M")}`;

/** Whether `money(n)` shows a minus, for the panels' red (T-722). */
export const moneyNegative = (n: number): boolean => n < 0 && Math.abs(n) >= 500;

/** To the dollar, for the cash (T-284: it moves as fares land): 5834215320 -> "$5,834,215,320". */
export function moneyExact(n: number): string {
  return (n < 0 ? "−" : "") + "$" + exact(Math.abs(n));
}

/** Three significant figures: 25.5, 133, 1.33. */
export const sig3 = (v: number): string => {
  const r = Number(v.toPrecision(3));
  return r >= 100 ? String(r) : r >= 10 ? r.toFixed(1) : r.toFixed(2);
};

/** Three significant figures, for the drawing tag (T-085): "$25.5M", "$133M", "$1.33B". */
export function moneyShort(n: number): string {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  // a figure that rounds up to the next unit (999.6k, 999.7M) is written in that unit
  if (a >= 999.5e6) return s + "$" + compactOf(sig3(a / 1e9), 1e9, "B");
  if (a >= 999.5e3) return s + "$" + compactOf(sig3(a / 1e6), 1e6, "M");
  return s + "$" + compactOf(String(Math.round(a / 1000)), 1e3, "k");
}

// Distances, lengths and speeds shown to the player (T-381). Every figure in the game is metric;
// these are the only places that turn one into miles, feet or mph, by the US units setting. Read
// while rendering, so a component that calls one redraws when the setting changes.

const M_PER_MILE = 1609.344;
const FT_PER_M = 3.280839895;

const us = () => display.value.usUnits;
const trim = (v: number, dp: number) => String(Number(v.toFixed(dp)));

/** 1520 -> "1.5 km" or "0.94 mi", 350 -> "0.35 km" or "0.22 mi" */
export function distance(m: number): string {
  if (us()) {
    const mi = m / M_PER_MILE;
    return (mi >= 1 ? mi.toFixed(1) : mi.toFixed(2)) + " mi";
  }
  return (m >= 1000 ? (m / 1000).toFixed(1) : (m / 1000).toFixed(2)) + " km";
}

/** A long length with thousands separators, whole above 100: "4,499 km", "62.4 km". */
export function bigDistance(m: number): string {
  const v = m / (us() ? M_PER_MILE : 1000);
  return `${tNumber(v, { maximumFractionDigits: v >= 100 ? 0 : 1 })} ${us() ? "mi" : "km"}`;
}

/** A position along something and its whole length: "1.2 of 5.4 km". */
export function distanceOf(part: number, whole: number): string {
  const u = us() ? M_PER_MILE : 1000;
  return t("format.distanceOf", { part: (part / u).toFixed(1), whole: `${(whole / u).toFixed(1)} ${us() ? "mi" : "km"}` });
}

/** A limit in a rule: 1000 -> "1 km" or "0.62 mi". */
export function roughDistance(m: number): string {
  return us() ? `${trim(m / M_PER_MILE, 2)} mi` : `${trim(m / 1000, 1)} km`;
}

/** Whole units, at least one, for "3 km north-east of" place names. */
export function wholeDistance(m: number): string {
  return us() ? `${Math.max(1, Math.round(m / M_PER_MILE))} mi` : `${Math.max(1, Math.round(m / 1000))} km`;
}

/** Short lengths to the unit: platforms, curve radii, a walk. 1250 -> "1,250 m" or "4,100 ft"; feet to the nearest 10. */
export function length(m: number): string {
  const v = us() ? Math.round((m * FT_PER_M) / 10) * 10 : Math.round(m);
  return `${tNumber(v)} ${us() ? "ft" : "m"}`;
}

/** A floor's area, three significant figures: 90700 -> "90,700 m²" or "976,000 sq ft" (T-2218). */
export function area(m2: number): string {
  const v = us() ? m2 * FT_PER_M * FT_PER_M : m2;
  return `${exact(Number(v.toPrecision(3)))} ${us() ? "sq ft" : "m²"}`;
}

/** A price per area, US$ a m²: 2000 -> "$2,000 per m²" or "$186 per sq ft". */
export function perArea(usdPerM2: number): string {
  return us() ? t("format.perSqFt", { amount: "$" + exact(usdPerM2 / (FT_PER_M * FT_PER_M)) }) : t("format.perM2", { amount: "$" + exact(usdPerM2) });
}

/** A distance from here: metres to the nearest 10, then kilometres; feet, then miles. */
export function shortDistance(m: number): string {
  if (us()) {
    const ft = m * FT_PER_M, mi = m / M_PER_MILE;
    if (ft < 528) return `${Math.round(ft / 10) * 10} ft`;
    return mi < 1 ? `${mi.toFixed(2)} mi` : mi < 10 ? `${mi.toFixed(1)} mi` : `${exact(mi)} mi`;
  }
  return m < 1000 ? `${Math.round(m / 10) * 10} m` : m < 10_000 ? `${(m / 1000).toFixed(1)} km` : `${exact(m / 1000)} km`;
}

/** 160 -> "160 km/h" or "99 mph"; `nearest` rounds in the shown unit (the speed key uses 5). */
export function speed(kmh: number, nearest = 1): string {
  const v = us() ? kmh / M_PER_MILE * 1000 : kmh;
  return `${Math.round(v / nearest) * nearest} ${us() ? "mph" : "km/h"}`;
}

/** Round lengths a scale bar can show, shortest first, in metres with their labels. */
export function scaleBars(): [number, string][] {
  if (us()) return [50, 100, 200, 500, 1000, 2000, 5280].map((ft) => [ft / FT_PER_M, ft === 5280 ? `${tNumber(1)} mi` : `${tNumber(ft)} ft`]);
  return [10, 20, 50, 100, 200, 500, 1000].map((m) => [m, m >= 1000 ? `${m / 1000} km` : `${m} m`]);
}

/** The unit a per-distance price is quoted in. */
export function distUnit(): string {
  return t("format.distUnit", { us: us() });
}

/** A price or quantity per km, per shown unit. */
export function perDistance(perKm: number): number {
  return us() ? perKm * (M_PER_MILE / 1000) : perKm;
}

const perUnitText = (perKm: number) => {
  const v = +perDistance(perKm).toFixed(2);
  return "$" + (Number.isInteger(v) ? v : v.toFixed(2));
};
/** T-807: "$9 per train-km plus $2 per car-km", in the shown unit. */
export function runCostRule(m: { runCostTrainKm: number; runCostCarKm: number }): string {
  return t("format.runCostRule", { train: perUnitText(m.runCostTrainKm), car: perUnitText(m.runCostCarKm), unit: distUnit() });
}
type UpkeepRates = { upkeepTrackKm: number; upkeepViaductKm: number; upkeepCutKm: number; upkeepBoredKm: number; upkeepStation: number; upkeepStationUnder: number; economy: number };
/** T-2676: an upkeep rate over a viaduct's, "0.7x" (the level whose cost multiple is 1.0x too). */
const upkeepMult = (v: number, m: UpkeepRates) => tNumber(m.upkeepViaductKm > 0 ? v / m.upkeepViaductKm : 1, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "x"; // i18n-ignore
/** T-867, T-1931, T-2619, T-2676: "$50k per km of viaduct per day, 0.7x that on the ground, 2.0x in cut and cover and 2.6x in a bored tunnel, plus $50k per station and $200k per underground station per day", at the game's economy, in the shown unit. */
export function upkeepRule(m: UpkeepRates): string {
  const km = (v: number) => money(perDistance(v * m.economy));
  return t("format.upkeepRule", { viaduct: km(m.upkeepViaductKm), unit: distUnit(), ground: upkeepMult(m.upkeepTrackKm, m), cut: upkeepMult(m.upkeepCutKm, m), bored: upkeepMult(m.upkeepBoredKm, m), station: money(m.upkeepStation * m.economy), under: money(m.upkeepStationUnder * m.economy) });
}
/** T-1838, T-1931, T-2619, T-2676: the Upkeep row's hover, seven lines: "$50k per km of viaduct per day", "Ground 0.7x", "Cut and cover 2.0x", "Bored tunnel 2.6x", "$50k per station per day", "$200k underground", "High speed rail 1.5x". */
export function upkeepLines(m: UpkeepRates & { upkeepHsr: number }): string {
  const km = (v: number) => money(perDistance(v * m.economy));
  return t("format.upkeepLines", { viaduct: km(m.upkeepViaductKm), unit: distUnit(), ground: upkeepMult(m.upkeepTrackKm, m), cut: upkeepMult(m.upkeepCutKm, m), bored: upkeepMult(m.upkeepBoredKm, m), station: money(m.upkeepStation * m.economy), under: money(m.upkeepStationUnder * m.economy), hsr: m.upkeepHsr });
}
/** T-2006, T-2676: the elevation hovers' line, "0.7x upkeep"; `mult` is the level's upkeep over a viaduct's, written (`ui/trackPrice.ts` `upkeepAt`). */
export function levelUpkeepLine(mult: string): string {
  return t("format.levelUpkeepLine", { mult });
}
/** T-1838: the Running trains row's hover: "Trains cost $4 per km, plus $2 per train car". */
export function runCostLine(m: { runCostTrainKm: number; runCostCarKm: number }): string {
  return t("format.runCostLine", { train: perUnitText(m.runCostTrainKm), unit: distUnit(), car: perUnitText(m.runCostCarKm) });
}
/** T-1893: the train length's hover line: "Running a train costs $4 per km, plus $2 per car". */
export function trainCostLine(m: { runCostTrainKm: number; runCostCarKm: number }): string {
  return t("format.trainCostLine", { train: perUnitText(m.runCostTrainKm), unit: distUnit(), car: perUnitText(m.runCostCarKm) });
}
/** A multiple to a hundredth, "0.39x" (T-2669). */
export const factorText = (f: number) => tNumber(f, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + "x"; // i18n-ignore
/** T-2669: the running cost rule's line for a country priced off the world list, "Running costs in Mexico: 0.39x". */
export const runCostHere = (name: string, f: number) => t("format.runCostHere", { name, factor: factorText(f) });
/** T-2669: a line's running costs over the world list, "Running costs on this line: 0.39x". */
export const runCostLineFactor = (f: number) => t("format.runCostLineFactor", { factor: factorText(f) });
/** The upkeep rule as a sentence; `lead` is "Upkeep" or "Once constructed, upkeep". */
export function upkeepText(m: UpkeepRates & { upkeepHsr: number; upkeepSingle: number }, lead: string): string {
  return t("format.upkeepText", { lead, rule: upkeepRule(m), hsr: m.upkeepHsr, single: m.upkeepSingle });
}
type Fleets = { carPrice: number; runCostTrainKm: number; runCostCarKm: number; hsCarPrice: number; hsRunCostTrainKm: number; hsRunCostCarKm: number; fleet: number; carsNeeded: number; fleetFast: number; carsNeededFast: number };
/** One kind of train's car price, running costs and spare cars (T-817: high-speed or ordinary). */
export function trainKind(m: Fleets, fast: boolean): { carPrice: number; runCostTrainKm: number; runCostCarKm: number; spare: number } {
  return fast
    ? { carPrice: m.hsCarPrice, runCostTrainKm: m.hsRunCostTrainKm, runCostCarKm: m.hsRunCostCarKm, spare: Math.max(0, m.fleetFast - m.carsNeededFast) }
    : { carPrice: m.carPrice, runCostTrainKm: m.runCostTrainKm, runCostCarKm: m.runCostCarKm, spare: Math.max(0, m.fleet - m.carsNeeded) };
}

export const busesText = (n: number) => t("format.buses", { n });
/** T-1464: "$3 per bus-km", in the shown unit. */
export const busRunRule = (m: { runCostBusKm: number }) => t("format.busRunRule", { amount: perUnitText(m.runCostBusKm), unit: distUnit() });
/** T-1464: a route's price on Construct, "$1M per km of route". */
export const busRouteRule = (m: { busRouteKm: number }) => t("format.busRouteRule", { amount: money(perDistance(m.busRouteKm * 1e6)), unit: distUnit() });
export const boatsText = (n: number) => t("format.boats", { n });
/** T-2194: "$40 per boat-km", in the shown unit. */
export const boatRunRule = (m: { runCostBoatKm: number }) => t("format.boatRunRule", { amount: perUnitText(m.runCostBoatKm), unit: distUnit() });

/** Mile, and the metres a ruler marking it stands for (the placing rings). */
export const unitMetres = () => (us() ? M_PER_MILE : 1000);
export const unitShort = () => (us() ? "mi" : "km");

/** headway in minutes from trains an hour: "5.5 min", "5.5分" */
export function headway(tph: number): string {
  if (!tph) return t("format.headwayNone");
  const m = 60 / tph;
  return t("inspector.min", { n: Number.isInteger(m) ? String(m) : m.toFixed(1) });
}

/** Whole minutes, at least one: "7 min", "7分". */
export const minutesText = (m: number): string => t("inspector.min", { n: Math.max(1, Math.round(m)) });

/** trains needed to run `tph` on a round trip of `rtS` seconds */
export function trainsNeeded(rtS: number, tph: number): number {
  return Math.ceil((rtS * tph) / 3600);
}

/** 83 -> "1m23s", 4330 -> "1h12m"; "1分23秒", "1時間12分" in Japanese */
export const duration = (s: number): string => tDuration(s);

export function clockTime(minute: number): string {
  const m = minute % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export function dayOf(minute: number): number {
  return Math.floor(minute / 1440);
}

export function levelLabel(l: number): string {
  return l > 0 ? "+" + l : l < 0 ? "−" + -l : "0";
}

const LEVEL_NAMES: Record<Level, () => string> = {
  3: () => t("format.levelHighViaduct"), 2: () => t("format.levelViaduct"), 1: () => t("format.levelViaduct"), 0: () => t("format.levelGround"),
  [-1]: () => t("format.levelCutCover"), [-2]: () => t("format.levelTunnel"), [-3]: () => t("format.levelDeepTunnel"),
};
export const levelName = (l: Level): string => LEVEL_NAMES[l]();
