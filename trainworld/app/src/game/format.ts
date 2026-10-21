// Number and time formatting for the UI. "−" (U+2212) for negatives, as in the mock.

import { display } from "./ui";

/** 312000 -> "312k", 1.84e6 -> "1.84M" */
export function count(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2).replace(/\.?0+$/, "") + "M";
  if (n >= 1e3) return Math.round(n / 1000) + "k";
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
  return a >= 999.95e6 ? s + "$" + sig(a / 1e9) + "B" : s + "$" + sig(a / 1e6) + "M";
}

/** Whether `money(n)` shows a minus, for the panels' red (T-722). */
export const moneyNegative = (n: number): boolean => n < 0 && Math.abs(n) >= 500;

/** To the dollar, for the cash (T-284: it moves as fares land): 5834215320 -> "$5,834,215,320". */
export function moneyExact(n: number): string {
  return (n < 0 ? "−" : "") + "$" + Math.round(Math.abs(n)).toLocaleString("en-US");
}

/** Three significant figures, for the drawing tag (T-085): "$25.5M", "$133M", "$1.33B". */
export function moneyShort(n: number): string {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  const sig = (v: number) => {
    const r = Number(v.toPrecision(3));
    return r >= 100 ? String(r) : r >= 10 ? r.toFixed(1) : r.toFixed(2);
  };
  // a figure that rounds up to the next unit (999.6k, 999.7M) is written in that unit
  if (a >= 999.5e6) return s + "$" + sig(a / 1e9) + "B";
  if (a >= 999.5e3) return s + "$" + sig(a / 1e6) + "M";
  return s + "$" + Math.round(a / 1000) + "k";
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

/** A position along something and its whole length: "1.2 of 5.4 km". */
export function distanceOf(part: number, whole: number): string {
  const u = us() ? M_PER_MILE : 1000;
  return `${(part / u).toFixed(1)} of ${(whole / u).toFixed(1)} ${us() ? "mi" : "km"}`;
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
  return `${v.toLocaleString("en-US")} ${us() ? "ft" : "m"}`;
}

/** A distance from here: metres to the nearest 10, then kilometres; feet, then miles. */
export function shortDistance(m: number): string {
  if (us()) {
    const ft = m * FT_PER_M, mi = m / M_PER_MILE;
    if (ft < 528) return `${Math.round(ft / 10) * 10} ft`;
    return mi < 1 ? `${mi.toFixed(2)} mi` : mi < 10 ? `${mi.toFixed(1)} mi` : `${Math.round(mi).toLocaleString("en-US")} mi`;
  }
  return m < 1000 ? `${Math.round(m / 10) * 10} m` : m < 10_000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m / 1000).toLocaleString("en-US")} km`;
}

/** 160 -> "160 km/h" or "99 mph"; `nearest` rounds in the shown unit (the speed key uses 5). */
export function speed(kmh: number, nearest = 1): string {
  const v = us() ? kmh / M_PER_MILE * 1000 : kmh;
  return `${Math.round(v / nearest) * nearest} ${us() ? "mph" : "km/h"}`;
}

/** Round lengths a scale bar can show, shortest first, in metres with their labels. */
export function scaleBars(): [number, string][] {
  if (us()) return [50, 100, 200, 500, 1000, 2000, 5280].map((ft) => [ft / FT_PER_M, ft === 5280 ? "1 mi" : `${ft.toLocaleString("en-US")} ft`]);
  return [10, 20, 50, 100, 200, 500, 1000].map((m) => [m, m >= 1000 ? `${m / 1000} km` : `${m} m`]);
}

/** The unit a per-distance price is quoted in. */
export function distUnit(): string {
  return us() ? "mile" : "km";
}

/** A price or quantity per km, per shown unit. */
export function perDistance(perKm: number): number {
  return us() ? perKm * (M_PER_MILE / 1000) : perKm;
}

const perUnitText = (perKm: number) => {
  const v = +perDistance(perKm).toFixed(2);
  return "$" + (Number.isInteger(v) ? v : v.toFixed(2));
};
/** T-807: "$9 a train-km plus $2 a car-km", in the shown unit. */
export function runCostRule(m: { runCostTrainKm: number; runCostCarKm: number }): string {
  return `${perUnitText(m.runCostTrainKm)} a train-${distUnit()} plus ${perUnitText(m.runCostCarKm)} a car-${distUnit()}`;
}
/** What a train of `cars` costs to run: "$27 a train-km". */
export function trainRunCost(m: { runCostTrainKm: number; runCostCarKm: number }, cars: number): string {
  return `${perUnitText(m.runCostTrainKm + cars * m.runCostCarKm)} a train-${distUnit()}`;
}

/** The inverse of `perDistance`. */
export function perKmFrom(perUnit: number): number {
  return us() ? perUnit / (M_PER_MILE / 1000) : perUnit;
}

/** Mile, and the metres a ruler marking it stands for (the placing rings). */
export const unitMetres = () => (us() ? M_PER_MILE : 1000);
export const unitShort = () => (us() ? "mi" : "km");

/** headway in minutes from trains an hour */
export function headway(tph: number): string {
  if (!tph) return "none";
  const m = 60 / tph;
  return (Number.isInteger(m) ? m : m.toFixed(1)) + " min";
}

/** trains needed to run `tph` on a round trip of `rtS` seconds */
export function trainsNeeded(rtS: number, tph: number): number {
  return Math.ceil((rtS * tph) / 3600);
}

/** 83 -> "1m23s", 4330 -> "1h12m" */
export function duration(s: number): string {
  s = Math.round(s);
  if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor(s / 60) % 60).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

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
