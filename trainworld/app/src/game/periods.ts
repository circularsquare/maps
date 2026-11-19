// The five periods of the day (SPEC 4.2, 6.3): when each demand level is scheduled (high 6-20,
// medium 20-24, low 0-6), and the periods demand is solved for. The definition is `PERIOD_HOURS`
// and `PERIOD_ROW` in sim/src/track/params.rs; this is its copy for the main thread, which has
// no WASM. The demand
// workers report the Rust table when they start and the demand client checks the two agree
// (workers/demandClient.ts), so an edit to one without the other shows up as a console error.

import { t } from "../i18n";
import type { Demand } from "./types";

export interface Period {
  name: string;
  /** hours of the day, [from, to) */
  from: number;
  to: number;
  demand: Demand;
}

export const PERIODS: readonly Period[] = [
  { get name() { return t("periods.morningPeak"); }, from: 6, to: 10, demand: "high" },
  { get name() { return t("periods.midday"); }, from: 10, to: 16, demand: "high" },
  { get name() { return t("periods.eveningPeak"); }, from: 16, to: 20, demand: "high" },
  { get name() { return t("periods.evening"); }, from: 20, to: 24, demand: "medium" },
  { get name() { return t("periods.night"); }, from: 0, to: 6, demand: "low" },
];

const LEVEL_INDEX: Record<Demand, number> = { high: 0, medium: 1, low: 2 };

/** T-3542: each period's code for a message's select ("in the morning peak", "at midday", ...) */
export const PERIOD_CODE = ["morning", "midday", "afternoon", "evening", "night"] as const; // i18n-ignore (select codes)

/** T-3542: the rows a line of 3, 4 or 5 rows sets its trains in, in the day's order: each its code
 * (`schedule.rowName`) and its periods. Three rows are high, medium and low demand; four split midday
 * out of high (rush is the morning and evening peaks), five the afternoon too (sim params.rs
 * `ROW_PLACES`, `PERIOD_PLACE`). */
export interface Row { code: "high" | "medium" | "low" | "rush" | "morning" | "midday" | "afternoon" | "evening" | "night"; periods: number[] }
export const ROWS_OF: Record<3 | 4 | 5, Row[]> = {
  3: [{ code: "high", periods: [0, 1, 2] }, { code: "medium", periods: [3] }, { code: "low", periods: [4] }],
  4: [{ code: "rush", periods: [0, 2] }, { code: "midday", periods: [1] }, { code: "evening", periods: [3] }, { code: "night", periods: [4] }],
  5: [{ code: "morning", periods: [0] }, { code: "midday", periods: [1] }, { code: "afternoon", periods: [2] }, { code: "evening", periods: [3] }, { code: "night", periods: [4] }],
};
/** The rows of a line of `n` rows (3 where `n` is none of 3, 4, 5). */
export const rowsOf = (n: number): Row[] => ROWS_OF[(n === 4 || n === 5 ? n : 3) as 3 | 4 | 5];
/** The row of a line of `n` rows that period `p` is in. */
export const rowOfPeriod = (n: number, p: number): number => rowsOf(n).findIndex((r) => r.periods.includes(p));
/** A row's hours, [from, to) in order, neighbouring periods joined (as `levelHours`). */
export function rowHours(r: Row): [number, number][] {
  const out: [number, number][] = [];
  for (const p of r.periods.map((i) => PERIODS[i]).sort((a, b) => a.from - b.from)) {
    const last = out[out.length - 1];
    if (last && last[1] === p.from) last[1] = p.to;
    else out.push([p.from, p.to]);
  }
  return out;
}

/** T-3542 stage 2: the hours a line splits its day at (sim params.rs `Cuts`): where its morning (rush,
 * high), midday, afternoon, evening and night stretches start, local whole hours, the later ones past
 * 24 where they reach past midnight. Stretch `i` is period `i`'s (`Row.periods`) and runs to the next
 * cut (the night to the morning's, a day on); each is `STRETCH_MIN_H` at least. */
export type Cuts = readonly number[];
export const CUTS_DEFAULT: Cuts = [6, 10, 16, 20, 24];
export const STRETCH_MIN_H = 2;
/** Stretch `i`'s hours [from, to), `to` past `from`. */
const stretch = (c: Cuts, i: number): [number, number] => [c[i], i === 4 ? c[0] + 24 : c[i + 1]];
/** Whether `c` is a line's hours (sim `cuts_ok`). */
export const cutsOk = (c: Cuts): boolean => c.length === 5 && c.every((h) => Number.isInteger(h)) && c[0] >= 0 && c[0] < 24 && [0, 1, 2, 3, 4].every((i) => stretch(c, i)[1] - stretch(c, i)[0] >= STRETCH_MIN_H);
/** A row's stretches as runs of neighbours, in the day's order from the morning: high's three as one,
 * rush's two apart. */
function rowGroups(r: Row): number[][] {
  const out: number[][] = [];
  for (const i of [...r.periods].sort((a, b) => a - b)) {
    const last = out[out.length - 1];
    if (last && last[last.length - 1] === i - 1) last.push(i);
    else out.push([i]);
  }
  return out;
}
/** A row's hours by cuts `c`: [from, to) per run of its stretches, `to` past `from`. */
export const rowSpans = (c: Cuts, r: Row): [number, number][] => rowGroups(r).map((g) => [stretch(c, g[0])[0], stretch(c, g[g.length - 1])[1]]);
/** A row's hours as its box shows them: "6-10", "16-20", "20-1", "0-6"; rush's two "6-10, 16-20". */
export const rowHoursText = (c: Cuts, r: Row): string => rowSpans(c, r).map(([a, b]) => `${a % 24}-${b % 24 || 24}`).join(", ");
/** A row's hours as the clock writes them, for hovers: ["06:00-10:00", ...]. */
export const rowHourSpans = (c: Cuts, r: Row): string[] => rowSpans(c, r).map(([a, b]) => hourSpan(a % 24, b % 24 || 24));
/** The row of a line of `n` rows in force at local time `t` (seconds), by its hours `c`. */
export function rowAt(c: Cuts, n: number, t: number): number {
  const h = (Math.floor(t / 3600) % 24 + 24) % 24;
  const i = [0, 1, 2, 3, 4].find((k) => { const [a, b] = stretch(c, k); return (h >= a && h < b) || (h + 24 >= a && h + 24 < b); }) ?? 0;
  return rowOfPeriod(n, i);
}
/** Row `r`'s hours typed into its box (`text`: "5-10", "5:00-10:00"; rush's two runs "6-10, 16-20"),
 * into the line's cuts `c`: the new cuts, or `short` the hours its shortest row needs where one comes
 * out shorter (a line of three rows' high holds three stretches, six hours), or null where the text
 * is not that many runs of whole hours. A line of three rows keeps its midday and afternoon cuts
 * inside high, moved only as far as high's new hours need. */
export function typedHours(c: Cuts, n: number, r: number, text: string): { cuts: number[] } | { short: number } | null {
  const rows = rowsOf(n), groups = rowGroups(rows[r]);
  const spans = [...text.matchAll(/(\d{1,2})(?::00)?\s*[-–]\s*(\d{1,2})(?::00)?/g)].map((m) => [+m[1], +m[2]]);
  if (spans.length !== groups.length || spans.some(([a, b]) => a > 24 || b > 24)) return null;
  const typed = [...c].map((h) => h % 24);
  groups.forEach((g, k) => { typed[g[0]] = spans[k][0] % 24; typed[(g[g.length - 1] + 1) % 5] = spans[k][1] % 24; });
  // the cuts a line of three rows shows: high's start and end and low's start; the others follow
  const shown = n === 3 ? [0, 3, 4] : [0, 1, 2, 3, 4];
  const out = [...c];
  let last = -1;
  for (const i of shown) {
    let h = i === 0 ? typed[0] : typed[i];
    while (h <= last) h += 24;
    out[i] = h;
    last = h;
  }
  if (n === 3) {
    // midday and afternoon where they were, as near as high's hours let them
    const at = (h: number) => { let x = h % 24; while (x <= out[0]) x += 24; return x; };
    out[1] = Math.min(Math.max(at(c[1]), out[0] + STRETCH_MIN_H), out[3] - 2 * STRETCH_MIN_H);
    out[2] = Math.min(Math.max(at(c[2]), out[1] + STRETCH_MIN_H), out[3] - STRETCH_MIN_H);
  }
  if (cutsOk(out)) return { cuts: out };
  // the shortest row's need: two hours a stretch
  const need = rows.map((row) => rowGroups(row).map((g) => g.length * STRETCH_MIN_H));
  const short = rows.flatMap((row, k) => rowSpans(out, row).map(([a, b], j) => (b - a < need[k][j] ? need[k][j] : 0))).find((x) => x > 0);
  return { short: short ?? STRETCH_MIN_H };
}

/** T-3542: a demand network's run times per stop, two runs by five periods (`(run * 5 + period) * n
 * + k` from the line's stops x 10; sim/src/demand/api.rs `TIMES_PER_STOP`); trains an hour are one
 * per period. */
export const TIMES_PER_STOP = 2 * PERIODS.length;

/** An hour of the day as the clock writes it, "06:00"; the day's end is "24:00" (T-2271). */
export const hourText = (h: number): string => `${String(h).padStart(2, "0")}:00`;
/** Hours `a` to `b` as players read them: "06:00-10:00". */
export const hourSpan = (a: number, b: number): string => `${hourText(a)}-${hourText(b)}`;

/** The period at time `t` (seconds), as an index into PERIODS. Callers pass local time (T-309):
 * each region's periods run on its own hours. */
export function periodAt(t: number): number {
  const h = (Math.floor(t / 3600) % 24 + 24) % 24;
  return PERIODS.findIndex((p) => h >= p.from && h < p.to);
}

/** The hours of the day a demand level is scheduled, [from, to) in order, neighbouring periods of
 * the same level joined. */
export function levelHours(level: Demand): [number, number][] {
  const out: [number, number][] = [];
  for (const p of [...PERIODS].sort((a, b) => a.from - b.from)) {
    if (p.demand !== level) continue;
    const last = out[out.length - 1];
    if (last && last[1] === p.from) last[1] = p.to;
    else out.push([p.from, p.to]);
  }
  return out;
}

/** T-3542: trains a day from trains an hour in each period (`tphP`): each level's periods at one rate
 * taken as one stretch, so a line of three rows gives the sum by level to the bit. */
export function trainsADay(tphP: readonly number[]): number {
  let out = 0;
  for (const lev of ["high", "medium", "low"] as const) {
    const ps = PERIODS.flatMap((p, i) => (p.demand === lev ? [i] : []));
    if (ps.every((i) => tphP[i] === tphP[ps[0]])) out += tphP[ps[0]] * ps.reduce((s, i) => s + PERIODS[i].to - PERIODS[i].from, 0);
    else for (const i of ps) out += tphP[i] * (PERIODS[i].to - PERIODS[i].from);
  }
  return out;
}

/** Check against the Rust table (per period: from, to, level index, ...; `stride` numbers each). */
export function periodsMatch(rust: ArrayLike<number>, stride: number): boolean {
  if (rust.length !== PERIODS.length * stride) return false;
  return PERIODS.every((p, i) => rust[i * stride] === p.from && rust[i * stride + 1] === p.to && rust[i * stride + 2] === LEVEL_INDEX[p.demand]);
}
