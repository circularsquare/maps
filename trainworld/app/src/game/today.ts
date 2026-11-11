// The past 24 hours (T-446; before it, today so far, T-293: Anita wanted counts, not a forecast
// with the future in it, and then no reset at midnight). The line and station panels count the
// day up to now: riders carried, boardings at each stop, fares taken, trains that left full
// and riders they left behind. Counted by the clock worker from the boarding account's departures
// (`TrackApi.past_day`, sim/src/track/boarding.rs) and each line's fares as the worker pays them
// (T-284); asked again four times a wall second while a panel shows them and the clock runs, as
// the cash moves. The names say "today" from T-293.

import { t } from "../i18n";
import { computed, signal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import { clock, lineSeconds } from "./clock";
import { stationsById, world } from "./world";
import { queryClock } from "../workers/clockClient";
import { accountVersion, type CrowdedStretch } from "./boarding";
import { demandView } from "./demand";
import { linesPending, type PendingKind } from "./lineFigures";
import { moneyState } from "./money";
import { lineRunFactor } from "./regionMoney";
import { clockTime, trainKind } from "./format";
import { hourSpan, PERIODS, periodAt } from "./periods";
import { crowdColour } from "./palette";
import { tipEnds, tipStrip } from "../ui/tip";
import { holidayOn, kindIn, lineCountry } from "./holidays";
import type { Demand, LineInput, LineStats } from "./types";
import type { Vehicle } from "./vehicle";
import { cameraMoving, guideVerbose } from "./ui";
import { perfOff } from "../perfFlags";

export interface LineToday {
  /** riders boarded in the past 24 hours, or yesterday (a change of line boards again), the line's own stops */
  riders: number;
  /** boardings at each stop, in the line's stop order */
  stops: number[];
  /** fares taken, US$ */
  fares: number;
  /** T-2475: fares a whole day of the line's local day now takes, US$: the expected profit's */
  dayFares: number;
  /** trains that left full, out of `trains` that ran (T-481), riders left behind */
  fullTrains: number;
  trains: number;
  leftBehind: number;
  /** T-472: trains in service now, and how many of them are packed full somewhere on their trip */
  trainsNow: number;
  fullNow: number;
  /** T-2236: the period now on today's kind of day: the busiest hour's load on the line's fullest
   * stretch as a share of the rated load (`period_peaks` in boarding.rs) */
  crowdLoad: number;
}

/** T-2236: grades of a load as a share of the rated load (`CROWD_STOPS`): 0 within it, 1 past 100%
 * as `loadText` rounds it (`CROWDED_FROM` in boarding.rs), 2 orange from 130%, 3 red from 140%. */
export const CROWD_GRADES = [1.005, 1.3, 1.4];
export function crowdGrade(load: number): 0 | 1 | 2 | 3 {
  return load >= CROWD_GRADES[2] ? 3 : load >= CROWD_GRADES[1] ? 2 : load >= CROWD_GRADES[0] ? 1 : 0;
}
/** Text the player should act on (T-472): orange and red grades. */
export function crowdClass(load: number): "" | "crowd-warn" | "crowd-bad" {
  const g = crowdGrade(load);
  return g === 3 ? "crowd-bad" : g === 2 ? "crowd-warn" : "";
}
/** "132%" */
export const loadText = (share: number) => `${Math.round(share * 100)}%`;
/** A crowded stretch's hover line (T-2236): "Peak load 132% at 08:00". */
export const peakLoadText = (c: CrowdedStretch) => t("today.peakAt", { load: loadText(c.load), time: clockTime(c.hour * 60) });

/** Numbers per period in `TrackApi.line_periods` (`PERIOD_STRIDE` in boarding.rs). */
export const PERIOD_STRIDE = 11;

/** Where the busiest hour's load per period and stretch starts in a line's `line_periods` numbers
 * `v` (T-2236, `period_peaks` in boarding.rs: `p * (stops - 1) + k`), whole or as the clock
 * worker's `linesPeriods` cuts it (the per-period block and the loads), or -1 where `v` has none (a
 * line of one stop, an older WASM build). */
function peaksAt(v: ArrayLike<number>, stops: number): number {
  const s = stops - 1, n = PERIODS.length;
  return s >= 1 && (v.length === n * (PERIOD_STRIDE + 2 * stops + 3 * s) || v.length === n * (PERIOD_STRIDE + s)) ? v.length - n * s : -1;
}

/** T-2279: the power the stretches' loads past 100% are averaged at. Above 1 the mean leans to the
 * fullest: one stretch packed to crush among 20 grades as 118% (1.5 - 1 over the cube root of 20). */
export const CROWD_MEAN_POWER = 2;

/** T-2279: a line's load from its stretches' loads (shares of the rated load): 100% plus the power
 * mean of each one's load past 100% (`CROWD_MEAN_POWER`), so it is the fullest stretch's load when
 * every stretch is that full and less as fewer are. */
export function stretchLoad(per: ArrayLike<number>): number {
  let peak = 0, sum = 0;
  for (let k = 0; k < per.length; k++) {
    peak = Math.max(peak, per[k]);
    if (per[k] > 1) sum += (per[k] - 1) ** CROWD_MEAN_POWER;
  }
  if (peak <= 1) return peak;
  return 1 + (sum / per.length) ** (1 / CROWD_MEAN_POWER);
}

/** T-2279: the Lines list's trains cell's load in period `p` from a line's `line_periods` numbers
 * (`stretchLoad` over each stretch's busiest hour then); null where `v` has no loads. */
export function periodLoad(v: ArrayLike<number>, p: number, stops: number): number | null {
  const at = peaksAt(v, stops), s = stops - 1;
  if (at < 0) return null;
  return stretchLoad(Array.from({ length: s }, (_, k) => v[at + p * s + k]));
}

/** T-477, T-515, T-2236, T-2334: one demand level's crowding from a line's `line_periods` numbers
 * `v`, as the line panel's dot and the Lines list's trains cell show it: the busiest hour's load on
 * each stretch over the level's periods, as a share of the rated load, graded over the stretches by
 * `stretchLoad` (`load`, the dot's fill) and at the fullest (`peak`, its ring), and the hover: the
 * peak, a bar of the stretches between their stations, and each period's load where the level has
 * several. T-2661: `flip` (`lineReversed`) draws the bar last stretch first, so it reads west to east.
 * Null when the level runs no trains. */
export function crowdLevel(v: ArrayLike<number>, d: Demand, stops: number, flip = false, ends?: readonly [string, string], vehicle: Vehicle = "train"): { load: number; peak: number; tip: string } | null {
  const at = peaksAt(v, stops), s = stops - 1;
  if (at < 0) return null;
  let trains = 0, peak = 0;
  const per = new Array<number>(s).fill(0), lines: string[] = [];
  const want = LEVEL_INDEX[d];
  for (let i = 0; i < PERIODS.length; i++) {
    const p = PERIODS[i];
    if (LEVEL_INDEX[p.demand] !== want || !(v[i * PERIOD_STRIDE] > 0)) continue;
    trains += v[i * PERIOD_STRIDE];
    let most = 0;
    for (let k = 0; k < s; k++) {
      const x = v[at + i * s + k];
      per[k] = Math.max(per[k], x);
      most = Math.max(most, x);
    }
    peak = Math.max(peak, most);
    lines.push(t("today.periodLoad", { span: hourSpan(p.from, p.to), load: loadText(most) }));
  }
  if (!(trains > 0)) return null;
  const over = per.filter((x) => x >= CROWD_GRADES[0]).length;
  // T-2263: the tutorial warns where only a short stretch is full
  const short = guideVerbose.value && over > 0 && over * 2 <= s;
  const tip = `${t("today.peak", { load: loadText(peak) })}\n${tipStrip((flip ? per.slice().reverse() : per).map(crowdColour), true)}`
    + (ends ? `
${flip ? tipEnds(ends[1], ends[0]) : tipEnds(ends[0], ends[1])}` : "")
    + (lines.length > 1 ? lines.map((x) => `\n${x}`).join("") : "")
    + (short ? "\n" + t("today.shortStretch", { vehicle }) : "");
  return { load: stretchLoad(per), peak, tip };
}

/** T-2772: the names of a line's first and last station, for the ends of `crowdLevel`'s bar; undefined when unknown */
export function endNames(l: { stops: readonly string[] }): [string, string] | undefined {
  const w = world.peek();
  if (!w || l.stops.length < 2) return undefined;
  const by = stationsById(w), a = by.get(l.stops[0])?.name, b = by.get(l.stops[l.stops.length - 1])?.name;
  return a && b ? [a, b] : undefined;
}

/** T-1893: the profit hover's line when the line's day is not a weekday, so a loss on a quiet day
 * does not alarm; null on a weekday. */
export function dayNote(l: LineInput, st: LineStats): string | null {
  const country = lineCountry(l.stops), day = Math.floor(lineSeconds(st) / 86400);
  const kind = kindIn(country, day);
  if (kind === 0) return null;
  if (kind === 1 || kind === 2) return t("today.weekend");
  if (kind === 3) {
    const name = country ? holidayOn(country, day) : null;
    return t("today.holiday", { named: !!name, name: name ?? "" });
  }
  return t("today.travelDay");
}

const LEVEL_INDEX = { high: 0, medium: 1, low: 2 } as const;
/** US$ of running cost over the past 24 hours (T-446): each hour's trains at its demand level, the
 * hours at either end in part, as the worker drains the cash (T-284). T-1492: at the trains and cars
 * the fares beside it were worked out for (`todayService`), so a train count change moves both
 * when its demand lands. */
export function runningToday(l: LineInput, st: LineStats): number {
  const m = moneyState.value;
  if (!m) return 0;
  // T-309: in the line's own local time
  const t = (clock.now() + st.shift) / 3600, t0 = t - 24;
  const held = todayService.value.get(l.id);
  const tph = held ? held.tph : [l.tph.high, l.tph.medium, l.tph.low];
  const cars = held?.cars ?? st.cars;
  const k = trainKind(m, st.fast, st.tram);
  let cost = 0;
  for (let h = Math.floor(t0); h < t; h++) {
    const lev = LEVEL_INDEX[PERIODS[periodAt(h * 3600)].demand];
    // T-1464: a bus line's buses cost a bus-km each
    const perKm = st.bus?.ferry ? m.runCostBoatKm : st.bus ? m.runCostBusKm : k.runCostTrainKm + cars * k.runCostCarKm;
    cost += tph[lev] * (st.oneWay ? 1 : 2) * st.km * perKm * m.economy *(Math.min(h + 1, t) - Math.max(h, t0));
  }
  // T-2669: at its countries' prices, as the clock worker drains it
  return cost * lineRunFactor(l.num);
}

/** T-2536: whether a line's profit has been worked out: it runs trains (or buses, boats) and a demand
 * result with an entry for it has landed. Before that its figures are only the upkeep. */
export function hasTrains(l: LineInput): boolean {
  const held = todayService.value.get(l.id);
  const tph = held ? held.tph : [l.tph.high, l.tph.medium, l.tph.low];
  return tph.some((x) => x > 0);
}
/** A line with no riders gets no demand entry, so once its edit is no longer pending it counts as calculated too. */
export const profitKnown = (l: LineInput) => {
  const dv = demandView.value;
  return hasTrains(l) && !!dv && (dv.lines.has(l.id) || !linesPending.value.has(l.id));
};

/** T-2658: why a line's riders show as being worked out: its change not yet solved (`linesPending`), or
 * solved with riders the boarding account has not counted yet (seconds on a big game: the figure would
 * read 0 meanwhile), "build" when it has no counts at all. Undefined once its figures show. */
export function pendingOf(l: LineInput): PendingKind | undefined {
  const k = linesPending.value.get(l.id);
  if (k) return k;
  const d = today.value.get(l.id);
  if (d && d.riders >= 1) return undefined;
  // (the line's trains as set, not as the account last counted them: those may be none yet)
  const runs = l.tph.high > 0 || l.tph.medium > 0 || l.tph.low > 0;
  return (demandView.value?.lines.get(l.id)?.ridersPerDay ?? 0) >= 1 && runs ? (d ? "change" : "build") : undefined;
}

/** T-2678: each line's last riders count of 1 or more (`today`), while the line exists */
const lastRiders = new Map<string, number>();
/** T-2773: each line's counts as last read while it was not pending */
const settled = new Map<string, LineToday>();

/** T-2678: the riders figure shown beside a pending line's label: its count now, else its last count (a
 * line stopped and built again), else the riders a day demand has for it (the account not counting them
 * yet, or a new line's first estimate); null with none, when the label stands alone. */
export function pendingRiders(l: LineInput): number | null {
  const d = today.value.get(l.id);
  if (d && d.riders >= 1) return d.riders;
  const last = lastRiders.get(l.id);
  if (last !== undefined) return last;
  const r = demandView.value?.lines.get(l.id)?.ridersPerDay ?? 0;
  return r >= 1 ? r : null;
}

/** Per line id; empty until the first answer (and for a line with no riders yet). */
export const today = signal<Map<string, LineToday>>(new Map());
/** T-1492: per line id, the service the demand result in force when `today` was asked had solved:
 * the fares in the answer come from that result's riders */
export const todayService = signal<Map<string, { tph: [number, number, number]; cars: number }>>(new Map());
/** whether any line has counts: lists sorted by today's figures re-sort when it turns */
export const todayLoaded = computed(() => today.value.size > 0);

export function readToday(d: Float64Array): Map<string, LineToday> {
  const out = new Map<string, LineToday>();
  const count = d[0];
  let i = 1;
  const at: { id: string; t: LineToday }[] = [];
  for (let k = 0; k < count; k++) {
    // d[i + 5] is the fullest train and d[i + 7 ..] the spans of full trains, which nothing shows
    const id = d[i], n = d[i + 1], nSpans = d[i + 6];
    const stops = Array.from(d.subarray(i + 7 + 2 * nSpans, i + 7 + 2 * nSpans + n));
    const e = i + 7 + 2 * nSpans + n;
    const t: LineToday = { riders: d[i + 2], fullTrains: d[i + 3], leftBehind: d[i + 4], stops, fares: 0, dayFares: 0, trainsNow: d[e], fullNow: d[e + 1], trains: d[e + 2], crowdLoad: d[e + 7] };
    at.push({ id: String(id), t });
    i = e + 8;
  }
  at.forEach((a, k) => { a.t.fares = (d[i + k] ?? 0) * 1e6; a.t.dayFares = (d[i + count + k] ?? 0) * 1e6; out.set(a.id, a.t); });
  return out;
}

/** Boardings in the past 24 hours per station id (stations with a counted line), summed once per answer. */
export const stationsToday = computed(() => {
  const out = new Map<string, number>();
  const t = today.value, w = world.peek();
  if (!w) return out;
  for (const l of w.save.lines) {
    const d = t.get(l.id);
    if (d) l.stops.forEach((s, k) => out.set(s, (out.get(s) ?? 0) + (d.stops[k] ?? 0)));
  }
  return out;
});

const ASK_MS = 250;
let users = 0, timer = 0, busy = false, askedAt = NaN, askedVersion = -1;
let askedDv: unknown = null;

function ask(force = false) {
  // T-1207: the Lines tab's rows re-rendered four times a second while the map was dragged
  if (busy || (!force && cameraMoving.now)) return;
  // the network or the account (new riders, the hour turning) changed
  const t = clock.now(), v = (world.peek()?.version ?? -1) * 1e6 + accountVersion.peek();
  // the income the worker holds as it answers came from this result (clockClient posts it on each new one)
  const dv = demandView.peek();
  // nothing moves while the clock stands, unless the network, the account or the demand result changed
  if (!force && t === askedAt && v === askedVersion && dv === askedDv) return;
  busy = true;
  askedAt = t;
  askedVersion = v;
  askedDv = dv;
  const service = new Map<string, { tph: [number, number, number]; cars: number }>();
  for (const [id, ld] of dv?.lines ?? []) if (ld.tph && ld.cars !== undefined) service.set(id, { tph: ld.tph, cars: ld.cars });
  void queryClock({ what: "today" }).then((d) => {
    busy = false;
    if (!d) return;
    const w = world.peek();
    readFor = new Map((w?.save.lines ?? []).map((l) => [l.id, l.stops]));
    const next = readToday(d), prev = today.peek();
    // A line's account is worked out again a while after its stops change (new riders, T-164);
    // until then its counts may be per the old stops. The ones moved to the new stops stand meanwhile.
    const pend = linesPending.peek();
    for (const l of w?.save.lines ?? []) {
      const t = next.get(l.id), p = prev.get(l.id);
      if (t && t.stops.length !== l.stops.length && p?.stops.length === l.stops.length) next.set(l.id, { ...t, stops: p.stops });
      // T-1419: no account at all for a running line yet (new riders being solved, or solved and not
      // yet counted, T-2793): the last counts stand, moved onto its stops above
      // (only while they can still be replaced: a line left with no riders has neither a pending mark nor a demand entry)
      else if (!t && p && w?.lineStats[l.id]?.status === "running" && (pend.has(l.id) || dv?.lines.has(l.id))) next.set(l.id, p);
    }
    // T-2773: a line whose riders are being worked out again keeps its last settled counts and fares, not the
    // half-done answer's (a local solve reads a long line far low); the final one replaces them when it lands
    for (const [id, x] of next) {
      const s = settled.get(id);
      if (pend.get(id) === "change" && s && !perfOff("settledHold")) next.set(id, { ...x, riders: s.riders, fullTrains: s.fullTrains, leftBehind: s.leftBehind, fares: s.fares, dayFares: s.dayFares, stops: s.stops.length === x.stops.length ? s.stops : x.stops });
      else settled.set(id, x);
    }
    for (const id of settled.keys()) if (!next.has(id) && !pend.has(id)) settled.delete(id);
    todayService.value = service;
    // T-2678: each line's last count, kept while the line exists (it may stop and be built again)
    for (const [id, x] of next) if (x.riders >= 1) lastRiders.set(id, x.riders);
    if (lastRiders.size > next.size) { const ids = new Set((w?.save.lines ?? []).map((l) => l.id)); for (const id of lastRiders.keys()) if (!ids.has(id)) lastRiders.delete(id); }
    today.value = next;
  });
}

/** the stops each line's counts are in the order of: the world's when they were read */
let readFor = new Map<string, string[]>();

/** T-648: a stop edit renumbers the line's stops, and the counts (until the next answer) would sit
 * one row off: they follow their stations to the new order now, a new stop at 0. */
world.subscribe((w) => {
  if (w && clock.held.peek()) { lastRiders.clear(); settled.clear(); } // (a game loading: its lines are other lines)
  const t = today.peek();
  if (!w || !t.size) return;
  let out: Map<string, LineToday> | null = null;
  for (const l of w.save.lines) {
    const d = t.get(l.id), was = readFor.get(l.id);
    if (!d || !was || (was.length === l.stops.length && was.every((s, k) => s === l.stops[k]))) continue;
    const used = was.map(() => false);
    const stops = l.stops.map((s) => {
      const k = was.findIndex((x, j) => !used[j] && x === s);
      if (k < 0) return 0;
      used[k] = true;
      return d.stops[k] ?? 0;
    });
    (out ??= new Map(t)).set(l.id, { ...d, stops });
    readFor.set(l.id, l.stops);
  }
  if (out) {
    today.value = out;
    if (users > 0) ask(); // the new stop's count, and a line that just lost its service, at once
  }
});

/** Keep `today` fresh while the calling panel is shown. */
export function useToday() {
  useEffect(() => {
    if (users++ === 0) {
      ask(true);
      timer = setInterval(ask, ASK_MS) as unknown as number;
    }
    return () => {
      if (--users === 0) clearInterval(timer);
    };
  }, []);
}
