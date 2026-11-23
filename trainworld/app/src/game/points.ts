// Points (T-684; SPEC 6): a measure per region, what its network earns on a forecast weekday at 20 a
// rail trip and 1 a passenger-km (T-694). A trip is a rider's whole rail journey, however many
// lines it uses, and counts in the region it starts in.
//
// Trips come from the solve, not from boardings: each region's rail trips (its own demand's, plus
// half of each regional corridor's at each end) are spread over the boardings at that region's
// stations, so a region's trips a day are exact while the hour they land follows its boardings.
// Passenger-km are each line's loads times its stretches' lengths, period by period, carried by
// that period's boardings onto the line (each boarding the line's average km), so the km count in
// the region where riders get on. The clock worker settles both as fares land (clock.worker.ts).
//
// T-3657: each line's points are times its cover, its fares over its running cost, at most 1, so free fares cannot
// buy points. A trip's points sit on its boardings, so each leg takes its own line's cover.

import { computed, effect, signal } from "@preact/signals";
import { CITIES, type CityId } from "./cities";
import { AVERAGE_DAY, awayNear, DAY_KINDS, DEMAND_DAYS, kindMade, PLAIN_KIND, rushRun } from "./days";
import { demandByDay, demandHeld, demandUpdating, finalWorld, networkWorld, type DayDemand } from "./demand";
import { loadProgress } from "./loading";
import { t } from "../i18n";
import { compactOf, sig3 } from "./format";
import { boatOf, boatRunKm } from "./boats";
import { incomeByDay, lineGeo, moneyState, moneyWorld } from "./money";
import { PERIODS } from "./periods";
import { lineRunFactor } from "./regionMoney";
import { stopsById } from "./world";
import { zonesLoaded } from "./zones";
import type { WorldView } from "./types";
import type { MoneyState } from "../workers/protocol";

export const POINTS_PER_TRIP = 20;
export const POINTS_PER_KM = 1;
import { GOAL_SHARES, STAR_SHARE } from "./goalRules";
export { GOAL_SHARES, STAR_SHARE };

/** The game second region `zone` reached each goal (`GOAL_SHARES` order), undefined where not yet. */
export function goalsReached(zone: string): (number | undefined)[] {
  const m = moneyState.value;
  return GOAL_SHARES.map((g) => (g === STAR_SHARE ? m?.stars?.[zone] : m?.goals?.[zone]?.[g]));
}

const PERIOD_HOURS = PERIODS.map((p) => p.to - p.from);

/** Points short, three significant digits (T-831): 980, 12.3k, 512k, 4.56M, 103M, 1.25B; 1.23万 in Japanese. */
export function pointsShort(n: number): string {
  // a figure that rounds up to the next unit (999.6k) is written in that unit
  if (n >= 999.5e6) return compactOf(sig3(n / 1e9), 1e9, "B");
  if (n >= 999.5e3) return compactOf(sig3(n / 1e6), 1e6, "M");
  if (n >= 1e3) return compactOf(sig3(n / 1e3), 1e3, "k");
  return String(Math.round(n));
}
const add = (m: Map<string, number>, k: string, v: number) => m.set(k, (m.get(k) ?? 0) + v);

/** Each region's rail trips a day: its own demand's, and the regional corridors' half at each end
 * (a corridor counts both ways; one within a place counts all to it). */
export function railTripsByRegion(d: DayDemand): Map<string, number> {
  const out = new Map<string, number>();
  for (const [id, v] of d.cities) add(out, id, v.railTripsPerDay);
  const byPlace = new Map<number, CityId[]>();
  for (const c of CITIES) { const l = byPlace.get(c.place); if (l) l.push(c.id); else byPlace.set(c.place, [c.id]); }
  const give = (place: number, v: number) => {
    const ids = byPlace.get(place);
    if (ids) for (const id of ids) add(out, id, v / ids.length);
  };
  const c = d.intercity?.corridors;
  for (let k = 0; c && k < c.railAt.length; k++) {
    const rail = c.longerRail[k] + c.commuteRail[k], from = c.railFrom[k], to = c.railTo[k];
    if (!(rail > 0)) continue;
    if (from === to) give(from, rail);
    else { give(from, rail / 2); give(to, rail / 2); }
  }
  return out;
}

/** Points per game hour of each running line on one kind of day, by region and period. */
function dayPoints(w: WorldView, d: DayDemand): Map<number, Map<string, number[]>> {
  const out = new Map<number, Map<string, number[]>>();
  const dv = d.view;
  if (!dv) return out;
  const byId = stopsById(w);
  const running = w.save.lines.filter((l) => w.lineStats[l.id]?.status === "running" && l.stops.length >= 2 && dv.lines.has(l.id));
  // boardings a day at each region's stations, to spread its trips over
  const board = new Map<string, number>();
  for (const l of running) {
    const z = lineGeo(w, l, byId).zones, b = dv.lines.get(l.id)!.boardings;
    for (let k = 0; k < b.length; k++) add(board, z[k] ?? "", b[k]);
  }
  const trips = railTripsByRegion(d);
  const perBoarding = new Map<string, number>();
  for (const [z, b] of board) if (b > 0) perBoarding.set(z, (trips.get(z) ?? 0) / b);
  const at: number[] = [];
  for (const l of running) {
    const st = w.lineStats[l.id], dm = dv.lines.get(l.id)!, n = l.stops.length;
    if (st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
    const { zones: z, kmAbs: km } = lineGeo(w, l, byId);
    const pkm: number[] = [];
    for (const seg of dm.loads) {
      let a = 0;
      if (seg.length === km.length) for (let k = 0; k < seg.length; k++) a += seg[k] * km[k];
      pkm.push(a);
    }
    let pkmAll = 0;
    for (const v of pkm) pkmAll += v;
    const regs = new Map<string, number[]>();
    for (let p = 0; p < PERIODS.length; p++) {
      // boardings at each stop in the period: both runs' (stop k is node k on run 0, 2n - 1 - k on
      // run 1), or the day's spread by the period's passenger-km when the solve kept no periods
      const bp = dm.board?.[p], both = !!bp && bp.length === 2 * n;
      let all = 0;
      for (let k = 0; k < n; k++) {
        at[k] = both ? bp![k] + bp![2 * n - 1 - k] : pkmAll > 0 ? dm.boardings[k] * pkm[p] / pkmAll : 0;
        all += at[k];
      }
      if (!(all > 0)) continue;
      const kmEach = (pkm[p] ?? 0) / all;
      for (let k = 0; k < n; k++) {
        const b = at[k];
        if (!(b > 0)) continue;
        let r = regs.get(z[k]);
        if (!r) regs.set(z[k], (r = PERIODS.map(() => 0)));
        r[p] += b * (POINTS_PER_TRIP * (perBoarding.get(z[k]) ?? 0) + POINTS_PER_KM * kmEach) / PERIOD_HOURS[p];
      }
    }
    if (regs.size) out.set(l.num, regs);
  }
  return out;
}

type RunPrices = Pick<MoneyState, "runCostTrainKm" | "runCostCarKm" | "hsRunCostTrainKm" | "hsRunCostCarKm" | "runCostBusKm" | "runCostBoatKm" | "runCostSmallBoatKm" | "runCostLargeBoatKm" | "economy">;
let runKey = "", runSame: RunPrices | null = null;
/** The running cost prices, the same object while they are the same (the money state changes every settle). */
const runPrices = computed<RunPrices | null>(() => {
  const m = moneyState.value;
  if (!m) return (runKey = "", runSame = null);
  const p: RunPrices = { runCostTrainKm: m.runCostTrainKm, runCostCarKm: m.runCostCarKm, hsRunCostTrainKm: m.hsRunCostTrainKm, hsRunCostCarKm: m.hsRunCostCarKm, runCostBusKm: m.runCostBusKm, runCostBoatKm: m.runCostBoatKm, runCostSmallBoatKm: m.runCostSmallBoatKm, runCostLargeBoatKm: m.runCostLargeBoatKm, economy: m.economy };
  const key = JSON.stringify(p);
  if (key !== runKey || !runSame) { runKey = key; runSame = p; }
  return runSame;
});

/** T-3657: each running line's cover, by line number: its fare income on the forecast weekday over its running cost
 * that day, at the trains and cars its riders were solved with (the running cost the line's money chart counts, no
 * upkeep), at most 1; 1 with no running cost. A line run on both pairs of quad track and its hidden twin share one,
 * from the two together. */
export const lineCover = computed<Map<number, number>>(() => {
  const out = new Map<number, number>();
  const w = moneyWorld.value, r = runPrices.value, inc = incomeByDay.value[0], dv = demandByDay.value[0]?.view;
  if (!w || !r || !inc) return out;
  const pair = new Map<string, { fares: number; run: number; nums: number[] }>();
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    let fares = 0;
    inc.lines.get(l.id)?.perHour.forEach((v, p) => { fares += v * PERIOD_HOURS[p] * 1e6; });
    const dm = dv?.lines.get(l.id);
    const tph = dm?.tph ?? l.tphP, cars = dm?.cars ?? st.cars;
    const [train, car] = st.fast ? [r.hsRunCostTrainKm, r.hsRunCostCarKm] : [r.runCostTrainKm, r.runCostCarKm];
    const perKm = st.bus?.ferry ? boatRunKm(r, boatOf(st.bus)) : st.bus ? r.runCostBusKm : train + cars * car;
    let run = 0;
    PERIODS.forEach((_, i) => { run += tph[i] * (st.oneWay ? 1 : 2) * st.km * perKm * r.economy * PERIOD_HOURS[i]; });
    run *= lineRunFactor(l.num); // T-2669
    const key = st.twinOf ?? l.id;
    let e = pair.get(key);
    if (!e) pair.set(key, (e = { fares: 0, run: 0, nums: [] }));
    e.fares += fares; e.run += run; e.nums.push(l.num);
  }
  for (const { fares, run, nums } of pair.values()) {
    const c = run > 0 ? Math.max(0, Math.min(1, fares / run)) : 1;
    for (const n of nums) out.set(n, c);
  }
  return out;
});

type LinePoints = { line: number; regions: [string, number[][]][] }[];
let pointsMemo: { w: WorldView; zl: number; days: DayDemand[]; per: Map<number, Map<string, number[]>>[]; out: LinePoints } | null = null;
const sameDay = (a: DayDemand, b: DayDemand) => a.view === b.view && a.cities === b.cities && a.intercity === b.intercity;
/** Every running line's points per game hour before its cover (`lineCover`). */
export const fullPoints = computed<LinePoints>(() => {
  const w = moneyWorld.value;
  if (!w) return [];
  const zl = zonesLoaded.value; // boundaries arriving re-place the stations
  // T-264, T-1373: a superholiday's days have their own points only while one is near, else the
  // ordinary day's they stand in for
  const near = awayNear.value, rush = rushRun.value;
  const days = DAY_KINDS.map((k) => demandByDay.value[kindMade(k, near, rush) ? k : PLAIN_KIND[k]]);
  const was = pointsMemo, same = !!was && was.w === w && was.zl === zl;
  if (same && days.every((d, k) => sameDay(d, was.days[k]))) return was.out;
  const own = days.map((d, k) => (!kindMade(k, near, rush) ? null : same && sameDay(d, was.days[k]) ? was.per[k] : dayPoints(w, d)));
  const per = own.map((x, k) => x ?? own[PLAIN_KIND[k]]!);
  const base = days.findIndex((d) => d.view);
  const out = base < 0 ? [] : assemble(days, per, base);
  pointsMemo = { w, zl, days, per, out };
  return out;
});
/** Every running line's points per game hour, `[region, perHour[dayKind][period]]` by line number,
 * as the clock worker settles them, times the line's cover (`lineCover`, the weekday's on every kind of day). A kind of
 * day not solved yet takes the weekday's. A day is redone only when its views, the network or the zone boundaries
 * change (T-1209). */
export const linePoints = computed<LinePoints>(() => {
  const cover = lineCover.value;
  return fullPoints.value.map(({ line, regions }) => {
    const c = cover.get(line) ?? 1;
    return c === 1 ? { line, regions } : { line, regions: regions.map(([z, per]): [string, number[][]] => [z, per.map((d) => d.map((v) => v * c))]) };
  });
});

function assemble(days: DayDemand[], per: Map<number, Map<string, number[]>>[], base: number): LinePoints {
  const lines = new Set(per.flatMap((m) => [...m.keys()]));
  return [...lines].map((line) => {
    const zones = new Set(per.flatMap((m) => [...(m.get(line)?.keys() ?? [])]));
    const zero = PERIODS.map(() => 0);
    return {
      line,
      regions: [...zones].map((z): [string, number[][]] => [z, DAY_KINDS.map((k) => (days[k].view ? per[k] : per[base]).get(line)?.get(z) ?? zero)]),
    };
  });
}

/** T-694: each region's points, the measure shown: what it earned in the past 24 whole game hours,
 * from the hour ledgers, so it moves once a game hour (fewer hours while the ledger is younger). */
export const pointsPast24h = computed<Map<string, number>>(() => {
  const m = moneyState.value, out = new Map<string, number>();
  const now = m?.hours.at(-1)?.hour;
  if (!m || now === undefined) return out;
  for (const [z, r] of Object.entries(m.regions ?? {})) {
    let t = 0;
    for (const e of r.hours) if (e.hour >= now - 24 && e.hour < now) t += e.points ?? 0;
    if (t > 0) out.set(z, t);
  }
  return out;
});

/** Each region's points a day at the riders now, per demand day (`DEMAND_DAYS`). */
export const pointsPerDay = computed<Map<string, number[]>>(() => {
  const out = new Map<string, number[]>();
  for (const { regions } of linePoints.value) {
    for (const [z, per] of regions) {
      let r = out.get(z);
      if (!r) out.set(z, (r = DEMAND_DAYS.map(() => 0)));
      for (const k of DAY_KINDS) r[k] += per[k].reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0);
    }
  }
  for (const r of out.values()) r[AVERAGE_DAY] = (5 * r[0] + r[1] + r[2]) / 7;
  return out;
});

/** T-3465: each region's points, the score shown and sent everywhere: its forecast weekday's, from the current solve, so an
 * edit moves it as soon as the solve answers and neither the clock, a weekend nor a holiday does. Until a solve exists,
 * what the region earned in the past 24 game hours. */
export const pointsScore = computed<Map<string, number>>(() => {
  const out = new Map<string, number>();
  let any = false;
  for (const [z, r] of pointsPerDay.value) {
    if (z) any = true;
    if (z && r[0] > 0) out.set(z, Math.round(r[0]));
  }
  // (T-3657: a region whose lines' fares pay none of their running cost has points of 0, not the past 24 hours')
  return any ? out : pointsPast24h.value;
});

/** T-3657: whether the network's riders earn points before cover, so a score of 0 is its fares paying none of the
 * running cost and goes up to the boards like any other. */
export const pointsBeforeCover = computed<boolean>(() => fullPoints.value.some(({ regions }) => regions.some(([z, per]) => !!z && per[0].some((v) => v > 0))));

/** T-3639, T-3644: whether the figures are a whole final of the network as it is ("solved"), or there is nothing running
 * to solve ("idle"). Before that a score is a load's or an edit's early figure, the past 24 hours' before any solve, or
 * (a game just opened in a running page) the last game's view. The board is sent only then, and the score on screen
 * shows only then. */
export const scoreSettled = computed<"solved" | "idle" | null>(() => {
  const w = moneyWorld.value, net = networkWorld.value;
  // (a load's focus figures come only while its whole solve runs; T-2533 can keep the pool's zone after it, a hair off)
  if (!w || demandUpdating.value || demandHeld.value || loadProgress.value || net < 0 || finalWorld.value < net) return null;
  const running = w.save.lines.filter((l) => w.lineStats[l.id]?.status === "running");
  if (!running.length) return "idle";
  const dv = demandByDay.value[0]?.view;
  if (!dv?.final) return null;
  // every running line solved on its stops as they are (editNote's `solvedForNow`, and none missing)
  for (const l of running) {
    const d = dv.lines.get(l.id);
    if (!d || (d.stops && (d.stops.length !== l.stops.length || d.stops.some((s, k) => s !== l.stops[k])))) return null;
  }
  return "solved";
});

const lastFinal = signal<Map<string, number> | null>(null);
effect(() => {
  if (scoreSettled.value) lastFinal.value = pointsScore.peek();
  else if (loadProgress.value) lastFinal.value = null; // another game's figures are not this one's
});
/** The scores on screen: the newest whole final's, kept through an edit's solve (the board holds it too), and null (a
 * dash, hover `pointsPending`) from a load's start until its first final. */
export const pointsShown = computed<Map<string, number> | null>(() => (scoreSettled.value ? pointsScore.value : lastFinal.value));
export const pointsText = (n: number): string => (pointsShown.value ? pointsShort(n) : "–");
export const pointsPending = () => t("lineFigures.pendingNew");
