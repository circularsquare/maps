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
import { CITIES, zoneCountry, type CityId } from "./cities";
import { AVERAGE_DAY, awayNear, DAY_KINDS, DEMAND_DAYS, kindMade, PLAIN_KIND, rushRun } from "./days";
import { mainDemandByDay, demandHeld, demandUpdating, finalWorld, networkWorld, type DayDemand, type DemandView, type LineDemand } from "./demand";
import { checkPatches, freshly, freshNow, perLineOf, sameBuilt, type PerLine } from "./worldDiff";
import { loadProgress } from "./loading";
import { t } from "../i18n";
import { compactOf, sig3 } from "./format";
import { boatOf, boatRunKm } from "./boats";
import { incomeByDay, lineGeo, moneyState, moneyWorld, type Income, type LineMoney } from "./money";
import { PERIODS } from "./periods";
import { lineShares, type Shares } from "./regionMoney";
import { runFactorOf } from "./income";
import { stopsById } from "./world";
import { zonesLoaded } from "./zones";
import type { LineInput, LineStats, WorldView } from "./types";
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

const tripsMemo = new WeakMap<object, { intercity: DayDemand["intercity"]; out: Map<string, number> }>();
/** Each region's rail trips a day: its own demand's, and the regional corridors' half at each end
 * (a corridor counts both ways; one within a place counts all to it). Kept per day's figures (T-4150). */
export function railTripsByRegion(d: DayDemand): Map<string, number> {
  const had = tripsMemo.get(d.cities);
  if (had && had.intercity === d.intercity && !freshNow()) return had.out;
  const out = railTripsOf(d);
  tripsMemo.set(d.cities, { intercity: d.intercity, out });
  return out;
}
function railTripsOf(d: DayDemand): Map<string, number> {
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

/** T-4284: what a running line gives one kind of day's points before its regions' trips a boarding, kept per line from the
 * world last read (`PerLine`, one per demand view): its boardings a day by region (each summed in stop order, the regions
 * as first met), and with both runs' stop positions its boardings by region (as first met with any) and period and each
 * period's passenger-km a boarding (`local`; null: it earns none). `pbs` and `regs`: the trips a boarding of each of
 * `local`'s regions its points were last made at, and those points. */
type PointsLine = { board: [string, number][]; local: { zones: string[]; at: number[][]; kmEach: number[] } | null; pbs: number[]; regs: Map<string, number[]> | null };
const pointsLinesBy = new Map<unknown, PerLine<PointsLine | null>>();

/** Points per game hour of each running line on one kind of day, by region and period: each line's own terms, kept while
 * the edit leaves it alone, and the trips a boarding of each region (its trips spread over its boardings), which any line's
 * boardings move, applied to them in a second step. */
function dayPoints(w: WorldView, d: DayDemand, kind: number): Map<number, Map<string, number[]>> {
  const out = new Map<number, Map<string, number[]>>();
  const dv = d.view;
  if (!dv) return out;
  const kept = perLineOf(pointsLinesBy, kind);
  kept.start(w, [zonesLoaded.peek()]);
  const running: PointsLine[] = [], nums: number[] = [];
  // boardings a day at each region's stations, to spread its trips over
  const board = new Map<string, number>();
  for (const l of w.save.lines) {
    const dm = dv.lines.get(l.id);
    let e = kept.kept(l.id, dm);
    if (e === undefined) e = kept.put(l.id, pointsLine(w, l, dv), dm);
    if (!e) continue;
    for (const [z, v] of e.board) add(board, z, v);
    running.push(e);
    nums.push(l.num);
  }
  kept.end();
  const trips = railTripsByRegion(d);
  const perBoarding = new Map<string, number>();
  for (const [z, b] of board) if (b > 0) perBoarding.set(z, (trips.get(z) ?? 0) / b);
  for (let i = 0; i < running.length; i++) {
    const e = running[i], loc = e.local;
    if (!loc) continue;
    let same = !!e.regs && !freshNow();
    for (let j = 0; same && j < loc.zones.length; j++) if (e.pbs[j] !== (perBoarding.get(loc.zones[j]) ?? 0)) same = false;
    if (!same) {
      e.pbs = loc.zones.map((z) => perBoarding.get(z) ?? 0);
      e.regs = regsOf(loc, e.pbs);
    }
    if (e.regs!.size) out.set(nums[i], e.regs!);
  }
  return out;
}

/** One running line's own terms in a kind of day's points (null: not running, or no figures). */
function pointsLine(w: WorldView, l: LineInput, dv: DemandView): PointsLine | null {
  const st = w.lineStats[l.id], dm = dv.lines.get(l.id), n = l.stops.length;
  if (st?.status !== "running" || n < 2 || !dm) return null;
  const g = lineGeo(w, l, stopsById(w)), z = g.zones, b = dm.boardings;
  const board: [string, number][] = [];
  for (let k = 0; k < b.length; k++) {
    const zone = z[k] ?? "";
    let e = board.find((x) => x[0] === zone);
    if (!e) board.push((e = [zone, 0]));
    e[1] += b[k];
  }
  if (st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) return { board, local: null, pbs: [], regs: null };
  const km = g.kmAbs;
  const pkm: number[] = [];
  for (const seg of dm.loads) {
    let a = 0;
    if (seg.length === km.length) for (let k = 0; k < seg.length; k++) a += seg[k] * km[k];
    pkm.push(a);
  }
  let pkmAll = 0;
  for (const v of pkm) pkmAll += v;
  const zones: string[] = [], at: number[][] = [], kmEach = PERIODS.map(() => 0), here: number[] = [];
  for (let p = 0; p < PERIODS.length; p++) {
    // boardings at each stop in the period: both runs' (stop k is node k on run 0, 2n - 1 - k on
    // run 1), or the day's spread by the period's passenger-km when the solve kept no periods
    const bp = dm.board?.[p], both = !!bp && bp.length === 2 * n;
    let all = 0;
    for (let k = 0; k < n; k++) {
      here[k] = both ? bp![k] + bp![2 * n - 1 - k] : pkmAll > 0 ? dm.boardings[k] * pkm[p] / pkmAll : 0;
      all += here[k];
    }
    if (!(all > 0)) continue;
    kmEach[p] = (pkm[p] ?? 0) / all;
    for (let k = 0; k < n; k++) {
      if (!(here[k] > 0)) continue;
      let j = zones.indexOf(z[k]);
      if (j < 0) { j = zones.push(z[k]) - 1; at.push(PERIODS.map(() => 0)); }
      at[j][p] += here[k];
    }
  }
  return { board, local: { zones, at, kmEach }, pbs: [], regs: null };
}
/** A line's points per game hour by region and period from its own terms and its regions' trips a boarding (`pbs`). */
function regsOf(loc: NonNullable<PointsLine["local"]>, pbs: number[]): Map<string, number[]> {
  const regs = new Map<string, number[]>();
  loc.zones.forEach((z, j) => {
    const at = loc.at[j];
    regs.set(z, PERIODS.map((_, p) => (at[p] > 0 ? at[p] * (POINTS_PER_TRIP * pbs[j] + POINTS_PER_KM * loc.kmEach[p]) / PERIOD_HOURS[p] : 0)));
  });
  return regs;
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
  const w = moneyWorld.value, r = runPrices.value, inc = incomeByDay.value[0], dv = mainDemandByDay.value[0]?.view;
  if (!w || !r || !inc) return (coverLast = new Map(), new Map());
  const shares = lineShares.value;
  const out = coverFor(w, r, inc, dv, shares, coverLast);
  coverLast = out.next;
  if (checkPatches() && !sameBuilt(out.cover, coverFor(w, r, inc, dv, shares, null).cover)) console.error("T-4284 check: the lines' cover kept per line differs from a whole build");
  return out.cover;
});
/** Each running line's cover from its fares and running cost, those of a line kept (`last`) while what they read is the
 * same objects (T-4284). */
function coverFor(w: WorldView, r: RunPrices, inc: Income, dv: DemandView | null | undefined, shares: ReadonlyMap<number, { run: Shares }>, last: ReadonlyMap<string, CoverLine> | null) {
  const cover = new Map<number, number>();
  const pair = new Map<string, { fares: number; run: number; nums: number[] }>();
  const next = new Map<string, CoverLine>();
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    const money = inc.lines.get(l.id), dm = dv?.lines.get(l.id), runShares = shares.get(l.num)?.run;
    let e = last?.get(l.id);
    if (!e || e.money !== money || e.dm !== dm || e.st !== st || e.l !== l || e.runShares !== runShares || e.r !== r) e = coverLine(l, st, money, dm, runShares, r);
    next.set(l.id, e);
    const key = st.twinOf ?? l.id;
    let p = pair.get(key);
    if (!p) pair.set(key, (p = { fares: 0, run: 0, nums: [] }));
    p.fares += e.fares; p.run += e.run; p.nums.push(l.num);
  }
  for (const { fares, run, nums } of pair.values()) {
    const c = run > 0 ? Math.max(0, Math.min(1, fares / run)) : 1;
    for (const n of nums) cover.set(n, c);
  }
  return { cover, next };
}
type CoverLine = { l: LineInput; st: LineStats; money: LineMoney | undefined; dm: LineDemand | undefined; runShares: Shares | undefined; r: RunPrices; fares: number; run: number };
let coverLast = new Map<string, CoverLine>();
/** One running line's fare income and running cost on the forecast weekday (`lineCover`). */
function coverLine(l: LineInput, st: LineStats, money: LineMoney | undefined, dm: LineDemand | undefined, runShares: Shares | undefined, r: RunPrices): CoverLine {
  let fares = 0;
  money?.perHour.forEach((v, p) => { fares += v * PERIOD_HOURS[p] * 1e6; });
  const tph = dm?.tph ?? l.tphP, cars = dm?.cars ?? st.cars;
  const [train, car] = st.fast ? [r.hsRunCostTrainKm, r.hsRunCostCarKm] : [r.runCostTrainKm, r.runCostCarKm];
  const perKm = st.bus?.ferry ? boatRunKm(r, boatOf(st.bus)) : st.bus ? r.runCostBusKm : train + cars * car;
  let run = 0;
  PERIODS.forEach((_, i) => { run += tph[i] * (st.oneWay ? 1 : 2) * st.km * perKm * r.economy * PERIOD_HOURS[i]; });
  run *= runFactorOf(runShares, zoneCountry); // T-2669, as `lineRunFactor`
  return { l, st, money, dm, runShares, r, fares, run };
}

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
  const days = DAY_KINDS.map((k) => mainDemandByDay.value[kindMade(k, near, rush) ? k : PLAIN_KIND[k]]);
  const was = pointsMemo, same = !!was && was.w === w && was.zl === zl;
  if (same && days.every((d, k) => sameDay(d, was.days[k]))) return was.out;
  const own = days.map((d, k) => (!kindMade(k, near, rush) ? null : same && sameDay(d, was.days[k]) ? was.per[k] : dayPoints(w, d, k)));
  if (checkPatches()) own.forEach((x, k) => { if (x && x !== was?.per[k] && !sameBuilt(x, freshly(() => dayPoints(w, days[k], k)))) console.error("T-4150 check: the points kept per line differ from a whole build"); });
  const per = own.map((x, k) => x ?? own[PLAIN_KIND[k]]!);
  const base = days.findIndex((d) => d.view);
  const out = base < 0 ? [] : assemble(days, per, base);
  if (checkPatches() && base >= 0 && !sameBuilt(out, freshly(() => assemble(days, per, base)))) console.error("T-4284 check: the points' entries kept per line differ from a whole build");
  pointsMemo = { w, zl, days, per, out };
  return out;
});
/** Every running line's points per game hour, `[region, perHour[dayKind][period]]` by line number,
 * as the clock worker settles them, times the line's cover (`lineCover`, the weekday's on every kind of day). A kind of
 * day not solved yet takes the weekday's. A day is redone only when its views, the network or the zone boundaries
 * change (T-1209). */
let coveredLast = new Map<number, { x: LinePoints[number]; c: number; out: LinePoints[number] }>();
export const linePoints = computed<LinePoints>(() => {
  const cover = lineCover.value;
  const next = new Map<number, { x: LinePoints[number]; c: number; out: LinePoints[number] }>();
  const out = fullPoints.value.map((x) => {
    const c = cover.get(x.line) ?? 1;
    // T-4150: a line's entry kept while its points and cover are the same
    let e = coveredLast.get(x.line);
    if (!e || e.x !== x || e.c !== c) e = { x, c, out: c === 1 ? { line: x.line, regions: x.regions } : { line: x.line, regions: x.regions.map(([z, per]): [string, number[][]] => [z, per.map((d) => d.map((v) => v * c))]) } };
    next.set(x.line, e);
    return e.out;
  });
  coveredLast = next;
  return out;
});

/** T-4150: each line's entry, kept while its days' points are the same objects (T-4284: compared one by one) */
let assembledLast = new Map<number, { base: number; has: boolean[]; per: (Map<string, number[]> | undefined)[]; out: LinePoints[number] }>();
function assemble(days: DayDemand[], per: Map<number, Map<string, number[]>>[], base: number): LinePoints {
  const lines = new Set<number>();
  for (const m of per) for (const line of m.keys()) lines.add(line);
  const has = DAY_KINDS.map((k) => !!days[k].view);
  const fresh = freshNow(), next: typeof assembledLast = new Map(), out: LinePoints = [];
  for (const line of lines) {
    let e = fresh ? undefined : assembledLast.get(line);
    let same = !!e && e.base === base;
    for (let k = 0; same && k < per.length; k++) if (e!.per[k] !== per[k].get(line) || e!.has[k] !== has[k]) same = false;
    if (!same) {
      const zones = new Set(per.flatMap((m) => [...(m.get(line)?.keys() ?? [])]));
      const zero = PERIODS.map(() => 0);
      e = { base, has, per: per.map((m) => m.get(line)), out: { line, regions: [...zones].map((z): [string, number[][]] => [z, DAY_KINDS.map((k) => (days[k].view ? per[k] : per[base]).get(line)?.get(z) ?? zero)]) } };
    }
    next.set(line, e!);
    out.push(e!.out);
  }
  if (!fresh) assembledLast = next;
  return out;
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
  for (const x of linePoints.value) {
    // (T-4284: each line's sums a day per region, kept per entry, added in line order)
    let sums = daySums.get(x);
    if (!sums) daySums.set(x, (sums = x.regions.map(([z, per]): [string, number[]] => [z, DAY_KINDS.map((k) => per[k].reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0))])));
    for (const [z, s] of sums) {
      let r = out.get(z);
      if (!r) out.set(z, (r = DEMAND_DAYS.map(() => 0)));
      for (const k of DAY_KINDS) r[k] += s[k];
    }
  }
  for (const r of out.values()) r[AVERAGE_DAY] = (5 * r[0] + r[1] + r[2]) / 7;
  if (checkPatches()) {
    const whole = new Map<string, number[]>();
    for (const { regions } of linePoints.peek()) for (const [z, per] of regions) {
      let r = whole.get(z);
      if (!r) whole.set(z, (r = DEMAND_DAYS.map(() => 0)));
      for (const k of DAY_KINDS) r[k] += per[k].reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0);
    }
    for (const r of whole.values()) r[AVERAGE_DAY] = (5 * r[0] + r[1] + r[2]) / 7;
    if (!sameBuilt(out, whole)) console.error("T-4284 check: the regions' points a day kept per line differ from a whole build");
  }
  return out;
});
const daySums = new WeakMap<LinePoints[number], [string, number[]][]>();

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
  const dv = mainDemandByDay.value[0]?.view;
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
