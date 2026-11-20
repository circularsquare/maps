// T-2626 (design in notes/T-2604.md): the network's profit change a day from a trains, headway or cars
// edit, before demand has solved it (the edit note's estimating state, game/editNote.ts). Running cost is
// exact: the clock's running cost per hour now, against the day's at the last settled state. Fares move
// by the wait term: each boarding waits half the headway, so a period's riders change by
// `boardings x (exp(e x dWait) - 1)`, e the per-minute value of `DemandView.fareSense.local` over the
// rail trips (the mode choice's own sensitivity, as game/fareEstimate.ts uses for a fare edit), and each
// trip gained or lost pays the network's mean fare a trip. A cars edit moves no wait, so it is running
// cost alone. Everything is read against the last settled state (`EstimateBase`), so several edits add up
// and a line's early local result does not move the figure. Fares sum to the shown day's.

import type { DemandView } from "./demand";
import { faresIn } from "./fareDefault";
import { fareScale, fareWeight } from "./income";
import { lineCountry } from "./holidays";
import { stationCity, type Income } from "./money";
import { stopsById } from "./world";
import { PERIODS } from "./periods";
import { trainKind } from "./format";
import { boatOf, boatRunKm } from "./boats";
import { lineRunFactor, lineShares, upkeepShares } from "./regionMoney";
import type { LineInput, LineStats, WorldView } from "./types";
import type { MoneyState } from "../workers/protocol";

const PERIOD_HOURS = PERIODS.map((p) => p.to - p.from);

/** What the last settled state gives the estimate. */
export interface EstimateBase {
  /** running cost a day then, US$ */
  running: number;
  /** per line id: the trains an hour (in each period, T-3542) its riders were solved with, and its boardings per period */
  lines: Map<string, { tph: number[]; board: number[] }>;
  /** per perceived minute added to every rail trip: the share of rail trips lost (negative) */
  perMinute: number;
  /** fare income a trip, US$ */
  farePerTrip: number;
}

/** Running cost a day, US$: the clock's running cost an hour in each of the day's periods. */
export function runningADay(m: MoneyState): number {
  let run = 0;
  PERIODS.forEach((_, i) => { run += m.runningPerHour[i] * PERIOD_HOURS[i]; });
  return run * 1e6;
}

/** The network's profit a day, US$: one kind of day's fare income (`inc`) minus the running cost and the upkeep, leaving
 * out new trains, construction and the loan (as Cities' `net`). The edit note's profit and, on the weekday's income, the
 * profit leaderboard's figure (T-3879). */
export function dayProfit(inc: Income, m: MoneyState): number {
  const fares = inc.perHour.reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0) * 1e6;
  return fares - runningADay(m) - (m.upkeep.track + m.upkeep.stations + m.upkeep.buses) * 1e6;
}

/** A running line's running cost over a whole day at its trains now, US$ (as `runningToday`, game/today.ts). */
export function lineRunningADay(l: LineInput, st: LineStats, m: MoneyState): number {
  const k = trainKind(m, st.fast, st.tram);
  const perKm = st.bus?.ferry ? boatRunKm(m, boatOf(st.bus)) : st.bus ? m.runCostBusKm : k.runCostTrainKm + st.cars * k.runCostCarKm;
  let run = 0;
  PERIODS.forEach((_, i) => { run += l.tphP[i] * 2 * st.km * perKm * m.economy * PERIOD_HOURS[i]; });
  return run * lineRunFactor(l.num); // T-2669
}

/** T-3885: `dayProfit` by region (a zone id; "" outside every zone), US$, split as the region ledgers are
 * (regionMoney.ts): each running line's fares by its fare shares, its running cost by its route km (scaled so the regions
 * add up to the clock's running cost), and the upkeep by where the track and stations are. Adds up to `dayProfit`. */
export function dayProfitByRegion(w: WorldView, inc: Income, m: MoneyState): Map<string, number> {
  const out = new Map<string, number>(), run = new Map<string, number>();
  const add = (map: Map<string, number>, k: string, v: number) => map.set(k, (map.get(k) ?? 0) + v);
  const shares = lineShares.peek();
  let runAll = 0;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    const sh = shares.get(l.num);
    const f = (inc.lines.get(l.id)?.perHour ?? []).reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0) * 1e6;
    for (const [z, s] of sh?.fare.length ? sh.fare : [["", 1] as [string, number]]) add(out, z, f * s);
    const r = lineRunningADay(l, st, m);
    runAll += r;
    for (const [z, s] of sh?.run.length ? sh.run : [["", 1] as [string, number]]) add(run, z, r * s);
  }
  // running cost scaled to the clock's
  const total = dayProfit(inc, m), upkeep = (m.upkeep.track + m.upkeep.stations + m.upkeep.buses) * 1e6;
  const scale = runAll > 0 ? runningADay(m) / runAll : 0;
  for (const [z, r] of run) add(out, z, -r * scale);
  const up = upkeepShares.peek();
  for (const [z, s] of up.length ? up : [["", 1] as [string, number]]) add(out, z, -upkeep * s);
  // what the splits leave over (a running cost with no running line, rounding) is outside every zone
  let sum = 0;
  for (const v of out.values()) sum += v;
  add(out, "", total - sum);
  return out;
}

/** The base from the settled demand `dv` and its income `inc`; fares are left out (sensitivity 0) where
 * demand carries no fare sensitivity. */
export function estimateBase(w: WorldView, dv: DemandView, inc: Income, m: MoneyState): EstimateBase {
  const lines = new Map<string, { tph: number[]; board: number[] }>();
  const byId = stopsById(w);
  let boardings = 0, minutes = 0;
  for (const l of w.save.lines) {
    const ld = dv.lines.get(l.id);
    if (!ld || w.lineStats[l.id]?.status !== "running") continue;
    const board = PERIODS.map((_, p) => { const b = ld.board?.[p]; let s = 0; if (b) for (let k = 0; k < b.length; k++) s += b[k]; return s; });
    lines.set(l.id, { tph: ld.tph ?? l.tphP, board });
    // the minutes a curve dollar costs, by the country the line is in, as the solve's sensitivity weighed it
    const c = lineCountry(l.stops) ?? undefined, b = board.reduce((a, v) => a + v, 0);
    // T-3766: with fares by city, its first stop's city's
    const first = m.fares.byCity ? byId.get(l.stops[0]) : undefined;
    boardings += b;
    minutes += b * fareWeight(c) * fareScale(c, faresIn(m.fares, first ? stationCity(first) : null).scales);
  }
  const fares = inc.perHour.reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0) * 1e6;
  const rail = dv.railTripsPerDay, local = dv.fareSense?.local ?? 0;
  const perMinute = boardings > 0 && rail > 0 && Number.isFinite(local) ? local / (minutes / boardings) / rail : 0;
  return { running: runningADay(m), lines, perMinute, farePerTrip: rail > 0 ? fares / rail : 0 };
}

export interface ProfitEstimate {
  /** running cost a day, US$, added (negative: saved) */
  running: number;
  /** fare income a day, US$, added */
  fares: number;
}

/** The change against `b` with the network as `w` and `m` have it now. */
export function estimateChange(b: EstimateBase, w: WorldView, m: MoneyState): ProfitEstimate {
  let trips = 0;
  for (const l of w.save.lines) {
    const was = b.lines.get(l.id);
    if (!was || w.lineStats[l.id]?.status !== "running") continue;
    const now = l.tphP;
    for (let p = 0; p < PERIODS.length; p++) {
      const t0 = was.tph[p], t1 = now[p];
      if (!(t0 > 0) || t1 === t0 || !(was.board[p] > 0)) continue;
      // a line with no trains left loses all its riders
      trips += was.board[p] * (t1 > 0 ? Math.exp(b.perMinute * 0.5 * (60 / t1 - 60 / t0)) - 1 : -1);
    }
  }
  return { running: runningADay(m) - b.running, fares: trips * b.farePerTrip };
}
