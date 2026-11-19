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
import type { WorldView } from "./types";
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
