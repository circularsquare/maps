// Money as the UI reads it (T-028; SPEC 6.4). The clock worker keeps the cash, the cars owned,
// the fare curve and the day ledger, and sends them with every state and whenever an hour ends
// (`moneyState`), and the cash and today's ledger several times a wall second (`liveMoney`, T-284).
// Fare income is worked out here, on the main thread, because the riders are here
// (game/demand.ts): each period's riders times the fare curve, sent back to the worker as income
// per game hour (workers/clockClient.ts). The worker pays each hour's fares into the cash as that
// hour's trains pick up riders, and its running cost evenly.
//
// Fare per ride = base + per km x the ride's length along the line, on the fare curve's first
// stretch. A ride that changes lines pays the base once: boardings are scaled to trips by the
// city's rail trips over all boardings. What the rest of a bent curve adds (or takes off) on each
// trip's length is demand's own figure, at the line the trip first boards (`LineDemand.fareX`,
// T-1143), so income is what demand charged.
//
// The economy runs at 100x (T-081, `ECONOMY` in sim/src/track/params.rs, sent as
// `MoneyState.economy`): a game day's fares and running costs are that many times a real day's,
// so a line pays back in about a game week. The fare the player sets stays a real fare.

import { computed, signal } from "@preact/signals";
import { DAY_KINDS } from "./days";
import { demandByDay, type DemandView } from "./demand";
import { cityById, isCity } from "./cities";
import { faresFor, faresOf } from "./fareCurve";
import { fareScale } from "./income";
import { stationZone } from "./listScope";
import { PERIODS } from "./periods";
import { stationsById, world } from "./world";
import type { StationInput } from "./types";
import type { MoneyState, RegionNow } from "../workers/protocol";
import type { DayLedger, HourLedger } from "../workers/saveFile";

export type { MoneyState };

export const moneyState = signal<MoneyState | null>(null);
/** T-284: the cash and today's ledger as they move, several times a wall second while the clock
 * runs (the worker's "cash" message; also set from every `moneyState`). Only the figures that show
 * them read it (the top bar's cash, the Money tab's cash and today), each its own small component,
 * so nothing else re-renders as fares land. */
export const liveMoney = signal<{ cash: number; today: DayLedger | null; hour: HourLedger | null; regions?: RegionNow }>({ cash: 0, today: null, hour: null });
export function setMoneyState(m: MoneyState) {
  moneyState.value = m;
  liveMoney.value = { cash: m.cash, today: m.ledger[m.ledger.length - 1] ?? null, hour: m.hours[m.hours.length - 1] ?? null };
}
/** When the last autosave was written (Date.now()), null before the first. */
export const lastSaved = signal<number | null>(null);

const PERIOD_HOURS = PERIODS.map((p) => p.to - p.from);

export interface LineMoney {
  /** fare income per game hour in each period, US$M (the worker's fares taken today, T-293) */
  perHour: number[];
  /** train rides per game hour in each period (T-565) */
  rides: number[];
}

export interface Income {
  /** fare income per game hour in each period, US$M */
  perHour: number[];
  /** train rides per game hour in each period, by the share of each period's passenger-km (T-565) */
  rides: number[];
  lines: Map<string, LineMoney>;
}

/** Fare income on each kind of day; the worker settles each hour by its own day's (T-260). The
 * average day (T-464) is a view only, so it has none. */
export const incomeByDay = computed<Income[]>(() => demandByDay.value.slice(0, DAY_KINDS.length).map((d) => incomeFor(d.view)));

function incomeFor(dv: DemandView | null): Income {
  const w = world.value, m = moneyState.value;
  const perPeriod = PERIODS.map(() => 0), ridesPeriod = PERIODS.map(() => 0);
  const lines = new Map<string, LineMoney>();
  if (!w || !m) return { perHour: perPeriod, rides: ridesPeriod, lines };
  const fares = faresOf(m.fares, m.fares);
  const x = m.economy;
  const byId = stationsById(w);
  const boardings = w.save.lines.reduce((a, l) => a + (dv?.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const tripsPerBoarding = dv && boardings > 0 ? Math.min(1, dv.railTripsPerDay / boardings) : 1;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    const linePer = PERIODS.map(() => 0), lineRides = PERIODS.map(() => 0);
    const dm = dv?.lines.get(l.id);
    const n = l.stops.length;
    if (dm && st.stopS[0]?.length === n && st.stopS[1]?.length === n) {
      // segment lengths, km: run 0's segments, then run 1's in its own order (as `loads`)
      const km: number[] = [];
      for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push((s[k + 1] - s[k]) / 1000);
      // T-1277: each stop's fare scale; a stretch pays its starting stop's, the base its boarding stop's
      const sc = stopScales(l.stops, byId, m.fares.scales);
      const from = [...sc.slice(0, n - 1), ...[...sc].reverse().slice(0, n - 1)];
      const pkm = dm.loads.map((seg) => (seg.length === km.length ? seg.reduce((a, v, k) => a + v * km[k], 0) : 0));
      const pkmPaid = dm.loads.map((seg) => (seg.length === km.length ? seg.reduce((a, v, k) => a + v * km[k] * from[k], 0) : 0));
      const total = pkm.reduce((a, b) => a + b, 0);
      const rides = dm.ridersPerDay * tripsPerBoarding;
      const boarded = dm.boardings.reduce((a, b) => a + b, 0);
      const baseScale = boarded > 0 ? dm.boardings.reduce((a, b, k) => a + b * (sc[k] ?? 1), 0) / boarded : (sc[0] ?? 1);
      pkm.forEach((pk, p) => {
        const share = total > 0 ? pk / total : 0;
        const r = (faresFor(fares, rides * share * baseScale, pkmPaid[p]) + (dm.fareX?.[p] ?? 0)) * x;
        perPeriod[p] += r;
        linePer[p] += r;
        ridesPeriod[p] += rides * share;
        lineRides[p] += rides * share;
      });
    }
    lines.set(l.id, { perHour: linePer.map((v, p) => v / PERIOD_HOURS[p] / 1e6), rides: lineRides.map((v, p) => v / PERIOD_HOURS[p]) });
  }
  return { perHour: perPeriod.map((v, p) => v / PERIOD_HOURS[p] / 1e6), rides: ridesPeriod.map((v, p) => v / PERIOD_HOURS[p]), lines };
}

/** T-1277: the country of the zone a station is in (null outside every zone). */
export function stationCountry(s: Pick<StationInput, "id" | "lng" | "lat">): string | null {
  const z = stationZone(s);
  return z && isCity(z) ? cityById(z).country ?? null : null;
}
/** T-1277: each stop's fare scale, by the country it is in; a stop outside every zone takes the
 * baseline's. */
export function stopScales(stops: string[], byId: Map<string, StationInput>, scales: Record<string, number> | undefined): number[] {
  return stops.map((id) => {
    const s = byId.get(id);
    return fareScale((s && stationCountry(s)) ?? undefined, scales);
  });
}
