// Money as the UI reads it (T-028; SPEC 6.4). The clock worker keeps the cash, the cars owned,
// the fare curve and the day ledger, and sends them with every state and whenever an hour ends
// (`moneyState`), and the cash and today's ledger several times a wall second (`liveMoney`, T-284).
// Fare income is worked out here, on the main thread, because the riders are here
// (game/demand.ts): each period's riders times the fare curve, sent back to the worker as income
// per game hour (workers/clockClient.ts). The worker pays each hour's fares into the cash as that
// hour's trains pick up riders, and its running cost evenly.
//
// Fare per ride = base + per km x the ride's length along the line. A ride that changes lines
// pays the base once: boardings are scaled to trips by the city's rail trips over all boardings.
//
// The economy runs at 100x (T-081, `ECONOMY` in sim/src/track/params.rs, sent as
// `MoneyState.economy`): a game day's fares and running costs are that many times a real day's,
// so a line pays back in about a game week. The fare the player sets stays a real fare.

import { computed, signal } from "@preact/signals";
import { DAY_KINDS } from "./days";
import { demandByDay, type DemandView } from "./demand";
import { PERIODS } from "./periods";
import { world } from "./world";
import type { MoneyState } from "../workers/protocol";
import type { DayLedger, HourLedger } from "../workers/saveFile";

export type { MoneyState };

export const moneyState = signal<MoneyState | null>(null);
/** T-284: the cash and today's ledger as they move, several times a wall second while the clock
 * runs (the worker's "cash" message; also set from every `moneyState`). Only the figures that show
 * them read it (the top bar's cash, the Money tab's cash and today), each its own small component,
 * so nothing else re-renders as fares land. */
export const liveMoney = signal<{ cash: number; today: DayLedger | null; hour: HourLedger | null }>({ cash: 0, today: null, hour: null });
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
}

export interface Income {
  /** fare income per game hour in each period, US$M */
  perHour: number[];
  lines: Map<string, LineMoney>;
}

/** Fare income on each kind of day; the worker settles each hour by its own day's (T-260). The
 * average day (T-464) is a view only, so it has none. */
export const incomeByDay = computed<Income[]>(() => demandByDay.value.slice(0, DAY_KINDS.length).map((d) => incomeFor(d.view)));

function incomeFor(dv: DemandView | null): Income {
  const w = world.value, m = moneyState.value;
  const perPeriod = PERIODS.map(() => 0);
  const lines = new Map<string, LineMoney>();
  if (!w || !m) return { perHour: perPeriod, lines };
  const { base, perKm } = m.fares;
  const x = m.economy;
  const boardings = w.save.lines.reduce((a, l) => a + (dv?.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const tripsPerBoarding = dv && boardings > 0 ? Math.min(1, dv.railTripsPerDay / boardings) : 1;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    const linePer = PERIODS.map(() => 0);
    const dm = dv?.lines.get(l.id);
    const n = l.stops.length;
    if (dm && st.stopS[0]?.length === n && st.stopS[1]?.length === n) {
      // segment lengths, km: run 0's segments, then run 1's in its own order (as `loads`)
      const km: number[] = [];
      for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push((s[k + 1] - s[k]) / 1000);
      const pkm = dm.loads.map((seg) => (seg.length === km.length ? seg.reduce((a, v, k) => a + v * km[k], 0) : 0));
      const total = pkm.reduce((a, b) => a + b, 0);
      const rides = dm.ridersPerDay * tripsPerBoarding;
      pkm.forEach((pk, p) => {
        const r = (base * rides * (total > 0 ? pk / total : 0) + perKm * pk) * x;
        perPeriod[p] += r;
        linePer[p] += r;
      });
    }
    lines.set(l.id, { perHour: linePer.map((v, p) => v / PERIOD_HOURS[p] / 1e6) });
  }
  return { perHour: perPeriod.map((v, p) => v / PERIOD_HOURS[p] / 1e6), lines };
}
