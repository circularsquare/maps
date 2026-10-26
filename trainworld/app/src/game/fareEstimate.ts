// T-1300: what a fare change would do to rail trips and fare income a day, shown while the player
// drags or edits the fare curve, until demand has solved the new fares. From demand's own fare
// sensitivity (`DemandView.fareSense`: each layer's rail trips per curve dollar on every fare, from
// the mode choice at its solved shares): each layer's rail trips go to exp(beta x dF) of today's,
// beta its sensitivity over its rail trips and dF the curve's change at its riders' mean trip length
// times its reach (the sensitivity-weighted mean trip km over the rider-weighted one), line by line
// and period by period (the market's on the longer trips' loads, the local layers' on the rest).
// Fare income: today's, plus the change on today's riders, times the riders kept.

import { computed, signal } from "@preact/signals";
import { demandUpdating, demandView, type DemandView } from "./demand";
import { fareAt, type FareNode } from "./fareCurve";
import { incomeByDay, moneyState, stopScales } from "./money";
import { DAY_KINDS } from "./days";
import { PERIODS } from "./periods";
import { stopsById, world } from "./world";

export interface FareEstimate {
  /** rail trips a day: as demand has them now, and at the new curve */
  riders: number;
  ridersNow: number;
  /** fare income a game day, US$ */
  fares: number;
  faresNow: number;
}

/** The curve the shown demand was solved with, from a fare edit until demand has solved it. */
const pending = signal<{ nodes: FareNode[]; version: number } | null>(null);

/** A fare curve edit is about to be sent: the shown demand keeps the curve before it until solved. */
export function noteFareEdit() {
  const m = moneyState.peek();
  if (!pending.peek() && m) pending.value = { nodes: m.fares.nodes.map((n) => [...n] as FareNode), version: demandView.peek()?.version ?? -1 };
}

/** Whether demand has solved every fare edit (and then forgets the curve before them). */
const settled = computed(() => {
  const p = pending.value, v = demandView.value;
  return !p || (!!v && v.final && v.version > p.version && !demandUpdating.value);
});

/** Over every running line, on `lines` (all riders or one purpose's), each period's trips times the
 * curve's change at their mean trip length: in curve dollars and as charged (times the line's fare
 * scale), and the trips. */
function curveChange(dv: DemandView, lines: Map<string, { ridersPerDay: number; boardings: number[]; loads: Float32Array[] }> | undefined, from: FareNode[], to: FareNode[], reach = 1) {
  const w = world.value, m = moneyState.value;
  let trips = 0, dCurve = 0, dCharged = 0;
  if (!w || !m || !lines) return { trips, dCurve, dCharged };
  const byId = stopsById(w);
  const boardings = w.save.lines.reduce((a, l) => a + (dv.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const perBoarding = boardings > 0 ? Math.min(1, dv.railTripsPerDay / boardings) : 1;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], dm = lines.get(l.id);
    const n = l.stops.length;
    if (!st || st.status !== "running" || !dm || st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
    const km: number[] = [];
    for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push((s[k + 1] - s[k]) / 1000);
    const sc = stopScales(l.stops, byId, m.fares.scales);
    const boarded = dm.boardings.reduce((a, b) => a + b, 0);
    const scale = boarded > 0 ? dm.boardings.reduce((a, b, k) => a + b * (sc[k] ?? 1), 0) / boarded : (sc[0] ?? 1);
    const pkm = dm.loads.map((seg) => (seg.length === km.length ? seg.reduce((a, v, k) => a + v * km[k], 0) : 0));
    const total = pkm.reduce((a, b) => a + b, 0);
    const rides = dm.ridersPerDay * perBoarding;
    if (!(total > 0 && rides > 0)) continue;
    for (const pk of pkm) {
      const t = (rides * pk) / total;
      if (!(t > 0)) continue;
      const d = fareAt(to, pk / t) - fareAt(from, pk / t);
      const dr = reach === 1 ? d : fareAt(to, (reach * pk) / t) - fareAt(from, (reach * pk) / t);
      trips += t;
      dCurve += t * dr;
      dCharged += t * d * scale;
    }
  }
  return { trips, dCurve, dCharged };
}

/** The estimate for the curve `to` (curve dollars, before any country's scale) against the curve
 * the shown demand was solved with; null without demand's sensitivity or fare income. */
export function estimateFares(to: FareNode[]): FareEstimate | null {
  const dv = demandView.value, m = moneyState.value;
  const fs = dv?.fareSense;
  if (!dv || !m || !fs || !(dv.railTripsPerDay > 0)) return null;
  const from = (settled.value ? null : pending.value?.nodes) ?? m.fares.nodes;
  const fin = (v: number, or = 0) => (Number.isFinite(v) ? v : or);
  // each layer's fare change at its trips as their sensitivity weighs them: a trip's length
  // stretched by the layer's reach (long trips weigh less a km than short ones, T-1300)
  const longerLines = dv.purposes?.get("longer")?.lines;
  const all = curveChange(dv, dv.lines, from, to, fin(fs.localReach, 1));
  const longerLocal = curveChange(dv, longerLines, from, to, fin(fs.localReach, 1));
  const longer = curveChange(dv, longerLines, from, to, fin(fs.marketReach, 1));
  const mean = (d: number, t: number) => (t > 0 ? d / t : all.trips > 0 ? all.dCurve / all.trips : 0);
  const rm = Math.min(fin(fs.marketRail), dv.railTripsPerDay), rl = dv.railTripsPerDay - rm;
  const kept = (r: number, sense: number, dF: number) => (r > 0 ? r * Math.exp((fin(sense) / r) * dF) : 0);
  const riders = kept(rl, fs.local, mean(all.dCurve - longerLocal.dCurve, all.trips - longerLocal.trips)) + kept(rm, fs.market, mean(longer.dCurve, longer.trips));
  // the day shown's fare income, a game day (the average day has none of its own: the weekday's)
  const day = dv.day < DAY_KINDS.length ? dv.day : 0;
  const inc = incomeByDay.value[day];
  if (!inc) return null;
  const faresNow = inc.perHour.reduce((a, v, p) => a + v * (PERIODS[p].to - PERIODS[p].from), 0) * 1e6;
  const sameRiders = faresNow + all.dCharged * m.economy;
  return { riders, ridersNow: dv.railTripsPerDay, fares: Math.max(0, sameRiders * (riders / dv.railTripsPerDay)), faresNow };
}

/** The estimate while the player drags the curve (`preview`, curve dollars) or demand has yet to
 * solve a fare edit; null otherwise. */
export function shownEstimate(preview: FareNode[] | null): FareEstimate | null {
  const m = moneyState.value;
  if (!m) return null;
  if (!preview && settled.value) {
    if (pending.peek()) queueMicrotask(() => { if (settled.peek()) pending.value = null; });
    return null;
  }
  return estimateFares(preview ?? m.fares.nodes);
}
