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
import { AWAY_DAY, awayNear, DAY_KINDS } from "./days";
import { demandByDay, type DemandView } from "./demand";
import { cityById, isCity } from "./cities";
import { faresFor, faresOf } from "./fareCurve";
import { fareScale } from "./income";
import { stationZone } from "./listScope";
import { PERIODS } from "./periods";
import { stationsById, world } from "./world";
import { zonesLoaded } from "./zones";
import type { LineInput, StationInput, WorldView } from "./types";
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

/** What the money, points and region figures read of `moneyState`: the same object while these are
 * the same, so a state or an hour's money that leaves them alone redoes no sums (T-1209). */
export type MoneyTerms = Pick<MoneyState, "fares" | "economy" | "upkeepTrackKm" | "upkeepStation" | "upkeepSingle" | "upkeepHsr">;
let termsKey = "", terms: MoneyTerms | null = null;
export const moneyTerms = computed<MoneyTerms | null>(() => {
  const m = moneyState.value;
  if (!m) return (termsKey = "", terms = null);
  const t: MoneyTerms = { fares: m.fares, economy: m.economy, upkeepTrackKm: m.upkeepTrackKm, upkeepStation: m.upkeepStation, upkeepSingle: m.upkeepSingle, upkeepHsr: m.upkeepHsr };
  const key = JSON.stringify(t);
  if (key !== termsKey || !terms) { termsKey = key; terms = t; }
  return terms;
});

const sameList = (a: readonly unknown[] | undefined, b: readonly unknown[] | undefined) => {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};
/** The same network as these figures read it: each line's id, number, stops, status and stop
 * positions, each station's place and whether it is built. */
function sameShape(a: WorldView, b: WorldView): boolean {
  const la = a.save.lines, lb = b.save.lines, sa = a.save.stations, sb = b.save.stations;
  if (la.length !== lb.length || sa.length !== sb.length) return false;
  for (let i = 0; i < sa.length; i++) {
    const x = sa[i], y = sb[i];
    if (x.id !== y.id || x.lng !== y.lng || x.lat !== y.lat || x.built !== y.built) return false;
  }
  for (let i = 0; i < la.length; i++) {
    const x = la[i], y = lb[i];
    if (x.id !== y.id || x.num !== y.num || !sameList(x.stops, y.stops)) return false;
    const p = a.lineStats[x.id], q = b.lineStats[y.id];
    if (!p || !q) { if (p !== q) return false; continue; }
    if (p.status !== q.status || !sameList(p.stopS[0], q.stopS[0]) || !sameList(p.stopS[1], q.stopS[1])) return false;
  }
  return true;
}
let shaped: WorldView | null = null;
/** `world` for the money, points and region figures: the last world while its network is the same
 * as they read it (`sameShape`), so the clock's states redo their sums only when that changes. */
export const moneyWorld = computed<WorldView | null>(() => {
  const w = world.value;
  if (!w || !shaped || !sameShape(shaped, w)) shaped = w;
  return shaped;
});

let incomeMemo: { w: WorldView | null; t: MoneyTerms | null; zl: number; views: (DemandView | null)[]; out: Income[] } | null = null;
/** Fare income on each kind of day; the worker settles each hour by its own day's (T-260). The
 * average day (T-464) is a view only, so it has none. A day whose view, network and fares are as
 * before keeps its figures. */
export const incomeByDay = computed<Income[]>(() => {
  const w = moneyWorld.value, t = moneyTerms.value, zl = zonesLoaded.value;
  // T-264: the away day's own figures only while one is near, else Sunday's
  const soon = awayNear.value !== null;
  const views = demandByDay.value.slice(0, DAY_KINDS.length).map((d, k) => (k === AWAY_DAY && !soon ? null : d.view));
  const was = incomeMemo, same = !!was && was.w === w && was.t === t && was.zl === zl;
  if (same && sameList(views, was.views)) return was.out;
  const own = views.map((v, k) => (k === AWAY_DAY && !soon ? null : same && v === was.views[k] ? was.out[k] : incomeFor(w, t, v)));
  const out = own.map((x) => x ?? own[2]!);
  incomeMemo = { w, t, zl, views, out };
  return out;
});

function incomeFor(w: WorldView | null, m: MoneyTerms | null, dv: DemandView | null): Income {
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
      const g = lineGeo(w, l, byId), km = g.km;
      // T-1277: a stretch pays its starting stop's fare scale, the base its boarding stop's
      const { sc, from } = scalesOf(g, l, byId, m.fares.scales);
      const pkm: number[] = [], pkmPaid: number[] = [];
      for (const seg of dm.loads) {
        let a = 0, b = 0;
        if (seg.length === km.length) for (let k = 0; k < seg.length; k++) { a += seg[k] * km[k]; b += seg[k] * km[k] * from[k]; }
        pkm.push(a);
        pkmPaid.push(b);
      }
      let total = 0;
      for (const v of pkm) total += v;
      const rides = dm.ridersPerDay * tripsPerBoarding;
      let boarded = 0, scaled = 0;
      for (let k = 0; k < dm.boardings.length; k++) { boarded += dm.boardings[k]; scaled += dm.boardings[k] * (sc[k] ?? 1); }
      const baseScale = boarded > 0 ? scaled / boarded : (sc[0] ?? 1);
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

/** What the figures read of a running line with both runs' stop positions: its stretches' km, run
 * 0's then run 1's in its own order (as `LineDemand.loads`), signed and not, and its stops' zones
 * ("" outside every zone); built once per network and zone boundaries (T-1209). */
export interface LineGeo {
  km: number[];
  kmAbs: number[];
  zones: string[];
  scales?: { of: Record<string, number> | undefined; sc: number[]; from: number[] };
}
let geoOf: { w: WorldView; zl: number; lines: Map<string, LineGeo> } | null = null;
export function lineGeo(w: WorldView, l: LineInput, byId: Map<string, StationInput>): LineGeo {
  const zl = zonesLoaded.peek();
  if (!geoOf || geoOf.w !== w || geoOf.zl !== zl) geoOf = { w, zl, lines: new Map() };
  let g = geoOf.lines.get(l.id);
  if (g) return g;
  const n = l.stops.length, km: number[] = [], kmAbs: number[] = [];
  for (const s of w.lineStats[l.id]?.stopS ?? []) for (let k = 0; k + 1 < n; k++) { km.push((s[k + 1] - s[k]) / 1000); kmAbs.push(Math.abs(s[k + 1] - s[k]) / 1000); }
  const zones = l.stops.map((id) => { const s = byId.get(id); return (s && stationZone(s)) ?? ""; });
  geoOf.lines.set(l.id, (g = { km, kmAbs, zones }));
  return g;
}
/** A line's stops' fare scales (`stopScales`) and each stretch's, its starting stop's (T-1277). */
export function scalesOf(g: LineGeo, l: LineInput, byId: Map<string, StationInput>, of: Record<string, number> | undefined) {
  if (g.scales?.of !== of || !g.scales) {
    const sc = stopScales(l.stops, byId, of), n = l.stops.length;
    g.scales = { of, sc, from: [...sc.slice(0, n - 1), ...[...sc].reverse().slice(0, n - 1)] };
  }
  return g.scales;
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
