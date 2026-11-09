// Money as the UI reads it (T-028; SPEC 6.4). The clock worker keeps the cash, the cars owned,
// the fare curve and the day ledger, and sends them with every state and whenever an hour ends
// (`moneyState`), and the cash and today's ledger several times a wall second (`liveMoney`, T-284).
// Fare income is worked out here, on the main thread, because the riders are here
// (game/demand.ts): each period's riders times the fare curve, sent back to the worker as income
// per game hour (workers/clockClient.ts). The worker pays each hour's fares into the cash as that
// hour's trains pick up riders, and its running cost evenly.
//
// Fare per ride = base + per km x the ride's length along the line, on the fare curve's first
// stretch. A ride that changes lines pays the base once: the network's base fares (its boardings
// scaled to trips by rail trips over all boardings) are shared between lines by `baseTrips`
// (T-2104). What the rest of a bent curve adds (or takes off) on each
// trip's length is demand's own figure, shared between the lines the trip rides by their km
// (`LineDemand.fareX`, T-1143, T-1516), so income is what demand charged and no line's is below its
// base and rate.
//
// The economy runs at 100x (T-081, `ECONOMY` in sim/src/track/params.rs, sent as
// `MoneyState.economy`): a game day's fares and running costs are that many times a real day's,
// so a line pays back in about a game week. The fare the player sets stays a real fare.

import { computed, signal } from "@preact/signals";
import { awayNear, DAY_KINDS, kindMade, PLAIN_KIND, rushRun } from "./days";
import { demandByDay, lineLoadsFor, type DemandView } from "./demand";
import { cityById, isCity } from "./cities";
import { faresFor, faresOf, lineFareMul, type Fares } from "./fareCurve";
import { roundedDefault } from "./fareDefault";
import { lineFares } from "./fareInEffect";
import { fareScale } from "./income";
import { stationZone } from "./listScope";
import { PERIODS } from "./periods";
import { setCarBuyCheck, setTrainBuyCheck, stopsById, world } from "./world";
import { zonesLoaded } from "./zones";
import type { LineId, LineInput, LineStats, StationInput, WorldView } from "./types";
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
/** T-1797: the rule the worker applies to a train count edit (`apply_paid`): a running line pays for
 * the cars its busiest level needs beyond the cars the network owns, and the edit is refused when
 * that comes to more than the cash. A sandbox is never refused. */
export function trainShortfall(id: LineId, trains: readonly number[]): { need: number; have: number } | null {
  const st = world.peek()?.lineStats[id];
  if (!st) return null;
  const more = Math.max(...trains) - Math.max(...st.trainsNeeded);
  return more > 0 ? carShortfall(id, more * (st.bus ? 1 : st.cars)) : null;
}
/** The same rule for `more` cars (buses on a bus line) a running line would need beyond the ones in
 * use, whether from more trains or longer ones (T-1844). */
export function carShortfall(id: LineId, more: number): { need: number; have: number } | null {
  const m = moneyState.peek(), st = world.peek()?.lineStats[id];
  if (!m || !st || st.status !== "running" || m.mode === "sandbox" || !(more > 0)) return null;
  const [owned, needed, price] = st.bus?.ferry ? [m.fleetFerry, m.boatsNeeded, m.boatPrice] : st.bus ? [m.fleetBus, m.busesNeeded, m.busPrice]
    : st.fast ? [m.fleetFast, m.carsNeededFast, m.hsCarPrice] : [m.fleet, m.carsNeeded, m.carPrice];
  const need = Math.max(0, needed + more - owned) * price, have = liveMoney.peek().cash;
  return need > 1e-9 && need > have + 1e-6 ? { need, have } : null;
}
setTrainBuyCheck(trainShortfall);
setCarBuyCheck(carShortfall);
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
export type MoneyTerms = Pick<MoneyState, "fares" | "economy" | "upkeepTrackKm" | "upkeepViaductKm" | "upkeepCutKm" | "upkeepBoredKm" | "upkeepStation" | "upkeepStationUnder" | "upkeepSingle" | "upkeepHsr" | "upkeepBusKm" | "upkeepPort">;
let termsKey = "", terms: MoneyTerms | null = null;
export const moneyTerms = computed<MoneyTerms | null>(() => {
  const m = moneyState.value;
  if (!m) return (termsKey = "", terms = null);
  const t: MoneyTerms = { fares: m.fares, economy: m.economy, upkeepTrackKm: m.upkeepTrackKm, upkeepViaductKm: m.upkeepViaductKm, upkeepCutKm: m.upkeepCutKm, upkeepBoredKm: m.upkeepBoredKm, upkeepStation: m.upkeepStation, upkeepStationUnder: m.upkeepStationUnder, upkeepSingle: m.upkeepSingle, upkeepHsr: m.upkeepHsr, upkeepBusKm: m.upkeepBusKm, upkeepPort: m.upkeepPort };
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
    // T-1626: high-speed trains change the line's fares
    if (p.status !== q.status || p.fast !== q.fast || !sameList(p.stopS[0], q.stopS[0]) || !sameList(p.stopS[1], q.stopS[1])) return false;
    if (p.bus !== q.bus && !sameBus(p.bus, q.bus)) return false;
  }
  return true;
}
/** T-1464: a bus line's stops in the same places, built or not, the same road length. */
function sameBus(a: LineStats["bus"], b: LineStats["bus"]): boolean {
  if (!a || !b) return a === b;
  if (a.built !== b.built || a.roadM !== b.roadM || a.stops.length !== b.stops.length) return false;
  return a.stops.every((s, i) => s.id === b.stops[i].id && s.lng === b.stops[i].lng && s.lat === b.stops[i].lat);
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
  // T-264, T-1373: a superholiday's days have their own figures only while one is near, else the
  // ordinary day's they stand in for
  const near = awayNear.value, rush = rushRun.value;
  const views = demandByDay.value.slice(0, DAY_KINDS.length).map((d, k) => (kindMade(k, near, rush) ? d.view : null));
  const was = incomeMemo, same = !!was && was.w === w && was.t === t && was.zl === zl;
  if (same && sameList(views, was.views)) return was.out;
  const own = views.map((v, k) => (!kindMade(k, near, rush) ? null : same && v === was.views[k] ? was.out[k] : incomeFor(w, t, v, was?.out[k])));
  const out = own.map((x, k) => x ?? own[PLAIN_KIND[k]]!);
  incomeMemo = { w, t, zl, views, out };
  return out;
});

function incomeFor(w: WorldView | null, m: MoneyTerms | null, dv: DemandView | null, prev?: Income): Income {
  const perPeriod = PERIODS.map(() => 0), ridesPeriod = PERIODS.map(() => 0);
  const lines = new Map<string, LineMoney>();
  if (!w || !m) return { perHour: perPeriod, rides: ridesPeriod, lines };
  const curves = new Map<Fares, Fares>();
  const x = m.economy;
  const byId = stopsById(w);
  const boardings = w.save.lines.reduce((a, l) => a + (dv?.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const tripsPerBoarding = dv && boardings > 0 ? Math.min(1, dv.railTripsPerDay / boardings) : 1;
  const baseOf = baseTrips(w, dv, m.fares, byId);
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== "running") continue;
    const linePer = PERIODS.map(() => 0), lineRides = PERIODS.map(() => 0);
    const dm = dv?.lines.get(l.id);
    const n = l.stops.length;
    // T-2793: figures solved for other stops (a stop edit, until its riders land) are carried onto the stops as they
    // are now, so the line earns as it did; without carried figures (a cached result) it keeps its last income
    const loads = dv ? lineLoadsFor(dv, l.id, "all", l.stops) : undefined;
    const old = dm && !loads?.every((p) => p.length === 2 * (n - 1)) ? prev?.lines.get(l.id) : undefined;
    if (old) {
      lines.set(l.id, old);
      old.perHour.forEach((v, p) => { perPeriod[p] += v * PERIOD_HOURS[p] * 1e6; });
      old.rides.forEach((v, p) => { ridesPeriod[p] += v * PERIOD_HOURS[p]; });
      continue;
    }
    if (dm && st.stopS[0]?.length === n && st.stopS[1]?.length === n) {
      const g = lineGeo(w, l, byId), km = g.km;
      // T-1277: a stretch pays its starting stop's fare scale (the base its boarding stop's, in `baseOf`)
      // T-2266: its riders pay the fares they were solved with
      const lf = lineFares(dm, m.fares);
      let fares = curves.get(lf);
      if (!fares) curves.set(lf, (fares = faresOf(lf, lf)));
      const { from } = scalesOf(g, l, byId, lf);
      const pkm: number[] = [], pkmPaid: number[] = [];
      for (const seg of loads ?? dm.loads) {
        let a = 0, b = 0;
        if (seg.length === km.length) for (let k = 0; k < seg.length; k++) { a += seg[k] * km[k]; b += seg[k] * km[k] * from[k]; }
        pkm.push(a);
        pkmPaid.push(b);
      }
      let total = 0;
      for (const v of pkm) total += v;
      const rides = dm.ridersPerDay * tripsPerBoarding;
      const based = baseOf.get(l.id) ?? 0;
      // T-1626, T-2195: a high-speed or ferry line's rides pay its multiplier on the rate; demand's extra has it already
      const mul = lineFareMul({ fast: st.fast, ferry: !!st.bus?.ferry }, lf);
      pkm.forEach((pk, p) => {
        const share = total > 0 ? pk / total : 0;
        const r = (faresFor(fares, based * share, pkmPaid[p], mul) + (dm.fareX?.[p] ?? 0)) * x;
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

/** A line's boardings' mean base fare multiplier, by the stop each boards at (T-1277). */
export function lineBaseScale(boardings: readonly number[], sc: readonly number[]): number {
  let boarded = 0, scaled = 0;
  for (let k = 0; k < boardings.length; k++) { boarded += boardings[k]; scaled += boardings[k] * (sc[k] ?? 1); }
  return boarded > 0 ? scaled / boarded : (sc[0] ?? 1);
}

/** Half of what the network's base fares are shared out by boardings, half by passenger-km. */
const BASE_BY_BOARDINGS = 0.5;

/** T-2104: each running line's part of the network's base fares a day, in base fares (each times
 * its boarding stop's fare scale, T-1277). The pool is what the base comes to over every line
 * (each boarding times rail trips over boardings), so the network's fares stay the same; it is
 * shared `byBoardings` by boardings and the rest by passenger-km, and no line takes more than one
 * base fare a boarding (a trip pays one base whatever it rides), the excess going to the others by
 * the same weights. Pool and weights are network-wide, since a trip's own legs are not known on
 * the main thread: a short leg (a feeder bus) gets less than an even share and a long one more.
 * `byBoardings` 1 is the even share per boarding. */
export function baseTrips(w: WorldView, dv: DemandView | null, fares: MoneyTerms["fares"], byId: Map<string, StationInput>, byBoardings = BASE_BY_BOARDINGS): Map<string, number> {
  const out = new Map<string, number>();
  if (!dv) return out;
  const boardingsAll = w.save.lines.reduce((a, l) => a + (dv.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const perBoarding = boardingsAll > 0 ? Math.min(1, dv.railTripsPerDay / boardingsAll) : 1;
  const ids: string[] = [], cap: number[] = [], byB: number[] = [], byKm: number[] = [];
  let sumB = 0, sumKm = 0, pool = 0;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], dm = dv.lines.get(l.id), n = l.stops.length;
    if (!st || st.status !== "running" || !dm || st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
    const g = lineGeo(w, l, byId), km = g.km;
    const sc = lineBaseScale(dm.boardings, scalesOf(g, l, byId, lineFares(dm, fares as Fares)).sc);
    let pk = 0;
    for (const seg of lineLoadsFor(dv, l.id, "all", l.stops) ?? dm.loads) if (seg.length === km.length) for (let k = 0; k < seg.length; k++) pk += seg[k] * km[k];
    // a line with no passenger-km takes no fare in `incomeFor`, so none of the pool either
    if (!(pk > 0)) continue;
    const b = dm.ridersPerDay * sc;
    ids.push(l.id); cap.push(b); byB.push(b); byKm.push(pk * sc);
    sumB += b; sumKm += pk * sc; pool += b * perBoarding;
  }
  const kmPart = sumKm > 0 ? 1 - byBoardings : 0;
  const wt = ids.map((_, i) => (sumB > 0 ? (1 - kmPart) * byB[i] / sumB : 0) + (kmPart > 0 ? kmPart * byKm[i] / sumKm : 0));
  // fill to the caps: lines whose share would pass one base a boarding are held there, the rest re-shared
  const got = ids.map(() => 0), held = ids.map(() => false);
  let left = pool;
  for (let round = 0; round < ids.length && left > 0; round++) {
    let free = 0;
    for (let i = 0; i < ids.length; i++) if (!held[i]) free += wt[i];
    if (!(free > 0)) break;
    const c = left / free;
    let more = false;
    for (let i = 0; i < ids.length; i++) if (!held[i] && c * wt[i] > cap[i]) { held[i] = true; got[i] = cap[i]; left -= cap[i]; more = true; }
    if (!more) { for (let i = 0; i < ids.length; i++) if (!held[i]) got[i] = c * wt[i]; break; }
  }
  ids.forEach((id, i) => out.set(id, got[i]));
  return out;
}

/** What the figures read of a running line with both runs' stop positions: its stretches' km, run
 * 0's then run 1's in its own order (as `LineDemand.loads`), signed and not, and its stops' zones
 * ("" outside every zone); built once per network and zone boundaries (T-1209). */
export interface LineGeo {
  km: number[];
  kmAbs: number[];
  zones: string[];
  scales?: { of: unknown; sc: number[]; from: number[] };
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
/** A line's stops' base fare multipliers (`sc`) and each stretch's rate multiplier, its starting
 * stop's (`from`), on the curve's first stretch (T-1277; T-1662: a country's rounded default
 * fares are not the curve times its scale). `fares` is the cache key: the same object, the same multipliers. */
export function scalesOf(g: LineGeo, l: LineInput, byId: Map<string, StationInput>, fares: Pick<Fares, "base" | "perKm" | "nodes" | "scales">) {
  if (g.scales?.of !== fares || !g.scales) {
    const n = l.stops.length, sc: number[] = [], rate: number[] = [];
    for (const id of l.stops) {
      const s = byId.get(id), c = (s && stationCountry(s)) ?? undefined;
      const scale = fareScale(c, fares.scales), r = roundedDefault(fares, c);
      sc.push(r && fares.base > 0 ? r.base / fares.base : scale);
      rate.push(r && fares.perKm > 0 ? r.perKm / fares.perKm : scale);
    }
    g.scales = { of: fares, sc, from: [...rate.slice(0, n - 1), ...[...rate].reverse().slice(0, n - 1)] };
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
