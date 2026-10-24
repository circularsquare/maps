// Points (T-684; SPEC 6): a measure per region, what it earned in the past 24 game hours at 20 a
// rail trip and 1 a passenger-km (T-694). A trip is a rider's whole rail journey, however many
// lines it uses, and counts in the region it starts in.
//
// Trips come from the solve, not from boardings: each region's rail trips (its own demand's, plus
// half of each regional corridor's at each end) are spread over the boardings at that region's
// stations, so a region's trips a day are exact while the hour they land follows its boardings.
// Passenger-km are each line's loads times its stretches' lengths, period by period, carried by
// that period's boardings onto the line (each boarding the line's average km), so the km count in
// the region where riders get on. The clock worker settles both as fares land (clock.worker.ts).

import { computed } from "@preact/signals";
import { CITIES, type CityId } from "./cities";
import { AVERAGE_DAY, DAY_KINDS, DEMAND_DAYS } from "./days";
import { demandByDay, type DayDemand } from "./demand";
import { sig3 } from "./format";
import { stationZone } from "./listScope";
import { moneyState } from "./money";
import { PERIODS } from "./periods";
import { stationsById, world } from "./world";
import { zonesLoaded } from "./zones";
import type { WorldView } from "./types";

export const POINTS_PER_TRIP = 20;
export const POINTS_PER_KM = 1;
/** T-685: the rail share (%) a region's star is earned at */
export const STAR_SHARE = 10;

const PERIOD_HOURS = PERIODS.map((p) => p.to - p.from);

/** Points short, three significant digits (T-831): 980, 12.3k, 512k, 4.56M, 103M, 1.25B. */
export function pointsShort(n: number): string {
  // a figure that rounds up to the next unit (999.6k) is written in that unit
  if (n >= 999.5e6) return sig3(n / 1e9) + "B";
  if (n >= 999.5e3) return sig3(n / 1e6) + "M";
  if (n >= 1e3) return sig3(n / 1e3) + "k";
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
    const rail = c.longerRail[k] + c.commuteRail[k], j = c.railAt[k], from = c.table.from[j], to = c.table.to[j];
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
  const byId = stationsById(w);
  const zone = (id: string) => { const s = byId.get(id); return (s && stationZone(s)) ?? ""; };
  const running = w.save.lines.filter((l) => w.lineStats[l.id]?.status === "running" && l.stops.length >= 2 && dv.lines.has(l.id));
  // boardings a day at each region's stations, to spread its trips over
  const board = new Map<string, number>();
  for (const l of running) dv.lines.get(l.id)!.boardings.forEach((v, k) => add(board, zone(l.stops[k]), v));
  const trips = railTripsByRegion(d);
  const perBoarding = new Map<string, number>();
  for (const [z, b] of board) if (b > 0) perBoarding.set(z, (trips.get(z) ?? 0) / b);
  for (const l of running) {
    const st = w.lineStats[l.id], dm = dv.lines.get(l.id)!, n = l.stops.length;
    if (st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
    const z = l.stops.map(zone);
    // run 0's stretches, then run 1's in its own order, as `loads`
    const km: number[] = [];
    for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push(Math.abs(s[k + 1] - s[k]) / 1000);
    const pkm = dm.loads.map((seg) => (seg.length === km.length ? seg.reduce((a, v, k) => a + v * km[k], 0) : 0));
    const pkmAll = pkm.reduce((a, b) => a + b, 0);
    const regs = new Map<string, number[]>();
    PERIODS.forEach((_, p) => {
      // boardings at each stop in the period: both runs' (stop k is node k on run 0, 2n - 1 - k on
      // run 1), or the day's spread by the period's passenger-km when the solve kept no periods
      const bp = dm.board?.[p];
      const at = l.stops.map((_, k) => (bp && bp.length === 2 * n ? bp[k] + bp[2 * n - 1 - k] : pkmAll > 0 ? dm.boardings[k] * pkm[p] / pkmAll : 0));
      const all = at.reduce((a, b) => a + b, 0);
      if (!(all > 0)) return;
      const kmEach = (pkm[p] ?? 0) / all;
      at.forEach((b, k) => {
        if (!(b > 0)) return;
        let r = regs.get(z[k]);
        if (!r) regs.set(z[k], (r = PERIODS.map(() => 0)));
        r[p] += b * (POINTS_PER_TRIP * (perBoarding.get(z[k]) ?? 0) + POINTS_PER_KM * kmEach) / PERIOD_HOURS[p];
      });
    });
    if (regs.size) out.set(l.num, regs);
  }
  return out;
}

/** Every running line's points per game hour, `[region, perHour[dayKind][period]]` by line number,
 * as the clock worker settles them. A kind of day not solved yet takes the weekday's. */
export const linePoints = computed<{ line: number; regions: [string, number[][]][] }[]>(() => {
  const w = world.value;
  if (!w) return [];
  void zonesLoaded.value; // boundaries arriving re-place the stations
  const days = DAY_KINDS.map((k) => demandByDay.value[k]);
  const per = days.map((d) => dayPoints(w, d));
  const base = days.findIndex((d) => d.view);
  if (base < 0) return [];
  const lines = new Set(per.flatMap((m) => [...m.keys()]));
  return [...lines].map((line) => {
    const zones = new Set(per.flatMap((m) => [...(m.get(line)?.keys() ?? [])]));
    const zero = PERIODS.map(() => 0);
    return {
      line,
      regions: [...zones].map((z): [string, number[][]] => [z, DAY_KINDS.map((k) => (days[k].view ? per[k] : per[base]).get(line)?.get(z) ?? zero)]),
    };
  });
});

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

/** Each region's points a day at the riders now, per demand day (weekday, weekend day, average). */
export const pointsPerDay = computed<Map<string, number[]>>(() => {
  const out = new Map<string, number[]>();
  for (const { regions } of linePoints.value) {
    for (const [z, per] of regions) {
      let r = out.get(z);
      if (!r) out.set(z, (r = DEMAND_DAYS.map(() => 0)));
      for (const k of DAY_KINDS) r[k] += per[k].reduce((a, v, p) => a + v * PERIOD_HOURS[p], 0);
    }
  }
  for (const r of out.values()) r[AVERAGE_DAY] = (5 * r[0] + 2 * r[1]) / 7;
  return out;
});
