// T-531: money by region. A region is a zone (a city pack or a non-city zone), the place the
// Lines and Stations tabs list (`stationZone`, T-332). The clock worker keeps a ledger per region
// beside the network's; this module works out how each line's money and each construction divides
// between regions, and reads the region ledgers back. "" holds whatever falls outside every zone,
// so the regions always add up to the network.
//
// Splits (docs/track-and-service.md 6.4): a line's fares by where its riders board (the base fare)
// and ride (the per-km part, each stretch between two stops half to each stop's region); its running
// cost by its route km the same way; construction by the cost of the track newly built in each
// region (its stations go with it), new trains by the line the edit started.

import { computed } from "@preact/signals";
import { demandByDay, type DemandView } from "./demand";
import { stationZone } from "./listScope";
import { lineGeo, liveMoney, moneyState, moneyTerms, moneyWorld, scalesOf, type MoneyTerms } from "./money";
import { placeAt, zonesLoaded } from "./zones";
import { groundDist } from "./coords";
import { stopsById, world } from "./world";
import type { WorldView } from "./types";
import type { MoneyState } from "../workers/protocol";
import type { DayLedger, HourLedger } from "../workers/saveFile";

/** [region, share] pairs adding up to 1 ("" outside every zone). */
export type Shares = [string, number][];

function norm(m: Map<string, number>): Shares {
  let t = 0;
  for (const v of m.values()) t += v;
  return t > 0 ? [...m].filter(([, v]) => v > 0).map(([k, v]) => [k, v / t]) : [];
}
const add = (m: Map<string, number>, k: string, v: number) => m.set(k, (m.get(k) ?? 0) + v);

type LineShares = Map<number, { fare: Shares; ride: Shares; run: Shares }>;
let sharesMemo: { w: WorldView; m: MoneyTerms; zl: number; dv: DemandView | null; out: LineShares } | null = null;
/** Each running line's split of its fares, its train rides (by where they board, T-565) and its
 * running cost between regions, by line number; redone only when the weekday's view, the network,
 * the fares or the zone boundaries change (T-1209). */
export const lineShares = computed(() => {
  const w = moneyWorld.value, m = moneyTerms.value;
  const out: LineShares = new Map();
  if (!w || !m) return out;
  const zl = zonesLoaded.value; // boundaries arriving re-place the stations
  const dv = demandByDay.value[0]?.view ?? null;
  const was = sharesMemo;
  if (was && was.w === w && was.m === m && was.zl === zl && was.dv === dv) return was.out;
  sharesMemo = { w, m, zl, dv, out };
  const byId = stopsById(w);
  const boardingsAll = w.save.lines.reduce((a, l) => a + (dv?.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const tripsPerBoarding = dv && boardingsAll > 0 ? Math.min(1, dv.railTripsPerDay / boardingsAll) : 1;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], n = l.stops.length;
    if (!st || st.status !== "running" || n < 2) continue;
    const g = lineGeo(w, l, byId), z = g.zones;
    const s0 = st.stopS[0] ?? [];
    const km = new Map<string, number>();
    for (let k = 0; k + 1 < n; k++) {
      const d = Math.max(0, ((s0[k + 1] ?? 0) - (s0[k] ?? 0)) / 1000);
      add(km, z[k], d / 2);
      add(km, z[k + 1], d / 2);
    }
    const run = norm(km);
    // run 0's segments run stop k to k + 1, run 1's the stops in reverse (LineDemand.loads)
    const dm = dv?.lines.get(l.id);
    let fare = run, ride = run;
    if (dm && st.stopS[1]?.length === n) {
      const ends: [string, string, number][] = [];
      for (let r = 0; r < 2; r++) {
        const s = st.stopS[r], zz = r === 0 ? z : [...z].reverse();
        for (let k = 0; k + 1 < n; k++) ends.push([zz[k], zz[k + 1], Math.abs(s[k + 1] - s[k]) / 1000]);
      }
      // T-1277: a stretch pays its starting stop's fare scale, the base its boarding stop's
      const { sc, from } = scalesOf(g, l, byId, m.fares.scales);
      const pkm = new Map<string, number>(), board = new Map<string, number>(), paid = new Map<string, number>();
      let pkmAll = 0;
      for (const per of dm.loads) {
        if (per.length !== ends.length) continue;
        ends.forEach(([a, b, d], k) => { const v = per[k] * d * from[k]; add(pkm, a, v / 2); add(pkm, b, v / 2); pkmAll += v; });
      }
      dm.boardings.forEach((v, k) => { add(board, z[k] ?? "", v); add(paid, z[k] ?? "", v * (sc[k] ?? 1)); });
      const byBoarding = norm(board), byPaid = norm(paid);
      if (byBoarding.length) ride = byBoarding;
      // a bent curve's extra goes with the per-km part, as it is counted on the km ridden (T-1516)
      const fx = dm.fareX?.reduce((a, b) => a + b, 0) ?? 0;
      const boarded = dm.boardings.reduce((a, b) => a + b, 0);
      const baseScale = boarded > 0 ? dm.boardings.reduce((a, b, k) => a + b * (sc[k] ?? 1), 0) / boarded : 1;
      const base = m.fares.base * dm.ridersPerDay * tripsPerBoarding * baseScale, perKm = Math.max(0, m.fares.perKm * pkmAll + fx);
      if (base + perKm > 0) {
        const mix = new Map<string, number>();
        for (const [k, v] of byPaid) add(mix, k, v * base);
        for (const [k, v] of norm(pkm)) add(mix, k, v * perKm);
        fare = norm(mix);
      }
    }
    out.set(l.num, { fare, ride, run });
  }
  return out;
});

/** T-867: a constructed edge's upkeep a game day, US$ (0 for blueprint and crossovers). */
export function edgeUpkeep(e: { built: boolean; crossover?: boolean; tracks: number; hsr?: boolean; lengthM: number }, m: Pick<MoneyState, "upkeepTrackKm" | "upkeepSingle" | "upkeepHsr" | "economy">): number {
  if (!e.built || e.crossover) return 0;
  return (e.lengthM / 1000) * m.upkeepTrackKm * (e.tracks === 1 ? m.upkeepSingle : e.hsr ? m.upkeepHsr : 1) * m.economy;
}
/** T-1464: a bus route's road km by region, each stretch between two stops half to each stop's
 * region (the drawn stops' ground distances, scaled to the road length). */
function busKmByRegion(w: WorldView, line: string): Map<string, number> {
  const bus = w.lineStats[line]?.bus, out = new Map<string, number>();
  if (!bus) return out;
  const byId = stopsById(w), p = bus.stops;
  const zone = (id: number) => { const s = byId.get(String(id)); return (s && stationZone(s)) ?? ""; };
  const d: number[] = [];
  let ground = 0;
  for (let k = 0; k + 1 < p.length; k++) ground += d[k] = groundDist(p[k].x, p[k].y, p[k + 1].x, p[k + 1].y);
  const scale = ground > 0 ? bus.roadM / 1000 / ground : 0;
  for (let k = 0; k + 1 < p.length; k++) {
    add(out, zone(p[k].id), d[k] * scale / 2);
    add(out, zone(p[k + 1].id), d[k] * scale / 2);
  }
  return out;
}
/** T-867: a constructed station's upkeep a game day, US$. */
export const stationUpkeep = (m: Pick<MoneyState, "upkeepStation" | "economy">) => m.upkeepStation * m.economy;

/** Each edge midpoint's zone, kept while the edge's ends and the zone boundaries stay (T-1209). */
const midZones = new Map<number, { lng: number; lat: number; zone: string }>();
let midZonesAt = -1;
function midZone(id: number, lng: number, lat: number, zl: number): string {
  if (midZonesAt !== zl) { midZones.clear(); midZonesAt = zl; }
  const c = midZones.get(id);
  if (c && c.lng === lng && c.lat === lat) return c.zone;
  const zone = placeAt(lng, lat) ?? "";
  midZones.set(id, { lng, lat, zone });
  return zone;
}
const sameShares = (a: Shares, b: Shares) => a.length === b.length && a.every(([k, v], i) => k === b[i][0] && v === b[i][1]);
let lastUpkeep: Shares = [];

/** T-867: how the network's upkeep divides between regions: each constructed edge by its midpoint,
 * each constructed station by its own region. The last shares while they come out the same. */
export const upkeepShares = computed<Shares>(() => {
  const w = world.value, m = moneyTerms.value;
  if (!w || !m || m.upkeepTrackKm === undefined) return (lastUpkeep = []);
  const zl = zonesLoaded.value;
  const out = new Map<string, number>();
  for (const e of w.edges) {
    const u = edgeUpkeep(e, m);
    const a = u > 0 && w.nodes.get(e.a), b = u > 0 && w.nodes.get(e.b);
    if (a && b) add(out, midZone(e.id, (a.lng + b.lng) / 2, (a.lat + b.lat) / 2, zl), u);
  }
  for (const s of w.save.stations) if (s.built) add(out, stationZone(s) ?? "", stationUpkeep(m));
  for (const l of w.save.lines) {
    if (!w.lineStats[l.id]?.bus?.built) continue;
    for (const [z, km] of busKmByRegion(w, l.id)) add(out, z, km * (m.upkeepBusKm ?? 0) * m.economy);
  }
  const next = norm(out);
  if (!sameShares(next, lastUpkeep)) lastUpkeep = next;
  return lastUpkeep;
});

/** How the last paid edit's charge divides: construction by the cost of the track it built in each
 * region (a station alone by its own region), new trains by `line`'s route km, else as construction. */
export function chargeShares(before: WorldView | null, after: WorldView, line?: number): { build: Shares; trains: Shares } {
  const was = new Map(before?.edges.map((e) => [e.id, e.built]) ?? []);
  const cost = new Map<string, number>();
  for (const e of after.edges) {
    if (!e.built || was.get(e.id)) continue;
    const a = after.nodes.get(e.a), b = after.nodes.get(e.b);
    if (!a || !b) continue;
    add(cost, placeAt((a.lng + b.lng) / 2, (a.lat + b.lat) / 2) ?? "", e.cost);
  }
  const perKm = (moneyState.peek()?.busRouteKm ?? 0) * 1e6;
  for (const l of after.save.lines) {
    if (!after.lineStats[l.id]?.bus?.built || before?.lineStats[l.id]?.bus?.built) continue;
    for (const [z, km] of busKmByRegion(after, l.id)) add(cost, z, km * perKm);
  }
  if (!cost.size) {
    const built = new Set(before?.save.stations.filter((s) => s.built).map((s) => s.id) ?? []);
    for (const s of after.save.stations) if (s.built && !built.has(s.id)) add(cost, stationZone(s) ?? "", 1);
  }
  const build = norm(cost);
  const runShares = line !== undefined ? lineShares.peek().get(line)?.run : undefined;
  return { build, trains: runShares?.length ? runShares : build };
}

/** A region's ledger as the Money tab reads it, with today and this hour as they move; null for the
 * network's. A region with nothing in it has empty lists. */
export function regionLedger(zone: string | null): { ledger: DayLedger[]; hours: HourLedger[]; today: DayLedger | null; hour: HourLedger | null } {
  const m = moneyState.value, live = liveMoney.value;
  if (zone === null || !m) return { ledger: m?.ledger ?? [], hours: m?.hours ?? [], today: live.today, hour: live.hour };
  const r = m.regions?.[zone], l = live.regions?.[zone];
  const day = live.today?.day ?? m.ledger.at(-1)?.day, hour = live.hour?.hour ?? m.hours.at(-1)?.hour;
  const lastDay = r?.ledger.at(-1), lastHour = r?.hours.at(-1);
  const zero = { fares: 0, running: 0, trains: 0, build: 0, rides: 0 };
  return {
    ledger: r?.ledger ?? [], hours: r?.hours ?? [],
    today: l?.today ?? (lastDay?.day === day ? lastDay! : day !== undefined ? { day, ...zero } : null),
    hour: l?.hour ?? (lastHour?.hour === hour ? lastHour! : hour !== undefined ? { hour, ...zero } : null),
  };
}
