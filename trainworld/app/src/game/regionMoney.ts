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
import { demandByDay } from "./demand";
import { stationZone } from "./listScope";
import { liveMoney, moneyState } from "./money";
import { placeAt, zonesLoaded } from "./zones";
import { stationsById, world } from "./world";
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

/** Each running line's split of its fares, its train rides (by where they board, T-565) and its
 * running cost between regions, by line number. */
export const lineShares = computed(() => {
  const w = world.value, m = moneyState.value;
  const out = new Map<number, { fare: Shares; ride: Shares; run: Shares }>();
  if (!w || !m) return out;
  void zonesLoaded.value; // boundaries arriving re-place the stations
  const dv = demandByDay.value[0]?.view ?? null;
  const byId = stationsById(w);
  const zone = (id: string) => { const s = byId.get(id); return (s && stationZone(s)) ?? ""; };
  const boardingsAll = w.save.lines.reduce((a, l) => a + (dv?.lines.get(l.id)?.ridersPerDay ?? 0), 0);
  const tripsPerBoarding = dv && boardingsAll > 0 ? Math.min(1, dv.railTripsPerDay / boardingsAll) : 1;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], n = l.stops.length;
    if (!st || st.status !== "running" || n < 2) continue;
    const z = l.stops.map(zone);
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
      const pkm = new Map<string, number>(), board = new Map<string, number>();
      let pkmAll = 0;
      for (const per of dm.loads) {
        if (per.length !== ends.length) continue;
        ends.forEach(([a, b, d], k) => { const v = per[k] * d; add(pkm, a, v / 2); add(pkm, b, v / 2); pkmAll += v; });
      }
      dm.boardings.forEach((v, k) => add(board, z[k] ?? "", v));
      const byBoarding = norm(board);
      if (byBoarding.length) ride = byBoarding;
      // a bent curve's extra goes with the base, by where its trips board (T-1143)
      const fx = dm.fareX?.reduce((a, b) => a + b, 0) ?? 0;
      const base = Math.max(0, m.fares.base * dm.ridersPerDay * tripsPerBoarding + fx), perKm = m.fares.perKm * pkmAll;
      if (base + perKm > 0) {
        const mix = new Map<string, number>();
        for (const [k, v] of byBoarding) add(mix, k, v * base);
        for (const [k, v] of norm(pkm)) add(mix, k, v * perKm);
        fare = norm(mix);
      }
    }
    out.set(l.num, { fare, ride, run });
  }
  return out;
});

/** T-867: a constructed edge's upkeep a game day, US$ (0 for blueprint and crossovers). */
export function edgeUpkeep(e: { built: boolean; crossover?: boolean; tracks: number; hsr?: boolean; lengthM: number }, m: MoneyState): number {
  if (!e.built || e.crossover) return 0;
  return (e.lengthM / 1000) * m.upkeepTrackKm * (e.tracks === 1 ? m.upkeepSingle : e.hsr ? m.upkeepHsr : 1) * m.economy;
}
/** T-867: a constructed station's upkeep a game day, US$. */
export const stationUpkeep = (m: MoneyState) => m.upkeepStation * m.economy;

/** T-867: how the network's upkeep divides between regions: each constructed edge by its midpoint,
 * each constructed station by its own region. */
export const upkeepShares = computed<Shares>(() => {
  const w = world.value, m = moneyState.value;
  if (!w || !m || m.upkeepTrackKm === undefined) return [];
  void zonesLoaded.value;
  const out = new Map<string, number>();
  for (const e of w.edges) {
    const u = edgeUpkeep(e, m);
    const a = u > 0 && w.nodes.get(e.a), b = u > 0 && w.nodes.get(e.b);
    if (a && b) add(out, placeAt((a.lng + b.lng) / 2, (a.lat + b.lat) / 2) ?? "", u);
  }
  for (const s of w.save.stations) if (s.built) add(out, stationZone(s) ?? "", stationUpkeep(m));
  return norm(out);
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
