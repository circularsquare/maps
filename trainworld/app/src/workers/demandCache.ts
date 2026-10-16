// T-328: the last final demand result, kept beside the save so a reload shows it at once while the
// workers solve again (docs/architecture.md 11 "Demand cache"). workers/demandClient.ts writes it
// when a final result lands and reads it when a whole game loads; IndexedDB in workers/saveFile.ts.
//
// It holds what the page shows from a final solve (game/demand.ts `DayDemand` for each kind of
// day: the network's view with its lines, stations and purposes, each solved zone's split, the
// market's trips), not worker state. Lines and stations are stored by their place in the network
// snapshot, not their ids, since a load renumbers ids; the key is made the same way.
//
// The key: a SHA-256 of the network demand reads (the canonical form below), the SHA-256 of every
// pack the solve covered and the market's, the app build, the worker count (sums are added in
// worker order) and the perfOff flags. Any difference means no cache, never a wrong one.

import type { CityId } from "../game/cities";
import type { DayDemand, DemandView, IntercityDemand, LineDemand, PurposeDemand, StationDemand } from "../game/demand";
import type { DemandDay } from "../game/days";
import type { PurposeKey } from "../game/purposes";
import type { ModeSplit } from "../game/types";

declare const __TW_BUILD__: string | undefined;
/** The app build: set by vite.config.ts at each build and each dev-server start. */
export const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "unknown";
export const CACHE_FORMAT = 1;
/** The view version a cached result is shown under: no solve has it, so the workers answer no
 * question about it (a click waits for the fresh solve, as before T-328). */
export const CACHE_VERSION = 0;

/** What the cache key is made of, besides the network's hash (its storage key). */
export interface CacheKey {
  format: number;
  build: string;
  workers: number;
  flags: string;
  /** zone id (and "market") -> SHA-256 of the pack's bytes; the zones are exactly the solve's */
  packs: Record<string, string>;
}

type Stations = [number, number, number][]; // place, boardings, alightings
interface StoredView {
  split: ModeSplit;
  trainShareByPeriod: [string, number][];
  peakHourFactor: number[];
  tripsPerDay: number;
  railTripsPerDay: number;
  lines: [number, Omit<LineDemand, "estimate">][];
  stations: Stations;
  purposes?: [PurposeKey, { lines: [number, PurposeDemand["lines"] extends Map<string, infer L> ? L : never][]; stations: Stations }][];
}
type CityFigures = Pick<DemandView, "split" | "trainShareByPeriod" | "tripsPerDay" | "railTripsPerDay">;
interface StoredDay { view: StoredView; cities: [CityId, CityFigures][]; intercity: Omit<IntercityDemand, "version"> | null }
/** `modes`: the viewed zone's bubbles on a weekday, all purposes (`CellModes`' six arrays one after
 * another, pack cell order), when they were on show as the result landed */
export interface StoredDemand extends CacheKey { at: number; round: number; days: StoredDay[]; modes?: { city: CityId; data: Float32Array } }

/** The service demand reads, as demandClient's `Service`. */
interface Svc {
  frame: unknown;
  fare: unknown;
  stations: { id: string; x: number; y: number; complex?: number }[];
  lines: { id: string; stops: string[]; tph: unknown; hopS: unknown; hopKm?: unknown; dwellS: unknown; cars: number }[];
}

/** The network as demand reads it with ids replaced by places (stations, lines) and complexes
 * numbered in order of first appearance; and the ids of each place now. */
export function canonical(svc: Svc): { text: string; stationIds: string[]; lineIds: string[] } {
  const place = new Map(svc.stations.map((s, i) => [s.id, i]));
  const cx = new Map<number, number>();
  const stations = svc.stations.map((s) => {
    let c: number | null = null;
    if (s.complex !== undefined) { if (!cx.has(s.complex)) cx.set(s.complex, cx.size); c = cx.get(s.complex)!; }
    return [s.x, s.y, c];
  });
  const lines = svc.lines.map((l) => ({ stops: l.stops.map((s) => place.get(s) ?? -1), tph: l.tph, hopS: l.hopS, hopKm: l.hopKm ?? null, dwellS: l.dwellS, cars: l.cars }));
  return { text: JSON.stringify({ frame: svc.frame, fare: svc.fare, stations, lines }), stationIds: svc.stations.map((s) => s.id), lineIds: svc.lines.map((l) => l.id) };
}

export async function sha256(data: BufferSource | string): Promise<string> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let s = "";
  for (const b of h) s += b.toString(16).padStart(2, "0");
  return s;
}

/** Whether the cache can work here (WebCrypto needs a secure context; localhost is one). */
export const cacheAvailable = (): boolean => typeof crypto !== "undefined" && !!crypto.subtle && typeof indexedDB !== "undefined";

export function sameKey(a: CacheKey, b: CacheKey): string | null {
  if (a.format !== b.format) return "format";
  if (a.build !== b.build) return "build";
  if (a.workers !== b.workers) return "workers";
  if (a.flags !== b.flags) return "flags";
  const ka = Object.keys(a.packs).sort(), kb = Object.keys(b.packs).sort();
  if (ka.join() !== kb.join()) return "zones";
  for (const k of ka) if (a.packs[k] !== b.packs[k]) return "pack " + k;
  return null;
}

/** A final result's days as stored: `cities` the zones the solve covered (bare zones are left out;
 * they come from the aux worker again). */
export function toStored(days: DayDemand[], cities: CityId[], ids: { stationIds: string[]; lineIds: string[] }): StoredDay[] {
  const st = new Map(ids.stationIds.map((id, i) => [id, i])), ln = new Map(ids.lineIds.map((id, i) => [id, i]));
  const stations = (m: Map<string, StationDemand>): Stations => [...m].map(([id, s]) => [st.get(id)!, s.boardings, s.alightings]);
  return days.map((d) => {
    const v = d.view!;
    const view: StoredView = {
      split: v.split, trainShareByPeriod: v.trainShareByPeriod, peakHourFactor: v.peakHourFactor, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay,
      lines: [...v.lines].map(([id, l]) => { const { estimate: _e, ...rest } = l; return [ln.get(id)!, rest]; }),
      stations: stations(v.stations),
      purposes: v.purposes && [...v.purposes].map(([k, p]) => [k, { lines: [...p.lines].map(([id, l]) => [ln.get(id)!, l]), stations: stations(p.stations) }]),
    };
    const own: [CityId, CityFigures][] = [...d.cities].filter(([id]) => cities.includes(id)).map(([id, c]) => [id, { split: c.split, trainShareByPeriod: c.trainShareByPeriod, tripsPerDay: c.tripsPerDay, railTripsPerDay: c.railTripsPerDay }]);
    const intercity = d.intercity && (({ version: _v, ...rest }) => rest)(d.intercity);
    return { view, cities: own, intercity };
  });
}

/** The stored days as the page shows them, under `CACHE_VERSION`, with today's ids. */
export function fromStored(stored: StoredDemand, ids: { stationIds: string[]; lineIds: string[] }): DayDemand[] {
  const stations = (s: Stations) => new Map(s.map(([i, b, a]) => [ids.stationIds[i], { boardings: b, alightings: a }]));
  return stored.days.map((d, day) => {
    const v = d.view;
    const view: DemandView = {
      version: CACHE_VERSION, day: day as DemandDay, round: stored.round, final: true,
      split: v.split, trainShareByPeriod: v.trainShareByPeriod, peakHourFactor: v.peakHourFactor, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay,
      lines: new Map(v.lines.map(([i, l]) => [ids.lineIds[i], l])),
      stations: stations(v.stations),
      purposes: v.purposes && new Map(v.purposes.map(([k, p]) => [k, { lines: new Map(p.lines.map(([i, l]) => [ids.lineIds[i], l])), stations: stations(p.stations) }])),
    };
    const cities = new Map<CityId, DemandView>(d.cities.map(([id, c]) => [id, { ...view, ...c }]));
    return { view, cities, intercity: d.intercity && { ...d.intercity, version: CACHE_VERSION } };
  });
}

/** Bytes of the typed arrays in a stored result, and the count of other numbers: its size, near enough. */
export function storedSize(days: StoredDay[]): number {
  let bytes = 0;
  const walk = (x: unknown) => {
    if (ArrayBuffer.isView(x)) bytes += x.byteLength;
    else if (typeof x === "number") bytes += 8;
    else if (typeof x === "string") bytes += x.length;
    else if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === "object") Object.values(x).forEach(walk);
  };
  walk(days);
  return bytes;
}
