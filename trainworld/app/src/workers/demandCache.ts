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
// pack the solve covered and the market's, the demand code's version (DEMAND_KEY, T-3926), the worker
// count (sums are added in worker order) and the perfOff flags. Any other difference means no cache, never a
// wrong one. T-4202: a result whose key differs only in the demand code's version or the worker count (an
// update, the same network, zones and packs) stands in while this build solves: shown as the cache is, its
// figures replaced by the fresh final (`standsIn`).

import type { CityId } from "../game/cities";
import type { CorridorRows, DayDemand, DemandView, IntercityDemand, LineDemand, PurposeDemand, StationDemand } from "../game/demand";
import type { DemandDay } from "../game/days";
import type { PurposeKey } from "../game/purposes";
import type { ModeSplit } from "../game/types";
import type { NetworkSnapshot } from "./protocol";

declare const __TW_BUILD__: string | undefined;
/** The app build: set by vite.config.ts at each build and each dev-server start. */
export const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "unknown";
declare const __TW_DEMAND__: string | undefined;
/** T-3926: the demand code's version, set by vite.config.ts: in a build a hash of every source and data file the
 * demand side runs (the same in every build of the same demand code), on the dev server new at each start. */
export const DEMAND_KEY = typeof __TW_DEMAND__ === "string" ? __TW_DEMAND__ : "unknown";
export const CACHE_FORMAT = 13; // raise when the cached result's layout changes (T-4202: an older demand code's result is read as a stand-in)
/** The view version a cached result is shown under: no solve has it, so the workers answer no
 * question about it (a click waits for the fresh solve). */
export const CACHE_VERSION = 0;

/** What the cache key is made of, besides the network's hash (its storage key). */
export interface CacheKey {
  format: number;
  /** DEMAND_KEY */
  demand: string;
  workers: number;
  flags: string;
  /** zone id (and "market") -> SHA-256 of the pack's bytes; the zones are exactly the solve's */
  packs: Record<string, string>;
}

type Stations = ([number, number, number] | [number, number, number, number])[]; // place, boardings, alightings, transfers if known (T-1530)
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
/** `modes`: the viewed zone's bubbles on the kind of day shown (`day`, a weekday when absent), all
 * purposes (`CellModes`' eight arrays one after another, pack cell order), when they were on show as
 * the result landed */
export interface StoredDemand extends CacheKey { at: number; round: number; days: StoredDay[]; modes?: { city: CityId; data: Float32Array; day?: number }; ahead?: CacheAhead;
  /** T-2136: the corridor rows the page held of the result's market, one place each (the zone in view's
   * and the built zones'), so a cached result shows them before any worker holds the market */
  rows?: { place: number; rows: CorridorRows }[];
  /** T-2188: the figures are a city edit's local results, not a whole solve's: a load schedules the quiet refresh */
  local?: true }
/** T-1539: the network a load of a save works out, when this result is that network's: `save` the
 * clock worker's key of the save's network inputs (`saveNetKey`), `snapshot` the network it sent */
export interface CacheAhead { save: string; snapshot: NetworkSnapshot }

/** T-1539: a key of what a save's network snapshot is made from: its track bytes (ids dense), the
 * header fields the snapshot reads, and the app build (T-3926: the clock worker that works the snapshot
 * out is not in DEMAND_KEY, so another build's stored snapshot may not be this build's). Equal keys give
 * the same snapshot. */
export async function saveNetKey(track: Uint8Array, h: { city?: string; fares?: unknown; fleet?: number; fleetFast?: number; fleetBus?: number; mode?: string }): Promise<string> {
  // T-2679: the reference country changes no fare, so no key
  const { ref: _, ...fares } = (h.fares ?? {}) as { ref?: unknown };
  const head = new TextEncoder().encode(JSON.stringify([h.city ?? "", h.fares ? fares : null, h.fleet ?? 0, h.fleetFast ?? 0, h.fleetBus ?? 0, h.mode ?? "normal", BUILD]));
  const net = withoutEras(track);
  const all = new Uint8Array(net.length + head.length);
  all.set(net);
  all.set(head, net.length);
  return sha256(all);
}

/** T-2588: a save's network bytes without the eras tail (sim `split_eras`): the trains changing over
 * after an edit (T-1071), which only the drawing reads and which every autosave prunes as the clock
 * runs, so a save that has any would key differently on each load. */
function withoutEras(track: Uint8Array): Uint8Array {
  const n = track.length;
  if (n < 8) return track;
  const tag = String.fromCharCode(...track.subarray(n - 4));
  if (tag !== "ERA2" && tag !== "ERA1" && tag !== "ERA3" && tag !== "ERA4") return track;
  const len = new DataView(track.buffer, track.byteOffset, n).getUint32(n - 8, true);
  return len <= n - 8 ? track.subarray(0, n - 8 - len) : track;
}

/** The service demand reads, as demandClient's `Service`. */
interface Svc {
  frame: unknown;
  fare: unknown;
  stations: { id: string; x: number; y: number; level?: number; complex?: number; upgrades?: number; fare?: number; street?: boolean }[];
  lines: { id: string; stops: string[]; back?: string[]; tph: unknown; hopS: unknown; hopKm?: unknown; fast?: boolean; ferry?: boolean; boat?: number; brt?: boolean; catches?: unknown; tram?: boolean; fareDist?: boolean; passFares?: unknown; dwellS: unknown; wrapS?: number; cars: number; span?: unknown; gaps?: unknown }[];
  passages?: [string, string][];
  apart?: [string, string][];
}

/** T-3542: five periods' values (morning, midday, afternoon, evening, night) as the three rows they
 * were keyed by before periods, where the day's three agree, so a game of three rows keeps its keys. */
function asRows(v: unknown): unknown {
  if (!Array.isArray(v) || v.length !== 5) return v;
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  return same(v[0], v[1]) && same(v[1], v[2]) ? [v[0], v[3], v[4]] : v;
}
/** `asRows` of each run's periods. */
function runsAsRows(v: unknown): unknown {
  return Array.isArray(v) ? v.map(asRows) : v;
}

/** The network as demand reads it with ids replaced by places (stations, lines) and complexes
 * numbered in order of first appearance; and the ids of each place now. */
export function canonical(svc: Svc): { text: string; stationIds: string[]; lineIds: string[] } {
  const place = new Map(svc.stations.map((s, i) => [s.id, i]));
  const cx = new Map<number, number>();
  const stations = svc.stations.map((s) => {
    let c: number | null = null;
    if (s.complex !== undefined) { if (!cx.has(s.complex)) cx.set(s.complex, cx.size); c = cx.get(s.complex)!; }
    // T-3740: a fare zone only where there is one, so other networks keep their keys; T-3888: a level (the stairs) alike
    if (s.level) return [s.x, s.y, c, s.upgrades ?? 0, s.fare ?? 0, s.level];
    return s.fare ?[s.x, s.y, c, s.upgrades ?? 0, s.fare] : s.upgrades ? [s.x, s.y, c, s.upgrades] : [s.x, s.y, c];
  });
  // a circle's stand (T-798), a service span (T-842) and shared stretches' gaps (T-4514) only where there is one, so other
  // networks keep their keys; T-3926: the way back's stations, the boat size, a busway and a bus's
  // catchment alike (the solve reads them, and the key no longer carries the clock worker's code)
  const lines = svc.lines.map((l) => ({ stops: l.stops.map((s) => place.get(s) ?? -1), ...(l.back ? { back: l.back.map((s) => place.get(s) ?? -1) } : {}), tph: asRows(l.tph), hopS: runsAsRows(l.hopS), hopKm: l.hopKm ?? null, ...(l.fast ? { fast: true } : {}), ...(l.ferry ? { ferry: true } : {}), ...(l.boat !== undefined ? { boat: l.boat } : {}), ...(l.brt ? { brt: true } : {}), ...(l.catches ? { catches: l.catches } : {}), ...(l.tram ? { tram: true } : {}), ...(l.fareDist ? { fareDist: true } : {}), ...(l.passFares ? { passFares: l.passFares } : {}), dwellS: runsAsRows(l.dwellS), ...(l.wrapS ? { wrapS: l.wrapS } : {}), cars: l.cars, ...(l.span ? { span: l.span } : {}), ...(l.gaps ? { gaps: l.gaps } : {}) }));
  // passages (T-1281) only where there are any
  const passages = svc.passages?.length ? { passages: svc.passages.map(([a, b]) => [place.get(a) ?? -1, place.get(b) ?? -1]) } : {};
  // T-4048: the separated pairs alike
  const apart = svc.apart?.length ? { apart: svc.apart.map(([a, b]) => [place.get(a) ?? -1, place.get(b) ?? -1]) } : {};
  return { text: JSON.stringify({ frame: svc.frame, fare: svc.fare, stations, lines, ...passages, ...apart }), stationIds: svc.stations.map((s) => s.id), lineIds: svc.lines.map((l) => l.id) };
}

export async function sha256(data: BufferSource | string): Promise<string> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)); // i18n-ignore
  let s = "";
  for (const b of h) s += b.toString(16).padStart(2, "0");
  return s;
}

/** Whether the cache can work here (WebCrypto needs a secure context; localhost is one). */
export const cacheAvailable = (): boolean => typeof crypto !== "undefined" && !!crypto.subtle && typeof indexedDB !== "undefined";

export function sameKey(a: CacheKey, b: CacheKey): string | null {
  if (a.format !== b.format) return "format";
  if (a.demand !== b.demand) return "demand";
  if (a.workers !== b.workers) return "workers";
  if (a.flags !== b.flags) return "flags";
  const ka = Object.keys(a.packs).sort(), kb = Object.keys(b.packs).sort();
  if (ka.join() !== kb.join()) return "zones";
  for (const k of ka) if (a.packs[k] !== b.packs[k]) return "pack " + k;
  return null;
}

/** T-4202: a kept result whose key `a` differs from this build's `b` only in the demand code's version or the
 * worker count, so it can stand in (its layout, zones and packs the same) while this build solves the network. */
export function standsIn(a: CacheKey, b: CacheKey): boolean {
  return (a.demand !== b.demand || a.workers !== b.workers) && sameKey({ ...a, demand: b.demand, workers: b.workers }, b) === null;
}

/** A final result's days as stored: `cities` the zones the solve covered (bare zones are left out;
 * they come from the aux worker again). */
export function toStored(days: DayDemand[], cities: CityId[], ids: { stationIds: string[]; lineIds: string[] }): StoredDay[] {
  const st = new Map(ids.stationIds.map((id, i) => [id, i])), ln = new Map(ids.lineIds.map((id, i) => [id, i]));
  const stations = (m: Map<string, StationDemand>): Stations => [...m].map(([id, s]) => s.transfers === undefined ? [st.get(id)!, s.boardings, s.alightings] : [st.get(id)!, s.boardings, s.alightings, s.transfers]);
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
  const stations = (s: Stations) => new Map(s.map(([i, b, a, t]): [string, StationDemand] => [ids.stationIds[i], t === undefined ? { boardings: b, alightings: a } : { boardings: b, alightings: a, transfers: t }]));
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

/** Bytes of the typed arrays in a stored result, and the count of other numbers: its size, near enough.
 * T-2499: an array the days share (the corridor table, T-1008) counts once, as it is stored. */
export function storedSize(days: StoredDay[]): number {
  let bytes = 0;
  const seen = new Set<unknown>();
  const walk = (x: unknown) => {
    if (x && typeof x === "object") { if (seen.has(x)) return; seen.add(x); }
    if (ArrayBuffer.isView(x)) bytes += x.byteLength;
    else if (typeof x === "number") bytes += 8;
    else if (typeof x === "string") bytes += x.length;
    else if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === "object") Object.values(x).forEach(walk);
  };
  walk(days);
  return bytes;
}
