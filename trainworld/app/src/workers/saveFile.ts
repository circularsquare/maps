// The game save (SPEC 11, T-029): the player's inputs only. Built and read in the clock worker, so
// the main thread never serialises or compresses anything.
//
// File layout, gzip-compressed as a whole:
//   "TWG1"                 magic
//   u32 little-endian      length of the JSON header in bytes
//   JSON header (UTF-8)    `SaveHeader`: city, clock, money, fares, the day and hour ledgers (the money
//                          and train rides history charts), each also per region
//   the rest               the track model's network save (`TrackApi.save()`: "TWT6" Mercator world frame since
//                          T-306; older saves, "TWT5" conic since T-125 and "TWT4" legacy frame, are
//                          converted to Mercator when they load, T-308; TWT2 and TWT3 no longer load,
//                          T-1146; a line saved with no trains is given them on load)
//
// Nothing in it depends on wall time, so saving the same game twice gives the same bytes.
// Autosaves live in IndexedDB (database "anitabuilder", store "saves", one key per slot; the original game keeps key "autosave";
// beside each slot its demand cache and backups under keys the saves list does not show).

import registry from "../game/registry.json";
import successors from "../game/zone-successors.json";
import { zoneName } from "../game/zoneNames";

export const SAVE_MAGIC = "TWG1";
export const SAVE_FORMAT = 1;
/** SPEC 8: a new game starts on day 1 at 07:00 (game seconds). */
export const START_CLOCK = 86400 + 7 * 3600;
/** SPEC 6.4: US$10B to start (T-2271), in US$M; a sandbox's money since the start is the cash minus this (T-728). */
export const START_CASH = 10000;
/** T-2271: a new game's lines start at 5 cars a train (Anita: players raise it where crowded); a save
 * keeps its own `newCars`. */
export const NEW_LINE_CARS = 5;

/** One game day's money, US$M (T-028). */
export interface DayLedger {
  day: number;
  fares: number;
  running: number;
  trains: number;
  build: number;
  /** track and station upkeep (T-867); absent in days saved before it, read as 0 */
  upkeep?: number;
  /** loan payments, interest included (T-2022); absent when none was paid */
  loan?: number;
  /** loan interest in saves from before T-2022, read into `loan` */
  interest?: number;
  /** train rides taken (T-565); absent in days saved before it, read as 0 */
  rides?: number;
  /** points earned (T-684); absent before it, read as 0 */
  points?: number;
}

/** One game hour's money, US$M (T-447); `hour` counts from day 0 00:00. */
export interface HourLedger {
  hour: number;
  fares: number;
  running: number;
  trains: number;
  build: number;
  /** upkeep (T-867), as in `DayLedger` */
  upkeep?: number;
  /** loan payments (T-2022) and old interest, as in `DayLedger` */
  loan?: number;
  interest?: number;
  /** train rides taken (T-565), as in `DayLedger` */
  rides?: number;
  /** points earned (T-684), as in `DayLedger` */
  points?: number;
}

export interface SaveHeader {
  game: "anitabuilder";
  format: number;
  city: string;
  /** The shared world (registry.json `world`); absent in legacy NYC saves. */
  region?: string;
  cities?: string[];
  /** game seconds since day 0 00:00 */
  clock: number;
  /** US$M */
  cash: number;
  /** T-1661: the loan out, US$M borrowed; absent for none */
  loan?: number;
  /** T-2022: its payment a day, US$M, and the game days of payments left; absent before T-2022 */
  loanPay?: number;
  loanDays?: number;
  /** cars owned */
  fleet: number;
  /** high-speed cars owned (T-817; absent in older saves) */
  fleetFast?: number;
  /** T-1464: buses owned; absent for none */
  fleetBus?: number;
  /** T-2194: boats owned; absent for none */
  fleetFerry?: number;
  /** fare curve (game/fareCurve.ts): its first stretch as a base and rate, and its nodes (none in a save from before T-883) */
  fares: { base: number; perKm: number; nodes?: [number, number][]; scales?: Record<string, number>; baseline?: string; basis?: "usd"; hsr?: number; ferry?: number; transfer?: number };
  /** recent days, oldest first (today last) */
  ledger: DayLedger[];
  /** recent hours, oldest first (T-447; absent in older saves) */
  hours?: HourLedger[];
  /** T-531: the same two per region (a zone id; "" outside every zone); absent in older saves */
  regions?: Record<string, { ledger: DayLedger[]; hours: HourLedger[] }>;
  /** T-685: each region's star, the game second it was earned; absent in older saves */
  stars?: Record<string, number>;
  /** T-1437: each region's higher goals by share (%), the game second each was reached; absent in older saves */
  goals?: Record<string, Record<string, number>>;
  /** T-706: each line's station note (semitones above C5) and the city it was chosen in, kept for the
   * line's life and after it so the others keep theirs; absent in older saves (map/stationLife.ts) */
  notes?: Record<string, [number, string]>;
  /** T-728: absent in older saves, read as "normal" */
  mode?: GameMode;
  /** T-728: the zone the game started in; absent for a start on the world map and in older saves */
  start?: string;
  /** T-728, T-729: the zones opened in this game, in the order they were opened; absent in older saves */
  opened?: string[];
  /** T-1223: the zones demand held fine (the rest coarse, notes/T-1206.md), so a reload solves the
   * same levels and shows the same figures; absent: every zone fine that fits */
  demandFine?: string[];
  /** T-2153: the zones of the last two builds (a station placed or a stop added), latest first, which
   * demand keeps fine (`builtKeep`); absent in older saves */
  demandBuilt?: string[][];
  /** T-1094: the train length last set on a line (0 for the most the platforms fit), a new line's start; absent in older saves */
  newCars?: number;
  /** T-1826: the save level of the build that wrote it (`TrackApi.save_level`, sim/src/track/save.rs
   * `LEVEL`); absent before it, read as 0 */
  level?: number;
  /** T-1822: the Getting started card's progress; absent where the game has none (older saves, sandboxes) */
  guide?: GuideSave;
  /** T-2124: the achievement steps earned, by key (game/achievements.ts); absent in older saves,
   * which work theirs out quietly on load */
  achievements?: Record<string, Achieved>;
}

/** T-2124: when a step was earned (game seconds) and what earned it, for the hovers: `of` the name of
 * the line, station or city, `v` the figure, `in` the countries or continents. Names are kept as they
 * were, so a later rename leaves them. */
export interface Achieved { at: number; of?: string; v?: number; in?: string[] }

/** T-1822: `done` is a bit per ticked step (`GUIDE_BITS` in game/guide.ts); `hidden`: closed with the
 * x, brought back in Settings; `closed`: finished and dismissed with Done; `card`: the second card is
 * up (T-2123); `zoom`: the game started on the world map, so the first card begins with a zoom step. */
export interface GuideSave { done: number; hidden?: true; closed?: true; card?: 2; zoom?: true }

/** T-1826: every header field this build reads, so a field added to `SaveHeader` must be listed. */
const HEADER_FIELDS: Record<keyof SaveHeader, true> = {
  game: true, format: true, city: true, region: true, cities: true, clock: true, cash: true, loan: true, loanPay: true, loanDays: true, fleet: true, fleetFast: true,
  fleetBus: true, fleetFerry: true, fares: true, ledger: true, hours: true, regions: true, stars: true, goals: true, notes: true, mode: true, start: true,
  opened: true, demandFine: true, demandBuilt: true, newCars: true, level: true, guide: true, achievements: true,
};

/** T-1826: whether a newer build wrote parts of this header that this one does not read: a higher
 * level, or (in a save with a level, so from a build that lists its fields) a field it does not know.
 * Saves from before levels can carry retired fields (`points`, T-694) and are judged by their
 * network bytes alone. */
export function headerUnread(h: SaveHeader, level: number): boolean {
  if (h.level === undefined) return false;
  return h.level > level || Object.keys(h).some((k) => !(k in HEADER_FIELDS));
}

/** T-728: a sandbox is the normal game with nothing refused for money and the cash not shown. */
export type GameMode = "normal" | "sandbox";

/** How a new game starts (T-728): its mode, its start zone (none for the world map), and for a
 * sandbox the start zone's real network as save bytes (T-731). */
export interface NewGameSpec { mode: GameMode; start?: string; real?: ArrayBuffer }

/**
 * Whether this build can open the save: it names the registry's world (or none, as legacy NYC
 * files do). Zones it names that the registry no longer has are `currentZones`' business (T-1053),
 * and which frame the track is in is the network bytes' (TWT5 world frame, TWT4 legacy, T-125).
 */
export function isNortheastSave(h: SaveHeader): boolean {
  return h.city === registry.cities[0].id && (h.region === undefined || h.region === registry.world);
}

const REGISTERED = new Set(registry.cities.map((c) => c.id));

/** T-1053: a zone id as the registry has it now. An id a replan removed becomes its successor in
 * game/zone-successors.json (pipeline/zone_successors.py), followed through later removals; null
 * where there is none. */
export function zoneNow(id: string, have: (id: string) => boolean = (z) => REGISTERED.has(z), table: Record<string, string | null> = successors): string | null {
  for (let hops = 0; !have(id); hops++) {
    const next = table[id];
    if (!next || hops > 32) return null;
    id = next;
  }
  return id;
}

/** T-1053: the header with every zone id it keeps (the cities, start and opened zones, the region
 * ledgers, stars and line notes) as the registry has it now: a removed zone's ledgers add into its
 * successor's (or "", outside every zone, where it has none), its star keeps the earlier time, and
 * an id with no successor is dropped. Track, stations and lines are positions and need nothing. */
export function currentZones(h: SaveHeader, now: (id: string) => string | null = (z) => zoneNow(z)): SaveHeader {
  const list = (ids: string[]) => [...new Set(ids.map(now).filter((z): z is string => z !== null))];
  const out: SaveHeader = { ...h };
  if (h.cities) out.cities = list(h.cities);
  if (h.opened) out.opened = list(h.opened);
  if (h.demandFine) out.demandFine = list(h.demandFine);
  if (h.demandBuilt) out.demandBuilt = h.demandBuilt.map(list).filter((b) => b.length);
  if (h.start !== undefined) out.start = now(h.start) ?? undefined;
  if (h.regions) {
    const regions: NonNullable<SaveHeader["regions"]> = {};
    for (const [k, r] of Object.entries(h.regions)) {
      const to = k === "" ? "" : now(k) ?? "", e = regions[to];
      regions[to] = e ? { ledger: [...e.ledger, ...(r.ledger ?? [])], hours: [...e.hours, ...(r.hours ?? [])] } : { ledger: [...(r.ledger ?? [])], hours: [...(r.hours ?? [])] };
    }
    out.regions = regions;
  }
  if (h.stars) {
    const stars: Record<string, number> = {};
    for (const [k, t] of Object.entries(h.stars)) {
      const to = now(k);
      if (to !== null) stars[to] = Math.min(stars[to] ?? Infinity, t);
    }
    out.stars = stars;
  }
  if (h.goals) {
    const goals: Record<string, Record<string, number>> = {};
    for (const [k, g] of Object.entries(h.goals)) {
      const to = now(k);
      if (to === null) continue;
      const e = (goals[to] ??= {});
      for (const [share, t] of Object.entries(g)) e[share] = Math.min(e[share] ?? Infinity, t);
    }
    out.goals = goals;
  }
  if (h.notes) out.notes = Object.fromEntries(Object.entries(h.notes).map(([line, [note, city]]) => [line, [note, now(city) ?? city]]));
  return out;
}

async function pipe(bytes: Uint8Array, s: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(s));
  return new Uint8Array(await out.arrayBuffer());
}

/**
 * Which frame a network's bytes are in, from their magic (T-308): TWT6 is the world's Mercator,
 * TWT5 the Northeast conic, TWT4 the legacy equirectangular frame. A load converts
 * anything but the new-game frame (`convert_save_frame`) before the track model reads it.
 */
export function trackFrameKind(track: Uint8Array): "merc" | "lcc" | "equirect" {
  const m = new TextDecoder().decode(track.subarray(0, 4));
  return m === "TWT6" ? "merc" : m === "TWT5" ? "lcc" : "equirect";
}

/** T-1146: whether the network bytes are in a format that no longer loads (TWT2, TWT3). */
export function isRetiredTrackFormat(track: Uint8Array): boolean {
  const m = new TextDecoder().decode(track.subarray(0, 4));
  return m === "TWT2" || m === "TWT3";
}

/** Header and network bytes -> the compressed file. */
export async function encodeSave(h: SaveHeader, track: Uint8Array): Promise<Uint8Array> {
  const json = new TextEncoder().encode(JSON.stringify(h));
  const raw = new Uint8Array(8 + json.length + track.length);
  raw.set(new TextEncoder().encode(SAVE_MAGIC), 0);
  new DataView(raw.buffer).setUint32(4, json.length, true);
  raw.set(json, 8);
  raw.set(track, 8 + json.length);
  return pipe(raw, new CompressionStream("gzip"));
}

/** The compressed file -> header and network bytes; throws a player-readable reason. */
export async function decodeSave(file: Uint8Array): Promise<{ header: SaveHeader; track: Uint8Array }> {
  let raw: Uint8Array;
  try {
    raw = await pipe(file, new DecompressionStream("gzip"));
  } catch {
    throw new Error("This file is not a trainsworld save.");
  }
  if (raw.length < 8 || new TextDecoder().decode(raw.subarray(0, 4)) !== SAVE_MAGIC) throw new Error("This file is not a trainsworld save.");
  const n = new DataView(raw.buffer, raw.byteOffset).getUint32(4, true);
  let header: SaveHeader;
  try {
    header = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + n)));
  } catch {
    throw new Error("This save is damaged.");
  }
  if (header.game !== "anitabuilder" || typeof header.format !== "number") throw new Error("This file is not a trainsworld save.");
  if (header.format > SAVE_FORMAT) throw new Error("This save is from a newer version of the game.");
  return { header, track: raw.slice(8 + n) };
}

// ---------------------------------------------------------------- IndexedDB

const DB = "anitabuilder";
const STORE = "saves";

/** What the saves list shows of a slot without opening it (T-241), made by the clock worker each
 * time it writes the slot. Absent in slots written before T-241 until they are next saved. */
export interface SlotSummary {
  /** a small picture of the network's track and lines in their colours: a PNG data URL, 480 x 320
   * px (drawn at half that size; 384 x 256 before T-894); absent for an empty network */
  thumb?: string;
  /** T-894: the picture's panels, one per city (the most stations first), as fractions of it; the
   * saves list writes each city's name over its panel. Absent in pictures from before T-894. */
  panels?: ThumbPanel[];
  /** the theme the picture was drawn in (T-543) */
  shade?: "light" | "dark";
  /** the cities the stations are in, most stations first (registry ids) */
  cities: string[];
  lines: number;
  stations: number;
  /** US$M */
  cash: number;
  /** T-728; absent in slots written before it */
  mode?: GameMode;
}

/** One city's panel in a save's picture (T-894): its registry id ("" for an unnamed panel over the
 * whole network), its stations, and where it is, as fractions of the picture's width and height. */
export interface ThumbPanel { city: string; stations: number; x: number; y: number; w: number; h: number; }

export interface StoredSave extends Partial<SlotSummary> {
  bytes: Uint8Array;
  /** Date.now() when written */
  at: number;
  day: number;
  name?: string;
  city?: string;
  /** T-2037: the id of the game this save descends from (a new game's own id; copies, conflicts and
   * restored backups keep their parent's); absent before T-2037 until `stampOrigins` */
  origin?: string;
  /** T-2037: Date.now() of the latest player change in these bytes (an edit, a list order, sandbox;
   * not the clock's own saves); absent before T-2037, read as `at` (versions.ts `editOf`) */
  edit?: number;
}

export interface SaveSlot extends Partial<SlotSummary> { key: string; name: string; at: number; day: number; /** T-988 */ cloud?: SyncRecord; /** T-2037 */ origin?: string; edit?: number; }

/** T-988: a slot's tie to a game on the account (`sync:<slot>`, deleted with the slot). Absent, or
 * without `acct`, for a slot tied to no account (a game made while signed out). The slot has changed
 * here since its last sync when its `at` is not `syncedAt`. */
export interface SyncRecord {
  /** the game id, also the leaderboards' */
  id: string;
  acct?: string;
  /** the account's rev this slot last uploaded or downloaded; 0 before its first upload */
  rev: number;
  /** the slot's `at` at that moment */
  syncedAt: number;
  /** T-2037: the slot's `edit` at that moment; absent before T-2037, read as `syncedAt` */
  edit?: number;
  /** Date.now() of the last upload that went through */
  upAt?: number;
  /** when the account's copy was saved (ms), for "Saved to your account" */
  okAt?: number;
  /** another device opened the game: this one stopped saving it, and its next load takes the
   * account's copy without a fork */
  gaveWay?: boolean;
  /** kept in this browser only (the sign-in question's answer) */
  local?: boolean;
  /** why the last upload failed */
  error?: string;
}
export const ORIGINAL_SLOT = "autosave";
export const ORIGINAL_NAME = "Original game";
export function slotName(key: string, save?: StoredSave | null): string { return save?.name || (key === ORIGINAL_SLOT ? ORIGINAL_NAME : "Saved game"); }

/** T-1829: a connection the browser closes (site data cleared, the database deleted in another tab)
 * is forgotten, so the next call opens the database again, made afresh if it went. */
function open(): Promise<IDBDatabase> {
  const p: Promise<IDBDatabase> = new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => {
      const d = r.result;
      d.onclose = () => { if (db === p) db = null; };
      d.onversionchange = () => { d.close(); if (db === p) db = null; };
      res(d);
    };
    r.onerror = () => rej(r.error);
  });
  return p;
}

let db: Promise<IDBDatabase> | null = null;
/** A store in a new transaction; a connection that fails (closed, or never opened) is dropped and
 * opened once more (T-1829). */
async function store(mode: IDBTransactionMode): Promise<IDBObjectStore> {
  for (let tries = 0; ; tries++) {
    const p = (db ??= open());
    try {
      return (await p).transaction(STORE, mode).objectStore(STORE);
    } catch (err) {
      if (db === p) db = null;
      if (tries) throw err;
    }
  }
}

/** T-1829: after a write failed, the connection is dropped (the next opens it again) and, when the
 * browser said it is out of room, every demand cache goes (rebuilt by the next solve); true if the
 * write is worth one more try. */
export async function makeRoom(err: unknown): Promise<boolean> {
  db = null;
  const name = (err as { name?: string } | null)?.name;
  if (name === "QuotaExceededError" || name === "UnknownError") await dropDemandCaches();
  return true;
}

/** T-1829: every slot's demand cache and its pointer (T-328, T-1539), which the next solve rebuilds. */
export async function dropDemandCaches(): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    for (const k of await req(s.getAllKeys())) if (typeof k === "string" && (k.startsWith(DEMAND_PREFIX) || k.startsWith(AHEAD_PREFIX))) s.delete(k);
    await done;
    return true;
  } catch {
    return false;
  }
}

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((res, rej) => {
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

export async function readStored(key: string): Promise<StoredSave | null> {
  try {
    return ((await req((await store("readonly")).get(key))) as StoredSave | undefined) ?? null;
  } catch {
    return null; // storage blocked (private window, previews): play without autosave
  }
}

function committed(tx: IDBTransaction): Promise<void> {
  const done = new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = tx.onerror = () => reject(tx.error ?? new Error("Browser storage is unavailable."));
  });
  // A request can fail before the caller reaches await done; retain the error without an
  // unhandled rejection while it unwinds through the request promise.
  void done.catch(() => {});
  return done;
}

export async function writeStored(key: string, v: StoredSave): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    await req(s.put(v, key));
    await done;
    return true;
  } catch {
    return false;
  }
}

/** Metadata only leaves the worker; unreadable-save recovery copies are kept out of the list. Only
 * the slots are read, not the demand caches beside them (several MB each, T-328). Also read by the
 * start menu on the main thread (T-677). */
export async function listSlots(): Promise<SaveSlot[]> {
  const s = await store("readonly");
  const keys = (await req(s.getAllKeys())).filter((k): k is string => typeof k === "string" && (k === ORIGINAL_SLOT || k.startsWith("slot-")));
  const saves = await Promise.all(keys.map((k) => req(s.get(k)) as Promise<StoredSave>));
  const syncs = await Promise.all(keys.map((k) => req(s.get(SYNC_PREFIX + k)) as Promise<SyncRecord | undefined>));
  return keys.flatMap((key, i) => {
    const s = saves[i];
    if (!s) return [];
    if (syncs[i]) seeSync(key, syncs[i]);
    return [{ key, name: slotName(key, s), at: s.at, day: s.day, thumb: s.thumb, panels: s.panels, shade: s.shade, cities: s.cities, lines: s.lines, stations: s.stations, cash: s.cash, mode: s.mode, cloud: syncs[i], origin: s.origin, edit: s.edit }];
  }).sort((a, b) => b.at - a.at);
}

/** Save while preserving a slot's latest name (it may have been renamed elsewhere; `lastName`, the
 * name last seen, where the slot has gone, T-1829) and its origin and latest change when `v` has none
 * (T-2037); true if the slot was not there before. */
export async function writeSlot(key: string, v: StoredSave, lastName?: string): Promise<boolean> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  const old = await req<StoredSave | undefined>(s.get(key));
  await req(s.put({ ...v, name: v.name ?? old?.name ?? lastName ?? slotName(key), origin: v.origin ?? old?.origin, edit: v.edit ?? old?.edit }, key));
  await done;
  return !old;
}

/** T-2037: an origin written into each slot that has none (`origins`, by slot key), in one transaction,
 * with the name `names` gives it, if any; nothing else in the slot changes. */
export async function stampOrigins(origins: Map<string, string>, names: Map<string, string> = new Map()): Promise<void> {
  if (!origins.size) return;
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    for (const [key, origin] of origins) {
      const old = await req<StoredSave | undefined>(s.get(key));
      if (old && !old.origin) s.put({ ...old, origin, ...(names.has(key) ? { name: names.get(key) } : {}) }, key);
    }
    await done;
  } catch { /* stamped next time */ }
}

const SYNC_PREFIX = "sync:";
/** T-1829: each slot's sync record as last read or written here, to put back if the database is
 * cleared under a running game. */
const syncSeen = new Map<string, SyncRecord>();
const seeSync = (key: string, v: SyncRecord | null | undefined) => { if (v) syncSeen.set(key, v); else syncSeen.delete(key); };
export function lastSync(key: string): SyncRecord | null { return syncSeen.get(key) ?? null; }
export async function readSync(key: string): Promise<SyncRecord | null> {
  try {
    const v = ((await req((await store("readonly")).get(SYNC_PREFIX + key))) as SyncRecord | undefined) ?? null;
    if (v) seeSync(key, v); // one found missing may have been cleared: the last seen is kept
    return v;
  } catch {
    return null;
  }
}
export async function writeSync(key: string, v: SyncRecord | null): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    if (v) await req(s.put(v, SYNC_PREFIX + key));
    else await req(s.delete(SYNC_PREFIX + key));
    await done;
    seeSync(key, v);
    return true;
  } catch {
    return false;
  }
}

/** T-988: games deleted here whose deletion has not reached the account yet */
const CLOUD_DELETES = "cloud-deletes";
export interface CloudDelete { id: string; acct: string }
export async function readCloudDeletes(): Promise<CloudDelete[]> {
  try {
    return ((await req((await store("readonly")).get(CLOUD_DELETES))) as CloudDelete[] | undefined) ?? [];
  } catch {
    return [];
  }
}
export async function writeCloudDeletes(v: CloudDelete[]): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    if (v.length) await req(s.put(v, CLOUD_DELETES));
    else await req(s.delete(CLOUD_DELETES));
    await done;
    return true;
  } catch {
    return false;
  }
}

export async function renameSlot(key: string, name: string): Promise<void> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  const old = await req<StoredSave | undefined>(s.get(key));
  if (!old) throw new Error("That save slot no longer exists.");
  await req(s.put({ ...old, name: name.trim() || slotName(key) }, key));
  await done;
}

export async function deleteSlot(key: string): Promise<void> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  await req(s.delete(key));
  await req(s.delete(SYNC_PREFIX + key));
  await req(s.delete(AHEAD_PREFIX + key));
  // and its demand cache (T-328) and backups (T-744)
  for (const k of await req(s.getAllKeys())) if (typeof k === "string" && (k.startsWith(`${DEMAND_PREFIX}${key}:`) || k.startsWith(`${BACKUP_PREFIX}${key}:`))) s.delete(k);
  await done;
  syncSeen.delete(key);
}

// T-1827 kept deleted slots 30 days under `deleted:<ms>-<id>` (backups `backup:deleted:...`); Anita
// removed it the same day (a deleted game should be gone for good). Anything it left is swept at start.
const DELETED_PREFIX = "deleted:", DELETED_BACKUP = "backup:deleted:";
export async function sweepDeleted(): Promise<void> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    for (const k of await req(s.getAllKeys())) if (typeof k === "string" && (k.startsWith(DELETED_PREFIX) || k.startsWith(DELETED_BACKUP))) s.delete(k);
    await done;
  } catch { /* storage unavailable */ }
}

/** The first "Game N" no slot is called (T-677, T-743). */
export function nextGameName(slots: { name: string }[]): string {
  const names = new Set(slots.map((s) => s.name));
  let n = slots.length + 1;
  while (names.has(`Game ${n}`)) n++;
  return `Game ${n}`;
}

/** A new game's slot name (T-728): its start city, or "World" for a map start, then " sandbox" for
 * a sandbox, then " 2", " 3" and on if a slot already has the name. */
export function newGameName(slots: { name: string }[], spec: NewGameSpec): string {
  const start = registry.cities.find((c) => c.id === spec.start);
  const base = (start ? zoneName(start.id, start.name) : "World") + (spec.mode === "sandbox" ? " sandbox" : "");
  const names = new Set(slots.map((s) => s.name));
  let name = base;
  for (let n = 2; names.has(name); n++) name = `${base} ${n}`;
  return name;
}

// T-744, T-1826: a slot's backups, the slot's record without its picture: four tiers
// `backup:<slot>:t0` to `:t3`, reaching back about 10 minutes, an hour, a day and a week, and
// `backup:<slot>:pull`, the slot as it was before the account's copy last replaced it. Backups from
// before T-1826 (`:0`, `:1`) are listed until the next tier write folds them in.
const BACKUP_PREFIX = "backup:";
const TIERS = ["t0", "t1", "t2", "t3"];
/** T-1826: wall ms. Tier 0 takes the slot at most once a step; the backup it held moves down to
 * tier i+1 only when it is at least step i+1 newer than the one there, else it goes. So tier 0 is up
 * to 10 min old, tier 1 up to about 1 h 10 min, tier 2 about a day, tier 3 about a week. */
export const BACKUP_STEPS = [10 * 60_000, 3600_000, 86400_000, 7 * 86400_000];
/** T-1826: the most a slot's tier backups hold, bytes; past it the oldest tiers are dropped (tier 0
 * is always kept). About 100 KB a save is usual. */
export const BACKUP_CAP = 8 << 20;
export interface SlotBackup { key: string; at: number; day: number; /** the slot before the account's copy replaced it */ pull?: boolean; /** T-2037 */ stations?: number; edit?: number }
const backupKey = (slot: string, tier: string) => `${BACKUP_PREFIX}${slot}:${tier}`;

export async function listBackups(slot: string): Promise<SlotBackup[]> {
  const s = await store("readonly");
  const keys = (await req(s.getAllKeys())).filter((k): k is string => typeof k === "string" && isBackupOf(k, slot));
  const out: SlotBackup[] = [];
  for (const key of keys) {
    const b = (await req(s.get(key))) as StoredSave | undefined;
    if (b) out.push({ key, at: b.at, day: b.day, stations: b.stations, edit: b.edit, ...(key.endsWith(":pull") ? { pull: true } : {}) });
  }
  return out.sort((a, b) => b.at - a.at);
}

/** T-1826: the tiers after `add` (newest first, `null` for an empty tier), or null when tier 0 is
 * less than a step old and nothing changes. */
export function planBackups<T extends { at: number }>(tiers: (T | null)[], add: T, steps: number[] = BACKUP_STEPS): (T | null)[] | null {
  if (tiers[0] && add.at - tiers[0].at < steps[0]) return null;
  const out = steps.map((_, i) => tiers[i] ?? null);
  let carry: T | null = add;
  for (let i = 0; i < steps.length && carry; i++) {
    const old: T | null = out[i];
    out[i] = carry;
    const next = out[i + 1];
    carry = old && i + 1 < steps.length && (!next || old.at - next.at >= steps[i + 1]) ? old : null;
  }
  return out;
}

/** An autosave's backup, written when tier 0 is a step old (`planBackups`), in one transaction. */
export async function writeBackup(slot: string, v: StoredSave, steps: number[] = BACKUP_STEPS): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const get = async (tier: string) => ((await req(s.get(backupKey(slot, tier)))) as StoredSave | undefined) ?? null;
    let tiers = [await get("t0")];
    if (tiers[0] && v.at - tiers[0].at < steps[0]) { await done; return true; }
    for (const t of TIERS.slice(1)) tiers.push(await get(t));
    const legacy = [await get("0"), await get("1")].filter((b): b is StoredSave => !!b).sort((a, b) => b.at - a.at);
    if (legacy.length && tiers.every((t) => !t)) tiers = [legacy[0], legacy[1] ?? null, null, null];
    const { thumb: _, ...rest } = v;
    const plan = planBackups(tiers, rest, steps) ?? tiers;
    for (let i = 1, sum = plan[0]?.bytes.length ?? 0; i < plan.length; i++) {
      sum += plan[i]?.bytes.length ?? 0;
      if (sum > BACKUP_CAP) plan[i] = null;
    }
    plan.forEach((b, i) => { if (b) s.put(b, backupKey(slot, TIERS[i])); else s.delete(backupKey(slot, TIERS[i])); });
    s.delete(backupKey(slot, "0"));
    s.delete(backupKey(slot, "1"));
    await done;
    return true;
  } catch {
    return false;
  }
}

/** T-1826: before anything replaces slot `key` with another copy (the account's), the slot as it
 * was, under `backup:<slot>:pull`. Nothing is written when the slot is empty or already holds `next`. */
export async function backupBeforeReplace(key: string, next: { at: number; bytes: Uint8Array }): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const old = (await req(s.get(key))) as StoredSave | undefined;
    if (old && !(old.at === next.at && old.bytes.length === next.bytes.length)) {
      const { thumb: _, ...rest } = old;
      await req(s.put(rest, backupKey(key, "pull")));
    }
    await done;
    return true;
  } catch {
    return false;
  }
}

export function isBackupOf(key: string, slot: string): boolean {
  return key.startsWith(`${BACKUP_PREFIX}${slot}:`);
}
/** T-2037: the slot a backup key belongs to ("" for none). */
export function backupSlot(key: string): string {
  return key.startsWith(BACKUP_PREFIX) ? key.slice(BACKUP_PREFIX.length, key.lastIndexOf(":")) : "";
}

// T-328: the last final demand result of each slot, kept beside it under `demand:<slot>:<network
// hash>` (workers/demandCache.ts says what is in it). One per slot: writing one removes the slot's
// others. Looked up by the network's hash alone, since a whole-game load sends its network before
// the slot it came from is announced; the same network gives the same result whatever slot holds it.
const DEMAND_PREFIX = "demand:";

/** The cached demand result for a network hash, from any slot (the given slot's first). */
export async function readDemandCache<T>(hash: string, slot: string): Promise<T | null> {
  try {
    const s = await store("readonly");
    const keys = (await req(s.getAllKeys())).filter((k): k is string => typeof k === "string" && k.startsWith(DEMAND_PREFIX) && k.endsWith(":" + hash));
    const key = keys.find((k) => k === `${DEMAND_PREFIX}${slot}:${hash}`) ?? keys[0];
    return key ? ((await req(s.get(key))) as T | undefined) ?? null : null;
  } catch {
    return null;
  }
}

/** Store a slot's demand result for a network hash, removing the slot's older one; and, T-1539, the
 * slot's pointer to it from the save inputs whose load works out that network (`ahead.save`), if any. */
export async function writeDemandCache(slot: string, hash: string, v: unknown): Promise<boolean> {
  const save = (v as { ahead?: { save?: string } } | null)?.ahead?.save;
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const key = `${DEMAND_PREFIX}${slot}:${hash}`;
    for (const k of await req(s.getAllKeys())) if (typeof k === "string" && k !== key && k.startsWith(`${DEMAND_PREFIX}${slot}:`)) s.delete(k);
    await req(s.put(v, key));
    if (save) await req(s.put({ save, hash }, AHEAD_PREFIX + slot));
    else s.delete(AHEAD_PREFIX + slot);
    await done;
    return true;
  } catch {
    return false;
  }
}

// T-1539: each slot's pointer from the key of a save's network inputs to its demand result, which
// then holds that network: a load shows the result before its network is worked out. Small, so a
// load reads every slot's to find its own.
const AHEAD_PREFIX = "ahead:";

/** The demand result kept for the save inputs `save` (demandCache.ts `saveNetKey`), if any. */
export async function readDemandAhead<T>(save: string): Promise<T | null> {
  try {
    const s = await store("readonly");
    const keys = (await req(s.getAllKeys())).filter((k): k is string => typeof k === "string" && k.startsWith(AHEAD_PREFIX));
    for (const k of keys) {
      const a = (await req(s.get(k))) as { save: string; hash: string } | undefined;
      if (a?.save !== save) continue;
      const v = (await req(s.get(`${DEMAND_PREFIX}${k.slice(AHEAD_PREFIX.length)}:${a.hash}`))) as T | undefined;
      if (v) return v;
    }
    return null;
  } catch {
    return null;
  }
}

/** T-1196: the game a reloaded page opens in place of a slot: a file loaded, a backup restored or a
 * new game, picked in the saves panel. The page reloads to free the last game's memory; the clock
 * worker takes this from storage as it starts. */
export type PendingLoad =
  | { kind: "import"; bytes: Uint8Array; name: string; from: string; /** T-2037: an account backup's game */ origin?: string; edit?: number }
  | { kind: "restore"; key: string; from: string }
  | { kind: "new"; spec: NewGameSpec; name?: string };
const PENDING = "pending-load";
export async function writePending(p: PendingLoad): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    await req(s.put(p, PENDING));
    await done;
    return true;
  } catch {
    return false;
  }
}
/** The pending load, removed from storage (null if none) */
export async function takePending(): Promise<PendingLoad | null> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const p = (await req(s.get(PENDING))) as PendingLoad | undefined;
    s.delete(PENDING);
    await done;
    return p ?? null;
  } catch {
    return null;
  }
}

/** T-1395: the folder handle T-1214's folder copies kept here; the copies are gone, the handle is
 * forgotten */
export async function forgetCopyFolder(): Promise<void> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    await req(s.delete("copy-folder"));
    await done;
  } catch { /* storage unavailable */ }
}

/** `lost` settles when another tab takes the lock with `steal` (T-1251). */
export interface SlotLease { release: () => void; lost: Promise<void>; }
/** A browser-managed lock lives for this worker's lifetime and releases even if the tab crashes.
 * With `steal` it is taken from the tab holding it, whose request then rejects (its `lost`). */
export async function leaseSlot(key: string, steal = false): Promise<SlotLease | null> {
  if (!navigator.locks) return null;
  let lose = () => {};
  const lost = new Promise<void>((r) => { lose = r; });
  let granted = false;
  return new Promise((resolve, reject) => {
    void navigator.locks.request("anitabuilder-save-" + key, steal ? { steal: true } : { ifAvailable: true }, (lock) => {
      if (!lock) { resolve(null); return; }
      granted = true;
      return new Promise<void>((release) => resolve({ release, lost }));
    }).catch((err) => { if (granted) lose(); else reject(err); });
  });
}
