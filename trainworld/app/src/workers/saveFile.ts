// The game save (SPEC 11, T-029): the player's inputs only. Built and read in the clock worker, so
// the main thread never serialises or compresses anything.
//
// File layout, gzip-compressed as a whole:
//   "TWG1"                 magic
//   u32 little-endian      length of the JSON header in bytes
//   JSON header (UTF-8)    `SaveHeader`: city, clock, money, fares, the day and hour ledgers (the money
//                          and train rides history charts), each also per region
//   the rest               the track model's network save (`TrackApi.save()`: "TWT6" Mercator world frame since
//                          T-306; older saves, "TWT5" conic since T-125, "TWT4" and TWT2/3 legacy frame, are
//                          converted to Mercator when they load, T-308)
//
// Nothing in it depends on wall time, so saving the same game twice gives the same bytes.
// Autosaves live in IndexedDB (database "anitabuilder", store "saves", one key per slot; the original game keeps key "autosave";
// beside each slot its demand cache and two backups under keys the saves list does not show).

import registry from "../game/registry.json";
import successors from "../game/zone-successors.json";
import { zoneName } from "../game/zoneNames";

export const SAVE_MAGIC = "TWG1";
export const SAVE_FORMAT = 1;
/** SPEC 8: a new game starts on day 1 at 07:00 (game seconds). */
export const START_CLOCK = 86400 + 7 * 3600;
/** SPEC 6.4: US$6B to start, in US$M; a sandbox's money since the start is the cash less this (T-728). */
export const START_CASH = 6000;

/** One game day's money, US$M (T-028). */
export interface DayLedger {
  day: number;
  fares: number;
  running: number;
  trains: number;
  build: number;
  /** track and station upkeep (T-867); absent in days saved before it, read as 0 */
  upkeep?: number;
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
  /** cars owned */
  fleet: number;
  /** high-speed cars owned (T-817; absent in older saves) */
  fleetFast?: number;
  /** fare curve (game/fareCurve.ts): its first stretch as a base and rate, and its nodes (none in a save from before T-883) */
  fares: { base: number; perKm: number; nodes?: [number, number][] };
  /** recent days, oldest first (today last) */
  ledger: DayLedger[];
  /** recent hours, oldest first (T-447; absent in older saves) */
  hours?: HourLedger[];
  /** T-531: the same two per region (a zone id; "" outside every zone); absent in older saves */
  regions?: Record<string, { ledger: DayLedger[]; hours: HourLedger[] }>;
  /** T-685: each region's star, the game second it was earned; absent in older saves */
  stars?: Record<string, number>;
  /** T-706: each line's station note (semitones above C5) and the city it was chosen in, kept for the
   * line's life and after it so the others keep theirs; absent in older saves (map/stationLife.ts) */
  notes?: Record<string, [number, string]>;
  /** T-728: absent in older saves, read as "normal" */
  mode?: GameMode;
  /** T-728: the zone the game started in; absent for a start on the world map and in older saves */
  start?: string;
  /** T-728, T-729: the zones opened in this game, in the order they were opened; absent in older saves */
  opened?: string[];
  /** T-1094: the train length last set on a line (0 for the most the platforms fit), a new line's start; absent in older saves */
  newCars?: number;
}

/** T-728: a sandbox is the normal game with nothing refused for money and the cash not shown. */
export type GameMode = "normal" | "sandbox";

/** How a new game starts (T-728): its mode, its start zone (none for the world map), and for a
 * sandbox the start zone's real network as save bytes (T-731). */
export interface NewGameSpec { mode: GameMode; start?: string; real?: ArrayBuffer }

/**
 * Whether this build can open the save: it names the registry's world (or none, as legacy NYC
 * files do). Zones it names that the registry no longer has are `currentZones`' business (T-1053),
 * and which frame the track is in is the network bytes' (TWT5 world frame, TWT4 and older legacy, T-125).
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
  if (h.notes) out.notes = Object.fromEntries(Object.entries(h.notes).map(([line, [note, city]]) => [line, [note, now(city) ?? city]]));
  return out;
}

async function pipe(bytes: Uint8Array, s: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(s));
  return new Uint8Array(await out.arrayBuffer());
}

/**
 * Which frame a network's bytes are in, from their magic (T-308): TWT6 is the world's Mercator,
 * TWT5 the Northeast conic, TWT4 and older the legacy equirectangular frame. A load converts
 * anything but the new-game frame (`convert_save_frame`) before the track model reads it.
 */
export function trackFrameKind(track: Uint8Array): "merc" | "lcc" | "equirect" {
  const m = new TextDecoder().decode(track.subarray(0, 4));
  return m === "TWT6" ? "merc" : m === "TWT5" ? "lcc" : "equirect";
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
    throw new Error("This file is not an anitabuilder save.");
  }
  if (raw.length < 8 || new TextDecoder().decode(raw.subarray(0, 4)) !== SAVE_MAGIC) throw new Error("This file is not an anitabuilder save.");
  const n = new DataView(raw.buffer, raw.byteOffset).getUint32(4, true);
  let header: SaveHeader;
  try {
    header = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + n)));
  } catch {
    throw new Error("This save is damaged.");
  }
  if (header.game !== "anitabuilder" || typeof header.format !== "number") throw new Error("This file is not an anitabuilder save.");
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
}

export interface SaveSlot extends Partial<SlotSummary> { key: string; name: string; at: number; day: number; }
export const ORIGINAL_SLOT = "autosave";
export const ORIGINAL_NAME = "Original game";
export function slotName(key: string, save?: StoredSave | null): string { return save?.name || (key === ORIGINAL_SLOT ? ORIGINAL_NAME : "Saved game"); }

function open(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

let db: Promise<IDBDatabase> | null = null;
function store(mode: IDBTransactionMode): Promise<IDBObjectStore> {
  db ??= open();
  return db.then((d) => d.transaction(STORE, mode).objectStore(STORE));
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
  return keys.flatMap((key, i) => {
    const s = saves[i];
    if (!s) return [];
    return [{ key, name: slotName(key, s), at: s.at, day: s.day, thumb: s.thumb, panels: s.panels, shade: s.shade, cities: s.cities, lines: s.lines, stations: s.stations, cash: s.cash, mode: s.mode }];
  }).sort((a, b) => b.at - a.at);
}

/** Save while preserving a slot's latest name (it may have been renamed elsewhere). */
export async function writeSlot(key: string, v: StoredSave): Promise<void> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  const old = await req<StoredSave | undefined>(s.get(key));
  await req(s.put({ ...v, name: v.name ?? old?.name ?? slotName(key) }, key));
  await done;
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
  // and its demand cache (T-328) and backups (T-744)
  for (const k of await req(s.getAllKeys())) if (typeof k === "string" && (k.startsWith(`${DEMAND_PREFIX}${key}:`) || k.startsWith(`${BACKUP_PREFIX}${key}:`))) s.delete(k);
  await done;
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

// T-744: two backups of each slot, `backup:<slot>:0` and `:1`, the older one written over next. A
// backup is the slot's record without its picture.
const BACKUP_PREFIX = "backup:";
export interface SlotBackup { key: string; at: number; day: number; }

export async function listBackups(slot: string): Promise<SlotBackup[]> {
  const out: SlotBackup[] = [];
  for (const i of [0, 1]) {
    const key = `${BACKUP_PREFIX}${slot}:${i}`, b = await readStored(key);
    if (b) out.push({ key, at: b.at, day: b.day });
  }
  return out.sort((a, b) => b.at - a.at);
}

/** Write a slot's backup over its older one (or into an empty place). */
export async function writeBackup(slot: string, v: StoredSave): Promise<boolean> {
  const have = await listBackups(slot);
  const free = [0, 1].map((i) => `${BACKUP_PREFIX}${slot}:${i}`).find((k) => !have.some((b) => b.key === k));
  const { thumb: _, ...rest } = v;
  return writeStored(free ?? have[have.length - 1].key, rest);
}

export function isBackupOf(key: string, slot: string): boolean {
  return key.startsWith(`${BACKUP_PREFIX}${slot}:`);
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

/** Store a slot's demand result for a network hash, removing the slot's older one. */
export async function writeDemandCache(slot: string, hash: string, v: unknown): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const key = `${DEMAND_PREFIX}${slot}:${hash}`;
    for (const k of await req(s.getAllKeys())) if (typeof k === "string" && k !== key && k.startsWith(`${DEMAND_PREFIX}${slot}:`)) s.delete(k);
    await req(s.put(v, key));
    await done;
    return true;
  } catch {
    return false;
  }
}

export interface SlotLease { release: () => void; }
/** A browser-managed lock lives for this worker's lifetime and releases even if the tab crashes. */
export async function leaseSlot(key: string): Promise<SlotLease | null> {
  if (!navigator.locks) return null;
  return new Promise((resolve, reject) => {
    void navigator.locks.request("anitabuilder-save-" + key, { ifAvailable: true }, (lock) => {
      if (!lock) { resolve(null); return; }
      return new Promise<void>((release) => resolve({ release }));
    }).catch(reject);
  });
}
