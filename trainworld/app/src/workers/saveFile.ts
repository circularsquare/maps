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
// Autosaves live in IndexedDB (database "anitabuilder", store "saves", one key per slot; the original game keeps key "autosave").

import registry from "../game/registry.json";

export const SAVE_MAGIC = "TWG1";
export const SAVE_FORMAT = 1;
/** SPEC 8: a new game starts on day 1 at 07:00 (game seconds). */
export const START_CLOCK = 86400 + 7 * 3600;

/** One game day's money, US$M (T-028). */
export interface DayLedger {
  day: number;
  fares: number;
  running: number;
  trains: number;
  build: number;
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
  /** fare curve: US$ a ride plus US$ a km */
  fares: { base: number; perKm: number };
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
}

/**
 * Whether this build can open the save: it names the registry's world (or none, as legacy NYC
 * files do) and only cities the registry lists. Which frame the track is in is not the header's
 * business: the network bytes say (TWT5 world frame, TWT4 and older legacy, T-125).
 */
export function isNortheastSave(h: SaveHeader): boolean {
  const ids = registry.cities.map(c => c.id);
  return h.city === ids[0] && (h.region === undefined || h.region === registry.world)
    && (h.cities === undefined || h.cities.every(c => ids.includes(c)));
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
  /** a small picture of the network's track and lines in their colours: a PNG data URL, 384 x 256
   * px (drawn at half that size); absent for an empty network */
  thumb?: string;
  /** the cities the stations are in, most stations first (registry ids) */
  cities: string[];
  lines: number;
  stations: number;
  /** US$M */
  cash: number;
}

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
    return [{ key, name: slotName(key, s), at: s.at, day: s.day, thumb: s.thumb, cities: s.cities, lines: s.lines, stations: s.stations, cash: s.cash }];
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
  // and its demand cache (T-328)
  for (const k of await req(s.getAllKeys())) if (typeof k === "string" && k.startsWith(`${DEMAND_PREFIX}${key}:`)) s.delete(k);
  await done;
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
