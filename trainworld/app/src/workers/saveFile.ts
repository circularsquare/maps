// The game save (SPEC 11, T-029): the player's inputs only. Built and read in the clock worker, so
// the main thread never serialises or compresses anything.
//
// File layout, gzip-compressed as a whole:
//   "TWG1"                 magic
//   u32 little-endian      length of the JSON header in bytes
//   JSON header (UTF-8)    `SaveHeader`: city, clock, money, fares, the day ledger
//   the rest               the track model's network save (`TrackApi.save()`, "TWT3" since T-079; "TWT2" loads)
//
// Nothing in it depends on wall time, so saving the same game twice gives the same bytes.
// Autosaves live in IndexedDB (database "anitabuilder", store "saves", key "autosave").

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
}

export interface SaveHeader {
  game: "anitabuilder";
  format: number;
  city: string;
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
}

async function pipe(bytes: Uint8Array, s: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(s));
  return new Uint8Array(await out.arrayBuffer());
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

export interface StoredSave {
  bytes: Uint8Array;
  /** Date.now() when written */
  at: number;
  day: number;
}

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

export async function writeStored(key: string, v: StoredSave): Promise<boolean> {
  try {
    await req((await store("readwrite")).put(v, key));
    return true;
  } catch {
    return false;
  }
}
