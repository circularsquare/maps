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
import { compressGzip, decompressGzip } from "./gzip";
import { compressGzipParts, GZIP_DAMAGED } from "./gzipCore";
import { editOf } from "./versions";
import type { LineShape } from "../game/lineShape";

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

/** One game hour's money, US$M (T-447); `hour` counts from day 0 00:00. The figures are `DayLedger`'s. */
export interface HourLedger extends Omit<DayLedger, "day"> {
  hour: number;
}

/** An empty entry for game day `day` or hour `hour`. */
export const newDay = (day: number): DayLedger => ({ day, fares: 0, running: 0, trains: 0, build: 0, rides: 0 });
export const newHour = (hour: number): HourLedger => ({ hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 });

/** Add (`sign` 1) or take away (-1) the figures a day's money and an hour's share: `e` into `to`. */
export function addEntry(to: Omit<DayLedger, "day">, e: Omit<DayLedger, "day">, sign = 1): void {
  to.fares += sign * e.fares;
  to.running += sign * e.running;
  to.upkeep = (to.upkeep ?? 0) + sign * (e.upkeep ?? 0);
  to.rides = (to.rides ?? 0) + sign * (e.rides ?? 0);
  to.points = (to.points ?? 0) + sign * (e.points ?? 0);
}

/** T-3543, T-3588: the headways new lines start with, by kind of line: seconds at high, medium and low demand, or
 * (T-3767, set from a line of four or five rows) one per period, morning to night (game/newLine.ts `headwaysIn`) */
export type Headways = number[];
export type NewHeadways = { rail?: Headways; tram?: Headways; fast?: Headways; bus?: Headways; ferry?: Headways };
/** The kinds of line a default headway is kept for (T-3703: bus and ferry too) */
export const NEW_HEADWAY_KINDS = ["rail", "tram", "fast", "bus", "ferry"] as const;
export const isHeadways = (v: unknown): v is Headways => Array.isArray(v) && (v.length === 3 || v.length === 5) && v.every((s) => typeof s === "number" && Number.isFinite(s) && s >= 0);
/** A header's `newHeadway` read back: the kinds it holds well-formed, none from anything else. */
export function cleanNewHeadways(v: unknown): NewHeadways {
  const out: NewHeadways = {};
  if (v && typeof v === "object" && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    for (const k of NEW_HEADWAY_KINDS) { const h = o[k]; if (isHeadways(h)) out[k] = [...h]; }
  }
  return out;
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
  /** T-2926: tram cars owned; absent for none */
  fleetTram?: number;
  /** fare curve (game/fareCurve.ts): its first stretch as a base and rate, and its nodes (none in a save from before T-883) */
  fares: { base: number; perKm: number; nodes?: [number, number][]; scales?: Record<string, number>; byCity?: boolean; cityScales?: Record<string, number>; baseline?: string; basis?: "usd"; income?: number; hsr?: number; ferry?: number; bus?: number; transfer?: number; ref?: string; lines?: Record<string, number>; zones?: { id: number; name: string; prices: number[]; off?: boolean }[]; zoneNext?: number };
  /** recent days, oldest first (today last) */
  ledger: DayLedger[];
  /** recent hours, oldest first (T-447; absent in older saves) */
  hours?: HourLedger[];
  /** T-531: the same two per region (a zone id; "" outside every zone); absent in older saves */
  regions?: Record<string, { ledger: DayLedger[]; hours: HourLedger[] }>;
  /** T-2787: each line's recent days of money, by line id: [day, fares, running cost] flattened, US$M,
   * oldest first (the last month); absent in older saves, which start with no history */
  lineDays?: Record<string, number[]>;
  /** T-685: each region's star, the game second it was earned; absent in older saves */
  stars?: Record<string, number>;
  /** T-1437: each region's higher goals by share (%), the game second each was reached; absent in older saves */
  goals?: Record<string, Record<string, number>>;
  /** T-706: each line's station note (semitones above C5) and the city it was chosen in, kept for the
   * line's life and after it so the others keep theirs; absent in older saves (map/stationLife.ts) */
  notes?: Record<string, [number, string]>;
  /** T-3319: the bullet shape of each line the player gave one, by line id; absent for none (older saves
   * too), every line then taking its mode's from Settings */
  shapes?: Record<string, LineShape>;
  /** T-3420: the ids of the lines the player hid on the map; absent for none (older saves too) */
  hidden?: string[];
  /** T-3409: the zones the schematic is of; absent for the whole network (older saves too); `[NO_CITIES]` for none (T-3886) */
  schematic?: string[];
  /** T-728: absent in older saves, read as "normal" */
  mode?: GameMode;
  /** T-728: the zone the game started in; absent for a start on the world map and in older saves */
  start?: string;
  /** T-3407: seconds the ledger days run ahead of the clock in a game with no start zone, set by its first built station; absent before one is built */
  dayShift?: number;
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
  /** T-3543: the headways, seconds at high, medium and low demand, every new line starts with (set from the timetables window); absent: the game's own counts */
  newHeadway?: NewHeadways;
  newWeekendShare?: number;
  /** T-1826: the save level of the build that wrote it (`TrackApi.save_level`, sim/src/track/save.rs
   * `LEVEL`); absent before it, read as 0 */
  level?: number;
  /** T-1822: the Getting started card's progress; absent where the game has none (older saves, sandboxes) */
  guide?: GuideSave;
  /** T-2124: the achievement steps earned, by key (game/achievements.ts); absent in older saves,
   * which work theirs out quietly on load */
  achievements?: Record<string, Achieved>;
  /** T-2395: where the game opens, a zone id or `WORLD_HOME`; absent: as `startOpenFit` picks */
  home?: string;
  /** T-3115: the Community share this game is a copy of (game/sharedSave.ts); a copy plays as normal but its
   * figures never reach the leaderboards. Absent for a game of the player's own */
  copy?: string;
  /** T-3707: why the game is modded (game/modded.ts: a mod's name, "unofficial build", "site:<host>"), every reason
   * seen, never removed; a modded game's figures never reach the leaderboards. Absent for the published game's */
  modded?: string[];
}

/** T-2395: `SaveHeader.home` for all the player's cities (the built area, zoomed out) */
export const WORLD_HOME = "world";

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
  fleetBus: true, fleetFerry: true, fleetTram: true, fares: true, ledger: true, hours: true, regions: true, lineDays: true, stars: true, goals: true, notes: true, shapes: true, hidden: true, schematic: true, mode: true, start: true,
  dayShift: true, opened: true, demandFine: true, demandBuilt: true, newCars: true, newHeadway: true, newWeekendShare: true, level: true, guide: true, achievements: true, home: true, copy: true, modded: true,
};
/** T-3115: the header fields, for checking a stranger's save (game/sharedSave.ts) */
export const SAVE_HEADER_FIELDS: ReadonlySet<string> = new Set(Object.keys(HEADER_FIELDS));

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
/** `rightHand` (T-3297): the running side picked, false for trains, trams and buses keeping left; absent: right */
export interface NewGameSpec { mode: GameMode; start?: string; real?: ArrayBuffer; rightHand?: boolean }

/**
 * Whether this build can open the save: it names the registry's world (or none, as legacy NYC
 * files do). Zones it names that the registry no longer has are `currentZones`' business (T-1053),
 * and which frame the track is in is the network bytes' (TWT5 world frame, TWT4 legacy, T-125).
 */
export function isThisWorld(h: SaveHeader): boolean {
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

/** T-1053: the header with every zone id it keeps (the cities, start, home and opened zones, the region
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
  if (h.home !== undefined && h.home !== WORLD_HOME) out.home = now(h.home) ?? undefined;
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

/** Header and network bytes -> the compressed file. In a worker (the clock worker's saves) it is made in parts
 * (`encodeSaveParts`) and joined; on the page the codec worker packs it. */
export async function encodeSave(h: SaveHeader, track: Uint8Array): Promise<Uint8Array> {
  if (typeof document === "undefined") return joinParts(encodeSaveParts(h, track));
  const json = new TextEncoder().encode(JSON.stringify(h));
  const raw = new Uint8Array(8 + json.length + track.length);
  raw.set(new TextEncoder().encode(SAVE_MAGIC), 0);
  new DataView(raw.buffer).setUint32(4, json.length, true);
  raw.set(json, 8);
  raw.set(track, 8 + json.length);
  return compressGzip(raw);
}

/** T-3869: a buffer the save asks for, refused as a browser short of memory refuses one; checks only
 * (`refuseSaveBuffers`), null otherwise */
let refuse: ((bytes: number) => boolean) | null = null;
export function refuseSaveBuffers(f: ((bytes: number) => boolean) | null) { refuse = f; }
/** V8's own words for a buffer it could not get */
export const NO_BUFFER = "Array buffer allocation failed"; // i18n-ignore: an engine's error text
function room(n: number) { if (refuse?.(n)) throw new RangeError(NO_BUFFER); }

/** T-3869: the compressed file in parts of at most GZ_PART bytes, the same bytes `encodeSave` gives joined, without
 * the file or the bytes it packs ever held whole: the header is written out a piece at a time (`savePieces`). Worker
 * only (it compresses on the thread that calls it). */
export function encodeSaveParts(h: SaveHeader, track: Uint8Array): Uint8Array<ArrayBuffer>[] {
  room(track.length);
  const json = JSON.stringify(h);
  return compressGzipParts(() => savePieces(json, track), room);
}

/** The file's bytes before compression, in pieces: the magic and the header's length, the header (UTF-8, at most
 * 16,384 characters a piece in one reused buffer, so each piece is read before the next is made), the network. */
function* savePieces(json: string, track: Uint8Array): Generator<Uint8Array> {
  const length = utf8Length(json);
  const head = new Uint8Array(8);
  head.set(MAGIC_BYTES, 0);
  new DataView(head.buffer).setUint32(4, length, true);
  yield head;
  const enc = new TextEncoder(), buf = new Uint8Array(3 * JSON_PIECE);
  let written = 0;
  for (let i = 0; i < json.length;) {
    let j = Math.min(json.length, i + JSON_PIECE);
    if (j < json.length && (json.charCodeAt(j - 1) & 0xfc00) === 0xd800) j--; // a surrogate pair stays whole
    const n = enc.encodeInto(json.slice(i, j), buf).written;
    written += n;
    i = j;
    yield buf.subarray(0, n);
  }
  if (written !== length) throw new Error("The save's header changed length as it was written."); // i18n-ignore: internal
  if (track.length) yield track;
}
const JSON_PIECE = 16384;
const MAGIC_BYTES = new TextEncoder().encode(SAVE_MAGIC);
/** bytes `TextEncoder` makes of `s` (a lone surrogate becomes U+FFFD, 3 bytes) */
function utf8Length(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) n += 1;
    else if (c < 0x800) n += 2;
    else if ((c & 0xfc00) === 0xd800 && i + 1 < s.length && (s.charCodeAt(i + 1) & 0xfc00) === 0xdc00) { n += 4; i++; }
    else n += 3;
  }
  return n;
}

/** Parts as one buffer (what a slot keeps) */
export function joinParts(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  let n = 0;
  for (const p of parts) n += p.length;
  room(n);
  const out = new Uint8Array(n);
  n = 0;
  for (const p of parts) { out.set(p, n); n += p.length; }
  return out;
}

/** T-3395: the message key a damaged save's load throws (a gzip that does not unpack to its end, a header
 * cut short, a network that does not decode), apart from a file that is no save at all. */
export const SAVE_DAMAGED = "clockWorker.saveDamaged";
const errorText = (err: unknown) => String((err as Error | null)?.message ?? err);
/** T-3395: whether an error is `SAVE_DAMAGED`'s. */
export const isDamagedError = (err: unknown): boolean => errorText(err).includes(SAVE_DAMAGED);
/** T-3395: the start of the error a save that could not be checked throws (the codec worker did not load,
 * memory ran out): nothing is known about its bytes, so it is never called damaged; the cause follows. */
export const SAVE_UNCHECKED = "The save could not be read"; // i18n-ignore: internal, shown through loadFailKey
/** T-3395: whether an error is `SAVE_UNCHECKED`'s. */
export const isUncheckedError = (err: unknown): boolean => errorText(err).includes(SAVE_UNCHECKED);
const NOT_SAVE = "This file is not a trainsworld save.";
const NEWER_SAVE = "This save is from a newer version of the game.";

/** The compressed file -> header and network bytes; throws a player-readable reason (`SAVE_DAMAGED`, a key,
 * for a damaged save). */
export async function decodeSave(file: Uint8Array): Promise<{ header: SaveHeader; track: Uint8Array }> {
  let raw: Uint8Array;
  try {
    raw = await decompressGzip(file);
  } catch (err) {
    // T-3395: a gzip stream that does not unpack (Safari's cut-short saves) is a damaged save; a codec that
    // failed for another reason (its worker, memory) has not read the bytes
    if (!(file.length >= 2 && file[0] === 0x1f && file[1] === 0x8b)) throw new Error(NOT_SAVE);
    throw new Error(errorText(err).includes(GZIP_DAMAGED) ? SAVE_DAMAGED : `${SAVE_UNCHECKED}: ${errorText(err)}`);
  }
  if (raw.length < 8 || new TextDecoder().decode(raw.subarray(0, 4)) !== SAVE_MAGIC) throw new Error(NOT_SAVE);
  const n = new DataView(raw.buffer, raw.byteOffset).getUint32(4, true);
  if (8 + n > raw.length) throw new Error(SAVE_DAMAGED);
  let header: SaveHeader;
  try {
    header = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + n)));
  } catch (err) {
    throw new Error(err instanceof SyntaxError ? SAVE_DAMAGED : `${SAVE_UNCHECKED}: ${errorText(err)}`);
  }
  if (!header || typeof header !== "object" || header.game !== "anitabuilder" || typeof header.format !== "number") throw new Error(NOT_SAVE);
  if (header.format > SAVE_FORMAT) throw new Error(NEWER_SAVE);
  return { header, track: raw.slice(8 + n) };
}

/** T-3395: what a strict read of a save finds: "intact" (it unpacks to its end with its checksum, its
 * header reads and, where the track model is at hand, its network decodes), "damaged", "other" (no
 * save, or one from a newer or retired version: kept as it is, never called damaged), or "unchecked"
 * (the check itself failed: the codec worker did not load, memory ran out; nothing is marked, try later). */
export type SaveCheck = "intact" | "damaged" | "other" | "unchecked";
let networkCheck: ((track: Uint8Array) => boolean) | null = null;
/** T-3395: the clock worker's network decode (`TrackApi`). The main thread has no track model, so its check
 * is the gzip and the header, which is what a cut-short save fails. */
export function setNetworkCheck(f: (track: Uint8Array) => boolean) { networkCheck = f; }
export async function checkSave(bytes: Uint8Array): Promise<SaveCheck> {
  let track: Uint8Array;
  try {
    ({ track } = await decodeSave(bytes));
  } catch (err) {
    if (isDamagedError(err)) return "damaged";
    const why = errorText(err);
    return why === NOT_SAVE || why === NEWER_SAVE ? "other" : "unchecked";
  }
  if (!networkCheck || isRetiredTrackFormat(track)) return "intact";
  try {
    return networkCheck(track) ? "intact" : "damaged";
  } catch {
    return "unchecked"; // a trap in the decode (memory) says nothing about the bytes
  }
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
  /** T-3115: a copy of a shared network (`SaveHeader.copy`); absent otherwise */
  copy?: boolean;
}

/** One city's panel in a save's picture (T-894): its registry id ("" for an unnamed panel over the
 * whole network, or `WHOLE_PANEL` for the panel over everything built, T-2394), its stations, and
 * where it is, as fractions of the picture's width and height. */
export interface ThumbPanel { city: string; stations: number; x: number; y: number; w: number; h: number; }
export const WHOLE_PANEL = "*";

export interface StoredSave extends Partial<SlotSummary> {
  bytes: Uint8Array;
  /** T-3395: these bytes failed the check (a load, or the account refusing them); the next write of the
   * slot drops it */
  damaged?: true;
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

export interface SaveSlot extends Partial<SlotSummary> { key: string; name: string; at: number; day: number; /** T-2569: an account card whose picture is still being fetched */ loading?: boolean; /** T-988 */ cloud?: SyncRecord; /** T-2037 */ origin?: string; edit?: number; /** T-3395 */ damaged?: boolean; /** T-3610: the browser could not read the record at all, so only the key is known (also `damaged`) */ unreadable?: boolean; }

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
  /** T-3395: the slot's `at` whose bytes the account refused as damaged: not sent again until the slot changes */
  refused?: number;
  /** T-3788: the account rev that failed to load in place: not fetched again until the account has a newer one */
  badRev?: number;
}
export const ORIGINAL_SLOT = "autosave";
export const ORIGINAL_NAME = "Original game"; // i18n-ignore: a slot's saved name
export function slotName(key: string, save?: StoredSave | null): string { return save?.name || (key === ORIGINAL_SLOT ? ORIGINAL_NAME : "Saved game"); } // i18n-ignore: a slot's saved name

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

/** T-2703: a read that browser storage answered with an error, every try; the slot may hold a game, so
 * nothing may take it as empty. The message is the last error's, for the failure screen. */
export class StorageReadError extends Error {
  override name = "StorageReadError";
}
/** waits before each try after the first (T-2703) */
const READ_BACKOFF_MS = [100, 400, 1600];
/** reads to fail on purpose, for checks: slot reads (T-2703, `?debug=1&readFails=N`) and slot lists
 * (T-2705, `?debug=1&listFails=N`) */
const forced = { read: 0, list: 0 };
export function failReads(n: number) { forced.read = Math.max(0, Math.floor(n)); }
export function failLists(n: number) { forced.list = Math.max(0, Math.floor(n)); }

/** Storage this page may not use at all (cookies blocked, no IndexedDB): nothing was ever saved here,
 * and every write fails the same way. */
export function storageBlocked(err: unknown): boolean {
  return typeof indexedDB === "undefined" || (err as { name?: string } | null)?.name === "SecurityError";
}

/** T-2703: `read` tried again on a connection opened afresh after each error, the error thrown as a
 * StorageReadError when it persists; a blocked storage's error is thrown at once, as it is. */
async function retried<T>(what: string, kind: keyof typeof forced, read: () => Promise<T>): Promise<T> {
  for (let tries = 0; ; tries++) {
    try {
      if (forced[kind] > 0) { forced[kind]--; throw new DOMException("forced by ?debug=1", "UnknownError"); } // i18n-ignore
      return await read();
    } catch (err) {
      if (storageBlocked(err)) throw err;
      if (tries >= READ_BACKOFF_MS.length) {
        console.error(`T-2703: ${what} could not be read after ${tries + 1} tries:`, err); // i18n-ignore
        throw new StorageReadError(String(err));
      }
      db = null;
      await new Promise((go) => setTimeout(go, READ_BACKOFF_MS[tries]));
    }
  }
}

/** Slot `key` as stored, or null when nothing is there (or storage is blocked: play without autosave).
 * A read that keeps failing throws a StorageReadError (T-2703). */
export async function readStored(key: string): Promise<StoredSave | null> {
  try {
    return await retried(`slot ${key}`, "read", async () => ((await req((await store("readonly")).get(key))) as StoredSave | undefined) ?? null);
  } catch (err) {
    if (storageBlocked(err)) return null;
    throw err;
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

const UNREADABLE = Symbol("unreadable");
/** T-3610: `key`'s record read in a transaction of its own, so that one the browser cannot read (Chrome
 * 130+ NotFoundError when a big value's file is gone, Firefox UnknownError) aborts only itself; tried
 * twice, then UNREADABLE. A blocked storage's error is thrown. */
async function readApart<T>(key: string): Promise<T | undefined | typeof UNREADABLE> {
  for (let tries = 0; ; tries++) {
    try {
      return (await req((await store("readonly")).get(key))) as T | undefined;
    } catch (err) {
      if (storageBlocked(err)) throw err;
      if (tries) return UNREADABLE;
    }
  }
}

/** Metadata only leaves the worker; unreadable-save recovery copies are kept out of the list. Only
 * the slots are read, not the demand caches beside them (several MB each, T-328). Also read by the
 * start menu on the main thread (T-677). Throws a blocked storage's error as it is, and a
 * StorageReadError when the reads keep failing (T-2705). */
export async function listSlots(): Promise<SaveSlot[]> {
  const [keys, saves, syncs] = await retried("the slot list", "list", async () => { // i18n-ignore: console text
    const keys = (await req((await store("readonly")).getAllKeys())).filter((k): k is string => typeof k === "string" && (k === ORIGINAL_SLOT || k.startsWith("slot-")));
    const saves = await Promise.all(keys.map((k) => readApart<StoredSave>(k)));
    const syncs = await Promise.all(keys.map(async (k) => { const v = await readApart<SyncRecord>(SYNC_PREFIX + k); return v === UNREADABLE ? undefined : v; }));
    return [keys, saves, syncs] as const;
  });
  return keys.flatMap((key, i): SaveSlot[] => {
    const s = saves[i];
    if (!s) return [];
    // T-3610: a record the browser cannot read (its value's file is gone) is that one game, damaged; its
    // backups and account copy are still offered, and `at` 0 puts it last
    if (s === UNREADABLE) return [{ key, name: slotName(key), at: 0, day: 0, damaged: true, unreadable: true, cloud: syncs[i] }];
    if (syncs[i]) seeSync(key, syncs[i]);
    return [{ key, name: slotName(key, s), at: s.at, day: s.day, thumb: s.thumb, panels: s.panels, shade: s.shade, cities: s.cities, lines: s.lines, stations: s.stations, cash: s.cash, mode: s.mode, ...(s.copy ? { copy: true } : {}), ...(s.damaged ? { damaged: true } : {}), cloud: syncs[i], origin: s.origin, edit: s.edit }];
  }).sort((a, b) => b.at - a.at);
}

/** T-3611: a write refused because another tab saved the slot since this one last read it. */
export class SlotTakenError extends Error { constructor() { super("The slot was saved by another tab."); } } // i18n-ignore: internal, never shown

/** Save while preserving a slot's latest name (it may have been renamed elsewhere; `lastName`, the
 * name last seen, where the slot has gone, T-1829) and its origin and latest change when `v` has none
 * (T-2037); true if the slot was not there before. With `seenEdit`, a slot whose latest change is newer
 * is left as it is and a SlotTakenError thrown (T-3611). */
export async function writeSlot(key: string, v: StoredSave, lastName?: string, seenEdit?: number): Promise<boolean> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  const old = await req<StoredSave | undefined>(s.get(key));
  // T-3611: the slot holds a change this tab never read (another tab saved it): nothing is written
  if (seenEdit !== undefined && old && editOf(old) > seenEdit) throw new SlotTakenError();
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
/** The list changed by `f` in one transaction (T-2738: both threads add and drop entries at once, and a
 * read then a write apart lost one, leaving its game on the account). */
export async function changeCloudDeletes(f: (v: CloudDelete[]) => CloudDelete[]): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const v = f(((await req(s.get(CLOUD_DELETES))) as CloudDelete[] | undefined) ?? []);
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

const UNREADABLE_PREFIX = "unreadable-";
/** T-743, T-3395: where slot `slot`'s copy written at `at` is kept aside when it cannot be read (out of the
 * slot list, the tiers and their trimming); deleted with the slot. */
export const unreadableKey = (slot: string, at: number) => `${UNREADABLE_PREFIX}${at}:${slot}`;

/** T-3774: how many copies are kept aside as unreadable, across every slot (a count for feedback reports). */
export async function countUnreadable(): Promise<number> {
  const keys = await req((await store("readonly")).getAllKeys());
  return keys.filter((k) => typeof k === "string" && k.startsWith(UNREADABLE_PREFIX)).length;
}

export async function deleteSlot(key: string): Promise<void> {
  const s = await store("readwrite");
  const done = committed(s.transaction);
  await req(s.delete(key));
  await req(s.delete(SYNC_PREFIX + key));
  await req(s.delete(AHEAD_PREFIX + key));
  // and its demand cache (T-328), backups (T-744) and copies kept aside as unreadable (T-743, T-3395)
  for (const k of await req(s.getAllKeys())) if (typeof k === "string" && (k.startsWith(`${DEMAND_PREFIX}${key}:`) || k.startsWith(`${BACKUP_PREFIX}${key}:`) || (k.startsWith(UNREADABLE_PREFIX) && k.endsWith(":" + key)))) s.delete(k);
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
  return `Game ${n}`; // i18n-ignore: a slot's saved name
}

/** A new game's slot name (T-728): its start city, or "World" for a map start, then " sandbox" for
 * a sandbox, then " 2", " 3" and on if a slot already has the name. */
export function newGameName(slots: { name: string }[], spec: NewGameSpec): string {
  const start = registry.cities.find((c) => c.id === spec.start);
  const base = (start ? zoneName(start.id, start.name) : "World") + (spec.mode === "sandbox" ? " sandbox" : ""); // i18n-ignore: a slot's saved name
  const names = new Set(slots.map((s) => s.name));
  let name = base;
  for (let n = 2; names.has(name); n++) name = `${base} ${n}`;
  return name;
}

// T-744, T-1826: a slot's backups, the slot's record without its picture: four tiers
// `backup:<slot>:t0` to `:t3`, reaching back about 10 minutes, an hour, a day and a week, and
// `backup:<slot>:pull`, the slot as it was before the account's copy last replaced it. Backups from
// before T-1826 (`:0`, `:1`) are listed until the next tier write folds them in. T-3395: `backup:<slot>:good`,
// the newest copy known to pass the check, outside the tiers and `BACKUP_CAP`; a damaged copy never
// becomes a backup.
const BACKUP_PREFIX = "backup:";
const TIERS = ["t0", "t1", "t2", "t3"];
const GOOD = "good";
/** T-1826: wall ms. Tier 0 takes the slot at most once a step; the backup it held moves down to
 * tier i+1 only when it is at least step i+1 newer than the one there, else it goes. So tier 0 is up
 * to 10 min old, tier 1 up to about 1 h 10 min, tier 2 about a day, tier 3 about a week. */
export const BACKUP_STEPS = [10 * 60_000, 3600_000, 86400_000, 7 * 86400_000];
/** T-1826: the most a slot's tier backups hold, bytes; past it the oldest tiers are dropped (tier 0
 * is always kept). About 100 KB a save is usual. */
export const BACKUP_CAP = 8 << 20;
export interface SlotBackup { key: string; at: number; day: number; /** the slot before the account's copy replaced it */ pull?: boolean; /** T-3395: the newest copy that passed the check */ good?: boolean; /** T-2037 */ stations?: number; edit?: number }
const backupKey = (slot: string, tier: string) => `${BACKUP_PREFIX}${slot}:${tier}`;

export async function listBackups(slot: string): Promise<SlotBackup[]> {
  const keys = (await req((await store("readonly")).getAllKeys())).filter((k): k is string => typeof k === "string" && isBackupOf(k, slot));
  const out: SlotBackup[] = [];
  for (const key of keys) {
    const b = await readApart<StoredSave>(key); // one unreadable backup is left out (T-3610)
    if (b && b !== UNREADABLE) out.push({ key, at: b.at, day: b.day, stations: b.stations, edit: b.edit, ...(key.endsWith(":pull") ? { pull: true } : {}), ...(key.endsWith(":" + GOOD) ? { good: true } : {}) });
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

/** An autosave's backup, written when tier 0 is a step old (`planBackups`), in one transaction. `v` is a
 * copy this build just wrote (`compressGzip` has read back every byte), so it is also the slot's good copy
 * (T-3395) whenever tier 0 takes it or there is none. */
export async function writeBackup(slot: string, v: StoredSave, steps: number[] = BACKUP_STEPS): Promise<boolean> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const get = async (tier: string) => ((await req(s.get(backupKey(slot, tier)))) as StoredSave | undefined) ?? null;
    const { thumb: _, ...rest } = v;
    const hasGood = (await req(s.getKey(backupKey(slot, GOOD)))) !== undefined;
    let tiers = [await get("t0")];
    if (tiers[0] && v.at - tiers[0].at < steps[0]) {
      if (!hasGood) s.put(rest, backupKey(slot, GOOD));
      await done;
      return true;
    }
    for (const t of TIERS.slice(1)) tiers.push(await get(t));
    const legacy = [await get("0"), await get("1")].filter((b): b is StoredSave => !!b).sort((a, b) => b.at - a.at);
    if (legacy.length && tiers.every((t) => !t)) tiers = [legacy[0], legacy[1] ?? null, null, null];
    const plan = planBackups(tiers, rest, steps) ?? tiers;
    if (plan[0] === rest || !hasGood) s.put(rest, backupKey(slot, GOOD));
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
 * was, under `backup:<slot>:pull`. Nothing is written when the slot is empty or already holds `next`.
 * T-3395: the slot is checked first: a damaged one is kept aside (`unreadableKey`, as a failed load keeps
 * it, T-743), never as a backup; an intact one newer than the good copy is the good copy too; one that
 * could not be checked is the pull backup as before. */
export async function backupBeforeReplace(key: string, next: { at: number; bytes: Uint8Array }): Promise<boolean> {
  try {
    const old = await readStored(key);
    if (!old || (old.at === next.at && old.bytes.length === next.bytes.length)) return true;
    // checked before the transaction, which an await on anything but storage would end
    const check = old.damaged ? "damaged" : await checkSave(old.bytes);
    const { thumb: _, damaged: _d, ...rest } = old;
    const s = await store("readwrite");
    const done = committed(s.transaction);
    if (check === "damaged") s.put(old, unreadableKey(key, old.at));
    else {
      s.put(rest, backupKey(key, "pull"));
      const good = (await req(s.get(backupKey(key, GOOD)))) as StoredSave | undefined;
      if (check === "intact" && (!good || good.at < old.at)) s.put(rest, backupKey(key, GOOD));
    }
    await done;
    return true;
  } catch {
    return false;
  }
}

/** T-3395: slot `key` marked damaged, if it still holds the copy written at `at` (nothing else in it changes). */
export async function markDamaged(key: string, at: number): Promise<void> {
  try {
    const s = await store("readwrite");
    const done = committed(s.transaction);
    const old = (await req(s.get(key))) as StoredSave | undefined;
    if (old && old.at === at && !old.damaged) s.put({ ...old, damaged: true }, key);
    await done;
  } catch { /* marked at its next check */ }
}

/** T-3395: for a slot found damaged, its newest backup that passes the check kept as its good copy (when
 * there was none newer); true when the slot has a good copy after. */
export async function rescueGood(key: string): Promise<boolean> {
  try {
    for (const b of await listBackups(key)) {
      if (b.good) return true;
      const copy = await readStored(b.key);
      if (!copy || copy.damaged || await checkSave(copy.bytes) !== "intact") continue;
      const { damaged: _, ...rest } = copy;
      return await writeStored(backupKey(key, GOOD), rest);
    }
  } catch { /* storage unavailable */ }
  return false;
}

/** T-3395: slot `key` checked: a damaged one is marked so and its good copy found (`rescueGood`). */
export async function checkSlot(key: string): Promise<SaveCheck | null> {
  const stored = await readStored(key).catch(() => null);
  if (!stored) return null;
  const check = await checkSave(stored.bytes);
  if (check === "damaged") {
    await markDamaged(key, stored.at);
    await rescueGood(key);
  }
  return check;
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
  | { kind: "import"; bytes: Uint8Array; name: string; from: string; /** T-2037: an account backup's game */ origin?: string; edit?: number;
    /** T-3257: a network viewed from Community, held in no slot until saved under `name` */ view?: boolean }
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

/** T-3886: the schematic's saved choice when every city is unticked; an empty choice is the whole network */
export const NO_CITIES = "none";
