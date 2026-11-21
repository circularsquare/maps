// T-4004: the search box's towns and neighbourhoods, found off the main thread (game/placeSearch.ts
// asks), so no query can hold up the game (T-3997: matching on the main thread stopped trains at 100x).
// Two sources: the place lists of the zones near the view (T-400, as before: substrings and short
// queries), and the world name index's shard for the query (pipeline/place_index.py), which finds a
// town anywhere. One fetch at a time, view lists first, then the index's towns, then its
// neighbourhoods and hamlets, then the next nearest zones' lists; the best MAX_HITS are posted after
// each arrives. Lists and shards kept are capped, the farthest list and the oldest shard dropped.

import { matchScore, normalise, PLACE_WEIGHT } from "../game/nameMatch";
import { M_PER_DEG } from "../game/frame";
import { patientFetch } from "./patientFetch";
import type { Place } from "../game/placeList";
import type { PlaceFound, PlaceSearchIn, ZoneInfo } from "../game/placeSearch";

const MAX_HITS = 8;
/** Zones whose boxes lie within this of the view: their lists load first (placeList.ts loadPlacesAt). */
const VIEW_M = 30_000;
/** T-400, T-3997: while fewer than WANT places match, the next nearest of the WIDEN_MAX nearest zones within WIDEN_M. */
const WANT = 5;
const WIDEN_M = 200_000;
const WIDEN_MAX = 48;
/** At equal match, a place this near the view comes before farther ones. */
const NEAR_M = 200_000;
/** A short query's far exact match scores past every real match (matchScore's worst is 5). */
const SHORT_FAR_SCORE = 10;
const MAX_LISTS = 56;
const MAX_SHARDS = 24;
/** A failed fetch is tried again after this long. */
const RETRY_MS = 30_000;

interface Manifest { format: number; buckets: number; kinds: string[]; keyLen: number; wideKeyLen: number; joined?: boolean; deep: Record<string, string[]> }
type Row = [string, number, number, number, number, string, string[]?];
const TIERS = ["a", "b"] as const;

let base = "";
let zones: ZoneInfo[] = [];
let manifest: (Manifest & { deepSets: Record<string, Set<string>> }) | null | undefined;
const lists = new Map<string, Place[]>();
const shards = new Map<string, Place[]>();
const failed = new Map<string, number>();
let cur: { seq: number; q: string; lng: number; lat: number } | null = null;
let busy = false, scheduled = false, matched = 0, posted = "";

self.onmessage = (e: MessageEvent<PlaceSearchIn>) => {
  const m = e.data;
  if (m.type === "init") {
    base = m.base;
    zones = m.zones;
    cells = null;
    return;
  }
  cur = { seq: m.seq, q: normalise(m.q), lng: m.lng, lat: m.lat };
  schedule();
};

function schedule() {
  if (scheduled) return;
  scheduled = true;
  // after the messages already queued, so a burst of typing is matched once, for its last query
  setTimeout(() => { scheduled = false; step(); }, 0);
}

function step() {
  const c = cur;
  if (!c) return;
  const hits = c.q ? match(c) : [];
  const next = busy ? null : nextFetch(c);
  const done = !busy && !next;
  // each answer re-runs the main thread's search: only a changed one is sent
  const sig = `${c.seq}|${done}|${hits.map((h) => `${h.place.city}|${h.place.name}|${h.place.lng}|${h.place.lat}|${h.score}`).join(";")}`;
  if (c.q && sig !== posted) {
    posted = sig;
    self.postMessage({ seq: c.seq, hits, done });
  }
  if (!next) return;
  busy = true;
  next().catch(() => {}).finally(() => { busy = false; schedule(); });
}

const tried = (k: string) => (failed.get(k) ?? 0) > Date.now() - RETRY_MS;

function boxDistance(z: ZoneInfo, lng: number, lat: number): number {
  const [w, s, e, n] = z.box, kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180);
  return Math.hypot(Math.max(0, w - lng, lng - e) * kx, Math.max(0, s - lat, lat - n) * M_PER_DEG);
}

function nextFetch(c: NonNullable<typeof cur>): (() => Promise<void>) | null {
  const byDist = zones.map((z) => ({ z, d: boxDistance(z, c.lng, c.lat) })).sort((a, b) => a.d - b.d);
  const view = byDist.filter((x) => x.d <= VIEW_M);
  for (const { z } of view.length ? view : byDist.slice(0, 1))
    if (!lists.has(z.id) && !tried("l" + z.id)) return () => loadList(z.id, c);
  if (manifest === undefined && !tried("m")) return loadManifest;
  if (!c.q) return null;
  for (const t of TIERS) {
    const b = bucketOf(c.q, t);
    if (b !== null && !shards.has(t + b) && !tried("s" + t + b)) return () => loadShard(t, b);
  }
  if (c.q.length < 3 || matched >= WANT) return null;
  for (const { z } of byDist.filter((x) => x.d <= WIDEN_M).slice(0, WIDEN_MAX))
    if (!lists.has(z.id) && !tried("l" + z.id)) return () => loadList(z.id, c);
  return null;
}

async function json<T>(path: string, key: string): Promise<T> {
  try {
    const r = await patientFetch(base + path);
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    return (await r.json()) as T;
  } catch (e) {
    failed.set(key, Date.now());
    throw e;
  }
}

const toPlace = (name: string, kind: string, lng: number, lat: number, pop: number, city: string, others?: string[]): Place => ({
  name, kind, lng, lat, pop, city, keys: [normalise(name), ...(others ?? []).map(normalise)], aka: others ?? [],
});

async function loadList(id: string, c: NonNullable<typeof cur>) {
  const f = await json<{ kinds: string[]; places: [string, number, number, number, number, string[]?][] }>(`${id}.places.json`, "l" + id);
  lists.set(id, f.places.map(([name, k, lng, lat, pop, others]) => toPlace(name, f.kinds[k] ?? "neighbourhood", lng, lat, pop, id, others)));
  if (lists.size <= MAX_LISTS) return;
  // the farthest from the view goes
  const at = new Map(zones.map((z) => [z.id, z]));
  const far = [...lists.keys()].map((k) => ({ k, d: at.has(k) ? boxDistance(at.get(k)!, c.lng, c.lat) : Infinity })).sort((a, b) => b.d - a.d);
  for (const { k } of far.slice(0, lists.size - MAX_LISTS)) lists.delete(k);
}

async function loadManifest() {
  try {
    const m = await json<Manifest>("place-index/index.json", "m");
    manifest = { ...m, deepSets: Object.fromEntries(Object.entries(m.deep).map(([t, ks]) => [t, new Set(ks)])) };
  } catch {
    // none (an older build, or offline): towns are found in the near lists only, and it is asked for again later
  }
}

async function loadShard(t: string, b: number) {
  const f = await json<{ rows: Row[] }>(`place-index/${t}/${b}.json`, "s" + t + b);
  const kinds = manifest!.kinds;
  shards.set(t + b, f.rows.map(([name, k, lng, lat, pop, zone, others]) => toPlace(name, kinds[k] ?? "neighbourhood", lng, lat, pop, zone, others)));
  while (shards.size > MAX_SHARDS) shards.delete(shards.keys().next().value!);
}

function fnv(key: string): number {
  let h = 2166136261;
  for (const ch of key) h = Math.imul(h ^ ch.codePointAt(0)!, 16777619) >>> 0;
  return h;
}

/** A word's key in tier `t`, ranked: 2 keyed to its end (a key with too many rows is split a letter
 * further), 1 a deep key whose shard keeps only its best rows, 0 a short word kept whole. */
function keyOf(w: string, deep: Set<string>, m: Manifest): { key: string; rank: number; len: number } {
  const cps = Array.from(w);
  const K = cps[0].codePointAt(0)! >= 0x2e80 ? m.wideKeyLen : m.keyLen;
  if (cps.length < K) return { key: w, rank: 0, len: cps.length };
  let n = K;
  while (n < cps.length && deep.has(cps.slice(0, n).join(""))) n++;
  const key = cps.slice(0, n).join("");
  return { key, rank: deep.has(key) ? 1 : 2, len: cps.length };
}

/** The shard of tier `t` holding the query's most telling word, longest first among equal ranks; the
 * words joined ("santa fe" as "santafe") when that ranks as well. Null when there is no index. */
function bucketOf(q: string, t: string): number | null {
  if (!manifest) return null;
  const deep = manifest.deepSets[t] ?? new Set<string>();
  const ws = q.split(" ").filter(Boolean);
  let best: { key: string; rank: number; len: number } | null = null;
  for (const w of ws) {
    const k = keyOf(w, deep, manifest);
    if (!best || k.rank > best.rank || (k.rank === best.rank && k.len > best.len)) best = k;
  }
  if (best && manifest.joined && ws.length > 1) {
    const k = keyOf(ws.join(""), deep, manifest);
    if (k.rank >= best.rank) best = k;
  }
  return best ? fnv(best.key) % manifest.buckets : null;
}

function match(c: NonNullable<typeof cur>): PlaceFound[] {
  const kx = M_PER_DEG * Math.cos((c.lat * Math.PI) / 180);
  const found: (PlaceFound & { weight: number })[] = [];
  const letters = Array.from(c.q.replace(/ /g, ""));
  // one or two letters (one CJK character): a far place only by its whole name, after every near one
  const short = letters.length < (letters[0].codePointAt(0)! >= 0x2e80 ? 2 : 3);
  const add = (p: Place) => {
    let score = -1;
    for (const k of p.keys) {
      const s = matchScore(k, c.q);
      if (s >= 0 && (score < 0 || s < score)) score = s;
    }
    if (score < 0 || isRegistryCity(p)) return;
    const near = Math.hypot((p.lng - c.lng) * kx, (p.lat - c.lat) * M_PER_DEG) <= NEAR_M;
    if (short && !near) {
      if (score > 0) return;
      score = SHORT_FAR_SCORE;
    }
    found.push({ place: p, score, near, weight: (PLACE_WEIGHT[p.kind] ?? 0) + p.pop });
  };
  for (const list of lists.values()) for (const p of list) add(p);
  if (manifest)
    for (const t of TIERS) {
      const b = bucketOf(c.q, t);
      // a zone whose list is in has every one of its places matched already
      for (const p of (b === null ? undefined : shards.get(t + b)) ?? []) if (!lists.has(p.city)) add(p);
    }
  matched = found.length;
  found.sort((a, b) => a.score - b.score || Number(b.near) - Number(a.near) || b.weight - a.weight || (a.place.name < b.place.name ? -1 : 1));
  return found.slice(0, MAX_HITS).map(({ place, score, near }) => ({ place, score, near }));
}

// ---- a place list's own entry for a registry city (search.ts used to hold this, T-3997)

/** How near a registry city's centre a place list's city or town is taken for it, degrees. */
const SAME_CITY_DEG = 0.15;
let cells: Map<string, { name: string; lng: number; lat: number }[]> | null = null;
const isCity = new WeakMap<Place, boolean>();

/** A place list's own entry for a registry city ("Washington" near Washington, DC's centre), which the
 * city's own result already covers. Kept per place: it does not depend on the query. */
function isRegistryCity(p: Place): boolean {
  if (p.kind !== "city" && p.kind !== "town") return false;
  let known = isCity.get(p);
  if (known !== undefined) return known;
  if (!cells) {
    cells = new Map();
    for (const z of zones) {
      const k = `${Math.floor(z.center[0])}:${Math.floor(z.center[1])}`;
      (cells.get(k) ?? cells.set(k, []).get(k)!).push({ name: normalise(z.name), lng: z.center[0], lat: z.center[1] });
    }
  }
  const name = p.keys[0], kx = Math.cos((p.lat * Math.PI) / 180), x = Math.floor(p.lng), y = Math.floor(p.lat);
  known = false;
  for (let i = -1; i <= 1 && !known; i++)
    for (let j = -1; j <= 1 && !known; j++)
      for (const z of cells.get(`${x + i}:${y + j}`) ?? [])
        if (z.name.startsWith(name) && Math.hypot((p.lng - z.lng) * kx, p.lat - z.lat) < SAME_CITY_DEG) { known = true; break; }
  isCity.set(p, known);
  return known;
}
