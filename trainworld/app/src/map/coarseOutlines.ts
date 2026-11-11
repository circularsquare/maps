// T-800: every registered zone's coarse outline, `packs/outlines.json` (pipeline/zone_outlines.py),
// so the map has an outline everywhere before any zone's own boundary has been fetched. Read once, by the
// page: the outlines, the realms' shade (T-1006) and the clock worker's place names (T-1265) share it,
// the workers as `PackedOutlines` (T-2802).
import type { MultiPolygon } from "geojson";
import { dataUrl } from "../game/dataBase";
import { setOutlineLookup, zonesTouching, ZONE_REACH_M, type CityId } from "../game/cities";
import { M_PER_DEG } from "../game/frame";
import { atZero, cellSays, cellsSayInside, distanceTo, type Footprint } from "../workers/cityReach";
import { smallDevice } from "../workers/deviceMemory";

/** `water` (T-1145): each coastal zone's sea, a little into its own land, for the focused zone's shade */
export type OutlineFile = { q: number; zones: Record<string, number[][]>; water?: Record<string, number[][]> };

type Outlines = ReadonlyMap<string, MultiPolygon>;
let read: Promise<{ zones: Outlines; water: Outlines; packed: PackedOutlines | null }> | null = null;

/** T-2802: the file's zones as typed arrays, which a worker takes as a copy in place of fetching and parsing the
 * file itself (an 11 MB string and about 35 MB of arrays a context): zone `k` (`ids[k]`) has rings `zoneRing[k]` to
 * `zoneRing[k + 1]`, and ring `r` is `data[ringAt[r]]` to `data[ringAt[r + 1]]`, the file's deltas. */
export type PackedOutlines = { q: number; ids: string[]; zoneRing: Uint32Array; ringAt: Uint32Array; data: Int32Array };

function pack(from: Record<string, number[][]> | undefined, q: number): PackedOutlines {
  const ids = Object.keys(from ?? {});
  let rings = 0, n = 0;
  for (const id of ids) for (const r of from![id]) { rings++; n += r.length; }
  const zoneRing = new Uint32Array(ids.length + 1), ringAt = new Uint32Array(rings + 1), data = new Int32Array(n);
  let ri = 0, at = 0;
  ids.forEach((id, k) => {
    zoneRing[k] = ri;
    for (const r of from![id]) { ringAt[ri++] = at; data.set(r, at); at += r.length; }
  });
  zoneRing[ids.length] = ri;
  ringAt[rings] = at;
  return { q, ids, zoneRing, ringAt, data };
}

/** Each zone's rings in `p` as views on its data, by zone id. */
export function packedRings(p: PackedOutlines): Map<string, Int32Array[]> {
  const out = new Map<string, Int32Array[]>();
  p.ids.forEach((id, k) => {
    const rs: Int32Array[] = [];
    for (let r = p.zoneRing[k]; r < p.zoneRing[k + 1]; r++) rs.push(p.data.subarray(p.ringAt[r], p.ringAt[r + 1]));
    out.set(id, rs);
  });
  return out;
}

/** T-2802: the zones' outlines for a worker (the clock's place names, the realm shade); null when the file is
 * unavailable. Posting it copies its arrays, about 12 MB. */
export function packedOutlines(): Promise<PackedOutlines | null> {
  return (read ??= load()).then((r) => r.packed);
}

let packedIn: PackedOutlines | null = null;
/** `packedOutlines` once the file is in, else null */
export const packedOutlinesNow = () => packedIn;

/** The coarse footprint of each zone in the file, by zone id; empty when the file is unavailable. */
export function coarseOutlines(): Promise<Outlines> {
  return (read ??= load()).then((r) => r.zones);
}

/** The sea each coastal zone's coarse outline takes (T-1145), by zone id. */
export function coarseWater(): Promise<Outlines> {
  return (read ??= load()).then((r) => r.water);
}

/** One zone's rings of the file as a MultiPolygon, each ring closed. */
function polygons(rings: ArrayLike<number>[], q: number): MultiPolygon {
  return {
    type: "MultiPolygon",
    coordinates: rings.map((flat) => {
      const d = decodeRing(flat, q), ring: number[][] = [];
      for (let i = 0; i < d.length; i += 2) ring.push([d[i], d[i + 1]]);
      ring.push([...ring[0]]);
      return [ring];
    }),
  };
}

/** T-2794: the outlines on a small device. Every zone's rings stay as the file's deltas in typed arrays (about
 * 8 MB in all) and become a MultiPolygon when asked for, the last few kept, where the decoded polygons of the
 * whole world take over 50 MB of a phone's page. */
class CompactOutlines implements Outlines {
  private recent = new Map<string, MultiPolygon>();
  constructor(private rings: Map<string, Int32Array[]>, private q: number) {}
  get size() { return this.rings.size; }
  has(id: string) { return this.rings.has(id); }
  get(id: string) {
    let g = this.recent.get(id);
    if (g) {
      this.recent.delete(id);
      this.recent.set(id, g);
      return g;
    }
    const r = this.rings.get(id);
    if (!r) return undefined;
    g = polygons(r, this.q);
    this.recent.set(id, g);
    if (this.recent.size > 16) this.recent.delete(this.recent.keys().next().value as string);
    return g;
  }
  keys() { return this.rings.keys(); }
  *values(): MapIterator<MultiPolygon> { for (const r of this.rings.values()) yield polygons(r, this.q); }
  *entries(): MapIterator<[string, MultiPolygon]> { for (const [id, r] of this.rings) yield [id, polygons(r, this.q)]; }
  [Symbol.iterator]() { return this.entries(); }
  forEach(f: (g: MultiPolygon, id: string, m: Outlines) => void) { for (const [id, g] of this.entries()) f(g, id, this); }
}

const compact = (p: PackedOutlines) => new CompactOutlines(packedRings(p), p.q);

/** One ring of the file as lon, lat pairs, flat and open. */
export function decodeRing(flat: ArrayLike<number>, q: number): Float64Array {
  const out = new Float64Array(flat.length);
  let x = 0, y = 0;
  for (let i = 0; i < flat.length; i += 2) { out[i] = (x += flat[i]) / q; out[i + 1] = (y += flat[i + 1]) / q; }
  return out;
}

/** T-2478: the waits before each new try of the file (a server mid-rebuild answers 404 for a minute or
 * two); after the last the page goes on without it */
const RETRY_MS = [1_000, 4_000, 10_000, 30_000, 60_000];

async function fetchText(url: string): Promise<string> {
  for (let k = 0; ; k++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      return await r.text();
    } catch (error) {
      if (k >= RETRY_MS.length) throw error;
      await new Promise((r) => setTimeout(r, RETRY_MS[k]));
    }
  }
}

async function load(): Promise<{ zones: Outlines; water: Outlines; packed: PackedOutlines | null }> {
  const out: { zones: Outlines; water: Outlines; packed: PackedOutlines | null } = { zones: new Map(), water: new Map(), packed: null };
  try {
    const { q, zones, water } = JSON.parse(await fetchText(dataUrl("outlines.json"))) as OutlineFile;
    out.packed = packedIn = pack(zones, q);
    if (smallDevice()) {
      out.zones = compact(out.packed);
      out.water = compact(pack(water, q));
    } else {
      out.zones = new Map(Object.entries(zones).map(([id, rings]) => [id, polygons(rings, q)]));
      out.water = new Map(Object.entries(water ?? {}).map(([id, rings]) => [id, polygons(rings, q)]));
    }
    placeBy((id) => out.zones.get(id)?.coordinates.flatMap((p) => p.map((r) => Float64Array.from(r.flat()))));
  } catch (error) {
    console.warn("Coarse zone outlines unavailable", error);
  }
  return out;
}

/** T-1265: name places by the outlines (game/cities.ts `cityWithin`): the zone whose outline holds the
 * place, else the one whose outline is nearest within `ZONE_REACH_M`, else null. `rings` gives a
 * zone's rings as flat lon, lat pairs; each becomes a footprint when a place near it is first asked. */
function placeBy(rings: (id: string) => Float64Array[] | undefined) {
  const fps = new Map<string, Footprint | null>();
  const fp = (id: CityId): Footprint | null => {
    let f = fps.get(id);
    if (f !== undefined) return f;
    const rs = rings(id);
    f = null;
    if (rs?.length) {
      const bbox: Footprint["bbox"] = [Infinity, Infinity, -Infinity, -Infinity];
      for (const r of rs) for (let k = 0; k < r.length; k += 2) {
        bbox[0] = Math.min(bbox[0], r[k]); bbox[1] = Math.min(bbox[1], r[k + 1]);
        bbox[2] = Math.max(bbox[2], r[k]); bbox[3] = Math.max(bbox[3], r[k + 1]);
      }
      f = { id, rings: rs, bbox };
    }
    fps.set(id, f);
    return f;
  };
  setOutlineLookup((lon, lat) => {
    let best: CityId | null = null, bd = ZONE_REACH_M;
    const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180);
    const near = zonesTouching(lon, lat, lon, lat, ZONE_REACH_M);
    // T-1348: the first zone at distance 0 wins, so look for it before measuring any distance
    for (const c of near) {
      const f = fp(c.id);
      if (f && atZero(f, lon, lat)) return c.id;
    }
    for (const c of near) {
      const f = fp(c.id);
      if (!f) continue;
      const [w, s, e, n] = f.bbox;
      if (Math.hypot(Math.max(0, w - lon, lon - e) * kx, Math.max(0, s - lat, lat - n) * M_PER_DEG) > bd) continue;
      const d = distanceTo(f, lon, lat);
      if (d === 0) return c.id;
      if (d <= bd) (bd = d), (best = c.id);
    }
    return best;
  });
}

/** The clock worker's copy (T-1265, T-2802): the page's packed rings, each zone decoded when a place near it is
 * first asked about. */
export function readOutlinesForPlaces(p: PackedOutlines): void {
  const rings = packedRings(p);
  placeBy((id) => rings.get(id)?.map((flat) => decodeRing(flat, p.q)));
}

/** An outline's edges by band of latitude (x0, y0, x1, y1 each), so the crossing test reads only the
 * edges of the point's band: a whole network's points against prefecture outlines took 0.6 s. */
interface Bands { s: number; h: number; nb: number; start: Uint32Array; seg: Float64Array }
const bands = new Map<string, Bands>();

function bandsOf(id: string, g: MultiPolygon): Bands {
  let b = bands.get(id);
  if (b) return b;
  let s = Infinity, n = -Infinity, edges = 0;
  for (const poly of g.coordinates) for (const r of poly) { edges += r.length; for (const [, y] of r) { s = Math.min(s, y); n = Math.max(n, y); } }
  // about four edges a band (T-1348), at least 64
  const BANDS = Math.max(64, Math.min(4096, Math.ceil(edges / 4)));
  const h = (n - s) / BANDS || 1;
  const band = (y: number) => Math.min(BANDS - 1, Math.max(0, Math.floor((y - s) / h)));
  const lists: number[][] = Array.from({ length: BANDS }, () => []);
  for (const poly of g.coordinates) for (const r of poly) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      for (let q = band(Math.min(yi, yj)); q <= band(Math.max(yi, yj)); q++) lists[q].push(xi, yi, xj, yj);
    }
  }
  const start = new Uint32Array(BANDS + 1);
  lists.forEach((l, q) => { start[q + 1] = start[q] + l.length; });
  const seg = new Float64Array(start[BANDS]);
  lists.forEach((l, q) => seg.set(l, start[q]));
  bands.set(id, (b = { s, h, nb: BANDS, start, seg }));
  return b;
}

const cellFootprints = new Map<string, Footprint>();
function cellFootprint(id: string, g: MultiPolygon): Footprint {
  let f = cellFootprints.get(id);
  if (f) return f;
  const bbox: Footprint["bbox"] = [Infinity, Infinity, -Infinity, -Infinity];
  const rings = g.coordinates.flatMap((p) => p.map((r) => {
    for (const [x, y] of r) { bbox[0] = Math.min(bbox[0], x); bbox[1] = Math.min(bbox[1], y); bbox[2] = Math.max(bbox[2], x); bbox[3] = Math.max(bbox[3], y); }
    return Float64Array.from(r.flat());
  }));
  cellFootprints.set(id, (f = { id, rings, bbox }));
  return f;
}

/** Whether (lon, lat) is inside zone `id`'s coarse outline `g` (even-odd), reading only the edges of
 * its band of latitude. */
export function inOutline(id: string, g: MultiPolygon, lon: number, lat: number): boolean {
  // T-1348: most places are in a cell of the outline's grid that no edge meets
  const said = cellSays(cellFootprint(id, g), lon, lat);
  if (said !== null) return said;
  const b = bandsOf(id, g), q = Math.floor((lat - b.s) / b.h);
  if (q < 0 || q >= b.nb) return false;
  let inside = false;
  for (let k = b.start[q], end = b.start[q + 1]; k < end; k += 4) {
    const xi = b.seg[k], yi = b.seg[k + 1], xj = b.seg[k + 2], yj = b.seg[k + 3];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** True when `inOutline` is surely true at every place in the box [w, s, e, n]: the box lies in
 * cells of the outline's grid wholly inside it (false says nothing). */
export function inOutlineBox(id: string, g: MultiPolygon, w: number, s: number, e: number, n: number): boolean {
  return cellsSayInside(cellFootprint(id, g), w, s, e, n);
}
