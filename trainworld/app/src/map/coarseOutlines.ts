// T-800: every registered zone's coarse outline, `packs/outlines.json` (pipeline/zone_outlines.py),
// so the map has an outline everywhere before any zone's own boundary has been fetched. Read once:
// the outlines and the realms' shade (T-1006) share it.
import type { MultiPolygon } from "geojson";
import { dataUrl } from "../game/dataBase";
import { setOutlineLookup, zonesTouching, ZONE_REACH_M, type CityId } from "../game/cities";
import { M_PER_DEG } from "../game/frame";
import { atZero, cellSays, distanceTo, type Footprint } from "../workers/cityReach";

/** `water` (T-1145): each coastal zone's sea, a little into its own land, for the focused zone's shade */
export type OutlineFile = { q: number; zones: Record<string, number[][]>; water?: Record<string, number[][]> };

let read: Promise<{ zones: Map<string, MultiPolygon>; water: Map<string, MultiPolygon> }> | null = null;
let text: string | null = null;

/** The coarse footprint of each zone in the file, by zone id; empty when the file is unavailable. */
export function coarseOutlines(): Promise<Map<string, MultiPolygon>> {
  return (read ??= load()).then((r) => r.zones);
}

/** The sea each coastal zone's coarse outline takes (T-1145), by zone id. */
export function coarseWater(): Promise<Map<string, MultiPolygon>> {
  return (read ??= load()).then((r) => r.water);
}

/** The file as fetched, once (the shade's worker reads its own copy, T-1026); null after that, or
 * before the file is in. */
export function takeOutlineText(): string | null {
  const t = text;
  text = null;
  return t;
}

/** One ring of the file as lon, lat pairs, flat and open. */
export function decodeRing(flat: number[], q: number): Float64Array {
  const out = new Float64Array(flat.length);
  let x = 0, y = 0;
  for (let i = 0; i < flat.length; i += 2) { out[i] = (x += flat[i]) / q; out[i + 1] = (y += flat[i + 1]) / q; }
  return out;
}

async function load() {
  const out = { zones: new Map<string, MultiPolygon>(), water: new Map<string, MultiPolygon>() };
  try {
    const r = await fetch(dataUrl("outlines.json"));
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const body = await r.text();
    const { q, zones, water } = JSON.parse(body) as OutlineFile;
    const polygons = (rings: number[][]): MultiPolygon => ({
      type: "MultiPolygon",
      coordinates: rings.map((flat) => {
        const d = decodeRing(flat, q), ring: number[][] = [];
        for (let i = 0; i < d.length; i += 2) ring.push([d[i], d[i + 1]]);
        ring.push([...ring[0]]);
        return [ring];
      }),
    });
    for (const id in zones) out.zones.set(id, polygons(zones[id]));
    for (const id in water ?? {}) out.water.set(id, polygons(water![id]));
    text = body;
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

/** The clock worker's copy (T-1265): only the zones' rings, kept as the file has them and decoded
 * per zone when a place near it is first asked about. */
export async function readOutlinesForPlaces(url: string): Promise<void> {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const { q, zones } = (await r.json()) as OutlineFile;
    placeBy((id) => zones[id]?.map((flat) => decodeRing(flat, q)));
  } catch (error) {
    console.warn("Coarse zone outlines unavailable", error);
  }
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
