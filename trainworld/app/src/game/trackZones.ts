// T-1047: the zones the player's track runs through (docs/zones.md 6, "Track"). A zone that built
// track crosses is in play as one a station reaches is, for what the player sees: it joins the
// game's opened list (a bubble zoomed out, not a dot), its realm is shown, and demand opens it bare
// for its figures (workers/demandClient.ts `openCrossed`). Blueprint track shows its realms only, as
// a blueprint station does. A zone is crossed when a point of the track's drawn line, every STEP_M,
// lies in its coarse outline (`packs/outlines.json`, the realms' too).

import { signal } from "@preact/signals";
import type { MultiPolygon } from "geojson";
import type { WorldView } from "./types";
import { zonesTouching, type CityId } from "./cities";
import { addOpened } from "./openedZones";
import { coarseOutlines } from "../map/coarseOutlines";
import { anchor, toLngLat } from "../map/geo";
import type { NetworkBuffers } from "../map/network";

/** ground metres between the points tested along the track */
const STEP_M = 1000;
const RAD = Math.PI / 180;

export interface TrackZones {
  /** crossed by built track */
  built: ReadonlySet<CityId>;
  /** crossed by blueprint track that is not switched off */
  planned: ReadonlySet<CityId>;
}
export const trackZones = signal<TrackZones>({ built: new Set(), planned: new Set() });

let outlines: Map<string, MultiPolygon> | null = null;
let waiting: { net: NetworkBuffers; w: WorldView } | null = null;
/** per edge id: the shape its zones were worked out for, and the zones */
const byEdge = new Map<number, { sig: string; zones: CityId[] }>();

/** A new network state (not a train count alone): work out the zones its track crosses. */
export function followTrack(net: NetworkBuffers | null, w: WorldView | null) {
  if (!net || !w) return;
  if (!outlines) {
    if (!waiting) void coarseOutlines().then((m) => { outlines = m; const q = waiting; waiting = null; if (q) followTrack(q.net, q.w); });
    waiting = { net, w };
    return;
  }
  const a = net.anchor ?? anchor();
  const built = new Set<CityId>(), planned = new Set<CityId>(), seen = new Set<number>();
  for (const e of w.edges) {
    if (!e.built && e.off) continue;
    const pts = net.edgePts.get(e.id), na = w.nodes.get(e.a), nb = w.nodes.get(e.b);
    if (!pts || pts.length < 3 || !na || !nb) continue;
    seen.add(e.id);
    // (the drawn line's own points move with the draw anchor)
    const sig = `${pts.length}:${e.lengthM.toFixed(1)}:${na.lng},${na.lat}:${nb.lng},${nb.lat}`;
    let k = byEdge.get(e.id);
    if (k?.sig !== sig) byEdge.set(e.id, (k = { sig, zones: crossed(pts, a) }));
    for (const z of k.zones) (e.built ? built : planned).add(z);
  }
  for (const id of byEdge.keys()) if (!seen.has(id)) byEdge.delete(id);
  for (const z of built) planned.delete(z);
  const prev = trackZones.peek();
  if (!same(prev.built, built) || !same(prev.planned, planned)) trackZones.value = { built, planned };
  // every state, since the game in hand may have changed under an unchanged set
  addOpened(built);
}

const same = (a: ReadonlySet<CityId>, b: ReadonlySet<CityId>) => a.size === b.size && [...a].every((z) => b.has(z));

/** The zones whose outlines hold some point of one edge's drawn line (x, y, z triples, local units). */
function crossed(pts: Float32Array, a: ReturnType<typeof anchor>): CityId[] {
  const out = new Set<CityId>();
  let last: CityId | null = null;
  const test = (lon: number, lat: number) => {
    if (last && inside(last, outlines!.get(last)!, lon, lat)) return;
    last = null;
    for (const c of zonesTouching(lon, lat, lon, lat)) {
      const g = outlines!.get(c.id);
      if (g && inside(c.id, g, lon, lat)) { last = c.id; out.add(c.id); return; }
    }
  };
  // A drawn line holds a point every few metres on curves: test one every STEP_M along it, measured
  // in local units (a ground metre is 1 / cos(lat) of them, lat from the last point tested)
  let cos = 1, px = pts[0], py = pts[1], since = 0;
  const at = (x: number, y: number) => { const [lon, lat] = toLngLat(x, y, a); test(lon, lat); cos = Math.cos(lat * RAD); };
  at(px, py);
  for (let k = 3; k + 2 < pts.length; k += 3) {
    const qx = pts[k], qy = pts[k + 1], len = Math.hypot(qx - px, qy - py) * cos;
    for (let s = STEP_M - since; s < len; s += STEP_M) at(px + ((qx - px) * s) / len, py + ((qy - py) * s) / len);
    since = (since + len) % STEP_M;
    px = qx; py = qy;
  }
  at(px, py);
  return [...out];
}

/** An outline's edges by band of latitude (x0, y0, x1, y1 each), so the crossing test reads only the
 * edges of the point's band: a whole network's points against prefecture outlines took 0.6 s. */
interface Bands { s: number; h: number; start: Uint32Array; seg: Float64Array }
const BANDS = 64;
const bands = new Map<string, Bands>();

function bandsOf(id: CityId, g: MultiPolygon): Bands {
  let b = bands.get(id);
  if (b) return b;
  let s = Infinity, n = -Infinity;
  for (const poly of g.coordinates) for (const r of poly) for (const [, y] of r) { s = Math.min(s, y); n = Math.max(n, y); }
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
  bands.set(id, (b = { s, h, start, seg }));
  return b;
}

function inside(id: CityId, g: MultiPolygon, lon: number, lat: number): boolean {
  const b = bandsOf(id, g), q = Math.floor((lat - b.s) / b.h);
  if (q < 0 || q >= BANDS) return false;
  let inside = false;
  for (let k = b.start[q], end = b.start[q + 1]; k < end; k += 4) {
    const xi = b.seg[k], yi = b.seg[k + 1], xj = b.seg[k + 2], yj = b.seg[k + 3];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
