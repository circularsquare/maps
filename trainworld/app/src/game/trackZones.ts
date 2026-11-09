// T-1047: the zones the player's track runs through (docs/zones.md 6, "Track"). A zone that built
// track crosses is in play as one a station reaches is, for what the player sees: it joins the
// game's opened list (a bubble zoomed out, not a dot), its realm is shown, and demand opens it bare
// for its figures (workers/demandClient.ts `openCrossed`). Blueprint track shows its realms only, as
// a blueprint station does. A zone is crossed when a point of the track's drawn line, every STEP_M,
// lies in its coarse outline (`packs/outlines.json`, the realms' too).

import { signal } from "@preact/signals";
import type { MultiPolygon } from "geojson";
import type { WorldView } from "./types";
import { cityById, zonesTouching, zoneBox, type CityId } from "./cities";
import { addOpened } from "./openedZones";
import { coarseOutlines, inOutline, inOutlineBox } from "../map/coarseOutlines";
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

let outlines: ReadonlyMap<string, MultiPolygon> | null = null;
let waiting: { net: NetworkBuffers; w: WorldView } | null = null;
/** per edge id: the shape its zones were worked out for (points, length, which counts the same to
 * 0.1 m, end positions), and the zones */
type EdgeZones = { n: number; lengthM: number; alng: number; alat: number; blng: number; blat: number; zones: CityId[] };
let byEdge = new Map<number, EdgeZones>();
const NONE: CityId[] = [];

/** A new network state (not a train count alone): work out the zones its track crosses. Not put off
 * past the paint: the zones it opens feed demand's rounds (T-1357). */
export function followTrack(net: NetworkBuffers | null, w: WorldView | null) {
  if (!net || !w) return;
  if (!outlines) {
    if (!waiting) void coarseOutlines().then((m) => { outlines = m; const q = waiting; waiting = null; if (q) followTrack(q.net, q.w); });
    waiting = { net, w };
    return;
  }
  const a = net.anchor ?? anchor();
  const next = new Map<number, EdgeZones>(), kept: EdgeZones[] = [], keptBuilt: boolean[] = [], todo: Float32Array[] = [], todoAt: number[] = [];
  for (let i = 0, edges = w.edges; i < edges.length; i++) {
    const e = edges[i];
    if (!e.built && e.off) continue;
    const pts = net.edgePts.get(e.id), na = w.nodes.get(e.a), nb = w.nodes.get(e.b);
    if (!pts || pts.length < 3 || !na || !nb) continue;
    // (the drawn line's own points move with the draw anchor)
    let k = byEdge.get(e.id);
    if (!k || k.n !== pts.length || (k.lengthM !== e.lengthM && k.lengthM.toFixed(1) !== e.lengthM.toFixed(1)) || k.alng !== na.lng || k.alat !== na.lat || k.blng !== nb.lng || k.blat !== nb.lat) {
      k = { n: pts.length, lengthM: e.lengthM, alng: na.lng, alat: na.lat, blng: nb.lng, blat: nb.lat, zones: NONE };
      todoAt.push(kept.length);
      todo.push(pts);
    }
    next.set(e.id, k);
    kept.push(k);
    keptBuilt.push(e.built);
  }
  // T-1358: the edges worked out together, so that one pass takes a first network's 100,000 points
  const found = zonesOfLines(todo, a);
  for (let i = 0; i < todo.length; i++) kept[todoAt[i]].zones = found[i];
  byEdge = next;
  const built = new Set<CityId>(), planned = new Set<CityId>();
  for (let i = 0; i < kept.length; i++) {
    const zs = kept[i].zones, to = keptBuilt[i] ? built : planned;
    for (let j = 0; j < zs.length; j++) to.add(zs[j]);
  }
  for (const z of built) planned.delete(z);
  const prev = trackZones.peek();
  if (!same(prev.built, built) || !same(prev.planned, planned)) trackZones.value = { built, planned };
  // every state, since the game in hand may have changed under an unchanged set
  addOpened(built);
}

const same = (a: ReadonlySet<CityId>, b: ReadonlySet<CityId>) => a.size === b.size && [...a].every((z) => b.has(z));

/** degrees round a line's box, far more than the rounding of the points tested along it */
const BOX_PAD = 1e-7;

/** `crossed` for each drawn line. T-1358: the lines' boxes in local units go in a tree of their unions
 * (lines 0..n-1 its leaves, node k's children 2k and 2k + 1); a node whose box is `aloneIn` a zone
 * gives each of its lines that zone alone, as each line would find it, and the others are opened
 * down to a few lines. */
function zonesOfLines(lines: Float32Array[], a: ReturnType<typeof anchor>): CityId[][] {
  const n = lines.length, out: CityId[][] = new Array(n);
  let p = 1;
  while (p < n) p *= 2;
  const t = new Float64Array(8 * p).fill(Infinity);
  for (let k = 1; k < 2 * p; k++) { t[4 * k + 2] = -Infinity; t[4 * k + 3] = -Infinity; }
  for (let i = 0; i < n; i++) {
    const pts = lines[i], m = pts.length - 2;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, k = 0, bad = false;
    // two points a turn, the smaller of each pair against the low end only: this loop meets every
    // point of a first network, before the JIT has warmed to it (a point not a number spoils the box)
    for (; k + 3 < m; k += 6) {
      const xa = pts[k], xb = pts[k + 3], ya = pts[k + 1], yb = pts[k + 4];
      if (xa < xb) { if (xa < x0) x0 = xa; if (xb > x1) x1 = xb; } else if (xb <= xa) { if (xb < x0) x0 = xb; if (xa > x1) x1 = xa; } else bad = true;
      if (ya < yb) { if (ya < y0) y0 = ya; if (yb > y1) y1 = yb; } else if (yb <= ya) { if (yb < y0) y0 = yb; if (ya > y1) y1 = ya; } else bad = true;
    }
    if (k < m) {
      const x = pts[k], y = pts[k + 1];
      if (x !== x || y !== y) bad = true;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (bad) x0 = NaN;
    const at = 4 * (p + i);
    t[at] = x0; t[at + 1] = y0; t[at + 2] = x1; t[at + 3] = y1;
  }
  for (let k = p - 1; k >= 1; k--) {
    const l = 8 * k, r = 8 * k + 4;
    t[4 * k] = Math.min(t[l], t[r]); t[4 * k + 1] = Math.min(t[l + 1], t[r + 1]);
    t[4 * k + 2] = Math.max(t[l + 2], t[r + 2]); t[4 * k + 3] = Math.max(t[l + 3], t[r + 3]);
  }
  /** node k's box in degrees: longitude follows x and latitude y alone, so the corners bound every
   * point tested */
  const boxOf = (k: number): Box => {
    const [w, latA] = toLngLat(t[4 * k], t[4 * k + 1], a), [e, latB] = toLngLat(t[4 * k + 2], t[4 * k + 3], a);
    return [w - BOX_PAD, Math.min(latA, latB) - BOX_PAD, e + BOX_PAD, Math.max(latA, latB) + BOX_PAD];
  };
  /** node k, which holds lines lo to hi (not included) */
  const visit = (k: number, lo: number, hi: number) => {
    if (lo >= n) return;
    if (hint && aloneIn(hint, boxOf(k))) {
      const z = [hint];
      for (let i = lo; i < Math.min(hi, n); i++) out[i] = z;
    } else if (hi - lo <= 4) {
      for (let i = lo; i < Math.min(hi, n); i++) out[i] = crossed(lines[i], a, boxOf(p + i));
    } else {
      const m = (lo + hi) >> 1;
      visit(2 * k, lo, m);
      visit(2 * k + 1, m, hi);
    }
  };
  if (n) visit(1, 0, p);
  return out;
}

/** The zones whose outlines hold some point of one edge's drawn line (x, y, z triples, local units),
 * `box` [w, s, e, n] in degrees holding every point tested. */
function crossed(pts: Float32Array, a: ReturnType<typeof anchor>, box: Box): CityId[] {
  // T-1358: a line whose box lies in cells of a zone's outline wholly inside it would find that zone
  // alone at every point tested, if the zone is the first holding its first point: the zone last
  // found when no zone before it has a box meeting the line's, or else the first point's own
  if (hint && aloneIn(hint, box)) return [hint];
  const out = new Set<CityId>();
  let last: CityId | null = null, entered = false;
  const test = (lon: number, lat: number) => {
    if (last && inOutline(last, outlines!.get(last)!, lon, lat)) return;
    last = firstHolding(lon, lat);
    if (last) (out.add(last), (entered = true));
  };
  // A drawn line holds a point every few metres on curves: test one every STEP_M along it, measured
  // in local units (a ground metre is 1 / cos(lat) of them, lat from the last point tested)
  let cos = 1, px = pts[0], py = pts[1], since = 0;
  const at = (x: number, y: number) => { const [lon, lat] = toLngLat(x, y, a); test(lon, lat); cos = Math.cos(lat * RAD); };
  at(px, py);
  if (last && inOutlineBox(last, outlines!.get(last)!, ...box)) return [last];
  entered = false;
  for (let k = 3; k + 2 < pts.length; k += 3) {
    const qx = pts[k], qy = pts[k + 1], len = Math.hypot(qx - px, qy - py) * cos;
    for (let s = STEP_M - since; s < len; s += STEP_M) {
      at(px + ((qx - px) * s) / len, py + ((qy - py) * s) / len);
      // T-1358: once the rest of the line, from this stretch's start, lies in cells wholly inside the
      // zone just entered, every point still to test holds it
      if (entered && ((entered = false), restIn(last!, pts, k - 3, a))) return [...out];
    }
    // (most stretches are short: the remainder only when it can differ, so the same to the bit)
    const to = since + len;
    since = to >= 0 && to < STEP_M ? to : to % STEP_M;
    px = qx; py = qy;
  }
  at(px, py);
  return [...out];
}

/** Whether a drawn line's points from index `from` (of x, y, z triples) on lie, by their box, in cells
 * of zone `id`'s outline wholly inside it; false for a point not a number. */
function restIn(id: CityId, pts: Float32Array, from: number, a: ReturnType<typeof anchor>): boolean {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = from; k + 2 < pts.length; k += 3) {
    const x = pts[k], y = pts[k + 1];
    if (x !== x || y !== y) return false;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  const [w, latA] = toLngLat(x0, y0, a), [e, latB] = toLngLat(x1, y1, a);
  return inOutlineBox(id, outlines!.get(id)!, w - BOX_PAD, Math.min(latA, latB) - BOX_PAD, e + BOX_PAD, Math.max(latA, latB) + BOX_PAD);
}

/** the zone `firstHolding` last found, tried first next time */
let hint: CityId | null = null;
/** per zone: its box, and the zones before it in registry order whose boxes meet it, with theirs */
type Box = readonly [number, number, number, number];
type Rivals = { box: Box; rivals: { id: CityId; box: Box; g: MultiPolygon }[] };
const rivalsOf = new Map<CityId, Rivals>();
const inBox = (b: Box, lon: number, lat: number) => lon <= b[2] && lon >= b[0] && lat <= b[3] && lat >= b[1];

function rivals(id: CityId): Rivals {
  let r = rivalsOf.get(id);
  if (r) return r;
  const box = zoneBox(cityById(id)), near = zonesTouching(...box), at = near.findIndex((c) => c.id === id);
  const before = near.slice(0, Math.max(0, at)).flatMap((c) => { const g = outlines!.get(c.id); return g ? [{ id: c.id, box: zoneBox(c), g }] : []; });
  // (a zone missing from its own box's list holds nothing here)
  rivalsOf.set(id, (r = { box: at < 0 ? [1, 1, 0, 0] : box, rivals: before }));
  return r;
}

/** The first zone in registry order whose box and outline hold (lon, lat), as a scan of
 * `zonesTouching` finds it; null for none. T-1358: the last one found holds most places asked next,
 * and only a zone before it whose box meets its own could come first there, so those alone are tried
 * before a scan. */
function firstHolding(lon: number, lat: number): CityId | null {
  if (hint) {
    const r = rivals(hint);
    if (inBox(r.box, lon, lat) && inOutline(hint, outlines!.get(hint)!, lon, lat) && !r.rivals.some((q) => inBox(q.box, lon, lat) && inOutline(q.id, q.g, lon, lat))) return hint;
  }
  for (const c of zonesTouching(lon, lat, lon, lat)) {
    const g = outlines!.get(c.id);
    if (g && inOutline(c.id, g, lon, lat)) return (hint = c.id);
  }
  return null;
}

/** Whether zone `id` is the first holding every place in box `b`, and its outline's cells say so:
 * `b` lies in its box and in cells wholly inside it, and meets the box of no zone before it. */
function aloneIn(id: CityId, b: Box): boolean {
  const r = rivals(id);
  return b[0] >= r.box[0] && b[2] <= r.box[2] && b[1] >= r.box[1] && b[3] <= r.box[3]
    && !r.rivals.some((q) => q.box[0] <= b[2] && q.box[2] >= b[0] && q.box[1] <= b[3] && q.box[3] >= b[1])
    && inOutlineBox(id, outlines!.get(id)!, ...b);
}

