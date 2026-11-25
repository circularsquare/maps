// The station buildings as the network renderer draws them (T-2219), worked out in the clock worker
// with the rest of the network (network.ts): each outline in local units, its members as drawn (the
// stations whose drawn platform touches it, game/buildings.ts's rule over the drawn lines), and its
// inside (game/buildingLayout.ts: shops, the cross-hall, the free floor for the hall's crowd, the
// room). The inside is kept per building and made again only when its outline or the platforms and
// track in and round it change.

import { BUILDING_TOUCH_M, platformTouches } from "../game/buildings";
import { layoutBuilding, type Rect } from "../game/buildingLayout";
import { PLATFORM_POINTS, ST_ROOM } from "./stationGeometry";
import { roomRect } from "./facilityPlace";
import { toLocal } from "./geo";
import { fault } from "./selfTest";
import type { BuildingInput } from "../game/types";

/** A station building as drawn, local units of the network's anchor. */
export interface BuildingDraw {
  id: number;
  /** the outline, x, y pairs */
  corners: Float32Array;
  /** its members, station indices in the network */
  members: number[];
  /** along the first edge (unit) */
  ux: number;
  uy: number;
  /** per shop SHOP_STRIDE numbers: centre x, y; half length and depth, m; colour; glyph */
  shops: Float32Array;
  /** per free floor cell CELL_STRIDE numbers: centre x, y; the nearest member (station index); its distance, m */
  cells: Float32Array;
  /** the middle of the cross-hall, and the short side, m */
  hall: [number, number];
  shortM: number;
  /** riders the hall holds */
  room: number;
  perM: number;
}

export const SHOP_STRIDE = 6, CELL_STRIDE = 4;

/** (T-4153: keyed by station ids, so a station added or removed elsewhere, which moves every index after it, keeps the
 * layouts; `ids` the station id of each index its draw holds) */
interface Kept { key: string; draw: BuildingDraw; ids: Map<number, number> }
const kept = new Map<number, Kept>();
/** T-4153: each edge's box (x0, y0, x1, y1) by id, in square cells of `size` local units (ids in no set order; one over
 * more than WIDE cells in `wide`), for the points map `map`; kept with the network's cache (map/network.ts) and patched
 * from the edges it says changed */
export interface EdgeBoxes { map: ReadonlyMap<number, Float32Array> | null; size: number; boxes: Map<number, Float64Array>; cells: Map<number, number[]>; wide: Set<number> }
export const newEdgeBoxes = (): EdgeBoxes => ({ map: null, size: 0, boxes: new Map(), cells: new Map(), wide: new Set() });
function boxOf(p: Float32Array): Float64Array {
  let e = boxOfPts.get(p);
  if (!e) {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (let j = 0; j + 2 < p.length; j += 3) {
      a = Math.min(a, p[j]); c = Math.max(c, p[j]);
      b = Math.min(b, p[j + 1]); d = Math.max(d, p[j + 1]);
    }
    boxOfPts.set(p, (e = Float64Array.of(a, b, c, d)));
  }
  return e;
}
/** `ix` made for `edgePts`: patched from the edges `changed` names when it was made for the map those are relative to */
function edgeBoxesFor(ix: EdgeBoxes, edgePts: ReadonlyMap<number, Float32Array>, size: number, changed?: { pts: ReadonlyMap<number, Float32Array>; edges: ReadonlySet<number> }) {
  if (ix.map === edgePts) return;
  const cellsOf = (b: Float64Array, f: (k: number) => void) => {
    const s = ix.size;
    for (let cx = Math.floor(b[0] / s); cx <= Math.floor(b[2] / s); cx++) for (let cy = Math.floor(b[1] / s); cy <= Math.floor(b[3] / s); cy++) f(cx * 1_000_003 + cy);
  };
  const wide = (b: Float64Array) => (Math.floor(b[2] / ix.size) - Math.floor(b[0] / ix.size) + 1) * (Math.floor(b[3] / ix.size) - Math.floor(b[1] / ix.size) + 1) > WIDE;
  const add = (id: number, p: Float32Array) => {
    const b = boxOf(p);
    ix.boxes.set(id, b);
    if (!(b[0] <= b[2])) return;
    if (wide(b)) { ix.wide.add(id); return; }
    cellsOf(b, (k) => { const v = ix.cells.get(k); if (v) v.push(id); else ix.cells.set(k, [id]); });
  };
  const drop = (id: number) => {
    const b = ix.boxes.get(id);
    if (!b) return;
    ix.boxes.delete(id);
    if (ix.wide.delete(id) || !(b[0] <= b[2])) return;
    cellsOf(b, (k) => {
      const v = ix.cells.get(k);
      if (!v) return;
      const at = v.indexOf(id);
      if (at < 0) return;
      v[at] = v[v.length - 1];
      v.pop();
      if (!v.length) ix.cells.delete(k);
    });
  };
  if (changed && ix.map && ix.map === changed.pts) {
    for (const id of changed.edges) {
      const p = edgePts.get(id);
      // (a new id has no box here yet and its points no box of their own: both undefined must not read as the same)
      const had = ix.boxes.get(id);
      if (p && (had || fault === "buildings") && had === boxOfPts.get(p)) continue;
      drop(id);
      if (p) add(id, p);
    }
  } else {
    ix.size = size;
    ix.boxes.clear();
    ix.cells.clear();
    ix.wide.clear();
    for (const [id, p] of edgePts) add(id, p);
  }
  ix.map = edgePts;
}
/** the edges whose cells meet a box, by id */
function edgesNear(ix: EdgeBoxes, x0: number, y0: number, x1: number, y1: number): number[] {
  const s = ix.size, out = new Set<number>(ix.wide);
  for (let cx = Math.floor(x0 / s); cx <= Math.floor(x1 / s); cx++)
    for (let cy = Math.floor(y0 / s); cy <= Math.floor(y1 / s); cy++) {
      const v = ix.cells.get(cx * 1_000_003 + cy);
      if (v) for (const id of v) out.add(id);
    }
  return [...out].sort((a, b) => a - b);
}
/** T-4153: each edge's box by its points (an edge whose points are as before keeps its box when the map is new) */
const boxOfPts = new WeakMap<Float32Array, Float64Array>();

/** T-4153: things by their boxes in square cells, so a building looks only at the stations and track near it, in the
 * order a scan of every one would meet them (each cell's list ascending, a query's union sorted) */
class BoxGrid {
  private cells = new Map<number, number[]>();
  /** things over more cells than WIDE: in every query */
  private wide: number[] = [];
  private stamp = new Int32Array(0);
  private serial = 0;
  constructor(private readonly size: number) {}
  add(i: number, x0: number, y0: number, x1: number, y1: number) {
    const s = this.size;
    if ((Math.floor(x1 / s) - Math.floor(x0 / s) + 1) * (Math.floor(y1 / s) - Math.floor(y0 / s) + 1) > WIDE) { this.wide.push(i); return; }
    for (let cx = Math.floor(x0 / s); cx <= Math.floor(x1 / s); cx++)
      for (let cy = Math.floor(y0 / s); cy <= Math.floor(y1 / s); cy++) {
        const k = cx * 1_000_003 + cy, v = this.cells.get(k);
        if (v) v.push(i);
        else this.cells.set(k, [i]);
      }
  }
  /** the things whose cells meet the box, ascending (`n`: how many things there are) */
  near(x0: number, y0: number, x1: number, y1: number, n: number): number[] {
    if (this.stamp.length < n) this.stamp = new Int32Array(n);
    const s = this.size, mark = ++this.serial, out: number[] = [];
    for (const i of this.wide) { this.stamp[i] = mark; out.push(i); }
    for (let cx = Math.floor(x0 / s); cx <= Math.floor(x1 / s); cx++)
      for (let cy = Math.floor(y0 / s); cy <= Math.floor(y1 / s); cy++) {
        const v = this.cells.get(cx * 1_000_003 + cy);
        if (v) for (const i of v) if (this.stamp[i] !== mark) { this.stamp[i] = mark; out.push(i); }
      }
    return out.sort((a, b) => a - b);
  }
}
/** the grids' cell, m (a station building and what is near it fit in one or a few); past WIDE cells a thing is in every
 * query (a track hundreds of km long) */
const GRID_M = 4000, WIDE = 64;

/** The buildings as drawn, and per station the building it is in (an index into them, -1 none). `ll`:
 * each building's corners as lng, lat pairs (the worker's frame). */
export function buildingsDrawn(
  buildings: readonly BuildingInput[] | undefined,
  ll: readonly ArrayLike<number>[] | undefined,
  st: readonly { id: number; lat: number; platform: number }[],
  stations: Float32Array,
  stationLine: Float32Array,
  edgePts: Map<number, Float32Array>,
  inside = true,
  boxes: EdgeBoxes = newEdgeBoxes(),
  changed?: { pts: ReadonlyMap<number, Float32Array>; edges: ReadonlySet<number> },
): { draw: BuildingDraw[]; memberOf: Int32Array } {
  const memberOf = new Int32Array(st.length).fill(-1);
  const draw: BuildingDraw[] = [];
  if (!buildings?.length || !ll || ll.length !== buildings.length) {
    if (inside) kept.clear();
    return { draw, memberOf };
  }
  const line = (i: number) => stationLine.subarray(i * 2 * PLATFORM_POINTS, (i + 1) * 2 * PLATFORM_POINTS);
  const seen = new Set<number>();
  // (T-4153: the stations by their places, each reaching as far as the longest platform)
  const perM0 = 1 / Math.cos(((st[0]?.lat ?? 0) * Math.PI) / 180), cellOf = GRID_M * perM0;
  const stGrid = new BoxGrid(cellOf);
  let longest = 0;
  for (let i = 0; i < st.length; i++) {
    longest = Math.max(longest, st[i].platform);
    stGrid.add(i, stations[i * 3], stations[i * 3 + 1], stations[i * 3], stations[i * 3 + 1]);
  }
  // each edge's box (T-4153: kept with the network's cache, patched where its edges changed)
  if (inside) edgeBoxesFor(boxes, edgePts, cellOf, changed);
  buildings.forEach((b, k) => {
    const g = ll[k], n = g.length / 2;
    if (n < 3) return;
    const corners = new Float32Array(2 * n);
    let lat = 0;
    for (let i = 0; i < n; i++) {
      corners.set(toLocal(g[2 * i], g[2 * i + 1]), 2 * i);
      lat += g[2 * i + 1] / n;
    }
    const perM = 1 / Math.cos((lat * Math.PI) / 180);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) {
      x0 = Math.min(x0, corners[2 * i]); x1 = Math.max(x1, corners[2 * i]);
      y0 = Math.min(y0, corners[2 * i + 1]); y1 = Math.max(y1, corners[2 * i + 1]);
    }
    // the platforms and track in and round it
    const pad = 60 * perM;
    const near: number[] = [], members: number[] = [];
    const reach = longest * perM + pad;
    for (const i of stGrid.near(x0 - reach, y0 - reach, x1 + reach, y1 + reach, st.length)) {
      const x = stations[i * 3], y = stations[i * 3 + 1], r = st[i].platform * perM + pad;
      if (x < x0 - r || x > x1 + r || y < y0 - r || y > y1 + r) continue;
      near.push(i);
      if (memberOf[i] < 0 && platformTouches(corners, x, y, line(i), BUILDING_TOUCH_M * perM)) {
        memberOf[i] = draw.length;
        members.push(i);
      }
    }
    if (!inside) {
      draw.push({ id: b.id, corners, members, ux: 1, uy: 0, shops: new Float32Array(0), cells: new Float32Array(0), hall: [0, 0], shortM: 0, room: 0, perM });
      return;
    }
    const tracks: { id: number; pts: number[] }[] = [];
    // (no `...rest` here: it made a copy of every edge's box for every building, most of a build's time with hundreds
    // of buildings, T-3659)
    // (by edge id: the layout reads the tracks as a set)
    for (const id of edgesNear(boxes, x0 - pad, y0 - pad, x1 + pad, y1 + pad)) {
      const p = edgePts.get(id)!, e = boxes.boxes.get(id)!;
      if (e[2] < x0 - pad || e[0] > x1 + pad || e[3] < y0 - pad || e[1] > y1 + pad) continue;
      // the pieces of it near the floor, each segment on its own
      for (let j = 0; j + 5 < p.length; j += 3) {
        const ax = p[j], ay = p[j + 1], bx = p[j + 3], by = p[j + 4];
        if (Math.max(ax, bx) < x0 - pad || Math.min(ax, bx) > x1 + pad || Math.max(ay, by) < y0 - pad || Math.min(ay, by) > y1 + pad) continue;
        tracks.push({ id, pts: [ax, ay, bx, by] });
      }
    }
    const plats = near.map((i) => {
      const o = line(i), x = stations[i * 3], y = stations[i * 3 + 1], abs: number[] = [];
      for (let j = 0; j + 1 < o.length; j += 2) abs.push(x + o[j], y + o[j + 1]);
      return { line: abs, member: members.includes(i) ? i : -1 };
    });
    const clear: Rect[] = near.filter((i) => Math.round(stations[i * 3 + 2]) & ST_ROOM).map((i) => roomRect(i, stations, stationLine, perM));
    const key = [b.id, Array.from(corners, (v) => v.toFixed(2)).join(","), plats.map((p) => `${p.member < 0 ? -1 : st[p.member].id}:${p.line.map((v) => v.toFixed(1)).join(",")}`).join("|"), tracks.map((t) => `${t.id}:${t.pts.length}:${t.pts[0]?.toFixed(1)},${t.pts[1]?.toFixed(1)},${t.pts.at(-2)?.toFixed(1)},${t.pts.at(-1)?.toFixed(1)}`).join("|"), clear.map((r) => `${r.x.toFixed(1)},${r.y.toFixed(1)}`).join("|")].join("/");
    seen.add(b.id);
    const old = kept.get(b.id);
    if (old && old.key === key) {
      draw.push(reindexed(old, st));
      return;
    }
    const lay = layoutBuilding({ corners, perM, platforms: plats, tracks: tracks.map((t) => t.pts), clear, seed: b.id, shops: members.length > 0 });
    const d: BuildingDraw = {
      id: b.id, corners, members, ux: lay.ux, uy: lay.uy,
      shops: new Float32Array(lay.shops), cells: new Float32Array(lay.cells),
      hall: lay.hall, shortM: lay.shortM, room: lay.room, perM,
    };
    kept.set(b.id, { key, draw: d, ids: new Map(near.map((i) => [i, st[i].id])) });
    draw.push(d);
  });
  if (inside) for (const id of [...kept.keys()]) if (!seen.has(id)) kept.delete(id);
  return { draw, memberOf };
}

/** A kept building's draw with its station indices (members, each floor cell's nearest member) moved to where those
 * stations are in `st` now: the same object when none moved. */
function reindexed(k: Kept, st: readonly { id: number }[]): BuildingDraw {
  let moved = false;
  for (const [i, id] of k.ids) if (st[i]?.id !== id) { moved = true; break; }
  if (!moved) return k.draw;
  const now = new Map<number, number>();
  for (const [i, id] of k.ids) {
    // (the kept key holds every one of these ids, so each is in `st`)
    let j = i;
    while (j < st.length && st[j].id !== id) j++;
    if (j === st.length) for (j = 0; j < st.length && st[j].id !== id; j++);
    now.set(i, j);
  }
  const d = k.draw, cells = d.cells.slice();
  for (let c = 2; c < cells.length; c += CELL_STRIDE) cells[c] = now.get(cells[c]) ?? cells[c];
  const draw = { ...d, members: d.members.map((i) => now.get(i)!), cells };
  k.draw = draw;
  k.ids = new Map([...k.ids].map(([i, id]) => [now.get(i)!, id]));
  return draw;
}
