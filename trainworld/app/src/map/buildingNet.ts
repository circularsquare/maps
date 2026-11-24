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
/** each edge's id, points and box (x0, y0, x1, y1), by the network's points map */
const boxesOf = new WeakMap<Map<number, Float32Array>, { ids: number[]; pts: Float32Array[]; box: Float64Array }>();

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
): { draw: BuildingDraw[]; memberOf: Int32Array } {
  const memberOf = new Int32Array(st.length).fill(-1);
  const draw: BuildingDraw[] = [];
  if (!buildings?.length || !ll || ll.length !== buildings.length) {
    if (inside) kept.clear();
    return { draw, memberOf };
  }
  const line = (i: number) => stationLine.subarray(i * 2 * PLATFORM_POINTS, (i + 1) * 2 * PLATFORM_POINTS);
  const seen = new Set<number>();
  // each edge's box, once (T-4153: kept while the network's points are the same map)
  let boxes = inside ? boxesOf.get(edgePts) : undefined;
  if (inside && !boxes) {
    const ids: number[] = [], pts: Float32Array[] = [], box = new Float64Array(edgePts.size * 4);
    for (const [id, p] of edgePts) {
      let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
      for (let j = 0; j + 2 < p.length; j += 3) {
        a = Math.min(a, p[j]); c = Math.max(c, p[j]);
        b = Math.min(b, p[j + 1]); d = Math.max(d, p[j + 1]);
      }
      box.set([a, b, c, d], ids.length * 4);
      ids.push(id);
      pts.push(p);
    }
    boxesOf.set(edgePts, (boxes = { ids, pts, box }));
  }
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
    for (let i = 0; i < st.length; i++) {
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
    for (let q = 0; boxes && q < boxes.ids.length; q++) {
      const id = boxes.ids[q], p = boxes.pts[q], e = boxes.box;
      if (e[q * 4 + 2] < x0 - pad || e[q * 4] > x1 + pad || e[q * 4 + 3] < y0 - pad || e[q * 4 + 1] > y1 + pad) continue;
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
