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

interface Kept { key: string; draw: BuildingDraw }
const kept = new Map<number, Kept>();

/** The buildings as drawn, and per station the building it is in (an index into them, -1 none). `ll`:
 * each building's corners as lng, lat pairs (the worker's frame). */
export function buildingsDrawn(
  buildings: readonly BuildingInput[] | undefined,
  ll: readonly ArrayLike<number>[] | undefined,
  st: readonly { lat: number; platform: number }[],
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
  // each edge's box, once
  const boxes: { id: number; p: Float32Array; x0: number; y0: number; x1: number; y1: number }[] = [];
  if (inside) for (const [id, p] of edgePts) {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (let j = 0; j + 2 < p.length; j += 3) {
      a = Math.min(a, p[j]); c = Math.max(c, p[j]);
      b = Math.min(b, p[j + 1]); d = Math.max(d, p[j + 1]);
    }
    boxes.push({ id, p, x0: a, y0: b, x1: c, y1: d });
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
    for (const { id, p, ...e } of boxes) {
      if (e.x1 < x0 - pad || e.x0 > x1 + pad || e.y1 < y0 - pad || e.y0 > y1 + pad) continue;
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
    const key = [b.id, Array.from(corners, (v) => v.toFixed(2)).join(","), plats.map((p) => `${p.member}:${p.line.map((v) => v.toFixed(1)).join(",")}`).join("|"), tracks.map((t) => `${t.id}:${t.pts.length}:${t.pts[0]?.toFixed(1)},${t.pts[1]?.toFixed(1)},${t.pts.at(-2)?.toFixed(1)},${t.pts.at(-1)?.toFixed(1)}`).join("|"), clear.map((r) => `${r.x.toFixed(1)},${r.y.toFixed(1)}`).join("|")].join("/");
    seen.add(b.id);
    const old = kept.get(b.id);
    if (old && old.key === key) {
      draw.push(old.draw);
      return;
    }
    const lay = layoutBuilding({ corners, perM, platforms: plats, tracks: tracks.map((t) => t.pts), clear, seed: b.id, shops: members.length > 0 });
    const d: BuildingDraw = {
      id: b.id, corners, members, ux: lay.ux, uy: lay.uy,
      shops: new Float32Array(lay.shops), cells: new Float32Array(lay.cells),
      hall: lay.hall, shortM: lay.shortM, room: lay.room, perM,
    };
    kept.set(b.id, { key, draw: d });
    draw.push(d);
  });
  if (inside) for (const id of [...kept.keys()]) if (!seen.has(id)) kept.delete(id);
  return { draw, memberOf };
}
