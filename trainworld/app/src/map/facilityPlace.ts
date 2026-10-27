// Where a complex's waiting room and car park go (T-1528, notes/T-1528.md), in the clock worker.
// Each is drawn once per station complex, however many of its stations have one. The waiting room
// is an annex joined to one box (stationGeometry.ts `stationAnnexes`): of the stations with one and
// the two sides of each, the spot with the fewest other platforms, tracks, car parks and the name
// label near it. The car park is a lot of its own beside the complex (map/parkingLots.ts): the
// nearest spot along one of the complex's platforms that overlaps no box, track, waiting room, other
// lot or the label, shrinking if none does, and the least bad spot if nothing fits.

import { PLATFORM_POINTS, ST_FLIP, ST_PARK, ST_ROOM } from "./stationGeometry";

/** A lot's numbers in `NetworkBuffers.lots`: centre x, y (local); half its length along it as a
 * vector (local); half length and half depth, m; bays; flags (LOT_*); the station its hover names
 * (index); how many of the complex's stations have park and ride. */
export const LOT_STRIDE = 10;
export const LOT_BLUEPRINT = 1, LOT_OFF = 2;
/** A car space and the aisle between two rows, m: a bay is a row each side of an aisle. */
export const SPACE_W = 2.5, SPACE_L = 5, AISLE = 6, BAY_M = 2 * SPACE_L + AISLE;
/** A lot runs along the longest platform with park and ride, within these lengths, m. */
const LOT_MIN_M = 80, LOT_MAX_M = 300, LOT_SHORT_M = 60;
/** At most this many bays deep: one for each station with park and ride. */
const MAX_BAYS = 3;
/** How far a lot's near edge is from the platform's centre line, m: past a box drawn at zoom 14,
 * where it is about 32 m wide on the ground (T-553). */
const LOT_GAPS_M = [20, 30, 42, 56, 72];
/** Kept clear round a lot, m: a box's centre line (half a zoom 14 box plus a little), a track, a
 * waiting room or another lot. */
const CLEAR_BOX_M = 18, CLEAR_TRACK_M = 6, CLEAR_LOT_M = 4;
/** Where a waiting room reaches, m from the platform's centre line and along it from its middle:
 * about the drawn size between zoom 14.5 and 16, where a box is 16-30 m wide on the ground. */
const ACROSS_M = [14, 24, 34], ALONG_M = [-28, -10, 10, 28];
const ROOM_NEAR_M = 14, ROOM_TRACK_M = 5;
/** The name label's patch beside the complex, m (stationLabels.ts: below the station where its
 * track runs east-west, else to its right), about its size at zoom 16. */
const LABEL_BELOW = { halfWide: 90, deep: 50 }, LABEL_RIGHT = { wide: 150, halfDeep: 25 };
const CELL_M = 40;

interface Rect { x: number; y: number; ux: number; uy: number; hl: number; hd: number }

/** Place every complex's waiting room and car park. `stations` flags: ST_ROOM and ST_PARK as built
 * at each station on entry; on return ST_ROOM is left only on the one station per complex whose box
 * draws the room, with ST_FLIP for its side. `complexOf`: each node's complex (0xffffffff none);
 * absent, each station is its own. Returns the lots, LOT_STRIDE numbers each. */
export function placeFacilities(
  st: { id: number; lat: number; stationBuilt?: boolean; stationOff?: boolean; platform: number }[],
  stations: Float32Array,
  stationLine: Float32Array,
  stationShape: Float32Array,
  edgePts: Map<number, Float32Array>,
  waysAt: Map<number, number[]>,
  complexOf?: ArrayLike<number>,
): Float32Array {
  const flag = (i: number) => Math.round(stations[i * 3 + 2]);
  const want: number[] = [];
  for (let i = 0; i < st.length; i++) if (flag(i) & (ST_ROOM | ST_PARK)) want.push(i);
  if (!want.length) return new Float32Array(0);
  const perM = 1 / Math.cos(st[want[0]].lat * Math.PI / 180), cell = CELL_M * perM;
  const keyOf = (i: number) => {
    const c = complexOf?.[st[i].id];
    return c === undefined || c === 0xffffffff ? -1 - i : c;
  };
  // the complexes with a facility, each with all its stations, in node order
  const groups = new Map<number, number[]>();
  for (const i of want) groups.set(keyOf(i), []);
  for (let i = 0; i < st.length; i++) groups.get(keyOf(i))?.push(i);

  // segments by grid cell: x0, y0, x1, y1, owner (edge id, or -1 - station index)
  const seg: number[] = [], grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => cx * 1_000_003 + cy;
  const add = (x0: number, y0: number, x1: number, y1: number, owner: number) => {
    const k = seg.length / 5;
    seg.push(x0, y0, x1, y1, owner);
    for (let cx = Math.floor(Math.min(x0, x1) / cell); cx <= Math.floor(Math.max(x0, x1) / cell); cx++)
      for (let cy = Math.floor(Math.min(y0, y1) / cell); cy <= Math.floor(Math.max(y0, y1) / cell); cy++) {
        const g = grid.get(key(cx, cy));
        if (g) g.push(k); else grid.set(key(cx, cy), [k]);
      }
  };
  // only what lies near a complex with a facility: its platforms and a lot's reach past them
  const near = new Set<number>();
  for (const members of groups.values()) for (const i of members) {
    const cx = Math.floor(stations[i * 3] / cell), cy = Math.floor(stations[i * 3 + 1] / cell);
    const r = Math.ceil(((st[i].platform / 2 + LOT_MAX_M / 2 + LOT_GAPS_M[LOT_GAPS_M.length - 1] + MAX_BAYS * BAY_M) * perM) / cell) + 1;
    for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) near.add(key(cx + dx, cy + dy));
  }
  const isNear = (x: number, y: number) => near.has(key(Math.floor(x / cell), Math.floor(y / cell)));
  // a straight run of track can be one long segment: tried a cell apart along it
  const passesNear = (x0: number, y0: number, x1: number, y1: number) => {
    const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / cell);
    for (let j = 0; j <= n; j++) if (isNear(x0 + (x1 - x0) * j / Math.max(1, n), y0 + (y1 - y0) * j / Math.max(1, n))) return true;
    return false;
  };
  for (const [id, p] of edgePts)
    for (let k = 0; k + 5 < p.length; k += 3) if (passesNear(p[k], p[k + 1], p[k + 3], p[k + 4])) add(p[k], p[k + 1], p[k + 3], p[k + 4], id);
  for (let i = 0; i < st.length; i++) {
    const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS;
    if (!isNear(x, y)) continue;
    for (let j = 0; j + 1 < PLATFORM_POINTS; j++) add(x + stationLine[o + j * 2], y + stationLine[o + j * 2 + 1], x + stationLine[o + j * 2 + 2], y + stationLine[o + j * 2 + 3], -1 - i);
  }
  const segsNear = (r: Rect, pad: number) => {
    const ext = Math.hypot(r.hl, r.hd) * perM + pad;
    const out = new Set<number>();
    for (let cx = Math.floor((r.x - ext) / cell); cx <= Math.floor((r.x + ext) / cell); cx++)
      for (let cy = Math.floor((r.y - ext) / cell); cy <= Math.floor((r.y + ext) / cell); cy++)
        for (const k of grid.get(key(cx, cy)) ?? []) out.add(k);
    return out;
  };

  /** a platform's middle, its way there and the left of it (+), local units */
  const frame = (i: number) => {
    const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS, m = PLATFORM_POINTS >> 1;
    let dx = stationLine[o + m * 2 + 2] - stationLine[o + m * 2 - 2], dy = stationLine[o + m * 2 + 3] - stationLine[o + m * 2 - 1];
    const dl = Math.hypot(dx, dy) || 1;
    dx /= dl; dy /= dl;
    return { mx: x + stationLine[o + m * 2], my: y + stationLine[o + m * 2 + 1], dx, dy, nx: -dy, ny: dx };
  };

  const lots: number[] = [], lotRects: Rect[] = [], rooms: Rect[] = [];
  const order = [...groups.values()].sort((a, b) => st[a[0]].id - st[b[0]].id);
  for (const members of order) {
    const room = members.filter((i) => flag(i) & ST_ROOM), park = members.filter((i) => flag(i) & ST_PARK);
    // the name label's patch: below or right of the complex's middle, by its first station's way
    let lx = 0, ly = 0;
    for (const i of members) { lx += stations[i * 3]; ly += stations[i * 3 + 1]; }
    lx /= members.length; ly /= members.length;
    const f0 = frame(members[0]), below = Math.abs(f0.dx) > Math.abs(f0.dy);
    // local y runs south, as the screen's does
    const label: Rect = below
      ? { x: lx, y: ly + LABEL_BELOW.deep / 2 * perM, ux: 1, uy: 0, hl: LABEL_BELOW.halfWide, hd: LABEL_BELOW.deep / 2 }
      : { x: lx + LABEL_RIGHT.wide / 2 * perM, y: ly, ux: 1, uy: 0, hl: LABEL_RIGHT.wide / 2, hd: LABEL_RIGHT.halfDeep };

    // the waiting room: one box of the complex draws it
    for (const i of room) stations[i * 3 + 2] -= ST_ROOM;
    if (room.length) {
      let best: { i: number; side: number; bad: number } | null = null;
      for (const i of room) {
        const f = frame(i), own = new Set(waysAt.get(st[i].id) ?? []);
        // + is to the left of the platform's way; it bends towards + when phi > 0, so its outside is -
        const outside = stationShape[i * 4 + 2] < 0 ? 1 : -1;
        for (const sg of [outside, -outside]) {
          let bad = 0;
          for (const a of ALONG_M) for (const c of ACROSS_M) {
            const px = f.mx + (f.dx * a + f.nx * sg * c) * perM, py = f.my + (f.dy * a + f.ny * sg * c) * perM;
            let worst = 0;
            const cx = Math.floor(px / cell), cy = Math.floor(py / cell);
            for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) for (const k of grid.get(key(cx + u, cy + v)) ?? []) {
              const owner = seg[k * 5 + 4];
              if (owner === -1 - i || own.has(owner)) continue;
              const d = distSeg(px, py, seg[k * 5], seg[k * 5 + 1], seg[k * 5 + 2], seg[k * 5 + 3]) / perM;
              if (owner < 0 && d < ROOM_NEAR_M) worst = 2;
              else if (owner >= 0 && d < ROOM_TRACK_M) worst = Math.max(worst, 1);
            }
            if (lotRects.some((r) => inRect(r, px, py, perM, 0))) worst = 2;
            if (inRect(label, px, py, perM, 0)) worst = Math.max(worst, 1);
            bad += worst;
          }
          if (!best || bad < best.bad) best = { i, side: sg, bad };
        }
      }
      const b = best!;
      stations[b.i * 3 + 2] += ST_ROOM;
      if (b.side > 0 && !(flag(b.i) & ST_FLIP)) stations[b.i * 3 + 2] += ST_FLIP;
      if (b.side < 0 && (flag(b.i) & ST_FLIP)) stations[b.i * 3 + 2] -= ST_FLIP;
      const f = frame(b.i);
      rooms.push({ x: f.mx + f.nx * b.side * 24 * perM, y: f.my + f.ny * b.side * 24 * perM, ux: f.dx, uy: f.dy, hl: 30, hd: 12 });
    }

    // the car park: a lot of its own beside the complex
    if (!park.length) continue;
    const longest = Math.max(...park.map((i) => st[i].platform));
    const full = Math.max(LOT_MIN_M, Math.min(LOT_MAX_M, longest)), bays = Math.min(park.length, MAX_BAYS);
    const sizes: [number, number][] = [[full, bays]];
    if (bays > 1) sizes.push([full, 1]);
    if (full * 0.6 >= LOT_SHORT_M) sizes.push([Math.max(LOT_SHORT_M, full * 0.6), 1]);
    // along every platform of the complex, those with park and ride first, longest first
    const along = [...members].sort((a, b) => Number(!!(flag(b) & ST_PARK)) - Number(!!(flag(a) & ST_PARK)) || st[b].platform - st[a].platform);
    const badness = (r: Rect) => {
      let bad = 0;
      for (const k of segsNear(r, CLEAR_BOX_M * perM)) {
        const owner = seg[k * 5 + 4], box = owner < 0;
        if (segHitsRect(seg[k * 5], seg[k * 5 + 1], seg[k * 5 + 2], seg[k * 5 + 3], r, perM, box ? CLEAR_BOX_M : CLEAR_TRACK_M)) bad += box ? 3 : 2;
      }
      for (const o of rooms) if (rectsMeet(r, o, perM, CLEAR_LOT_M)) bad += 2;
      for (const o of lotRects) if (rectsMeet(r, o, perM, CLEAR_LOT_M)) bad += 2;
      if (rectsMeet(r, label, perM, 0)) bad += 1;
      return bad;
    };
    let pick: { r: Rect; bays: number; bad: number } | null = null;
    search: for (const [len, nb] of sizes) {
      const hl = Math.floor(len / SPACE_W) * SPACE_W / 2, hd = nb * BAY_M / 2;
      for (const gap of LOT_GAPS_M) for (const i of along) {
        const f = frame(i);
        // the side away from the label first
        const sides = (below ? f.ny : f.nx) > 0 ? [-1, 1] : [1, -1];
        for (const sg of sides) for (const a of [0, -0.25, 0.25, -0.5, 0.5]) {
          const off = gap + hd, t = a * st[i].platform;
          const r: Rect = { x: f.mx + (f.dx * t + f.nx * sg * off) * perM, y: f.my + (f.dy * t + f.ny * sg * off) * perM, ux: f.dx, uy: f.dy, hl, hd };
          const bad = badness(r);
          if (!pick || bad < pick.bad) pick = { r, bays: nb, bad };
          if (bad === 0) break search;
        }
      }
    }
    const p = pick!;
    lotRects.push(p.r);
    const built = park.some((i) => st[i].stationBuilt !== false), on = park.some((i) => !st[i].stationOff);
    lots.push(p.r.x, p.r.y, p.r.ux * p.r.hl * perM, p.r.uy * p.r.hl * perM, p.r.hl, p.r.hd, p.bays,
      (built ? 0 : LOT_BLUEPRINT) | (on ? 0 : LOT_OFF), park[0], park.length);
  }
  return new Float32Array(lots);
}

/** Whether a point (local) is within `pad` m of a rectangle. */
function inRect(r: Rect, px: number, py: number, perM: number, pad: number) {
  const u = ((px - r.x) * r.ux + (py - r.y) * r.uy) / perM, v = (-(px - r.x) * r.uy + (py - r.y) * r.ux) / perM;
  return Math.abs(u) <= r.hl + pad && Math.abs(v) <= r.hd + pad;
}

/** Whether a segment (local) comes within `pad` m of a rectangle: clipped to it grown by `pad`. */
export function segHitsRect(x0: number, y0: number, x1: number, y1: number, r: Rect, perM: number, pad: number) {
  const tu = (x: number, y: number) => ((x - r.x) * r.ux + (y - r.y) * r.uy) / perM;
  const tv = (x: number, y: number) => (-(x - r.x) * r.uy + (y - r.y) * r.ux) / perM;
  const u0 = tu(x0, y0), v0 = tv(x0, y0), du = tu(x1, y1) - u0, dv = tv(x1, y1) - v0;
  const hu = r.hl + pad, hv = r.hd + pad;
  let lo = 0, hi = 1;
  for (const [p, q] of [[-du, u0 + hu], [du, hu - u0], [-dv, v0 + hv], [dv, hv - v0]]) {
    if (Math.abs(p) < 1e-12) { if (q < 0) return false; continue; }
    const t = q / p;
    if (p < 0) lo = Math.max(lo, t); else hi = Math.min(hi, t);
    if (lo > hi) return false;
  }
  return true;
}

/** Whether two rectangles come within `pad` m of each other (separating axes). */
function rectsMeet(a: Rect, b: Rect, perM: number, pad: number) {
  const dx = (b.x - a.x) / perM, dy = (b.y - a.y) / perM;
  for (const [ax, ay] of [[a.ux, a.uy], [-a.uy, a.ux], [b.ux, b.uy], [-b.uy, b.ux]]) {
    const ra = a.hl * Math.abs(a.ux * ax + a.uy * ay) + a.hd * Math.abs(-a.uy * ax + a.ux * ay);
    const rb = b.hl * Math.abs(b.ux * ax + b.uy * ay) + b.hd * Math.abs(-b.uy * ax + b.ux * ay);
    if (Math.abs(dx * ax + dy * ay) > ra + rb + pad) return false;
  }
  return true;
}

function distSeg(px: number, py: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
  return Math.hypot(x0 + t * dx - px, y0 + t * dy - py);
}
