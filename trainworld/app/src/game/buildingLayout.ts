// The inside of a station building as drawn (T-2219, notes/T-2208.md 3 "The drawing"): the little shops
// on the floor left over, the cross-hall that holds the name, the free floor the hall's crowd stands on,
// and the hall's room. Pure geometry in any units with `perM` of them to a metre (the clock worker's
// local units for the map, pack units for the panel); the work is done in the building's own frame, in
// metres, along its first edge.

/** A shop's cell along the platforms and across them, the gap between cells, and the clear cross-hall
 * through the middle, m. A shop is one cell, or two along the platforms as one long shop. */
export const SHOP_L_M = 13, SHOP_D_M = 15, SHOP_GAP_M = 4, HALL_M = 28;
/** The share of cells with shops in them, and of shops two cells long. */
const SHOP_SHARE = 0.85, LONG_SHARE = 0.4;
/** Shops keep this clear of the outline, of a platform's drawn band and of a line's drawn stroke, m. */
const CLEAR_EDGE_M = 3, CLEAR_PLATFORM_M = 4, CLEAR_LINE_M = 3;
/** Half a platform's drawn box on the ground from zoom 15 (game/complexes.ts BOX_TOUCH_M), and half a
 * line's drawn stroke there about zoom 16.5, m. */
export const PLATFORM_HALF_M = 8;
const LINE_HALF_M = 7;
/** The hall's crowd stands on free floor in cells this big, m. */
export const HALL_CELL_M = 4;
/** One waiting rider to this much of the floor not under its own platforms, m² (half the floor shops,
 * stairs and gates, the rest at 1 m² a rider). */
const M2_PER_RIDER = 2;
/** The room is counted on a grid this fine, m. */
const ROOM_GRID_M = 2.5;
/** Shop colours (T-2208: peach, cream, sage, sky, rose, sand, yellow), and their rims 16% darker. */
export const SHOP_COLOURS = ["#f6d6c4", "#f2e9d0", "#d6e3cc", "#cfe0ee", "#efd4dd", "#e4d9c4", "#f5e4a8"];
export const SHOP_GLYPHS = 4;

export interface Rect { x: number; y: number; ux: number; uy: number; hl: number; hd: number }

export interface LayoutInput {
  /** the outline, x, y pairs */
  corners: ArrayLike<number>;
  perM: number;
  /** each platform's centre line (x, y pairs) and whether it is a member's */
  platforms: { line: ArrayLike<number>; member: number }[];
  /** the track's centre lines through and near the building, x, y pairs */
  tracks: ArrayLike<number>[];
  /** rectangles the shops leave room for (a waiting room on the floor), half sizes in m */
  clear: Rect[];
  /** picks the shops, so they never shuffle */
  seed: number;
  shops: boolean;
}

export interface Layout {
  /** along the first edge, and across it into the floor (units, unit length) */
  ux: number; uy: number; vx: number; vy: number;
  /** per shop 6 numbers: centre x, y, half its length along the first edge and half its depth (m),
   * colour (SHOP_COLOURS), glyph */
  shops: number[];
  /** per free floor cell 4 numbers: centre x, y, the nearest member's `member`, its distance m;
   * nearest first */
  cells: number[];
  /** the middle of the cross-hall, and the building's short side, m */
  hall: [number, number];
  shortM: number;
  room: number;
}

/** The building's frame: its first edge's way, across it towards the floor, and the outline in it, m. */
export function buildingFrame(c: ArrayLike<number>, perM: number) {
  const n = c.length / 2, x0 = c[0], y0 = c[1];
  let ux = c[2] - x0, uy = c[3] - y0;
  const ul = Math.hypot(ux, uy) || 1;
  ux /= ul; uy /= ul;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += c[2 * i] / n; my += c[2 * i + 1] / n; }
  const side = (mx - x0) * -uy + (my - y0) * ux >= 0 ? 1 : -1;
  const vx = -uy * side, vy = ux * side;
  const toU = (x: number, y: number) => ((x - x0) * ux + (y - y0) * uy) / perM;
  const toV = (x: number, y: number) => ((x - x0) * vx + (y - y0) * vy) / perM;
  const pts: number[] = [];
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const u = toU(c[2 * i], c[2 * i + 1]), v = toV(c[2 * i], c[2 * i + 1]);
    pts.push(u, v);
    u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
  }
  const back = (u: number, v: number): [number, number] => [x0 + (ux * u + vx * v) * perM, y0 + (uy * u + vy * v) * perM];
  return { ux, uy, vx, vy, toU, toV, back, pts, u0, u1, v0, v1 };
}

/** The riders the hall holds: one to M2_PER_RIDER of the floor not under its members' platforms. */
export function hallRoom(corners: ArrayLike<number>, perM: number, members: ArrayLike<number>[]): number {
  const f = buildingFrame(corners, perM);
  const lines = members.map((l) => toFrame(l, f));
  let free = 0;
  for (let u = f.u0 + ROOM_GRID_M / 2; u < f.u1; u += ROOM_GRID_M)
    for (let v = f.v0 + ROOM_GRID_M / 2; v < f.v1; v += ROOM_GRID_M) {
      if (!inside(f.pts, u, v)) continue;
      if (!lines.some((l) => lineGap(l, u, v) < PLATFORM_HALF_M)) free++;
    }
  return Math.round((free * ROOM_GRID_M * ROOM_GRID_M) / M2_PER_RIDER);
}

export function layoutBuilding(inp: LayoutInput): Layout {
  const f = buildingFrame(inp.corners, inp.perM);
  const plats = inp.platforms.map((p) => ({ l: toFrame(p.line, f), member: p.member }));
  const tracks = inp.tracks.map((l) => toFrame(l, f));
  const edges = f.pts;
  const clear = inp.clear.map((r) => {
    const cu = f.toU(r.x, r.y), cv = f.toV(r.x, r.y);
    // the rectangle's corners in the frame, as a box round them
    const au = r.ux * f.ux + r.uy * f.uy, av = r.ux * f.vx + r.uy * f.vy;
    const eu = Math.abs(au) * r.hl + Math.abs(av) * r.hd, ev = Math.abs(av) * r.hl + Math.abs(au) * r.hd;
    return [cu - eu, cv - ev, cu + eu, cv + ev];
  });
  const longU = f.u1 - f.u0 >= f.v1 - f.v0;
  const um = (f.u0 + f.u1) / 2, vm = (f.v0 + f.v1) / 2;
  const [hx, hy] = f.back(um, vm);
  const shortM = Math.min(f.u1 - f.u0, f.v1 - f.v0);

  // the shops: cells on a grid lined up with the first edge, long along the platforms, outward from the
  // cross-hall on the building's long axis
  const shops: number[] = [], shopBoxes: number[][] = [];
  if (inp.shops) {
    // the platforms' way in the frame, by length
    let su = 0, sv = 0;
    for (const p of plats) if (p.member >= 0) for (let i = 0; i + 3 < p.l.length; i += 2) { su += Math.abs(p.l[i + 2] - p.l[i]); sv += Math.abs(p.l[i + 3] - p.l[i + 1]); }
    const alongU = su >= sv;
    const [lu, lv] = alongU ? [SHOP_L_M, SHOP_D_M] : [SHOP_D_M, SHOP_L_M];
    const starts = (lo: number, hi: number, mid: number, size: number, hall: boolean) => {
      const out: number[] = [], pitch = size + SHOP_GAP_M;
      if (hall) {
        for (let a = mid - HALL_M / 2 - size; a >= lo; a -= pitch) out.unshift(a);
        for (let a = mid + HALL_M / 2; a + size <= hi; a += pitch) out.push(a);
      } else for (let a = lo + CLEAR_EDGE_M; a + size <= hi; a += pitch) out.push(a);
      return out;
    };
    const us = starts(f.u0, f.u1, um, lu, longU), vs = starts(f.v0, f.v1, vm, lv, !longU);
    const ok = (box: number[]) => boxInside(edges, box, CLEAR_EDGE_M)
      && !plats.some((p) => boxGap(p.l, box) < PLATFORM_HALF_M + CLEAR_PLATFORM_M)
      && !tracks.some((l) => boxGap(l, box) < LINE_HALF_M + CLEAR_LINE_M)
      && !clear.some((r) => r[0] < box[2] && r[2] > box[0] && r[1] < box[3] && r[3] > box[1]);
    // rows across the platforms, walked along them
    const [rows, along] = alongU ? [vs, us] : [us, vs];
    for (const r of rows) {
      for (let k = 0; k < along.length; k++) {
        const a = along[k], key = [Math.round(a * 10), Math.round(r * 10)] as const;
        if (hash(inp.seed, ...key) >= SHOP_SHARE) continue;
        const cell = (t: number) => (alongU ? [t, r, t + lu, r + lv] : [r, t, r + lu, t + lv]);
        let box = cell(a);
        if (!ok(box)) continue;
        // a long shop takes the next cell too, when it follows on
        const next = along[k + 1];
        if (next !== undefined && Math.abs(next - a - (alongU ? lu : lv) - SHOP_GAP_M) < 1e-6 && hash(inp.seed + 7, ...key) < LONG_SHARE && ok(cell(next))) {
          const b2 = cell(next);
          box = [box[0], box[1], b2[2], b2[3]];
          k++;
        }
        shopBoxes.push(box);
        const [x, y] = f.back((box[0] + box[2]) / 2, (box[1] + box[3]) / 2);
        shops.push(x, y, (box[2] - box[0]) / 2, (box[3] - box[1]) / 2, Math.floor(hash(inp.seed + 13, ...key) * SHOP_COLOURS.length) % SHOP_COLOURS.length, Math.floor(hash(inp.seed + 29, ...key) * SHOP_GLYPHS) % SHOP_GLYPHS);
      }
    }
  }

  // the free floor, nearest a member's platform first
  const cells: [number, number, number, number][] = [];
  const members = plats.filter((p) => p.member >= 0);
  if (members.length) {
    const s = HALL_CELL_M;
    for (let u = f.u0 + s / 2; u < f.u1; u += s)
      for (let v = f.v0 + s / 2; v < f.v1; v += s) {
        if (!inside(edges, u, v) || edgeGap(edges, u, v) < s / 2) continue;
        if (shopBoxes.some((b) => u > b[0] - 1 && u < b[2] + 1 && v > b[1] - 1 && v < b[3] + 1)) continue;
        if (plats.some((p) => lineGap(p.l, u, v) < PLATFORM_HALF_M + 1)) continue;
        if (tracks.some((l) => lineGap(l, u, v) < LINE_HALF_M + 1)) continue;
        let best = -1, d = Infinity;
        for (const p of members) {
          const g = lineGap(p.l, u, v);
          if (g < d) { d = g; best = p.member; }
        }
        const [x, y] = f.back(u, v);
        cells.push([x, y, best, d - PLATFORM_HALF_M]);
      }
    cells.sort((a, b) => a[3] - b[3]);
  }
  return {
    ux: f.ux, uy: f.uy, vx: f.vx, vy: f.vy,
    shops,
    cells: cells.flat(),
    hall: [hx, hy],
    shortM,
    room: hallRoom(inp.corners, inp.perM, inp.platforms.filter((p) => p.member >= 0).map((p) => p.line)),
  };
}

// ---------------------------------------------------------------- geometry in the frame, m

function toFrame(l: ArrayLike<number>, f: ReturnType<typeof buildingFrame>): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < l.length; i += 2) out.push(f.toU(l[i], l[i + 1]), f.toV(l[i], l[i + 1]));
  return out;
}

function inside(c: ArrayLike<number>, x: number, y: number): boolean {
  const n = c.length / 2;
  let odd = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const ax = c[2 * i], ay = c[2 * i + 1], bx = c[2 * j], by = c[2 * j + 1];
    if ((ay > y) !== (by > y) && x < ax + ((bx - ax) * (y - ay)) / (by - ay)) odd = !odd;
  }
  return odd;
}

function segGap(px: number, py: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
  return Math.hypot(x0 + t * dx - px, y0 + t * dy - py);
}

/** A point's distance from a polyline (a single point counts). */
function lineGap(l: number[], x: number, y: number): number {
  if (l.length < 4) return l.length >= 2 ? Math.hypot(l[0] - x, l[1] - y) : Infinity;
  let d = Infinity;
  for (let i = 0; i + 3 < l.length; i += 2) d = Math.min(d, segGap(x, y, l[i], l[i + 1], l[i + 2], l[i + 3]));
  return d;
}

/** A point's distance from the outline's edges. */
function edgeGap(c: number[], x: number, y: number): number {
  const n = c.length / 2;
  let d = Infinity;
  for (let i = 0, j = n - 1; i < n; j = i++) d = Math.min(d, segGap(x, y, c[2 * j], c[2 * j + 1], c[2 * i], c[2 * i + 1]));
  return d;
}

/** A segment's distance from a box [u0, v0, u1, v1]: 0 where it crosses it. */
function segBoxGap(x0: number, y0: number, x1: number, y1: number, b: number[]): number {
  // clipped to the box: crossing it (Liang-Barsky)
  let lo = 0, hi = 1, hits = true;
  const dx = x1 - x0, dy = y1 - y0;
  for (const [p, q] of [[-dx, x0 - b[0]], [dx, b[2] - x0], [-dy, y0 - b[1]], [dy, b[3] - y0]]) {
    if (Math.abs(p) < 1e-12) { if (q < 0) { hits = false; break; } continue; }
    const t = q / p;
    if (p < 0) lo = Math.max(lo, t); else hi = Math.min(hi, t);
    if (lo > hi) { hits = false; break; }
  }
  if (hits) return 0;
  const box = (x: number, y: number) => Math.hypot(Math.max(b[0] - x, 0, x - b[2]), Math.max(b[1] - y, 0, y - b[3]));
  return Math.min(box(x0, y0), box(x1, y1), segGap(b[0], b[1], x0, y0, x1, y1), segGap(b[2], b[1], x0, y0, x1, y1), segGap(b[0], b[3], x0, y0, x1, y1), segGap(b[2], b[3], x0, y0, x1, y1));
}

function boxGap(l: number[], b: number[]): number {
  if (l.length < 4) return l.length >= 2 ? segBoxGap(l[0], l[1], l[0], l[1], b) : Infinity;
  let d = Infinity;
  for (let i = 0; i + 3 < l.length && d > 0; i += 2) d = Math.min(d, segBoxGap(l[i], l[i + 1], l[i + 2], l[i + 3], b));
  return d;
}

/** The box lies inside the outline, at least `pad` from its edges. */
function boxInside(c: number[], b: number[], pad: number): boolean {
  if (!inside(c, b[0], b[1]) || !inside(c, b[2], b[1]) || !inside(c, b[0], b[3]) || !inside(c, b[2], b[3])) return false;
  const n = c.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) if (segBoxGap(c[2 * j], c[2 * j + 1], c[2 * i], c[2 * i + 1], b) < pad) return false;
  return true;
}

/** 0-1 from three integers, the same every time. */
function hash(a: number, b: number, c: number): number {
  let h = (Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul(b | 0, 0x165667b1) ^ Math.imul(c | 0, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}
