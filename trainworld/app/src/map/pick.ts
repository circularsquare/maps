// What is under the pointer. The network is on our own canvas, so MapLibre's queryRenderedFeatures
// cannot see it; this projects the network with the overlay's camera. Within each layer the
// last drawn object containing the pointer wins, matching the visible stacking.
// Trains beat stations (a train is drawn over the station it stands at; Anita, T-085) beat
// lines beat junctions beat track (SPEC 8), blueprint track and its lines above built track and
// its lines (T-695). Clicking the line already
// selected selects the track under it instead. Strokes, trains and stations are first narrowed to
// those near the pointer in local units (T-180, T-724), so a pick costs well under a millisecond.

import { clock } from "../game/clock";
import type { Selection } from "../game/types";
import { F_BLUEPRINT, profileS, trainCarAt, tripAt, type NetworkBuffers } from "./network";
import type { Overlay } from "./overlay";
import { CellLevels } from "./cellIndex";

export const SLOP_PX = 7;
/** projected points, reused (topSeg runs over every segment on each pointer move) */
const SCRATCH = new Float64Array(6);

/**
 * The pointer in local units, and local units per CSS px with a quarter to spare (T-180), when the
 * camera has no perspective: local units are mercator, so screen position is then an affine
 * function of them, the same scale every way. Null with perspective (the map cannot be pitched
 * now) or no camera; `topSeg` then projects every segment.
 */
export function pointerLocal(ov: Overlay, px: number, py: number): [number, number, number] | null {
  const m = ov.camMatrix;
  if (!m) return null;
  // on the globe (T-033) the camera is affine only near a point: taken round the pointer's own
  // place, close enough for the narrowing within a few slops
  let x0 = 0, y0 = 0;
  if (ov.globe) {
    const at = ov.screenToLocal(px, py);
    if (!at) return null;
    [x0, y0] = at;
  } else if (m[3] !== 0 || m[7] !== 0) return null;
  const L = 1000, p = SCRATCH;
  ov.toScreenInto(x0, y0, p, 0);
  ov.toScreenInto(x0 + L, y0, p, 2);
  ov.toScreenInto(x0, y0 + L, p, 4);
  const a = (p[2] - p[0]) / L, d = (p[3] - p[1]) / L, b = (p[4] - p[0]) / L, e = (p[5] - p[1]) / L;
  const det = a * e - b * d;
  const pxPerUnit = Math.min(Math.hypot(a, d), Math.hypot(b, e));
  if (!det || !Number.isFinite(det) || !(pxPerUnit > 0)) return null;
  const sx = px - p[0], sy = py - p[1];
  return [x0 + (e * sx - b * sy) / det, y0 + (a * sy - d * sx) / det, 1.25 / pxPerUnit];
}


/** A stroke set's grid (T-724); until it is built in idle time, queries scan every segment. */
const grids = new WeakMap<Float32Array, { count: number; levels: CellLevels }>();
/** the candidate indices of the last `segsNear`, highest first */
let NEAR = new Int32Array(1024);
/** per stroke set, its last query and answer: a pointer move asks the same set at the same point
 * several times (the cursor's pick, the tag's pick, the tag's own segment; T-630) */
const lastNear = new WeakMap<Float32Array, { count: number; lx: number; ly: number; reach: number; m: number; idx: Int32Array }>();
/** scratch marks for ordering many candidates without a sort */
let MARK = new Uint8Array(0);
/** the widest drawn line of a width array (topSeg's reach), worked out once per array */
const widest = new WeakMap<Float32Array, number>();

/** T-1030: the grids built for the network drawn before an edit, carried over to the one after it
 * (posted as changes to it, `NetPatch`) instead of built again. */
export function carryGrids(was: NetworkBuffers | null, n: NetworkBuffers, check = false) {
  const p = n.patch;
  if (!was || !p || was.serial !== p.base) return;
  for (const [name, a, b, runs] of [["track", was.track, n.track, p.track], ["lines", was.lines, n.lines, p.lines]] as const) {
    const g = grids.get(a.seg);
    if (!g || g.count !== a.count || grids.has(b.seg)) continue;
    const seg = b.seg;
    const levels = CellLevels.carried(g.levels, runs, () => seg, b.count);
    grids.set(seg, { count: b.count, levels });
    const bad = check ? levels.verify() : [];
    if (bad.length) console.error(`T-1030 check: the ${name} pick grid carried over differs from one built anew at ${bad.join(", ")}`);
  }
}

/** Every segment whose box comes within `reach` (local units) of (lx, ly), into NEAR highest index
 * first, perhaps with others; their number, or -1 when the caller is to scan them all. */
function segsNear(seg: Float32Array, count: number, lx: number, ly: number, reach: number): number {
  const was = lastNear.get(seg);
  if (was && was.count === count && was.lx === lx && was.ly === ly && was.reach === reach) {
    if (NEAR.length < was.m) NEAR = new Int32Array(was.idx.length);
    NEAR.set(was.idx.subarray(0, was.m));
    return was.m;
  }
  let g = grids.get(seg);
  if (!g || g.count !== count) grids.set(seg, (g = { count, levels: new CellLevels(() => seg, count) }));
  const grid = g.levels.for(reach);
  if (!grid) return -1;
  let n = 0;
  grid.near(lx, ly, reach, (i) => {
    if (n === NEAR.length) {
      const more = new Int32Array(NEAR.length * 2);
      more.set(NEAR);
      NEAR = more;
    }
    NEAR[n++] = i;
  });
  let m = 0;
  if (n * 32 > count) {
    // zoomed far out a query holds a whole city (T-630): marking them and reading the marks back
    // from the top is one pass over the set, where a sort of that many costs milliseconds
    if (MARK.length < count) MARK = new Uint8Array(count);
    for (let k = 0; k < n; k++) MARK[NEAR[k]] = 1;
    for (let i = count - 1; i >= 0; i--) if (MARK[i]) (MARK[i] = 0), (NEAR[m++] = i);
  } else {
    const v = NEAR.subarray(0, n).sort().reverse();
    // a segment in two cells comes twice
    for (let k = 0; k < n; k++) if (!m || v[k] !== v[m - 1]) v[m++] = v[k];
  }
  lastNear.set(seg, { count, lx, ly, reach, m, idx: NEAR.slice(0, m) });
  return m;
}

/** Topmost segment of a stroke set within the slop: [index, squared px distance]. `offset` and
 * `slot`: segments shifted sideways as the renderer draws them (lines side by side, T-062).
 * Also used by the track tooltip (trackHover.ts, T-157) to find the segment `pick` hit.
 * `wide` and `widePx`: per segment, the drawn width as a multiple of `widePx` (line widths by
 * riders, T-227); a segment drawn wider than the slop is hit anywhere across it. */
export function topSeg(ov: Overlay, seg: Float32Array, count: number, px: number, py: number, best: number, offset?: Float32Array, slot = 0, accept?: (index: number) => boolean, wide?: Float32Array | null, widePx = 0, only?: Int32Array): [number, number] {
  // T-180: a segment whose box, grown by the slop and its sideways shift (and a pixel to spare),
  // misses the pointer cannot be within the slop, so it is skipped before the exact sum: first in
  // local units without projecting it (the pointer taken into local units once), then on screen.
  // T-724: and only the segments a grid cell near the pointer holds are looked at. Same answers.
  const slop = Math.sqrt(best);
  const p = SCRATCH;
  const at = pointerLocal(ov, px, py);
  // off the globe nothing is under the pointer
  if (!at && ov.globe) return [-1, best];
  const lx = at ? at[0] : 0, ly = at ? at[1] : 0, perPx = at ? at[2] : 0;
  let near: Int32Array | null = null, nearCount = 0;
  // `only`: the caller's own candidates, highest first (the blueprint's few segments, T-630)
  if (only) (near = only), (nearCount = only.length);
  else if (at && !offset) {
    let maxWide = 0;
    if (wide && widePx) {
      const had = widest.get(wide);
      if (had !== undefined) maxWide = had;
      else {
        for (let i = 0; i < wide.length; i++) if (wide[i] > maxWide) maxWide = wide[i];
        widest.set(wide, maxWide);
      }
    }
    const reachMax = (Math.max(slop, 0.5 * maxWide * widePx) + 1) * perPx;
    nearCount = segsNear(seg, count, lx, ly, reachMax);
    if (nearCount >= 0) near = NEAR;
  }
  for (let k = near ? 0 : count - 1, end = near ? nearCount : -1; k !== end; k += near ? 1 : -1) {
    const i = near ? near[k] : k;
    if (accept && !accept(i)) continue;
    // a line drawn wider than the slop is hit anywhere on it (T-227)
    const half = wide && widePx ? Math.max(slop, 0.5 * wide[i] * widePx) : slop;
    const reach = half + 1;
    if (at) {
      const rl = (reach + Math.abs(offset ? offset[i] * slot : 0)) * perPx;
      const ax = seg[i * 4], ay = seg[i * 4 + 1], bx = seg[i * 4 + 2], by = seg[i * 4 + 3];
      if ((ax > lx + rl && bx > lx + rl) || (ax < lx - rl && bx < lx - rl) || (ay > ly + rl && by > ly + rl) || (ay < ly - rl && by < ly - rl)) continue;
    }
    ov.toScreenInto(seg[i * 4], seg[i * 4 + 1], p, 0);
    ov.toScreenInto(seg[i * 4 + 2], seg[i * 4 + 3], p, 2);
    let x0 = p[0], y0 = p[1], x1 = p[2], y1 = p[3];
    const o = offset ? offset[i] * slot : 0;
    const r = reach + Math.abs(o);
    if ((x0 > px + r && x1 > px + r) || (x0 < px - r && x1 < px - r) || (y0 > py + r && y1 > py + r) || (y0 < py - r && y1 < py - r)) continue;
    let dx = x1 - x0, dy = y1 - y0;
    if (o) {
      // the renderer's normal, in screen px with y down: (dy, -dx) for a unit direction
      const l = Math.hypot(dx, dy) || 1;
      const ox = (dy / l) * o, oy = (-dx / l) * o;
      x0 += ox, y0 += oy, x1 += ox, y1 += oy;
      dx = x1 - x0, dy = y1 - y0;
    }
    const l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
    const d = (x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2;
    if (d < Math.max(best, half * half)) {
      return [i, d];
    }
  }
  return [-1, best];
}

/** Up to `max` different lines with a stroke under the pointer, in `pick`'s order: the selected
 * line, blueprint lines, then the rest, each top first (the resting hover's list, T-732). */
export function linesAt(ov: Overlay, px: number, py: number, current: Selection, max: number): string[] {
  const n = ov.renderer.net;
  if (!n || !ov.camMatrix || !ov.renderer.slotPx) return [];
  const reach = Math.max(SLOP_PX, ov.renderer.laneReachPx);
  const slop = reach * reach + 1;
  const L = n.lines, wide = ov.renderer.lineWidths, widePx = ov.renderer.slotPx;
  const tiers: ((i: number) => boolean)[] = [];
  if (current?.kind === "line") tiers.push((i) => String(L.colour[i]) === current.line);
  if (ov.renderer.hasBlueprint) tiers.push((i) => (L.flags[i] & F_BLUEPRINT) !== 0);
  tiers.push(() => true);
  const out: string[] = [];
  for (const tier of tiers) {
    while (out.length < max) {
      const [i] = topSeg(ov, L.seg, L.count, px, py, slop, undefined, 0, (j) => tier(j) && !out.includes(String(L.colour[j])), wide, widePx);
      if (i < 0) break;
      out.push(String(L.colour[i]));
    }
  }
  return out;
}

/** Each line's samples in blocks of BLOCK stretches (T-724): every block's box, local units, and
 * the first block of each line; `mark` is the scratch of `markNearBlocks`. */
interface LineBlocks {
  first: Int32Array;
  box: Float32Array;
  mark: Uint8Array;
}
const BLOCK = 16;
const blocksOf = new WeakMap<NetworkBuffers, LineBlocks>();

/** A network's line blocks, built when first asked for (T-1020: a scan of every trip until an idle
 * build was the slower path on a big network). */
function lineBlocks(n: NetworkBuffers): LineBlocks {
  let lb = blocksOf.get(n);
  if (!lb) blocksOf.set(n, (lb = buildBlocks(n)));
  return lb;
}

function buildBlocks(n: NetworkBuffers): LineBlocks {
  const T = n.lineTable, S = n.samples, lines = T.length / 4;
  const blocks = (count: number) => Math.max(1, Math.ceil((count - 1) / BLOCK));
  let total = 0;
  for (let l = 0; l < lines; l++) total += blocks(Math.max(1, T[l * 4 + 1]));
  const first = new Int32Array(lines + 1), box = new Float32Array(total * 4);
  let at = 0;
  for (let l = 0; l < lines; l++) {
    first[l] = at;
    const s0 = T[l * 4], count = Math.max(1, T[l * 4 + 1]);
    for (let b = 0; b === 0 || b * BLOCK < count - 1; b++) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let j = b * BLOCK; j <= Math.min(count - 1, (b + 1) * BLOCK); j++) {
        const x = S[(s0 + j) * 2], y = S[(s0 + j) * 2 + 1];
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
      box[at * 4] = x0, box[at * 4 + 1] = y0, box[at * 4 + 2] = x1, box[at * 4 + 3] = y1;
      at++;
    }
  }
  first[lines] = at;
  return { first, box, mark: new Uint8Array(at) };
}

/** Marks the blocks whose box comes within `reach` (local units) of (lx, ly). */
function markNearBlocks(n: NetworkBuffers, lx: number, ly: number, reach: number): LineBlocks {
  const lb = lineBlocks(n);
  const b = lb.box, m = lb.mark;
  for (let k = 0; k < m.length; k++)
    m[k] = b[k * 4] - reach <= lx && b[k * 4 + 2] + reach >= lx && b[k * 4 + 1] - reach <= ly && b[k * 4 + 3] + reach >= ly ? 1 : 0;
  return lb;
}

/** Whether any point `tripAt` and `trainCarAt` could give for the trip at time `t` lies in a marked
 * block: its offset along the line, give or take a train's length (cars at 20 m) and the 5 m and
 * half-car look-ups beside it. False only when neither can hit. */
function tripMayBeNear(n: NetworkBuffers, lb: LineBlocks, profile: number, dep: number, t: number): boolean {
  const tau = t - dep;
  if (tau < 0 || !n.meta[profile * 4 + 1]) return false;
  const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
  if (line * 4 + 3 >= n.lineTable.length) return true;
  const s = profileS(n, profile, tau), plen = n.meta[profile * 4 + 3];
  const d = run === 0 ? s : plen - s;
  const count = n.lineTable[line * 4 + 1], len = n.lineTable[line * 4 + 2], step = n.lineTable[line * 4 + 3];
  const f = lb.first[line], nb = lb.first[line + 1] - f;
  if (!(step > 0) || count < 2) return lb.mark[f] === 1;
  const margin = Math.max(1, n.lineCars?.[line] ?? 8) * 20 + 30;
  const blockAt = (dd: number) => Math.min(nb - 1, Math.floor(Math.min(Math.floor(Math.max(0, Math.min(len, dd)) / step), count - 2) / BLOCK));
  for (let b = blockAt(d - margin), hi = blockAt(d + margin); b <= hi; b++) if (lb.mark[f + b]) return true;
  return false;
}

const blueprints = new WeakMap<Float32Array, Int32Array>();
/** A stroke set's blueprint segments, highest first: a handful beside the built network's tens of
 * thousands, which a pick far out would otherwise all look at (T-630). */
function blueprintOf(s: { flags: Float32Array; count: number }): Int32Array {
  let had = blueprints.get(s.flags);
  if (had) return had;
  const out: number[] = [];
  for (let i = s.count - 1; i >= 0; i--) if ((s.flags[i] & F_BLUEPRINT) !== 0) out.push(i);
  blueprints.set(s.flags, (had = Int32Array.from(out)));
  return had;
}

/** How many different lines have strokes on an edge. */
function linesOnEdge(s: { colour: Uint32Array; edge: Uint32Array; count: number }, edge: number): number {
  let first = -1;
  for (let i = 0; i < s.count; i++) {
    if (s.edge[i] !== edge) continue;
    if (first < 0) first = s.colour[i];
    else if (s.colour[i] !== first) return 2;
  }
  return 1;
}

/** `chooseShared`: a click (not a hover) on track several lines share picks the track, whose
 * inspector lists the lines, instead of the line drawn on top (T-654). */
export function pick(ov: Overlay, px: number, py: number, trainsShown: boolean, current: Selection = null, chooseShared = false): Selection {
  const n = ov.renderer.net;
  if (!n || !ov.camMatrix) return null;
  const best = SLOP_PX * SLOP_PX + 1;
  // close in, either of double track's two tracks (T-658)
  const reach = Math.max(SLOP_PX, ov.renderer.laneReachPx);
  const slop = reach * reach + 1;

  // trains are not drawn zoomed far out (T-205)
  if (trainsShown && !ov.renderer.trainsHidden) {
    const t = clock.now();
    const r = ov.renderer;
    const trips = r.trips;
    // the selected train is drawn over the others (T-242), so it is tried first
    const first = current?.kind === "train" ? r.tripIndex(current.profile, current.dep - r.tripsEpoch) : -1;
    const at = pointerLocal(ov, px, py);
    // T-724: only trips on a stretch of line near the pointer are placed (same answers)
    const marked = at ? markNearBlocks(n, at[0], at[1], (Math.sqrt(best) + 1) * at[2] + 3 * r.laneShiftM + 1) : null;
    for (let k = r.tripCount - (first >= 0 ? 0 : 1); k >= 0; k--) {
      const i = k === r.tripCount ? first : k;
      if (k < r.tripCount && i === first) continue;
      const prof = trips[i * 2], dep = trips[i * 2 + 1] + r.tripsEpoch;
      if (marked && !tripMayBeNear(n, marked, prof, dep, t)) continue;
      const at = tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, r.laneShiftM);
      if (!at) continue;
      if (r.detailedTrains) {
        const cars = Math.max(1, n.lineCars?.[at.line] ?? 8);
        for (let car = 0; car < cars; car++) {
          const p = trainCarAt(n, at, car);
          const [x0, y0] = ov.toScreen(...p.back), [x1, y1] = ov.toScreen(...p.front);
          const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
          const u = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (len * len)));
          const d = (x0 + u * dx - px) ** 2 + (y0 + u * dy - py) ** 2;
          if (d < best) {
            return { kind: "train", line: String(at.line), profile: prof, dep };
          }
        }
        continue;
      }
      const [x, y] = ov.toScreen(at.x, at.y);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        return { kind: "train", line: String(at.line), profile: prof, dep };
      }
    }
  }

  const station=ov.renderer.stationAt(px,py,(x,y)=>ov.toScreen(x,y),SLOP_PX);
  if (station>=0) return {kind:"station",station:n.stationIds[station]};

  // Height/speed views do not draw line strokes; their hidden line buffers cannot win a hit.
  // A selected line has a final highlight pass above the other line strokes. Preserve the
  // existing repeat-click-to-track interaction even where that pass crosses another line.
  // lines wider than the slop (traffic thickness, T-227) are hit across their drawn width
  const wide = ov.renderer.lineWidths, widePx = ov.renderer.slotPx;
  const [selectedLi] = ov.renderer.slotPx && current?.kind === "line"
    ? topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, undefined, 0, (i) => String(n.lines.colour[i]) === current.line, wide, widePx)
    : [-1];
  // blueprint track and the lines planned on it are drawn over built track and its lines (T-695)
  const bp = ov.renderer.hasBlueprint;
  let li = selectedLi;
  if (li < 0 && bp && ov.renderer.slotPx) [li] = topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, undefined, 0, undefined, wide, widePx, blueprintOf(n.lines));
  if (li < 0 && bp) {
    const [ti] = topSeg(ov, n.track.seg, n.track.count, px, py, slop, undefined, 0, undefined, null, 0, blueprintOf(n.track));
    if (ti >= 0) return { kind: "track", edge: n.track.edge[ti] };
  }
  if (li < 0 && ov.renderer.slotPx) [li] = topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, undefined, 0, undefined, wide, widePx);
  const lineHit = li >= 0 ? String(n.lines.colour[li]) : null;
  if (lineHit && !(current?.kind === "line" && current.line === lineHit)) {
    // track several lines run over opens the track's list of them rather than the line on top (T-654)
    if (chooseShared && linesOnEdge(n.lines, n.lines.edge[li]) > 1) return { kind: "track", edge: n.lines.edge[li] };
    return { kind: "line", line: lineHit };
  }

  for (let i = n.junctionIds.length - 1; i >= 0; i--) {
    ov.toScreenInto(n.junctions[i * 2], n.junctions[i * 2 + 1], SCRATCH, 0);
    const d = (SCRATCH[0] - px) ** 2 + (SCRATCH[1] - py) ** 2;
    if (d < best) {
      return { kind: "junction", node: n.junctionIds[i] };
    }
  }

  const [ti] = topSeg(ov, n.track.seg, n.track.count, px, py, slop);
  if (ti >= 0) return { kind: "track", edge: n.track.edge[ti] };
  return lineHit ? { kind: "line", line: lineHit } : null;
}
