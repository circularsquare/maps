// What is under the pointer. The network is on our own canvas, so MapLibre's queryRenderedFeatures
// cannot see it; this projects the network with the overlay's camera. Within each layer the
// last drawn object containing the pointer wins, matching the visible stacking.
// Trains beat stations (a train is drawn over the station it stands at; Anita, T-085) beat
// lines beat junctions beat track (SPEC 8). Clicking the line already
// selected selects the track under it instead. Linear scans: fine for a city.

import { clock } from "../game/clock";
import type { Selection } from "../game/types";
import { trainCarAt, tripAt } from "./network";
import type { Overlay } from "./overlay";

export const SLOP_PX = 7;
/** projected points, reused (topSeg runs over every segment on each pointer move) */
const SCRATCH = new Float64Array(6);

/**
 * The pointer in local units, and local units per CSS px with a quarter to spare (T-180), when the
 * camera has no perspective: local units are mercator, so screen position is then an affine
 * function of them, the same scale every way. Null with perspective (the map cannot be pitched
 * now) or no camera; `topSeg` then projects every segment.
 */
function pointerLocal(ov: Overlay, px: number, py: number): [number, number, number] | null {
  const m = ov.camMatrix;
  if (!m || m[3] !== 0 || m[7] !== 0) return null;
  const L = 1000, p = SCRATCH;
  ov.toScreenInto(0, 0, p, 0);
  ov.toScreenInto(L, 0, p, 2);
  ov.toScreenInto(0, L, p, 4);
  const a = (p[2] - p[0]) / L, d = (p[3] - p[1]) / L, b = (p[4] - p[0]) / L, e = (p[5] - p[1]) / L;
  const det = a * e - b * d;
  const pxPerUnit = Math.min(Math.hypot(a, d), Math.hypot(b, e));
  if (!det || !Number.isFinite(det) || !(pxPerUnit > 0)) return null;
  const sx = px - p[0], sy = py - p[1];
  return [(e * sx - b * sy) / det, (a * sy - d * sx) / det, 1.25 / pxPerUnit];
}

/** Topmost segment of a stroke set within the slop: [index, squared px distance]. `offset` and
 * `slot`: segments shifted sideways as the renderer draws them (lines side by side, T-062).
 * Also used by the track tooltip (trackHover.ts, T-157) to find the segment `pick` hit.
 * `wide` and `widePx`: per segment, the drawn width as a multiple of `widePx` (line widths by
 * riders, T-227); a segment drawn wider than the slop is hit anywhere across it. */
export function topSeg(ov: Overlay, seg: Float32Array, count: number, px: number, py: number, best: number, offset?: Float32Array, slot = 0, accept?: (index: number) => boolean, wide?: Float32Array | null, widePx = 0): [number, number] {
  // T-180: a segment whose box, grown by the slop and its sideways shift (and a pixel to spare),
  // misses the pointer cannot be within the slop, so it is skipped before the exact sum: first in
  // local units without projecting it (the pointer taken into local units once), then on screen.
  // Hovering the real New York network cost about 25 ms a pointer move before. Same answers.
  const slop = Math.sqrt(best);
  const p = SCRATCH;
  const at = pointerLocal(ov, px, py);
  const lx = at ? at[0] : 0, ly = at ? at[1] : 0, perPx = at ? at[2] : 0;
  for (let i = count - 1; i >= 0; i--) {
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

export function pick(ov: Overlay, px: number, py: number, trainsShown: boolean, current: Selection = null): Selection {
  const n = ov.renderer.net;
  if (!n || !ov.camMatrix) return null;
  const slop = SLOP_PX * SLOP_PX + 1;
  const best = slop;

  // trains are not drawn zoomed far out (T-205)
  if (trainsShown && !ov.renderer.trainsHidden) {
    const t = clock.now();
    const r = ov.renderer;
    const trips = r.trips;
    // the selected train is drawn over the others (T-242), so it is tried first
    const first = current?.kind === "train" ? r.tripIndex(current.profile, current.dep - r.tripsEpoch) : -1;
    for (let k = r.tripCount - (first >= 0 ? 0 : 1); k >= 0; k--) {
      const i = k === r.tripCount ? first : k;
      if (k < r.tripCount && i === first) continue;
      const prof = trips[i * 2], dep = trips[i * 2 + 1] + r.tripsEpoch;
      const at = tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
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
  const [li] = selectedLi >= 0 ? [selectedLi] : ov.renderer.slotPx ? topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, undefined, 0, undefined, wide, widePx) : [-1];
  const lineHit = li >= 0 ? String(n.lines.colour[li]) : null;
  if (lineHit && !(current?.kind === "line" && current.line === lineHit)) return { kind: "line", line: lineHit };

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
