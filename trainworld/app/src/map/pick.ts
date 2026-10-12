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

const SLOP_PX = 7;

/** Topmost segment of a stroke set within the slop: [index, squared px distance]. `offset` and
 * `slot`: segments shifted sideways as the renderer draws them (lines side by side, T-062). */
function topSeg(ov: Overlay, seg: Float32Array, count: number, px: number, py: number, best: number, offset?: Float32Array, slot = 0, accept?: (index: number) => boolean): [number, number] {
  for (let i = count - 1; i >= 0; i--) {
    if (accept && !accept(i)) continue;
    let [x0, y0] = ov.toScreen(seg[i * 4], seg[i * 4 + 1]);
    let [x1, y1] = ov.toScreen(seg[i * 4 + 2], seg[i * 4 + 3]);
    let dx = x1 - x0, dy = y1 - y0;
    const o = offset ? offset[i] * slot : 0;
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
    if (d < best) {
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

  if (trainsShown) {
    const t = clock.now();
    const r = ov.renderer;
    const trips = r.trips;
    for (let i = r.tripCount - 1; i >= 0; i--) {
      const prof = trips[i * 2], dep = trips[i * 2 + 1] + r.tripsEpoch;
      const at = tripAt(n, prof, dep, t);
      if (!at) continue;
      if (r.detailedTrains) {
        const cars = Math.max(1, n.lineCars?.[at.line] ?? 8);
        for (let car = 0; car < cars; car++) {
          const p = trainCarAt(n, at, car);
          let [x0, y0] = ov.toScreen(...p.back), [x1, y1] = ov.toScreen(...p.front);
          const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
          const o = p.off * r.slotPx;
          x0 += dy / len * o; y0 -= dx / len * o;
          x1 += dy / len * o; y1 -= dx / len * o;
          const u = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (len * len)));
          const d = (x0 + u * dx - px) ** 2 + (y0 + u * dy - py) ** 2;
          if (d < best) {
            return { kind: "train", line: String(at.line), profile: prof, dep };
          }
        }
        continue;
      }
      let [x, y] = ov.toScreen(at.x, at.y);
      const o = at.off * r.slotPx;
      if (o) {
        const [x1, y1] = ov.toScreen(at.x + at.dx, at.y + at.dy);
        const l = Math.hypot(x1 - x, y1 - y) || 1;
        const ox = ((y1 - y) / l) * o, oy = (-(x1 - x) / l) * o;
        x += ox;
        y += oy;
      }
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
  const [selectedLi] = ov.renderer.slotPx && current?.kind === "line"
    ? topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, n.lines.offset, ov.renderer.slotPx, (i) => String(n.lines.colour[i]) === current.line)
    : [-1];
  const [li] = selectedLi >= 0 ? [selectedLi] : ov.renderer.slotPx ? topSeg(ov, n.lines.seg, n.lines.count, px, py, slop, n.lines.offset, ov.renderer.slotPx) : [-1];
  const lineHit = li >= 0 ? String(n.lines.colour[li]) : null;
  if (lineHit && !(current?.kind === "line" && current.line === lineHit)) return { kind: "line", line: lineHit };

  for (let i = n.junctionIds.length - 1; i >= 0; i--) {
    const [x, y] = ov.toScreen(n.junctions[i * 2], n.junctions[i * 2 + 1]);
    const d = (x - px) ** 2 + (y - py) ** 2;
    if (d < best) {
      return { kind: "junction", node: n.junctionIds[i] };
    }
  }

  const [ti] = topSeg(ov, n.track.seg, n.track.count, px, py, slop);
  if (ti >= 0) return { kind: "track", edge: n.track.edge[ti] };
  return lineHit ? { kind: "line", line: lineHit } : null;
}
