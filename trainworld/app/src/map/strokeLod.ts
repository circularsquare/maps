// Fewer stroke instances per draw (T-1639, notes/T-1618.md): zoomed out, each network stroke set is
// drawn as a simplified copy whose runs of segments are merged where the merged segment strays from
// them by less than a fraction of a pixel; closer in, only the chunks whose box touches the view are
// drawn. Both keep the segments' order, which decides which line is drawn over which (T-190).

import { BAND_STRIDE } from "./lineBands";
import { F_EASE, type Strokes } from "./network";

/** whether segment b's band differs from a's, or a's changes along it (an ease, T-1799) */
function bandDiffers(bd: Int16Array, a: number, b: number): boolean {
  const oa = a * BAND_STRIDE, ob = b * BAND_STRIDE;
  for (let x = 0; x < BAND_STRIDE; x++) if (bd[ob + x] !== bd[oa + x]) return true;
  return bd[oa] !== bd[oa + 1] || bd[oa + 2] !== bd[oa + 3] || bd[oa + 4] !== bd[oa + 5];
}

/** instances per culled chunk, and boxes per box of the next level up */
export const CHUNK = 16;
const FAN = 8;
/** a merged run's speeds (the Speed view's colour) stay within this many km/h of its first's */
const SPEED_STEP = 10;
/** longest run simplified as one, which bounds Douglas-Peucker's worst case */
const MAX_RUN = 512;

/** Boxes (min x, min y, max x, max y, local units) in levels: level 0 one per chunk of CHUNK instances,
 * each level above one per FAN boxes of the one below, up to a level of at most FAN boxes. */
export type ChunkBoxes = Float32Array[];

export function chunkBoxes(seg: Float32Array, count: number): ChunkBoxes {
  const chunks = Math.max(1, Math.ceil(count / CHUNK));
  const levels = [new Float32Array(chunks * 4)];
  const l0 = levels[0];
  for (let c = 0; c < chunks; c++) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = c * CHUNK, e = Math.min(count, i + CHUNK) * 4; i * 4 < e; i++) {
      const o = i * 4;
      x0 = Math.min(x0, seg[o], seg[o + 2]);
      x1 = Math.max(x1, seg[o], seg[o + 2]);
      y0 = Math.min(y0, seg[o + 1], seg[o + 3]);
      y1 = Math.max(y1, seg[o + 1], seg[o + 3]);
    }
    l0[c * 4] = x0;
    l0[c * 4 + 1] = y0;
    l0[c * 4 + 2] = x1;
    l0[c * 4 + 3] = y1;
  }
  for (let below = l0; below.length / 4 > FAN; ) {
    const n = Math.ceil(below.length / 4 / FAN), up = new Float32Array(n * 4);
    for (let g = 0; g < n; g++) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let k = g * FAN; k < Math.min(below.length / 4, (g + 1) * FAN); k++) {
        x0 = Math.min(x0, below[k * 4]);
        y0 = Math.min(y0, below[k * 4 + 1]);
        x1 = Math.max(x1, below[k * 4 + 2]);
        y1 = Math.max(y1, below[k * 4 + 3]);
      }
      up.set([x0, y0, x1, y1], g * 4);
    }
    levels.push(up);
    below = up;
  }
  return levels;
}

/** Ranges (start, count, ...) with each gap of `gap` instances or fewer drawn through, as one range */
export function mergeRanges(r: number[], gap: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < r.length; i += 2) {
    const k = out.length;
    if (k && out[k - 2] + out[k - 1] >= r[i] - gap) out[k - 1] = r[i] + r[i + 1] - out[k - 2];
    else out.push(r[i], r[i + 1]);
  }
  return out;
}

/** The instance ranges (start, count, ...) whose chunks may show on a `w` x `h` CSS px view, a box
 * counted off only when all its corners project to one side of it, `margin` px out; null: all. A box
 * wholly inside the view, or on `minLevel` or below, is drawn without testing the boxes in it. */
export function visibleRanges(levels: ChunkBoxes, count: number, project: (x: number, y: number) => [number, number], w: number, h: number, margin: number, minLevel = 0): number[] | null {
  const out: number[] = [];
  let all = true;
  const emit = (start: number, end: number) => {
    const k = out.length;
    if (k && out[k - 2] + out[k - 1] >= start) out[k - 1] = end - out[k - 2];
    else out.push(start, end - start);
  };
  const visit = (level: number, k: number) => {
    const b = levels[level], span = CHUNK * FAN ** level, start = k * span, end = Math.min(count, start + span);
    if (start >= end) return;
    let l = 0, r = 0, t = 0, d = 0, inside = 0;
    for (let j = 0; j < 4; j++) {
      const [px, py] = project(b[k * 4 + (j & 1 ? 2 : 0)], b[k * 4 + (j & 2 ? 3 : 1)]);
      if (!(px === px && py === py)) { l = r = t = d = 0; inside = -1; break; }
      if (px < -margin) l++;
      if (px > w + margin) r++;
      if (py < -margin) t++;
      if (py > h + margin) d++;
      if (px >= -margin && px <= w + margin && py >= -margin && py <= h + margin) inside++;
    }
    if (l === 4 || r === 4 || t === 4 || d === 4) { all = false; return; }
    if (level <= minLevel || inside === 4) emit(start, end);
    else for (let c = k * FAN; c < Math.min(levels[level - 1].length / 4, (k + 1) * FAN); c++) visit(level - 1, c);
  };
  const top = levels.length - 1;
  for (let k = 0; k < levels[top].length / 4; k++) visit(top, k);
  return all ? null : out;
}

/** Per segment, the tolerance (local units) below which its start point may be dropped: Douglas-
 * Peucker's distance where the point splits its run, no more than the point that split before it,
 * so every tolerance's kept points are DP's at that tolerance. Infinity where a run starts: runs are
 * joined segments of one line or track, edge, level, flags, stretch and (within SPEED_STEP) speed.
 * Worked out from run start `a` until `deadline` (performance.now ms) has passed: returns where it
 * stopped, `s.count` once done. */
export function significance(s: Strokes, speed: Float32Array | null, sig: Float32Array, a = 0, deadline = Infinity): number {
  const n = s.count, seg = s.seg;
  const st = s.stretch, bd = s.band, stack: number[] = [];
  for (let k = 0; a < n; k++) {
    if ((k & 255) === 255 && performance.now() > deadline) return a;
    let b = a + 1;
    for (; b < n && b - a < MAX_RUN; b++) {
      const o = b * 4;
      if (Math.abs(seg[o] - seg[o - 2]) + Math.abs(seg[o + 1] - seg[o - 1]) > 1e-3) break;
      if (s.colour[b] !== s.colour[a] || s.edge[b] !== s.edge[a] || s.level[b] !== s.level[a] || s.flags[b] !== s.flags[a]) break;
      if (st && st[b] !== st[a]) break;
      // a band's shift easing at a node (T-1799) is never merged away
      // (nor one where only the Traffic view's shift eases, T-2132: merged, its S-curve became a chord)
      if (bd && (bandDiffers(bd, a, b) || (s.flags[a] & F_EASE) !== 0)) break;
      if (speed && Math.abs(speed[b] - speed[a]) > SPEED_STEP) break;
    }
    sig[a] = Infinity;
    // vertex v of the run: segment v's start, or for v = b the last segment's end
    const vx = (v: number) => (v < b ? seg[v * 4] : seg[v * 4 - 2]);
    const vy = (v: number) => (v < b ? seg[v * 4 + 1] : seg[v * 4 - 1]);
    stack.push(a, b, Infinity);
    while (stack.length) {
      const cap = stack.pop()!, hi = stack.pop()!, lo = stack.pop()!;
      if (hi - lo < 2) continue;
      const ax = vx(lo), ay = vy(lo), dx = vx(hi) - ax, dy = vy(hi) - ay, len2 = dx * dx + dy * dy;
      let m = lo + 1, best = -1;
      for (let v = lo + 1; v < hi; v++) {
        const px = seg[v * 4] - ax, py = seg[v * 4 + 1] - ay;
        const t = len2 > 0 ? Math.max(0, Math.min(1, (px * dx + py * dy) / len2)) : 0;
        const ex = px - t * dx, ey = py - t * dy, d = ex * ex + ey * ey;
        if (d > best) { best = d; m = v; }
      }
      const sm = Math.min(Math.sqrt(best), cap);
      sig[m] = sm;
      stack.push(lo, m, sm, m, hi, sm);
    }
    a = b;
  }
  return n;
}

/** The strokes with every start point under `tol` dropped, each merged segment carrying its first
 * segment's values; `first` is that segment's index in `s`, for the per-segment arrays rewritten
 * later (the demand widths, the traffic marks). */
export function simplified(s: Strokes, sig: Float32Array, tol: number, speed: Float32Array | null): { strokes: Strokes; first: Uint32Array } {
  const out = simplifiedLater(s, sig, tol, speed);
  out.fill(0, out.strokes.count);
  return out;
}

/** `simplified`, its arrays filled by `fill(from, to)` (merged segments `from` to `to`), so the work
 * can be spread over several tasks (T-2403). */
export function simplifiedLater(s: Strokes, sig: Float32Array, tol: number, speed: Float32Array | null): { strokes: Strokes; first: Uint32Array; fill: (from: number, to: number) => void } {
  const n = s.count;
  let m = 0;
  for (let i = 0; i < n; i++) if (sig[i] > tol) m++;
  const first = new Uint32Array(m);
  for (let i = 0, k = 0; i < n; i++) if (sig[i] > tol) first[k++] = i;
  const seg = new Float32Array(m * 4), colour = new Uint32Array(m), edge = new Uint32Array(m), level = new Float32Array(m), flags = new Float32Array(m);
  const sp = speed ? new Float32Array(m) : undefined, lane = s.lane ? new Uint8Array(m * 2) : undefined, stretch = s.stretch ? new Float32Array(m) : undefined, band = s.band ? new Int16Array(m * BAND_STRIDE) : undefined;
  const fill = (from: number, to: number) => {
    for (let k = from; k < Math.min(to, m); k++) {
      const i = first[k], last = (k + 1 < m ? first[k + 1] : n) - 1;
      seg[k * 4] = s.seg[i * 4];
      seg[k * 4 + 1] = s.seg[i * 4 + 1];
      seg[k * 4 + 2] = s.seg[last * 4 + 2];
      seg[k * 4 + 3] = s.seg[last * 4 + 3];
      colour[k] = s.colour[i];
      edge[k] = s.edge[i];
      level[k] = s.level[i];
      flags[k] = s.flags[i];
      if (sp) sp[k] = speed![i];
      if (stretch) stretch[k] = s.stretch![i];
      if (lane) {
        lane[k * 2] = s.lane![i * 2];
        lane[k * 2 + 1] = s.lane![last * 2 + 1];
      }
      if (band) {
        const b = s.band!, o = k * BAND_STRIDE, oi = i * BAND_STRIDE, ol = last * BAND_STRIDE;
        band[o] = b[oi];
        band[o + 1] = b[ol + 1];
        band[o + 2] = b[oi + 2];
        band[o + 3] = b[ol + 3];
        band[o + 4] = b[oi + 4];
        band[o + 5] = b[ol + 5];
      }
    }
  };
  return { strokes: { seg, colour, edge, level, flags, speed: sp, dist: new Float32Array(0), stretch, lane, band, count: m }, first, fill };
}
