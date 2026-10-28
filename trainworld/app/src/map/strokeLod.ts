// Fewer stroke instances per draw (T-1639, notes/T-1618.md): zoomed out, each network stroke set is
// drawn as a simplified copy whose runs of segments are merged where the merged segment strays from
// them by less than a fraction of a pixel; closer in, only the chunks whose box touches the view are
// drawn. Both keep the segments' order, which decides which line is drawn over which (T-190).

import type { Strokes } from "./network";

/** instances per culled chunk, and chunks per group tested first */
export const CHUNK = 256;
const GROUP = 16;
/** a merged run's speeds (the Speed view's colour) stay within this many km/h of its first's */
const SPEED_STEP = 10;
/** longest run simplified as one, which bounds Douglas-Peucker's worst case */
const MAX_RUN = 512;

/** Per chunk of CHUNK instances its box (min x, min y, max x, max y, local units); then per group of
 * GROUP chunks the same, after the chunks. */
export function chunkBoxes(seg: Float32Array, count: number): { boxes: Float32Array; chunks: number } {
  const chunks = Math.ceil(count / CHUNK), groups = Math.ceil(chunks / GROUP);
  const out = new Float32Array((chunks + groups) * 4);
  for (let c = 0; c < chunks; c++) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = c * CHUNK, e = Math.min(count, i + CHUNK) * 4; i * 4 < e; i++) {
      const o = i * 4;
      x0 = Math.min(x0, seg[o], seg[o + 2]);
      x1 = Math.max(x1, seg[o], seg[o + 2]);
      y0 = Math.min(y0, seg[o + 1], seg[o + 3]);
      y1 = Math.max(y1, seg[o + 1], seg[o + 3]);
    }
    out.set([x0, y0, x1, y1], c * 4);
    const g = (chunks + Math.floor(c / GROUP)) * 4;
    if (c % GROUP === 0) out.set([x0, y0, x1, y1], g);
    else {
      out[g] = Math.min(out[g], x0);
      out[g + 1] = Math.min(out[g + 1], y0);
      out[g + 2] = Math.max(out[g + 2], x1);
      out[g + 3] = Math.max(out[g + 3], y1);
    }
  }
  return { boxes: out, chunks };
}

/** The instance ranges (start, count, ...) whose chunks may show on a `w` x `h` CSS px view, a box
 * counted off only when all its corners project to one side of it, `margin` px out; null: all. */
export function visibleRanges(b: { boxes: Float32Array; chunks: number }, count: number, project: (x: number, y: number) => [number, number], w: number, h: number, margin: number): number[] | null {
  const { boxes, chunks } = b;
  const off = (k: number) => {
    let l = 0, r = 0, t = 0, d = 0;
    for (let j = 0; j < 4; j++) {
      const [px, py] = project(boxes[k * 4 + (j & 1 ? 2 : 0)], boxes[k * 4 + (j & 2 ? 3 : 1)]);
      if (!(px === px && py === py)) return false;
      if (px < -margin) l++;
      if (px > w + margin) r++;
      if (py < -margin) t++;
      if (py > h + margin) d++;
    }
    return l === 4 || r === 4 || t === 4 || d === 4;
  };
  const out: number[] = [];
  let all = true;
  for (let g = 0; g * GROUP < chunks; g++) {
    if (off(chunks + g)) { all = false; continue; }
    for (let c = g * GROUP; c < Math.min(chunks, (g + 1) * GROUP); c++) {
      if (off(c)) { all = false; continue; }
      const start = c * CHUNK, n = Math.min(count, start + CHUNK) - start, k = out.length;
      // a gap of a chunk or two is drawn through: each range costs a rebinding per pass
      if (k && out[k - 2] + out[k - 1] >= start - 2 * CHUNK) out[k - 1] = start + n - out[k - 2];
      else out.push(start, n);
    }
  }
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
  const st = s.stretch, stack: number[] = [];
  for (let k = 0; a < n; k++) {
    if ((k & 255) === 255 && performance.now() > deadline) return a;
    let b = a + 1;
    for (; b < n && b - a < MAX_RUN; b++) {
      const o = b * 4;
      if (Math.abs(seg[o] - seg[o - 2]) + Math.abs(seg[o + 1] - seg[o - 1]) > 1e-3) break;
      if (s.colour[b] !== s.colour[a] || s.edge[b] !== s.edge[a] || s.level[b] !== s.level[a] || s.flags[b] !== s.flags[a]) break;
      if (st && st[b] !== st[a]) break;
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
  const n = s.count;
  let m = 0;
  for (let i = 0; i < n; i++) if (sig[i] > tol) m++;
  const first = new Uint32Array(m);
  for (let i = 0, k = 0; i < n; i++) if (sig[i] > tol) first[k++] = i;
  const seg = new Float32Array(m * 4), colour = new Uint32Array(m), edge = new Uint32Array(m), level = new Float32Array(m), flags = new Float32Array(m);
  const sp = speed ? new Float32Array(m) : undefined, lane = s.lane ? new Uint8Array(m * 2) : undefined, stretch = s.stretch ? new Float32Array(m) : undefined;
  for (let k = 0; k < m; k++) {
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
  }
  return { strokes: { seg, colour, edge, level, flags, speed: sp, dist: new Float32Array(0), stretch, lane, count: m }, first };
}
