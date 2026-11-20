// T-3824: a ferry on the network diagram drawn over the diagram's own water (water.ts, drawn before the thin lines):
// the drawn sea, bands and fillets on a grid, the way over it between two stops, and a port moved onto the drawn coast.
// Pure: no DOM.

import type { DiagramWater } from "./water";

/** the drawn water on a grid: `cell` layout units a side from (x0, y0), 1 where wet */
export interface WaterGrid { x0: number; y0: number; cell: number; w: number; h: number; wet: Uint8Array }

/** path commands as subpaths of points (layout units), each curve in 4 pieces */
function flatten(cmd: ArrayLike<number>): number[][] {
  const out: number[][] = [];
  let cur: number[] = [];
  for (let i = 0; i < cmd.length; ) {
    if (cmd[i] === 0) (cur = [cmd[i + 1], cmd[i + 2]]), out.push(cur), (i += 3);
    else if (cmd[i] === 1) cur.push(cmd[i + 1], cmd[i + 2]), (i += 3);
    else {
      const ax = cur[cur.length - 2], ay = cur[cur.length - 1];
      for (let t = 1; t <= 4; t++) { const u = t / 4; cur.push((1 - u) ** 2 * ax + 2 * u * (1 - u) * cmd[i + 1] + u * u * cmd[i + 3], (1 - u) ** 2 * ay + 2 * u * (1 - u) * cmd[i + 2] + u * u * cmd[i + 4]); }
      i += 5;
    }
  }
  return out;
}

/** The drawn water `wa` on a grid over the box (layout units) at `cell`. */
export function waterGrid(wa: DiagramWater, box: [number, number, number, number], cell: number): WaterGrid {
  const [x0, y0, x1, y1] = box, w = Math.max(1, Math.ceil((x1 - x0) / cell)), h = Math.max(1, Math.ceil((y1 - y0) / cell)), wet = new Uint8Array(w * h);
  /** rings filled even-odd: each row's crossings, from every edge across it (left of the box too) */
  const fill = (rings: number[][]) => {
    const rows: number[][] = Array.from({ length: h }, () => []);
    for (const r of rings) for (let a = 0, m = r.length; a < m; a += 2) {
      const b = (a + 2) % m, ay = r[a + 1], by = r[b + 1];
      if (ay === by) continue;
      const j0 = Math.max(0, Math.ceil((Math.min(ay, by) - y0) / cell - 0.5)), j1 = Math.min(h - 1, Math.floor((Math.max(ay, by) - y0) / cell - 0.5));
      for (let j = j0; j <= j1; j++) { const y = y0 + (j + 0.5) * cell; if (ay > y !== by > y) rows[j].push(r[a] + ((y - ay) * (r[b] - r[a])) / (by - ay)); }
    }
    rows.forEach((xs, j) => {
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) for (let i = Math.max(0, Math.ceil((xs[k] - x0) / cell - 0.5)); i <= Math.min(w - 1, Math.floor((xs[k + 1] - x0) / cell - 0.5)); i++) wet[j * w + i] = 1;
    });
  };
  if (wa.sea?.length) fill(flatten(wa.sea));
  // each fillet a ring of its own, so overlapping ones do not cancel
  if (wa.fillets?.length) for (const r of flatten(wa.fillets)) fill([r]);
  const disk = (cx: number, cy: number, ax: number, ay: number, hw: number) => {
    const dx = ax - cx, dy = ay - cy, l2 = dx * dx + dy * dy;
    const i0 = Math.max(0, Math.floor((Math.min(cx, ax) - hw - x0) / cell)), i1 = Math.min(w - 1, Math.floor((Math.max(cx, ax) + hw - x0) / cell));
    const j0 = Math.max(0, Math.floor((Math.min(cy, ay) - hw - y0) / cell)), j1 = Math.min(h - 1, Math.floor((Math.max(cy, ay) + hw - y0) / cell));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const x = x0 + (i + 0.5) * cell, y = y0 + (j + 0.5) * cell, t = l2 > 0 ? Math.max(0, Math.min(1, ((x - cx) * dx + (y - cy) * dy) / l2)) : 0;
      if ((x - cx - dx * t) ** 2 + (y - cy - dy * t) ** 2 <= hw * hw) wet[j * w + i] = 1;
    }
  };
  wa.rivers.forEach((cmd, k) => { const hw = wa.bands[k] / 2; for (const p of flatten(cmd)) for (let q = 0; q + 3 < p.length; q += 2) disk(p[q], p[q + 1], p[q + 2], p[q + 3], hw); });
  return { x0, y0, cell, w, h, wet };
}

export const wetAt = (G: WaterGrid, x: number, y: number): boolean => {
  const i = Math.floor((x - G.x0) / G.cell), j = Math.floor((y - G.y0) / G.cell);
  return i >= 0 && j >= 0 && i < G.w && j < G.h && G.wet[j * G.w + i] === 1;
};

/** the drawn coast nearest (x, y) within `reach` (layout units): from a place on land the first wet point toward the
 * nearest water, from one on water the last wet point toward the nearest land; null if none so near */
export function toCoast(G: WaterGrid, x: number, y: number, reach: number): [number, number] | null {
  const wet0 = wetAt(G, x, y), ci = Math.floor((x - G.x0) / G.cell), cj = Math.floor((y - G.y0) / G.cell), R = Math.ceil(reach / G.cell);
  let best = -1, bd = Infinity;
  for (let j = cj - R; j <= cj + R; j++) for (let i = ci - R; i <= ci + R; i++) {
    if (i < 0 || j < 0 || i >= G.w || j >= G.h || (G.wet[j * G.w + i] === 1) === wet0) continue;
    const d = (i - ci) ** 2 + (j - cj) ** 2;
    if (d < bd) (bd = d), (best = j * G.w + i);
  }
  if (best < 0 || Math.sqrt(bd) > R) return null;
  const tx = G.x0 + ((best % G.w) + 0.5) * G.cell, ty = G.y0 + (Math.floor(best / G.w) + 0.5) * G.cell, n = Math.max(2, Math.ceil((4 * Math.hypot(tx - x, ty - y)) / G.cell));
  let px = x, py = y;
  for (let t = 1; t <= n; t++) {
    const qx = x + ((tx - x) * t) / n, qy = y + ((ty - y) * t) / n;
    if (wetAt(G, qx, qy) !== wet0) return wet0 ? [px, py] : [qx, qy];
    (px = qx), (py = qy);
  }
  return wet0 ? [px, py] : [tx, ty];
}

/** a step over land costs this many times one over water */
const LAND = 12;

/** The cheapest way over the grid from (ax, ay) to (bx, by), layout units, as points (cell centres between the two
 * ends), steps of a side or a corner, land LAND times dearer; null if either end is off the grid. */
export function waterPath(G: WaterGrid, ax: number, ay: number, bx: number, by: number): number[] | null {
  const { w, h, cell } = G, cellOf = (x: number, y: number) => { const i = Math.floor((x - G.x0) / cell), j = Math.floor((y - G.y0) / cell); return i < 0 || j < 0 || i >= w || j >= h ? -1 : j * w + i; };
  const s = cellOf(ax, ay), t = cellOf(bx, by);
  if (s < 0 || t < 0) return null;
  const N = w * h, g = new Float64Array(N).fill(Infinity), from = new Int32Array(N).fill(-1), done = new Uint8Array(N);
  const ti = t % w, tj = Math.floor(t / w), hOf = (c: number) => Math.hypot((c % w) - ti, Math.floor(c / w) - tj);
  // a binary heap of (f, cell)
  const hf: number[] = [], hc: number[] = [];
  const push = (f: number, c: number) => {
    let k = hf.length;
    hf.push(f), hc.push(c);
    while (k > 0) { const p = (k - 1) >> 1; if (hf[p] <= f) break; (hf[k] = hf[p]), (hc[k] = hc[p]), (k = p); }
    (hf[k] = f), (hc[k] = c);
  };
  const pop = () => {
    const c = hc[0], lf = hf.pop()!, lc = hc.pop()!;
    if (hf.length) {
      let k = 0;
      for (;;) { const a = 2 * k + 1, b = a + 1; let m = k, mf = lf; if (a < hf.length && hf[a] < mf) (m = a), (mf = hf[a]); if (b < hf.length && hf[b] < mf) (m = b), (mf = hf[b]); if (m === k) break; (hf[k] = hf[m]), (hc[k] = hc[m]), (k = m); }
      (hf[k] = lf), (hc[k] = lc);
    }
    return c;
  };
  g[s] = 0;
  push(hOf(s), s);
  while (hf.length) {
    const c = pop();
    if (done[c]) continue;
    done[c] = 1;
    if (c === t) break;
    const i = c % w, j = (c - i) / w;
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      if (!di && !dj) continue;
      const ii = i + di, jj = j + dj;
      if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue;
      const u = jj * w + ii, step = (di && dj ? Math.SQRT2 : 1) * (G.wet[u] ? 1 : LAND);
      if (g[c] + step < g[u]) (g[u] = g[c] + step), (from[u] = c), push(g[u] + hOf(u), u);
    }
  }
  if (!done[t]) return null;
  const cells: number[] = [];
  for (let c = t; c >= 0; c = from[c]) cells.push(c);
  const out: number[] = [ax, ay];
  for (let k = cells.length - 2; k >= 1; k--) out.push(G.x0 + ((cells[k] % w) + 0.5) * cell, G.y0 + (Math.floor(cells[k] / w) + 0.5) * cell);
  out.push(bx, by);
  return out;
}
