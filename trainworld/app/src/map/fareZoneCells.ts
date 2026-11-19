// T-3740 stage Z2: the Fare zones tint's cells, as vector outlines. Each painted site (a complex, or a station on
// its own) owns the ground nearer to it than to any other painted site, out to its reach: its Voronoi cell cut to a
// disc. Every area's sites are in the one diagram, so no two cells overlap. Built in Mercator 0..1 (the anchor and
// the camera don't change them) when the paint or the reach changes; map/fareZoneTool.ts projects and draws them.

/** A painted site, Mercator 0..1; `r` its reach in the same units. */
export interface Site { x: number; y: number; r: number; code: number }

/** One piece of a cell's outline, relative to its site: a straight side, which is a rim (an edge of the zone's
 * patch) when the site across it has another code; or an arc of the disc, always a rim. Angles rise along the
 * outline, a1 > a0. */
export type Piece =
  | { arc: false; x0: number; y0: number; x1: number; y1: number; rim: boolean }
  | { arc: true; a0: number; a1: number };

export interface Cell { x: number; y: number; r: number; code: number; pieces: Piece[] }

/** sites nearer than this share a spot (Mercator 0..1, about 4 mm): the earlier one keeps it */
const SAME = 1e-10;

export function buildCells(sites: readonly Site[]): Cell[] {
  let rmax = 0;
  for (const s of sites) rmax = Math.max(rmax, s.r);
  if (!sites.length || !(rmax > 0)) return [];
  const G = 2 * rmax, grid = new Map<number, number[]>();
  const key = (gx: number, gy: number) => gx * 1e6 + gy;
  sites.forEach((s, i) => {
    const k = key(Math.floor(s.x / G), Math.floor(s.y / G)), b = grid.get(k);
    if (b) b.push(i); else grid.set(k, [i]);
  });
  const out: Cell[] = [];
  let xs: number[] = [], ys: number[] = [], nb: number[] = [];
  let nx: number[] = [], ny: number[] = [], nn: number[] = [];
  for (let i = 0; i < sites.length; i++) {
    const s = sites[i], h = s.r * 1.1;
    // a square round the disc, its sides no site's (-1), corners in order of rising angle
    xs = [-h, h, h, -h]; ys = [-h, -h, h, h]; nb = [-1, -1, -1, -1];
    const gx = Math.floor(s.x / G), gy = Math.floor(s.y / G);
    let gone = false;
    for (let ax = gx - 1; ax <= gx + 1 && !gone; ax++) for (let ay = gy - 1; ay <= gy + 1 && !gone; ay++) {
      for (const j of grid.get(key(ax, ay)) ?? []) {
        if (j === i) continue;
        const t = sites[j], dx = t.x - s.x, dy = t.y - s.y, d2 = dx * dx + dy * dy, lim = s.r + t.r;
        if (d2 > lim * lim) continue;
        if (d2 < SAME * SAME) { if (j < i) { gone = true; break; } continue; }
        // keep the side nearer s: p . d <= |d|^2 / 2
        const c = d2 / 2;
        nx = []; ny = []; nn = [];
        const n = xs.length;
        for (let k = 0; k < n; k++) {
          const k1 = (k + 1) % n;
          const pa = xs[k] * dx + ys[k] * dy - c, pb = xs[k1] * dx + ys[k1] * dy - c;
          if (pa <= 0) { nx.push(xs[k]); ny.push(ys[k]); nn.push(nb[k]); }
          if ((pa <= 0) !== (pb <= 0)) {
            const f = pa / (pa - pb), ix = xs[k] + f * (xs[k1] - xs[k]), iy = ys[k] + f * (ys[k1] - ys[k]);
            nx.push(ix); ny.push(iy);
            // leaving: the side from here runs along j's bisector; entering: the rest of the old side
            nn.push(pa <= 0 ? j : nb[k]);
          }
        }
        [xs, ys, nb] = [nx, ny, nn];
        if (xs.length < 3) { gone = true; break; }
      }
    }
    if (gone) continue;
    const pieces = cutToDisc(xs, ys, nb, s.r, (j) => j >= 0 && sites[j].code !== s.code);
    out.push({ x: s.x, y: s.y, r: s.r, code: s.code, pieces });
  }
  return out;
}

/** The convex outline (xs, ys; side k from corner k to k + 1 is site nb[k]'s) cut to the disc of radius r round the
 * origin, which is inside it: its sides' parts in the disc, joined by arcs. */
function cutToDisc(xs: number[], ys: number[], nb: number[], r: number, rimOf: (j: number) => boolean): Piece[] {
  const lines: Piece[] = [];
  const n = xs.length, r2 = r * r;
  for (let k = 0; k < n; k++) {
    const k1 = (k + 1) % n, ax = xs[k], ay = ys[k], dx = xs[k1] - ax, dy = ys[k1] - ay;
    const A = dx * dx + dy * dy;
    if (A <= 0) continue;
    const B = 2 * (ax * dx + ay * dy), C = ax * ax + ay * ay - r2, D = B * B - 4 * A * C;
    if (D <= 0) continue;
    const sq = Math.sqrt(D), t0 = Math.max(0, (-B - sq) / (2 * A)), t1 = Math.min(1, (-B + sq) / (2 * A));
    if (t1 - t0 < 1e-9) continue;
    lines.push({ arc: false, x0: ax + t0 * dx, y0: ay + t0 * dy, x1: ax + t1 * dx, y1: ay + t1 * dy, rim: nb[k] < 0 || rimOf(nb[k]) });
  }
  if (!lines.length) return [{ arc: true, a0: 0, a1: 2 * Math.PI }];
  const out: Piece[] = [];
  const gap = r * 1e-7;
  for (let k = 0; k < lines.length; k++) {
    const p = lines[k] as Extract<Piece, { arc: false }>, q = lines[(k + 1) % lines.length] as Extract<Piece, { arc: false }>;
    out.push(p);
    if (Math.hypot(q.x0 - p.x1, q.y0 - p.y1) <= gap) continue;
    const a0 = Math.atan2(p.y1, p.x1);
    let a1 = Math.atan2(q.y0, q.x0);
    while (a1 < a0) a1 += 2 * Math.PI;
    out.push({ arc: true, a0, a1 });
  }
  return out;
}

/** Whether (x, y), relative to the cell's site, is inside the cell (for checks). */
export function inCell(c: Cell, x: number, y: number): boolean {
  if (x * x + y * y > c.r * c.r) return false;
  // the outline is convex and runs round the site with rising angle: inside is left of every straight side
  for (const p of c.pieces) if (!p.arc && (p.x1 - p.x0) * (y - p.y0) - (p.y1 - p.y0) * (x - p.x0) < 0) return false;
  return true;
}
