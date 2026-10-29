// T-1791, T-1809: the group view's combined diagram, laid out (pure, for checks/t1791-layout.mjs).
// Rows: one place each (a station complex, `key`), top to bottom. They start from the group's line
// with most stops in its path order; each other line is matched to them by place, turned round if it
// runs the other way, and its own places go in as one block: between the rows it leaves and rejoins,
// or above its first shared row or below its last, whichever leaves the shorter bar beside them (so a
// branch's stops stay together and it hangs off the row where it leaves).
// Columns bend like a schematic (notes/T-1809.md): lines together on a stretch run GAP apart, the rest
// SPREAD apart, in an order per row that keeps lines alike side by side and puts the one that differs
// soonest outside. Only key stops get a named row unless `all`: termini, where lines meet and part,
// the stops on either side of a stretch one line skips, and transfers to lines outside the group.

import { pillHeight, ROUND_MAX, TALL_MAX } from "./relatedLayout";

/** One place on a line's path, in path order: a stop or a station it passes. */
export interface PathPlace { key: string; station: string; stop: boolean; /** the stop's index in its line's stops, -1 for a station passed */ index: number }
export interface GroupLineIn { num: number; places: PathPlace[] }
export interface GroupCell { station: string; stop: boolean; index: number; key: string; /** a key stop: its row is named */ named: boolean }
export interface GroupRow {
  /** a cell per column (by the input's order), null where the line has no place here */
  cells: (GroupCell | null)[];
  /** a named row (a full row with the station's name); else a tick row (a bare one where every
   * line passes, so a ghost there can add the stop) */
  named: boolean;
  /** the lines present here, left to right */
  order: number[];
  /** centre and height, px */
  y: number;
  h: number;
}
/** A line's bullet at one of its ends (0 its first stop, 1 its last). */
export interface GroupBullet { col: number; end: 0 | 1; x: number; y: number }
export interface GroupLayout {
  rows: GroupRow[];
  /** per column, its centre's x at each row, NaN outside its span */
  xs: number[][];
  /** per column, its first and last rows (-1, -1 for a line with no places) */
  spans: [number, number][];
  /** per column, its bar as an SVG path through its rows */
  paths: string[];
  bullets: GroupBullet[];
  width: number;
  height: number;
  /** columns `a` and `b` run together at row `r` (drawn GAP apart) */
  together: (a: number, b: number, r: number) => boolean;
  /** column `k`'s centre at height `y` (NaN where it is not drawn) */
  xAt: (k: number, y: number) => number;
}
export interface GroupOpts {
  /** every row named (the "all stops" switch) */
  all?: boolean;
  /** lines outside the group stop at this place */
  transfer?: (key: string) => boolean;
  /** the crossing bars drawn through this place's row (past ROUND_MAX the row grows to fit them) */
  crossings?: (key: string) => number;
}

/** A bar's width, the gap between lines running together, lines apart, px. */
export const BAR = 5, GAP = 2, PITCH = BAR + GAP, SPREAD = 24;
/** A named row's height, a tick row's, the bullets' pitch, room above and below, px. */
export const NAME_H = 18, TICK_H = 8, CHIP = 18;
const PAD = 22, MARGIN = 11;
/** Rows looked ahead when ordering; a crossing's cost against a row of likeness. */
const AHEAD = 60, CROSS = 8;
/** Orderings tried per row at most; past it the order holds. */
const MAX_TRIES = 5040;

/** A line's places from its stops and the stations it passes (`[gap, node]`, gap i after stop i). */
export function pathPlaces(stops: readonly string[], passed: readonly { gap: number; node: number }[], key: (station: string) => string): PathPlace[] {
  const out: PathPlace[] = [];
  stops.forEach((s, i) => {
    out.push({ key: key(s), station: s, stop: true, index: i });
    for (const p of passed) if (p.gap === i) out.push({ key: key(String(p.node)), station: String(p.node), stop: false, index: -1 });
  });
  return out;
}

/** A loop's path ends where it began: its places once round. */
function onceRound(p: PathPlace[]): PathPlace[] {
  return p.length > 2 && p[0].key === p[p.length - 1].key ? p.slice(0, -1) : p;
}

function perms<T>(xs: T[]): T[][] {
  if (xs.length <= 1) return [xs.slice()];
  const out: T[][] = [];
  xs.forEach((x, i) => { for (const p of perms([...xs.slice(0, i), ...xs.slice(i + 1)])) out.push([x, ...p]); });
  return out;
}
const fact = (m: number): number => (m <= 1 ? 1 : m * fact(m - 1));

/** Centres for clusters in order, each wanting `want` (null: no wish) with weight `w`, cluster j at
 * least `gap[j]` right of j-1: weighted least squares under the order (pool adjacent violators). */
function place(want: (number | null)[], w: number[], gap: number[]): number[] {
  const m = want.length, off: number[] = [];
  for (let j = 0; j < m; j++) off.push(j ? off[j - 1] + gap[j] : 0);
  const t: (number | null)[] = want.map((d, j) => (d === null ? null : d - off[j]));
  // a cluster with no wish sits tight against its left neighbour (the first, against its right one)
  const known = t.findIndex((v) => v !== null);
  for (let j = 0; j < m; j++) if (t[j] === null) t[j] = j && t[j - 1] !== null ? t[j - 1] : known >= 0 ? t[known] : 0;
  const blocks: { v: number; w: number; n: number }[] = [];
  for (let j = 0; j < m; j++) {
    blocks.push({ v: t[j]!, w: want[j] === null ? 1e-6 : w[j], n: 1 });
    while (blocks.length > 1 && blocks[blocks.length - 2].v > blocks[blocks.length - 1].v) {
      const b = blocks.pop()!, a = blocks.pop()!;
      blocks.push({ v: (a.v * a.w + b.v * b.w) / (a.w + b.w), w: a.w + b.w, n: a.n + b.n });
    }
  }
  const out: number[] = [];
  for (const b of blocks) for (let i = 0; i < b.n; i++) out.push(b.v + off[out.length]);
  return out;
}

export function groupLayout(lines: GroupLineIn[], o: GroupOpts = {}): GroupLayout {
  const n = lines.length, all = !!o.all;
  const none: GroupLayout = { rows: [], xs: [], spans: [], paths: [], bullets: [], width: 2 * MARGIN, height: 2 * PAD, together: () => false, xAt: () => NaN };
  if (!n) return none;
  const ps = lines.map((l) => onceRound(l.places));

  // ---- rows: one place each, every line down them in its path order, each line's own places a block
  type Node = { key: string; cells: (GroupCell | null)[] };
  const seq: Node[] = [];
  const mk = (k: number, p: PathPlace): Node => {
    const cells = Array<GroupCell | null>(n).fill(null);
    cells[k] = { station: p.station, stop: p.stop, index: p.index, key: p.key, named: false };
    return { key: p.key, cells };
  };
  const nStops = (k: number) => ps[k].filter((p) => p.stop).length;
  const spine = ps.reduce((b, _, k) => (nStops(k) > nStops(b) ? k : b), 0);
  seq.push(...ps[spine].map((p) => mk(spine, p)));
  for (let k = 0; k < n; k++) {
    if (k === spine) continue;
    let pl = ps[k];
    const first = (p: PathPlace) => seq.findIndex((s) => s.key === p.key);
    const hits = pl.map(first).filter((r) => r >= 0);
    if (hits.length >= 2 && hits[0] > hits[hits.length - 1]) pl = [...pl].reverse();
    let last = -1, pending: PathPlace[] = [];
    const block = (at: number) => {
      seq.splice(at, 0, ...pending.map((p) => mk(k, p)));
      const added = pending.length;
      pending = [];
      return added;
    };
    for (const p of pl) {
      let r = -1;
      for (let i = last + 1; i < seq.length; i++) if (seq[i].key === p.key && !seq[i].cells[k]) { r = i; break; }
      if (r < 0) { pending.push(p); continue; }
      // above its first shared row: just above it, or at the top when that leaves a shorter bar
      r += block(last < 0 && pending.length > r ? 0 : r);
      seq[r].cells[k] = { station: p.station, stop: p.stop, index: p.index, key: p.key, named: false };
      last = r;
    }
    if (last < 0) { block(seq.length); continue; } // nothing shared: below everything
    const rest = seq.length - last - 1;
    block(pending.length <= rest ? last + 1 : seq.length);
  }
  const cell = (k: number, r: number) => seq[r].cells[k];
  const stopAt = (k: number, r: number) => !!seq[r].cells[k]?.stop;
  const pk = (a: number, b: number) => (a < b ? a * n + b : b * n + a);
  const name = (k: number, r: number) => { const c = cell(k, r); if (c?.stop) c.named = true; };
  let R = 0, spans: [number, number][] = [], tog = new Map<number, Uint8Array>();
  const present = (k: number, r: number) => spans[k][0] >= 0 && spans[k][0] <= r && r <= spans[k][1];
  const together = (a: number, b: number, r: number) => a === b || !!tog.get(pk(a, b))?.[r];
  // ---- together: from a stop both make to the next, unless each makes stops the other does not
  // (skip-stop: drawn apart); when `naming`, the two stops either side of a stretch one skips are named
  const analyse = (naming: boolean) => {
    R = seq.length;
    spans = lines.map((_, k): [number, number] => {
      let a = -1, b = -1;
      for (let r = 0; r < R; r++) if (seq[r].cells[k]) { if (a < 0) a = r; b = r; }
      return [a, b];
    });
    tog = new Map();
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      const t = new Uint8Array(R);
      const common: number[] = [];
      for (let r = 0; r < R; r++) if (stopAt(a, r) && stopAt(b, r) && cell(a, r)!.key === cell(b, r)!.key) common.push(r);
      common.forEach((c, i) => {
        t[c] = 1;
        const d = common[i + 1];
        if (d === undefined) return;
        let oa = false, ob = false;
        for (let r = c + 1; r < d; r++) { oa ||= stopAt(a, r); ob ||= stopAt(b, r); }
        if (oa && ob) return;
        t.fill(1, c, d + 1);
        if (naming && (oa || ob)) for (const k of [a, b]) { name(k, c); name(k, d); }
      });
      // where they meet and where they part
      if (naming) for (let r = 0; r < R; r++) if (t[r] && (!r || !t[r - 1] || r === R - 1 || !t[r + 1])) { name(a, r); name(b, r); }
      tog.set(pk(a, b), t);
    }
  };
  analyse(true);
  lines.forEach((_, k) => { if (spans[k][0] >= 0) { name(k, spans[k][0]); name(k, spans[k][1]); } });
  if (o.transfer) for (const s of seq) for (const c of s.cells) if (c?.stop && o.transfer(c.key)) c.named = true;

  // ---- branches side by side share rows where both have only ticks (Anita, T-1817): a tick row
  // moves up into the first tick row after its lines' previous places, beside lines apart from them
  if (!all) {
    const named = (s: Node) => s.cells.some((c) => c?.named);
    const at0 = seq.map((_, r) => r); // each row's index in the analysis above
    for (let r = 1; r < seq.length; r++) {
      const s = seq[r];
      if (named(s)) continue;
      const ks = s.cells.map((c, k) => (c ? k : -1)).filter((k) => k >= 0);
      let lo = 0, starts = false;
      for (const k of ks) {
        let q = r - 1;
        while (q >= 0 && !seq[q].cells[k]) q--;
        if (q < 0) starts = true;
        lo = Math.max(lo, q + 1);
      }
      if (starts) continue;
      for (let q = lo; q < r; q++) {
        const t = seq[q];
        if (named(t) || ks.some((k) => t.cells[k])) continue;
        const others = t.cells.map((c, j) => (c ? j : -1)).filter((j) => j >= 0);
        if (others.some((j) => ks.some((k) => together(j, k, at0[q]) || together(j, k, at0[r])))) continue;
        for (const k of ks) t.cells[k] = s.cells[k];
        seq.splice(r, 1);
        at0.splice(r, 1);
        r--;
        break;
      }
    }
    if (seq.length < R) analyse(false);
  }

  // ---- order per row: clusters of lines together side by side; inside one, lines alike for longer
  // ahead nearer; apart, lines that will run alike soon nearer; few crossings against the rows next
  // to it. Swept down, up and down again, so an order a meeting needs reaches back up the branches.
  const alike = (a: number, b: number, r: number) => present(a, r) && present(b, r) && together(a, b, r) && stopAt(a, r) === stopAt(b, r);
  const cluster: number[][] = []; // per row, each line's cluster id (-1 absent)
  const rowsIn: { P: number[]; groups: number[][]; wt: Map<number, number> }[] = [];
  for (let r = 0; r < R; r++) {
    const P = lines.map((_, k) => k).filter((k) => present(k, r));
    const id = lines.map((_, k) => (present(k, r) ? k : -1));
    const root = (k: number): number => (id[k] === k ? k : (id[k] = root(id[k])));
    for (const a of P) for (const b of P) if (a < b && together(a, b, r)) id[root(a)] = root(b);
    const cid = lines.map((_, k) => (present(k, r) ? root(k) : -1));
    cluster.push(cid);
    const groups = [...new Set(P.map((k) => cid[k]))].map((c) => P.filter((k) => cid[k] === c));
    const run = (a: number, b: number) => { let d = 0; while (r + d < R && d < AHEAD && alike(a, b, r + d)) d++; return d; };
    const soon = (a: number, b: number) => { let s = 0; for (let d = 1; d <= AHEAD && r + d < R; d++) if (alike(a, b, r + d)) s += 1 - d / (AHEAD + 1); return s; };
    const wt = new Map<number, number>();
    for (const a of P) for (const b of P) if (a < b) wt.set(pk(a, b), cid[a] === cid[b] ? run(a, b) : soon(a, b));
    rowsIn.push({ P, groups, wt });
  }
  const order: number[][] = Array.from({ length: R }, (_, r) => rowsIn[r].groups.flat());
  const orderRow = (r: number, nb: (number[] | undefined)[]) => {
    const { P, groups: gs0, wt } = rowsIn[r];
    const pos = nb.filter((x): x is number[] => !!x).map((x) => new Map(x.map((k, i) => [k, i])));
    const near = (k: number) => { for (const p of pos) { const v = p.get(k); if (v !== undefined) return v; } return 99 + k; };
    const groups = gs0.map((g) => [...g].sort((a, b) => near(a) - near(b))).sort((x, y) => near(x[0]) - near(y[0]));
    const cost = (ord: number[]) => {
      let c = 0;
      for (let i = 0; i < ord.length; i++) for (let j = i + 1; j < ord.length; j++) {
        const a = ord[i], b = ord[j];
        c += (j - i) * wt.get(pk(a, b))!;
        for (const p of pos) { const pa = p.get(a), pb = p.get(b); if (pa !== undefined && pb !== undefined && pa > pb) c += CROSS; }
        if (a > b) c += 0.01;
      }
      return c;
    };
    let best = groups.flat();
    const tries = fact(groups.length) * groups.reduce((m, g) => m * fact(g.length), 1);
    if (tries <= MAX_TRIES && P.length > 1) {
      let bc = cost(best);
      const inner = groups.map((g) => perms(g));
      const walk = (gs: number[][], i: number, acc: number[]) => {
        if (i === gs.length) { const c = cost(acc); if (c < bc - 1e-9) { bc = c; best = acc.slice(); } return; }
        for (const p of inner[groups.indexOf(gs[i])]) walk(gs, i + 1, [...acc, ...p]);
      };
      for (const gs of perms(groups)) walk(gs, 0, []);
    }
    order[r] = best;
  };
  for (let r = 0; r < R; r++) orderRow(r, [order[r - 1]]);
  for (let r = R - 1; r >= 0; r--) orderRow(r, [order[r + 1]]);
  for (let r = 0; r < R; r++) orderRow(r, [order[r - 1], order[r + 1]]);

  // ---- x: each cluster tight, clusters SPREAD apart, each line as straight as it can stay
  const xs = lines.map(() => new Array<number>(R).fill(NaN));
  for (let r = 0; r < R; r++) {
    const ord = order[r], cid = cluster[r];
    const cl: number[][] = [];
    for (const k of ord) { if (cl.length && cid[cl[cl.length - 1][0]] === cid[k]) cl[cl.length - 1].push(k); else cl.push([k]); }
    const off = (m: number, i: number) => (i - (m - 1) / 2) * PITCH;
    const want: (number | null)[] = [], w: number[] = [], gap: number[] = [];
    cl.forEach((g, j) => {
      let s = 0, c = 0;
      g.forEach((k, i) => { const p = r ? xs[k][r - 1] : NaN; if (Number.isFinite(p)) { s += p - off(g.length, i); c++; } });
      want.push(c ? s / c : null);
      w.push(c);
      gap.push(j ? ((cl[j - 1].length - 1) / 2) * PITCH + ((g.length - 1) / 2) * PITCH + SPREAD : 0);
    });
    const at = place(want, w, gap);
    cl.forEach((g, j) => g.forEach((k, i) => { xs[k][r] = at[j] + off(g.length, i); }));
  }

  // ---- rows' heights and centres; a bend gets room to run at about 50 degrees
  const rows: GroupRow[] = seq.map((s, r) => ({ cells: s.cells, named: all || s.cells.some((c) => c?.named), order: order[r], y: NaN, h: 0 }));
  let y = PAD, last = -1;
  for (let r = 0; r < R; r++) {
    const row = rows[r];
    const bars = row.named && o.crossings ? Math.min(TALL_MAX, o.crossings(seq[r].key)) : 0;
    // past ROUND_MAX the bullets after the name go in two rows of 16 px
    row.h = row.named ? Math.max(NAME_H, pillHeight(bars, true) + 8, bars > ROUND_MAX ? 38 : 0) : TICK_H;
    if (last >= 0) {
      let bend = 0;
      for (let k = 0; k < n; k++) if (present(k, r) && present(k, last)) bend = Math.max(bend, Math.abs(xs[k][r] - xs[k][last]));
      y += Math.max(0, bend * 0.85 - (rows[last].h + row.h) / 2);
    }
    row.y = y + row.h / 2;
    y += row.h;
    last = r;
  }
  const height = y + PAD;

  // shift so the leftmost bar sits MARGIN in
  let minX = Infinity;
  for (const x of xs.flat()) if (Number.isFinite(x)) minX = Math.min(minX, x);
  if (!Number.isFinite(minX)) return none;
  for (const xk of xs) for (let r = 0; r < R; r++) xk[r] += MARGIN - minX;

  // ---- bars: through the rows, an S-curve where the x changes
  const pts = lines.map((_, k) => {
    const out: [number, number][] = [];
    if (spans[k][0] < 0) return out;
    for (let r = spans[k][0]; r <= spans[k][1]; r++) out.push([xs[k][r], rows[r].y]);
    return out;
  });
  const xAt = (k: number, yy: number) => {
    const p = pts[k];
    if (!p?.length || yy < p[0][1] || yy > p[p.length - 1][1]) return NaN;
    for (let i = 1; i < p.length; i++) {
      if (yy > p[i][1]) continue;
      const [x0, y0] = p[i - 1], [x1, y1] = p[i];
      if (y1 === y0) return x1;
      const t = (yy - y0) / (y1 - y0), s = t * t * (3 - 2 * t); // the S-curve, near enough
      return x0 + (x1 - x0) * s;
    }
    return p[0][0];
  };

  // ---- bullets above a line's first stop and below its last; side by side where they meet, and out
  // to the side where another line runs through
  const bullets: GroupBullet[] = [];
  lines.forEach((_, k) => {
    const [a, b] = spans[k];
    if (a < 0) return;
    bullets.push({ col: k, end: 0, x: xs[k][a], y: rows[a].y - rows[a].h / 2 - CHIP / 2 + 3 });
    bullets.push({ col: k, end: 1, x: xs[k][b], y: rows[b].y + rows[b].h / 2 + CHIP / 2 - 3 });
  });
  const near = new Map<string, GroupBullet[]>();
  for (const bu of bullets) { const key = `${bu.end}:${Math.round(bu.y)}`; near.set(key, [...(near.get(key) ?? []), bu]); }
  for (const g of near.values()) {
    g.sort((p, q) => p.x - q.x);
    // runs of bullets closer than CHIP spread round their mean
    let i = 0;
    while (i < g.length) {
      let j = i;
      while (j + 1 < g.length && g[j + 1].x - g[j].x < CHIP) j++;
      const m = (g[i].x + g[j].x) / 2;
      for (let q = i; q <= j; q++) g[q].x = m + (q - i - (j - i) / 2) * CHIP;
      i = j + 1;
    }
  }
  for (const bu of bullets) {
    for (let tries = 0; tries < 3; tries++) {
      const hit = lines.map((_, j) => xAt(j, bu.y)).filter((x, j) => j !== bu.col && Number.isFinite(x) && Math.abs(x - bu.x) < CHIP / 2 + BAR / 2);
      if (!hit.length) break;
      const mid = hit.reduce((s, x) => s + x, 0) / hit.length;
      bu.x = bu.x < mid ? Math.min(...hit) - CHIP / 2 - BAR / 2 - 1 : Math.max(...hit) + CHIP / 2 + BAR / 2 + 1;
    }
  }
  // room for bullets pushed past the left edge
  const lo = Math.min(...bullets.map((bu) => bu.x - CHIP / 2));
  if (lo < 1) {
    const dx = 1 - lo;
    for (const xk of xs) for (let r = 0; r < R; r++) xk[r] += dx;
    for (const bu of bullets) bu.x += dx;
    for (const p of pts) for (const q of p) q[0] += dx;
  }
  const paths = pts.map((p) => {
    if (!p.length) return "";
    let d = `M${p[0][0]},${p[0][1]}`;
    for (let i = 1; i < p.length; i++) {
      const [x0, y0] = p[i - 1], [x1, y1] = p[i];
      if (Math.abs(x1 - x0) < 0.25) d += ` V${y1}`;
      else { const ym = (y0 + y1) / 2; d += ` C${x0},${ym} ${x1},${ym} ${x1},${y1}`; }
    }
    return d;
  });
  const width = Math.max(...xs.flat().filter(Number.isFinite).map((x) => x + MARGIN), ...bullets.map((bu) => bu.x + CHIP / 2 + 1));
  return { rows, xs, spans, paths, bullets, width: Math.ceil(width), height, together, xAt };
}
