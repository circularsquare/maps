// Lines sharing track drawn side by side (T-1799, notes/T-1799.md): each edge's lines in an order,
// left to right, and each line's sideways shift along its path in line widths, which the stroke and
// train shaders scale by the line width on screen.
//
// The order is the pairwise rule of metro-line crossing minimisation: of two lines on an edge, follow
// both ahead to the node where they part and put the one turning more to the left on the left; if
// they never part ahead, the same looking back; same route both ways, by line id. "Ahead" is the lower
// id line's own direction, whichever way each edge was drawn. One parting decides
// every edge of a stretch they share, so bands cross only at a node where their routes make them.
//
// Side by side is decided per place, not per edge (T-1848): each edge is sampled (corridors.ts), the
// ribbon at a sample is every line on the edges beside it there, and along each line a ribbon that
// lasts less than CORRIDOR_MIN_M is smoothed away, so edges cut at different places on two tracks do
// not make a line step sideways.

import { besideSamples, type Beside } from "./corridors";
import { kAtY } from "./geo";

/** T-1821: the Line view's line width as a multiple of the width the Traffic view's scale is set
 * against (Anita, 2026-10-29: "raise normal line thickness by like 30% relative to traffic mode") */
export const LINE_VIEW_WIDTH = 1.3;
/** a band's width as a share of a line's (Anita, 2026-10-29) */
export const BAND_SHARE = 0.8;
/** px of casing left between two bands side by side, at every zoom (Anita, T-2126: "a little like ~1px gap") */
export const BAND_GAP_PX = 1;
/** line widths a metre across is drawn as, for the gap between the centre lines of tracks side by
 * side (8 m between a four-track trunk's pairs): 0.4 of a line width for 8 m, which is exact near
 * zoom 14.5, 1.4 px of overlap at zoom 13 and 0.9 px of gap at 15, a few px of gap by 16 (a metre is
 * fixed on the ground, a line width is not; notes/T-1799.md, stage 3) */
export const BAND_LW_PER_M = 0.05;
/** where a line's slot changes at a node, its shift eases over this much of the edge, m */
export const BAND_RAMP_M = 30;
/** where the ribbon changes along a track (a track comes beside or leaves), the shift eases over this, m */
export const CORRIDOR_EASE_M = 80;
/** a ribbon that lasts less than this along a line is smoothed over, m: a neighbour track splaying
 * round a platform (up to about 200 m long) or touching at a junction */
export const CORRIDOR_MIN_M = 200;
/** a ribbon wider than this (m, outermost centre lines) is a throat or yard, not a trunk: its tracks
 * are left as they are (a four-track trunk is 8 m across; the Long Island Rail Road's main line with
 * the Port Washington branch beside it at Woodside, seventeen lines on the real New York save, 58 m) */
const SPAN_M = 70;
/** at most this many tracks side by side in one ribbon, against runaway work (the Long Island Rail
 * Road's main line on the real New York save is thirteen branches' edges on top of each other, and
 * with the Port Washington branch beside seventeen); SPAN_M is what keeps yards out */
const MAX_TRACKS = 32;
/** edges whose centre lines are closer than this (m, averaged along) are one bundle of single
 * tracks or lie on top of each other (the real New York save's lines each on their own edge, 4 m
 * apart or less, swapping sides at interlockings), not the double-track edges of a four-track trunk
 * (8 m apart): their order on the ground is not kept */
const COINCIDE_M = 6;
/** a band's shift is stored as a share of this many line widths (`Strokes.band`, Int16 normalised) */
export const BAND_RANGE = 16;
/** numbers per knot of a band profile: where along the path entry (0-1, its travel direction), the
 * shift there (line widths, left of travel), `1 - width share`, and which recipe (-1 between two) */
export const KNOT = 4;
/** shifts closer than this (line widths) along one ribbon are one value (a track's spacing wandering) */
const SHIFT_TOL = 0.05;

/** T-1799 stage 3's corridors: tracks side by side drawn as one ribbon. */
export const CORRIDORS = true;

/** T-2058 (notes/T-2058.md): "smooth" every change of a band's shift a long S-curve, changes that
 * overlap added together; "order" each pair of tracks' order in a ribbon decided once for the
 * network; "fit" a change at a node on the side with more lines where the side with fewer is shorter
 * than its curve; "snap" a ribbon change just ahead of a merge's or split's node placed as at the
 * node (T-2059, T-2060); "" T-1848's eases and per-ribbon order (`?perfOff=bandsSmooth`,
 * `bandsOrder`, `bandsPlace`: `bandTry`) */
let bandMode = "smooth,order,fit,snap";
export function setBandMode(m: string): void {
  if (m === bandMode) return;
  bandMode = m;
  last = null;
}
/** "smooth": a change's length, m, for ribbons of up to two lines, and more per line beyond (all of a
 * ribbon's lines take the same length at a change, so they move together) */
const SMOOTH_M = 250;
const SMOOTH_PER_LINE_M = 50;
const SMOOTH_MAX_M = 800;
/** knots per change, along its S-curve */
const SMOOTH_KNOTS = 9;

interface BandLine { id: number; ok: boolean; path: number[]; bus?: unknown }
interface BandEdge { id: number; a: number; b: number }

/**
 * A line's bands along its path: `[n, start of each path entry's knots (n + 1), knots...]`, KNOT
 * numbers a knot, linear between knots. Each entry has a knot at 0 and at 1.
 */
export type BandProfile = Float32Array;

export interface LineBands {
  /** per line id: its profile; absent for a line on no shared track */
  of: Map<number, BandProfile>;
  /** per edge with two or more lines: their ids left to right in its a to b direction */
  order: Map<number, number[]>;
  /** the pairwise rule on an edge between two of its lines (checks/t1799-bands.mjs): < 0 the first on the left in a to b */
  rule: (edge: number, x: number, y: number) => number;
  /** the Traffic view's recipe for every knot (`trafficBands`), made into one array to post */
  corridor: Float32Array;
}

/** how far along an edge its leaving direction is measured, m, so a short first segment does not decide it */
const BEARING_M = 60;

/** The ribbon at a place, seen from one of its tracks. */
interface Ribbon {
  /** the lines left to right, left of that track's a to b, and the edge each is on */
  lines: number[];
  edges: number[];
  /** its lines in order left to right of that track's a to b; `back` right to left, as seen
   * travelling b to a */
  key: string;
  back: string;
  /** metres from this track's centre line to the ribbon's middle, left of its a to b */
  mid: number;
  /** the tracks and lines in order, shared by every place they are side by side alike; `flip` when
   * this track sees them right to left */
  shape: { rid: [number, number] };
  flip: boolean;
  /** its Traffic recipes by slot and sign, once worked out */
  rc?: number[];
}

/** The bands of every line. `pts`: each edge's local polyline (x, y, z triples, a to b). */
export function lineBands(lines: readonly BandLine[], edges: readonly BandEdge[], pts: ReadonlyMap<number, Float32Array>, opts?: { corridors?: boolean; within?: number; debug?: Map<number, unknown> }): LineBands {
  const ends = new Map(edges.map((e) => [e.id, e]));
  /** per edge: the lines on it, each with its first entry in its path */
  const on = new Map<number, { line: BandLine; k: number }[]>();
  const rail = lines.filter((l) => l.ok && !l.bus && l.path.length);
  for (const l of [...rail].sort((a, b) => a.id - b.id)) {
    l.path.forEach((p, k) => {
      const e = p >> 1;
      let v = on.get(e);
      if (!v) on.set(e, (v = []));
      if (!v.some((x) => x.line === l)) v.push({ line: l, k });
    });
  }
  // the same paths over the same edges as last time (a rebuild for trains or a timetable): the same bands
  let sig = `${opts?.corridors ?? CORRIDORS},${opts?.within ?? ""},${bandMode};`;
  for (const l of rail) sig += `${l.id}:${l.path.join(",")};`;
  const used = [...on.keys()].map((e) => [pts.get(e), ends.get(e)?.a, ends.get(e)?.b] as const);
  if (!opts?.debug && last && last.sig === sig && last.used.length === used.length && used.every((u, i) => u[0] === last!.used[i][0] && u[1] === last!.used[i][1] && u[2] === last!.used[i][2])) return { ...last.out, corridor: last.corridor.slice() };
  const out = bands(rail, on, ends, pts, opts);
  // (a copy: the clock worker transfers the array it posts)
  last = { sig, used, out, corridor: out.corridor.slice() };
  return out;
}

let last: { sig: string; used: (readonly [Float32Array | undefined, number | undefined, number | undefined])[]; out: LineBands; corridor: Float32Array } | null = null;

function bands(rail: BandLine[], on: Map<number, { line: BandLine; k: number }[]>, ends: Map<number, BandEdge>, pts: ReadonlyMap<number, Float32Array>, opts?: { corridors?: boolean; within?: number; debug?: Map<number, unknown> }): LineBands {
  // the unit direction an edge leaves a node in, y up (local y points south)
  const leave = new Map<number, [number, number]>();
  const dirAt = (edge: number, node: number): [number, number] => {
    const key = edge * 2 + (ends.get(edge)?.a === node ? 0 : 1);
    let v = leave.get(key);
    if (v) return v;
    const p = pts.get(edge), e = ends.get(edge);
    v = [1, 0];
    if (p && e && p.length >= 6) {
      const n = p.length / 3, fromA = e.a === node;
      const at = (i: number) => (fromA ? i : n - 1 - i);
      const x0 = p[at(0) * 3], y0 = p[at(0) * 3 + 1];
      let x = x0, y = y0, d = 0;
      const reach = BEARING_M * kAtY(y0);
      for (let i = 1; i < n && d < reach; i++) {
        const nx = p[at(i) * 3], ny = p[at(i) * 3 + 1];
        d += Math.hypot(nx - x, ny - y);
        x = nx;
        y = ny;
      }
      const l = Math.hypot(x - x0, y - y0);
      if (l > 0) v = [(x - x0) / l, -(y - y0) / l];
    }
    leave.set(key, v);
    return v;
  };
  /** how far left (rad) leaving along `out` turns from arriving along `inEdge`, at `node` */
  const turn = (inEdge: number, out: number, node: number) => {
    const [ax, ay] = dirAt(inEdge, node), [bx, by] = dirAt(out, node);
    // arriving is the reverse of leaving along the edge in
    return Math.atan2(-ax * by + ay * bx, -ax * bx - ay * by);
  };
  /** line `l`'s entries walked from entry k in direction `fwd` of its path: the edge and the node it
   * is entered from, step by step; null past the path's end */
  const step = (l: BandLine, k: number, fwd: boolean, j: number): [number, number] | null => {
    const i = fwd ? k + j : k - j;
    if (i < 0 || i >= l.path.length) return null;
    const p = l.path[i], e = ends.get(p >> 1);
    if (!e) return null;
    // entered from a when run forwards a to b (low bit clear), or walked backwards b to a
    const fromA = ((p & 1) === 0) === fwd;
    return [p >> 1, fromA ? e.a : e.b];
  };
  /** of two lines on `edge`, walking towards `toB` (its b end) or away: < 0 when `x` turns left of
   * `y` where they part, > 0 right, 0 if they never part */
  const part = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }, toB: boolean): number => {
    // each line's own direction on this edge: walk its path forwards when it runs the way we walk
    const fx = ((x.line.path[x.k] & 1) === 0) === toB, fy = ((y.line.path[y.k] & 1) === 0) === toB;
    let prev = edge;
    const max = Math.min(x.line.path.length, y.line.path.length);
    for (let j = 1; j <= max; j++) {
      const a = step(x.line, x.k, fx, j), b = step(y.line, y.k, fy, j);
      if (!a || !b) return 0;
      if (a[0] !== b[0] || a[1] !== b[1]) {
        if (a[1] !== b[1]) return 0;
        const ta = turn(prev, a[0], a[1]), tb = turn(prev, b[0], b[1]);
        return ta > tb ? -1 : ta < tb ? 1 : 0;
      }
      prev = a[0];
    }
    return 0;
  };
  const order = new Map<number, number[]>();
  /** < 0: `x` left of `y` in the edge's a to b direction. Looked at first ahead in the lower-id
   * line's own direction, so every edge of a stretch the two share asks the same parting node
   * whatever way each edge was drawn; a walk towards a sees left of travel as right of a to b. */
  const compare = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }) => {
    const lead = x.line.id < y.line.id ? x : y;
    const toB = (lead.line.path[lead.k] & 1) === 0;
    const first = part(edge, x, y, toB), second = first ? 0 : part(edge, x, y, !toB);
    // never parting: the lower id on the left of the lead line's travel
    return (toB ? first : -first) || (toB ? -second : second) || (toB ? 1 : -1) * (x.line.id - y.line.id);
  };
  for (const [edge, v] of on) {
    if (v.length < 2) continue;
    // by how many of the others each is left of: the rule's own order where it is one, and where
    // three or more lines' partings go round in a circle (some pair has to cross somewhere), a
    // fixed order rather than whatever a sort makes of a comparison that contradicts itself
    const m = v.length, wins = new Int32Array(m), c = new Int8Array(m * m);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
      const r = Math.sign(compare(edge, v[i], v[j]));
      c[i * m + j] = r;
      c[j * m + i] = -r;
      if (r < 0) wins[i]++;
      else if (r > 0) wins[j]++;
    }
    const idx = v.map((_, i) => i).sort((i, j) => wins[j] - wins[i] || c[i * m + j] || v[i].line.id - v[j].line.id);
    order.set(edge, idx.map((i) => v[i].line.id));
  }
  const rule = (edge: number, x: number, y: number) => {
    const v = on.get(edge), a = v?.find((s) => s.line.id === x), b = v?.find((s) => s.line.id === y);
    return a && b ? compare(edge, a, b) : 0;
  };
  const linesOf = (e: number) => order.get(e) ?? on.get(e)!.map((x) => x.line.id);

  // ---- the ribbon at each place
  const useCorridors = (opts?.corridors ?? CORRIDORS) && on.size > 1;
  const beside: Beside = useCorridors ? besideSamples([...on.keys()], pts, ends, opts?.within) : { edges: new Map(), pair: new Map() };
  const samples = beside.edges;
  const lenOf = new Map<number, number>();
  const edgeLen = (e: number) => {
    let v = lenOf.get(e);
    if (v !== undefined) return v;
    const s = samples.get(e);
    if (s) v = s.len;
    else {
      const p = pts.get(e);
      v = 0;
      if (p) for (let i = 3; i < p.length; i += 3) v += Math.hypot(p[i] - p[i - 3], p[i + 1] - p[i - 2]);
      v = p && p.length >= 6 ? v / kAtY(p[1]) : 0;
    }
    lenOf.set(e, v);
    return v;
  };
  const keysOf = (ids: readonly number[]) => ({ key: ids.join(","), back: [...ids].reverse().join(",") });
  const own = new Map<number, Ribbon>();
  const ownRibbon = (e: number): Ribbon => {
    let r = own.get(e);
    if (!r) {
      const ids = linesOf(e), edges = ids.map(() => e), { key, back } = keysOf(ids);
      const shape: Shape = { ids, edges, rids: [...ids].reverse(), redges: edges, key, back, rid: [-1, -1] };
      own.set(e, (r = { shape, flip: false, lines: ids, edges, key, back, mid: 0, rc: undefined }));
    }
    return r;
  };
  const ribbons = new Map<number, (Ribbon | undefined)[]>();
  /** keep ribbon r for sample i of edge e unless it has one */
  const put = (e: number, i: number, r: Ribbon) => {
    let memo = ribbons.get(e);
    if (!memo) ribbons.set(e, (memo = new Array(samples.get(e)?.n ?? 1)));
    if (!memo[i]) memo[i] = r;
  };
  /** the ribbon at sample i of edge e: the tracks side by side there, passed on from track to track */
  const ribbonAt = (e: number, i: number): Ribbon => {
    const S = samples.get(e);
    if (!S || S.first[i] === S.first[i + 1]) return ownRibbon(e);
    let memo = ribbons.get(e);
    if (!memo) ribbons.set(e, (memo = new Array(S.n)));
    const old = memo[i];
    if (old) return old;
    const all = [{ e, i, off: 0, rel: 1 }];
    const seen = new Set([e]);
    let big = false;
    for (let q = 0; q < all.length && !big; q++) {
      const M = all[q], T = samples.get(M.e)!;
      for (let j = T.first[M.i]; j < T.first[M.i + 1]; j++) {
        const f = T.edge[j];
        if (seen.has(f) || !on.has(f)) continue;
        seen.add(f);
        all.push({ e: f, i: T.at[j], off: M.off + M.rel * T.off[j], rel: M.rel * T.rel[j] });
        if (all.length > 2 * MAX_TRACKS) big = true;
      }
    }
    // a line is in a ribbon once: of two edges with a line in common (two stretches of one track
    // meeting at a node here) the lower id is in, whichever track the ribbon is seen from
    const has = new Set<number>();
    const mem = [...all].sort((a, b) => a.e - b.e).filter((m) => {
      const v = linesOf(m.e);
      if (v.some((id) => has.has(id))) return false;
      for (const id of v) has.add(id);
      return true;
    });
    let lo = Infinity, hi = -Infinity;
    for (const m of mem) {
      lo = Math.min(lo, m.off);
      hi = Math.max(hi, m.off);
    }
    if (big || mem.length > MAX_TRACKS || hi - lo > SPAN_M || mem.length < 2 || !mem.some((m) => m.e === e)) {
      // the same for each track of it at this place, which need not work it out again
      for (const m of all) put(m.e, m.i, ownRibbon(m.e));
      return (memo[i] = ownRibbon(e));
    }
    // worked out in the travel direction of the ribbon's lowest id line, the same from whichever
    // track and along the line from one edge to the next
    let c = mem[0], low = Infinity;
    for (const m of mem) {
      const x = lowest(m.e);
      if (x < low) (low = x), (c = m);
    }
    const lead = on.get(c.e)!.find((x) => x.line.id === low)!;
    const cw = c.rel * ((lead.line.path[lead.k] & 1) === 1 ? -1 : 1);
    let sig = "";
    for (const m of mem) sig += `${m.e}${m.rel * cw > 0 ? "+" : "-"}`;
    let sh = shapes.get(sig);
    if (!sh) shapes.set(sig, (sh = shapeOf(mem.map((m) => [m.e, m.rel * cw]))));
    const mid = (cw * (lo + hi)) / 2;
    // the ribbon seen from each of its tracks there, in that track's frame
    for (const m of mem) {
      const fwd = m.rel * cw > 0, mm = mid - cw * m.off;
      const v: Ribbon = fwd
        ? { shape: sh, flip: false, lines: sh.ids, edges: sh.edges, key: sh.key, back: sh.back, mid: mm, rc: undefined }
        : { shape: sh, flip: true, lines: sh.rids, edges: sh.redges, key: sh.back, back: sh.key, mid: -mm, rc: undefined };
      if (m.e === e) memo[i] = v;
      else put(m.e, m.i, v);
    }
    return memo[i]!;
  };
  interface Shape { ids: number[]; edges: number[]; rids: number[]; redges: number[]; key: string; back: string; rid: [number, number] }
  const shapes = new Map<string, Shape>();
  const lowest = (e: number) => Math.min(...linesOf(e));
  // T-2058 "order": which of two tracks' lines is on the left, decided once for the whole network
  /** +1 a to b along edge e for line id, -1 b to a */
  const dirOn = (e: number, id: number) => {
    const x = on.get(e)!.find((s) => s.line.id === id)!;
    return (x.line.path[x.k] & 1) === 1 ? -1 : 1;
  };
  /** per pair of lines `l,m` (l < m, each the lowest line of a track): metres m's track is left of
   * l's in l's travel, summed over every place the two are beside, and the places */
  let pairSums: Map<string, [number, number]> | null = null;
  /** +1: line l is left of line m (l < m) in l's travel; 0 no place they are beside; ties within
   * COINCIDE_M: +1, the lower number on the left of its travel, wherever the two meet */
  const relOf = (l: number, m: number): number => {
    if (!pairSums) {
      pairSums = new Map();
      for (const [e, nb] of beside.pair) {
        if (!on.has(e)) continue;
        const le = lowest(e);
        for (const [f, x] of nb) {
          if (!on.has(f)) continue;
          const lf = lowest(f);
          if (le >= lf) continue;
          const key = `${le},${lf}`, s = pairSums.get(key) ?? [0, 0];
          s[0] += x[0] * dirOn(e, le) * x[2];
          s[1] += x[2];
          pairSums.set(key, s);
        }
      }
    }
    const s = pairSums.get(`${l},${m}`);
    if (!s) return 0;
    return Math.abs(s[0] / s[1]) < COINCIDE_M ? 1 : s[0] > 0 ? -1 : 1;
  };
  /** a ribbon's tracks left to right of its frame by the network-wide pairs (Copeland: by how many
   * of the others each is left of), the ground's order where two tracks are never beside */
  const networkOrder = (mem: [number, number][], pos: Map<number, number>): [number, number][] => {
    const k = mem.length, wins = new Int32Array(k), fr = mem[0][1] < 0 ? -1 : 1;
    const lo = mem.map((m) => lowest(m[0])), sg = mem.map((m, i) => m[1] * dirOn(m[0], lo[i]));
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
      let left: boolean;
      const a = lo[i] < lo[j] ? i : j, b = a === i ? j : i, r = lo[a] === lo[b] ? 0 : relOf(lo[a], lo[b]);
      if (r) left = (r * sg[a] > 0) === (a === i);
      else {
        const d = ((pos.get(mem[i][0]) ?? 0) - (pos.get(mem[j][0]) ?? 0)) * fr;
        left = Math.abs(d) >= COINCIDE_M ? d > 0 : (sg[a] > 0) === (a === i);
      }
      if (left) wins[i]++;
      else wins[j]++;
    }
    return mem.map((m, i) => [m, i] as const).sort((x, y) => wins[y[1]] - wins[x[1]] || lo[x[1]] - lo[y[1]]).map((x) => x[0]);
  };
  /** a ribbon's tracks (edge, direction against the frame) in order left to right of the frame (the
   * travel of its lowest id line): each track's place from every pair's offset averaged over all the
   * places the two are side by side, so the order holds along the ribbon. Tracks within COINCIDE_M
   * of each other (as on the real New York save, where each line has its own edge) have no order on
   * the ground worth keeping: by line id. */
  const shapeOf = (mem: [number, number][]): Shape => {
    const pos = new Map<number, number>([[mem[0][0], 0]]), way = new Map<number, number>([[mem[0][0], 1]]);
    const queue = [mem[0][0]], inMem = new Set(mem.map((m) => m[0]));
    for (let q = 0; q < queue.length; q++) {
      const a = queue[q], pa = pos.get(a)!, ra = way.get(a)!;
      const nb = beside.pair.get(a);
      if (!nb) continue;
      for (const f of [...inMem].sort((x, y) => x - y)) {
        const x = nb.get(f);
        if (!x || pos.has(f)) continue;
        pos.set(f, pa + ra * x[0]);
        way.set(f, ra * x[1]);
        queue.push(f);
      }
    }
    // left to right of the first's a to b, then of the frame `mem` is in (its directions are against that)
    let order = [...mem].sort((a, b) => (pos.get(b[0]) ?? 0) - (pos.get(a[0]) ?? 0) || lowest(a[0]) - lowest(b[0]));
    if (mem[0][1] < 0) order.reverse();
    if (bandMode.includes("order")) order = networkOrder(mem, pos);
    else for (let c0 = 0; c0 < order.length; ) {
      let c1 = c0 + 1;
      while (c1 < order.length && Math.abs((pos.get(order[c1 - 1][0]) ?? 0) - (pos.get(order[c1][0]) ?? 0)) < COINCIDE_M) c1++;
      // by line id, left to right of the lowest id line's travel: a new track joining the bundle
      // leaves the others' order as it was
      if (c1 - c0 > 1) order.splice(c0, c1 - c0, ...order.slice(c0, c1).sort((a, b) => lowest(a[0]) - lowest(b[0])));
      c0 = c1;
    }
    const ids: number[] = [], edges: number[] = [];
    for (const [f, rel] of order) for (const id of rel > 0 ? linesOf(f) : [...linesOf(f)].reverse()) {
      ids.push(id);
      edges.push(f);
    }
    const { key, back } = keysOf(ids);
    return { ids, edges, rids: [...ids].reverse(), redges: [...edges].reverse(), key, back, rid: [-1, -1] };
  };
  /** a line's shift left of e's a to b in ribbon r, line widths: its slot, the ribbon centred on its
   * middle, from the line's own track */
  const shiftIn = (r: Ribbon, slot: number) => BAND_SHARE * ((r.lines.length - 1) / 2 - slot) + r.mid * BAND_LW_PER_M;

  // ---- the Traffic view's table: ribbons, recipes, and per line and edge its knots' recipes
  const tRibbons: number[] = [];
  let nRibbons = 0;
  const ridOf = (r: Ribbon) => {
    const f = r.flip ? 1 : 0;
    if (r.shape.rid[f] < 0) {
      r.shape.rid[f] = nRibbons++;
      tRibbons.push(r.lines.length);
      r.lines.forEach((id, s) => tRibbons.push(id, r.edges[s]));
    }
    return r.shape.rid[f];
  };
  const tRecipes: number[] = [];
  const recipeIds = new Map<number, Map<number, number>>();
  const recipeOf = (r: Ribbon, slot: number, sgn: number) => {
    const memo = (r.rc ??= []), mk = slot * 2 + (sgn > 0 ? 1 : 0);
    if (memo[mk] !== undefined) return memo[mk];
    const rid = ridOf(r), mid = Math.fround(r.mid * BAND_LW_PER_M), key = (rid * 4096 + slot) * 2 + (sgn > 0 ? 1 : 0);
    let byMid = recipeIds.get(key);
    if (!byMid) recipeIds.set(key, (byMid = new Map()));
    let id = byMid.get(mid);
    if (id === undefined) {
      tRecipes.push(rid, slot, sgn, mid);
      byMid.set(mid, (id = tRecipes.length / RECIPE - 1));
    }
    return (memo[mk] = id);
  };
  // (filled in place: about 3M numbers on the real New York save)
  let tRecords = new Float32Array(1 << 16), nRec = 0;
  const need = (k: number) => {
    if (nRec + k <= tRecords.length) return;
    const b = new Float32Array(Math.max(tRecords.length * 2, nRec + k));
    b.set(tRecords.subarray(0, nRec));
    tRecords = b;
  };
  let nRecords = 0;

  // ---- along each line
  const of = new Map<number, BandProfile>();
  for (const l of rail) {
    const n = l.path.length;
    // its samples in travel order: metres along its path, half their spacing, entry, ribbon, slot
    const M: number[] = [], H: number[] = [], K: number[] = [], R: Ribbon[] = [], S: number[] = [], dbgF: number[] = [];
    const start = new Float64Array(n + 1);
    let any = false;
    for (let k = 0; k < n; k++) {
      const p = l.path[k], e = p >> 1, rev = (p & 1) === 1, len = edgeLen(e);
      start[k + 1] = start[k] + len;
      if (!on.has(e)) continue;
      const T = samples.get(e), ns = T?.n ?? 1;
      let alone = -1;
      for (let s = 0; s < ns; s++) {
        const i = rev ? ns - 1 - s : s;
        // samples with nothing beside, one after another, are one (long edges in open country)
        if (T && T.first[i] === T.first[i + 1] && alone >= 0 && !opts?.debug) {
          const m1 = start[k] + ((s + 1) / ns) * len, m0 = M[alone] - H[alone];
          M[alone] = (m0 + m1) / 2;
          H[alone] = (m1 - m0) / 2;
          continue;
        }
        const r = ribbonAt(e, i);
        let slot = -1;
        for (let q = 0; q < r.lines.length && slot < 0; q++) if (r.lines[q] === l.id && r.edges[q] === e) slot = q;
        if (slot < 0) continue;
        if (r.lines.length > 1) any = true;
        if (opts?.debug) dbgF.push((i + 0.5) / ns);
        alone = T && T.first[i] === T.first[i + 1] ? M.length : -1;
        M.push(start[k] + ((s + 0.5) / ns) * len);
        H.push(len / ns / 2);
        K.push(k);
        R.push(r);
        S.push(slot);
      }
    }
    if (!any) continue;
    const N = M.length, sgnAt = (j: number) => ((l.path[K[j]] & 1) === 1 ? -1 : 1);
    const V = new Float64Array(N), KEY: string[] = new Array(N);
    for (let j = 0; j < N; j++) {
      const sg = sgnAt(j);
      V[j] = sg * shiftIn(R[j], S[j]);
      // its lines in order left to right of travel
      KEY[j] = sg > 0 ? R[j].key : R[j].back;
    }
    const val = (j: number) => V[j];
    // runs of one ribbon along the path
    type Run = { key: string; s0: number; s1: number; r: Ribbon };
    const runs: Run[] = [];
    for (let j = 0; j < N; j++) {
      const last = runs[runs.length - 1];
      if (last && last.key === KEY[j]) last.s1 = j;
      else runs.push({ key: KEY[j], s0: j, s1: j, r: R[j] });
    }
    const dropped = new Uint8Array(N);
    const lenRun = (u: Run) => M[u.s1] + H[u.s1] - (M[u.s0] - H[u.s0]);
    /** whether every line on the line's own tracks along run u is in the ribbon of run x */
    const keeps = (u: Run, x: Run | undefined) => {
      if (!x) return true;
      for (let j = u.s0; j <= u.s1; j++) for (const id of linesOf(R[j].edges[S[j]])) if (!x.r.lines.includes(id)) return false;
      return true;
    };
    // a short ribbon goes, its stretch then eased across from the ribbons each side, unless a line
    // on the line's own track there would go missing from them (lines on one edge always side by side)
    for (;;) {
      let best = -1;
      for (let u = 0; u < runs.length && runs.length > 1; u++) {
        if (lenRun(runs[u]) >= CORRIDOR_MIN_M || (best >= 0 && lenRun(runs[u]) >= lenRun(runs[best]))) continue;
        if (keeps(runs[u], runs[u - 1]) && keeps(runs[u], runs[u + 1])) best = u;
      }
      if (best < 0) break;
      const u = runs[best];
      for (let j = u.s0; j <= u.s1; j++) dropped[j] = 1;
      const a = runs[best - 1], b = runs[best + 1];
      if (a && b && a.key === b.key) {
        a.s1 = b.s1;
        runs.splice(best, 2);
      } else runs.splice(best, 1);
    }
    opts?.debug?.set(l.id, { M, f: dbgF, key: KEY, v: [...V], entry: K, dropped: [...dropped], runs: runs.map((u) => [u.key, u.s0, u.s1]) });
    // knots in path metres: per run its value, eased into the next over CORRIDOR_EASE_M, or at a
    // node between different lines over BAND_RAMP_M on the side with fewer (T-1799 stage 4)
    // (each knot's Traffic recipes: recipe and weight pairs in kf, from ko, kn of them)
    const km: number[] = [], kv: number[] = [], kq: number[] = [], ko: number[] = [], kn: number[] = [], kf: number[] = [];
    const push = (m: number, j: number) => {
      if (km.length && m <= km[km.length - 1] + 1e-3) m = km[km.length - 1] + 1e-3;
      km.push(m);
      kv.push(val(j));
      kq.push(R[j].lines.length > 1 ? 1 - BAND_SHARE : 0);
      ko.push(kf.length);
      kn.push(1);
      kf.push(recipeOf(R[j], S[j], sgnAt(j)), 1);
    };
    const ease: [number, number][] = runs.map(() => [0, start[n]]);
    for (let u = 0; u + 1 < runs.length; u++) {
      const a = runs[u], b = runs[u + 1];
      // the last sample of a and first of b not smoothed away
      let ja = a.s1, jb = b.s0;
      while (ja > a.s0 && dropped[ja]) ja--;
      while (jb < b.s1 && dropped[jb]) jb++;
      const g0 = M[ja] + H[ja], g1 = M[jb] - H[jb];
      const la = lenRun(a), lb = lenRun(b);
      if (g1 - g0 < 0.5 && K[ja] !== K[jb]) {
        const ra = Math.min(BAND_RAMP_M, la / 2), rb = Math.min(BAND_RAMP_M, lb / 2);
        ease[u][1] = a.r.lines.length > b.r.lines.length ? g0 : g0 - ra;
        ease[u + 1][0] = b.r.lines.length > a.r.lines.length ? g1 : g1 + rb;
      } else {
        const c = (g0 + g1) / 2, half = Math.max(CORRIDOR_EASE_M, g1 - g0) / 2;
        ease[u][1] = Math.max(c - half, M[a.s0] - H[a.s0] + la / 2);
        ease[u + 1][0] = Math.min(c + half, M[b.s1] + H[b.s1] - lb / 2);
      }
    }
    /** T-2058 "smooth": the pieces' values as steps, each eased over a long S-curve and added up, so
     * changes closer together than their length make one smooth curve rather than steps */
    const smoothKnots = (byRun: [number, number, number, number][][], rs: Run[]) => {
      const qOf = (j: number) => (R[j].lines.length > 1 ? 1 - BAND_SHARE : 0);
      const pl: { a: number; b: number; j0: number; j1: number; first: number; last: number; run: number; rc: number }[] = [];
      byRun.forEach((ps, ui) => {
        for (const p of ps) pl.push({ a: M[p[0]] - H[p[0]], b: M[p[1]] + H[p[1]], first: p[0], last: p[1], j0: p[2], j1: p[3], run: ui, rc: recipeOf(R[p[2]], S[p[2]], sgnAt(p[2])) });
      });
      if (!pl.length) return;
      // a step between each two pieces (pl[i] to pl[i + 1]): its middle and length, m along the path
      const steps: { c: number; L: number }[] = [];
      for (let i = 0; i + 1 < pl.length; i++) {
        const A = pl[i], B = pl[i + 1];
        // (each piece one value, its first's, so the steps add up to every piece's own value)
        // (two pieces alike: a step of no length, which changes nothing)
        if (A.rc === B.rc && Math.abs(val(B.j0) - val(A.j0)) < 1e-6 && Math.abs(qOf(B.j0) - qOf(A.j0)) < 1e-6) {
          steps.push({ c: (A.b + B.a) / 2, L: 0 });
          continue;
        }
        const ra = rs[A.run].r.lines.length, rb = rs[B.run].r.lines.length;
        // (a node between them even where samples smoothed away leave a gap: the node is where B's entry starts)
        const node = A.run !== B.run && K[A.last] !== K[B.first], g0 = node ? start[K[B.first]] : A.b, g1 = node ? g0 : B.a;
        const L = Math.min(SMOOTH_MAX_M, SMOOTH_M + SMOOTH_PER_LINE_M * Math.max(0, Math.max(R[A.j1].lines.length, R[B.j0].lines.length) - 2));
        let c = (g0 + g1) / 2;
        // at a node between ribbons of different lines: on the side with fewer, from the node (stage 4),
        // counted first on the line's own edges (a ribbon near a junction takes in the tracks beside)
        const linesAt = (k: number) => on.get(l.path[k] >> 1)?.length ?? 1;
        let oa = linesAt(K[A.last]), ob = linesAt(K[B.first]), nodeM = node ? g0 : -1;
        if (!node && A.run !== B.run && ra !== rb && bandMode.includes("snap")) {
          // tracks coming beside along the way that join one edge soon after, or leave one edge just
          // before (a merge or split whose ribbon forms ahead of its node): placed as at that node, so
          // the lines are in their places where the tracks meet (T-2059, the Jersey City merge)
          if (rb > ra) {
            for (let k = K[B.first] + 1; k < n && start[k] < g1 + L / 2 && nodeM < 0; k++) if (linesAt(k) !== linesAt(k - 1)) (nodeM = start[k]), (oa = linesAt(k - 1)), (ob = linesAt(k));
          } else for (let k = K[A.last]; k > 0 && start[k] > g0 - L / 2 && nodeM < 0; k--) if (linesAt(k) !== linesAt(k - 1)) (nodeM = start[k]), (oa = linesAt(k - 1)), (ob = linesAt(k));
        }
        if (nodeM >= 0 && oa !== ob) c = oa > ob ? nodeM + L / 2 : nodeM - L / 2;
        else if (node && ra !== rb) {
          // on the side with fewer lines where the curve fits in that run, else on the side with
          // more (a track beside or apart for less than the curve's length, T-2060)
          const fewer = ra > rb ? 1 : -1, fits = !bandMode.includes("fit") || lenRun(rs[ra > rb ? B.run : A.run]) >= L;
          c += (fits ? fewer : -fewer) * (L / 2);
        }
        steps.push({ c, L });
      }
      const end = start[n];
      const at: number[] = [0, end];
      for (const s of steps) if (s.L > 0) for (let i = 0; i < SMOOTH_KNOTS; i++) {
        const m = s.c + s.L * (i / (SMOOTH_KNOTS - 1) - 0.5);
        if (m > 0 && m < end) at.push(Math.round(m * 100) / 100);
      }
      const ms = Float64Array.from(at).sort();
      const S3 = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
      // each knot a mix of the pieces: piece p weighs how far the step into it has gone less how far
      // the step out of it has, so the Traffic view's widths mix exactly as the Line view's shifts
      // (the first piece, plus per step k its S times piece k + 1 less piece k: swept along, the steps
      // passed folded into `base`, only those under way worked out at each knot)
      // (T-2059: in flat arrays, each list kept in the order a Map or Set would keep it, so the sums
      // are added up in the same order as before)
      const P = pl.length, pv = new Float64Array(P), pq = new Float64Array(P), prc = new Int32Array(P);
      for (let p = 0; p < P; p++) {
        pv[p] = val(pl[p].j0);
        pq[p] = qOf(pl[p].j0);
        prc[p] = pl[p].rc;
      }
      const ns = steps.length, lo = steps.map((s) => s.c - s.L / 2), hi = steps.map((s) => s.c + s.L / 2);
      const byLo = [...steps.keys()].sort((a, b) => lo[a] - lo[b]), byHi = [...steps.keys()].sort((a, b) => hi[a] - hi[b]);
      /** the steps passed, per piece, and the pieces whose sum is not zero */
      const base = new Float64Array(P), bk: number[] = [0];
      base[0] = 1;
      const bput = (p: number, w: number) => {
        const x = base[p] + w;
        if (Math.abs(x) < 1e-9) {
          if (base[p] !== 0) bk.splice(bk.indexOf(p), 1);
          base[p] = 0;
        } else {
          if (base[p] === 0) bk.push(p);
          base[p] = x;
        }
      };
      const going: number[] = [], inGo = new Uint8Array(ns);
      const W = new Float64Array(P), mark = new Uint8Array(P), touched: number[] = [];
      const touch = (p: number, w: number) => {
        if (!mark[p]) {
          mark[p] = 1;
          touched.push(p);
        }
        W[p] += w;
      };
      let iL = 0, iH = 0;
      for (let x = 0; x < ms.length; x++) {
        const m = ms[x];
        if (x && m === ms[x - 1]) continue;
        while (iL < ns && lo[byLo[iL]] < m) {
          const k = byLo[iL++];
          inGo[k] = 1;
          going.push(k);
        }
        while (iH < ns && hi[byHi[iH]] <= m) {
          const k = byHi[iH++];
          if (inGo[k]) {
            inGo[k] = 0;
            going.splice(going.indexOf(k), 1);
          }
          bput(k + 1, 1);
          bput(k, -1);
        }
        if (km.length && m <= km[km.length - 1] + 1e-3) continue;
        touched.length = 0;
        for (const p of bk) touch(p, base[p]);
        for (const k of going) {
          const s = S3((m - steps[k].c) / steps[k].L + 0.5);
          touch(k + 1, s);
          touch(k, -s);
        }
        let v = 0, q = 0;
        const o = kf.length;
        for (const p of touched) {
          const w = W[p];
          W[p] = 0;
          mark[p] = 0;
          if (Math.abs(w) < 1e-9) continue;
          v += w * pv[p];
          q += w * pq[p];
          // its recipe's weight, summed over the pieces sharing it
          let i = o;
          while (i < kf.length && kf[i] !== prc[p]) i += 2;
          if (i < kf.length) kf[i + 1] += w;
          else kf.push(prc[p], w);
        }
        let e = o;
        for (let i = o; i < kf.length; i += 2) if (Math.abs(kf[i + 1]) > 1e-6) (kf[e] = kf[i]), (kf[e + 1] = kf[i + 1]), (e += 2);
        kf.length = e;
        if (e === o) kf.push(prc[0], 1);
        km.push(m);
        kv.push(v);
        kq.push(Math.max(0, q));
        ko.push(o);
        kn.push((kf.length - o) / 2);
      }
    };
    /** run u's own samples in pieces of one value: [first, last sample, the samples whose values they take] */
    const piecesOf = (u: Run) => {
      // each sample stands for the median of the five around it on its edge, so one sample where a
      // track is just coming beside (the middle not yet settled) is no bump
      const kept: number[] = [];
      for (let j = u.s0; j <= u.s1; j++) if (!dropped[j]) kept.push(j);
      const medOf = (x: number) => {
        const j = kept[x], a = kept[x - 1], b = kept[x + 1];
        // (only where it stands out from a neighbour)
        if ((a === undefined || Math.abs(V[a] - V[j]) <= SHIFT_TOL) && (b === undefined || Math.abs(V[b] - V[j]) <= SHIFT_TOL)) return j;
        const w: number[] = [];
        for (let y = Math.max(0, x - 2); y <= Math.min(kept.length - 1, x + 2); y++) if (K[kept[y]] === K[j]) w.push(kept[y]);
        w.sort((p, q) => V[p] - V[q]);
        return w[w.length >> 1];
      };
      // its own samples in pieces of one value: [first, last sample, the samples whose values they take]
      const pieces: [number, number, number, number][] = [];
      for (let x = 0; x < kept.length; x++) {
        const j0 = kept[x], j = medOf(x);
        // (a new piece at each node too, where the Traffic view's widths may change)
        const p = pieces[pieces.length - 1];
        if (p && Math.abs(val(j) - val(p[2])) <= SHIFT_TOL && K[j0] === K[p[0]]) (p[1] = j0), (p[3] = j);
        else pieces.push([j0, j0, j, j]);
      }
      return pieces;
    };
    if (bandMode) smoothKnots(runs.map(piecesOf), runs);
    else runs.forEach((u, ui) => {
      const [t0, t1] = ease[ui];
      const pieces = piecesOf(u);
      if (!pieces.length) return;
      let first = pieces.findIndex((p) => M[p[1]] >= t0);
      if (first < 0) first = pieces.length - 1;
      let last = pieces.length - 1;
      while (last > first && M[pieces[last][0]] > t1) last--;
      push(t0, pieces[first][2]);
      for (let q = first; q < last; q++) {
        const ma = M[pieces[q][1]], mb = M[pieces[q + 1][0]];
        if (ma > t0 && ma < t1) push(ma, pieces[q][3]);
        if (mb > t0 && mb < t1) push(mb, pieces[q + 1][2]);
      }
      push(Math.max(t1, km[km.length - 1] + 1e-3), pieces[last][3]);
    });
    // per path entry: its knots, with one at each end where it lies between two
    const knots: number[] = [];
    const starts: number[] = [];
    const recMark = nRec, recCount = nRecords;
    let q = 0, ki = 0, nonzero = false;
    /** knot i's recipe list as an array */
    const listOf = (i: number) => kf.slice(ko[i], ko[i] + 2 * kn[i]);
    /** knots i and i + 1's recipe lists mixed, t of the way */
    const mixed = (i: number, t: number) => {
      const a = ko[i], na = 2 * kn[i], b = ko[i + 1], nb = 2 * kn[i + 1];
      if (t <= 0 || (na === 2 && nb === 2 && kf[a] === kf[b] && kf[a + 1] === kf[b + 1])) return listOf(i);
      const m = new Map<number, number>();
      for (let x = a; x < a + na; x += 2) m.set(kf[x], (m.get(kf[x]) ?? 0) + kf[x + 1] * (1 - t));
      for (let x = b; x < b + nb; x += 2) m.set(kf[x], (m.get(kf[x]) ?? 0) + kf[x + 1] * t);
      const out: number[] = [];
      for (const [r, w] of m) if (Math.abs(w) > 1e-6) out.push(r, w);
      return out;
    };
    const at = (m: number): [number, number, number[]] => {
      while (q + 1 < km.length && km[q + 1] < m) q++;
      if (m <= km[0]) return [kv[0], kq[0], listOf(0)];
      const z = km.length - 1;
      if (q + 1 >= km.length || m >= km[z]) return [kv[z], kq[z], listOf(z)];
      const t = (m - km[q]) / (km[q + 1] - km[q]);
      return [kv[q] + (kv[q + 1] - kv[q]) * t, kq[q] + (kq[q + 1] - kq[q]) * t, mixed(q, t)];
    };
    for (let k = 0; k < n; k++) {
      starts.push(knots.length / KNOT);
      const m0 = start[k], m1 = start[k + 1], len = m1 - m0;
      const p = l.path[k], e = p >> 1;
      const ed = ends.get(e), P = pts.get(e);
      const rec = !!(ed && P && P.length >= 6), head = nRec;
      if (rec) {
        need(6);
        tRecords[nRec++] = l.id;
        tRecords[nRec++] = e;
        tRecords[nRec++] = p & 1;
        tRecords[nRec++] = P![0];
        tRecords[nRec++] = P![1];
        nRec++;
      }
      let nk = 0;
      const add = (f: number, v: number, w: number, list: readonly number[], o: number, c: number) => {
        // (the one recipe when it is one alone, else -1: between two, for `bandPoints`' eases)
        knots.push(f, v, w, c === 1 && list[o + 1] === 1 ? list[o] : -1);
        if (rec) {
          need(2 + 2 * c);
          tRecords[nRec++] = f;
          tRecords[nRec++] = c;
          for (let x = o; x < o + 2 * c; x++) tRecords[nRec++] = list[x];
        }
        nk++;
        if (Math.abs(v) > 1e-6 || w > 0) nonzero = true;
      };
      const a0 = at(m0);
      add(0, a0[0], a0[1], a0[2], 0, a0[2].length / 2);
      while (ki < km.length && km[ki] <= m0 + 1e-3) ki++;
      for (; ki < km.length && km[ki] < m1 - 1e-3; ki++) add(len > 0 ? (km[ki] - m0) / len : 0, kv[ki], kq[ki], kf, ko[ki], kn[ki]);
      const a1 = at(m1);
      add(1, a1[0], a1[1], a1[2], 0, a1[2].length / 2);
      if (rec) {
        tRecords[head + 5] = nk;
        nRecords++;
      }
    }
    if (!nonzero) {
      nRec = recMark;
      nRecords = recCount;
      continue;
    }
    starts.push(knots.length / KNOT);
    const out = new Float32Array(1 + starts.length + knots.length);
    out[0] = n;
    out.set(starts, 1);
    out.set(knots, 1 + starts.length);
    of.set(l.id, out);
  }
  const corridor = new Float32Array(3 + tRibbons.length + tRecipes.length + nRec);
  corridor[0] = nRibbons;
  corridor.set(tRibbons, 1);
  corridor[1 + tRibbons.length] = tRecipes.length / RECIPE;
  corridor.set(tRecipes, 2 + tRibbons.length);
  corridor[2 + tRibbons.length + tRecipes.length] = nRecords;
  corridor.set(tRecords.subarray(0, nRec), 3 + tRibbons.length + tRecipes.length);
  return { of, order, rule, corridor };
}

/** numbers per recipe in `LineBands.corridor`: ribbon, slot, sign (+1 a to b), the ribbon middle from the line's track (line widths, left of a to b) */
const RECIPE = 4;

/** Path entry `k`'s knots in profile `b`: the offset of its first and how many. */
export function bandKnots(b: BandProfile, k: number): [number, number] {
  const n = b[0], o = 2 + n;
  return [o + b[1 + k] * KNOT, b[2 + k] - b[1 + k]];
}

/** Path entry `k`'s band at `f` (0-1 along it, in its travel direction): [shift, 1 - width share]. */
export function profileAt(b: BandProfile, k: number, f: number): [number, number] {
  const [o, c] = bandKnots(b, k);
  if (c <= 0) return [0, 0];
  if (f <= b[o]) return [b[o + 1], b[o + 2]];
  for (let i = 1; i < c; i++) {
    const x = o + i * KNOT;
    if (f <= b[x]) {
      const p = x - KNOT, d = b[x] - b[p], t = d > 0 ? (f - b[p]) / d : 1;
      return [b[p + 1] + (b[x + 1] - b[p + 1]) * t, b[p + 2] + (b[x + 2] - b[p + 2]) * t];
    }
  }
  const x = o + (c - 1) * KNOT;
  return [b[x + 1], b[x + 2]];
}

/** `TrafficBands.pick` holds a shift as a share of this many times BAND_RANGE line widths, since a
 * trunk of busy lines is wider than BAND_RANGE */
export const TRAFFIC_PICK_DIV = 4;

export interface TrafficBands {
  /** per line stroke segment, 2: its band's shift at its start and end, line widths left of its direction */
  seg: Float32Array;
  /** per train sample: its line's band's shift there, line widths left of run 0 (as `sampleBand`) */
  sample: Float32Array;
  /** per segment, `Strokes.band`'s layout for picking: the shifts over TRAFFIC_PICK_DIV, full width */
  pick: Int16Array;
}

interface TrafficStrokes { seg: Float32Array; colour: Uint32Array; edge: Uint32Array; band?: Int16Array; count: number }

/**
 * The Traffic view's bands (T-1799 stage 2): the Line view's ribbons and eases (`lineBands`, whose
 * recipes `table` holds), each band as wide as its line's riders (`width`, per segment, line widths),
 * so a ribbon is their sum. `samples` and `sampleStroke` as `NetworkBuffers`'. Null when no line
 * has a band.
 */
export function trafficBands(L: TrafficStrokes, width: Float32Array, samples: Float32Array, sampleStroke: Int32Array | undefined, table?: Float32Array): TrafficBands | null {
  const n = L.count, seg = L.seg;
  if (!table || table.length < 3 || !L.band || L.band.length < n * 4) return null;
  // the table: ribbons, recipes, then per line and edge its knots' recipes
  let o = 0;
  const nRib = table[o++], ribAt = new Int32Array(nRib);
  for (let r = 0; r < nRib; r++) {
    ribAt[r] = o;
    o += 1 + 2 * table[o];
  }
  const nRec = table[o++], recAt = o;
  o += nRec * RECIPE;
  const nLin = table[o++];
  const recs = new Map<number, Map<number, number[]>>();
  for (let r = 0; r < nLin; r++) {
    const line = table[o], edge = table[o + 1];
    let m = recs.get(line);
    if (!m) recs.set(line, (m = new Map()));
    let v = m.get(edge);
    if (!v) m.set(edge, (v = []));
    v.push(o);
    // its knots: where along, how many recipes, then recipe and weight pairs
    const c = table[o + 5];
    o += 6;
    for (let k = 0; k < c; k++) o += 2 + 2 * table[o + 1];
  }
  if (!nLin) return null;
  // runs: one line's consecutive segments on one edge
  const start: number[] = [];
  for (let i = 0; i < n; ) {
    let j = i + 1;
    while (j < n && L.edge[j] === L.edge[i] && L.colour[j] === L.colour[i]) j++;
    start.push(i);
    i = j;
  }
  const R = start.length;
  start.push(n);
  const runOf = new Int32Array(n);
  const widthOf = new Map<number, number>();
  for (let r = 0; r < R; r++) {
    for (let i = start[r]; i < start[r + 1]; i++) runOf[i] = r;
    const key = L.colour[start[r]] * 67108864 + L.edge[start[r]];
    if (!widthOf.has(key)) widthOf.set(key, width[start[r]]);
  }
  // a ribbon's slots' places with these widths: per ribbon, its slots' middles from its middle and
  // the left edge of each slot (cumulative), lazily
  const placed = new Map<number, Float64Array>();
  const placesOf = (rid: number) => {
    let v = placed.get(rid);
    if (v) return v;
    const b = ribAt[rid], K = table[b];
    // v[s]: slot s's left side from the ribbon's left edge; v[K]: the ribbon's width
    v = new Float64Array(K + 1);
    for (let s = 0; s < K; s++) v[s + 1] = v[s] + (widthOf.get(table[b + 1 + 2 * s] * 67108864 + table[b + 2 + 2 * s]) ?? 0);
    placed.set(rid, v);
    return v;
  };
  const valueOf = (rc: number) => {
    const x = recAt + rc * RECIPE;
    const v = placesOf(table[x]), slot = table[x + 1], W = v[v.length - 1];
    return table[x + 2] * (W / 2 - (v[slot] + v[slot + 1]) / 2 + table[x + 3]);
  };
  const out = new Float32Array(n * 2), pick = new Int16Array(n * 4);
  const P = 32767 / (BAND_RANGE * TRAFFIC_PICK_DIV);
  let any = false;
  const fs: number[] = [], vs: number[] = [];
  for (let r = 0; r < R; r++) {
    const i0 = start[r], i1 = start[r + 1];
    const list = recs.get(L.colour[i0])?.get(L.edge[i0]);
    if (!list) continue;
    // which way the run goes: its first point nearer the edge's a end than its last
    const ax0 = seg[i0 * 4], ay0 = seg[i0 * 4 + 1], ax1 = seg[(i1 - 1) * 4 + 2], ay1 = seg[(i1 - 1) * 4 + 3];
    let rec = list[0];
    for (const x of list) {
      const dir = table[x + 2], ax = table[x + 3], ay = table[x + 4];
      const fromA = Math.hypot(ax0 - ax, ay0 - ay) <= Math.hypot(ax1 - ax, ay1 - ay);
      if ((dir === 0) === fromA) rec = x;
    }
    const c = table[rec + 5];
    fs.length = vs.length = 0;
    for (let k = 0, y = rec + 6; k < c; k++) {
      const nr = table[y + 1];
      let v = 0;
      for (let i = 0; i < nr; i++) v += valueOf(table[y + 2 + 2 * i]) * table[y + 3 + 2 * i];
      fs.push(table[y]);
      vs.push(v);
      y += 2 + 2 * nr;
    }
    let len = 0;
    for (let i = i0; i < i1; i++) len += Math.hypot(seg[i * 4 + 2] - seg[i * 4], seg[i * 4 + 3] - seg[i * 4 + 1]);
    const value = (f: number) => {
      if (f <= fs[0]) return vs[0];
      for (let k = 1; k < fs.length; k++) if (f <= fs[k]) {
        const d = fs[k] - fs[k - 1];
        return d > 0 ? vs[k - 1] + (vs[k] - vs[k - 1]) * ((f - fs[k - 1]) / d) : vs[k];
      }
      return vs[vs.length - 1];
    };
    let at = 0;
    for (let i = i0; i < i1; i++) {
      const l = Math.hypot(seg[i * 4 + 2] - seg[i * 4], seg[i * 4 + 3] - seg[i * 4 + 1]);
      const s0 = value(len > 0 ? at / len : 0), s1 = value(len > 0 ? (at + l) / len : 1);
      at += l;
      out[i * 2] = s0;
      out[i * 2 + 1] = s1;
      pick[i * 4] = Math.round(s0 * P);
      pick[i * 4 + 1] = Math.round(s1 * P);
      if (s0 || s1) any = true;
    }
  }
  if (!any) return null;
  // a train sample's run is its stroke's (the first segment of its path entry); where it is along
  // the run, by walking the run's segments forwards with the line's samples
  const ns = samples.length / 2, sample = new Float32Array(ns);
  let lastRun = -1, s = 0;
  for (let q = 0; sampleStroke && q < ns; q++) {
    const f = sampleStroke[q];
    if (f < 0 || f >= n) continue;
    const r = runOf[f], i1 = start[r + 1];
    if (r !== lastRun) (lastRun = r), (s = start[r]);
    const x = samples[q * 2], y = samples[q * 2 + 1];
    for (;;) {
      const ax = seg[s * 4], ay = seg[s * 4 + 1], dx = seg[s * 4 + 2] - ax, dy = seg[s * 4 + 3] - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0;
      if (t > 1 && s < i1 - 1) {
        s++;
        continue;
      }
      const u = Math.max(0, Math.min(1, t));
      sample[q] = out[s * 2] + (out[s * 2 + 1] - out[s * 2]) * u;
      break;
    }
  }
  return { seg: out, sample, pick };
}
