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

import { besideSamples, newBesideState, shiftBeside, type Beside, type BesideState, type EdgeSamples } from "./corridors";
import { kAtY } from "./geo";

/** T-1821: the Line view's line width as a multiple of the width the Traffic view's scale is set
 * against (Anita, 2026-10-29: "raise normal line thickness by like 30% relative to traffic mode") */
export const LINE_VIEW_WIDTH = 1.3;
/** a band's width as a share of a line's (Anita, 2026-10-29) */
export const BAND_SHARE = 0.8;
/** px of casing left between two bands side by side, at every zoom (Anita, T-2126: "a little like ~1px gap") */
export const BAND_GAP_PX = 1;
/** line widths a metre across is taken as when deciding which shifts along a ribbon are one value
 * (a line width is about 20 m of ground near zoom 14.5); the metres themselves are drawn at the zoom
 * in view (T-2132) */
const BAND_LW_PER_M = 0.05;
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
/** the pair rule's metres are summed in parts of a metre this fine (T-4153) */
const PAIR_FIX = 1024;
/** a band's shift is stored as a share of this many line widths (`Strokes.band`, Int16 normalised) */
export const BAND_RANGE = 16;
/** and the metres from its track to its ribbon's middle as a share of this many (SPAN_M halved, with room) */
export const BAND_M_RANGE = 64;
/** numbers per segment of `Strokes.band`: the shift at its start and end, `1 - width share` at both,
 * the metres at both (T-2132) */
export const BAND_STRIDE = 6;
/** numbers per knot of a band profile: where along the path entry (0-1, its travel direction), the
 * shift there (line widths, left of travel), `1 - width share`, which recipe (-1 between two), and
 * the metres from the line's track to its ribbon's middle (left of travel), which the shaders draw at
 * the zoom in view so a ribbon is centred on its tracks at every zoom (T-2132) */
export const KNOT = 5;
/** shifts closer than this (line widths) along one ribbon are one value (a track's spacing wandering) */
const SHIFT_TOL = 0.05;

/** T-1799 stage 3's corridors: tracks side by side drawn as one ribbon. */
export const CORRIDORS = true;

/** T-2058 (notes/T-2058.md): "smooth" every change of a band's shift a long S-curve, changes that
 * overlap added together; "order" each pair of tracks' order in a ribbon decided once for the
 * network; "fit" a change at a node on the side with more lines where the side with fewer is shorter
 * than its curve; "snap" a ribbon change just ahead of a merge's or split's node placed as at the
 * node (T-2059, T-2060); "hold" a ribbon whose tracks part for a while held across the gap (T-2132);
 * "align" the lines of one ribbon change easing at one place (T-2140); "" T-1848's eases and
 * per-ribbon order (`?perfOff=bandsSmooth`, `bandsOrder`, `bandsPlace`, `bandsHold`, `bandsAlign`:
 * `bandTry`) */
let bandMode = "smooth,order,fit,snap,hold,align";
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
/** "hold" (T-2132): a ribbon left and rejoined with the same lines is held across a gap up to this
 * long (m) while the tracks stay within HOLD_M of each other (centre lines; a ribbon is at most
 * SPAN_M across) */
const HOLD_RUN_M = 2000;
const HOLD_M = 50;
/** "align" (T-2140): ribbon changes of one ribbon whose places (on each line's track) are within this
 * many metres are one change; a line's curve is moved only to a point of its track within ALIGN_NEAR_M */
const ALIGN_M = 100;
const ALIGN_NEAR_M = 80;

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
  /** T-4153: per line id, its profile with its own numbering of recipes, a new array only when its bands are worked out
   * again (so the same array: the same profile but for the recipes' numbers) */
  own: Map<number, BandProfile>;
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
  shape: object;
  flip: boolean;
}

/** The bands of every line. `pts`: each edge's local polyline (x, y, z triples, a to b). A later call compares its maps
 * with this call's, so a caller does not change a map it has passed. `changed` (T-4153): the edges whose row or points
 * may differ from those in the maps given (the network's last finished build), so a call made right after one on those
 * maps follows only them and the lines whose paths changed. */
export function lineBands(lines: readonly BandLine[], edges: readonly BandEdge[], pts: ReadonlyMap<number, Float32Array>, opts?: { corridors?: boolean; within?: number; debug?: Map<number, unknown>; state?: BandState; ends?: Map<number, BandEdge>; changed?: BandChanges }): LineBands {
  const st = useBands(opts?.state), beside = st.beside;
  const o = { ...opts, beside };
  const ends = opts?.ends ?? new Map(edges.map((e) => [e.id, e]));
  if (!opts?.debug) { for (const k in bandLaps) delete bandLaps[k]; bandT = performance.now(); }
  if (!opts?.debug) { bandWork.orders = 0; bandWork.pairs = 0; bandWork.passed = 0; bandWork.lines = 0; bandWork.whole = 0; bandWork.spliced = 0; bandWork.swept = 0; bandWork.copied = 0; bandWork.samplesKept = 0; bandWork.samplesMade = 0; }
  const rail = lines.filter((l) => l.ok && !l.bus && l.path.length);
  const mode = `${opts?.corridors ?? CORRIDORS},${opts?.within ?? ""},${bandMode}`;
  // (a fresh pass, for `rule` and the check, lays every path out again)
  const fresh = () => linesOn(rail);
  if (opts?.debug) return bands(rail, fresh(), ends, pts, o);
  // T-4153: the lines on each edge kept, changed only where a line's path did
  const kept = (st.on ??= { on: new Map(), rows: new Map() });
  const moved = patchOn(kept, rail);
  const on = kept.on;
  /** the edges whose row or points may differ from the last call's, when the network says (null: any may) */
  const delta = last && last.mode === mode && opts?.changed && opts.changed.pts === last.pts && opts.changed.ends === last.ends ? opts.changed.edges : null;
  // the same paths, and the same points and ends of every edge they use: the same bands, known without laying out a path
  if (last && last.mode === mode && !moved.lines.size && sameUsed(on, pts, ends, last, delta)) {
    let whole: LineBands | null = null;
    const rule = (edge: number, x: number, y: number) => (whole ??= bands(rail, fresh(), ends, pts, { ...o, fresh: true })).rule(edge, x, y);
    last = { ...last, pts, ends, rail };
    return { ...last.out, rule, corridor: last.corridor.slice() };
  }
  // (T-3939: `rule` is asked of a kept result by one whole pass, on this call's points)
  let whole: LineBands | null = null;
  const rule = (edge: number, x: number, y: number) => (whole ??= bands(rail, fresh(), ends, pts, { ...o, fresh: true })).rule(edge, x, y);
  const follow = !!delta && memo?.mode === mode;
  if (follow) bandWork.followed++;
  const out = bands(rail, on, ends, pts, { ...o, delta: follow ? { edges: delta!, moved } : undefined });
  if (bandCheck) {
    // (T-4153: the corridors and the lines on each edge from scratch too, or a slip in what is kept is in both)
    const bad = bandsDiffer(out, bands(rail, fresh(), ends, pts, { ...o, fresh: true, beside: newBesideState() }));
    if (bad) console.error(`T-2238 check: bands worked out again for an edit differ from a whole pass: ${bad}`);
  }
  // (a copy: the clock worker transfers the array it posts.) T-3939: kept without `rule`, whose closure holds the
  // whole of `bands`' working set (18 MB on a 13-zone Tokyo game)
  last = { pts, ends, rail, mode, out: { of: out.of, own: out.own, order: out.order, rule, corridor: out.corridor }, corridor: out.corridor.slice() };
  return out;
}

/** T-4153: the edges whose row or points may differ from those of the maps given (`pts`, `ends`) */
export interface BandChanges { pts: ReadonlyMap<number, Float32Array>; ends: ReadonlyMap<number, BandEdge>; edges: ReadonlySet<number> }

/** every edge the lines use has the same points and ends as at the last call (only those `delta` names may not) */
function sameUsed(on: ReadonlyMap<number, unknown>, pts: ReadonlyMap<number, Float32Array>, ends: ReadonlyMap<number, BandEdge>, was: LastBands, delta: ReadonlySet<number> | null): boolean {
  const same = (e: number) => pts.get(e) === was.pts.get(e) && ends.get(e)?.a === was.ends.get(e)?.a && ends.get(e)?.b === was.ends.get(e)?.b;
  if (delta) {
    for (const e of delta) if (on.has(e) && !same(e)) return false;
    return true;
  }
  for (const e of on.keys()) if (!same(e)) return false;
  return true;
}

/** T-4153: two rail lines the bands read alike (their id, path and whether a bus) */
const sameBandLine = (a: BandLine, b: BandLine) => a === b || (a.id === b.id && !a.bus === !b.bus && (a.path === b.path || (a.path.length === b.path.length && a.path.every((p, k) => p === b.path[k]))));

/** `?patch=check` (T-2238): every build's bands also worked out whole and compared */
let bandCheck = false;
export function setBandCheck(on: boolean): void {
  bandCheck = on;
}

/** How two lots of bands differ, "" for not at all: order, profiles and Traffic records, with the
 * recipes as what they are (their numbers can differ) and numbers exactly (T-4153: a path's metres lie on the path grid,
 * so a cut moves those after it by one exact amount). */
function bandsDiffer(a: LineBands, b: LineBands): string {
  const r4 = (v: number) => v;
  const read = (x: LineBands) => {
    const { recipes, recs } = tableRead(x.corridor);
    const prof = new Map<number, string>();
    for (const [id, p] of x.of) {
      const k0 = 2 + p[0];
      prof.set(id, Array.from(p, (v, i) => (i >= k0 && (i - k0) % KNOT === 3 ? (v < 0 ? "-" : recipes[v]) : r4(v))).join(","));
    }
    return { recs, prof };
  };
  const x = read(a), y = read(b);
  const lines = [...new Set([...x.prof.keys(), ...y.prof.keys()])].filter((id) => x.prof.get(id) !== y.prof.get(id));
  const recs = x.recs.length === y.recs.length ? x.recs.filter((s, i) => s !== y.recs[i]).length : -1;
  const order = [...b.order].filter(([e, v]) => a.order.get(e)?.join() !== v.join()).length + (a.order.size === b.order.size ? 0 : 1);
  return lines.length || recs || order ? `lines ${lines.slice(0, 8).join(", ")}${lines.length > 8 ? "..." : ""}; Traffic records ${recs}; edges' orders ${order}` : "";
}

/** a Traffic table's recipes and records as text, each recipe as what it is (its ribbon, slot and sign), not its number */
function tableRead(t: Float32Array): { recipes: string[]; recs: string[] } {
  let o = 0;
  const ribs: string[] = [], recipes: string[] = [], recs: string[] = [];
  for (let r = 0, n = t[o++]; r < n; r++) {
    ribs.push(Array.from(t.subarray(o + 1, o + 1 + 2 * t[o])).join(","));
    o += 1 + 2 * t[o];
  }
  for (let r = 0, n = t[o++]; r < n; r++, o += RECIPE) recipes.push(`${ribs[t[o]]}/${t[o + 1]}/${t[o + 2]}`);
  for (let r = 0, n = t[o++]; r < n; r++) {
    let s = Array.from(t.subarray(o, o + 6)).join(",");
    const c = t[o + 5];
    o += 6;
    for (let k = 0; k < c; k++) {
      s += `;${t[o]}:`;
      for (let i = 0; i < t[o + 1]; i++) s += `${recipes[t[o + 2 + 2 * i]]}*${t[o + 3 + 2 * i]} `;
      o += 2 + 2 * t[o + 1];
    }
    recs.push(s);
  }
  return { recipes, recs };
}
/** T-4153 (`?patch=check`): two Traffic tables alike but for how they number their ribbons and recipes, which a kept
 * one numbers in the order they came, so one left by an edit keeps its number (and one no line uses any more stays) */
export function sameCorridor(a: Float32Array, b: Float32Array): boolean {
  const x = tableRead(a).recs, y = tableRead(b).recs;
  return x.length === y.length && x.every((s, i) => s === y[i]);
}

/** the last call: the maps it was given, its lines and what it gave */
interface LastBands { pts: ReadonlyMap<number, Float32Array>; ends: ReadonlyMap<number, BandEdge>; rail: readonly BandLine[]; mode: string; out: LineBands; corridor: Float32Array }
type OnEdge = { line: BandLine; k: number };
/** T-4153: per edge the lines on it, as `linesOn` lays them out, and the rows they are of, by id */
interface KeptOn { on: Map<number, OnEdge[]>; rows: Map<number, BandLine> }
/** what `patchOn` changed: the lines whose paths changed, came or went, and the edges on their paths as they were and are */
interface OnMoved { lines: Set<number>; edges: Set<number> }
/** `kept` brought to `rail`: a line on a new path taken off the edges of its last one and put on those of its new one (in
 * id order on each, at its first entry there), a line on the same path given its new row */
function patchOn(kept: KeptOn, rail: readonly BandLine[]): OnMoved {
  const out: OnMoved = { lines: new Set(), edges: new Set() };
  const { on, rows } = kept;
  const drop = (l: BandLine) => {
    for (const p of l.path) {
      const e = p >> 1, v = on.get(e);
      out.edges.add(e);
      if (!v) continue;
      const i = v.findIndex((x) => x.line.id === l.id);
      if (i < 0) continue;
      v.splice(i, 1);
      if (!v.length) on.delete(e);
    }
  };
  const put = (l: BandLine) => {
    l.path.forEach((p, k) => {
      const e = p >> 1;
      let v = on.get(e);
      if (!v) on.set(e, (v = []));
      out.edges.add(e);
      let i = 0;
      while (i < v.length && v[i].line.id < l.id) i++;
      if (i < v.length && v[i].line.id === l.id) return;
      v.splice(i, 0, { line: l, k });
    });
  };
  let n = 0;
  for (const l of rail) {
    const was = rows.get(l.id);
    if (was) n++;
    if (was === l) continue;
    if (was && sameBandLine(was, l)) {
      // (the same path: its entries read the new row)
      for (const p of l.path) for (const x of on.get(p >> 1) ?? []) if (x.line.id === l.id) x.line = l;
    } else {
      if (was) drop(was);
      put(l);
      out.lines.add(l.id);
    }
    rows.set(l.id, l);
  }
  if (n < rows.size) {
    const now = new Set(rail.map((l) => l.id));
    for (const [id, l] of [...rows]) if (!now.has(id)) { drop(l); rows.delete(id); out.lines.add(id); }
  }
  return out;
}

/** per edge: the lines on it, each with its first entry in its path */
function linesOn(rail: readonly BandLine[]): Map<number, { line: BandLine; k: number }[]> {
  const on = new Map<number, { line: BandLine; k: number }[]>();
  for (const l of [...rail].sort((a, b) => a.id - b.id)) {
    l.path.forEach((p, k) => {
      const e = p >> 1;
      let v = on.get(e);
      if (!v) on.set(e, (v = []));
      if (!v.some((x) => x.line === l)) v.push({ line: l, k });
    });
  }
  return on;
}
let last: LastBands | null = null;

/** T-4153: what `lineBands` keeps between calls (the last result, the bands' memo, the kept orders, the corridors'
 * samples), one per network cache (map/network.ts `NetCache`), so a build from scratch beside the kept one
 * (`?patch=check`) neither reads nor moves it; the module's variables hold the one in use */
export interface BandState { last: LastBands | null; memo: BandMemo | null; orders: ReturnType<typeof newOrders>; beside: BesideState; on?: KeptOn }
export const newBandState = (): BandState => ({ last: null, memo: null, orders: newOrders(), beside: newBesideState() });
let sharedBands: BandState | null = null;
let bandsInUse: BandState | null = null;
function useBands(st: BandState | undefined): BandState {
  const inUse: BandState = (bandsInUse ??= sharedBands ??= { last, memo, orders, beside: newBesideState() });
  const want = st ?? sharedBands!;
  if (want === inUse) return want;
  Object.assign(inUse, { last, memo, orders });
  ({ last, memo, orders } = want);
  bandsInUse = want;
  return want;
}

/** T-4103: the view's anchor moved (`shiftNetCache`): the kept bands (by line and edge, in metres) and corridors carry
 * over to each edge's points in the moved frame (`moved`, old array to new), as if worked out on them. */
export function shiftKept(moved: ReadonlyMap<Float32Array, Float32Array>, state?: BandState) {
  const st = useBands(state);
  if (last) last = { ...last, pts: new Map([...last.pts].map(([e, p]) => [e, moved.get(p) ?? p])) };
  shiftBeside(moved, st.beside);
}

/** a recipe of a line's own: its ribbon (`rk` its lines and edges as text, `rib` them interleaved), slot and sign */
interface OwnRecipe { key: string; rk: string; rib: number[]; slot: number; sgn: number }
/** T-4153: the recipe of every knot where a line is alone on its track: a ribbon of one slot of no line (its width none),
 * whose Traffic shift is none, as a ribbon of the line alone has on any edge. One for all such places, so a line's
 * profile there has no knots but its entries' ends, the same across an edit elsewhere on it, and a cut there makes no
 * recipe. */
const ALONE: OwnRecipe = { key: "|/0/1", rk: "|", rib: [-1, -1], slot: 0, sgn: 1 };
/** a ribbon change of a line for T-2140's alignment: its group (the bigger ribbon's lines), where, and
 * how far it reaches, local units */
interface AlignAt {
  key: string;
  x: number;
  y: number;
  r: number;
  /** what its group's lining up reads of it (`alignChanges`): priority, places on its own track
   * (m) and the point at its curve's middle */
  prio: number;
  enter: boolean;
  g: number;
  c: number;
  L: number;
  px: number;
  py: number;
}
/** past this (local units, or metres along a path) two ribbon changes are not the same: a path's
 * metres can move in the last bits when an edge on it is cut in two */
const ALIGN_SAME = 1e-3;
/** (T-4153: its curve's place is compared from the change, `c - g`, as the other lines of its group read it, not in path
 * metres, which a cut earlier on the line moves) */
const sameChange = (a: AlignAt, b: AlignAt) => a.key === b.key && a.prio === b.prio && a.enter === b.enter
  && Math.abs(a.x - b.x) < ALIGN_SAME && Math.abs(a.y - b.y) < ALIGN_SAME && Math.abs((a.c - a.g) - (b.c - b.g)) < ALIGN_SAME
  && Math.abs(a.L - b.L) < ALIGN_SAME && (Object.is(a.px, b.px) || Math.abs(a.px - b.px) < ALIGN_SAME) && (Object.is(a.py, b.py) || Math.abs(a.py - b.py) < ALIGN_SAME);
/** T-2238: one line's bands as last worked out, with its recipes numbered in the order it first used
 * them, and what they were made from: the edges it read (`deps`), the boxes it looked for tracks in
 * to hold a ribbon across a gap (`holds`, x0, y0, x1, y1 each), and its ribbon changes */
interface LineMemo {
  path: number[];
  deps: number[];
  /** the network-wide pair rules it read (`BandMemo.rel` keys) */
  rels: string[];
  holds: number[];
  changes: AlignAt[];
  recipes: OwnRecipe[];
  /** its profile and Traffic records with its own recipe numbers; `recPos` where those are in `recs` */
  profile: Float32Array | null;
  recs: Float32Array;
  recPos: Int32Array;
  nRecords: number;
  /** T-4153: its recipes' numbers in the network's table (`BandMemo.net`), and its profile and records numbered so;
   * `netWas`: the numbers its recipes had, which a line worked out again keeps (its own numbers are kept too) */
  net?: { map: Int32Array; profile: Float32Array | null; recs: Float32Array };
  netWas?: Int32Array;
  /** T-4153: what its knots were swept from, and where each entry's record starts in `recs` (-1: none) */
  sweep?: SweepIn;
  recAt?: Int32Array;
  /** T-4153: per entry whether any of its knots is other than none */
  nzE?: Uint8Array;
  /** T-4153: its samples and what each entry's were read from */
  smp?: LineSamplesMemo;
}
/** T-4153: a line's samples (metres along its path, half their spacing, ribbon, slot) and per entry where its samples start
 * (`js`), its start, whether any is in a ribbon of more than one line, what its ribbons read (edges, pair rules), its
 * edge's corridor samples and its length */
interface LineSamplesMemo { js: Int32Array; start: Float64Array; any: Uint8Array; deps: number[][]; rels: (string[] | null)[]; T: (EdgeSamples | undefined)[]; len: Float64Array; M: number[]; H: number[]; R: Ribbon[]; S: number[] }
/** the ribbon at a sample, the edges it was worked out from and in which call */
interface RibbonEntry { r: Ribbon; deps: number[]; at: number; /** the pair rules its order was worked out from */ rels?: string[] }
/** What the last call worked out (T-2238), so an edit works out again only the lines whose bands it
 * can change: per edge, what other lines' bands read of it (`fp`, its points and samples), the ribbon
 * at each sample, the network-wide pair rule's answers, and per line its `LineMemo`. */
interface BandMemo {
  mode: string;
  /** per edge, what other lines' bands read of it: its ends, its lines in order and their directions, its points
   * and samples (changed in place) */
  fp: Map<number, { a: number; b: number; ids: number[]; dirs: number[]; pts: Float32Array | undefined; samples: EdgeSamples | undefined }>;
  /** per edge, its samples' ribbons (from the call `at`, when it had that many samples) */
  ribbons: Map<number, { at: number; list: (RibbonEntry | undefined)[] }>;
  rel: Map<string, number>;
  lines: Map<number, LineMemo>;
  /** the calls so far, and per edge the last one it changed in: a ribbon from before any of its
   * edges changed holds */
  serial: number;
  changedAt: Map<number, number>;
  /** T-4153: the network's numbers of ribbons and recipes, kept from call to call */
  net: NetNumbers;
}
/** T-4153: the table's ribbons and recipes by key, numbered in the order they first came; never renumbered while the memo
 * lasts (one no line uses any more stays, unread), so a line's numbers hold while it is not worked out again */
interface NetNumbers { ribIds: Map<string, number>; recIds: Map<string, number>; ribs: Nums; nRib: number; recs: Nums; nRec: number }
const newNetNumbers = (): NetNumbers => ({ ribIds: new Map(), recIds: new Map(), ribs: new Nums(), nRib: 0, recs: new Nums(), nRec: 0 });
let memo: BandMemo | null = null;
/** T-4153: one edge's order of its lines (`LineBands.order`), with what it was worked out from: its lines (each
 * with its first entry on it and its path), per pair of them the rule's answer and the steps its walks ended at, and
 * the edges read where they ended (their ends and points) */
interface OrderMemo {
  lines: BandLine[];
  ks: number[];
  paths: number[][];
  /** per pair of lines i < j, row by row: the rule's sign, and the step its walk ahead ended at and its walk back's (-1
   * when not walked); 0 for a walk that cannot be kept */
  sign: Int8Array;
  walks: Int32Array;
  reads: Int32Array;
  order: number[];
}
/** the kept orders; per edge the orders that read it; each edge's row and points at the last call */
/** (`readBy` only grows, so it can name an order that no longer reads an edge, which is then worked out again for
 * nothing; it is laid out again from the kept orders once it holds three times what they read, `live`) */
const newOrders = () => ({ memo: new Map<number, OrderMemo>(), readBy: new Map<number, number[]>(), entries: 0, live: 0, rows: new Map<number, BandEdge>(), pts: new Map<number, Float32Array | undefined>(), order: new Map<number, number[]>() });
let orders = newOrders();
/** (`added`: the reads `m` has that the edge's last order did not) */
function setOrder(edge: number, m: OrderMemo | null, added: ArrayLike<number> = m?.reads ?? []) {
  const old = orders.memo.get(edge);
  if (old) orders.live -= old.reads.length;
  if (!m) { orders.memo.delete(edge); return; }
  orders.memo.set(edge, m);
  orders.live += m.reads.length;
  addReads(edge, added);
  if (orders.entries > 3 * orders.live + 1024) {
    orders.readBy.clear();
    orders.entries = 0;
    for (const [e, k] of orders.memo) addReads(e, k.reads);
  }
}
function addReads(edge: number, reads: ArrayLike<number>) {
  for (let i = 0; i < reads.length; i++) {
    const r = reads[i], v = orders.readBy.get(r);
    if (v) v.push(edge);
    else orders.readBy.set(r, [edge]);
  }
  orders.entries += reads.length;
}
/** the edges whose kept order read an edge whose row or points changed since the last call; `gone` gets the edges
 * gone since */
function staleOrders(ends: ReadonlyMap<number, BandEdge>, pts: ReadonlyMap<number, Float32Array>, gone: number[], only?: ReadonlySet<number>): Set<number> {
  const out = new Set<number>();
  const moved = (e: number) => { for (const x of orders.readBy.get(e) ?? []) out.add(x); };
  const look = (e: number, row: BandEdge) => {
    const p = pts.get(e);
    if (orders.rows.get(e) === row && orders.pts.get(e) === p) return;
    moved(e);
    orders.rows.set(e, row);
    orders.pts.set(e, p);
  };
  const lost = (e: number) => { moved(e); gone.push(e); orders.rows.delete(e); orders.pts.delete(e); };
  // (T-4153: only the edges that may have changed, when the network says which)
  if (only) {
    for (const e of only) {
      const row = ends.get(e);
      if (row) look(e, row);
      else if (orders.rows.has(e)) lost(e);
    }
    return out;
  }
  for (const [e, row] of ends) look(e, row);
  if (orders.rows.size > ends.size) for (const e of [...orders.rows.keys()]) if (!ends.has(e)) lost(e);
  return out;
}
/** checks/t4153-net.mjs --self-test: every kept order trusted, which its check must catch */
let selfTest = false;
export function bandSelfTest(on: boolean) { selfTest = on; }
/** T-4153: what the last call worked out again (counts, for the network build's work line): orders with any pair of
 * lines walked again, those pairs, and pairs kept whose lines' paths changed within their walks */
/** T-4153: per stage of the last call, ms (for measuring what an edit's bands still pay for) */
export const bandLaps: Record<string, number> = {};
let bandT = 0;
const blap = (k: string) => { const t = performance.now(); bandLaps[k] = (bandLaps[k] ?? 0) + t - bandT; bandT = t; };
export const bandWork = { orders: 0, pairs: 0, passed: 0, lines: 0, whole: 0, passedAll: 0, followed: 0, spliced: 0, swept: 0, copied: 0, splicedAll: 0, sweptAll: 0, copiedAll: 0, samplesKept: 0, samplesMade: 0 };
/** T-4153 self-test: a line worked out again after an edit sweeps only its entries that changed, not those beside them
 * whose knots the change reaches (a station on a long line's shared track must fail the check) */
let spliceTest = false;
export function spliceSelfTest(on: boolean) { spliceTest = on; }
function sameOrderLines(m: OrderMemo, v: readonly { line: BandLine; k: number }[], diffOf: (q: number[], p: number[]) => PathDiff | null): boolean {
  if (m.lines.length !== v.length) return false;
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (x.line.id !== m.lines[i].id || x.k !== m.ks[i] || diffOf(m.paths[i], x.line.path)) return false;
  }
  return true;
}
/** T-4153: how a line's path changed since a kept order read it: old entries [pre, `was` - suf) became new entries
 * [pre, p.length - suf), the rest alike; null when not at all */
interface PathDiff { p: number[]; pre: number; suf: number; was: number }
function pathDiff(q: number[], p: number[]): PathDiff | null {
  if (q === p) return null;
  const n = Math.min(q.length, p.length);
  let pre = 0, suf = 0;
  while (pre < n && q[pre] === p[pre]) pre++;
  if (pre === n && q.length === p.length) return null;
  while (suf < n - pre && q[q.length - 1 - suf] === p[p.length - 1 - suf]) suf++;
  return { p, pre, suf, was: q.length };
}
/** the changed part of a line's path in the steps of a walk from its entry k (forwards or back in its path): the first
 * step of it and the step past it, in the old path, and how many new entries took its place */
function stepsOf(d: PathDiff, k: number, fwd: boolean): [number, number, number] {
  const w0 = d.pre, w1 = d.was - d.suf, n = d.p.length - d.suf - d.pre;
  return fwd ? [w0 - k, w1 - k, n] : [k - w1 + 1, k - w0 + 1, n];
}
/** past this many edges changed, every line is worked out again */
const MEMO_EDGES = 400;
/** T-4153: a line's path metres in its bands lie on a grid of 1/1024 m (its edges' lengths, its entries' starts, its
 * samples' bounds), so a cut, which moves the metres after it, moves every place there by one exact amount and every
 * difference of them (a knot from its entry's start, a curve's middle from a sample) comes out the same bit for bit */
const PATH_GRID = 1024;
const onGrid = (m: number) => Math.round(m * PATH_GRID) / PATH_GRID;

/** `delta` (T-4153): the edges whose row or points may differ from the last call's and the lines `patchOn` moved, so the
 * passes over every edge follow those alone */
function bands(rail: BandLine[], on: Map<number, OnEdge[]>, ends: Map<number, BandEdge>, pts: ReadonlyMap<number, Float32Array>, opts?: { corridors?: boolean; within?: number; debug?: Map<number, unknown>; fresh?: boolean; beside?: BesideState; delta?: { edges: ReadonlySet<number>; moved: OnMoved } }): LineBands {
  // the unit direction an edge leaves a node in, y up (local y points south)
  const leave = new Map<number, [number, number]>();
  /** T-4153: the edges an edge's order read (their ends and points), while it is worked out */
  let orderReads: number[] | null = null;
  const dirAt = (edge: number, node: number): [number, number] => {
    orderReads?.push(edge);
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
  /** T-4153: the step the last walk of `part` ended at (where the two lines part or a path ends); 0 when it cannot be
   * kept, having passed two entries that differ but read alike (both ways along an edge from a node to itself) */
  let walkEnd = 0;
  /** a line's entry `j` steps from entry k, as the way it is walked (forwards, or back with its direction turned) */
  const entryAt = (x: { line: BandLine; k: number }, fwd: boolean, j: number) => (fwd ? x.line.path[x.k + j] : x.line.path[x.k - j] ^ 1);
  /** of two lines on `edge`, walking towards `toB` (its b end) or away: < 0 when `x` turns left of
   * `y` where they part, > 0 right, 0 if they never part */
  const part = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }, toB: boolean): number => {
    // each line's own direction on this edge: walk its path forwards when it runs the way we walk
    const fx = ((x.line.path[x.k] & 1) === 0) === toB, fy = ((y.line.path[y.k] & 1) === 0) === toB;
    let prev = edge, odd = false;
    const max = Math.min(x.line.path.length, y.line.path.length);
    for (let j = 1; j <= max; j++) {
      const a = step(x.line, x.k, fx, j), b = step(y.line, y.k, fy, j);
      if (!a || !b || a[0] !== b[0] || a[1] !== b[1]) {
        walkEnd = odd ? 0 : j;
        // (T-4153: what a kept walk depends on beyond the two paths: the rows and points of the edges where it ends,
        // and of the one before, which `turn` reads)
        if (orderReads) {
          const ix = fx ? x.k + j : x.k - j, iy = fy ? y.k + j : y.k - j;
          if (ix >= 0 && ix < x.line.path.length) orderReads.push(x.line.path[ix] >> 1);
          if (iy >= 0 && iy < y.line.path.length) orderReads.push(y.line.path[iy] >> 1);
        }
        if (!a || !b || a[1] !== b[1]) return 0;
        const ta = turn(prev, a[0], a[1]), tb = turn(prev, b[0], b[1]);
        return ta > tb ? -1 : ta < tb ? 1 : 0;
      }
      if (orderReads && entryAt(x, fx, j) !== entryAt(y, fy, j)) odd = true;
      prev = a[0];
    }
    walkEnd = 0;
    return 0;
  };
  // (T-4153: the kept orders' own map, set and deleted where an edge's order is worked out)
  const order = !opts?.debug && !opts?.fresh ? orders.order : new Map<number, number[]>();
  /** T-4153: the steps the last `compare`'s walks ended at (`walkEnd`), the second -1 when not walked */
  let walk1 = 0, walk2 = -1;
  /** < 0: `x` left of `y` in the edge's a to b direction. Looked at first ahead in the lower-id
   * line's own direction, so every edge of a stretch the two share asks the same parting node
   * whatever way each edge was drawn; a walk towards a sees left of travel as right of a to b. */
  const compare = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }) => {
    const lead = x.line.id < y.line.id ? x : y;
    const toB = (lead.line.path[lead.k] & 1) === 0;
    const first = part(edge, x, y, toB);
    walk1 = walkEnd;
    walk2 = -1;
    const second = first ? 0 : part(edge, x, y, !toB);
    if (!first) walk2 = walkEnd;
    // never parting: the lower id on the left of the lead line's travel
    return (toB ? first : -first) || (toB ? -second : second) || (toB ? 1 : -1) * (x.line.id - y.line.id);
  };
  // T-4153: an edge's order kept while its lines, their paths and every edge the comparisons read are as last time;
  // and where some changed, each pair's answer kept while its walks read the same of their paths (`walkNow`)
  const keepOrders = !opts?.debug && !opts?.fresh;
  const delta = keepOrders ? opts?.delta : undefined;
  const gone: number[] = [];
  const stale = keepOrders ? staleOrders(ends, pts, gone, delta?.edges) : null;
  // (the lines on an edge gone that their paths still name: a walk along it would now end there)
  const broken = new Set<number>(), brokenPaths: number[][] = [];
  for (const e of gone) for (const x of on.get(e) ?? []) if (!broken.has(x.line.id)) (broken.add(x.line.id), brokenPaths.push(x.line.path));
  const diffs = new Map<number[], { p: number[]; d: PathDiff | null }>();
  const diffOf = (q: number[], p: number[]) => {
    let x = diffs.get(q);
    if (!x || x.p !== p) diffs.set(q, (x = { p, d: pathDiff(q, p) }));
    return x.d;
  };
  /** a kept walk of two lines (x the lead) that ended at step J, as their paths are now: the step it ends at, 0 when it
   * is to be walked again. It is the same where neither path changed within it, and where both changed alike within the
   * stretch they walk together (a piece of track they share cut in two), short of its last two steps (where they part,
   * and the one before, whose way off the node the parting reads) */
  const walkNow = (J: number, dx: PathDiff | null, kx: number, fx: boolean, dy: PathDiff | null, ky: number, fy: boolean): number => {
    if (J < 1) return 0;
    const wx = dx && stepsOf(dx, kx, fx), wy = dy && stepsOf(dy, ky, fy);
    const ax = !!wx && wx[1] > 0 && wx[0] <= J, ay = !!wy && wy[1] > 0 && wy[0] <= J;
    if (!ax && !ay) return J;
    if (!ax || !ay || wx![0] !== wy![0] || wx![1] !== wy![1] || wx![2] !== wy![2] || wx![0] < 1 || wx![1] > J - 1) return 0;
    const n = wx![2];
    for (let t = 0; t < n; t++) {
      const ex = fx ? dx!.p[dx!.pre + t] : dx!.p[dx!.pre + n - 1 - t] ^ 1, ey = fy ? dy!.p[dy!.pre + t] : dy!.p[dy!.pre + n - 1 - t] ^ 1;
      if (ex !== ey || !ends.has(ex >> 1)) return 0;
    }
    return J + n - (wx![1] - wx![0]);
  };
  // T-4153: with `delta`, only the edges whose order can differ: those whose lines or their paths changed (`patchOn`), those
  // whose kept order read an edge that changed, and those of a line on an edge gone
  let visit: Iterable<number> = on.keys();
  if (delta) {
    const v = new Set(delta.moved.edges);
    for (const e of stale!) v.add(e);
    for (const p of brokenPaths) for (const q of p) v.add(q >> 1);
    visit = v;
  }
  const visited = new Set<number>();
  for (const edge of visit) {
    const v = on.get(edge);
    if (!v || v.length < 2) {
      if (delta) {
        order.delete(edge);
        if (orders.memo.has(edge)) setOrder(edge, null);
      }
      continue;
    }
    visited.add(edge);
    const was = stale ? orders.memo.get(edge) : undefined;
    const fresh = !was || stale!.has(edge);
    if (was && (selfTest || (!fresh && sameOrderLines(was, v, diffOf)))) {
      order.set(edge, was.order);
      continue;
    }
    if (keepOrders) orderReads = [];
    const m = v.length, nPairs = (m * (m - 1)) / 2, sign = new Int8Array(nPairs), walks = new Int32Array(2 * nPairs);
    // per line, where it is in the kept order (-1: not there, or its pairs walked again) and how its path changed
    const from = new Int32Array(m).fill(-1), dif: (PathDiff | null)[] = new Array(m).fill(null);
    if (!fresh) for (let i = 0, o = 0; i < m; i++) {
      const id = v[i].line.id;
      while (o < was!.lines.length && was!.lines[o].id < id) o++;
      if (o === was!.lines.length || was!.lines[o].id !== id || broken.has(id)) continue;
      const k = was!.ks[o], d = diffOf(was!.paths[o], v[i].line.path);
      // (its entry on this edge where it was, moved by what changed before it)
      if ((!d ? k : k < d.pre ? k : k >= d.was - d.suf ? k + d.p.length - d.was : -1) !== v[i].k) continue;
      from[i] = o;
      dif[i] = d;
    }
    const om = was?.lines.length ?? 0;
    let walked = 0;
    for (let i = 0, p = 0; i < m; i++) for (let j = i + 1; j < m; j++, p++) {
      const oi = from[i], oj = from[j];
      if (oi >= 0 && oj >= 0) {
        const op = (oi * (2 * om - oi - 1)) / 2 + oj - oi - 1, J1 = was!.walks[2 * op], J2 = was!.walks[2 * op + 1];
        let n1 = J1, n2 = J2;
        if (dif[i] || dif[j]) {
          // (the lead, the lower id, walks its own way first; the other the same way along the edge)
          const ki = was!.ks[oi], kj = was!.ks[oj], fy = (was!.paths[oj][kj] & 1) === (was!.paths[oi][ki] & 1);
          n1 = walkNow(J1, dif[i], ki, true, dif[j], kj, fy);
          n2 = J2 < 0 || !n1 ? J2 : walkNow(J2, dif[i], ki, false, dif[j], kj, !fy);
          if (n1 && n2 && keepOrders) bandWork.passed++, bandWork.passedAll++;
        }
        if (n1 && n2) {
          sign[p] = was!.sign[op];
          walks[2 * p] = n1;
          walks[2 * p + 1] = n2;
          continue;
        }
      }
      walked++;
      sign[p] = Math.sign(compare(edge, v[i], v[j]));
      walks[2 * p] = walk1;
      walks[2 * p + 1] = walk2;
    }
    if (keepOrders && walked) {
      bandWork.orders++;
      bandWork.pairs += walked;
    }
    // by how many of the others each is left of: the rule's own order where it is one, and where
    // three or more lines' partings go round in a circle (some pair has to cross somewhere), a
    // fixed order rather than whatever a sort makes of a comparison that contradicts itself
    const wins = new Int32Array(m), c = new Int8Array(m * m);
    for (let i = 0, p = 0; i < m; i++) for (let j = i + 1; j < m; j++, p++) {
      const r = sign[p];
      c[i * m + j] = r;
      c[j * m + i] = -r;
      if (r < 0) wins[i]++;
      else if (r > 0) wins[j]++;
    }
    const idx = v.map((_, i) => i).sort((i, j) => wins[j] - wins[i] || c[i * m + j] || v[i].line.id - v[j].line.id);
    let ids = idx.map((i) => v[i].line.id);
    if (was && ids.length === was.order.length && ids.every((x, k) => x === was.order[k])) ids = was.order;
    order.set(edge, ids);
    if (orderReads) {
      // (the edges where every walk kept ended were read by the kept order: those, with what the walks made now read)
      const add = Int32Array.from(new Set(orderReads));
      const reads = walked === nPairs ? add : add.length ? Int32Array.from(new Set([...was!.reads, ...add])) : was!.reads;
      setOrder(edge, { lines: v.map((x) => x.line), ks: v.map((x) => x.k), paths: v.map((x) => x.line.path), sign, walks, reads, order: ids }, add);
      orderReads = null;
    }
  }
  if (keepOrders && !delta) {
    for (const edge of [...orders.memo.keys()]) if (!visited.has(edge)) setOrder(edge, null);
    for (const edge of [...order.keys()]) if (!visited.has(edge)) order.delete(edge);
  }
  const rule = (edge: number, x: number, y: number) => {
    const v = on.get(edge), a = v?.find((s) => s.line.id === x), b = v?.find((s) => s.line.id === y);
    return a && b ? compare(edge, a, b) : 0;
  };
  const linesOf = (e: number) => order.get(e) ?? on.get(e)!.map((x) => x.line.id);

  blap("orders");
  // ---- the ribbon at each place
  const useCorridors = (opts?.corridors ?? CORRIDORS) && on.size > 1;
  // (T-4153: with `delta`, the corridors look again only at the edges that changed or whose lines did)
  const besideIn = delta && { edges: [delta.edges, delta.moved.edges], has: (e: number) => on.has(e) };
  const beside: Beside = useCorridors ? besideSamples(on, pts, ends, opts?.within, opts?.beside, besideIn || undefined) : { edges: new Map(), pair: new Map() };
  const samples = beside.edges;
  const lowest = (e: number) => {
    const od = order.get(e);
    let m = Infinity;
    if (od) for (const id of od) m = Math.min(m, id);
    else for (const x of on.get(e)!) m = Math.min(m, x.line.id);
    return m;
  };
  /** +1 a to b along edge e for line id, -1 b to a */
  const dirOn = (e: number, id: number) => {
    const x = on.get(e)!.find((s) => s.line.id === id)!;
    return (x.line.path[x.k] & 1) === 1 ? -1 : 1;
  };
  // T-2058 "order": which of two tracks' lines is on the left, decided once for the whole network.
  // Per pair of lines `l,m` (l < m, each the lowest line of a track): metres m's track is left of l's
  // in l's travel, summed over every place the two are beside, and the places; then +1: line l is left
  // of line m in l's travel (ties within COINCIDE_M: +1, the lower number on the left of its travel,
  // wherever the two meet), -1 right, none: never beside
  // (T-4153: the metres summed as whole PAIR_FIX parts, so the sums are the same whatever order the corridors' records
  // come in, which they keep from call to call)
  blap("corridors");
  const pairSums = new Map<string, [number, number]>();
  for (const [e, nb] of beside.pair) {
    if (!on.has(e)) continue;
    const le = lowest(e);
    for (const [f, x] of nb) {
      if (!on.has(f)) continue;
      const lf = lowest(f);
      if (le >= lf) continue;
      const key = `${le},${lf}`, s = pairSums.get(key) ?? [0, 0];
      s[0] += Math.round(x[0] * x[2] * PAIR_FIX) * dirOn(e, le);
      s[1] += x[2];
      pairSums.set(key, s);
    }
  }
  const rel = new Map<string, number>();
  for (const [key, s] of pairSums) rel.set(key, Math.abs(s[0] / s[1]) < COINCIDE_M * PAIR_FIX ? 1 : s[0] > 0 ? -1 : 1);
  /** the pair rules read since `relReads` was set (what a ribbon's order depends on) */
  let relReads: string[] | null = null;
  const relOf = (l: number, m: number): number => {
    const key = `${l},${m}`;
    relReads?.push(key);
    return rel.get(key) ?? 0;
  };

  blap("pairs");
  // ---- T-2238: what changed since the last call, and so the lines to work out again
  const mode = `${useCorridors},${opts?.within ?? ""},${bandMode}`;
  const mmKept = !opts?.debug && !opts?.fresh && !!memo && memo.mode === mode;
  const mm: BandMemo = mmKept ? memo! : { mode, fp: new Map(), ribbons: new Map(), rel: new Map(), lines: new Map(), serial: 0, changedAt: new Map(), net: newNetNumbers() };
  if (!opts?.debug && !opts?.fresh) memo = mm;
  /** edges whose part in any line's bands may differ: their points, samples, lines or those lines' directions */
  const changedEdges = new Set<number>();
  // (T-4153: each edge's record changed in place, no text made of it; `oldPts` the points a changed one had)
  const oldPts = new Map<number, Float32Array | undefined>();
  const record = (e: number, v: readonly OnEdge[]) => {
    // (an edge's lines in order, or as `on` lists them where it has no order, read in place)
    const ed = ends.get(e), od = order.get(e), n = od ? od.length : v.length, p = pts.get(e), sm = samples.get(e);
    let f = mm.fp.get(e);
    if (!f) mm.fp.set(e, (f = { a: -1, b: -1, ids: [], dirs: [], pts: undefined, samples: undefined }));
    let same = f.a === (ed?.a ?? -1) && f.b === (ed?.b ?? -1) && f.pts === p && f.samples === sm && f.ids.length === n && f.dirs.length === v.length;
    for (let i = 0; same && i < n; i++) same = f.ids[i] === (od ? od[i] : v[i].line.id);
    for (let i = 0; same && i < v.length; i++) same = f.dirs[i] === (v[i].line.path[v[i].k] & 1);
    if (same) return;
    changedEdges.add(e);
    oldPts.set(e, f.pts);
    f.a = ed?.a ?? -1; f.b = ed?.b ?? -1; f.pts = p; f.samples = sm;
    f.ids = linesOf(e).slice();
    f.dirs = v.map((x) => x.line.path[x.k] & 1);
  };
  const dropRecord = (e: number) => {
    const f = mm.fp.get(e);
    if (!f) return;
    changedEdges.add(e);
    oldPts.set(e, f.pts);
    mm.fp.delete(e);
    mm.ribbons.delete(e);
  };
  if (delta && beside.remade && mmKept) {
    // T-4153: only the edges whose record can differ: those whose row or points may have, whose lines did, whose order was
    // worked out again, and whose samples the corridors made again
    const look = new Set<number>(delta.edges);
    for (const e of delta.moved.edges) look.add(e);
    for (const e of visit) look.add(e);
    for (const e of beside.remade) look.add(e);
    for (const e of look) {
      const v = on.get(e);
      if (v) record(e, v);
      else dropRecord(e);
    }
  } else {
    for (const [e, v] of on) record(e, v);
    for (const e of [...mm.fp.keys()]) if (!on.has(e)) dropRecord(e);
  }
  blap("records");
  // a pair rule that is new, gone or changed: only what read it is worked out again
  const relChanged = new Set<string>();
  for (const [k, v] of rel) if (mm.rel.get(k) !== v) relChanged.add(k);
  for (const k of mm.rel.keys()) if (!rel.has(k)) relChanged.add(k);
  if (!opts?.debug && !opts?.fresh) bandWork.whole = changedEdges.size;
  if (changedEdges.size > MEMO_EDGES) {
    mm.lines.clear();
    mm.net = newNetNumbers();
    mm.ribbons.clear();
    mm.changedAt.clear();
  }
  const serial = ++mm.serial;
  for (const e of changedEdges) mm.changedAt.set(e, serial);
  // the boxes of the changed edges, as they were and are, for the "hold" stretches near them
  const boxOfPts = (p: Float32Array | undefined, out: number[]) => {
    if (!p) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i + 1 < p.length; i += 3) (x0 = Math.min(x0, p[i])), (y0 = Math.min(y0, p[i + 1])), (x1 = Math.max(x1, p[i])), (y1 = Math.max(y1, p[i + 1]));
    out.push(x0, y0, x1, y1);
  };
  const changedBoxes: number[] = [];
  for (const e of changedEdges) {
    boxOfPts(oldPts.get(e), changedBoxes);
    if (pts.get(e) !== oldPts.get(e)) boxOfPts(pts.get(e), changedBoxes);
  }
  mm.rel = rel;
  const byId = new Map(rail.map((l) => [l.id, l]));
  /** the ribbon changes of lines gone, and of every line kept by group */
  const goneChanges: AlignAt[] = [];
  for (const [id, lm] of mm.lines) if (!byId.has(id)) {
    goneChanges.push(...lm.changes);
    mm.lines.delete(id);
  }
  const byKey = new Map<string, [number, AlignAt][]>();
  for (const [id, lm] of mm.lines) for (const c of lm.changes) {
    let v = byKey.get(c.key);
    if (!v) byKey.set(c.key, (v = []));
    v.push([id, c]);
  }
  /** lines new, on another path, reading an edge that changed, or holding a ribbon near one */
  const redo = new Set<number>();
  // (counted after the loop below)
  for (const l of rail) {
    const lm = mm.lines.get(l.id);
    if (!lm || lm.path.length !== l.path.length || l.path.some((p, k) => p !== lm.path[k]) || lm.deps.some((d) => changedEdges.has(d)) || lm.rels.some((k) => relChanged.has(k))) redo.add(l.id);
    else for (let h = 0; h < lm.holds.length && !redo.has(l.id); h += 4)
      for (let b = 0; b < changedBoxes.length; b += 4)
        if (!(changedBoxes[b + 2] < lm.holds[h] || changedBoxes[b] > lm.holds[h + 2] || changedBoxes[b + 3] < lm.holds[h + 1] || changedBoxes[b + 1] > lm.holds[h + 3])) {
          redo.add(l.id);
          break;
        }
  }

  if (!opts?.debug && !opts?.fresh) bandWork.lines = redo.size;
  blap("redo");
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
    v = onGrid(v);
    lenOf.set(e, v);
    return v;
  };
  const keysOf = (ids: readonly number[]) => ({ key: ids.join(","), back: [...ids].reverse().join(",") });
  const own = new Map<number, Ribbon>();
  const ownRibbon = (e: number): Ribbon => {
    let r = own.get(e);
    if (!r) {
      const ids = linesOf(e), edges = ids.map(() => e), { key, back } = keysOf(ids);
      const shape: Shape = { ids, edges, rids: [...ids].reverse(), redges: edges, key, back, rels: [] };
      own.set(e, (r = { shape, flip: false, lines: ids, edges, key, back, mid: 0 }));
    }
    return r;
  };
  const ribbons = mm.ribbons;
  /** the edges the line being worked out has read (T-2238) */
  let deps = new Set<number>();
  /** the pair rules the line being worked out has read */
  let relDeps = new Set<string>();
  /** the boxes, padded, the line being worked out has looked for tracks in to hold a ribbon across a gap (T-2132) */
  let holdBoxes: number[] = [];
  /** whether a ribbon kept from an earlier call holds: none of its edges changed since (the tracks of
   * one place share one list) */
  const holding = new Map<number[], boolean>();
  const holds = (x: RibbonEntry | undefined): x is RibbonEntry => {
    if (!x) return false;
    let h = holding.get(x.deps);
    if (h === undefined) holding.set(x.deps, (h = x.deps.every((d) => (mm.changedAt.get(d) ?? -1) <= x.at) && (x.at === serial || !x.rels?.some((k) => relChanged.has(k)))));
    return h;
  };
  /** edge e's ribbons, none kept from before it changed */
  const ribbonsOf = (e: number) => {
    let m = ribbons.get(e);
    if (!m || (mm.changedAt.get(e) ?? -1) > m.at) ribbons.set(e, (m = { at: serial, list: new Array(samples.get(e)?.n ?? 1) }));
    return m.list;
  };
  /** keep ribbon r for sample i of edge e unless it has one */
  const put = (e: number, i: number, r: Omit<RibbonEntry, "at">) => {
    const memo = ribbonsOf(e);
    if (!holds(memo[i])) memo[i] = { ...r, at: serial };
  };
  /** the ribbon at sample i of edge e: the tracks side by side there, passed on from track to track */
  const ribbonAt = (e: number, i: number): Ribbon => {
    const S = samples.get(e);
    if (!S || S.first[i] === S.first[i + 1]) return ownRibbon(e);
    const memo = ribbonsOf(e);
    const old = memo[i];
    if (holds(old)) {
      for (const d of old.deps) deps.add(d);
      for (const k of old.rels ?? []) relDeps.add(k);
      return old.r;
    }
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
    const from = all.map((m) => m.e);
    for (const d of from) deps.add(d);
    if (big || mem.length > MAX_TRACKS || hi - lo > SPAN_M || mem.length < 2 || !mem.some((m) => m.e === e)) {
      // the same for each track of it at this place, which need not work it out again
      for (const m of all) put(m.e, m.i, { r: ownRibbon(m.e), deps: from });
      return (memo[i] = { r: ownRibbon(e), deps: from, at: serial }).r;
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
    if (!sh) {
      relReads = [];
      sh = shapeOf(mem.map((m) => [m.e, m.rel * cw]));
      sh.rels = relReads;
      relReads = null;
      shapes.set(sig, sh);
    }
    for (const k of sh.rels) relDeps.add(k);
    const mid = (cw * (lo + hi)) / 2;
    // the ribbon seen from each of its tracks there, in that track's frame
    for (const m of mem) {
      const fwd = m.rel * cw > 0, mm = mid - cw * m.off;
      const v: Ribbon = fwd
        ? { shape: sh, flip: false, lines: sh.ids, edges: sh.edges, key: sh.key, back: sh.back, mid: mm }
        : { shape: sh, flip: true, lines: sh.rids, edges: sh.redges, key: sh.back, back: sh.key, mid: -mm };
      if (m.e === e) memo[i] = { r: v, deps: from, at: serial, rels: sh.rels };
      else put(m.e, m.i, { r: v, deps: from, rels: sh.rels });
    }
    return memo[i]!.r;
  };
  interface Shape { ids: number[]; edges: number[]; rids: number[]; redges: number[]; key: string; back: string; rels: string[] }
  const shapes = new Map<string, Shape>();
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
    return { ids, edges, rids: [...ids].reverse(), redges: [...edges].reverse(), key, back, rels: [] };
  };
  /** a line's slot left of e's a to b in ribbon r, line widths from the ribbon's middle (which is
   * `r.mid` metres from the line's own track) */
  const shiftIn = (r: Ribbon, slot: number) => BAND_SHARE * ((r.lines.length - 1) / 2 - slot);

  // ---- T-2132 "hold": how far a line's track gets from the other tracks of a ribbon it leaves and rejoins
  const cumOf = new Map<number, Float64Array>(), boxOf = new Map<number, Float64Array>();
  const cum = (e: number) => {
    let c = cumOf.get(e);
    if (!c) {
      const p = pts.get(e)!;
      c = new Float64Array(p.length / 3);
      for (let i = 1; i < c.length; i++) c[i] = c[i - 1] + Math.hypot(p[i * 3] - p[i * 3 - 3], p[i * 3 + 1] - p[i * 3 - 2]);
      cumOf.set(e, c);
    }
    return c;
  };
  /** the point a share f of the way along edge e, a to b, local units */
  const pointOn = (e: number, f: number): [number, number] => {
    const p = pts.get(e)!, c = cum(e), want = f * c[c.length - 1];
    let i = 0;
    while (i < c.length - 2 && c[i + 1] < want) i++;
    const sl = c[i + 1] - c[i], t = sl > 0 ? Math.max(0, Math.min(1, (want - c[i]) / sl)) : 0;
    return [p[i * 3] + (p[i * 3 + 3] - p[i * 3]) * t, p[i * 3 + 1] + (p[i * 3 + 4] - p[i * 3 + 1]) * t];
  };
  /** local units from (x, y) to edge e's centre line; Infinity when its box is further than `reach` */
  const boxFor = (e: number) => {
    let b = boxOf.get(e);
    if (!b) {
      const p = pts.get(e)!;
      b = Float64Array.of(Infinity, Infinity, -Infinity, -Infinity);
      for (let i = 0; i < p.length; i += 3) (b[0] = Math.min(b[0], p[i])), (b[1] = Math.min(b[1], p[i + 1])), (b[2] = Math.max(b[2], p[i])), (b[3] = Math.max(b[3], p[i + 1]));
      boxOf.set(e, b);
    }
    return b;
  };
  const distTo = (e: number, x: number, y: number, reach: number) => {
    const p = pts.get(e)!, b = boxFor(e);
    if (x < b[0] - reach || x > b[2] + reach || y < b[1] - reach || y > b[3] + reach) return Infinity;
    let best = Infinity;
    for (let i = 3; i < p.length; i += 3) {
      const ax = p[i - 3], ay = p[i - 2], dx = p[i] - ax, dy = p[i + 1] - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      best = Math.min(best, Math.hypot(ax + t * dx - x, ay + t * dy - y));
    }
    return best;
  };
  /** the most metres, over line l's samples j0 to j1 (`at`: entry and path metres; `own`: the lines on
   * its own track there), from its track to the nearest track of each other line of `ribbon`;
   * Infinity once over HOLD_M */
  const sepAlong = (l: BandLine, j0: number, j1: number, ribbon: readonly number[], at: (j: number) => [number, number], start: Float64Array, own: (j: number) => readonly number[]) => {
    const mine = new Set<number>();
    for (let j = j0; j <= j1; j++) for (const id of own(j)) mine.add(id);
    const others = ribbon.filter((id) => !mine.has(id));
    // the stretch's points (every other sample, 50 m apart, and the last), and per other line its
    // edges near them
    const P: number[] = [];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, perM = 1;
    for (let j = j0; j <= j1; j = j < j1 && j + 2 > j1 ? j1 : j + 2) {
      const [k, m] = at(j), p = l.path[k], e = p >> 1, len = start[k + 1] - start[k];
      if (!pts.get(e) || pts.get(e)!.length < 6) continue;
      const f = len > 0 ? (m - start[k]) / len : 0, [x, y] = pointOn(e, (p & 1) === 1 ? 1 - f : f);
      P.push(x, y);
      (x0 = Math.min(x0, x)), (y0 = Math.min(y0, y)), (x1 = Math.max(x1, x)), (y1 = Math.max(y1, y));
      perM = kAtY(y);
    }
    const reach = HOLD_M * perM;
    holdBoxes.push(x0 - reach, y0 - reach, x1 + reach, y1 + reach);
    const near = others.map((id) => (byId.get(id)?.path ?? []).map((q) => q >> 1).filter((g) => {
      const gp = pts.get(g);
      if (!gp || gp.length < 6) return false;
      const b = boxFor(g);
      return !(b[2] < x0 - reach || b[0] > x1 + reach || b[3] < y0 - reach || b[1] > y1 + reach);
    }));
    let worst = 0;
    for (let i = 0; i < P.length; i += 2) {
      for (const gs of near) {
        let best = Infinity;
        for (const g of gs) best = Math.min(best, distTo(g, P[i], P[i + 1], reach));
        if (best > reach) return Infinity;
        worst = Math.max(worst, best / perM);
      }
    }
    return worst;
  };

  // ---- the Traffic view's table, line by line (T-2238): the recipes a line uses, numbered in the order
  // it first uses them, and per edge of its path its knots' recipes; the network's numbering (ribbons
  // and recipes in the order the lines first use them) is laid over them at the end
  interface Own {
    recipes: OwnRecipe[];
    idx: Map<string, number>;
    recs: Nums;
    recPos: Nums;
    nRecords: number;
    profile: Float32Array | null;
    changes: [string, Change][];
    later: (() => void) | null;
    deps: Set<number>;
    rels: Set<string>;
    holds: number[];
    /** its ribbon changes as they were before lining up */
    aligns?: AlignAt[];
    sweep?: SweepIn;
    recAt?: Int32Array;
    nzE?: Uint8Array;
    smp?: LineSamplesMemo;
  }
  const alignOf = (key: string, c: Change): AlignAt => {
    const [px, py] = c.pointAt(c.step.c);
    return { key, x: c.at[0], y: c.at[1], r: ALIGN_M * kAtY(c.at[1]), prio: c.prio, enter: c.enter, g: c.g, c: c.step.c, L: c.step.L, px, py };
  };
  /** every ribbon change, of the lines kept as they were and of those worked out as they are now,
   * linked to one of `from` through changes of its group within reach of each other */
  const linked = (from: AlignAt[], visit: (id: number, c: AlignAt) => void) => {
    const now = new Map<string, [number, AlignAt][]>();
    for (const [id, w] of owns) for (const a of w.aligns ?? []) {
      let v = now.get(a.key);
      if (!v) now.set(a.key, (v = []));
      v.push([id, a]);
    }
    const seen = new Set<AlignAt>(from), todo = [...from];
    for (let f = 0; f < todo.length; f++) {
      const a = todo[f];
      for (const list of [byKey.get(a.key), now.get(a.key)]) for (const [id, c] of list ?? []) {
        if (seen.has(c) || Math.hypot(c.x - a.x, c.y - a.y) > Math.max(a.r, c.r)) continue;
        seen.add(c);
        visit(id, c);
        todo.push(c);
      }
    }
  };
  const ribKeys = new WeakMap<Ribbon, string>();
  const recipeIn = (o: Own, r: Ribbon, slot: number, sgn: number) => {
    // T-4153: a line alone on its track has one recipe wherever it is (`ALONE`): its Traffic shift is none on any edge
    if (r.lines.length === 1) {
      let id = o.idx.get(ALONE.key);
      if (id === undefined) {
        o.recipes.push(ALONE);
        o.idx.set(ALONE.key, (id = o.recipes.length - 1));
      }
      return id;
    }
    let rk = ribKeys.get(r);
    if (rk === undefined) ribKeys.set(r, (rk = `${r.lines.join(",")}|${r.edges.join(",")}`));
    const key = `${rk}/${slot}/${sgn > 0 ? 1 : 0}`;
    let id = o.idx.get(key);
    if (id === undefined) {
      const rib: number[] = [];
      r.lines.forEach((x, s) => rib.push(x, r.edges[s]));
      o.recipes.push({ key, rk, rib, slot, sgn });
      o.idx.set(key, (id = o.recipes.length - 1));
    }
    return id;
  };

  // ---- T-2140 "align": a ribbon change is a curve per line, each placed by its own track, so a
  // change at a node on one track can be along the way on the one beside it, and the lines of one
  // ribbon then ease at different places (in the Traffic view, where bands are wide, a bulge). The
  // changes of one ribbon within ALIGN_M of each other take one placement: the curve of the line
  // whose change says most about the place (a line joining its track, then a node, then along the
  // way; then the lowest id), its middle at the same point on every line's track
  interface Change { id: number; step: { c: number; L: number }; prio: number; g: number; enter: boolean; at: [number, number]; pointAt: (m: number) => [number, number]; nearM: (P: [number, number], m: number, reach: number) => number }
  const alignChanges = (all: Map<string, Change[]>) => {
    for (const list of all.values()) {
      if (list.length < 2) continue;
      const used = new Uint8Array(list.length);
      for (let i = 0; i < list.length; i++) {
        if (used[i]) continue;
        used[i] = 1;
        const a = list[i], grp = [a], ids = new Set([a.id]), reach = ALIGN_M * kAtY(a.at[1]);
        for (let j = i + 1; j < list.length; j++) {
          const b = list[j];
          if (used[j] || ids.has(b.id) || Math.hypot(b.at[0] - a.at[0], b.at[1] - a.at[1]) > reach) continue;
          used[j] = 1;
          ids.add(b.id);
          grp.push(b);
        }
        if (grp.length < 2) continue;
        let lead = grp[0];
        for (const x of grp) if (x.prio > lead.prio || (x.prio === lead.prio && x.id < lead.id)) lead = x;
        const P = lead.pointAt(lead.step.c);
        if (!Number.isFinite(P[0])) continue;
        for (const x of grp) if (x !== lead) {
          const m = x.nearM(P, x.g, lead.step.L);
          // (its track parted from the leader's there: as far from its own change, into or out of
          // the bigger ribbon alike)
          x.step.c = Number.isFinite(m) ? m : x.g + (lead.step.c - lead.g) * (lead.enter === x.enter ? 1 : -1);
          x.step.L = lead.step.L;
        }
      }
    }
  };

  // ---- along each line to work out again (T-2238: the others keep theirs); and then any line kept that
  // shares a T-2140 alignment group with one worked out, as the group is lined up together
  const owns = new Map<number, Own>();
  const queue = rail.filter((l) => redo.has(l.id));
  for (let li = 0; ; ) {
   for (; li < queue.length; li++) {
    const l = queue[li];
    blap("q.other");
    // (T-4153: its own recipes numbered as its last bands had them, a new one after them)
    const lmWas = mmKept ? mm.lines.get(l.id) : undefined;
    const mine: Own = { recipes: lmWas ? lmWas.recipes.slice() : [], idx: lmWas ? new Map(lmWas.recipes.map((rc, i) => [rc.key, i])) : new Map(), recs: new Nums(), recPos: new Nums(), nRecords: 0, profile: null, changes: [], later: null, deps: new Set(l.path.map((p) => p >> 1)), rels: new Set(), holds: [] };
    owns.set(l.id, mine);
    deps = mine.deps;
    relDeps = mine.rels;
    holdBoxes = mine.holds;
    const recipeOf = (r: Ribbon, slot: number, sgn: number) => recipeIn(mine, r, slot, sgn);
    const n = l.path.length;
    // its samples in travel order: metres along its path, half their spacing, entry, ribbon, slot
    const M: number[] = [], H: number[] = [], K: number[] = [], R: Ribbon[] = [], S: number[] = [], dbgF: number[] = [];
    const start = new Float64Array(n + 1);
    let any = false;
    // T-4153: an entry's samples as the line's last bands had them where nothing they read has changed since (its piece of
    // track and every edge and pair rule its ribbons were worked out from), moved along by its start's change (the path grid)
    const was0 = lmWas?.smp && !opts?.debug ? lmWas.smp : null, pd = was0 ? pathDiff(lmWas!.path, l.path) : null;
    const preK = !was0 ? 0 : pd ? pd.pre : n, sufK = !was0 || !pd ? 0 : pd.suf, grow = pd ? n - pd.was : 0;
    const smp: LineSamplesMemo = { js: new Int32Array(n + 1), start, any: new Uint8Array(n), deps: new Array(n), rels: new Array(n), T: new Array(n), len: new Float64Array(n), M, H, R, S };
    const lineDeps = mine.deps, lineRels = mine.rels;
    const holdsAll = (ds: readonly number[], rs: readonly string[] | null) => ds.every((d) => !changedEdges.has(d)) && (!rs || rs.every((x) => !relChanged.has(x)));
    for (let k = 0; k < n; k++) {
      const p = l.path[k], e = p >> 1, rev = (p & 1) === 1, len = edgeLen(e);
      start[k + 1] = start[k] + len;
      smp.js[k] = M.length;
      smp.len[k] = len;
      smp.deps[k] = [];
      smp.rels[k] = null;
      if (!on.has(e)) continue;
      const T = samples.get(e), ns = T?.n ?? 1;
      smp.T[k] = T;
      const ko = !was0 ? -1 : k < preK ? k : k >= n - sufK ? k - grow : -1;
      if (ko >= 0 && was0!.T[ko] === T && was0!.len[ko] === len && holdsAll(was0!.deps[ko], was0!.rels[ko])) {
        const a = was0!.js[ko], b = was0!.js[ko + 1], d = start[k] - was0!.start[ko];
        for (let j = a; j < b; j++) {
          M.push(was0!.M[j] + d);
          H.push(was0!.H[j]);
          K.push(k);
          R.push(was0!.R[j]);
          S.push(was0!.S[j]);
        }
        if (was0!.any[ko]) (any = true), (smp.any[k] = 1);
        smp.deps[k] = was0!.deps[ko];
        smp.rels[k] = was0!.rels[ko];
        for (const x of smp.deps[k]) lineDeps.add(x);
        for (const x of smp.rels[k] ?? []) lineRels.add(x);
        bandWork.samplesKept += b - a;
        continue;
      }
      // (what the entry's ribbons read, its own edge's lines among it)
      const ed = new Set<number>([e]), er = new Set<string>();
      deps = ed;
      relDeps = er;
      let alone = -1, b1 = start[k];
      for (let s = 0; s < ns; s++) {
        const i = rev ? ns - 1 - s : s;
        // (its bounds on the path grid, from its entry's start)
        const b0 = b1;
        b1 = start[k] + onGrid(((s + 1) / ns) * len);
        // samples with nothing beside, one after another, are one (long edges in open country)
        if (T && T.first[i] === T.first[i + 1] && alone >= 0 && !opts?.debug) {
          const m0 = M[alone] - H[alone];
          M[alone] = (m0 + b1) / 2;
          H[alone] = (b1 - m0) / 2;
          continue;
        }
        const r = ribbonAt(e, i);
        let slot = -1;
        for (let q = 0; q < r.lines.length && slot < 0; q++) if (r.lines[q] === l.id && r.edges[q] === e) slot = q;
        if (slot < 0) continue;
        if (r.lines.length > 1) (any = true), (smp.any[k] = 1);
        if (opts?.debug) dbgF.push((i + 0.5) / ns);
        alone = T && T.first[i] === T.first[i + 1] ? M.length : -1;
        M.push((b0 + b1) / 2);
        H.push((b1 - b0) / 2);
        K.push(k);
        R.push(r);
        S.push(slot);
      }
      deps = lineDeps;
      relDeps = lineRels;
      smp.deps[k] = [...ed];
      smp.rels[k] = er.size ? [...er] : null;
      for (const x of ed) lineDeps.add(x);
      for (const x of er) lineRels.add(x);
      bandWork.samplesMade += M.length - smp.js[k];
    }
    smp.js[n] = M.length;
    mine.smp = smp;
    blap("q.sample");
    if (!any) continue;
    /** the point at path metre m, local units */
    const pointAt = (m: number): [number, number] => {
      let lo = 0, hi = n - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (start[mid] <= m) lo = mid;
        else hi = mid - 1;
      }
      const p = l.path[lo], e = p >> 1, len = start[lo + 1] - start[lo], P = pts.get(e);
      if (!P || P.length < 6) return P && P.length >= 3 ? [P[0], P[1]] : [NaN, NaN];
      const f = len > 0 ? Math.max(0, Math.min(1, (m - start[lo]) / len)) : 0;
      return pointOn(e, (p & 1) === 1 ? 1 - f : f);
    };
    /** the path metre within `reach` of `m` nearest point P, or NaN where none is within ALIGN_NEAR_M */
    const nearM = (P: [number, number], m: number, reach: number) => {
      const d = (x: number) => {
        const q = pointAt(x);
        return Math.hypot(q[0] - P[0], q[1] - P[1]);
      };
      let best = m, bd = Infinity;
      for (let x = Math.max(0, m - reach); x <= Math.min(start[n], m + reach); x += 10) {
        const v = d(x);
        if (v < bd) (bd = v), (best = x);
      }
      for (let x = Math.max(0, best - 10); x <= Math.min(start[n], best + 10); x += 1) {
        const v = d(x);
        if (v < bd) (bd = v), (best = x);
      }
      return bd / kAtY(P[1]) <= ALIGN_NEAR_M ? best : NaN;
    };
    const N = M.length, sgnAt = (j: number) => ((l.path[K[j]] & 1) === 1 ? -1 : 1);
    // per sample its slot (line widths) and its ribbon's middle (m), left of travel, and the two as
    // one value at BAND_LW_PER_M, which decides what is one piece
    const VS = new Float64Array(N), VM = new Float64Array(N), V = new Float64Array(N), KEY: string[] = new Array(N);
    for (let j = 0; j < N; j++) {
      const sg = sgnAt(j);
      VS[j] = sg * shiftIn(R[j], S[j]);
      VM[j] = sg * R[j].mid;
      V[j] = VS[j] + VM[j] * BAND_LW_PER_M;
      // its lines in order left to right of travel
      KEY[j] = sg > 0 ? R[j].key : R[j].back;
    }
    const val = (j: number) => VS[j], valM = (j: number) => VM[j];
    // runs of one ribbon along the path
    type Run = { key: string; s0: number; s1: number; r: Ribbon };
    const runs: Run[] = [];
    for (let j = 0; j < N; j++) {
      const last = runs[runs.length - 1];
      if (last && last.key === KEY[j]) last.s1 = j;
      else runs.push({ key: KEY[j], s0: j, s1: j, r: R[j] });
    }
    blap("q.runs");
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
    // T-2132: a ribbon whose tracks part for a while and come together again, as they were, is held
    // across the gap while they stay within HOLD_M of each other, so zoomed out (where a line is
    // wider than the gap) the bands do not close up onto their tracks and spread out again
    const holds: [number, number][] = [];
    if (bandMode.includes("hold")) for (let u = 1; u + 1 < runs.length; u++) {
      const a = runs[u - 1], x = runs[u], b = runs[u + 1];
      if (a.key !== b.key || lenRun(x) > HOLD_RUN_M || !x.r.lines.every((id) => a.r.lines.includes(id)) || !keeps(x, a)) continue;
      const sep = sepAlong(l, x.s0, x.s1, a.r.lines, (j) => [K[j], M[j]], start, (j) => linesOf(R[j].edges[S[j]]));
      holds.push([Math.round(lenRun(x)), Math.round(sep)]);
      if (sep > HOLD_M) continue;
      for (let j = x.s0; j <= x.s1; j++) dropped[j] = 1;
      a.s1 = b.s1;
      runs.splice(u, 2);
      u--;
    }
    blap("q.drophold");
    opts?.debug?.set(l.id, { M, f: dbgF, key: KEY, v: [...V], entry: K, dropped: [...dropped], runs: runs.map((u) => [u.key, u.s0, u.s1]), holds });
    // knots in path metres: per run its value, eased into the next over CORRIDOR_EASE_M, or at a
    // node between different lines over BAND_RAMP_M on the side with fewer (T-1799 stage 4)
    // (each knot's Traffic recipes: recipe and weight pairs in kf, from ko, kn of them)
    const km: number[] = [], kv: number[] = [], kw: number[] = [], kq: number[] = [], ko: number[] = [], kn: number[] = [], kf: number[] = [];
    const push = (m: number, j: number) => {
      if (km.length && m <= km[km.length - 1] + 1e-3) m = km[km.length - 1] + 1e-3;
      km.push(m);
      kv.push(val(j));
      kw.push(valM(j));
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
        if (A.rc === B.rc && Math.abs(val(B.j0) - val(A.j0)) < 1e-6 && Math.abs(valM(B.j0) - valM(A.j0)) < 1e-6 && Math.abs(qOf(B.j0) - qOf(A.j0)) < 1e-6) {
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
        // T-2140: each line places its curve by its own track (a node on one, along the way on the
        // other), so the lines of one ribbon change are lined up afterwards (`alignChanges`)
        if (A.run !== B.run && ra !== rb && bandMode.includes("align")) {
          const big = [...(ra > rb ? rs[A.run] : rs[B.run]).r.lines].sort((x, y) => x - y).join(",");
          mine.changes.push([big, { id: l.id, step: steps[steps.length - 1], prio: nodeM >= 0 && oa !== ob ? 3 : node ? 2 : 1, g: (g0 + g1) / 2, enter: rb > ra, at: pointAt((g0 + g1) / 2), pointAt, nearM }]);
        }
      }
      // (each piece's values and recipe, for the sweep)
      const P = pl.length, pv = new Float64Array(P), pw = new Float64Array(P), pq = new Float64Array(P), prc = new Int32Array(P);
      for (let p = 0; p < P; p++) {
        pv[p] = val(pl[p].j0);
        pw[p] = valM(pl[p].j0);
        pq[p] = qOf(pl[p].j0);
        prc[p] = pl[p].rc;
      }
      return { pl, steps, pv, pw, pq, prc };
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
        if (p && Math.abs(V[j] - V[p[2]]) <= SHIFT_TOL && K[j0] === K[p[0]]) (p[1] = j0), (p[3] = j);
        else pieces.push([j0, j0, j, j]);
      }
      return pieces;
    };
    // (the rest once every line's changes are lined up)
    const sm = bandMode ? smoothKnots(runs.map(piecesOf), runs) : undefined;
    blap("q.smooth");
    mine.later = () => {
    if (bandMode && !sm) return;
    const sw: SweepOut = { km, kv, kw, kq, ko, kn, kf, nv: new Float64Array(n + 1), nq: new Float64Array(n + 1), nw: new Float64Array(n + 1), no: new Int32Array(n + 1), nn: new Int32Array(n + 1) };
    // (the line's last bands, which an edit's lines lend their entries from)
    const was = !opts?.debug && !opts?.fresh ? mm.lines.get(l.id) : undefined;
    let plan: Splice | null = null, map: Int32Array | null = null, now: SweepIn | null = null;
    if (sm) {
      const { pl, steps } = sm;
      const dbgL = opts?.debug?.get(l.id) as { steps?: number[][] } | undefined;
      if (dbgL) dbgL.steps = steps.map((s, i) => [s.c, s.L, val(pl[i].j0), valM(pl[i].j0), val(pl[i + 1].j0), valM(pl[i + 1].j0)]);
      const ns = steps.length, sc = new Float64Array(ns), sL = new Float64Array(ns);
      for (let k = 0; k < ns; k++) (sc[k] = steps[k].c), (sL[k] = steps[k].L);
      now = { start, pv: sm.pv, pw: sm.pw, pq: sm.pq, prc: sm.prc, sc, sL };
      mine.sweep = now;
      // T-4153: only the entries whose knots this call can change are worked out again, from a quiet place before them
      if (was?.sweep && was.recAt) {
        map = recipeMap(was.recipes, mine.idx);
        plan = spliceOf(was.sweep, was.path, now, l.path, map, (e) => changedEdges.has(e));
        if (plan && spliceTest) plan = { ...plan, e0: Math.max(plan.e0, plan.pre), e1: Math.max(Math.max(plan.e0, plan.pre), Math.min(plan.e1, n - plan.suf)) };
      }
      if (plan) sweepKnots(now, plan.k0, plan.k1, plan.e0, plan.e1, plan.first, plan.last, sw);
      else sweepKnots(now, 0, ns, 0, n, true, true, sw);
    } else {
    runs.forEach((u, ui) => {
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
      knotsAtNodes(start, sw);
    }
    // per path entry: its knots (its ends' nodes' and those inside it) and its Traffic record; with a plan, the entries
    // before e0 and from e1 on as the line's last bands had them
    const knots = new Nums(), starts = new Nums(), recs = mine.recs, recPos = mine.recPos, recAt = new Int32Array(n).fill(-1), nzE = new Uint8Array(n);
    const acc = { nonzero: false, records: 0, nzE };
    // (its recipes numbered as before where its own are, T-4153: the copied knots and records then copied as they are)
    let same = !!map;
    for (let i = 0; same && i < map!.length; i++) same = map![i] === i;
    let done = false;
    if (plan && was) {
      if (was.profile) {
        done = copyEntries(was, 0, plan.e0, map!, same, 0, knots, starts, recs, recPos, recAt, acc);
        entryKnots(l.id, l.path, start, ends, pts, sw, plan.e0, plan.e1, knots, starts, recs, recPos, recAt, acc);
        done = done && copyEntries(was, plan.e1 - plan.grow, n - plan.grow, map!, same, plan.e1, knots, starts, recs, recPos, recAt, acc);
      } else {
        // (its last bands were none anywhere, so the entries it lends are none: it has none now if those worked out are)
        entryKnots(l.id, l.path, start, ends, pts, sw, plan.e0, plan.e1, knots, starts, recs, recPos, recAt, acc);
        done = !acc.nonzero;
      }
      if (done) {
        bandWork.spliced++;
        bandWork.swept += plan.e1 - plan.e0;
        bandWork.copied += n - (plan.e1 - plan.e0);
        bandWork.splicedAll++;
        bandWork.sweptAll += plan.e1 - plan.e0;
        bandWork.copiedAll += n - (plan.e1 - plan.e0);
      } else {
        knots.length = starts.length = recs.length = recPos.length = 0;
        recAt.fill(-1);
        nzE.fill(0);
        acc.nonzero = false;
        acc.records = 0;
        km.length = kv.length = kw.length = kq.length = ko.length = kn.length = kf.length = 0;
        sweepKnots(now!, 0, now!.sc.length, 0, n, true, true, sw);
      }
    }
    if (!done) entryKnots(l.id, l.path, start, ends, pts, sw, 0, n, knots, starts, recs, recPos, recAt, acc);
    mine.nRecords = acc.records;
    mine.recAt = recAt;
    mine.nzE = nzE;
    if (!acc.nonzero) {
      recs.length = recPos.length = mine.nRecords = 0;
      return;
    }
    starts.push(knots.length / KNOT);
    const out = new Float32Array(1 + starts.length + knots.length);
    out[0] = n;
    out.set(starts.view(), 1);
    out.set(knots.view(), 1 + starts.length);
    mine.profile = out;
    };
   }
   // T-2140's lining up pairs only changes of one group within reach of each other, greedily in line
   // order, so it can come out otherwise only among changes linked to a new, gone or moved change
   // through changes within reach: the lines kept with one of those are worked out again
   for (const w of owns.values()) w.aligns ??= w.changes.map(([k, c]) => alignOf(k, c));
   const front: AlignAt[] = [...goneChanges];
   for (const [id, w] of owns) {
     const old = mm.lines.get(id)?.changes ?? [];
     for (const a of w.aligns!) if (!old.some((b) => sameChange(a, b))) front.push(a);
     for (const b of old) if (!w.aligns!.some((a) => sameChange(a, b))) front.push(b);
   }
   const more = new Set<number>();
   linked(front, (id) => { if (!owns.has(id)) more.add(id); });
   if (!more.size) break;
   queue.push(...rail.filter((l) => more.has(l.id)));
  }
  /** T-2140: every line's ribbon changes by the bigger ribbon's lines, in line order; with those of
   * lines kept that the lines worked out are lined up with, as they were (what they come to is
   * theirs already) */
  blap("lines");
  const kept = new Map<number, Set<AlignAt>>();
  linked([...owns.values()].flatMap((w) => w.aligns!), (id, c) => {
    if (owns.has(id)) return;
    let s = kept.get(id);
    if (!s) kept.set(id, (s = new Set()));
    s.add(c);
  });
  const changes = new Map<string, Change[]>();
  const into = (k: string, c: Change) => {
    let v = changes.get(k);
    if (!v) changes.set(k, (v = []));
    v.push(c);
  };
  for (const l of rail) {
    const w = owns.get(l.id), s = kept.get(l.id);
    if (w) for (const [k, c] of w.changes) into(k, c);
    else if (s) for (const a of mm.lines.get(l.id)!.changes) if (s.has(a))
      into(a.key, { id: l.id, step: { c: a.c, L: a.L }, prio: a.prio, g: a.g, enter: a.enter, at: [a.x, a.y], pointAt: () => [a.px, a.py], nearM: () => NaN });
  }
  blap("alignPrep");
  alignChanges(changes);
  blap("align");
  for (const l of rail) {
    const w = owns.get(l.id);
    if (!w) continue;
    w.later?.();
    mm.lines.set(l.id, {
      path: l.path.slice(), deps: [...w.deps], rels: [...w.rels], holds: w.holds, changes: w.aligns!, recipes: w.recipes,
      profile: w.profile, recs: Float32Array.from(w.recs.view()), recPos: Int32Array.from(w.recPos.view()), nRecords: w.nRecords, sweep: w.sweep, recAt: w.recAt, nzE: w.nzE, smp: w.smp, netWas: mm.lines.get(l.id)?.net?.map,
    });
  }
  blap("later");
  // ---- the network's table and profiles. T-4153: ribbons and recipes keep their numbers from call to call (`BandMemo.net`),
  // a new one numbered after the rest, so a line's profile and records are numbered again only when it is worked out again
  const reg = mm.net;
  const numberOf = (rc: OwnRecipe) => {
    let id = reg.recIds.get(rc.key);
    if (id !== undefined) return id;
    let rid = reg.ribIds.get(rc.rk);
    if (rid === undefined) {
      reg.ribIds.set(rc.rk, (rid = reg.nRib++));
      reg.ribs.push(rc.rib.length / 2);
      for (const x of rc.rib) reg.ribs.push(x);
    }
    reg.recs.push(rid);
    reg.recs.push2(rc.slot, rc.sgn);
    reg.recIds.set(rc.key, (id = reg.nRec++));
    return id;
  };
  const of = new Map<number, BandProfile>(), ownOf = new Map<number, BandProfile>();
  let nRec = 0, nRecords = 0;
  for (const l of rail) {
    const lm = mm.lines.get(l.id)!;
    let nt = lm.net;
    if (!nt) {
      const was = lm.netWas, map = new Int32Array(lm.recipes.length);
      for (let k = 0; k < map.length; k++) map[k] = was && k < was.length ? was[k] : numberOf(lm.recipes[k]);
      let profile: Float32Array | null = null;
      if (lm.profile) {
        profile = lm.profile.slice();
        for (let i = 2 + profile[0] + 3; i < profile.length; i += KNOT) if (profile[i] >= 0) profile[i] = map[profile[i]];
      }
      const recs = lm.recs.slice();
      for (const x of lm.recPos) recs[x] = map[lm.recs[x]];
      lm.net = nt = { map, profile, recs };
      lm.netWas = undefined;
    }
    if (nt.profile) {
      of.set(l.id, nt.profile);
      ownOf.set(l.id, lm.profile!);
    }
    nRec += nt.recs.length;
    nRecords += lm.nRecords;
  }
  blap("maps");
  // (the records after the ribbons and recipes, each line's as its `net` has them)
  const rb = reg.ribs.view(), rr = reg.recs.view(), base = 3 + rb.length + rr.length, corridor = new Float32Array(base + nRec);
  corridor[0] = reg.nRib;
  corridor.set(rb, 1);
  corridor[1 + rb.length] = reg.nRec;
  corridor.set(rr, 2 + rb.length);
  corridor[2 + rb.length + rr.length] = nRecords;
  let at = base;
  for (const l of rail) {
    const r = mm.lines.get(l.id)!.net!.recs;
    corridor.set(r, at);
    at += r.length;
  }
  blap("table");
  return { of, own: ownOf, order, rule, corridor };
}

/** T-4153: numbers pushed one at a time or copied a block at a time (a line's knots and records lent from its last
 * bands), in float64 as a number array holds them */
class Nums {
  a = new Float64Array(256);
  length = 0;
  room(n: number) {
    if (this.length + n <= this.a.length) return;
    const b = new Float64Array(Math.max(this.a.length * 2, this.length + n));
    b.set(this.a.subarray(0, this.length));
    this.a = b;
  }
  push(x: number) {
    this.room(1);
    this.a[this.length++] = x;
  }
  push2(x: number, y: number) {
    this.room(2);
    this.a[this.length++] = x;
    this.a[this.length++] = y;
  }
  block(src: Float32Array, from: number, to: number) {
    this.room(to - from);
    this.a.set(src.subarray(from, to), this.length);
    this.length += to - from;
  }
  view() {
    return this.a.subarray(0, this.length);
  }
}

/** T-4153: what a line's knots are swept from: its entries' starts (path metres, on the path grid), its pieces' values
 * (shift, the ribbon's middle, Traffic width) and own recipes, and its steps' middles and lengths; kept with its bands so
 * the next call can tell what changed */
interface SweepIn { start: Float64Array; pv: Float64Array; pw: Float64Array; pq: Float64Array; prc: Int32Array; sc: Float64Array; sL: Float64Array }
/** a sweep's knots (place, shift, ribbon's middle, Traffic width, and recipe pairs in kf from ko, kn of them) and its value
 * at each node swept (nv, nw, nq, recipes from no, nn) */
interface SweepOut { km: number[]; kv: number[]; kw: number[]; kq: number[]; ko: number[]; kn: number[]; kf: number[]; nv: Float64Array; nq: Float64Array; nw: Float64Array; no: Int32Array; nn: Int32Array }

/** T-2058 "smooth": the pieces' values as steps, each eased over an S-curve and added up, swept along the path over steps
 * [k0, k1) from a quiet place (the steps before k0 passed, none under way: piece k0 whole), with knots at their curves'
 * knot places (and at the path's start and end when `first`, `last`) and the value at nodes [a, b] (past b only when
 * `last`). Each knot a mix of the pieces: piece p weighs how far the step into it has gone less how far the step out of it
 * has (the first piece, plus per step k its S times piece k + 1 less piece k: the steps passed folded into `base`, only
 * those under way worked out at each place), so the Traffic view's widths mix exactly as the Line view's shifts. T-2059:
 * in flat arrays, each list kept in the order a Map or Set would keep it, so the sums are added in the same order however
 * far back the sweep starts. */
function sweepKnots(s: SweepIn, k0: number, k1: number, a: number, b: number, first: boolean, last: boolean, o: SweepOut): void {
  const { start, pv, pw, pq, prc, sc, sL } = s, { km, kv, kw, kq, ko, kn, kf, nv, nq, nw, no, nn } = o;
  const end = start[start.length - 1];
  const at: number[] = [];
  if (first) at.push(0);
  if (last) at.push(end);
  for (let k = k0; k < k1; k++) if (sL[k] > 0) for (let i = 0; i < SMOOTH_KNOTS; i++) {
    const m = sc[k] + sL[k] * (i / (SMOOTH_KNOTS - 1) - 0.5);
    if (m > 0 && m < end) at.push(m);
  }
  const ms = Float64Array.from(at).sort();
  const P = pv.length, ns = k1 - k0;
  const lo = new Float64Array(ns), hi = new Float64Array(ns);
  for (let k = 0; k < ns; k++) {
    lo[k] = sc[k0 + k] - sL[k0 + k] / 2;
    hi[k] = sc[k0 + k] + sL[k0 + k] / 2;
  }
  const byLo = [...lo.keys()].sort((x, y) => lo[x] - lo[y]), byHi = [...hi.keys()].sort((x, y) => hi[x] - hi[y]);
  /** the steps passed, per piece, and the pieces whose sum is not zero */
  const base = new Float64Array(P), bk: number[] = [k0];
  base[k0] = 1;
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
  // (a step of no length whose place is one swept passes there before it can start: `gone` keeps it from starting after,
  // so whether a step is under way at a place does not depend on the places swept before it)
  const going: number[] = [], inGo = new Uint8Array(ns), gone = new Uint8Array(ns);
  const W = new Float64Array(P), mark = new Uint8Array(P), touched: number[] = [];
  const touch = (p: number, w: number) => {
    if (!mark[p]) {
      mark[p] = 1;
      touched.push(p);
    }
    W[p] += w;
  };
  let v = 0, vm = 0, q = 0;
  /** the mix at m into v, vm, q, its recipes' weights onto kf: where they start */
  const mix = (m: number) => {
    touched.length = 0;
    for (const p of bk) touch(p, base[p]);
    for (const k of going) {
      const t = S3((m - sc[k0 + k]) / sL[k0 + k] + 0.5);
      touch(k0 + k + 1, t);
      touch(k0 + k, -t);
    }
    v = vm = q = 0;
    const o = kf.length;
    for (const p of touched) {
      const w = W[p];
      W[p] = 0;
      mark[p] = 0;
      if (Math.abs(w) < 1e-9) continue;
      v += w * pv[p];
      vm += w * pw[p];
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
    return o;
  };
  let iL = 0, iH = 0, x = 0, y = a;
  for (;;) {
    const mk = x < ms.length ? ms[x] : Infinity, mn = y <= b ? start[y] : Infinity, m = mk < mn ? mk : mn;
    if (m === Infinity || (!last && y > b)) break;
    while (iL < ns && lo[byLo[iL]] < m) {
      const k = byLo[iL++];
      if (gone[k]) continue;
      inGo[k] = 1;
      going.push(k);
    }
    while (iH < ns && hi[byHi[iH]] <= m) {
      const k = byHi[iH++];
      gone[k] = 1;
      if (inGo[k]) {
        inGo[k] = 0;
        going.splice(going.indexOf(k), 1);
      }
      bput(k0 + k + 1, 1);
      bput(k0 + k, -1);
    }
    // (a place twice is one knot, and a place within 1e-3 of the last knot none)
    let knot = false;
    if (mk === m) {
      knot = !(x > 0 && m === ms[x - 1]) && !(km.length && m <= km[km.length - 1] + 1e-3);
      x++;
    }
    const node = mn === m;
    if (!knot && !node) continue;
    const off = mix(m);
    if (knot) {
      km.push(m);
      kv.push(v);
      kw.push(vm);
      kq.push(Math.max(0, q));
      ko.push(off);
      kn.push((kf.length - off) / 2);
    }
    if (node) {
      nv[y] = v;
      nq[y] = Math.max(0, q);
      nw[y] = vm;
      no[y] = off;
      nn[y] = (kf.length - off) / 2;
      y++;
    }
  }
}
const S3 = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/** the value at each node from knots alone (no "smooth"): the knots either side mixed, as far along as it lies */
function knotsAtNodes(start: Float64Array, o: SweepOut): void {
  const { km, kv, kw, kq, ko, kn, kf, nv, nq, nw, no, nn } = o;
  let q = 0;
  const listOf = (i: number) => kf.slice(ko[i], ko[i] + 2 * kn[i]);
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
  for (let k = 0; k < start.length; k++) {
    const m = start[k];
    while (q + 1 < km.length && km[q + 1] < m) q++;
    const z = km.length - 1;
    let r: [number, number, number[], number];
    if (m <= km[0]) r = [kv[0], kq[0], listOf(0), kw[0]];
    else if (q + 1 >= km.length || m >= km[z]) r = [kv[z], kq[z], listOf(z), kw[z]];
    else {
      const t = (m - km[q]) / (km[q + 1] - km[q]);
      r = [kv[q] + (kv[q + 1] - kv[q]) * t, kq[q] + (kq[q + 1] - kq[q]) * t, mixed(q, t), kw[q] + (kw[q + 1] - kw[q]) * t];
    }
    nv[k] = r[0];
    nq[k] = r[1];
    nw[k] = r[3];
    no[k] = kf.length;
    nn[k] = r[2].length / 2;
    for (const x of r[2]) kf.push(x);
  }
}

/** T-4153: entries [e0, e1) of a line to work out again, the rest as its last bands had them (those from e1 on from its
 * entry e1 - grow on); the sweep for them over steps [k0, k1) (`first`, `last`: from the path's start, to its end); its
 * entries alike from the start (pre) and the end (suf) */
interface Splice { e0: number; e1: number; k0: number; k1: number; first: boolean; last: boolean; grow: number; pre: number; suf: number }
/** T-4153: what of a line's last bands an edit leaves. Its entries alike from the start (the same entry and start, on an
 * edge whose row and points did not change) and from the end (moved by one exact amount, the path grid), and its pieces
 * and steps alike from the start and (moved) from the end, give the metres where the sweep's values can differ; the
 * entries over them are worked out again, swept from a quiet place before them (no step under way, the steps before it
 * all passed, so the sweep there is one piece whole, as from the path's start) to the first node past a quiet place after
 * them, from which the sweep goes on as it went. */
function spliceOf(was: SweepIn, wasPath: readonly number[], now: SweepIn, path: readonly number[], map: Int32Array, changed: (e: number) => boolean): Splice | null {
  const n = path.length, m = wasPath.length, lim = Math.min(n, m), S = now.start, Sw = was.start;
  const shift = S[n] - Sw[m];
  let pre = 0, suf = 0;
  while (pre < lim && path[pre] === wasPath[pre] && S[pre + 1] === Sw[pre + 1] && !changed(path[pre] >> 1)) pre++;
  if (pre === n && n === m) suf = n;
  else while (suf < lim - pre && path[n - 1 - suf] === wasPath[m - 1 - suf] && S[n - 1 - suf] - shift === Sw[m - 1 - suf] && !changed(path[n - 1 - suf] >> 1)) suf++;
  const np = now.pv.length, mp = was.pv.length, ns = now.sc.length, ms = was.sc.length;
  if (!np || !mp) return null;
  const alike = (i: number, j: number) => now.pv[i] === was.pv[j] && now.pw[i] === was.pw[j] && now.pq[i] === was.pq[j] && now.prc[i] === map[was.prc[j]];
  const head = alike(0, 0), tail = alike(np - 1, mp - 1);
  let P = 0, Q = 0;
  if (head) while (P < ns && P < ms && now.sc[P] === was.sc[P] && now.sL[P] === was.sL[P] && alike(P + 1, P + 1)) P++;
  if (tail) while (Q < Math.min(ns, ms) - P && now.sc[ns - 1 - Q] - shift === was.sc[ms - 1 - Q] && now.sL[ns - 1 - Q] === was.sL[ms - 1 - Q] && alike(np - 2 - Q, mp - 2 - Q)) Q++;
  if (pre === n && n === m && head && P === ns && ns === ms) return { e0: n, e1: n, k0: ns, k1: ns, first: false, last: false, grow: 0, pre, suf };
  // the metres where the sweep's values can differ: the entries changed, the steps after those alike from the start and
  // before those alike from the end (the last ones' where they lay, moved either way)
  let x0 = Infinity, x1 = -Infinity;
  if (pre < n - suf || pre < m - suf) (x0 = S[pre]), (x1 = S[n - suf]);
  for (let k = P; k < ns; k++) x0 = Math.min(x0, now.sc[k] - now.sL[k] / 2);
  for (let k = P; k < ms; k++) x0 = Math.min(x0, was.sc[k] - was.sL[k] / 2 + Math.min(0, shift));
  for (let k = 0; k < ns - Q; k++) x1 = Math.max(x1, now.sc[k] + now.sL[k] / 2);
  for (let k = 0; k < ms - Q; k++) x1 = Math.max(x1, was.sc[k] + was.sL[k] / 2 + Math.max(0, shift));
  if (!head) x0 = -Infinity;
  if (!tail) x1 = Infinity;
  // the entries from the one where x0 lies (those before it read only steps alike); the sweep from a quiet place before it:
  // the steps before k0 all ended over a metre before it (and past the path's first knot), the rest start over a metre after
  let e0 = Math.min(pre, n);
  if (x0 === -Infinity) e0 = 0;
  else while (e0 > 0 && S[e0] > x0) e0--;
  const pmh = new Float64Array(ns + 1), sml = new Float64Array(ns + 1);
  pmh[0] = -Infinity;
  for (let k = 0; k < ns; k++) pmh[k + 1] = Math.max(pmh[k], now.sc[k] + now.sL[k] / 2);
  sml[ns] = Infinity;
  for (let k = ns - 1; k >= 0; k--) sml[k] = Math.min(sml[k + 1], now.sc[k] - now.sL[k] / 2);
  let k0 = 0;
  for (let K = Math.min(P, ns); K > 0; K--) if (pmh[K] > 1 && pmh[K] + 1 < S[e0] && pmh[K] + 2 < sml[K]) {
    k0 = K;
    break;
  }
  // to the first node past a quiet place after x1 (the last bands' sweep was quiet there too, its steps before alike or
  // ended before x1), with the steps that start before that node or at it (one of no length there has passed when the
  // node's value is taken, `sweepKnots`)
  let e1 = n, k1 = ns, last = true;
  if (x1 < Infinity) for (let K = Math.max(0, ns - Q); K <= ns; K++) {
    const g = Math.max(x1, pmh[K] + 1);
    if (!(g + 1 < sml[K])) continue;
    let k = Math.max(e0, n - suf);
    while (k < n && S[k] <= g) k++;
    if (k >= n) break;
    e1 = k;
    last = false;
    k1 = K;
    while (k1 < ns && sml[k1] <= S[e1]) k1++;
    break;
  }
  return { e0, e1, k0, k1, first: k0 === 0, last, grow: n - m, pre, suf };
}

/** T-4153: per recipe of a line's last bands, its number now (-1: not one of its recipes now) */
function recipeMap(was: readonly OwnRecipe[], idx: ReadonlyMap<string, number>): Int32Array {
  const out = new Int32Array(was.length);
  for (let i = 0; i < was.length; i++) out[i] = idx.get(was[i].key) ?? -1;
  return out;
}

/** T-4153: entries [a, b) of a line's path, from its sweep: per entry its knots (its first node's value, the knots inside
 * it, its last node's value) onto `knots` (where they start onto `starts`), its Traffic record onto `recs` (where it starts
 * in `recAt`, its recipes' places onto `recPos`); `acc`: whether any knot is other than none (and per entry, `nzE`), and
 * the records */
function entryKnots(id: number, path: readonly number[], start: Float64Array, ends: ReadonlyMap<number, BandEdge>, pts: ReadonlyMap<number, Float32Array>, o: SweepOut, a: number, b: number, knots: Nums, starts: Nums, recs: Nums, recPos: Nums, recAt: Int32Array, acc: { nonzero: boolean; records: number; nzE: Uint8Array }): void {
  const { km, kv, kw, kq, ko, kn, kf, nv, nq, nw, no, nn } = o;
  let ki = 0, rec = false, nk = 0, k = a;
  const add = (f: number, v: number, w: number, list: readonly number[], o: number, c: number, vm: number) => {
    // (the one recipe when it is one alone, else -1: between two, for `bandPoints`' eases)
    knots.push(f);
    knots.push(v);
    knots.push(w);
    knots.push(c === 1 && list[o + 1] === 1 ? list[o] : -1);
    knots.push(vm);
    if (rec) {
      recs.push2(f, c);
      for (let x = o; x < o + 2 * c; x += 2) {
        recPos.push(recs.length);
        recs.push2(list[x], list[x + 1]);
      }
    }
    nk++;
    if (Math.abs(v) > 1e-6 || w > 0 || Math.abs(vm) > 1e-3) (acc.nonzero = true), (acc.nzE[k] = 1);
  };
  for (; k < b; k++) {
    starts.push(knots.length / KNOT);
    const m0 = start[k], m1 = start[k + 1], len = m1 - m0;
    const p = path[k], e = p >> 1;
    const ed = ends.get(e), P = pts.get(e), head = recs.length;
    rec = !!(ed && P && P.length >= 6);
    nk = 0;
    if (rec) {
      recAt[k] = head;
      recs.push2(id, e);
      recs.push2(p & 1, P![0]);
      recs.push2(P![1], 0);
    }
    add(0, nv[k], nq[k], kf, no[k], nn[k], nw[k]);
    while (ki < km.length && km[ki] <= m0 + 1e-3) ki++;
    for (; ki < km.length && km[ki] < m1 - 1e-3; ki++) add(len > 0 ? (km[ki] - m0) / len : 0, kv[ki], kq[ki], kf, ko[ki], kn[ki], kw[ki]);
    add(1, nv[k + 1], nq[k + 1], kf, no[k + 1], nn[k + 1], nw[k + 1]);
    if (rec) {
      recs.a[head + 5] = nk;
      acc.records++;
    }
  }
}

/** T-4153: a line's entries [a, b) as its last bands had them (`was`), as its entries from `to` on, their recipes numbered as
 * now (`map`; `same`: as they were, so its knots and records are copied a block each), onto the arrays `entryKnots`
 * writes; false when one of their recipes is not the line's now */
function copyEntries(was: LineMemo, a: number, b: number, map: Int32Array, same: boolean, to: number, knots: Nums, starts: Nums, recs: Nums, recPos: Nums, recAt: Int32Array, acc: { nonzero: boolean; records: number; nzE: Uint8Array }): boolean {
  if (a >= b) return true;
  const op = was.profile!, om = op[0], or = was.recs, oat = was.recAt!, onz = was.nzE!;
  if (same) {
    const k0 = op[1 + a], base = knots.length / KNOT;
    for (let j = a; j < b; j++) starts.push(base + op[1 + j] - k0);
    knots.block(op, 2 + om + k0 * KNOT, 2 + om + op[1 + b] * KNOT);
    for (let j = a; j < b; j++) if (onz[j]) (acc.nonzero = true), (acc.nzE[to + j - a] = 1);
    // (the records of entries [a, b) lie one after another: from the first's start to the next one's after b)
    let h0 = -1;
    for (let j = a; j < b; j++) {
      const h = oat[j];
      if (h < 0) continue;
      if (h0 < 0) h0 = h;
      recAt[to + j - a] = recs.length + h - h0;
      acc.records++;
    }
    if (h0 < 0) return true;
    let h1 = or.length;
    for (let j = b; j < oat.length; j++) if (oat[j] >= 0) { h1 = oat[j]; break; }
    const shift = recs.length - h0, P = was.recPos;
    let lo = 0, hi = P.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (P[mid] < h0) lo = mid + 1; else hi = mid; }
    for (let i = lo; i < P.length && P[i] < h1; i++) recPos.push(P[i] + shift);
    recs.block(or, h0, h1);
    return true;
  }
  for (let j = a; j < b; j++) {
    starts.push(knots.length / KNOT);
    for (let x = op[1 + j]; x < op[2 + j]; x++) {
      const i = 2 + om + x * KNOT, r = op[i + 3], v = op[i + 1], w = op[i + 2], vm = op[i + 4];
      const nr = r < 0 ? -1 : map[r];
      if (r >= 0 && nr < 0) return false;
      knots.push(op[i]);
      knots.push(v);
      knots.push(w);
      knots.push(nr);
      knots.push(vm);
      if (Math.abs(v) > 1e-6 || w > 0 || Math.abs(vm) > 1e-3) (acc.nonzero = true), (acc.nzE[to + j - a] = 1);
    }
    const h = oat[j];
    if (h < 0) continue;
    recAt[to + j - a] = recs.length;
    for (let i = h; i < h + 6; i++) recs.push(or[i]);
    let i = h + 6;
    for (let c = 0, nk = or[h + 5]; c < nk; c++) {
      const cnt = or[i + 1];
      recs.push2(or[i], cnt);
      i += 2;
      for (let y = 0; y < cnt; y++, i += 2) {
        const nr = map[or[i]];
        if (nr < 0) return false;
        recPos.push(recs.length);
        recs.push2(nr, or[i + 1]);
      }
    }
    acc.records++;
  }
  return true;
}

/** numbers per recipe in `LineBands.corridor`: ribbon, slot, sign (+1 a to b); the ribbon's middle
 * from the line's track is the strokes' metres (`Strokes.band`), the same in both views */
const RECIPE = 3;

/** Path entry `k`'s knots in profile `b`: the offset of its first and how many. */
export function bandKnots(b: BandProfile, k: number): [number, number] {
  const n = b[0], o = 2 + n;
  return [o + b[1 + k] * KNOT, b[2 + k] - b[1 + k]];
}

/** Path entry `k`'s band at `f` (0-1 along it, in its travel direction): [shift, 1 - width share,
 * metres to the ribbon's middle]. */
export function profileAt(b: BandProfile, k: number, f: number): [number, number, number] {
  const [o, c] = bandKnots(b, k);
  if (c <= 0) return [0, 0, 0];
  if (f <= b[o]) return [b[o + 1], b[o + 2], b[o + 4]];
  for (let i = 1; i < c; i++) {
    const x = o + i * KNOT;
    if (f <= b[x]) {
      const p = x - KNOT, d = b[x] - b[p], t = d > 0 ? (f - b[p]) / d : 1;
      return [b[p + 1] + (b[x + 1] - b[p + 1]) * t, b[p + 2] + (b[x + 2] - b[p + 2]) * t, b[p + 4] + (b[x + 4] - b[p + 4]) * t];
    }
  }
  const x = o + (c - 1) * KNOT;
  return [b[x + 1], b[x + 2], b[x + 4]];
}

/** `profileAt`'s shift and metres written to `out[o]` and `out[o + 1]` (turned round with `flip`), the same numbers
 * made without arrays (T-4153: a line's train samples read it once each) */
export function profileInto(b: BandProfile, k: number, f: number, out: Float32Array, o: number, flip: boolean): void {
  const n = b[0], k0 = 2 + n + b[1 + k] * KNOT, c = b[2 + k] - b[1 + k], s = flip ? -1 : 1;
  if (c <= 0) {
    out[o] = 0;
    out[o + 1] = 0;
    return;
  }
  if (f <= b[k0]) {
    out[o] = s * b[k0 + 1];
    out[o + 1] = s * b[k0 + 4];
    return;
  }
  for (let i = 1; i < c; i++) {
    const x = k0 + i * KNOT;
    if (f <= b[x]) {
      const p = x - KNOT, d = b[x] - b[p], t = d > 0 ? (f - b[p]) / d : 1;
      out[o] = s * (b[p + 1] + (b[x + 1] - b[p + 1]) * t);
      out[o + 1] = s * (b[p + 4] + (b[x + 4] - b[p + 4]) * t);
      return;
    }
  }
  const x = k0 + (c - 1) * KNOT;
  out[o] = s * b[x + 1];
  out[o + 1] = s * b[x + 4];
}

/** `TrafficBands.pick` holds a shift as a share of this many times BAND_RANGE line widths, since a
 * trunk of busy lines is wider than BAND_RANGE */
export const TRAFFIC_PICK_DIV = 4;

export interface TrafficBands {
  /** per line stroke segment, 2: its band's shift at its start and end, line widths left of its direction */
  seg: Float32Array;
  /** per train sample, 2: its line's band's shift there, line widths left of run 0, and its metres (as `sampleBand`) */
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
  if (!table || table.length < 3 || !L.band || L.band.length < n * BAND_STRIDE) return null;
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
    return table[x + 2] * (W / 2 - (v[slot] + v[slot + 1]) / 2);
  };
  const out = new Float32Array(n * 2), pick = new Int16Array(n * BAND_STRIDE), LB = L.band;
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
      pick[i * BAND_STRIDE] = Math.round(s0 * P);
      pick[i * BAND_STRIDE + 1] = Math.round(s1 * P);
      pick[i * BAND_STRIDE + 4] = LB[i * BAND_STRIDE + 4];
      pick[i * BAND_STRIDE + 5] = LB[i * BAND_STRIDE + 5];
      if (s0 || s1) any = true;
    }
  }
  if (!any) return null;
  // a train sample's run is its stroke's (the first segment of its path entry); where it is along
  // the run, by walking the run's segments forwards with the line's samples
  const ns = samples.length / 2, sample = new Float32Array(ns * 2), MU = BAND_M_RANGE / 32767;
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
      sample[q * 2] = out[s * 2] + (out[s * 2 + 1] - out[s * 2]) * u;
      sample[q * 2 + 1] = (LB[s * BAND_STRIDE + 4] + (LB[s * BAND_STRIDE + 5] - LB[s * BAND_STRIDE + 4]) * u) * MU;
      break;
    }
  }
  return { seg: out, sample, pick };
}
