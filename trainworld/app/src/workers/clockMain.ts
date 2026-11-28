// The state built for the main thread, and sending it (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { anchor, anchorShift, EARTH_C, toMerc01, type Anchor } from "../map/geo";
import { PERIODS } from "../game/periods";
import { DAY_KINDS } from "../game/days";
import { runLag } from "../game/moneyTiming";
import { networkFromState, networkSteps, newNetCache, runsTrains, shiftNetCache, type NetworkBuffers } from "../map/network";
import { terminalEnds } from "../map/terminalBays";
import { packCross } from "../map/markData";
import { alignList, StateRows, upgradesOf, type RowEnv } from "./stateRows";
import { PlatformLines } from "./platformLines";
import type { ClockEdge, ClockLine, ClockNode, ClockState } from "./protocol";
import { publishSlots, saves } from "./slots";
import { META_PER_LINE, profileOf } from "../map/profiles";
import { buildSnapshot, busRow, complexInputs, complexOfNodes, forgetKeeps, groundK, loadsLater, namedKeeps, putXferSet, sendTrips, shownBuildings, snapTimer } from "./clockDemand";
import { pinFareRef } from "./clockFares";
import { moneyState } from "./clockMoney";
import { keep } from "./clockPanic";
import { patchOutside } from "./clockPatch";
import { checkRows, checkShift, DiagramIn, diagramOn, keepPosted, netForPost, NO_PHASES, patchMode, postDiagram, posted, postState, postWhole, SAME_FIELDS, sameBytes } from "./clockPost";
import { anchorSpec, api, applyRunFactors, clk, diagramCities, frame, frameSpec, freezeCheckpoint, hiddenLines, home, hourShape, lineShapes, midLoad, newLineHeadway, opened, packLngLat, packLocal, partCost, partHour, pinDayShift, placeLineCountries, putClk, putIncomePerHour, putLineIncome, putLinePoints, putLineRides, putMidLoad, putPartCost, putPartFare, putPartHour, putPiece, putRidesPerHour, putSettledT, SAMPLE_STEP, send, startZone, syncFrame, syncHolidays, timing, wallNow } from "./clockState";

/** T-2679: the first built station's lng and lat (lowest node id), as the last state was built. */
export let firstBuiltAt: [number, number] | null = null;

/** T-867: hours a day in each demand period, which weigh each line's trains a day for its upkeep share
 * (T-3542: `TrackApi.line_upkeep_periods`) */
const PERIOD_HOURS = new Float64Array(PERIODS.map((p) => p.to - p.from));

export let version = 0;

export let moneyLag = 0;

/** each line's first stop, lng and lat (null with none), as the last state was built */
export let lineFirstStops = new Map<number, [number, number] | null>();
/** T-2980: the places at each line's ends, as the last state was built */
export let termEnds: ReturnType<typeof terminalEnds> = new Map();

/** Loads so far (T-715): a "clock" message the main thread sent before it saw the latest still
 * carries the previous game's time, and settling to it charged the gap's running costs and then
 * held the money until the clock caught up. T-3734: the stamp of the game in hand, on the page's `gameStamp`. */
export let game = 0;

// ---------------------------------------------------------------- state for the main thread

export function hex(c: number) {
  return "#" + c.toString(16).padStart(6, "0");
}

/** T-4291: a map that notes, for each reader it hands out, the keys set or deleted since that reader last emptied its
 * record (`all`: cleared since) */
class NotedMap<K, V> extends Map<K, V> {
  private readers?: { keys: Set<K>; all: boolean }[];
  reader(): { keys: Set<K>; all: boolean } {
    const r = { keys: new Set<K>(), all: true };
    (this.readers ??= []).push(r);
    return r;
  }
  set(k: K, v: V): this {
    if (this.readers) for (const r of this.readers) r.keys.add(k);
    return super.set(k, v);
  }
  delete(k: K): boolean {
    if (this.readers) for (const r of this.readers) r.keys.add(k);
    return super.delete(k);
  }
  clear() {
    if (this.readers) for (const r of this.readers) { r.all = true; r.keys.clear(); }
    super.clear();
  }
}

// What the renderer draws, per edge and per line, in local units, kept between edits: an edit
// re-renders only the edges and lines it dirtied (T-063). Cleared when a game is loaded.
export const edgeDrawn = new NotedMap<number, Float32Array>();
const edgeSpeedDrawn = new Map<number, Float32Array>();
export const lineDrawn = new Map<number, Float32Array>();
/** T-1170: the same points in mercator 0..1 (float64; z or nothing after x, y), which no anchor move
 * changes, so a move re-expresses them without asking the track model */
export const edgeMerc = new Map<number, Float64Array>();
const lineMerc = new Map<number, Float64Array>();
/** T-4075: each line's run 0 bends (`bendsOf`), made with its `lineMerc` */
const lineBends = new Map<number, Float32Array>();
/** How far (ground m) a train may be drawn off its track between two samples before the interval is cut (T-4075) */
const BEND_TOL_M = 0.25;
/** `run_bends`'s rows (interval, inner points, then the points in track units) as the same rows with each point
 * given as its offset from the straight between the interval's samples (`merc`, a run's `drawnLine` points),
 * local units, which no anchor move changes */
function bendsOf(raw: ArrayLike<number>, merc: Float64Array): Float32Array {
  const out: number[] = [];
  for (let i = 0; i < raw.length; ) {
    const k = raw[i], m = raw[i + 1];
    const ax = merc[k * 2], ay = merc[k * 2 + 1], bx = merc[k * 2 + 2], by = merc[k * 2 + 3];
    out.push(k, m);
    i += 2;
    for (let j = 1; j <= m; j++, i += 2) {
      const [x, y] = toMerc01(...packLngLat(raw[i], raw[i + 1])), f = j / (m + 1);
      out.push((x - ax - (bx - ax) * f) * EARTH_C, (y - ay - (by - ay) * f) * EARTH_C);
    }
  }
  return new Float32Array(out);
}
/** The renderer's bend tables (T-4075) from each run's first sample and bends: per sample its interval's first point
 * in `bends` times 32 plus its inner points' count (0 none), and the points' offsets (x, y each). */
function bendTables(nSamples: number, runs: [number, Float32Array][]): { sampleBend: Float32Array; bends: Float32Array } {
  let total = 0;
  for (const [, b] of runs) for (let i = 0; i < b.length; i += 2 + 2 * b[i + 1]) total += b[i + 1];
  const sampleBend = new Float32Array(nSamples), bends = new Float32Array(total * 2);
  let at = 0;
  for (const [first, b] of runs) for (let i = 0; i < b.length; i += 2 + 2 * b[i + 1]) {
    const m = b[i + 1];
    // (a float holds the packed index exactly below 2^24: past that, an interval is drawn straight)
    if (at * 32 + m >= 1 << 24) break;
    sampleBend[first + b[i]] = at * 32 + m;
    bends.set(b.subarray(i + 2, i + 2 + 2 * m), at * 2);
    at += m;
  }
  return { sampleBend, bends: at * 2 === bends.length ? bends : bends.slice(0, at * 2) };
}
/** mercator points (`stride` numbers each, x and y first) in the anchor's local units */
export function localOf(m: Float64Array, stride: number): Float32Array {
  const a = anchor(), ax = a.merc[0], ay = a.merc[1], d = new Float32Array(m.length);
  for (let k = 0; k < m.length; k += stride) {
    d[k] = (m[k] - ax) * EARTH_C;
    d[k + 1] = (m[k + 1] - ay) * EARTH_C;
    for (let j = 2; j < stride; j++) d[k + j] = m[k + j];
  }
  return d;
}
export function forget(all = false) {
  if (all) {
    edgeDrawn.clear();
    edgeSpeedDrawn.clear();
    lineDrawn.clear();
    edgeMerc.clear();
    lineMerc.clear();
    lineBends.clear();
    backDrawn.clear();
    pathsMoved.clear();
    dropped.clear();
    droppedBack.clear();
    pathsAllMoved = true;
    // (the model's list of lines that moved starts again with this: every line is sampled anew anyway)
    api?.take_moved_lines();
    api?.take_moved_spans();
    drift.clear();
    shiftFrom = null;
    stateRows.reset();
    platformLines.clear();
    postWhole();
    return;
  }
  const a = api!, edges = a.dirty_edges();
  for (const e of edges) { edgeDrawn.delete(e); edgeSpeedDrawn.delete(e); edgeMerc.delete(e); platformLines.forgetEdge(e); }
  // T-4149: a line's runs are sampled again only where they lie anywhere new on the track; the other lines whose profiles
  // were worked out again (a settle's timing: most of a big network's lines on shared track) or that were rebuilt onto the
  // same track run where they ran. A bus or ferry line's runs follow its route and keep the older rule.
  const moved = a.take_moved_lines(), spans = a.take_moved_spans();
  if (moved[0] === 0xffffffff) {
    for (const l of lineMerc.keys()) keepRun0(l);
    for (const l of backDrawn.keys()) keepBack(l);
    lineDrawn.clear(); lineMerc.clear(); lineBends.clear(); backDrawn.clear(); drift.clear();
  } else {
    // T-4281: a run that moved only in a stretch (`take_moved_spans`) is sampled again there alone; T-4149: a way back
    // too (`note_moved` looks at both runs: their entries and the touched edges under each)
    const near = new Map<number, ArrayLike<number>>();
    for (let k = 0; k + 5 < spans.length; k += 6) near.set(runKey(spans[k], spans[k + 1]), spans.subarray(k, k + 6));
    for (const l of moved) {
      if (!spliceRun0(a, l, near.get(runKey(l, 0)))) dropRun0(l);
      if (!spliceBack(a, l, near.get(runKey(l, 1)))) dropBack(l);
    }
  }
  for (const l of a.dirty_lines()) if (a.is_bus(l) || a.is_ferry(l)) dropDrawnLine(l);
}

/** T-4280: the run samples and bends of lines dropped since the last network was posted, as they were drawn: a line sampled
 * again is named in `pathsMoved` only where its new ones differ (bus lines are dropped on most edits, onto the same route) */
const dropped = new Map<number, { merc: Float64Array; bends: Float32Array }>();
const droppedBack = new Map<number, { merc: Float64Array; bends: Float32Array }>();
function keepRun0(l: number) {
  const m = lineMerc.get(l);
  if (m) dropped.set(l, { merc: m, bends: lineBends.get(l) ?? NO_PHASES });
}
function keepBack(l: number) {
  const b = backDrawn.get(l);
  if (b) droppedBack.set(l, { merc: b.merc, bends: b.bends });
}
/** whether a run sampled again (`merc`, `bends`) differs from its samples before (`was`, none: a run new to the drawing) */
function resampled(was: { merc: Float64Array; bends: Float32Array } | undefined, merc: Float64Array, bends: Float32Array): boolean {
  if (!was || was.merc.length !== merc.length || was.bends.length !== bends.length) return true;
  for (let k = 0; k < merc.length; k++) if (was.merc[k] !== merc[k]) return true;
  for (let k = 0; k < bends.length; k++) if (was.bends[k] !== bends[k]) return true;
  return false;
}
function dropDrawnLine(l: number) {
  dropRun0(l); dropBack(l);
}
function dropRun0(l: number) {
  keepRun0(l);
  lineDrawn.delete(l); lineMerc.delete(l); lineBends.delete(l); drift.delete(runKey(l, 0));
}
function dropBack(l: number) {
  keepBack(l);
  backDrawn.delete(l); drift.delete(runKey(l, 1));
}
const runKey = (l: number, run: number) => l * 2 + run;
/** T-4281: per run (`runKey`), how far (ground m) its kept samples may lie along the track from where a whole resample
 * would put them (`m`): the summed length changes of the cuts since its last whole sampling (a station splitting an edge
 * moves the pieces' lengths by a few mm). T-4149: samples before `from` are a whole resample's, bit for bit (before the
 * first cut that moved the rest) */
const drift = new Map<number, { m: number; from: number }>();
/** T-4281: the most `drift` a line's samples may carry before they are taken whole again (a train is drawn at most this
 * far along its track from its place; `BEND_TOL_M` is a train's leeway across it) */
const DRIFT_MAX_M = 0.05;
/** T-4281: line `l`'s kept run 0 samples and bends with the stretch of `span` sampled again (`splice`); false (nothing
 * changed) where it must be sampled whole */
function spliceRun0(a: TrackApi, l: number, span: ArrayLike<number> | undefined): boolean {
  const m = lineMerc.get(l), b = lineBends.get(l);
  const s = m && b && splice(a, l, 0, m, b, span);
  if (!s) return false;
  lineMerc.set(l, s.merc);
  lineBends.set(l, s.bends);
  lineDrawn.delete(l);
  if (s.changed) pathsMoved.add(l);
  return true;
}
/** T-4149: the same for a line's own way back (`drawnBack`'s), kept under its path now */
function spliceBack(a: TrackApi, l: number, span: ArrayLike<number> | undefined): boolean {
  const c = backDrawn.get(l);
  const s = c && splice(a, l, 1, c.merc, c.bends, span);
  if (!s) return false;
  backDrawn.set(l, { key: a.line_back_path(l).join(), merc: s.merc, bends: s.bends, at: null, d: null });
  if (s.changed) pathsMoved.add(l);
  return true;
}
/** T-4281: run `run` of line `l`, kept as `m` and `b` (mercator samples, bends), with the stretch of `span`
 * (`[line, run, lo, hi, drift, len]`) sampled again as a whole resample samples it there, and its drift noted; null where
 * it must be sampled whole: no span, the drift past `DRIFT_MAX_M` or the count of samples changed */
function splice(a: TrackApi, l: number, run: 0 | 1, m: Float64Array, b: Float32Array, span: ArrayLike<number> | undefined): { merc: Float64Array; bends: Float32Array; changed: boolean } | null {
  // (`?patch=walk` samples every moved run whole, as before, to time against)
  if (!span || patchMode === "walk") return null;
  const [, , lo, hi, moved, len] = Array.from(span);
  const was = drift.get(runKey(l, run));
  const d = (was?.m ?? 0) + moved, n = Math.ceil(len / SAMPLE_STEP);
  if (!(d <= DRIFT_MAX_M) || m.length !== (n + 1) * 2) return null;
  const k0 = Math.max(0, Math.floor(lo / SAMPLE_STEP) - 1), k1 = Math.min(n, Math.ceil(hi / SAMPLE_STEP) + 1);
  const p = a.run_polyline_range(l, run, SAMPLE_STEP, k0, k1);
  if (p.length !== (k1 - k0 + 1) * 2) return null;
  const merc = m.slice();
  for (let k = 0; k < p.length; k += 2) merc.set(toMerc01(...packLngLat(p[k], p[k + 1])), k0 * 2 + k);
  // the bends of every interval with a sample sampled again
  const b0 = Math.max(0, k0 - 1), b1 = Math.min(n, k1 + 1);
  const fresh = bendsOf(a.run_bends_range(l, run, SAMPLE_STEP, BEND_TOL_M, b0, b1), merc);
  const out: number[] = [];
  let at = 0, cut = 0;
  for (; at < b.length && b[at] < b0; at += 2 + 2 * b[at + 1]) cut = at + 2 + 2 * b[at + 1];
  let tail = at;
  for (; tail < b.length && b[tail] < b1; tail += 2 + 2 * b[tail + 1]);
  for (let k = 0; k < cut; k++) out.push(b[k]);
  for (let k = 0; k < fresh.length; k++) out.push(fresh[k]);
  for (let k = tail; k < b.length; k++) out.push(b[k]);
  const bends = new Float32Array(out);
  let changed = bends.length !== b.length;
  for (let k = k0 * 2; !changed && k <= k1 * 2 + 1; k++) changed = merc[k] !== m[k];
  for (let k = cut; !changed && k < cut + fresh.length; k++) changed = bends[k] !== b[k];
  // (past the stretch the samples moved along by this cut's length change)
  if (d > 0) drift.set(runKey(l, run), { m: d, from: moved > 0 ? Math.min(was?.from ?? n + 1, k1 + 1) : was?.from ?? n + 1 });
  return { merc, bends, changed };
}
/** T-4149: the lines whose drawn runs moved since the last network was posted (`ClockState.touched.paths`), or all of them;
 * T-4280: a line is named where it was sampled again into other samples or bends than before (`resampled`) */
export const pathsMoved = new Set<number>();
export let pathsAllMoved = false;
/** T-4149: every drawn position moved since the last network was posted (the view's anchor): `touched.all` */
export let drawnAllMoved = false;
export function pathsPosted() { pathsMoved.clear(); pathsAllMoved = false; drawnAllMoved = false; dropped.clear(); droppedBack.clear(); }
/** The view's anchor moved (T-307): every drawn position again, relative to the new anchor */
export function forgetDrawn() { edgeDrawn.clear(); lineDrawn.clear(); drawnAllMoved = true; }

/** `?patch=check` (T-4149): every line's kept run samples and bends against the model's now. A line whose runs moved and
 * whose samples were kept anyway draws its trains off its track. */
function checkSamples(a: TrackApi, lines: readonly ClockLine[]) {
  const bad: number[] = [];
  const same = (x: ArrayLike<number> | undefined, y: ArrayLike<number>) => !!x && x.length === y.length && Array.prototype.every.call(y, (v: number, k: number) => Object.is(v, x[k]));
  const fresh = (id: number, run: number) => {
    const p = a.run_polyline(id, run, SAMPLE_STEP), m = new Float64Array(p.length);
    for (let k = 0; k < p.length; k += 2) m.set(toMerc01(...packLngLat(p[k], p[k + 1])), k);
    return { m, bends: bendsOf(a.run_bends(id, run, SAMPLE_STEP, BEND_TOL_M), m) };
  };
  // T-4281: samples kept past a stretch sampled again lie within the run's drift (ground m) of a whole resample; T-4149:
  // those before `from`, and the bends between them, are a whole resample's bit for bit
  const near = (x: Float64Array, y: Float64Array, tol: number, from: number) => x.length === y.length && x.every((v, k) => {
    if (k % 2) return true;
    if (k < from * 2) return Object.is(v, y[k]) && Object.is(x[k + 1], y[k + 1]);
    const ground = Math.hypot(v - y[k], x[k + 1] - y[k + 1]) * EARTH_C / Math.cosh(Math.PI * (1 - 2 * y[k + 1]));
    return ground <= tol + 1e-3;
  });
  const bendsBefore = (b: Float32Array, from: number) => {
    let k = 0;
    while (k < b.length && b[k] + 1 < from) k += 2 + 2 * b[k + 1];
    return b.subarray(0, k);
  };
  const kept = (id: number, run: number, m: Float64Array, b: Float32Array | undefined) => {
    const f = fresh(id, run), d = drift.get(runKey(id, run));
    if (!d) return same(m, f.m) && same(b, f.bends);
    return !!b && near(m, f.m, d.m, d.from) && same(bendsBefore(b, d.from), bendsBefore(f.bends, d.from));
  };
  for (const l of lines) {
    if (!l.ok) continue;
    const m = lineMerc.get(l.id), b = backDrawn.get(l.id);
    if (m && !kept(l.id, 0, m, lineBends.get(l.id))) bad.push(l.id);
    if (b && l.back?.length && !kept(l.id, 1, b.merc, b.bends)) bad.push(l.id);
  }
  if (bad.length) console.error(`T-4149 check: the kept run samples of lines ${bad.slice(0, 8).join(", ")} differ from the model's`);
}

/** The track model's mark rows (`crossings_info`, `preview_marks`) with x, y moved to local units. */
function markPoints(rows: Float64Array, stride: number): Float32Array {
  const out = new Float32Array(rows.length);
  for (let k = 0; k < rows.length; k += stride) {
    const [x, y] = packLocal(rows[k], rows[k + 1]);
    out.set([x, y], k);
    for (let j = 2; j < stride; j++) out[k + j] = rows[k + j];
  }
  return out;
}

/** T-3535: `preview_marks` (x, y, kind, tram tracks) as the preview message's marks: x, y in local units, then 1
 * for a junction, or for a crossing its tram tracks packed as a placed crossing's level carries them (`packCross`
 * at level 0), so the ghost draws the mark the placed track will. */
export function previewMarkRows(rows: Float64Array): Float32Array {
  const m = markPoints(rows, 4), out = new Float32Array((m.length / 4) * 3);
  for (let k = 0, j = 0; k + 3 < m.length; k += 4, j += 3) out.set([m[k], m[k + 1], m[k + 2] === 1 ? 1 : packCross(0, m[k + 3])], j);
  return out;
}

/** T-1859: each station's platform line, for the complexes */
const platformLines = new PlatformLines();
export const platformLineOf = (a: TrackApi, id: number, x: number, y: number, platform: number) => platformLines.of(a, id, x, y, platform, groundK(y));
export const rowEnv: RowEnv = { lngLat: packLngLat, periodHours: PERIOD_HOURS, busRow, platformLine: platformLineOf, complexInputs,
  platformLineNow: (a, id, x, y, platform, changed) => platformLines.peekOf(a, id, x, y, platform, groundK(y), changed) };
/** the rows of the last states, read again only where the track model says something changed (T-1382);
 * `?patch=0` reads them whole */
export const stateRows = new StateRows();

/** an edge's points in local units (x, y, z each), made and kept if not yet */
export function drawnEdge(a: TrackApi, id: number): Float32Array {
  let d = edgeDrawn.get(id);
  if (!d) {
    // T-1170: at a new anchor the kept mercator points move over; otherwise the track model's
    let m = edgeMerc.get(id);
    if (!m) {
      const r = a.edge_render(id);
      m = new Float64Array(r.length);
      for (let k = 0; k < r.length; k += 3) {
        const [mx, my] = toMerc01(...packLngLat(r[k], r[k + 1]));
        m[k] = mx;
        m[k + 1] = my;
        m[k + 2] = r[k + 2];
      }
      edgeMerc.set(id, m);
    }
    d = localOf(m, 3);
    edgeDrawn.set(id, d);
    // the speeds are in no local unit: kept over an anchor move, so their pieces are too
    if (!edgeSpeedDrawn.has(id)) edgeSpeedDrawn.set(id, a.edge_render_speeds(id));
  }
  return d;
}
/** T-4148: an edge's points in local units, read from the track model again (the edit's word is built before the dirty
 * edges' kept points are dropped) */
export function drawnEdgeNow(a: TrackApi, id: number): Float32Array {
  edgeDrawn.delete(id);
  edgeMerc.delete(id);
  return drawnEdge(a, id);
}
/** a line's first run, sampled, in local units (x, y each), made and kept if not yet */
export function drawnLine(a: TrackApi, id: number): Float32Array {
  let d = lineDrawn.get(id);
  if (!d) {
    let m = lineMerc.get(id);
    if (!m) {
      const p = a.run_polyline(id, 0, SAMPLE_STEP);
      m = new Float64Array(p.length);
      for (let k = 0; k < p.length; k += 2) m.set(toMerc01(...packLngLat(p[k], p[k + 1])), k);
      lineMerc.set(id, m);
      const bends = bendsOf(a.run_bends(id, 0, SAMPLE_STEP, BEND_TOL_M), m);
      lineBends.set(id, bends);
      drift.delete(runKey(id, 0));
      if (resampled(dropped.get(id), m, bends)) pathsMoved.add(id);
      dropped.delete(id);
    }
    d = localOf(m, 2);
    lineDrawn.set(id, d);
  }
  return d;
}
/** T-3250: a line's return run, sampled as `drawnLine`, where it takes its own way back (`back`, its
 * path; kept while that is unchanged) */
const backDrawn = new Map<number, { key: string; merc: Float64Array; bends: Float32Array; at: Anchor | null; d: Float32Array | null }>();
function drawnBack(a: TrackApi, id: number, back: readonly number[]): Float32Array {
  const key = back.join();
  let c = backDrawn.get(id);
  if (!c || c.key !== key) {
    const p = a.run_polyline(id, 1, SAMPLE_STEP);
    const m = new Float64Array(p.length);
    for (let k = 0; k < p.length; k += 2) m.set(toMerc01(...packLngLat(p[k], p[k + 1])), k);
    const was = c ?? droppedBack.get(id);
    droppedBack.delete(id);
    backDrawn.set(id, (c = { key, merc: m, bends: bendsOf(a.run_bends(id, 1, SAMPLE_STEP, BEND_TOL_M), m), at: null, d: null }));
    drift.delete(runKey(id, 1));
    if (resampled(was, m, c.bends)) pathsMoved.add(id);
  }
  if (!c.d || c.at !== anchor()) {
    c.d = localOf(c.merc, 2);
    c.at = anchor();
  }
  return c.d;
}

/** T-4291: each edge's points and speeds (`drawnEdge`) beside the edges' list, as the last full state's had them: only those of
 * rows that changed, came or went since, and of edges whose points were dropped or made again since (`edgeDrawn`'s own
 * record), are looked up again */
let listed: { read: number; edges: readonly ClockEdge[]; drawn: Float32Array[]; speeds: (Float32Array | undefined)[] } | null = null;
const listedFresh = edgeDrawn.reader();
function drawnLists(a: TrackApi, edges: readonly ClockEdge[]): { drawn: Float32Array[]; speeds: (Float32Array | undefined)[] } {
  const read = stateRows.readNumber, rows = listed && !listedFresh.all && patchMode !== "walk" ? stateRows.changedSince(listed.read, 0) : null;
  let out: { drawn: Float32Array[]; speeds: (Float32Array | undefined)[] };
  if (!listed || !rows) out = { drawn: edges.map((e) => drawnEdge(a, e.id)), speeds: [] };
  else {
    const ids = new Set([...rows, ...listedFresh.keys]);
    out = ids.size || edges !== listed.edges ? { drawn: alignList(listed.drawn, listed.edges, edges, ids, (id) => drawnEdge(a, id)), speeds: [] } : listed;
    if (out !== listed) out.speeds = alignList(listed.speeds, listed.edges, edges, ids, (id) => edgeSpeedDrawn.get(id));
  }
  if (!listed || !rows) out.speeds = edges.map((e) => edgeSpeedDrawn.get(e.id));
  // (after the points made just now, which note themselves)
  listedFresh.all = false;
  listedFresh.keys.clear();
  listed = { read, edges, ...out };
  if (patchMode === "check" && edges.some((e, k) => out.drawn[k] !== edgeDrawn.get(e.id) || out.speeds[k] !== edgeSpeedDrawn.get(e.id))) console.error("T-4291 check: the edges' points kept beside their list differ from those drawn");
  return out;
}

/** T-4113: the view's bounds, west, south, east, north, from the page's "view"; a rows state reads the regions there
 * first */
let view: [number, number, number, number] | undefined;
export function putView(w: number, s: number, e: number, n: number) { view = [w, s, e, n]; }
function viewRect(): [number, number, number, number] | undefined {
  if (!view) return undefined;
  const [x0, y0] = frame.fromLonLat(view[0], view[1]), [x1, y1] = frame.fromLonLat(view[2], view[3]);
  return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
}

/** `rowsOnly`: for a rows state (`sendRows`), which carries no drawn pieces: none are made, and the rows read the
 * regions in view and the edit's own first when many are dirty (T-4113) */
function buildState(rowsOnly = false): ClockState {
  const tq0 = performance.now();
  const a = api!;
  // T-4272: the runs and rows of what answered edits still owe are their old ones until the follow-up's first step
  if (patchMode === "check" && a.owes()) console.error("T-4272 check: a state was built while answered edits still owe their services");
  if (patchMode === "0") stateRows.reset();
  stateRows.bind(a, frame);
  // T-4113: `?patch=walk` makes every record's own calls on every read, as before regions (for timing against)
  StateRows.bounded = patchMode !== "walk";
  StateRows.alternate = patchMode === "check";
  // (with `?patch=walk` a rows state is built as a full one, as before)
  rowsOnly &&= StateRows.bounded;
  // T-4534: ground that re-priced patched track with no patch since (a road tile mid-burst) is patched again first;
  // T-4363: any laid since the last state posted, a read dropped since included
  patchOutside(a, true);
  stateRows.begin(a, rowsOnly ? { rect: viewRect() } : undefined);
  if (patchMode === "check") {
    const [n, e] = stateRows.regions.counts(), ni = a.nodes_info().length / 10, ei = a.edges_info().length / 10;
    if (n !== ni || e !== ei) console.error(`T-4299 check: the regions hold ${n} nodes and ${e} edges, the model ${ni} and ${ei}`);
  }
  const edges = stateRows.edges(a);
  const { drawn, speeds: speedParts } = rowsOnly ? { drawn: [], speeds: [] } : drawnLists(a, edges);
  const tq1 = performance.now(); timing.sEdges = tq1 - tq0;
  platformLines.sync(edges, stateRows.changes(0), (id) => stateRows.edgeNow(id));
  if (patchMode === "check") { const bad = platformLines.differs(edges); if (bad.length) console.error(`T-4291 check: the platforms' ${bad.join(", ")} laid from changes differ from a whole sync`); }
  let nodes = stateRows.nodes(a, rowEnv);
  // T-3759: an older save's typed stations that showed their own names keep them, unbound (once a load,
  // not on undo)
  if (a.take_old_names()) {
    const apart = stateRows.oldApart();
    if (apart.length && a.adopt_old_names(new Uint32Array(apart)) === 0) {
      forget();
      nodes = stateRows.nodes(a, rowEnv);
    }
  }
  const stationCount = nodes.length;
  const tq2 = performance.now(); timing.sNodes = tq2 - tq1;
  const lines = stateRows.lines(a, rowEnv, nodes);
  platformLines.setLines(lines, stateRows.changes(2), (id) => stateRows.lineNow(id));
  if (patchMode === "check") { const bad = platformLines.differs(edges, lines); if (bad.length) console.error(`T-4291 check: the platforms' ${bad.join(", ")} from changes differ from whole ones`); }
  const tq3 = performance.now(); timing.sLines = tq3 - tq2;
  const w = stateRows.work;
  timing.sFold = w.foldMs;
  timing.nEdges = w.edges; timing.nNodes = w.nodes; timing.nLines = w.lines; timing.nBuses = w.buses; timing.nRegions = w.leaves; timing.nLeft = w.left;
  if (patchMode === "check") checkRows({ edges, nodes, lines }, a, upgradesOf(a));
  // a bus line's stops that are no station were added to the nodes by its row
  const busNodes = new Map(nodes.slice(stationCount).map((n) => [n.id, n]));
  lineFirstStops = new Map(lines.map((l) => {
    const n = stateRows.nodeRow(l.stops[0]) ?? busNodes.get(l.stops[0]);
    return [l.id, n ? [n.lng, n.lat] : null];
  }));
  const tq3a = performance.now();
  placeLineCountries();
  // T-4291: from the rows by id, not a pass over every edge and node
  termEnds = terminalEnds(edges, nodes, lines, { edgeOf: { get: (id) => stateRows.edgeNow(id) }, nodeOf: { get: (id) => stateRows.nodeNow(id) }, endCount: { get: (n) => platformLines.endsAt(n) } });
  if (patchMode === "check" && JSON.stringify([...termEnds]) !== JSON.stringify([...terminalEnds(edges, nodes, lines)])) console.error("T-4291 check: the line ends read by id differ from those of every row");
  timing.sEnds = performance.now() - tq3a;
  const tq3b = performance.now();
  let first: ClockNode | null = null;
  for (let k = 0; k < stationCount; k++) if (nodes[k].stationBuilt && (!first || nodes[k].id < first.id)) first = nodes[k];
  firstBuiltAt = first ? [first.lng, first.lat] : null;
  pinFareRef();
  pinDayShift();
  syncHolidays();
  applyRunFactors();
  const tq3c = performance.now(); timing.sSync = tq3c - tq3b;
  // Keyframes: phase tables, profile meta, and run 0 of every line resampled; a line whose return run
  // takes its own way back (T-3094) has that run resampled too, in the table's second half (T-3250,
  // network.ts `backSlot`)
  const meta = a.profile_meta();
  const nLines = meta.length / META_PER_LINE;
  const backs = lines.filter((l) => l.ok && l.back?.length && l.id < nLines);
  const lineTable = new Float32Array(Math.max(1, nLines) * (backs.length ? 8 : 4));
  const runs: Float32Array[] = [];
  // (T-4075: each run's first sample and its bends)
  const runBends: [number, Float32Array][] = [];
  let nSamples = 0;
  if (!rowsOnly) for (const l of lines) {
    if (!l.ok) continue;
    const d = drawnLine(a, l.id);
    runs.push(d);
    runBends.push([nSamples, lineBends.get(l.id) ?? NO_PHASES]);
    lineTable.set([nSamples, d.length / 2, l.lengthM, SAMPLE_STEP], l.id * 4);
    nSamples += d.length / 2;
  }
  if (!rowsOnly) for (const l of backs) {
    const d = drawnBack(a, l.id, l.back!);
    if (d.length < 4) continue;
    runs.push(d);
    runBends.push([nSamples, backDrawn.get(l.id)!.bends]);
    lineTable.set([nSamples, d.length / 2, meta[profileOf(l.id, 1, 0) * 4 + 3] || l.lengthM, SAMPLE_STEP], (nLines + l.id) * 4);
    nSamples += d.length / 2;
  }
  if (patchMode === "check" && !rowsOnly) checkSamples(a, lines);
  const samples = new Float32Array(nSamples * 2);
  runs.reduce((o, d) => (samples.set(d, o), o + d.length), 0);
  const { sampleBend, bends } = bendTables(nSamples, runBends);
  const tq3d = performance.now(); timing.sSamples = tq3d - tq3c;
  // T-3542: per period, read once per slot serving periods
  const slots = a.period_slots();
  const bySlot = new Map<number, ClockState["markers"][number]>();
  const markers = PERIODS.map((_, p) => {
    const had = bySlot.get(slots[p]);
    if (had) return had;
    const m = a.markers_in(p);
    const out: ClockState["markers"][number] = [];
    for (let k = 0; k < m.length; k += 7) {
      const [x, y] = packLocal(m[k], m[k + 1]);
      out.push({ x, y, rho: m[k + 2], delayS: m[k + 3], kind: m[k + 4], tph: m[k + 5], held: m[k + 6] > 0 });
    }
    bySlot.set(slots[p], out);
    return out;
  });
  const conflicts = a.guard_conflicts();
  const singles = a.guard_singles();
  // T-3062: a flat crossing's level carries its tram tracks, for the mark's drawing (markData.ts)
  const info = markPoints(a.crossings_info(), 6), crossings = new Float32Array((info.length / 6) * 5);
  for (let k = 0, j = 0; k + 5 < info.length; k += 6, j += 5) {
    crossings.set(info.subarray(k, k + 5), j);
    if (info[k + 2] === 0) crossings[j + 3] = packCross(info[k + 3], info[k + 5]);
  }
  const tq4 = performance.now(); timing.sRest = tq4 - tq3; timing.sMarks = tq4 - tq3d;
  const fullBySlot = new Map<number, Float64Array>();
  const fullTrack = PERIODS.map((_, p) => fullBySlot.get(slots[p]) ?? fullBySlot.set(slots[p], a.full_track_in(p)).get(slots[p])!);
  const tq4a = performance.now(); timing.sFull = tq4a - tq4;
  const [totalCost, builtCost] = a.costs();
  // T-4299: the money kept per record against the same worked out from nothing
  if (patchMode === "check") { const d = a.kept_check(PERIOD_HOURS); if (d) console.error(`T-4299 check: ${d}`); }
  const blueprintCost = Math.max(totalCost - builtCost, 0);
  const tq5 = performance.now(); timing.sSums = tq5 - tq4a; timing.sCosts = tq5 - tq4;
  // (a rows state carries none: the tables changed stay pending for the next full one)
  const phases = rowsOnly ? NO_PHASES : stateRows.phases(a, meta);
  timing.sPhases = performance.now() - tq5; timing.nPhaseLines = rowsOnly ? 0 : stateRows.phaseReads;
  if (patchMode === "check" && !rowsOnly && !sameBytes(phases, a.phases())) console.error("T-1540 check: the phase tables read per line differ from a whole read");
  return {
    version: ++version,
    markers,
    fullTrack,
    turnLimits: a.turn_limits(),
    periods: a.game_periods(),
    hours: Array.from(a.game_hours()),
    crossings,
    conflicts,
    singles,
    rightHand: a.right_hand(),
    cash: a.cash(),
    builtCost,
    blueprintCost,
    canUndo: a.can_undo(),
    canRedo: a.can_redo(),
    edges,
    nodes,
    lines,
    links: Array.from(a.station_links()),
    apart: Array.from(a.station_apart()),
    buildings: timed("sBld", () => shownBuildings(a)),
    keeps: namedKeeps(a),
    render: {
      edgeParts: drawn,
      edgeSpeedParts: speedParts,
      phases,
      meta,
      samples,
      lineTable,
      sampleBend,
      bends,
    },
    money: moneyState(),
    anchor: anchorSpec(),
    rowsRead: stateRows.lastRead,
  };
}

/** T-4149: `f`'s answer, its time in `timing[k]` (the edit's answer carries them) */
function timed<T>(k: string, f: () => T): T {
  const t0 = performance.now(), v = f();
  timing[k] = performance.now() - t0;
  return v;
}

/** whether the game just loaded was converted from an older frame (T-308), for the "loaded" message */
export let convertedOnLoad = false;

export function sendOpened() {
  send({ kind: "opened", zones: opened ? [...opened] : null, start: startZone });
}
export function sendHome() {
  send({ kind: "home", home: home ?? null });
}

/** After a load or a new game: the clock, fresh state, a new snapshot for demand. */
export function restart(t: number, from: "autosave" | "file" | "new", reason?: "rebuilt") {
  putMidLoad(false);
  forgetKeeps();
  putClk({ t, rate: clk.rate, at: wallNow() });
  game++;
  saves.edited = false;
  // T-284: the loaded hour counts as paid up to the clock, evenly (its trains' timing is not known yet)
  putSettledT(t);
  putPartHour(Math.floor(t / 3600));
  putPartCost((t - partHour * 3600) / 3600); putPartFare(partCost);
  hourShape.clear();
  putIncomePerHour(DAY_KINDS.map(() => PERIODS.map(() => 0))); // the main thread sends the new network's once demand lands
  putRidesPerHour(DAY_KINDS.map(() => PERIODS.map(() => 0)));
  putLineIncome(new Map()); // T-715: the old game's per line, which regions' fares were paid from
  putLineRides(new Map());
  putLinePoints(new Map());
  api!.clear_riders(); // and its riders (T-164)
  putXferSet(null);
  forget(true);
  syncFrame();
  keep();
  sendOpened();
  sendHome();
  send({ kind: "listOrder", lines: [...api!.list_order(0)], stations: [...api!.list_order(1)] });
  send({ kind: "lineShapes", shapes: Object.fromEntries(lineShapes) });
  send({ kind: "hiddenLines", lines: [...hiddenLines] });
  send({ kind: "diagramCities", cities: diagramCities });
  send({ kind: "newHeadway", seconds: newLineHeadway });
  send({ kind: "loaded", from, clock: t, frame: frameSpec(), converted: convertedOnLoad || undefined, game, slot: saves.loadingInto ?? saves.activeSlot, reason });
  // T-3734: a game started in the active slot itself is that slot's now, and the page hears so
  if (saves.loadingInto === null) { saves.slotGame = game; void publishSlots(); }
  convertedOnLoad = false;
  sendState();
  clearTimeout(snapTimer);
  send({ kind: "snapshot", snapshot: buildSnapshot() });
}

/** the last build's pieces of the renderer's buffers (T-496) */
const netCache = newNetCache();
/** the lines with trains at the last full build */
let runningLines = "";
/** T-1170: the anchor the drawn pieces were last laid at, while they are to move to the new one */
export let shiftFrom: Anchor | null = null;
/** each edge's points the pieces were laid from, at `shiftFrom` */
export let shiftWas = new Map<number, Float32Array>();
export let anchorMoves = 0;
/** every this many anchor moves the pieces are laid exactly again, so float32 rounding never adds up */
export const SHIFTS_EXACT = 8;
/** local units: a move farther than this is laid exactly (float32's step is 0.25 there, so a moved
 * point near the new anchor is off by at most 12.5 cm; about 3,000 km of ground in China) */
export const SHIFT_REACH = 2 ** 22;

/** T-2766: a state of the rows alone (`ClockState.rows`): the world and its money, without the network or trips,
 * which a full state brings after it. False, nothing sent, before the first network or while the drawn pieces are to
 * move to a new anchor (that state must be a full one). */
export function sendRows(built?: ClockState): boolean {
  if (!posted.net || shiftFrom) return false;
  const t0 = performance.now();
  // T-3803: rows a follow-up built before an edit came between its pieces, with the money as it is now
  const s: ClockState = built ? { ...built, ...nowMoney(), effectiveAt: undefined, ghostFirst: undefined } : buildState(true);
  s.rows = true;
  s.noTrips = true;
  s.render = { phases: NO_PHASES, meta: NO_PHASES, samples: NO_PHASES, lineTable: NO_PHASES };
  s.ms = [performance.now() - t0, 0];
  timing.rows = s.ms[0];
  const t1 = performance.now();
  postState(s, []);
  timing.rowsPost = performance.now() - t1;
  return true;
}
/** T-4114: whether a rows state can go now (`sendRows`'s own test) */
export const rowsCanGo = () => !!posted.net && !shiftFrom;
/** the parts of a state that can change between a follow-up's pieces without an edit: the cash and the money record
 * (the money tick, demand's money results) */
const nowMoney = () => ({ cash: api!.cash(), money: moneyState() });

/** `service`: only the service changed (a train count, T-229) or a name (T-3803): the renderer's buffers are not
 * built again, which was about half the edit's time on New York's real network.
 * `split`: the trips' next hour a task later (`sendTrips`), as after an edit (T-1407) */
export function sendState(service = false, effectiveAt?: number, trips = true, ghostFirst = false, split = false) {
  const g = stateSteps(service, effectiveAt, trips, ghostFirst, split);
  while (!g.next().done);
}

/** T-4281: what each of the drawn network's pieces works out (map/network.ts `networkSteps`, a yield after each), the last
 * with the post and the trips, as an edit's answer names what it waited behind */
const NET_PIECES = ["network: counts and splits", "network: track", "network: lines", "network: speeds", "network: stations", "network: rest and post"];
/** `sendState` in pieces of about 0.1 s on the biggest saves (T-3803): the rows, then the network's pieces
 * (`networkSteps`), then its post with the trips, the last never split (T-648: a state's trips follow it before
 * anything else). It yields only where an edit may come between (a first network posted, no anchor move pending);
 * `rowsBuilt` is told the rows, which go out as a rows state if one does. `boardAt`: a service edit's follow-up, begun
 * then (`loadsLater`, T-3655). */
export function* stateSteps(service = false, effectiveAt?: number, trips = true, ghostFirst = false, split = false, rowsBuilt?: (s: ClockState) => void, boardAt?: number): Generator<void, void, void> {
  if (midLoad) return; // (`restart` sends the loaded game's)
  freezeCheckpoint("build-state-start");
  putPiece("rows");
  const t0 = performance.now();
  const s = buildState();
  freezeCheckpoint("build-state-end", { ms: performance.now() - t0 });
  const t1 = performance.now();
  if (effectiveAt !== undefined) s.effectiveAt = effectiveAt;
  if (!trips) s.noTrips = true;
  if (ghostFirst) s.ghostFirst = true;
  moneyLag = runLag(s.lines.filter((l) => l.ok && l.running).map((l) => ({ tph: l.tph[1], lengthM: l.lengthM, cars: l.cars, oneWayS: l.roundTrip[1] / 2 - l.turnaround })));
  // a line gaining its first train or losing its last changes the draw order (T-533)
  const running = s.lines.filter((l) => l.ok && runsTrains(l)).map((l) => l.id).join();
  const r = s.render;
  // the per-edge arrays stay here (T-1029): the network carries what the main thread needs of them.
  // `r.phases` is the rows' own array (T-1540), never transferred: a copy goes where it is wanted.
  s.render = { phases: NO_PHASES, meta: r.meta, samples: r.samples, lineTable: r.lineTable };
  if (service && running === runningLines && posted.net) {
    s.service = true;
    s.ms = [t1 - t0, 0];
    timing.state = t1 - t0;
    s.render.phases = r.phases.slice();
    posted.arrays.set("phases", r.phases);
    keepPosted("meta", r.meta);
    const later = trips && loadsLater(boardAt);
    if (later) s.loadsFollow = true;
    postState(s, [s.render.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
    timing.post = performance.now() - t1;
    if (trips) sendTrips(split, undefined, !later);
    return;
  }
  // The renderer's buffers are built here too, so the main thread only uploads them (T-063).
  // the complexes: a waiting room or car park is drawn once for one (T-1528), and members whose boxes
  // do not touch are drawn joined (T-1859)
  const complexOf = timed("cxOf", complexOfNodes);
  const shifted = !!shiftFrom;
  const pieces = !shifted && !!posted.net;
  if (shiftFrom) {
    shiftNetCache(netCache, shiftWas, edgeDrawn, ...anchorShift(shiftFrom, anchor()));
    shiftFrom = null;
    shiftWas = new Map();
  }
  const tb0 = performance.now();
  const buildingLL = s.buildings?.map((b) => b.corners.flatMap((_, i) => (i % 2 ? [] : frame.toLonLat(b.corners[i], b.corners[i + 1]))));
  timing.bldLL = performance.now() - tb0;
  const steps = networkSteps({ ...s, render: r, complexOf, buildingLL }, netCache, true);
  let n: NetworkBuffers;
  /** the time between pieces, no part of the build's */
  let between = 0;
  if (pieces) {
    rowsBuilt?.(s);
    for (let k = 0; ; k++) {
      const ty = performance.now();
      yield;
      between += performance.now() - ty;
      putPiece(NET_PIECES[k] ?? "network " + k);
      const x = steps.next();
      if (x.done) { n = x.value; break; }
    }
    // what can change between pieces without an edit (the money tick, demand's money results)
    Object.assign(s, nowMoney());
  } else for (;;) { const x = steps.next(); if (x.done) { n = x.value; break; } }
  runningLines = running;
  const t2 = performance.now();
  // a moved build is off the exact one by float32 rounding: `?patch=shiftcheck` measures by how much
  if (patchMode === "shiftcheck" && shifted) checkShift(n, networkFromState({ ...s, render: r, complexOf, buildingLL }));
  const out = (s.net = netForPost(n, patchMode === "check" && !shifted ? networkFromState({ ...s, render: r, complexOf, buildingLL }) : undefined));
  s.ms = [t1 - t0, t2 - t1 - between];
  timing.state = t1 - t0;
  timing.buffers = t2 - t1 - between;
  const bufs = new Set<ArrayBufferLike>([r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
  for (const st of [out.track, out.lines]) for (const v of Object.values(st)) if (ArrayBuffer.isView(v) && v.byteLength) bufs.add(v.buffer);
  for (const k of SAME_FIELDS) { const v = out[k]; if (v) bufs.add(v.buffer); }
  for (const v of out.edgePts.values()) bufs.add(v.buffer);
  if (out.patch) for (const v of [out.patch.track, out.patch.lines, out.patch.edgeIds]) bufs.add(v.buffer);
  const later = trips && loadsLater(boardAt);
  if (later) s.loadsFollow = true;
  postState(s, [...bufs] as Transferable[]);
  timing.post = performance.now() - t2;
  diagramIn = { lines: s.lines, edges: s.edges, pts: n.edgePts };
  if (diagramOn) postDiagram();
  if (trips) sendTrips(split, undefined, !later);
}

export let diagramIn: DiagramIn | null = null;
export function putAnchorMoves(v: typeof anchorMoves) { anchorMoves = v; }
export function putConvertedOnLoad(v: typeof convertedOnLoad) { convertedOnLoad = v; }
export function putShiftFrom(v: typeof shiftFrom) { shiftFrom = v; }
export function putShiftWas(v: typeof shiftWas) { shiftWas = v; }
