// The demand views the network renderer draws itself (T-084, SPEC 8): line width by riders and
// trains filled by how full they are. Built on the main thread from the
// network now drawn, the world and the demand result, whenever one of them, the period in force
// or a display setting changes: one pass over the line stroke segments and train samples (a few ms on New York's real network, notes/T-084.md), never per frame.
//
// Lines sharing track are drawn side by side (T-1799), each band its own line's width, so a trunk's
// ribbon is their sum; folded back over each other zoomed out (renderer.ts).

import { gauge } from "../game/boarding";
import { lineLoadsFor, type DemandView } from "../game/demand";
import { loadFailure, tripsOff } from "../game/loading";
import { PERIODS } from "../game/periods";
import type { StopKey } from "../game/carryLoads";
import type { PurposeFilter } from "../game/purposes";
import type { WorldView } from "../game/types";
import type { NetworkBuffers } from "./network";
import type { NetworkLayers } from "./renderer";
import { LINE_VIEW_WIDTH, trafficBands } from "./lineBands";
import { checkPatches, freshly, freshNow, PerRecord, sameBuilt } from "../game/worldDiff";
import { backBaseOf } from "./network";
import { rowKey } from "./trainSlots";
import { stepSince, timed } from "../game/stepTimes";
/** two float arrays bit for bit (NaN as NaN) */
const sameWords = (a: Float32Array, b: Float32Array) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };

export interface LayerSwitches {
  lineLoad: boolean;
  /** the map's purpose filter (T-158): line widths count this purpose only; train fill counts
   * everyone */
  purpose?: PurposeFilter;
  /** every kind of day's view (T-260): the widest line is the busiest stretch-day of any of
   * them, so a weekend's lines are drawn thinner than a weekday's rather than rescaled */
  days?: (DemandView | null)[];
  /** a station and its complex are one place when a line's riders are carried onto stops it was
   * not solved for (T-1419) */
  stopKey?: StopKey;
}

/**
 * Line width (T-227, Anita: "the scaling should be linear. the most used line should be a fixed
 * width like 3x what it is now and the least used lines should be like 0.5x normal width"): riders
 * a day on a stretch, both directions together, as a straight share of the network's busiest
 * stretch-day (`busiest`: of the day shown and the other kind of day, the purpose shown), from
 * `W_MIN` of the normal line width for none to `W_MAX` (3 times T-084's 2.2) for the busiest. The
 * whole day, not the hour on the clock (T-274, Anita: "lets show demand for the whole day rather
 * than the current hour"), so widths agree with the riders a day in the panels and the hover, and
 * do not change as the periods pass; the platforms and trains show the hour.
 */
export const W_MIN = 0.5, W_MAX = 6.6;
/** T-1799 stage 2: the Traffic view's widths drawn at 85% of the width before T-1821, since lines
 * sharing track now sit side by side and a trunk is their sum (Anita, 2026-10-29: "adjust default
 * line width down a tad") */
export const TRAFFIC_WIDTH = 0.85 / LINE_VIEW_WIDTH;
export function widthFor(ridersPerDay: number, busiest: number): number {
  return W_MIN + (W_MAX - W_MIN) * (busiest > 0 ? Math.min(1, Math.max(0, ridersPerDay) / busiest) : 0);
}

/** A train's fill when its line runs and train loads are shown but demand has no figures for it
 * yet (T-324); -1 is no gauge at all (train loads off, or the line not running). */
export const PENDING = -2;

/** Riders a day on each stop-to-stop stretch of run 0, both directions, every period (the purpose
 * filter's purpose only); null when demand has no answer that fits the line's `stops`, its last
 * answer for other stops carried onto them (T-1419). */
function stretchDay(dv: DemandView | null, lineId: string, stops: readonly string[], purpose: PurposeFilter, key?: StopKey): Float32Array | null {
  const n = stops.length;
  const all = dv ? lineLoadsFor(dv, lineId, purpose, stops, key) : undefined;
  if (!all?.length || n < 2) return null;
  const day = new Float32Array(n - 1);
  for (const shown of all) {
    if (!shown || shown.length !== 2 * (n - 1)) return null;
    // run 1 runs the stops backwards: its stretch j covers run 0's stretch n - 2 - j
    for (let k = 0; k < n - 1; k++) day[k] += shown[k] + shown[n - 1 + (n - 2 - k)];
  }
  return day;
}

/** Per line id: per stop-to-stop stretch of run 0, riders a day (both directions; the purpose
 * filter's purpose only) and the two runs' gauges in the period in force (everyone aboard); absent
 * when demand has no answer for it that fits its stops. */
type LineNums = { perDay: Float32Array; g0: Float32Array | null; g1: Float32Array | null };
/** T-4284: each line's numbers and busiest stretch-day, kept per demand view while what they are made from is the same (its
 * row, its figures, the period, the purpose shown): an edit makes again only the lines it changed */
const numsKept = new WeakMap<DemandView, PerRecord<string, LineNums | null>>();
const busiestKept = new WeakMap<DemandView, PerRecord<string, number>>();
const keptOf = <V>(by: WeakMap<DemandView, PerRecord<string, V>>, dv: DemandView): PerRecord<string, V> => by.get(dv) ?? by.set(dv, new PerRecord()).get(dv)!;

function lineNumbers(w: WorldView, dv: DemandView | null, period: number, purpose: PurposeFilter = "all", key?: StopKey) {
  // gauges are null for a line whose riders are carried from before its edit (T-1419): its trains
  // stay unknown, as the boarding account has not run them yet
  const out = new Map<number, LineNums>();
  if (!dv) return out;
  const P = PERIODS[period];
  const hours = P.to - P.from;
  const kept = keptOf(numsKept, dv);
  kept.trim(w.save.lines.length);
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    const v = kept.get(l.id, [l, st, period, purpose, key], (): LineNums | null => {
      const dm = dv.lines.get(l.id);
      const n = l.stops.length;
      const loads = dm?.loads[period];
      const perDay = stretchDay(dv, l.id, l.stops, purpose, key);
      if (!st || st.status !== "running" || !perDay || n < 2) return null;
      if (!loads || loads.length !== 2 * (n - 1)) return { perDay, g0: null, g1: null };
      const trains = l.tphP[period] * hours;
      const factor = dv.peakHourFactor[period] ?? 1;
      const g0 = new Float32Array(n - 1), g1 = new Float32Array(n - 1);
      for (let k = 0; k < n - 1; k++) {
        const a = loads[k], b = loads[n - 1 + (n - 2 - k)];
        g0[k] = trains > 0 ? gauge(a / trains * factor, st.cars) : -1;
        g1[k] = trains > 0 ? gauge(b / trains * factor, st.cars) : -1;
      }
      return { perDay, g0, g1 };
    });
    if (v) out.set(l.num, v);
  }
  return out;
}

/** The network's busiest stretch, riders a day both directions together (the purpose shown): the
 * widest line (T-227, T-274). */
export function busiestDay(w: WorldView, dv: DemandView | null, purpose: PurposeFilter = "all", key?: StopKey): number {
  let best = 0;
  if (!dv) return best;
  // (a max over each line's max is the max over every stretch, whatever the order)
  const kept = keptOf(busiestKept, dv);
  kept.trim(w.save.lines.length);
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (st?.status !== "running") continue;
    best = Math.max(best, kept.get(l.id, [l, st, purpose, key], () => {
      let most = 0;
      const day = stretchDay(dv, l.id, l.stops, purpose, key);
      if (day) for (const v of day) most = Math.max(most, v);
      return most;
    }));
  }
  return best;
}

/** T-2274: the Traffic view's widths of bus and ferry lines (drawn by `busLayer.ts`): per running
 * line's number, each stretch's width as the rail lines' rule gives it (a share of a rail line's
 * width in the Line view), so a stretch is as wide as a rail stretch with as many riders. */
export function busWidths(w: WorldView, dv: DemandView | null, on: LayerSwitches): Map<number, Float32Array> {
  const out = new Map<number, Float32Array>();
  const busiest = Math.max(busiestDay(w, dv, on.purpose, on.stopKey), ...(on.days ?? []).map((d) => busiestDay(w, d, on.purpose, on.stopKey)));
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st?.bus || st.status !== "running") continue;
    const day = stretchDay(dv, l.id, l.stops, on.purpose ?? "all", on.stopKey);
    if (day) out.set(l.num, day.map((v) => TRAFFIC_WIDTH * widthFor(v, busiest)));
  }
  return out;
}

/** the last layers built: their network, line numbers, fill and pending lines, and the period (T-4284) */
let lastLayers: { net: NetworkBuffers; nums: Map<number, LineNums>; fill: Float32Array; period: number; pending: Set<number> } | null = null;
let plainArr: Float32Array | null = null;
const plainWidths = (n: number) => (plainArr?.length === n ? plainArr : (plainArr = new Float32Array(n).fill(1)));

/** Build the renderer's layers for `net`. */
export function buildLayers(net: NetworkBuffers, w: WorldView, dv: DemandView | null, period: number, on: LayerSwitches): NetworkLayers {
  const L = net.lines;
  const nSeg = L.count;
  const nSamples = net.samples.length / 2;
  const fresh = freshNow();
  let t = import.meta.env.DEV ? performance.now() : 0;
  const nums = lineNumbers(w, dv, period, on.purpose, on.stopKey);
  t = stepSince("layers.nums", t);
  if (import.meta.env.DEV && checkPatches() && timed("dev.check.nums", () => !sameBuilt(nums, freshly(() => lineNumbers(w, dv, period, on.purpose, on.stopKey))))) console.error("T-4284 check: the demand layers' line numbers kept per line differ from a whole build");
  const stretchOf = L.stretch;

  // The segments of one line along one edge are consecutive and share everything below: work per
  // run of them (a few thousand on New York's real network, against 118k segments). Only the riders' widths read them.
  const runs: number[] = [];
  for (let i = 0; on.lineLoad && i < nSeg; ) {
    let j = i + 1;
    while (j < nSeg && L.edge[j] === L.edge[i] && L.colour[j] === L.colour[i]) j++;
    runs.push(i, j);
    i = j;
  }
  t = stepSince("layers.runs", t);

  // widths per stroke segment, the day's riders against the busiest stretch-day (T-227, T-274)
  // (T-4284: the plain widths one array while the count is the same)
  const lineWidth = on.lineLoad ? new Float32Array(nSeg).fill(TRAFFIC_WIDTH) : plainWidths(nSeg);
  if (on.lineLoad && stretchOf) {
    const busiest = Math.max(busiestDay(w, dv, on.purpose, on.stopKey), ...(on.days ?? []).map((d) => busiestDay(w, d, on.purpose, on.stopKey)));
    if (import.meta.env.DEV && checkPatches() && !sameBuilt(busiest, freshly(() => Math.max(busiestDay(w, dv, on.purpose, on.stopKey), ...(on.days ?? []).map((d) => busiestDay(w, d, on.purpose, on.stopKey)))))) console.error("T-4284 check: the busiest stretch kept per line differs from a whole build");
    for (let r = 0; r < runs.length; r += 2) {
      const i = runs[r];
      const v = nums.get(L.colour[i]);
      if (v) lineWidth.fill(TRAFFIC_WIDTH * widthFor(v.perDay[stretchOf[i]] ?? 0, busiest), i, runs[r + 1]);
    }
  }
  t = stepSince("layers.widths", t);
  // lines sharing track side by side, each band its riders' width (T-1799 stage 2)
  const bands = on.lineLoad ? trafficBands(L, lineWidth, net.samples, net.sampleStroke, net.corridor) : null;
  t = stepSince("layers.bands", t);

  // the trains: gauges per sample; PENDING for a running line demand has no figures for yet
  // (T-324: drawn striped, never like a full train)
  const fill = new Float32Array(nSamples * 2).fill(-1);
  const strokeOf = net.sampleStroke;
  const pending = new Set<number>();
  if (!loadFailure.peek() && !tripsOff.peek()) for (const l of w.save.lines) if (w.lineStats[l.id]?.status === "running" && l.stops.length >= 2 && !nums.get(l.num)?.g0) pending.add(l.num);
  const fillSamples = (k0: number, k1: number, fill: Float32Array) => {
    let lastLine = -1;
    let v: ReturnType<typeof nums.get>;
    if (strokeOf && stretchOf)
      for (let k = k0; k < k1; k++) {
        const i = strokeOf[k];
        if (i < 0) continue;
        if (L.colour[i] !== lastLine) {
          lastLine = L.colour[i];
          v = nums.get(lastLine);
        }
        if (!v?.g0 || !v.g1) {
          if (pending.has(lastLine)) fill[k * 2] = fill[k * 2 + 1] = PENDING;
          continue;
        }
        const s = stretchOf[i];
        fill[k * 2] = v.g0[s] ?? -1;
        fill[k * 2 + 1] = v.g1[s] ?? -1;
      }
  };
  // T-4284: a line table row whose line has the numbers, pending mark and samples' stretches it had in the last layers keeps
  // their fill there; the others are filled again (an edit: its lines; new trips: every line)
  const was = !fresh && lastLayers && lastLayers.period === period && (lastLayers.net === net || (!!net.patch && lastLayers.net.serial === net.patch.base)) ? lastLayers : null;
  let fillChanged: NetworkLayers["fillChanged"];
  if (was && strokeOf && stretchOf) {
    const keys = new Set<number>(), T = net.lineTable, base = backBaseOf(T, net.meta);
    const P = was.net.lineTable, pBase = backBaseOf(P, was.net.meta);
    const moved = was.net === net ? null : new Set(net.patch!.stretches);
    let covered = 0;
    for (let r = 0; r * 4 + 1 < T.length; r++) {
      const first = T[r * 4], c = T[r * 4 + 1];
      if (!(c > 0)) continue;
      covered += c;
      const key = rowKey(r, base), line = key >> 1;
      // (the row of that key in the last network's table)
      const pr = key & 1 ? (pBase > 0 ? pBase + line : -1) : line;
      const same = pr >= 0 && pr * 4 + 1 < P.length && P[pr * 4 + 1] === c && !moved?.has(line) && was.nums.get(line) === nums.get(line) && was.pending.has(line) === pending.has(line);
      if (same) fill.set(was.fill.subarray(P[pr * 4] * 2, (P[pr * 4] + c) * 2), first * 2);
      else { fillSamples(first, first + c, fill); keys.add(key); }
    }
    // (samples in no row: as a whole fill would have them)
    if (covered !== nSamples) { fill.fill(-1); fillSamples(0, nSamples, fill); }
    else fillChanged = { from: was.fill, keys };
  } else fillSamples(0, nSamples, fill);
  if (import.meta.env.DEV && was && checkPatches()) timed("dev.check.fill", () => {
    const whole = new Float32Array(nSamples * 2).fill(-1);
    fillSamples(0, nSamples, whole);
    if (!sameWords(whole, fill)) console.error("T-4284 check: the trains' fill kept per line row differs from a whole build");
  });
  if (!fresh) lastLayers = { net, nums, fill, period, pending };
  stepSince("layers.fill", t);

  return { lineWidth, fill, traffic: on.lineLoad, bands, plain: !on.lineLoad, fillChanged };
}
