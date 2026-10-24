// The demand views the network renderer draws itself (T-084, SPEC 8): line width by riders, station
// use circles (T-482), trains filled by how full they are. Built on the main thread from the
// network now drawn, the world and the demand result, whenever one of them, the period in force
// or a display setting changes: one pass over the line stroke segments, train samples and
// stations (a few ms on New York's real network, notes/T-084.md), never per frame.
//
// Lines sharing track are drawn over each other (T-190), so a line's width moves nothing else; a
// station's box covers the widest line through it.

import { gauge } from "../game/boarding";
import type { Complexes } from "../game/complexes";
import { lineLoadsFor, type DemandView } from "../game/demand";
import { loadFailure } from "../game/loading";
import { PERIODS } from "../game/periods";
import type { PurposeFilter } from "../game/purposes";
import type { WorldView } from "../game/types";
import type { NetworkBuffers } from "./network";
import type { NetworkLayers } from "./renderer";

export interface LayerSwitches {
  lineLoad: boolean;
  /** riders a day a station's use circle stands for (T-482, game/stationUse.ts `useFigure`), or
   * absent for no circles */
  stationFigure?: ((id: string) => number | undefined) | null;
  trainLoad: boolean;
  /** station complexes (T-146): one use circle per complex, at its stations' mean position */
  complexes?: Complexes | null;
  /** the map's purpose filter (T-158): line widths count this purpose only; train fill and the
   * station circles count everyone */
  purpose?: PurposeFilter;
  /** every kind of day's view (T-260): the widest line is the busiest stretch-day of any of
   * them, so a weekend's lines are drawn thinner than a weekday's rather than rescaled */
  days?: (DemandView | null)[];
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
export function widthFor(ridersPerDay: number, busiest: number): number {
  return W_MIN + (W_MAX - W_MIN) * (busiest > 0 ? Math.min(1, Math.max(0, ridersPerDay) / busiest) : 0);
}

/** Station use circle area is proportional to its riders a day (T-482: origins, transfers or both;
 * T-722, no cap). The radius is `stationHaloRadius(zoom)` times this scale, so STATION_FULL riders
 * have that radius at any zoom; a circle with any riders is never smaller than STATION_MIN_SCALE of
 * it, so tiny stations stay visible. 140,000 puts the real New York save's biggest total (Penn
 * Station's complex, 681,000) at 100 px radius at zoom 13 (Anita: the largest about 200 px across). */
export const STATION_FULL = 140000, STATION_MIN_SCALE = 0.1;
export function stationScale(ridersPerDay: number): number {
  return ridersPerDay > 0 ? Math.max(STATION_MIN_SCALE, Math.sqrt(ridersPerDay / STATION_FULL)) : 0;
}

/** A train's fill when its line runs and train loads are shown but demand has no figures for it
 * yet (T-324); -1 is no gauge at all (train loads off, or the line not running). */
export const PENDING = -2;

/** Riders a day on each stop-to-stop stretch of run 0, both directions, every period (the purpose
 * filter's purpose only); null when demand has no answer that fits the line's `n` stops. */
function stretchDay(dv: DemandView | null, lineId: string, n: number, purpose: PurposeFilter): Float32Array | null {
  const all = dv ? lineLoadsFor(dv, lineId, purpose) : undefined;
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
function lineNumbers(w: WorldView, dv: DemandView | null, period: number, purpose: PurposeFilter = "all") {
  const out = new Map<number, { perDay: Float32Array; g0: Float32Array; g1: Float32Array }>();
  if (!dv) return out;
  const P = PERIODS[period];
  const hours = P.to - P.from;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    const dm = dv.lines.get(l.id);
    const n = l.stops.length;
    const loads = dm?.loads[period];
    const perDay = stretchDay(dv, l.id, n, purpose);
    if (!st || st.status !== "running" || !loads || !perDay || n < 2 || loads.length !== 2 * (n - 1)) continue;
    const trains = l.tph[P.demand] * hours;
    const factor = dv.peakHourFactor[period] ?? 1;
    const g0 = new Float32Array(n - 1), g1 = new Float32Array(n - 1);
    for (let k = 0; k < n - 1; k++) {
      const a = loads[k], b = loads[n - 1 + (n - 2 - k)];
      g0[k] = trains > 0 ? gauge(a / trains * factor, st.cars) : -1;
      g1[k] = trains > 0 ? gauge(b / trains * factor, st.cars) : -1;
    }
    out.set(l.num, { perDay, g0, g1 });
  }
  return out;
}

/** The network's busiest stretch, riders a day both directions together (the purpose shown): the
 * widest line (T-227, T-274). */
export function busiestDay(w: WorldView, dv: DemandView | null, purpose: PurposeFilter = "all"): number {
  let best = 0;
  if (!dv) return best;
  for (const l of w.save.lines) {
    if (w.lineStats[l.id]?.status !== "running") continue;
    const day = stretchDay(dv, l.id, l.stops.length, purpose);
    if (day) for (const v of day) best = Math.max(best, v);
  }
  return best;
}

/** Build the renderer's layers for `net`. */
export function buildLayers(net: NetworkBuffers, w: WorldView, dv: DemandView | null, period: number, on: LayerSwitches): NetworkLayers {
  const L = net.lines;
  const nSeg = L.count;
  const nSamples = net.samples.length / 2;
  const nums = lineNumbers(w, dv, period, on.purpose);
  const info = L.info;

  // The segments of one line along one edge are consecutive and share everything below: work per
  // run of them (a few thousand on New York's real network, against 118k segments).
  const runs: number[] = [];
  for (let i = 0; i < nSeg; ) {
    let j = i + 1;
    while (j < nSeg && L.edge[j] === L.edge[i] && L.colour[j] === L.colour[i]) j++;
    runs.push(i, j);
    i = j;
  }

  // widths per stroke segment, the day's riders against the busiest stretch-day (T-227, T-274)
  const lineWidth = new Float32Array(nSeg).fill(1);
  if (on.lineLoad && info) {
    const busiest = Math.max(busiestDay(w, dv, on.purpose), ...(on.days ?? []).map((d) => busiestDay(w, d, on.purpose)));
    for (let r = 0; r < runs.length; r += 2) {
      const i = runs[r];
      const v = nums.get(L.colour[i]);
      if (v) lineWidth.fill(widthFor(v.perDay[info[i * 4]] ?? 0, busiest), i, runs[r + 1]);
    }
  }

  // the trains: gauges per sample; PENDING for a running line demand has no figures for yet
  // (T-324: drawn striped, never like a full train)
  const fill = new Float32Array(nSamples * 2).fill(-1);
  const strokeOf = net.sampleStroke;
  const pending = new Set<number>();
  if (on.trainLoad && !loadFailure.peek()) for (const l of w.save.lines) if (w.lineStats[l.id]?.status === "running" && l.stops.length >= 2 && !nums.has(l.num)) pending.add(l.num);
  let lastLine = -1;
  let v: ReturnType<typeof nums.get>;
  if (strokeOf && on.trainLoad && info)
    for (let k = 0; k < nSamples; k++) {
      const i = strokeOf[k];
      if (i < 0) continue;
      if (L.colour[i] !== lastLine) {
        lastLine = L.colour[i];
        v = nums.get(lastLine);
      }
      if (!v) {
        if (pending.has(lastLine)) fill[k * 2] = fill[k * 2 + 1] = PENDING;
        continue;
      }
      const s = info[i * 4];
      fill[k * 2] = v.g0[s] ?? -1;
      fill[k * 2 + 1] = v.g1[s] ?? -1;
    }

  // stations: riders (a box no longer grows to cover a wide traffic line, T-239)
  const stationSize = new Float32Array(net.stationCount * 2);
  const fig = on.stationFigure;
  if (fig) for (let i = 0; i < net.stationCount; i++) stationSize[i * 2 + 1] = stationScale(fig(net.stationIds[i]) ?? 0);
  const stationHalo = fig && on.complexes ? complexCircles(net, fig, on.complexes, stationSize) : undefined;

  return { lineWidth, fill, stationSize, stationHalo };
}

/**
 * One use circle per station complex (T-146): its stations' figures (`fig`) together, drawn by
 * one constructed station (the first in the network's order) and moved to the mean position of
 * the complex's constructed stations. Rewrites the circle scale in `stationSize`; returns the
 * per-station circle offset, local units. Blueprint stations have no demand and draw no circle
 * (renderer.ts), so they neither hold nor place one.
 */
export function complexCircles(net: Pick<NetworkBuffers, "stationCount" | "stations" | "stationIds">, fig: (id: string) => number | undefined, cx: Complexes, stationSize: Float32Array): Float32Array {
  const halo = new Float32Array(net.stationCount * 2);
  const groups = new Map<number, number[]>();
  for (let i = 0; i < net.stationCount; i++) {
    if ((Math.round(net.stations[i * 3 + 2]) & 2) !== 0) continue;
    const c = cx.of.get(net.stationIds[i]);
    if (c === undefined) continue;
    const g = groups.get(c);
    if (g) g.push(i);
    else groups.set(c, [i]);
  }
  for (const g of groups.values()) {
    if (g.length < 2) continue;
    let riders = 0, x = 0, y = 0;
    for (const i of g) {
      riders += fig(net.stationIds[i]) ?? 0;
      x += net.stations[i * 3];
      y += net.stations[i * 3 + 1];
      stationSize[i * 2 + 1] = 0;
    }
    const a = g[0];
    stationSize[a * 2 + 1] = riders > 0 ? stationScale(riders) : 0;
    halo[a * 2] = x / g.length - net.stations[a * 3];
    halo[a * 2 + 1] = y / g.length - net.stations[a * 3 + 1];
  }
  return halo;
}
