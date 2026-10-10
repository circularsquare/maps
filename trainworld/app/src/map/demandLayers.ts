// The demand views the network renderer draws itself (T-084, SPEC 8): line width by riders, station
// circles by riders, trains filled by how full they are. Built on the main thread from the
// network now drawn, the world and the demand result, whenever one of them, the period in force
// or a display setting changes: one pass over the line stroke segments, train samples and
// stations (a few ms on New York's real network, notes/T-084.md), never per frame.
//
// Line widths change how wide a bundle of lines sharing track is, so the lines on an edge are laid
// side by side again at their own widths (T-062's slots, kept in `Strokes.info`), and the trains
// and the picking follow: the new offsets are written into the network's own `lines.offset` and
// `sampleOff`, which pick.ts and `tripAt` read; the offsets as built are kept here.

import type { DemandView } from "../game/demand";
import { PERIODS } from "../game/periods";
import type { WorldView } from "../game/types";
import type { NetworkBuffers } from "./network";
import type { NetworkLayers } from "./renderer";

export interface LayerSwitches {
  lineLoad: boolean;
  stationRiders: boolean;
  trainLoad: boolean;
  trainLoadBasis?: "average" | "busiest";
}

/**
 * Line width: riders an hour on a stretch, both directions together, in the period in force.
 * Width grows with the square root (a line ten times as busy is about three times as wide), from
 * 0.45 of the normal line width when empty to 2.2 times at `LINE_FULL` and above.
 */
export const LINE_FULL = 30000;
const W_MIN = 0.45, W_MAX = 2.2;
export function widthFor(ridersPerHour: number): number {
  return W_MIN + (W_MAX - W_MIN) * Math.sqrt(Math.min(1, Math.max(0, ridersPerHour) / LINE_FULL));
}

/** Station size: boardings plus alightings a day; the circle grows with the square root up to
 * 2.6 times its normal radius at `STATION_FULL` and above (the shader's `1 + 1.6 x`). */
export const STATION_FULL = 150000;
export function stationScale(ridersPerDay: number): number {
  return Math.sqrt(Math.min(1, Math.max(0, ridersPerDay) / STATION_FULL));
}

/** Seats and crush load per 20 m car: as ui/inspector/others.tsx (T-083) and
 * `SEATS_PER_CAR`, `CRUSH_PER_CAR` in sim/src/demand/api.rs. */
const SEATS_PER_CAR = 44, CRUSH_PER_CAR = 160;
/** The train gauge: half full when every seat is taken, full at crush; 2 = over full. */
export function gauge(aboard: number, cars: number): number {
  const seats = SEATS_PER_CAR * cars, crush = CRUSH_PER_CAR * cars;
  if (aboard <= seats) return seats > 0 ? (0.5 * Math.max(0, aboard)) / seats : 0;
  if (aboard <= crush) return 0.5 + (0.5 * (aboard - seats)) / (crush - seats);
  return 2;
}

/** The offsets as built, per network (the arrays themselves are rewritten for picking). */
const pristine = new WeakMap<NetworkBuffers, { offset: Float32Array; sampleOff: Float32Array }>();

/** Per line id: per stop-to-stop stretch of run 0, riders an hour (both directions) and the two
 * runs' gauges; absent when demand has no answer for it that fits its stops. */
function lineNumbers(w: WorldView, dv: DemandView | null, period: number, basis: "average" | "busiest") {
  const out = new Map<number, { perHour: Float32Array; g0: Float32Array; g1: Float32Array }>();
  if (!dv) return out;
  const P = PERIODS[period];
  const hours = P.to - P.from;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    const dm = dv.lines.get(l.id);
    const n = l.stops.length;
    const loads = dm?.loads[period];
    if (!st || st.status !== "running" || !loads || n < 2 || loads.length !== 2 * (n - 1)) continue;
    const trains = l.tph[P.demand] * hours;
    const factor = basis === "busiest" ? dv.peakHourFactor[period] ?? 1 : 1;
    const perHour = new Float32Array(n - 1), g0 = new Float32Array(n - 1), g1 = new Float32Array(n - 1);
    for (let k = 0; k < n - 1; k++) {
      // run 1 runs the stops backwards: its stretch j covers run 0's stretch n - 2 - j
      const a = loads[k], b = loads[n - 1 + (n - 2 - k)];
      perHour[k] = (a + b) / hours;
      g0[k] = trains > 0 ? gauge(a / trains * factor, st.cars) : -1;
      g1[k] = trains > 0 ? gauge(b / trains * factor, st.cars) : -1;
    }
    out.set(l.num, { perHour, g0, g1 });
  }
  return out;
}

/** Build the renderer's layers for `net`; also rewrites `net.lines.offset` and `net.sampleOff`. */
export function buildLayers(net: NetworkBuffers, w: WorldView, dv: DemandView | null, period: number, on: LayerSwitches): NetworkLayers {
  const L = net.lines;
  const nSeg = L.count;
  const nSamples = net.samples.length / 2;
  let base = pristine.get(net);
  if (!base) {
    base = { offset: (L.offset ?? new Float32Array(nSeg)).slice(), sampleOff: (net.sampleOff ?? new Float32Array(nSamples)).slice() };
    pristine.set(net, base);
  }
  const nums = lineNumbers(w, dv, period, on.trainLoadBasis ?? "busiest");
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

  // widths per stroke segment
  const lineWidth = new Float32Array(nSeg).fill(1);
  if (on.lineLoad && info)
    for (let r = 0; r < runs.length; r += 2) {
      const i = runs[r];
      const v = nums.get(L.colour[i]);
      if (v) lineWidth.fill(widthFor(v.perHour[info[i * 4]] ?? 0), i, runs[r + 1]);
    }

  // side by side at those widths: per edge, the width in each slot, then each line's place
  const slotW = new Map<number, Float64Array>();
  if (info)
    for (let r = 0; r < runs.length; r += 2) {
      const i = runs[r];
      const k = info[i * 4 + 2];
      let a = slotW.get(L.edge[i]);
      if (!a) slotW.set(L.edge[i], (a = new Float64Array(Math.max(1, k)).fill(1)));
      a[info[i * 4 + 1]] = lineWidth[i];
    }
  const lineOffset = base.offset.slice();
  if (info)
    for (let r = 0; r < runs.length; r += 2) {
      const i = runs[r];
      const a = slotW.get(L.edge[i]);
      if (!a || a.length < 2) continue;
      const slot = info[i * 4 + 1];
      let before = 0, all = 0;
      for (let s = 0; s < a.length; s++) {
        if (s < slot) before += a[s];
        all += a[s];
      }
      lineOffset.fill(info[i * 4 + 3] * (before + a[slot] / 2 - all / 2), i, runs[r + 1]);
    }

  // the trains: offsets and gauges per sample
  const sampleOff = base.sampleOff.slice();
  const fill = new Float32Array(nSamples * 2).fill(-1);
  const strokeOf = net.sampleStroke;
  let lastLine = -1;
  let v: ReturnType<typeof nums.get>;
  if (strokeOf)
    for (let k = 0; k < nSamples; k++) {
      const i = strokeOf[k];
      if (i < 0) continue;
      sampleOff[k] = lineOffset[i];
      if (!on.trainLoad || !info) continue;
      if (L.colour[i] !== lastLine) {
        lastLine = L.colour[i];
        v = nums.get(lastLine);
      }
      if (!v) continue;
      const s = info[i * 4];
      fill[k * 2] = v.g0[s] ?? -1;
      fill[k * 2 + 1] = v.g1[s] ?? -1;
    }

  // stations: the widest bundle through each, and riders
  const bundleAt = new Map<number, number>();
  if (on.lineLoad) {
    const ends = new Map(w.edges.map((e) => [e.id, e]));
    for (const [edge, a] of slotW) {
      const e = ends.get(edge);
      if (!e) continue;
      let all = 0;
      for (const v of a) all += v;
      for (const n of [e.a, e.b]) bundleAt.set(n, Math.max(bundleAt.get(n) ?? 0, all));
    }
  }
  const stationSize = new Float32Array(net.stationCount * 2);
  for (let i = 0; i < net.stationCount; i++) {
    const plain = Math.floor((net.stations[i * 3 + 2] + 0.5) / 4);
    const id = net.stationIds[i];
    stationSize[i * 2] = on.lineLoad ? bundleAt.get(Number(id)) ?? plain : plain;
    const d = on.stationRiders ? dv?.stations.get(id) : undefined;
    stationSize[i * 2 + 1] = d ? stationScale(d.boardings + d.alightings) : 0;
  }

  // picking and `tripAt` read these
  if (L.offset) L.offset.set(lineOffset);
  if (net.sampleOff) net.sampleOff.set(sampleOff);
  return { lineWidth, lineOffset, sampleOff, fill, stationSize };
}
