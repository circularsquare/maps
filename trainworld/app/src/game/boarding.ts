// The boarding account on the main thread (T-164, notes/T-164.md, sim/src/track/boarding.rs).
//
// The clock worker turns demand's riders per period into a load per train and a crowd per platform,
// a whole day at a time, and sends each game hour's slice with that hour's trips. This file packs
// demand's riders for the worker, reads an hour's slice, and evaluates a platform's crowd at a
// moment: the crowd left by the last train to leave it plus the riders who have arrived since.

import { signal } from "@preact/signals";
import type { DemandView } from "./demand";

/** Bumped each time an hour of the account arrives, so panels that ask the worker for a line's
 * day in the account ask again (T-230). */
export const accountVersion = signal(0);

/** Numbers per platform in an hour's slice (`QUEUE_STRIDE` in boarding.rs). */
export const QUEUE_STRIDE = 12;

/** Demand's riders as `TrackApi.set_riders` reads them: the busiest-hour factors, then per line
 * its id, stop count and, per period, boardings then alightings at each route node. Null while a
 * result has no per-node riders. */
export function packRiders(dv: DemandView): Float32Array | null {
  let size = 5;
  for (const [, l] of dv.lines) if (l.board && l.alight) size += 2 + l.board.reduce((a, b) => a + b.length * 2, 0);
  if (size === 5) return null;
  const out = new Float32Array(size);
  for (let p = 0; p < 5; p++) out[p] = dv.peakHourFactor[p] ?? 1;
  let i = 5;
  for (const [id, l] of dv.lines) {
    if (!l.board || !l.alight) continue;
    out[i++] = Number(id);
    out[i++] = (l.board[0]?.length ?? 0) / 2;
    for (let p = 0; p < l.board.length; p++) {
      out.set(l.board[p], i);
      i += l.board[p].length;
      out.set(l.alight[p], i);
      i += l.alight[p].length;
    }
  }
  return out;
}

/** The riders coming to each platform (T-1255) as `TrackApi.set_arrivals` reads them: per line its
 * id, stop count and, per period, the riders at each route node. Null while a result has none. */
export function packArrivals(dv: DemandView): Float32Array | null {
  let size = 0;
  for (const [, l] of dv.lines) if (l.board && l.alight && l.arrive) size += 2 + l.arrive.reduce((a, b) => a + b.length, 0);
  if (size === 0) return null;
  const out = new Float32Array(size);
  let i = 0;
  for (const [id, l] of dv.lines) {
    if (!l.board || !l.alight || !l.arrive) continue;
    out[i++] = Number(id);
    out[i++] = (l.arrive[0]?.length ?? 0) / 2;
    for (const v of l.arrive) {
      out.set(v, i);
      i += v.length;
    }
  }
  return out;
}

/** One game hour of the account, as views into the worker's array. */
export interface BoardHour {
  epoch: number;
  trips: number;
  /** per trip: index of its first load or -1, seconds from departure to its last stop */
  offs: Float32Array;
  /** riders aboard per stop-to-stop stretch, numbered as on run 0 */
  loads: Float32Array;
  /** per platform QUEUE_STRIDE numbers: node, line, run, offset along run 0 (m), riders arriving
   * steadily a second at mid-hour, riders left behind this hour, longest wait this hour (s),
   * riders arriving this hour, first event, events, first pulse, pulses */
  queues: Float32Array;
  queueCount: number;
  /** per train leaving a platform: seconds from `epoch`, riders it left there */
  events: Float32Array;
  /** per platform in time order, riders changing onto it from another line's train (T-554):
   * seconds from `epoch` they reach it, riders */
  pulses: Float32Array;
  /** per event, when its train stopped there (T-1123): its crowd boards then */
  arrive: Float32Array;
  /** per pulse, the station node its riders walk from and the walk, s (T-1124) */
  pulseFrom: Float32Array;
  /** the crowded stretches (T-2236: the busiest hour's trains past the rated load, `CROWDED_FROM` in
   * boarding.rs): line id and run 0 stretch, pairs */
  crowdStretches: Float32Array;
  /** per crowded stretch, its busiest hour's load as a share of the rated load, the fuller way's */
  crowdLoads: Float32Array;
  /** per crowded stretch, that hour of the day in the line's local time */
  crowdHours: Float32Array;
}

/** A crowded stretch on the map (T-2236): its busiest hour's load as a share of the rated load, and
 * that hour of the day in the line's local time. */
export interface CrowdedStretch { load: number; hour: number }

export function readBoardHour(epoch: number, b: Float32Array): BoardHour {
  const [nt, nl, nq, ne, np] = [b[0], b[1], b[2], b[3], b[4]];
  let i = 5;
  const offs = b.subarray(i, (i += 2 * nt));
  const loads = b.subarray(i, (i += nl));
  const queues = b.subarray(i, (i += nq * QUEUE_STRIDE));
  const events = b.subarray(i, (i += ne * 2));
  const pulses = b.subarray(i, (i += np * 2));
  const arrive = b.subarray(i, (i += ne));
  const pulseFrom = b.subarray(i, (i += np * 2));
  const nf = b[i++] ?? 0;
  const crowdStretches = b.subarray(i, (i += nf * 2));
  const crowdLoads = b.subarray(i, (i += nf));
  const crowdHours = b.subarray(i, i + nf);
  return { epoch, trips: nt, offs, loads, queues, queueCount: nq, events, pulses, arrive, pulseFrom, crowdStretches, crowdLoads, crowdHours };
}

/** The most riders waiting on one platform side (T-1067, `WAITING_CAP` in boarding.rs): the account
 * holds its crowds to it as trains leave, and a crowd drawn between two trains stops there too. */
export const WAITING_CAP = 5000;

/** Riders who changed onto platform `q` from other lines' trains in (`t0`, `t`] (T-554). */
export function pulsesBetween(h: BoardHour, q: number, t0: number, t: number): number {
  const base = q * QUEUE_STRIDE, P = h.pulses;
  let lo = h.queues[base + 10], hi = lo + h.queues[base + 11];
  // the first pulse after t0
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (P[mid * 2] <= t0) lo = mid + 1;
    else hi = mid;
  }
  const end = h.queues[base + 10] + h.queues[base + 11];
  let sum = 0;
  for (let p = lo; p < end && P[p * 2] <= t; p++) sum += P[p * 2 + 1];
  return sum;
}

/** When event `e`'s train stopped (T-1123), or when it left without an arrival. */
export function stoppedAt(h: BoardHour, e: number): number {
  const a = h.arrive[e];
  return a === undefined ? h.events[e * 2] : a;
}

/** The last train at platform `q` to have stopped by `t` (its event index), or -1 without events. */
export function lastStopped(h: BoardHour, q: number, t: number): number {
  const base = q * QUEUE_STRIDE;
  const first = h.queues[base + 8], count = h.queues[base + 9];
  if (count <= 0) return -1;
  let lo = first, hi = first + count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (stoppedAt(h, mid) <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Riders at platform `q` at `t` (>= its departure) after event `e`'s train left. */
function grownAt(h: BoardHour, q: number, e: number, t: number): number {
  const t0 = h.events[e * 2];
  return Math.min(WAITING_CAP, h.events[e * 2 + 1] + (t - t0) * h.queues[q * QUEUE_STRIDE + 4] + pulsesBetween(h, q, t0, t));
}

/** Riders waiting at platform `q` at `t` seconds from the hour's epoch: those the last train to
 * stop there left, plus, once it has gone, who has arrived since (T-164). While a train stands
 * there its riders get on evenly over the stand (T-1785: the train's drawn load fills alike, in the
 * renderer's train shader), from the crowd it found to the crowd it leaves. 0 without events. */
export function waitingAt(h: BoardHour, q: number, t: number): number {
  const e = lastStopped(h, q, t);
  if (e < 0) return 0;
  const t0 = h.events[e * 2], left = h.events[e * 2 + 1];
  if (t >= t0) return grownAt(h, q, e, t);
  const first = h.queues[q * QUEUE_STRIDE + 8], stop = stoppedAt(h, e);
  if (e <= first || t < stop || t0 - stop < 1) return left;
  const found = grownAt(h, q, e - 1, stop);
  return found + (left - found) * ((t - stop) / (t0 - stop));
}

/** Riders waiting at platform `q` at `t` seconds from the hour's epoch, whether the last train
 * to stop there left anyone behind, and how many it left (T-284). */
export function crowdAt(h: BoardHour, q: number, t: number): { waiting: number; leftBehind: boolean; left: number } {
  const e = lastStopped(h, q, t);
  if (e < 0) return { waiting: 0, leftBehind: false, left: 0 };
  const left = h.events[e * 2 + 1];
  return { waiting: waitingAt(h, q, t), leftBehind: left >= 1, left };
}

/** Usable width one direction's riders wait on, m: half of a 6 m island platform less the edge
 * strip kept clear (T-219). */
export const PLATFORM_SIDE_WIDTH_M = 2.5;
/** Space a waiting rider takes on a full platform, m²: Fruin's queuing level of service E (0.2-0.3
 * m² a person, standing in contact with others, as the TCQSM uses for platforms), at its roomier
 * end, about the standing density of a train at crush (T-219). */
export const PLATFORM_M2_PER_RIDER = 0.3;
/** Riders who fit waiting on one side of a platform `lengthM` long (T-219): 1,250 for 150 m. */
export function platformRoom(lengthM: number): number {
  return (Math.max(20, lengthM) * PLATFORM_SIDE_WIDTH_M) / PLATFORM_M2_PER_RIDER;
}

/** Seats, riders at the rated load (100%) and at crush (150%, the most a car takes) per car:
 * `SEATS_PER_CAR`, `RATED_PER_CAR` and `CRUSH_PER_CAR` in sim/src/demand/api.rs (T-2186). */
export const SEATS_PER_CAR = 44;
export const RATED_PER_CAR = 160;
export const CRUSH_PER_CAR = 240;
/** T-1464: a bus holds half a car's riders, 22 seated and 80 at crush (`BUS_CRUSH` in
 * sim/src/track/params.rs). */
export const BUS_CARS = 0.5;
/** A car's length, m. */
export const CAR_M = 20;

/** Seats, riders at the rated load and at crush of a train of `cars` cars, or of a bus
 * (`BUS_CARS`), which is full at its rated load (T-2186). */
export function capacity(cars: number): { seats: number; rated: number; crush: number } {
  return { seats: SEATS_PER_CAR * cars, rated: RATED_PER_CAR * cars, crush: (cars === BUS_CARS ? RATED_PER_CAR : CRUSH_PER_CAR) * cars };
}

/** The train gauge (T-2186): half full when every seat is taken, 1 at the rated load, then a
 * second band from 1 to 2 at crush; 2 = packed to crush. */
export function gauge(aboard: number, cars: number): number {
  const { seats, rated, crush } = capacity(cars);
  if (aboard <= seats) return seats > 0 ? (0.5 * Math.max(0, aboard)) / seats : 0;
  if (aboard <= rated) return 0.5 + (0.5 * (aboard - seats)) / (rated - seats);
  if (aboard < crush - 0.5 && crush > rated) return Math.min(1.99, 1 + (aboard - rated) / (crush - rated));
  return 2;
}

/** The platform lengths a player can set, m, a car apart. */
export const PLATFORM_MIN_M = 60;
export const PLATFORM_MAX_M = 400;
/** A platform length stepped `dir` cars longer (negative: shorter), kept within the limits. */
export const stepPlatform = (lengthM: number, dir: number): number =>
  Math.max(PLATFORM_MIN_M, Math.min(PLATFORM_MAX_M, lengthM + dir * CAR_M));

/** One platform of a station in the inspector: a line and direction. */
export interface PlatformNow {
  line: string;
  run: number;
  waiting: number;
  leftBehind: number;
  longestWait: number;
  arriving: number;
}

/** The platforms of station `node` now. */
export function platformsAt(h: BoardHour, node: number, t: number): PlatformNow[] {
  const out: PlatformNow[] = [];
  for (let q = 0; q < h.queueCount; q++) {
    const b = q * QUEUE_STRIDE;
    if (h.queues[b] !== node) continue;
    out.push({
      line: String(h.queues[b + 1]),
      run: h.queues[b + 2],
      waiting: crowdAt(h, q, t).waiting,
      leftBehind: h.queues[b + 5],
      longestWait: h.queues[b + 6],
      arriving: h.queues[b + 7],
    });
  }
  return out;
}

/** When the next train stops at any platform of station `node` after `t` (seconds from the hour's
 * epoch), or Infinity: the moment its crowd drops (T-238, T-1123). */
export function nextLeavingAt(h: BoardHour, node: number, t: number): number {
  let best = Infinity;
  const Q = h.queues;
  for (let q = 0; q < h.queueCount; q++) {
    const b = q * QUEUE_STRIDE;
    if (Q[b] !== node) continue;
    const first = Q[b + 8], end = first + Q[b + 9];
    for (let e = first; e < end; e++) {
      const s = stoppedAt(h, e);
      if (s > t) { best = Math.min(best, s); break; }
    }
  }
  return best;
}

/** Per trip of the hour, its train's stand at the terminal before its next departure (T-1785),
 * four numbers the train shader reads: the stand's start and end (s from the epoch, the platform
 * event's, so train and platform flow together), the next trip's first load in `h.loads` (-1 when
 * that trip is in a later hour), and then the riders the train leaves with by the platform (-1
 * none). All -1 where the trip has no next departure or its platform no event for it.
 * `nexts`: per trip, its train's next departure from the epoch (NaN none). */
export function trainStands(h: BoardHour, trips: Float32Array, nexts: Float32Array | null): Float32Array {
  const n = trips.length / 2, out = new Float32Array(Math.max(1, n) * 4).fill(-1);
  if (!nexts) return out;
  const Q = h.queues;
  // each line and run's first stop: the lowest offset along run 0 for run 0, the highest for run 1
  const firstStop = new Map<number, number>();
  for (let q = 0; q < h.queueCount; q++) {
    const b = q * QUEUE_STRIDE, key = Q[b + 1] * 2 + Q[b + 2], was = firstStop.get(key);
    if (was === undefined || (Q[b + 2] === 0 ? Q[b + 3] < Q[was * QUEUE_STRIDE + 3] : Q[b + 3] > Q[was * QUEUE_STRIDE + 3])) firstStop.set(key, q);
  }
  const tripAt = new Map<string, number>();
  for (let i = 0; i < n; i++) tripAt.set(`${trips[i * 2]}|${Math.round(trips[i * 2 + 1])}`, i);
  for (let i = 0; i < n; i++) {
    const next = nexts[i];
    if (!Number.isFinite(next)) continue;
    const prof = trips[i * 2], line = Math.floor(prof / 6), run = 1 - (Math.floor(prof / 3) % 2);
    const q = firstStop.get(line * 2 + run);
    if (q === undefined) continue;
    const b = q * QUEUE_STRIDE, first = Q[b + 8], end = first + Q[b + 9];
    let e = first;
    while (e < end && h.events[e * 2] < next - 0.5) e++;
    if (e >= end || h.events[e * 2] > next + 0.5) continue;
    const stop = stoppedAt(h, e);
    out[i * 4] = stop;
    out[i * 4 + 1] = h.events[e * 2];
    // the next trip: the other run's at any level (profile = line * 6 + run * 3 + level)
    for (let lv = 0; lv < 3; lv++) {
      const j = tripAt.get(`${line * 6 + run * 3 + lv}|${Math.round(next)}`);
      if (j !== undefined && h.offs[j * 2] >= 0) out[i * 4 + 2] = h.offs[j * 2];
    }
    out[i * 4 + 3] = Math.max(0, waitingAt(h, q, stop - 1e-3) + Q[b + 4] * (h.events[e * 2] - stop) + pulsesBetween(h, q, stop, h.events[e * 2]) - h.events[e * 2 + 1]);
  }
  return out;
}

/** Riders aboard trip `i` of the hour on stretch `s` (run 0's numbering), or null if the trip has
 * no account. */
export function tripLoad(h: BoardHour, i: number, s: number): number | null {
  const o = h.offs[i * 2];
  if (!(o >= 0)) return null;
  return h.loads[o + s] ?? null;
}
