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
  /** the stretches where trains run packed to crush (T-1125): line id and run 0 stretch, pairs */
  fullStretches: Float32Array;
  /** per full stretch, the share of the day's trains packed to crush there (T-1152); empty from a
   * WASM build before it */
  fullShares: Float32Array;
  /** per full stretch, the riders' average wait there for a train with room, s (T-1160); empty from
   * a WASM build before it */
  fullWaits: Float32Array;
}

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
  const fullStretches = b.subarray(i, (i += nf * 2));
  const fullShares = b.length >= i + nf ? b.subarray(i, i + nf) : new Float32Array(0);
  const fullWaits = b.length >= i + 2 * nf ? b.subarray(i + nf, i + 2 * nf) : new Float32Array(0);
  return { epoch, trips: nt, offs, loads, queues, queueCount: nq, events, pulses, arrive, pulseFrom, fullStretches, fullShares, fullWaits };
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

/** Riders waiting at platform `q` at `t` seconds from the hour's epoch: those the last train to
 * stop there left, plus, once it has gone, who has arrived since (T-164, T-1123: a train's riders
 * get on as it stops). 0 without events. */
export function waitingAt(h: BoardHour, q: number, t: number): number {
  const e = lastStopped(h, q, t);
  if (e < 0) return 0;
  const t0 = h.events[e * 2], left = h.events[e * 2 + 1];
  if (t < t0) return left;
  return Math.min(WAITING_CAP, left + (t - t0) * h.queues[q * QUEUE_STRIDE + 4] + pulsesBetween(h, q, t0, t));
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

/** Seats and riders at crush load per car: `SEATS_PER_CAR` and `CRUSH_PER_CAR` in
 * sim/src/demand/api.rs (a New York subway car). */
export const SEATS_PER_CAR = 44;
export const CRUSH_PER_CAR = 160;
/** A car's length, m. */
export const CAR_M = 20;

/** The train gauge: half full when every seat is taken, full at crush; 2 = over full. */
export function gauge(aboard: number, cars: number): number {
  const seats = SEATS_PER_CAR * cars, crush = CRUSH_PER_CAR * cars;
  if (aboard <= seats) return seats > 0 ? (0.5 * Math.max(0, aboard)) / seats : 0;
  if (aboard <= crush) return 0.5 + (0.5 * (aboard - seats)) / (crush - seats);
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

/** Riders aboard trip `i` of the hour on stretch `s` (run 0's numbering), or null if the trip has
 * no account. */
export function tripLoad(h: BoardHour, i: number, s: number): number | null {
  const o = h.offs[i * 2];
  if (!(o >= 0)) return null;
  return h.loads[o + s] ?? null;
}
