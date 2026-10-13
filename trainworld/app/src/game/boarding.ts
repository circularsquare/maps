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
export const QUEUE_STRIDE = 10;

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

/** One game hour of the account, as views into the worker's array. */
export interface BoardHour {
  epoch: number;
  trips: number;
  /** per trip: index of its first load or -1, seconds from departure to its last stop */
  offs: Float32Array;
  /** riders aboard per stop-to-stop stretch, numbered as on run 0 */
  loads: Float32Array;
  /** per platform QUEUE_STRIDE numbers: node, line, run, offset along run 0 (m), riders arriving
   * a second at mid-hour, riders left behind this hour, longest wait this hour (s), riders arriving
   * this hour, first event, events */
  queues: Float32Array;
  queueCount: number;
  /** per train leaving a platform: seconds from `epoch`, riders it left there */
  events: Float32Array;
}

export function readBoardHour(epoch: number, b: Float32Array): BoardHour {
  const [nt, nl, nq, ne] = [b[0], b[1], b[2], b[3]];
  let i = 4;
  const offs = b.subarray(i, (i += 2 * nt));
  const loads = b.subarray(i, (i += nl));
  const queues = b.subarray(i, (i += nq * QUEUE_STRIDE));
  const events = b.subarray(i, i + ne * 2);
  return { epoch, trips: nt, offs, loads, queues, queueCount: nq, events };
}

/** Riders waiting at platform `q` at `t` seconds from the hour's epoch, whether the last train
 * to leave it left anyone behind, and how many it left (T-284). */
export function crowdAt(h: BoardHour, q: number, t: number): { waiting: number; leftBehind: boolean; left: number } {
  const Q = h.queues, E = h.events;
  const base = q * QUEUE_STRIDE;
  const first = Q[base + 8], count = Q[base + 9];
  let lo = first, hi = first + count - 1;
  if (count <= 0) return { waiting: 0, leftBehind: false, left: 0 };
  // the last train to leave at or before t
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (E[mid * 2] <= t) lo = mid;
    else hi = mid - 1;
  }
  const t0 = E[lo * 2], left = E[lo * 2 + 1];
  return { waiting: left + Math.max(0, t - t0) * Q[base + 4], leftBehind: left >= 1, left };
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

/** When the next train leaves any platform of station `node` after `t` (seconds from the hour's
 * epoch), or Infinity: the moment its crowd drops (T-238). */
export function nextLeavingAt(h: BoardHour, node: number, t: number): number {
  let best = Infinity;
  const Q = h.queues, E = h.events;
  for (let q = 0; q < h.queueCount; q++) {
    const b = q * QUEUE_STRIDE;
    if (Q[b] !== node) continue;
    const first = Q[b + 8], end = first + Q[b + 9];
    for (let e = first; e < end; e++) {
      if (E[e * 2] > t) { best = Math.min(best, E[e * 2]); break; }
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
