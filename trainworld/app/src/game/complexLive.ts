// A station complex now (T-246): the riders waiting on each of its platforms, from the boarding
// account the map's platform fills are drawn from (game/boarding.ts, T-164, T-219), and the trains
// standing at its platforms or pulling in. Read from the renderer's hour of the account and its trip
// buffer, the same numbers the map draws, so the complex view and the map agree.

import { clock } from "./clock";
import { crowdAt, platformRoom, QUEUE_STRIDE } from "./boarding";
import type { LineId, LineInput, StationId, StationInput } from "./types";
import { world } from "./world";
import { live } from "../map/live";
import { tripAt } from "../map/network";

/** A train this many game seconds from a platform or nearer is drawn pulling in. */
export const PULLING_IN_S = 60;

/** One line and direction's riders on one platform now. */
export interface QueueNow {
  station: StationId;
  line: LineId;
  run: number;
  /** +1: its trains run along the station's heading, -1 against it */
  dir: number;
  waiting: number;
  /** the last train to leave left riders behind */
  leftBehind: boolean;
}

/** A train standing at one of the complex's platforms, or pulling in. */
export interface TrainNow {
  station: StationId;
  line: LineId;
  run: number;
  dir: number;
  /** game seconds until it stops there; 0 standing */
  inS: number;
  /** where it goes */
  to: StationId;
}

export interface PlatformNow {
  /** riders on its + side (trains along the heading) and - side */
  side: [number, number];
  /** room to stand on a side (both sides' room when trains use one side only, as the map draws it) */
  room: number;
  /** trains use one side only: the platform fills as one */
  oneSide: boolean;
  leftBehind: [boolean, boolean];
}

export interface ComplexNow {
  queues: QueueNow[];
  trains: TrainNow[];
  platforms: Map<StationId, PlatformNow>;
  /** changes whenever anything drawn from it does */
  key: string;
}

/** Which way line `l`'s run `run` passes station `s` (its stop `k`): +1 along the station's heading,
 * -1 against it, from where the previous stop (or the next, at the first) lies. */
function runDir(l: LineInput, run: number, k: number, s: StationInput, byId: Map<StationId, StationInput>): number {
  const stops = l.stops, n = stops.length;
  // run 0 travels stop 0 to n - 1, run 1 back
  const prev = run === 0 ? k - 1 : k + 1, next = run === 0 ? k + 1 : k - 1;
  const ux = Math.cos(s.heading), uy = Math.sin(s.heading);
  const along = (j: number) => {
    const o = j >= 0 && j < n ? byId.get(stops[j]) : undefined;
    return o ? (o.x - s.x) * ux + (o.y - s.y) * uy : NaN;
  };
  const p = along(prev);
  if (Number.isFinite(p) && Math.abs(p) > 1) return p < 0 ? 1 : -1;
  const q = along(next);
  if (Number.isFinite(q) && Math.abs(q) > 1) return q > 0 ? 1 : -1;
  return run === 0 ? 1 : -1;
}

/** The complex `members` now, or null before the boarding account has an hour drawn. */
export function complexNow(members: StationId[]): ComplexNow | null {
  const w = world.value, r = live.overlay?.renderer, net = r?.net;
  if (!w || !r || !net) return null;
  const inside = new Set(members);
  const byId = new Map(w.save.stations.map((s) => [s.id, s]));
  const byNum = new Map(members.map((m) => [byId.get(m)?.num, m]));
  const lines = new Map(w.save.lines.filter((l) => l.stops.some((st) => inside.has(st))).map((l) => [l.id, l]));
  const now = clock.now();
  const queues: QueueNow[] = [];
  const platforms = new Map<StationId, PlatformNow>();
  const b = r.board;
  if (b) {
    const t = now - r.tripsEpoch;
    for (let q = 0; q < b.queueCount; q++) {
      const base = q * QUEUE_STRIDE;
      const station = byNum.get(b.queues[base]);
      if (station === undefined) continue;
      const line = String(b.queues[base + 1]), run = b.queues[base + 2], l = lines.get(line), s = byId.get(station);
      if (!l || !s) continue;
      // a line stopping here twice (T-797): the visit at the queue's offset along run 0
      const offs = w.lineStats[line]?.stopS[0] ?? [];
      let k = l.stops.indexOf(station);
      for (let j = k + 1; j < l.stops.length; j++)
        if (l.stops[j] === station && Math.abs((offs[j] ?? Infinity) - b.queues[base + 3]) < Math.abs((offs[k] ?? Infinity) - b.queues[base + 3])) k = j;
      const c = crowdAt(b, q, t);
      queues.push({ station, line, run, dir: runDir(l, run, k, s, byId), waiting: c.waiting, leftBehind: c.leftBehind });
    }
    for (const m of members) {
      const qs = queues.filter((x) => x.station === m);
      if (!qs.length) continue;
      const side: [number, number] = [0, 0], bad: [boolean, boolean] = [false, false];
      let sides = 0;
      for (const x of qs) {
        const i = x.dir > 0 ? 0 : 1;
        side[i] += x.waiting;
        bad[i] ||= x.leftBehind;
        sides |= 1 << i;
      }
      const room = platformRoom(byId.get(m)?.length ?? 0);
      const oneSide = sides !== 3;
      platforms.set(m, { side, room: oneSide ? 2 * room : room, oneSide, leftBehind: oneSide ? [bad[0] || bad[1], bad[0] || bad[1]] : bad });
    }
  }
  // trains: every trip of the hour on a line stopping here, standing at one of its stops here or
  // reaching one within PULLING_IN_S
  const trains: TrainNow[] = [];
  const trips = r.trips, ends = r.tripEnds, epoch = r.tripsEpoch;
  for (let i = 0; i < trips.length / 2; i++) {
    const profile = trips[i * 2], line = String(Math.floor(profile / 6)), l = lines.get(line);
    if (!l) continue;
    const st = w.lineStats[line];
    if (!st) continue;
    const dep = trips[i * 2 + 1] + epoch, end = ends ? ends[i] + epoch : -Infinity;
    const at = tripAt(net, profile, dep, now, end);
    if (!at) continue;
    const stops = at.run === 0 ? l.stops : [...l.stops].reverse(), S = st.stopS[at.run] ?? [];
    for (let k = 0; k < stops.length; k++) {
      if (!inside.has(stops[k]) || S[k] === undefined) continue;
      const gap = S[k] - at.s;
      let inS = -1;
      if (Math.abs(gap) <= 2) inS = 0;
      else if (gap > 2) {
        const later = tripAt(net, profile, dep, now + PULLING_IN_S, end);
        if (later && later.s >= S[k] - 2) {
          // when it reaches the stop: halve the minute until within a couple of seconds
          let lo = 0, hi = PULLING_IN_S;
          for (let it = 0; it < 6; it++) {
            const mid = (lo + hi) / 2, p = tripAt(net, profile, dep, now + mid, end);
            if (p && p.s >= S[k] - 2) hi = mid;
            else lo = mid;
          }
          inS = hi;
        }
      }
      if (inS < 0) continue;
      const kk = at.run === 0 ? k : l.stops.length - 1 - k;
      const s = byId.get(stops[k])!;
      trains.push({ station: stops[k], line, run: at.run, dir: runDir(l, at.run, kk, s, byId), inS, to: stops[stops.length - 1] });
      break;
    }
  }
  const key = queues.map((x) => Math.round(x.waiting) + (x.leftBehind ? "!" : "")).join(",") + "/" +
    trains.map((x) => `${x.station}:${x.line}:${x.run}:${Math.ceil(x.inS / 5)}`).join(",");
  return { queues, trains, platforms, key };
}
