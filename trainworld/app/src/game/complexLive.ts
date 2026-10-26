// A station complex now (T-246): the riders waiting on each of its platforms, from the boarding
// account the map's platform fills are drawn from (game/boarding.ts, T-164, T-219), and the trains
// standing at its platforms or pulling in. Read from the renderer's hour of the account and its trip
// buffer, the same numbers the map draws, so the complex view and the map agree.

import { clock } from "./clock";
import { crowdAt, platformRoom, QUEUE_STRIDE } from "./boarding";
import type { LineId, LineInput, StationId, StationInput } from "./types";
import { stationsById, world } from "./world";
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

/** A train standing at one of the complex's platforms, pulling in or pulling out (T-1096): the times
 * it stops and leaves, so the view moves it on every frame between workings-out. */
export interface TrainNow {
  /** its trip in the renderer's hour, with the station: who it is between workings-out */
  id: string;
  station: StationId;
  line: LineId;
  run: number;
  dir: number;
  /** game time it stops there (-Infinity: standing already) */
  arriveAt: number;
  /** game time it leaves (Infinity: not within the look ahead) */
  leaveAt: number;
  /** its trip ends here: it turns back rather than pulling out */
  ends: boolean;
  /** where it goes */
  to: StationId;
}

/** Where a train is at game time `t`: seconds until it stops (`inS`), since it left (`outS`), both 0
 * standing; null when more than PULLING_IN_S from the platform. */
export function trainPhase(x: TrainNow, t: number): { inS: number; outS: number } | null {
  if (t < x.arriveAt) return x.arriveAt - t > PULLING_IN_S ? null : { inS: x.arriveAt - t, outS: 0 };
  if (t < x.leaveAt) return { inS: 0, outS: 0 };
  return x.ends || t - x.leaveAt > PULLING_IN_S ? null : { inS: 0, outS: t - x.leaveAt };
}

export interface ComplexPlatform {
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
  platforms: Map<StationId, ComplexPlatform>;
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

/** The complex `members` now, or null before the boarding account has an hour drawn. Trains are
 * those within PULLING_IN_S of a platform now or within `lookS` game seconds, so the view has them
 * until it next works this out. */
export function complexNow(members: StationId[], lookS = 30): ComplexNow | null {
  const w = world.value, r = live.overlay?.renderer, net = r?.net;
  if (!w || !r || !net) return null;
  const inside = new Set(members);
  const byId = stationsById(w);
  const byNum = new Map(members.map((m) => [byId.get(m)?.num, m]));
  const lines = new Map(w.save.lines.filter((l) => !w.lineStats[l.id]?.bus && l.stops.some((st) => inside.has(st))).map((l) => [l.id, l]));
  const now = clock.now();
  const queues: QueueNow[] = [];
  const platforms = new Map<StationId, ComplexPlatform>();
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
  // trains: every trip of the hour on a line stopping here, standing at one of its stops here,
  // reaching one within PULLING_IN_S + lookS, or gone from one within PULLING_IN_S
  const trains: TrainNow[] = [];
  const trips = r.trips, ends = r.tripEnds, epoch = r.tripsEpoch;
  /** the first time in [lo, hi] at which `ok` holds, given it holds from then on (to 0.1 s) */
  const first = (lo: number, hi: number, ok: (t: number) => boolean) => {
    while (hi - lo > 0.1) {
      const mid = (lo + hi) / 2;
      if (ok(mid)) hi = mid; else lo = mid;
    }
    return hi;
  };
  for (let i = 0; i < trips.length / 2; i++) {
    const profile = trips[i * 2], line = String(Math.floor(profile / 6)), l = lines.get(line);
    if (!l) continue;
    const st = w.lineStats[line];
    if (!st) continue;
    const dep = trips[i * 2 + 1] + epoch, end = ends ? ends[i] + epoch : -Infinity;
    if (dep > now + lookS) continue;
    const run = Math.floor(profile / 3) % 2;
    const stops = run === 0 ? l.stops : [...l.stops].reverse(), S = st.stopS[run] ?? [];
    // how far along its run at time t: at its first stop before it sets off, past all once gone
    const sAt = (t: number) => tripAt(net, profile, dep, t, end)?.s ?? (t < dep ? S[0] ?? 0 : Infinity);
    const sNow = sAt(now);
    if (sNow === Infinity) continue;
    const gone = dep + net.meta[profile * 4 + 2];
    let found: TrainNow | null = null, left: TrainNow | null = null;
    for (let k = 0; k < stops.length && !found; k++) {
      if (!inside.has(stops[k]) || S[k] === undefined) continue;
      const kk = run === 0 ? k : l.stops.length - 1 - k, last = k === stops.length - 1;
      const train = (arriveAt: number, leaveAt: number): TrainNow => ({
        id: `${epoch}:${i}:${stops[k]}`, station: stops[k], line, run, dir: runDir(l, run, kk, byId.get(stops[k])!, byId),
        arriveAt, leaveAt, ends: last, to: stops[stops.length - 1],
      });
      const leaves = (from: number) => {
        if (last) return Math.max(end, gone);
        const hz = from + PULLING_IN_S + lookS;
        return sAt(hz) > S[k] + 2 ? first(from, hz, (t) => sAt(t) > S[k] + 2) : Infinity;
      };
      const gap = S[k] - sNow;
      if (Math.abs(gap) <= 2) found = train(-Infinity, leaves(now));
      else if (gap > 2) {
        const hz = now + PULLING_IN_S + lookS;
        if (sAt(hz) >= S[k] - 2) {
          const arrive = first(now, hz, (t) => sAt(t) >= S[k] - 2);
          found = train(arrive, leaves(arrive));
        }
        break;
      } else if (!last && sAt(now - PULLING_IN_S) <= S[k] + 2) {
        left = train(-Infinity, first(now - PULLING_IN_S, now, (t) => sAt(t) > S[k] + 2));
      }
    }
    if (found ?? left) trains.push((found ?? left)!);
  }
  // the trains' times are left out: the view moves them itself
  const key = queues.map((x) => Math.round(x.waiting) + (x.leftBehind ? "!" : "")).join(",") + "/" +
    trains.map((x) => x.id).join(",");
  return { queues, trains, platforms, key };
}
