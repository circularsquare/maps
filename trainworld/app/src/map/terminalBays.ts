// T-2980 (notes/T-2974.md 5): where a train stands at the end of its run. A double-track dead end
// or a through station with a terminal (T-1281) turns on both platforms, and the capacity sums count
// both, so the drawing does too: each train arriving takes the platform there that frees first, and
// stands on it whole (T-3053: nothing is drawn past a line's last station). A dead end without a terminal turns on one platform (T-2986): the train ahead
// holds it, and the next waits short of the crossover until it leaves. The pick is worked
// out in the clock worker from the timetable (an hour of trips before the window as well), so every
// window draws the same; the renderer moves the train across to it (`TRAIN_VS`), and the train guard
// holds one whose place is still taken.
import type { ClockEdge, ClockLine, ClockNode } from "../workers/protocol";
import { hasUpgrade, UPGRADE } from "../game/upgrades";

/** Length of a turn-back crossover's diagonal, m (T-658d). */
export const TURNBACK_M = 50;
/** T-3202: a station on single track is drawn as two tracks along its platform (network.ts
 * `singleSplits`), parting from the single track over this far either side, m. */
export const SINGLE_SPLIT_LEAD_M = 60;
/** Places at a line's end: the arriving train's own track, or the other track (it crosses the
 * scissors on the way in). */
export const BAY_OWN = 0, BAY_OTHER = 1;
/** A trip's place distance (the 2nd and 4th number of its `BAY_STRIDE`) at a through station is
 * written as `-p - 1`, so below 0 marks an end the track runs on past, where the train stands with
 * its middle at the stop and may be drawn that far over the line's end (T-3053). */
export const bayThru = (v: number) => v < 0;
/** The metres a place distance stands for. */
export const bayP = (v: number) => (v < 0 ? -v - 1 : v);
/** The nearest a train's middle may be to a run's end (T-3053): half its length, less the half
 * platform of a through end it may overhang. */
export const endClear = (v: number, trainM: number) => (v < 0 ? Math.max(0, trainM / 2 + v + 1) : trainM / 2);
/** Floats per trip in `assignBays`'s result: where it starts (code + 4 x (group + 1), -1 for no
 * place), the metres from its first stop to where it crosses over, where it ends (the same), and
 * the metres from its last stop to the crossing (each of the two distances `bayThru` at a through end). */
export const BAY_STRIDE = 4;
/** A place is free once the train leaving it is this far past the platform's end, m (its own length,
 * a long train's, and the crossover), or, where its next trip is not known, this long after it
 * leaves, s. */
const CLEAR_M = 200 + TURNBACK_M, CLEAR_S = 60;
/** How far before the window the pick starts, s: past the longest a train stands at an end (an hour,
 * `HANDOVER_MAX_S`), so a train standing as the window opens has the place it was given before. */
export const BAY_LOOKBACK_S = 3600;

/** A line end's places: `group` is shared by the lines arriving there along the same track. */
export interface TerminalEnd {
  group: number;
  /** 0 one place at a through station (no pick), 1 two platforms, 3 a dead end's one platform (no
   * terminal, T-2986: the arrival track's, held as a place) */
  kind: 0 | 1 | 3;
  /** m from the stop to the crossover (a dead end's platform, half a through station's) */
  p: number;
  /** the track runs on past the stop, which stands mid-platform */
  thru: boolean;
  /** a train turning on a through platform with no terminal leaves over the crossover behind it */
  cross: boolean;
  /** T-3202: a station on single track, its two places the sides it is drawn split into: each
   * arriving train takes the one free longest, so the one it meets there is on the other */
  single?: boolean;
}

/**
 * Per line, the end of each run (run 0 ends at its last stop, run 1 at its first), null where
 * nothing is drawn differently (a bus, no platform, a loop).
 */
export function terminalEnds(edges: readonly ClockEdge[], nodes: readonly ClockNode[], lines: readonly ClockLine[]): Map<number, [TerminalEnd | null, TerminalEnd | null]> {
  const endCount = new Map<number, number>(), edgeOf = new Map<number, ClockEdge>();
  for (const e of edges) {
    endCount.set(e.a, (endCount.get(e.a) ?? 0) + 1).set(e.b, (endCount.get(e.b) ?? 0) + 1);
    edgeOf.set(e.id, e);
  }
  const nodeOf = new Map<number, ClockNode>();
  for (const n of nodes) nodeOf.set(n.id, n);
  const groups = new Map<string, number>();
  const out = new Map<number, [TerminalEnd | null, TerminalEnd | null]>();
  for (const l of lines) {
    const last = l.stops.length - 1;
    if (!l.ok || last < 1 || !l.path.length || l.stops[0] === l.stops[last]) continue;
    const at = (run: 0 | 1): TerminalEnd | null => {
      const node = nodeOf.get(run === 0 ? l.stops[last] : l.stops[0]);
      const entry = run === 0 ? l.path[l.path.length - 1] : l.path[0];
      const edge = edgeOf.get(entry >> 1);
      if (!node || !node.platform || !edge) return null;
      const dead = endCount.get(node.id) === 1;
      if (edge.tracks !== 2 && (edge.tracks !== 1 || l.bus)) return dead ? null : { group: -1, kind: 0, p: node.platform / 2, thru: true, cross: false };
      const key = `${node.id}|${edge.id}`;
      let group = groups.get(key);
      if (group === undefined) groups.set(key, (group = groups.size));
      // (the crossover past where the split joins the single track, so a train for the other side
      // eases straight across to it)
      if (edge.tracks === 1) return { group, kind: 1, p: (dead ? node.platform : node.platform / 2) + SINGLE_SPLIT_LEAD_M, thru: !dead, cross: false, single: true };
      const terminal = hasUpgrade(node.upgrades, UPGRADE.terminal);
      // (both platforms of a dead end only with a terminal, as the sim's `dead_end_places`)
      if (dead) return { group, kind: terminal ? 1 : 3, p: node.platform, thru: false, cross: false };
      if (terminal) return { group, kind: 1, p: node.platform / 2, thru: true, cross: false };
      return { group, kind: 0, p: node.platform / 2, thru: true, cross: true };
    };
    const ends: [TerminalEnd | null, TerminalEnd | null] = [at(0), at(1)];
    if (ends[0] || ends[1]) out.set(l.id, ends);
  }
  return out;
}

/**
 * Per profile of `lines`: seconds into its trip when it reaches its last stop (its turnaround and
 * any wait are stood after that, inside the trip), and a function of the seconds into it when it is
 * `m` metres along its run (holds on the way included). `meta` is `TrackApi.profile_meta`,
 * `phases(line)` `TrackApi.line_phases`.
 */
export function profileTimes(meta: Float32Array, lines: Iterable<number>, phases: (line: number) => Float32Array): Map<number, ProfileTimes> {
  const out = new Map<number, ProfileTimes>();
  for (const l of lines) {
    const ph = phases(l), base = meta[l * 6 * 4];
    for (let p = l * 6; p < l * 6 + 6; p++) {
      const first = meta[p * 4] - base, count = meta[p * 4 + 1];
      if (!count || (first + count) * 4 > ph.length) continue;
      const k = (first + count - 1) * 4, a = ph[k + 3];
      const arrive = a < 0 ? ph[k] + ph[k + 2] / -a : ph[k];
      const at = (m: number) => {
        // the phase (t, s, v, a) in which it passes m
        for (let j = first; j < first + count; j++) {
          const t = ph[j * 4], s = ph[j * 4 + 1], v = ph[j * 4 + 2], acc = ph[j * 4 + 3];
          const end = j + 1 < first + count ? ph[(j + 1) * 4 + 1] : Infinity;
          if (end < m || (v === 0 && acc === 0)) continue;
          const d = m - s;
          if (d <= 0) return t;
          if (Math.abs(acc) < 1e-9) return t + d / v;
          const disc = v * v + 2 * acc * d;
          return disc < 0 ? t : t + (-v + Math.sqrt(disc)) / acc;
        }
        return arrive;
      };
      out.set(p, { arrive, at });
    }
  }
  return out;
}

/** `profileTimes` of one profile. */
export interface ProfileTimes { arrive: number; at: (m: number) => number }

/** One trip as the worker's `TrackApi.trips` gives it, in absolute game seconds. */
interface Trip { profile: number; draw: number; end: number; next: number; i: number }

/**
 * Each trip's place at its start and its end (`BAY_STRIDE` per trip of `cur`, the trips of the hour
 * from `epoch`: profile, departure, end and next departure relative to the epoch, as `TrackApi.trips`).
 * `prev` is the hour before's, from `epoch - BAY_LOOKBACK_S`, so the pick is the same from any window.
 * `times(profile)`: when it reaches its last stop and where it is when (`profileTimes`).
 */
export function assignBays(cur: Float32Array, epoch: number, prev: Float32Array | null, ends: Map<number, [TerminalEnd | null, TerminalEnd | null]>, times: (profile: number) => ProfileTimes | undefined): Float32Array {
  const n = cur.length / 4;
  const out = new Float32Array(n * BAY_STRIDE).fill(-1);
  for (let i = 0; i < n; i++) out[i * BAY_STRIDE + 1] = out[i * BAY_STRIDE + 3] = 0;
  if (!ends.size) return out;
  const key = (profile: number, draw: number) => `${profile}|${Math.round(draw * 100)}`;
  const trips: Trip[] = [];
  const byKey = new Map<string, Trip>();
  const add = (raw: Float32Array, e: number, own: boolean) => {
    for (let k = 0; k < raw.length / 4; k++) {
      const profile = raw[k * 4], line = Math.floor(profile / 6);
      if (!ends.has(line)) continue;
      const t: Trip = { profile, draw: raw[k * 4 + 1] + e, end: raw[k * 4 + 2] + e, next: raw[k * 4 + 3] + e, i: own ? k : -1 };
      const kk = key(profile, t.draw);
      const was = byKey.get(kk);
      if (was) {
        if (own) was.i = k;
        continue;
      }
      byKey.set(kk, t);
      trips.push(t);
    }
  };
  if (prev) add(prev, epoch - BAY_LOOKBACK_S, false);
  add(cur, epoch, true);
  // per group, by arrival: each train takes the place that frees first, its own track on a tie
  const endOf = (t: Trip) => ends.get(Math.floor(t.profile / 6))![Math.floor(t.profile / 3) % 2];
  // a trip by line, run and departure (its next trip can be of another level)
  const runKey = (line: number, run: number, draw: number) => `${line}|${run}|${Math.round(draw * 100)}`;
  const byRun = new Map<string, Trip>();
  for (const t of trips) byRun.set(runKey(Math.floor(t.profile / 6), Math.floor(t.profile / 3) % 2, t.draw), t);
  const nextOf = (t: Trip) => (Number.isFinite(t.next) ? byRun.get(runKey(Math.floor(t.profile / 6), 1 - (Math.floor(t.profile / 3) % 2), t.next)) : undefined);
  const arrivals = trips.map((t) => ({ t, at: t.draw + (times(t.profile)?.arrive ?? 0) })).sort((a, b) => a.at - b.at || a.t.profile - b.t.profile);
  const free = new Map<number, [number, number]>();
  const endCode = new Map<Trip, number>();
  for (const { t, at } of arrivals) {
    const end = endOf(t);
    if (!end || end.kind === 0) continue;
    let f = free.get(end.group);
    if (!f) free.set(end.group, (f = [-Infinity, -Infinity]));
    // (wanted from when it comes up to the crossover, some 30 s before it stops)
    const k = end.kind === 3 || (!end.single && f[0] <= at - 30) ? 0 : !end.single && f[1] <= at - 30 ? 1 : f[0] <= f[1] ? 0 : 1;
    // taken until its train on its next trip is off the platform (a hold there first included)
    const nt = nextOf(t), nx = nt && times(nt.profile);
    f[k] = Math.max(at, nt && nx ? nt.draw + nx.at(end.p + CLEAR_M) : t.end + CLEAR_S);
    endCode.set(t, BAY_OWN + k);
  }
  for (const t of trips) {
    const line = Math.floor(t.profile / 6), run = Math.floor(t.profile / 3) % 2;
    const end = ends.get(line)![run];
    if (!end) continue;
    const code = endCode.get(t);
    const pEnc = end.thru ? -end.p - 1 : end.p;
    if (t.i >= 0) {
      if (code !== undefined) out[t.i * BAY_STRIDE + 2] = code + 4 * (end.group + 1);
      out[t.i * BAY_STRIDE + 3] = pEnc;
    }
    // its train's next trip leaves from where it stood: the arriving track is that trip's other one
    const nt = nextOf(t);
    if (!nt || nt.i < 0) continue;
    out[nt.i * BAY_STRIDE + 1] = pEnc;
    const start = code === undefined ? (end.cross ? BAY_OTHER : -1) : code === BAY_OWN ? BAY_OTHER : BAY_OWN;
    if (start < 0) continue;
    // (the group goes with it: the guard keeps the platform it leaves from taken until it is clear)
    out[nt.i * BAY_STRIDE] = start + (code !== undefined ? 4 * (end.group + 1) : 0);
  }
  return out;
}

/** A place code's own part (0-3), -1 for none. */
export const bayCode = (v: number) => (v < 0 ? -1 : v % 4);
/** A place code's group, -1 for none. */
export const bayGroup = (v: number) => (v < 4 ? -1 : Math.floor(v / 4) - 1);

const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** The train shader's `acrossAt`: how far across to its own track the point `x` m along run 0 is
 * drawn, for a trip with places `bay` (BAY_STRIDE numbers from `o`) on run `run` of length `len`. */
export function acrossAt(bay: ArrayLike<number>, o: number, run: number, len: number, x: number, trainM: number): number {
  const q = run === 0 ? len - x : x, r = run === 0 ? x : len - x;
  const sc = bayCode(bay[o]), ec = bayCode(bay[o + 2]);
  if (r < q) {
    if (sc !== BAY_OTHER) return 1;
    const p = Math.max(bayP(bay[o + 1]), trainM + 5);
    return -1 + 2 * smooth(p, p + TURNBACK_M, r);
  }
  if (ec === BAY_OTHER) {
    const p = Math.max(bayP(bay[o + 3]), trainM + 5);
    return -1 + 2 * smooth(p, p + TURNBACK_M, q);
  }
  return 1;
}
