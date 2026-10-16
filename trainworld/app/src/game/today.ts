// The past 24 hours (T-446; before it, today so far, T-293: Anita wanted counts, not a forecast
// with the future in it, and then no reset at midnight). The line and station panels count the
// day up to now: riders carried, boardings at each stop, fares taken, trains that left packed full
// and riders they left behind. Counted by the clock worker from the boarding account's departures
// (`TrackApi.past_day`, sim/src/track/boarding.rs) and each line's fares as the worker pays them
// (T-284); asked again four times a wall second while a panel shows them and the clock runs, as
// the cash moves. The names say "today" from T-293.

import { computed, signal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import { clock } from "./clock";
import { world } from "./world";
import { queryClock } from "../workers/clockClient";
import { accountVersion } from "./boarding";

export interface LineToday {
  /** riders boarded in the past 24 hours, or yesterday (a change of line boards again), the line's own stops */
  riders: number;
  /** boardings at each stop, in the line's stop order */
  stops: number[];
  /** fares taken, US$ */
  fares: number;
  /** trains that left packed full, out of `trains` that ran (T-481), the fullest one's share of
   * crush, riders left behind */
  fullTrains: number;
  trains: number;
  fullest: number;
  leftBehind: number;
  /** spans of full trains (s from the line's local midnight; negative for yesterday's) */
  spans: [number, number][];
  /** running cost of the trains over the day, US$; only for a whole day (yesterday's) */
  running: number;
  /** T-472: trains in service now, and how many of them are packed full somewhere on their trip */
  trainsNow: number;
  fullNow: number;
  /** T-491: the period now: rider-seconds waited after full trains, riders arriving, rider-seconds aboard */
  crowdWait: number;
  crowdRiders: number;
  crowdRide: number;
}

/** T-491: how much crowding costs a line's riders: the average rider's extra wait for a train
 * with room (min), and that wait against the square root of their average ride (min, at least 5),
 * so ten minutes weigh more on a short ride than on a long one. */
export function crowding(waitS: number, riders: number, rideS: number): { wait: number; ride: number; score: number } {
  if (!(riders > 0)) return { wait: 0, ride: 0, score: 0 };
  const wait = waitS / riders / 60, ride = rideS / riders / 60;
  return { wait, ride, score: wait / Math.sqrt(Math.max(5, ride)) };
}

/** T-491 grades of `crowding`'s score (DECISIONS 2026-10-16): 0 none to speak of, 1 some, 2 orange,
 * 3 red. */
export const CROWD_GRADES = [0.1, 0.3, 0.6];
export function crowdGrade(score: number): 0 | 1 | 2 | 3 {
  return score >= CROWD_GRADES[2] ? 3 : score >= CROWD_GRADES[1] ? 2 : score >= CROWD_GRADES[0] ? 1 : 0;
}
/** Text the player should act on (T-472): orange and red grades. */
export function crowdClass(score: number): "" | "crowd-warn" | "crowd-bad" {
  const g = crowdGrade(score);
  return g === 3 ? "crowd-bad" : g === 2 ? "crowd-warn" : "";
}
/** "2.3 min", or "under 0.1 min" */
export const waitText = (min: number) => (min < 0.05 ? "under 0.1 min" : `${min < 10 ? min.toFixed(1) : Math.round(min)} min`);

/** Per line id; empty until the first answer (and for a line with no riders yet). */
export const today = signal<Map<string, LineToday>>(new Map());
/** each line's last complete day (T-408), asked for while a panel shows it */
export const yesterday = signal<Map<string, LineToday>>(new Map());
/** whether any line has counts: lists sorted by today's figures re-sort when it turns */
export const todayLoaded = computed(() => today.value.size > 0);

export function readToday(d: Float64Array): Map<string, LineToday> {
  const out = new Map<string, LineToday>();
  const count = d[0];
  let i = 1;
  const at: { id: string; t: LineToday }[] = [];
  for (let k = 0; k < count; k++) {
    const id = d[i], n = d[i + 1], nSpans = d[i + 6];
    const spans: [number, number][] = [];
    for (let s = 0; s < nSpans; s++) spans.push([d[i + 7 + 2 * s], d[i + 8 + 2 * s]]);
    const stops = Array.from(d.subarray(i + 7 + 2 * nSpans, i + 7 + 2 * nSpans + n));
    const e = i + 7 + 2 * nSpans + n;
    const t: LineToday = { riders: d[i + 2], fullTrains: d[i + 3], leftBehind: d[i + 4], fullest: d[i + 5], spans, stops, fares: 0, running: 0, trainsNow: d[e], fullNow: d[e + 1], trains: d[e + 2], crowdWait: d[e + 3], crowdRiders: d[e + 4], crowdRide: d[e + 5] };
    at.push({ id: String(id), t });
    i = e + 6;
  }
  at.forEach((a, k) => { a.t.fares = (d[i + k] ?? 0) * 1e6; a.t.running = d[i + at.length + k] ?? 0; out.set(a.id, a.t); });
  return out;
}

/** Boardings in the past 24 hours per station id (stations with a counted line), summed once per answer. */
export const stationsToday = computed(() => {
  const out = new Map<string, number>();
  const t = today.value, w = world.peek();
  if (!w) return out;
  for (const l of w.save.lines) {
    const d = t.get(l.id);
    if (d) l.stops.forEach((s, k) => out.set(s, (out.get(s) ?? 0) + (d.stops[k] ?? 0)));
  }
  return out;
});

const ASK_MS = 250;
let users = 0, timer = 0, busy = false, askedAt = NaN, askedVersion = -1;

function ask(force = false) {
  if (busy) return;
  // the network or the account (new riders, the hour turning) changed
  const t = clock.now(), v = (world.peek()?.version ?? -1) * 1e6 + accountVersion.peek();
  // nothing moves while the clock stands, unless the network or the account changed
  if (!force && t === askedAt && v === askedVersion) return;
  busy = true;
  askedAt = t;
  askedVersion = v;
  void queryClock({ what: "today" }).then((d) => {
    busy = false;
    if (!d) return;
    today.value = readToday(d);
  });
}

/** Ask for yesterday's figures while the calling panel shows them (T-408): again when the day turns
 * or the network or the riders change. */
export function useYesterday(on: boolean) {
  const key = (world.value?.version ?? -1) * 1e6 + accountVersion.value;
  const day = Math.floor(clock.local.value / 1440);
  useEffect(() => {
    if (!on) return;
    let live = true;
    void queryClock({ what: "yesterday" }).then((d) => { if (live && d) yesterday.value = readToday(d); });
    return () => { live = false; };
  }, [on, key, day]);
}

/** Keep `today` fresh while the calling panel is shown. */
export function useToday() {
  useEffect(() => {
    if (users++ === 0) {
      ask(true);
      timer = setInterval(ask, ASK_MS) as unknown as number;
    }
    return () => {
      if (--users === 0) clearInterval(timer);
    };
  }, []);
}
