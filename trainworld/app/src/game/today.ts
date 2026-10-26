// The past 24 hours (T-446; before it, today so far, T-293: Anita wanted counts, not a forecast
// with the future in it, and then no reset at midnight). The line and station panels count the
// day up to now: riders carried, boardings at each stop, fares taken, trains that left full
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
import { moneyState } from "./money";
import { trainKind } from "./format";
import { PERIODS, periodAt } from "./periods";
import type { LineInput, LineStats } from "./types";
import { cameraMoving } from "./ui";

export interface LineToday {
  /** riders boarded in the past 24 hours, or yesterday (a change of line boards again), the line's own stops */
  riders: number;
  /** boardings at each stop, in the line's stop order */
  stops: number[];
  /** fares taken, US$ */
  fares: number;
  /** trains that left full, out of `trains` that ran (T-481), riders left behind */
  fullTrains: number;
  trains: number;
  leftBehind: number;
  /** T-472: trains in service now, and how many of them are packed full somewhere on their trip */
  trainsNow: number;
  fullNow: number;
  /** T-491: the period now: rider-seconds waited after full trains and riders arriving, at the
   * platforms where trains come full (T-1255), rider-seconds aboard; T-818: the first two times
   * the km those riders ride (0 since T-1255: every rider counts alike) */
  crowdWait: number;
  crowdRiders: number;
  crowdRide: number;
  crowdWaitKm: number;
  crowdRidersKm: number;
}

/** T-491, T-818: how much crowding costs a line's riders: their average extra wait for a train
 * with room (min), each rider counted by the fare they pay on the line (the base plus the per-km
 * rate times the km they ride; `waitKmS` and `ridersKm` are the waits and riders times those km),
 * so a long ride weighs more than a short hop. A rider a full platform turned away counts the
 * 30 minutes they would not wait (T-1255, the account's). The score is that wait (T-1255: no
 * longer against the ride, so the Trains to add count reads as the schedule's dots do). `weight` is
 * the fares the wait was averaged over, for averaging periods. */
export function crowding(waitS: number, riders: number, rideS: number, waitKmS = 0, ridersKm = 0): { wait: number; ride: number; score: number; weight: number } {
  if (!(riders > 0)) return { wait: 0, ride: 0, score: 0, weight: 0 };
  const f = moneyState.peek()?.fares;
  const paid = f ? f.base * riders + f.perKm * ridersKm : 0;
  const wait = (paid > 0 ? (f!.base * waitS + f!.perKm * waitKmS) / paid : waitS / riders) / 60, ride = rideS / riders / 60;
  return { wait, ride, score: wait, weight: paid > 0 ? paid : riders };
}

/** Grades of `crowding`'s score, minutes (T-1255, the colour scale's yellow, orange and red;
 * `CROWD_STOPS`): 0 none to speak of, 1 some, 2 orange, 3 red. */
export const CROWD_GRADES = [6, 9, 15];
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

/** US$ of running cost over the past 24 hours (T-446): each hour's trains at its demand level, the
 * hours at either end in part, as the worker drains the cash (T-284). */
export function runningToday(l: LineInput, st: LineStats): number {
  const m = moneyState.value;
  if (!m) return 0;
  // T-309: in the line's own local time
  const t = (clock.now() + st.shift) / 3600, t0 = t - 24;
  const tph = { high: l.tph.high, medium: l.tph.medium, low: l.tph.low };
  const k = trainKind(m, st.fast);
  let cost = 0;
  for (let h = Math.floor(t0); h < t; h++) {
    const lev = PERIODS[periodAt(h * 3600)].demand;
    cost += tph[lev] * 2 * st.km * (k.runCostTrainKm + st.cars * k.runCostCarKm) * m.economy *(Math.min(h + 1, t) - Math.max(h, t0));
  }
  return cost;
}

/** Per line id; empty until the first answer (and for a line with no riders yet). */
export const today = signal<Map<string, LineToday>>(new Map());
/** whether any line has counts: lists sorted by today's figures re-sort when it turns */
export const todayLoaded = computed(() => today.value.size > 0);

export function readToday(d: Float64Array): Map<string, LineToday> {
  const out = new Map<string, LineToday>();
  const count = d[0];
  let i = 1;
  const at: { id: string; t: LineToday }[] = [];
  for (let k = 0; k < count; k++) {
    // d[i + 5] is the fullest train and d[i + 7 ..] the spans of full trains, which nothing shows
    const id = d[i], n = d[i + 1], nSpans = d[i + 6];
    const stops = Array.from(d.subarray(i + 7 + 2 * nSpans, i + 7 + 2 * nSpans + n));
    const e = i + 7 + 2 * nSpans + n;
    const t: LineToday = { riders: d[i + 2], fullTrains: d[i + 3], leftBehind: d[i + 4], stops, fares: 0, trainsNow: d[e], fullNow: d[e + 1], trains: d[e + 2], crowdWait: d[e + 3], crowdRiders: d[e + 4], crowdRide: d[e + 5], crowdWaitKm: d[e + 6], crowdRidersKm: d[e + 7] };
    at.push({ id: String(id), t });
    i = e + 8;
  }
  at.forEach((a, k) => { a.t.fares = (d[i + k] ?? 0) * 1e6; out.set(a.id, a.t); });
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
  // T-1207: the Lines tab's rows re-rendered four times a second while the map was dragged
  if (busy || (!force && cameraMoving.now)) return;
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
    const w = world.peek();
    readFor = new Map((w?.save.lines ?? []).map((l) => [l.id, l.stops]));
    const next = readToday(d), prev = today.peek();
    // A line's account is worked out again a while after its stops change (new riders, T-164);
    // until then its counts may be per the old stops. The ones moved to the new stops stand meanwhile.
    for (const l of w?.save.lines ?? []) {
      const t = next.get(l.id), p = prev.get(l.id);
      if (t && t.stops.length !== l.stops.length && p?.stops.length === l.stops.length) next.set(l.id, { ...t, stops: p.stops });
    }
    today.value = next;
  });
}

/** the stops each line's counts are in the order of: the world's when they were read */
let readFor = new Map<string, string[]>();

/** T-648: a stop edit renumbers the line's stops, and the counts (until the next answer) would sit
 * one row off: they follow their stations to the new order now, a new stop at 0. */
world.subscribe((w) => {
  const t = today.peek();
  if (!w || !t.size) return;
  let out: Map<string, LineToday> | null = null;
  for (const l of w.save.lines) {
    const d = t.get(l.id), was = readFor.get(l.id);
    if (!d || !was || (was.length === l.stops.length && was.every((s, k) => s === l.stops[k]))) continue;
    const used = was.map(() => false);
    const stops = l.stops.map((s) => {
      const k = was.findIndex((x, j) => !used[j] && x === s);
      if (k < 0) return 0;
      used[k] = true;
      return d.stops[k] ?? 0;
    });
    (out ??= new Map(t)).set(l.id, { ...d, stops });
    readFor.set(l.id, l.stops);
  }
  if (out) {
    today.value = out;
    if (users > 0) ask(); // the new stop's count, and a line that just lost its service, at once
  }
});

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
