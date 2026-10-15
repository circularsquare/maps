// Today so far (T-293, Anita: "lets not even show the whole day forecast cuz it shows stuff in
// the future. lets only show so far today, and call it 'today'"). The line and station panels
// count the day from midnight to now: riders carried, boardings at each stop, fares taken, trains
// that left packed full and riders they left behind. Counted by the clock worker from the boarding
// account's departures up to now (`TrackApi.today`, sim/src/track/boarding.rs) and each line's
// fares as the worker pays them (T-284); asked again four times a wall second while a panel shows
// them and the clock runs, as the cash moves.

import { computed, signal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import { clock } from "./clock";
import { world } from "./world";
import { queryClock } from "../workers/clockClient";
import { accountVersion } from "./boarding";

export interface LineToday {
  /** riders boarded since midnight (a change of line boards again), the line's own stops */
  riders: number;
  /** boardings at each stop, in the line's stop order */
  stops: number[];
  /** fares taken, US$ */
  fares: number;
  /** trains that left packed full, the fullest one's share of crush, riders left behind */
  fullTrains: number;
  fullest: number;
  leftBehind: number;
  /** spans of full trains (s into the day) */
  spans: [number, number][];
  /** running cost of the trains over the day, US$; only for a whole day (yesterday's) */
  running: number;
}

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
    const t: LineToday = { riders: d[i + 2], fullTrains: d[i + 3], leftBehind: d[i + 4], fullest: d[i + 5], spans, stops, fares: 0, running: 0 };
    at.push({ id: String(id), t });
    i += 7 + 2 * nSpans + n;
  }
  at.forEach((a, k) => { a.t.fares = (d[i + k] ?? 0) * 1e6; a.t.running = d[i + at.length + k] ?? 0; out.set(a.id, a.t); });
  return out;
}

/** Boardings today per station id (stations with a counted line), summed once per answer. */
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
