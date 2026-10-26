// A line's schedule settings copied for pasting onto another line (T-1293): its trains or
// headways per level, train length and kind of train. Kept in the browser, so it outlasts the game.

import { signal } from "@preact/signals";
import { CAR_M } from "./boarding";
import { andList, plural } from "./format";
import type { LineInput, LineStats } from "./types";
import { clickSound, edit, headwayShown, say, stationsById, trainsShown, world } from "./world";

type Levels = [number, number, number];

export interface ScheduleCopy {
  /** the line it came from, for the paste button's hover */
  from: string;
  /** seconds per level when the line kept a headway, else null and `trains` holds */
  headway: Levels | null;
  trains: Levels;
  cars: number;
  fast: boolean;
}

const STORE = "anitabuilder-schedule-copy";

function load(): ScheduleCopy | null {
  try {
    const v = JSON.parse(localStorage.getItem(STORE) || "null") as ScheduleCopy | null;
    return v && Array.isArray(v.trains) && v.trains.length === 3 && v.cars > 0 ? v : null;
  } catch {
    return null;
  }
}

export const scheduleCopy = signal<ScheduleCopy | null>(load());

/** A line's settings as shown (a change still in flight counts). */
export const scheduleOf = (l: LineInput, st: LineStats): ScheduleCopy => ({
  from: l.name,
  headway: headwayShown(l.id),
  trains: trainsShown(l.id) ?? (st.trainsNeeded as Levels),
  cars: st.cars,
  fast: st.fast,
});

export function copySchedule(l: LineInput, st: LineStats) {
  if (st.bus) return; // a bus has no cars or train kind to carry over
  const c = scheduleOf(l, st);
  scheduleCopy.value = c;
  try { localStorage.setItem(STORE, JSON.stringify(c)); } catch { /* storage unavailable: the copy lasts this visit */ }
  say(`Schedule of ${l.name} copied.`);
}

/** The most cars a line's shortest platform fits, and that platform; null with no stations. */
export function carsCap(l: LineInput): { cap: number; name: string; m: number } | null {
  const stations = world.value ? stationsById(world.value) : null;
  let short: { name: string; length: number } | undefined;
  for (const s of l.stops) {
    const p = stations?.get(s);
    if (p && (!short || p.length < short.length)) short = p;
  }
  return short ? { cap: Math.max(1, Math.floor(short.length / CAR_M)), name: short.name, m: short.length } : null;
}

const mins = (s: number) => (s > 0 ? `${+(s / 60).toFixed(2)} min` : "none");

/** "Headways 3 min, 5 min and 10 min, 6 cars, high speed", high, medium and low in order. */
export function scheduleText(c: ScheduleCopy): string {
  const levels = c.headway ? `Headways ${andList(c.headway.map(mins))}` : `${andList(c.trains.map(String))} trains`;
  return `${levels}, ${plural(c.cars, "car")}, ${c.fast ? "high speed" : "ordinary"}`;
}

/** Put the copy on a line: kind of train, length (cut to what its platforms fit), then the
 * trains or headways, each its own edit; a refused edit (cash) stops the rest. */
export async function pasteSchedule(l: LineInput, st: LineStats) {
  const c = scheduleCopy.peek();
  if (!c || st.bus) return;
  clickSound({ op: "lineCars" }); // one sound for the paste, not one for each of its edits
  if (c.fast !== st.fast && !(await edit({ op: "lineFast", line: l.num, fast: c.fast }, false, true)).ok) return;
  const fit = carsCap(l);
  const cap = fit?.cap ?? c.cars;
  const cars = Math.min(c.cars, cap);
  if (cars !== st.cars && !(await edit({ op: "lineCars", line: l.num, cars: cars >= cap ? 0 : cars }, false, true)).ok) return;
  if (c.headway) {
    if (!(await edit({ op: "setHeadway", line: l.num, seconds: c.headway }, false, true)).ok) return;
  } else {
    if (headwayShown(l.id) && !(await edit({ op: "setHeadway", line: l.num, seconds: null }, false, true)).ok) return;
    // T-798: a loop's trains are half each way, so its counts are even
    const k = st.loop ? 2 : 1;
    const trains = c.trains.map((n) => Math.ceil(n / k) * k) as Levels;
    if (!(await edit({ op: "setTrainCount", line: l.num, trains }, false, true)).ok) return;
  }
  if (fit && cars < c.cars) say(`Trains of ${plural(cars, "car")} on ${l.name}: the platform at ${fit.name} fits no more.`);
}
