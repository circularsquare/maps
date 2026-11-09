// A line's schedule settings copied for pasting onto another line (T-1293): its trains or
// headways per level, train length and kind of train. Kept in the browser, so it outlasts the game.

import { t, tList, tNumber } from "../i18n";
import { signal } from "@preact/signals";
import { CAR_M } from "./boarding";

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
  /** T-2919: what the line runs; absent in a copy made before it, a train line's */
  vehicle?: "train" | "bus" | "boat";
}

const vehicleOf = (st: LineStats) => (st.bus?.ferry ? "boat" : st.bus ? "bus" : "train");
/** Whether pasting `c` on a line carries its train length and kind: only from a train line to a train line. */
export const wholeCopy = (c: ScheduleCopy, st: LineStats): boolean => !st.bus && (c.vehicle ?? "train") === "train";

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
  cars: Math.max(1, st.cars),
  fast: st.fast,
  vehicle: vehicleOf(st),
});

export function copySchedule(l: LineInput, st: LineStats) {
  const c = scheduleOf(l, st);
  scheduleCopy.value = c;
  try { localStorage.setItem(STORE, JSON.stringify(c)); } catch { /* storage unavailable: the copy lasts this visit */ }
  say(t("scheduleCopy.copied", { line: l.name }));
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

const mins = (s: number) => (s > 0 ? t("scheduleCopy.minutes", { n: tNumber(+(s / 60).toFixed(2)) }) : t("format.headwayNone"));

/** "Headways 3 min, 5 min and 10 min, 6 cars, high speed", high, medium and low in order; without
 * `whole`, what a paste carries to or from a bus or ferry line: "Headways 8 min, 15 min and 30 min" or
 * "4, 3 and 2 buses". */
export function scheduleText(c: ScheduleCopy, whole = true): string {
  if (!whole) return c.headway ? t("scheduleCopy.headwaysOnly", { list: tList(c.headway.map(mins)) }) : t("scheduleCopy.countsOnly", { list: tList(c.trains.map(String)), vehicle: c.vehicle ?? "train" });
  return t(c.headway ? "scheduleCopy.withHeadways" : "scheduleCopy.withTrains", { list: tList((c.headway ? c.headway.map(mins) : c.trains.map(String))), cars: c.cars, fast: c.fast });
}

/** Put the copy on a line: kind of train, length (cut to what its platforms fit), then the
 * trains or headways, each its own edit; a refused edit (cash) stops the rest. */
export async function pasteSchedule(l: LineInput, st: LineStats) {
  const c = scheduleCopy.peek();
  if (!c) return;
  clickSound({ op: "lineCars" }); // one sound for the paste, not one for each of its edits
  // T-2919: train length and kind only from a train line to a train line
  const whole = wholeCopy(c, st);
  if (whole && c.fast !== st.fast && !(await edit({ op: "lineFast", line: l.num, fast: c.fast }, false, true)).ok) return;
  const fit = whole ? carsCap(l) : null;
  const cap = fit?.cap ?? c.cars;
  const cars = Math.min(c.cars, cap);
  if (whole && cars !== st.cars && !(await edit({ op: "lineCars", line: l.num, cars: cars >= cap ? 0 : cars }, false, true)).ok) return;
  if (c.headway) {
    if (!(await edit({ op: "setHeadway", line: l.num, seconds: c.headway }, false, true)).ok) return;
  } else {
    if (headwayShown(l.id) && !(await edit({ op: "setHeadway", line: l.num, seconds: null }, false, true)).ok) return;
    // T-798: a loop's trains are half each way, so its counts are even
    const k = st.loop ? 2 : 1;
    const trains = c.trains.map((n) => Math.ceil(n / k) * k) as Levels;
    if (!(await edit({ op: "setTrainCount", line: l.num, trains }, false, true)).ok) return;
  }
  if (fit && cars < c.cars) say(t("scheduleCopy.tooShort", { station: fit.name, wanted: c.cars, line: l.name, kept: cars }));
}
