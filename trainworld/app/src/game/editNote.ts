// "What this edit did" (T-2125, after T-1478's daily rides note): once demand has settled after the
// player's edits, the change in train trips a day and profit a day since the last settled state
// (ui/EditNote.tsx shows it). Every edit through `edit()` counts but the look, group and loan ops
// (editCap's NEUTRAL); edits that land while one is settling add up into one note. Both figures are
// for the day shown (the weekday's for the average day, which has no fares of its own): train trips
// are the trips taken by rail (`railTrips`, each trip once however many lines it rides, T-2237: a
// line that saves people a change lowers boardings while trips rise), profit is that day's fare
// income (`incomeByDay`, what the clock pays) minus its running cost and the upkeep. A one-line
// edit's line figures are its riders (`LineDemand.ridersPerDay`, its boardings) and its profit.
// Zones keep their levels of detail while edits settle (`editsSettling`, T-2180), and figures from
// other levels are never compared. Held on the main thread only and not saved.

import { effect, signal } from "@preact/signals";
import { demandByDay, demandUpdating, editsSettling, roughZones, type DayDemand, type DemandView } from "./demand";
import { DAY_KINDS, DEMAND_DAY_PHRASE, levelOn } from "./days";
import { editStamp, lastEdit } from "./editCap";
import { trainKind } from "./format";
import { loadProgress } from "./loading";
import { incomeByDay, moneyState, type Income } from "./money";
import { PERIODS } from "./periods";
import { perfOff } from "../perfFlags";
import { DEMAND_INDEX, type WorldView } from "./types";
import { shownDay, shownDayPhrase } from "./ui";
import { world } from "./world";
import type { MoneyState } from "../workers/protocol";

/** Wall ms the note stays (while the pointer is off it). */
export const NOTE_MS = 8000;
/** Wall ms the numbers take to ease up to their figures. */
export const NOTE_COUNT_MS = 900;
/** Wall ms after the last edit, or the last moment demand was updating, before an edit that started
 * no solve (a refused one, a blueprint) is settled with the figures as they are (lineFigures' wait). */
const NO_SOLVE_MS = 8000;
/** Wall ms after which edits demand never settled are forgotten. */
const GIVE_UP_MS = 5 * 60_000;

export interface FromTo { from: number; to: number }
export interface EditNote {
  /** train trips a day (each trip once) and the network's profit a day (US$), before and after */
  trips: FromTo;
  money: FromTo;
  /** the line, when the edits were about exactly one line (named in them, or new): its riders (its
   * boardings) and profit */
  line?: { id: string; riders: FromTo; money: FromTo };
  /** "per weekday": the day the figures are for */
  day: string;
  /** performance.now() when it landed */
  at: number;
}

/** The note showing, null for none. */
export const editNote = signal<EditNote | null>(null);

/** The figure to show now: eased out from 0 (fast at first, slow to finish). */
export function shownChange(delta: number, at: number, now = performance.now()): number {
  const t = Math.min(1, (now - at) / NOTE_COUNT_MS);
  return delta * (1 - (1 - t) ** 3);
}

interface Figures {
  trips: number;
  money: number;
  lines: Map<string, { riders: number; money: number }>;
  /** the day the figures are for, and the demand result they came from */
  day: number;
  version: number;
  /** the zones shown from a coarse solve: figures at other levels of detail are not compared (T-2180) */
  rough: string;
}

const PERIOD_HOURS = PERIODS.map((p) => p.to - p.from);

/** Whether `dv` has figures for the lines as they are now: a line whose figures are for other stops
 * is still being solved. */
function solvedForNow(w: WorldView, dv: DemandView): boolean {
  for (const l of w.save.lines) {
    const d = dv.lines.get(l.id);
    if (!d?.stops || w.lineStats[l.id]?.status !== "running") continue;
    if (d.stops.length !== l.stops.length || d.stops.some((s, k) => s !== l.stops[k])) return false;
  }
  return true;
}

/** Trips taken by rail a day in `d` (T-2237): every zone's own (its view's `railTripsPerDay`, the
 * city views' train trips) and the market's between places (longer trips and commutes), each trip
 * once however many lines it rides. Summed from the zones rather than read off the network view,
 * whose total a zone's later figures (a local or focus result) put in leave as it was. */
export function railTrips(d: DayDemand): number {
  // a solve with no market gives the market's figures as NaN (no summary to add up)
  const ok = (v: number | undefined) => (v !== undefined && Number.isFinite(v) ? v : 0);
  let n = ok(d.intercity?.railTripsPerDay);
  for (const v of d.cities.values()) n += ok(v.railTripsPerDay);
  return n;
}

function figuresOf(w: WorldView, d: DayDemand | null, kind: number, inc: Income, m: MoneyState): Figures {
  const dv = d?.view ?? null;
  const lines = new Map<string, { riders: number; money: number }>();
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st) continue;
    const r = st.status === "running" ? dv?.lines.get(l.id)?.ridersPerDay ?? 0 : 0;
    let fares = 0, run = 0;
    const lm = inc.lines.get(l.id);
    if (lm) lm.perHour.forEach((v, p) => { fares += v * PERIOD_HOURS[p] * 1e6; });
    if (st.status === "running") {
      // as `runningToday` (game/today.ts), over the whole day of this kind
      const k = trainKind(m, st.fast);
      const perKm = st.bus?.ferry ? m.runCostBoatKm : st.bus ? m.runCostBusKm : k.runCostTrainKm + st.cars * k.runCostCarKm;
      const tph = [l.tph.high, l.tph.medium, l.tph.low];
      PERIODS.forEach((p, i) => { run += tph[levelOn(kind, DEMAND_INDEX[p.demand])] * 2 * st.km * perKm * m.economy * PERIOD_HOURS[i]; });
    }
    lines.set(l.id, { riders: r, money: fares - run - st.upkeep });
  }
  let fares = 0, run = 0;
  inc.perHour.forEach((v, p) => { fares += v * PERIOD_HOURS[p]; });
  PERIODS.forEach((p, i) => { run += m.runningPerHour[levelOn(kind, DEMAND_INDEX[p.demand])] * PERIOD_HOURS[i]; });
  const upkeep = m.upkeep.track + m.upkeep.stations + m.upkeep.buses;
  return { trips: d ? railTrips(d) : 0, money: (fares - run - upkeep) * 1e6, lines, day: kind, version: dv?.version ?? -1, rough: [...roughZones.peek()].sort().join() };
}

/** Start watching for edits. Call once. */
export function startEditNote() {
  /** the figures at the last settled state */
  let base: Figures | null = null;
  /** edits since `base`: when the first was made, and the last activity (an edit, demand updating or
   * a city loading). A city that loads meanwhile is no worry: one no line reaches has no train trips,
   * and track into a new zone loads it as part of the edit. */
  let pending: { since: number; last: number; lines: Set<number> } | null = null;
  const setPending = (p: typeof pending) => { pending = p; editsSettling.value = !!p; };
  let hide = 0, wake = 0;
  const tick = signal(0);
  const later = (ms: number) => { clearTimeout(wake); wake = setTimeout(() => { tick.value++; }, ms) as unknown as number; };

  let stamp = editStamp.peek();
  effect(() => {
    if (editStamp.value === stamp) return;
    stamp = editStamp.value;
    // the note shown is about the network before this edit
    if (editNote.peek()) { clearTimeout(hide); editNote.value = null; }
    const now = performance.now();
    if (!base) return;
    if (pending) pending.last = now;
    else setPending({ since: now, last: now, lines: new Set() });
    // the line it is about (a copy's source is not: the copy is new)
    const op = lastEdit;
    if (op && "line" in op && typeof op.line === "number" && op.op !== "copyLine") pending!.lines.add(op.line);
    later(NO_SOLVE_MS);
  });

  let day = -1;
  effect(() => {
    void tick.value;
    void roughZones.value;
    const w = world.value, updating = demandUpdating.value, loading = !!loadProgress.value;
    const shown = shownDay.value, kind = shown < DAY_KINDS.length ? shown : 0;
    const dd = demandByDay.value[kind] ?? null, dv = dd?.view ?? null, inc = incomeByDay.value[kind], m = moneyState.value;
    if (!w) { base = null; setPending(null); return; }
    // another day's figures are other figures
    if (kind !== day) { day = kind; base = null; setPending(null); }
    const now = performance.now();
    if (pending) {
      if (updating || loading) pending.last = now;
      if (now - pending.since > GIVE_UP_MS) { setPending(null); base = null; }
    }
    // a network with nothing running has nothing for demand to solve: its train trips are 0 (a new game's
    // first line counts from there)
    const idle = !w.save.lines.some((l) => w.lineStats[l.id]?.status === "running");
    if (updating || loading || !inc || !m || (!idle && (!dv?.final || !solvedForNow(w, dv)))) return;
    const version = idle ? -1 : dv!.version;
    if (pending && base && version === base.version && now - pending.last < NO_SOLVE_MS) {
      later(NO_SOLVE_MS - (now - pending.last) + 50);
      return;
    }
    const after = figuresOf(w, idle ? null : dd, kind, inc, m);
    const was = pending ? base : null, edited = pending?.lines;
    base = after;
    setPending(null);
    // a zone's level of detail changed between them (a change already under way at the first edit):
    // coarse and fine figures differ by more than a small edit moves (T-2180)
    if (!was || !edited || (was.rough !== after.rough && !perfOff("lodHold"))) return;
    const trips = { from: Math.round(was.trips), to: Math.round(after.trips) };
    const money = { from: was.money, to: after.money };
    if (trips.from === trips.to && Math.abs(money.to - money.from) < 500) return;
    // the lines the edits named, and the new ones; other lines' figures move with them, so it takes the edits
    const touched = w.save.lines.filter((l) => after.lines.has(l.id) && (edited.has(l.num) || !was.lines.has(l.id))).map((l) => l.id);
    let line: EditNote["line"];
    if (touched.length === 1) {
      const id = touched[0], f = after.lines.get(id)!, b = was.lines.get(id);
      line = { id, riders: { from: Math.round(b?.riders ?? 0), to: Math.round(f.riders) }, money: { from: b?.money ?? 0, to: f.money } };
    }
    clearTimeout(hide);
    editNote.value = { trips, money, line, day: shown < DAY_KINDS.length ? shownDayPhrase.peek() : DEMAND_DAY_PHRASE[0], at: now };
    holdNote(false);
  });

  /** While the pointer is on the note it stays; it goes NOTE_MS after it lands or the pointer leaves. */
  holdNote = (on: boolean) => {
    clearTimeout(hide);
    if (!on) hide = setTimeout(() => { editNote.value = null; }, NOTE_MS) as unknown as number;
  };
}

/** The pointer came onto the note (true) or left it. */
export let holdNote: (on: boolean) => void = () => {};
