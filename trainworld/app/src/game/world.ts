// The newest world snapshot, and the edits the UI can ask for.
//
// `world` is read-only for the UI: components read it, never mutate it. It is replaced each time
// the clock worker sends a new state (workers/clockClient.ts). Edits go to the worker as ops
// (workers/protocol.ts); a refused edit puts its reasons, in plain words, in `notice`.

import { t } from "../i18n";
import { signal, type Signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import { breakDeleteRun } from "./deleteHint";
import type { EditAnswer } from "../workers/clockClient";
import { describeIssues, type Buys } from "./issues";
import { tidyNodes, type FareNode } from "./fareCurve";
import { groundDist } from "./coords";
import { noteEdit } from "./editCap";
import type { Demand, LineId, LineInput, Save, StationInput, WorldView } from "./types";
import { DEMAND_INDEX, loopHalves } from "./types";

export const world = signal<WorldView | null>(null);
/** A short message for the player: why an edit was refused, what a construct cost. */
export const notice = signal<{ text: string; bad: boolean; at: number } | null>(null);

let sink: ((op: EditOp, early?: boolean) => Promise<EditAnswer>) | null = null;
/** workers/clockClient.ts hands over its edit function. */
export function setEditSink(f: (op: EditOp, early?: boolean) => Promise<EditAnswer>) {
  sink = f;
}

/** T-1659: told of every refusal (map/buildSound.ts plays the knock). */
let onRefuse: (() => void) | null = null;
export function setRefuseHook(f: () => void) {
  onRefuse = f;
}

/** The refusal knock alone (T-1844): a step the controls refused for a limit, with nothing to explain. */
export function knock() {
  onRefuse?.();
}

export function say(text: string, bad = false) {
  if (bad) onRefuse?.();
  notice.value = { text, bad, at: performance.now() };
}

/** Edits sent other than fares, undo and redo: a fare change is the last action while this holds
 * still (T-1130, game/fareUndo.ts). */
export const editsSent = signal(0);

/** What the player's click is about to do; a click behind folded or pasted edits names only the op. */
export type Intent = EditOp | { op: "setTrainCount" | "setHeadway" | "lineCars" };

/** Told of every edit the player makes the moment they make it, before the worker answers
 * (map/buildSound.ts, T-1474): its sound plays at the click. Checks that send edits to the client
 * directly are not heard. */
let beforeEdit: ((i: Intent) => void) | null = null;
export function setBeforeEdit(f: (i: Intent) => void) {
  beforeEdit = f;
}
export function clickSound(i: Intent) {
  beforeEdit?.(i);
}

/** Told of every edit the worker accepted, with its answer (map/buildSound.ts, T-1683): the sounds of
 * a construct or an upgrade play here, so a refusal for want of cash is heard as the knock alone. */
let afterEdit: ((op: EditOp, a: EditAnswer) => void) | null = null;
export function setAfterEdit(f: (op: EditOp, a: EditAnswer) => void) {
  afterEdit = f;
}

/** Send an edit; a refusal is explained in `notice`. Resolves with the answer either way. `silent`:
 * its sound was played already (the click that sent it, or the one sound of a run of edits). */
export async function edit(op: EditOp, quiet = false, silent = false, early = false): Promise<EditAnswer> {
  breakDeleteRun();
  if (!silent) beforeEdit?.(op);
  if (!sink) return { ok: false, issues: [], charge: 0 };
  noteEdit(op, world.peek());
  if (op.op !== "fares" && op.op !== "undo" && op.op !== "redo") editsSent.value++;
  // `early` (T-1407): the answer may come before `world` holds the edit (`a.early`)
  const a = await sink(op, early);
  if (!a.ok && !quiet && a.issues.length) say(describeIssues(a.issues, buysOf(op)), true);
  else if (a.ok && !silent) afterEdit?.(op, a);
  return a;
}

const TRAIN_OPS = new Set(["setStops", "insertStop", "dropStop", "endLineAt", "stopPassed", "orderStops", "sortStops", "setTrainCount", "setHeadway", "lineCars", "copyLine", "undo", "redo"]);
const BUILD_OPS = new Set(["route", "construct", "constructAll", "constructLine", "addStation", "crossover", "doubleUp", "stationUpgrade"]);
/** What a refused-for-money edit would have paid for: line edits cost trains only, track edits construction. */
function buysOf(op: EditOp): Buys | undefined {
  return TRAIN_OPS.has(op.op) ? "trains" : BUILD_OPS.has(op.op) ? "construction" : undefined;
}

const num = (id: LineId) => Number(id);

/** Train counts asked for but not yet in `world` (T-229): the stepper shows them at once, and a
 * second click before the first edit lands counts from them, not from the old figure. */
export const pendingTrains = signal<ReadonlyMap<LineId, [number, number, number]>>(new Map());

/** A line's trains on the entire line per demand level, as the player last set them. */
export function trainsShown(id: LineId): [number, number, number] | null {
  return pendingTrains.value.get(id) ?? (world.value?.lineStats[id]?.trainsNeeded as [number, number, number] | undefined) ?? null;
}

/** Step a line's trains for one demand level, `by` from what it shows now (T-229). Clicks while an
 * edit is in flight are folded into one edit sent when it lands, so fast clicking never queues a
 * recompute per click. */
export function stepTrainCount(id: LineId, demand: Demand, by: number): boolean {
  return changeTrains(id, demand, (now, k) => Math.max(0, Math.round(now / k + by) * k));
}

export function setTrainCount(id: LineId, demand: Demand, trains: number) {
  changeTrains(id, demand, (_, k) => Math.max(0, Math.ceil(Math.round(trains) / k) * k));
}

/** False when the count could not move (none to take away): the stepper shakes (T-1844). */
function changeTrains(id: LineId, demand: Demand, to: (now: number, step: number) => number): boolean {
  const base = trainsShown(id);
  if (!base) return true;
  const t: [number, number, number] = [...base] as [number, number, number];
  const k = trainStep(id);
  const lev = DEMAND_INDEX[demand];
  t[lev] = to(t[lev], k);
  if (t[lev] === base[lev]) return false;
  setTrainCounts(id, t);
  return true;
}

/** The step of a line's train count: 2 on a loop run both ways, whose trains are half each way (T-798). */
export function trainStep(id: LineId): number {
  const st = world.value?.lineStats[id];
  return st && loopHalves(st) ? 2 : 1;
}

/** T-625: a train count or headway edit refused for want of cash: the line, the levels it changed,
 * when (the panel shakes the control), and the US$M it needed. The panel paints those levels'
 * train counts red until the cash covers it or another edit to the line goes through. */
export const refusedForCash = signal<{ id: LineId; levs: boolean[]; at: number; need: number } | null>(null);
function noteAnswer(id: LineId, a: EditAnswer, sent: number[], had: readonly number[] | null | undefined) {
  if (a.ok) {
    if (refusedForCash.peek()?.id === id) refusedForCash.value = null;
    return;
  }
  const m = a.issues.find((k) => k.startsWith("Money:"));
  if (m) refusedForCash.value = { id, levs: sent.map((v, i) => v !== had?.[i]), at: performance.now(), need: Number(m.split(":")[1]) };
}

/** Shows a line's value in `pending` at once and sends it with `send`, one edit in flight per line:
 * a value set meanwhile waits and goes when the edit lands, only the newest. `send` says whether
 * the edit was accepted; a refusal (money) drops what waited and the line goes back to the
 * world's value. */
function folded<V>(pending: Signal<ReadonlyMap<LineId, V>>, send: (id: LineId, v: V) => Promise<boolean>): (id: LineId, v: V) => void {
  const jobs = new Map<LineId, { busy: boolean; next: { v: V } | null }>();
  return (id, v) => {
    const p = new Map(pending.peek());
    p.set(id, v);
    pending.value = p;
    const job = jobs.get(id) ?? { busy: false, next: null };
    jobs.set(id, job);
    job.next = { v };
    if (job.busy) return;
    void (async () => {
      job.busy = true;
      while (job.next) {
        const { v } = job.next;
        job.next = null;
        if (!(await send(id, v))) job.next = null;
      }
      job.busy = false;
      jobs.delete(id);
      // the world now holds the last value sent (or, refused, the one before it)
      const q = new Map(pending.peek());
      q.delete(id);
      pending.value = q;
    })();
  };
}

const sendTrainCounts = folded(pendingTrains, async (id, trains) => {
  const had = world.peek()?.lineStats[id]?.trainsNeeded;
  const a = await edit({ op: "setTrainCount", line: num(id), trains }, false, true);
  noteAnswer(id, a, trains, had);
  return a.ok;
});
/** T-1797: what new train counts for a line cost beyond the cash, US$M, or null when the cash
 * covers them (game/money.ts sets it; the worker applies the same rule when it gets the edit). */
let trainBuyCheck: ((id: LineId, trains: readonly number[]) => { need: number; have: number } | null) | null = null;
export function setTrainBuyCheck(f: typeof trainBuyCheck) {
  trainBuyCheck = f;
}

/** T-1844: the same for cars a longer train needs: the cash short for `more` cars on line `id`, or null. */
let carBuyCheck: ((id: LineId, more: number) => { need: number; have: number } | null) | null = null;
export function setCarBuyCheck(f: typeof carBuyCheck) {
  carBuyCheck = f;
}

/** Each click is heard as it is made, however the edits behind it are folded. A count the cash
 * cannot cover is refused here, with the knock alone and nothing sent. */
const setTrainCounts = (id: LineId, trains: [number, number, number]) => {
  const was = trainsShown(id);
  const short = trainBuyCheck?.(id, trains);
  if (short) {
    refusedForCash.value = { id, levs: trains.map((v, i) => v !== was?.[i]), at: performance.now(), need: short.need };
    say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains"), true); // i18n-ignore
    return;
  }
  if (!was || trains.some((n, i) => n !== was[i])) clickSound({ op: "setTrainCount" });
  sendTrainCounts(id, trains);
};

/** T-1526: a line's train length asked for but not yet in `world`, folded like the train counts: a
 * click counts from it, not from the old length. `to` is the cars shown, `cap` what the line's
 * shortest platform fits (the most that fits is stored as 0, so the line follows its platforms). */
export const pendingCars = signal<ReadonlyMap<LineId, { to: number; cap: number }>>(new Map());
const sendCars = folded(pendingCars, async (id, v) => {
  const a = await edit({ op: "lineCars", line: num(id), cars: v.to >= v.cap ? 0 : v.to }, false, true);
  return a.ok;
});

/** Step a line's train length by `by` cars from what it shows now, within 1 to `cap`; `shown` is the
 * length the world gives it. False when it could not move, at 1 car or at `cap`; a promise of false
 * when the cash is short for the cars (nothing sent, the notice says so), as for a train count (T-1844). */
export function stepLineCars(id: LineId, shown: number, cap: number, by: number, checkCash = true): boolean | Promise<boolean> {
  const now = pendingCars.peek().get(id)?.to ?? shown;
  const to = Math.max(1, Math.min(cap, now + by));
  if (to === now) return false;
  const st = world.peek()?.lineStats[id];
  const short = checkCash && st && to > st.cars ? carBuyCheck?.(id, (to - st.cars) * Math.max(...st.trainsNeeded)) : null;
  if (short) {
    say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains"), true); // i18n-ignore
    return Promise.resolve(false);
  }
  clickSound({ op: "lineCars" });
  sendCars(id, { to, cap });
  return true;
}

/** T-625: a line whose trains follow a headway. The headways asked for but not yet in `world`,
 * folded like the train counts above; null is a switch back to counts set by hand (T-747). */
export const pendingHeadway = signal<ReadonlyMap<LineId, [number, number, number] | null>>(new Map());

/** Seconds per level the line keeps, as last set, or null when its trains are counted by hand. */
export function headwayShown(id: LineId): [number, number, number] | null {
  const p = pendingHeadway.value;
  if (p.has(id)) return p.get(id) ?? null;
  return world.value?.lineStats[id]?.autoS ?? null;
}

/** The headway limits (0 is no service) and the step from one headway to the next (T-1872): to the
 * whole minute above or below, so 5.37 min goes to 6 or 5 and 5 min to 6 or 4; with `minutes` 5
 * (shift-click, T-1888) to the multiple of 5, so 7 goes to 5 or 10 and 5 to 10 or the 1-minute limit. From no service,
 * plus gives 15 minutes and minus 5; minus at the 1-minute limit gives no service. From 1 hour up
 * the step is a multiple of 15 minutes (T-2519). The cap is 4 hours, the shortest stretch of one
 * demand level in the day (20:00 to 24:00, `level_stretches`): the timetable lays at least one
 * departure in each stretch, so a longer headway would run more trains than it asks for. */
export const HEADWAY_MIN_S = 60;
export const HEADWAY_MAX_S = 14400;
const LONG_HEADWAY_S = 3600, LONG_STEP_MIN = 15;
export function nextHeadway(s: number, dir: number, minutes = 1): number {
  if (s <= 0) return dir > 0 ? 900 : 300;
  const long = dir > 0 ? s >= LONG_HEADWAY_S : s > LONG_HEADWAY_S;
  const k = Math.max(long ? LONG_STEP_MIN : 1, minutes), m = s / 60 / k, near = Math.round(m), whole = Math.abs(m - near) < 1e-6;
  if (dir > 0) return Math.min(HEADWAY_MAX_S, (whole ? near + 1 : Math.ceil(m)) * k * 60);
  if (s <= HEADWAY_MIN_S) return 0;
  return Math.max(HEADWAY_MIN_S, (whole ? near - 1 : Math.floor(m)) * k * 60);
}
/** The fewest trains that keep a headway over a round trip, as the sim works it out; with the
 * line's shortest headway `limitS`, no more than fit when the headway is at that limit (T-679,
 * `clamp_auto`). A loop (T-798) keeps it each way over half its round trip with half its trains. */
export function autoTrainsFor(roundTripS: number, headwayS: number, limitS = 0, loop = false): number {
  if (loop) return 2 * autoTrainsFor(roundTripS / 2, headwayS, limitS);
  const want = headwayS > 0 && roundTripS > 0 ? Math.min(9999, Math.ceil(roundTripS / (headwayS * AUTO_SLACK) - 1e-6)) : 0;
  if (!(want > 0) || !(limitS > 0) || !Number.isFinite(limitS) || headwayS * AUTO_SLACK < limitS) return want;
  return Math.min(want, Math.max(1, Math.floor(roundTripS / limitS + 1e-6)));
}
/** `AUTO_SLACK` in sim/src/track/world.rs */
export const AUTO_SLACK = 1.005;
/** A typed number of minutes as seconds, to 5 s inside the limits; 0 or less is no service. */
export function headwayFromMinutes(min: number): number {
  if (!(min > 0)) return 0;
  return Math.max(HEADWAY_MIN_S, Math.min(HEADWAY_MAX_S, Math.round((min * 60) / 5) * 5));
}

export function setHeadway(id: LineId, demand: Demand, seconds: number) {
  const base = headwayShown(id);
  if (!base) return;
  const h: [number, number, number] = [...base] as [number, number, number];
  h[DEMAND_INDEX[demand]] = seconds;
  queueHeadway(id, h);
}

const sendHeadway = folded(pendingHeadway, async (id, secs) => {
  const had = world.peek()?.lineStats[id]?.autoS;
  const a = await edit({ op: "setHeadway", line: num(id), seconds: secs }, false, true);
  if (secs) noteAnswer(id, a, secs, had);
  return a.ok;
});
/** T-1800: each level's shortest headway the line's inspector last read (`lineHeld`), 0 where none. */
const headwayLimits = new Map<LineId, readonly number[]>();
export function noteHeadwayLimits(id: LineId, s: readonly number[]) {
  headwayLimits.set(id, s);
}

/** T-1800: the trains a headway edit will leave each level with, as far as the panel can tell, or
 * null when it cannot. A level kept as it is keeps its trains. A headway at or over the level's
 * shortest is the round trip over the headway, as `autoTrainsFor`; a shorter one is trimmed by the
 * sim to what the track takes, over several rounds the panel cannot reproduce, and a line whose
 * limits the panel has not read (a bus line's too) is unknown, so the worker decides. */
function headwayTrains(id: LineId, secs: readonly number[], was: readonly number[] | null): number[] | null {
  const st = world.peek()?.lineStats[id], lim = headwayLimits.get(id);
  if (!st || st.bus || !lim) return null;
  const out: number[] = [];
  for (let i = 0; i < secs.length; i++) {
    if (secs[i] === was?.[i]) out.push(st.trainsNeeded[i]);
    else if (secs[i] * AUTO_SLACK < lim[i]) return null;
    else out.push(autoTrainsFor(st.roundTripByLevel[i], secs[i], lim[i], loopHalves(st)));
  }
  return out;
}

/** Each click is heard as it is made. A headway whose trains the cash cannot cover is refused here,
 * with the knock alone and nothing sent (T-1800, as `setTrainCounts`). */
const queueHeadway = (id: LineId, secs: [number, number, number] | null) => {
  const was = headwayShown(id);
  const trains = secs && trainBuyCheck ? headwayTrains(id, secs, was) : null;
  if (trains) {
    const short = trainBuyCheck!(id, trains);
    if (short) {
      refusedForCash.value = { id, levs: secs!.map((v, i) => v !== was?.[i]), at: performance.now(), need: short.need };
      say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains"), true); // i18n-ignore
      return;
    }
  }
  if (JSON.stringify(secs) !== JSON.stringify(was)) clickSound({ op: "setHeadway" });
  sendHeadway(id, secs);
};

/** False when the headway is at its limit already (T-1844): the stepper shakes. `dir`'s size is the
 * step in minutes (the stepper's 1, 5 or 20), its sign the direction. */
export function stepHeadway(id: LineId, demand: Demand, dir: number): boolean {
  const base = headwayShown(id);
  if (!base) return true;
  const now = base[DEMAND_INDEX[demand]], to = nextHeadway(now, Math.sign(dir), Math.abs(dir));
  if (to === now) return false;
  setHeadway(id, demand, to);
  return true;
}

/** Switch a line between counting its trains by hand and keeping a headway: it starts with the
 * headway its trains give now, so no train comes or goes. */
export function setAutoTrains(id: LineId, on: boolean) {
  const st = world.value?.lineStats[id];
  if (!st) return;
  if (!on) {
    queueHeadway(id, null);
    return;
  }
  const n = trainsShown(id) ?? st.trainsNeeded;
  // the headway its trains give, rounded to the coarsest of 1 minute, 30, 10 or 5 seconds that keeps them
  const seconds = n.map((k, lev) => {
    const trip = st.roundTripByLevel[lev];
    if (!(k > 0) || !(trip > 0)) return 0;
    const exact = Math.ceil(trip / k);
    for (const step of [60, 30, 10, 5]) {
      const h = Math.ceil(exact / step) * step;
      if (autoTrainsFor(trip, h, 0, loopHalves(st)) === k) return h;
    }
    return exact;
  }) as [number, number, number];
  queueHeadway(id, seconds);
}

/** The fare curve for the whole network (T-883, game/fareCurve.ts): its nodes, km and US$. */
export function setFares(nodes: FareNode[]) {
  return edit({ op: "fares", nodes: tidyNodes(nodes) });
}
/** T-1277: a country's fare scale, null for its income's. */
export function setFareScale(country: string, scale: number | null) {
  return edit({ op: "fareScale", country, scale });
}
/** T-2679: the reference country, whose fares the Money tab's multiples are of. */
export function setFareRef(country: string) {
  return edit({ op: "fareRef", country });
}
/** T-1626: the fare multiplier on high-speed lines. */
export function setFareHsr(mul: number) {
  return edit({ op: "fareHsr", mul });
}
/** T-2194: the fare multiplier on ferry lines. */
export function setFareFerry(mul: number) {
  return edit({ op: "fareFerry", mul });
}
/** T-3028: the fare multiplier on bus lines. */
export function setFareBus(mul: number) {
  return edit({ op: "fareBus", mul });
}
/** T-1801: the transfer fare, in the curve's units. */
export function setFareTransfer(usd: number) {
  return edit({ op: "fareTransfer", usd });
}

/** T-1661: take a loan of `amount` US$M, or repay the one out; the answer says whether it was done. */
export function borrow(amount: number) {
  return edit({ op: "borrow", amount }, true, true);
}
export function repay() {
  return edit({ op: "repay" }, true, true);
}

export function renameLine(id: LineId, name: string) {
  const n = name.trim();
  const l = lineById(id);
  if (n && l) void edit({ op: "lineLook", line: num(id), name: n });
}

export const undo = () => edit({ op: "undo" }, true);
export const redo = () => edit({ op: "redo" }, true);

// The lookups below are built once per save: a new snapshot makes a new `save` object and nothing
// edits one in place, so a WeakMap keyed on it never goes stale.
const looked = new WeakMap<Save, Map<string, unknown>>();
function lookup<T>(w: WorldView, key: string, build: (s: Save) => T): T {
  let m = looked.get(w.save);
  if (!m) looked.set(w.save, (m = new Map()));
  if (!m.has(key)) m.set(key, build(w.save));
  return m.get(key) as T;
}
export const stationsById = (w: WorldView) => lookup(w, "stations", (s) => new Map(s.stations.map((x) => [x.id, x])));
const stopMemo = new WeakMap<WorldView, { byId: Map<string, StationInput>; names: Map<string, string>; bus: Map<string, StationInput> }>();
/** T-1464: the stations and a bus line's other stops (no station, no name of their own) by id: a stop's
 * place for the figures, its words for the panels. A bus stop is named for the station within 1 km
 * (the nearest), else "Bus stop". `stationsById` stays stations only. */
function stopsOf(w: WorldView) {
  let m = stopMemo.get(w);
  if (m) return m;
  const byId = new Map<string, StationInput>(w.save.stations.map((x) => [x.id, x]));
  const names = new Map<string, string>(w.save.stations.map((x) => [x.id, x.name]));
  const bus = new Map<string, StationInput>();
  for (const st of Object.values(w.lineStats)) {
    for (const p of st.bus?.stops ?? []) {
      const id = String(p.id);
      if (p.station >= 0 || byId.has(id)) continue;
      const stop: StationInput = { id, num: p.id, name: "", lng: p.lng, lat: p.lat, x: p.x, y: p.y, level: 0, length: 0, built: st.bus!.built, heading: 0 };
      byId.set(id, stop);
      bus.set(id, stop);
      let near = "", best = 1000;
      for (const s of w.save.stations) {
        const d = groundDist(s.x, s.y, p.x, p.y);
        if (s.name && d < best) { best = d; near = s.name; }
      }
      names.set(id, near ? t("world.busStopNear", { name: near }) : t("world.busStop"));
    }
  }
  stopMemo.set(w, (m = { byId, names, bus }));
  return m;
}
export const stopsById = (w: WorldView) => stopsOf(w).byId;
/** Only the bus stops that are no station. */
export const busStopsById = (w: WorldView) => stopsOf(w).bus;
/** Every stop's name by id, a bus stop's included. */
export const stationNames = (w: WorldView) => stopsOf(w).names;
export const stationsByNum = (w: WorldView) => lookup(w, "stationNums", (s) => new Map(s.stations.map((x) => [x.num, x])));
export const linesById = (w: WorldView) => lookup(w, "lines", (s) => new Map(s.lines.map((l) => [l.id, l])));
export const linesByNum = (w: WorldView) => lookup(w, "lineNums", (s) => new Map(s.lines.map((l) => [l.num, l])));
/** The lines that stop at each station, in line order. */
export const linesAtMap = (w: WorldView) => lookup(w, "linesAt", (s) => {
  const at = new Map<string, LineInput[]>();
  for (const l of s.lines) for (const st of new Set(l.stops)) (at.get(st) ?? at.set(st, []).get(st)!).push(l);
  return at;
});

export const lineById = (id: LineId) => (world.value ? linesById(world.value).get(id) : undefined);
export const linesAt = (station: string) => (world.value ? linesAtMap(world.value).get(station) : undefined) ?? [];
