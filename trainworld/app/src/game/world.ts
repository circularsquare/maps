// The newest world snapshot, and the edits the UI can ask for.
//
// `world` is read-only for the UI: components read it, never mutate it. It is replaced each time
// the clock worker sends a new state (workers/clockClient.ts). Edits go to the worker as ops
// (workers/protocol.ts); a refused edit puts its reasons, in plain words, in `notice`.

import { t } from "../i18n";
import { editTrace, trace, tracing } from "./freezeTrace";
import { signal, type Signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import { breakDeleteRun } from "./deleteHint";
import type { EditAnswer } from "../workers/clockClient";
import { describeIssues, type Buys } from "./issues";
import { vehicleOf, type Vehicle } from "./vehicle";
import { tidyNodes, type FareNode } from "./fareCurve";
import { groundDist, groundScale } from "./coords";
import { busPlaces } from "./busLinks";
import { noteEdit } from "./editCap";
import { checkPatches, freshly, freshNow, sameBuilt, worldDiff } from "./worldDiff";
import type { Demand, LineId, LineInput, LineStats, RowFigs, Save, StationInput, WorldView } from "./types";
import { DEMAND_INDEX, DEMANDS, loopHalves } from "./types";

export const world = signal<WorldView | null>(null);
/** T-3159: the world just set is a stop click's rows alone (`ClockState.rows`); the network it goes with comes later */
export const worldIsRows = { value: false };
/** T-3297: the game's running side, false where trains, trams and buses keep left (`ClockState.rightHand`) */
export const rightHand = signal(true);
/** T-3297: the last side switch (ui/SideSwitch.tsx): the side it went to, and its sites as the sim gives them (`kind`
 * as `FLIP_*`, `id` the edge, line or node); null in a game not switched since it opened */
export const flipSites = signal<{ right: boolean; sites: { kind: number; id: number }[] } | null>(null);
/** A short message for the player: why an edit was refused, what a construct cost. */
export const notice = signal<{ text: string; bad: boolean; at: number } | null>(null);

let sink: ((op: EditOp, early?: boolean | "rows" | "click", game?: number) => Promise<EditAnswer>) | null = null;
/** workers/clockClient.ts hands over its edit function. */
export function setEditSink(f: (op: EditOp, early?: boolean | "rows" | "click", game?: number) => Promise<EditAnswer>) {
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

/** T-3731: the line tool's stop clicks, each sent once the one before is answered (it reads the stops that one
 * left), but a plain click (`sendStopEdit`, T-4114). A click is kept when stop editing ends; only a game loaded since
 * drops it. */
let stopQueue: Promise<void> = Promise.resolve();
/** stop clicks made and not yet answered */
export const stopsQueued = signal(0);
/** when the queue last moved (a click made into an empty queue, or one answered) */
let stopsMoved = 0;
/** a waiter gives up on clicks that have had no answer for this long, ms (a worker that never answers holds nothing
 * for ever; clockClient's `stopOut` is as long) */
const STOPS_GIVE_UP_MS = 10000;
export function queueStopEdit(f: () => Promise<void>): Promise<void> {
  if (stopsQueued.peek() === 0) stopsMoved = performance.now();
  stopsQueued.value++;
  chained++;
  const p = stopQueue.then(f).finally(() => { chained--; stopsQueued.value--; stopsMoved = performance.now(); });
  stopQueue = p.catch(() => {});
  return p;
}
/** queued stop edits not yet answered (`queueStopEdit`) */
let chained = 0;
/** T-4114: a plain stop click (`toggleStop`, which the worker reads against the stops it holds), sent at once, not after
 * the clicks still out; edits queued after it wait for it as for the others. Behind a queued edit it keeps its place. */
export function sendStopEdit(f: () => Promise<void>): Promise<void> {
  if (chained > 0) return queueStopEdit(f);
  if (stopsQueued.peek() === 0) stopsMoved = performance.now();
  stopsQueued.value++;
  const p = f().finally(() => { stopsQueued.value--; stopsMoved = performance.now(); });
  const was = stopQueue;
  stopQueue = Promise.all([was, p.catch(() => {})]).then(() => {});
  return p;
}
const stopsStuck = () => performance.now() - stopsMoved >= STOPS_GIVE_UP_MS;
/** Resolves once every stop click made so far is answered (at once when none is out), or when none has been answered
 * for `STOPS_GIVE_UP_MS`. */
export async function afterStopEdits(): Promise<void> {
  while (stopsQueued.peek() > 0 && !stopsStuck()) {
    await Promise.race([stopQueue, new Promise((r) => setTimeout(r, STOPS_GIVE_UP_MS - (performance.now() - stopsMoved)))]);
  }
}

/** T-4214: whether a map click waits for the map or is being carried out (map/toolWait.ts) */
let clicksBusy: () => boolean = () => false;
let clickWaiters: (() => void)[] = [];
/** a waiter gives up on the clicks after this long, ms (a click waits for the map at most 5 s, then for its edit's answer) */
const CLICKS_GIVE_UP_MS = 10000;
export function setClicksBusy(f: () => boolean) { clicksBusy = f; }
/** map/toolWait.ts: the clicks moved on */
export function clicksMoved() {
  if (!clickWaiters.length || clicksBusy()) return;
  const w = clickWaiters;
  clickWaiters = [];
  for (const f of w) f();
}
function afterClicks(): Promise<unknown> {
  return Promise.race([new Promise<void>((r) => clickWaiters.push(r)), new Promise((r) => setTimeout(r, CLICKS_GIVE_UP_MS))]);
}

/** Send an edit; a refusal is explained in `notice`. Resolves with the answer either way. `silent`:
 * its sound was played already (the click that sent it, or the one sound of a run of edits). `game`: the game stamp
 * (clockClient `gameStamp`) of an edit sent later than it was made, a queued stop edit's: refused under another. */
export async function edit(op: EditOp, quiet = false, silent = false, early: boolean | "rows" | "click" = false, game?: number): Promise<EditAnswer> {
  if (tracing()) {
    const snapshot = world.peek();
    trace("edit-click", { edit: editTrace(op), version: snapshot?.version,
      lines: snapshot?.save.lines.length, stations: snapshot?.save.stations.length,
      lineStops: "line" in op ? snapshot?.save.lines.find((l) => l.num === op.line)?.stops.slice(0, 256) : undefined });
  }
  breakDeleteRun();
  if (!silent) beforeEdit?.(op);
  if (!sink) return { ok: false, issues: [], charge: 0 };
  // T-3731: an undo or redo takes back the last stop click shown, so it waits for the clicks still out; T-4214: and the
  // last map click, waiting for the map or being carried out
  if ((op.op === "undo" || op.op === "redo") && stopsQueued.peek() > 0) await afterStopEdits();
  if ((op.op === "undo" || op.op === "redo") && clicksBusy()) await afterClicks();
  noteEdit(op, world.peek());
  if (op.op !== "fares" && op.op !== "undo" && op.op !== "redo") editsSent.value++;
  // `early` (T-1407): the answer may come before `world` holds the edit (`a.early`); "rows" (T-2766, a stop click):
  // the answer comes with `world` holding the edit, its network a moment later; "click" (T-4214, a station or a delete
  // clicked on the map): the answer may come once the map the tools pick on holds it (map/clickMap.ts), before `world`
  const a = await sink!(op, early, game);
  if (!a.ok && !quiet && a.issues.length) say(describeIssues(a.issues, buysOf(op), lineVehicle((op as { line?: number }).line)), true);
  else if (a.ok && !silent) afterEdit?.(op, a);
  return a;
}

const TRAIN_OPS = new Set(["setStops", "insertStop", "toggleStop", "oneWayStop", "dropStop", "endLineAt", "stopPassed", "orderStops", "sortStops", "turnEnd", "openLoop", "setTrainCount", "setHeadway", "weekendShare", "weekendCounts", "weekendShareAll", "setPeriods", "setHours", "setRowTrains", "setRowHeadways", "rowWeekend", "lineCars", "copyLine", "undo", "redo"]);
const BUILD_OPS = new Set(["route", "construct", "constructAll", "constructLine", "addStation", "addBrtStation", "addPort", "crossover", "doubleUp", "stationUpgrade", "rebuild"]);
/** What a refused-for-money edit would have paid for: line edits cost trains only, track edits construction. */
function buysOf(op: EditOp): Buys | undefined {
  return TRAIN_OPS.has(op.op) ? "trains" : BUILD_OPS.has(op.op) ? "construction" : undefined;
}

const num = (id: LineId) => Number(id);

/** What a line runs, by its id or number (a train where it is unknown), for words about buying its vehicles. */
const lineVehicle = (line: LineId | number | undefined): Vehicle => { const st = line === undefined ? undefined : world.peek()?.lineStats[String(line)]; return st ? vehicleOf(st) : "train"; };

/** Train counts asked for but not yet in `world` (T-229): the stepper shows them at once, and a
 * second click before the first edit lands counts from them, not from the old figure. */
export const pendingTrains = signal<ReadonlyMap<LineId, [number, number, number]>>(new Map());

/** A line's trains on the entire line per demand level, as the player last set them. */
export function trainsShown(id: LineId): [number, number, number] | null {
  const w = world.value, st = w?.lineStats[id];
  return pendingTrains.value.get(id) ?? (st ? pairStats(w!, st).trainsNeeded as [number, number, number] : null);
}

/** Step a line's trains for one demand level, `by` from what it shows now (T-229). Clicks while an
 * edit is in flight are folded into one edit sent when it lands, so fast clicking never queues a
 * recompute per click. */
export function stepTrainCount(id: LineId, demand: Demand, by: number): boolean {
  return changeTrains(id, demand, (now, k) => Math.max(0, Math.round(now / k + by) * k));
}

export function setTrainCount(id: LineId, demand: Demand, trains: number): boolean {
  return changeTrains(id, demand, (_, k) => Math.max(0, Math.ceil(Math.round(trains) / k) * k));
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

// ---- T-3521: weekend trains (`LineStats.weekend`): a percent of the weekdays' or counts set by
// hand; the sim settles weekends on the track on their own and buys what the busier kind of day needs.

/** The percents a line's weekends may run at, and the step (sim params.rs `WEEKEND_SHARE_MIN`). */
export const WEEKEND_SHARE_MIN = 10, WEEKEND_SHARE_MAX = 200, WEEKEND_SHARE_STEP = 5;

/** A line's weekends as they show: its percent, whether its counts are set by hand (the percent
 * then no longer says what runs), and its whole trains per level on weekends. */
export interface WeekendShown { share: number; custom: boolean; counts: [number, number, number]; headways?: [number, number, number] }

/** Weekend edits sent and not yet in the world, as `pendingTrains`: a second click steps from the
 * first's. */
const pendingWeekend = signal<ReadonlyMap<LineId, WeekendShown>>(new Map());

/** A line's weekends as they show (`WeekendShown`); null without figures. A count line following
 * its weekdays at a percent follows a weekday count still on its way too. */
export function weekendShown(id: LineId): WeekendShown | null {
  const p = pendingWeekend.value.get(id);
  if (p) return p;
  const st = world.value?.lineStats[id], wk = trainsShown(id);
  if (!st || !wk) return null;
  const x = st.weekend, share = x ? x[0] : 100, custom = !!x && x[1] === 1;
  const sim = x ? ([x[2], x[3], x[4]] as [number, number, number]) : wk;
  const headways = x && x.length >= 8 && (x[5] || x[6] || x[7]) ? ([x[5], x[6], x[7]] as [number, number, number]) : undefined;
  if (custom || st.autoS || !pendingTrains.value.has(id)) return { share, custom, counts: sim, headways };
  return { share, custom, counts: wk.map((n) => weekendCount(n, share)) as [number, number, number] };
}

/** A weekday count at a percent: rounded up (sim world.rs `weekend_counts`). */
export const weekendCount = (n: number, share: number) => Math.ceil((n * share) / 100 - 1e-9);

/** A line's trains per level on kind of day `row` (0 weekdays, 1 weekends) as they show. */
export function restShown(id: LineId, row: number): [number, number, number] | null {
  return row === 0 ? trainsShown(id) : weekendShown(id)?.counts ?? null;
}

function sendWeekend(id: LineId, next: WeekendShown, op: EditOp) {
  pendingWeekend.value = new Map(pendingWeekend.value).set(id, next);
  void edit(op, false, true).finally(() => {
    if (pendingWeekend.peek().get(id) !== next) return;
    const m = new Map(pendingWeekend.peek());
    m.delete(id);
    pendingWeekend.value = m;
  });
}

/** Run a line's weekends at `share` percent of its weekdays (kept between the stepper's ends), its
 * counts set by hand dropped; false when that changes nothing (the stepper shakes). */
export function setWeekendShare(id: LineId, share: number): boolean {
  const cur = weekendShown(id), wk = trainsShown(id);
  if (!cur || !wk) return false;
  const s = Math.max(WEEKEND_SHARE_MIN, Math.min(WEEKEND_SHARE_MAX, Math.round(share)));
  if (s === cur.share && !cur.custom) return false;
  sendWeekend(id, { share: s, custom: false, counts: wk.map((n) => weekendCount(n, s)) as [number, number, number] }, { op: "weekendShare", line: num(id), share: s });
  return true;
}

/** Set a line's weekend trains at a level by hand, kept between none and `most` (`trainCap`, as a
 * weekday count); false when that changes nothing. */
export function setWeekendCount(id: LineId, demand: Demand, want: number, most: number): boolean {
  const cur = weekendShown(id);
  if (!cur) return false;
  const i = DEMAND_INDEX[demand], to = Math.max(0, Math.min(most, want));
  if (to === cur.counts[i]) return false;
  const counts = [...cur.counts] as [number, number, number];
  counts[i] = to;
  // a headway stepped to stays at the levels this leaves alone (sim `set_line_weekend_counts`)
  const headways = cur.headways?.map((h, k) => (k === i ? 0 : h)) as [number, number, number] | undefined;
  sendWeekend(id, { share: cur.share, custom: true, counts, headways: headways?.some(Boolean) ? headways : undefined }, { op: "weekendCounts", line: num(id), counts });
  return true;
}

/** A weekday headway at a weekend percent, to a whole minute from a minute up (sim `weekend_headway`). */
export function weekendHeadwayAt(headwayS: number, share: number): number {
  if (!(headwayS > 0) || !(share > 0)) return 0;
  const raw = Math.floor((headwayS * 100) / share);
  return raw >= 60 ? Math.floor(raw / 60) * 60 : Math.max(1, raw);
}

/** The trains an hour the weekends' row shows at level `lev` with `n` trains. On a headway line it
 * is the stepped headway the weekends are asked to run, as the weekdays' row shows its stored one:
 * the hand-set headway, or the weekday headway at the percent (sim `weekend_headway`). The settled
 * rate (`restTph`) is a decimal wherever the train count cannot hit the headway, and the round
 * trip over a count still on its way is another; neither is shown there (T-3603). */
export function restTphShown(id: LineId, st: LineStats, lev: number, n: number): number {
  const trip = st.roundTripByLevel[lev];
  const wk = weekendShown(id), auto = headwayShown(id);
  if (n > 0 && wk) {
    const given = wk.custom ? wk.headways?.[lev] ?? 0 : auto ? weekendHeadwayAt(auto[lev], wk.share) : 0;
    if (given > 0) return 3600 / given;
  }
  const ran = st.weekend ? st.weekend[2 + lev] : st.trainsNeeded[lev];
  if (!pendingWeekend.value.has(id) && n === ran && st.restTph) return st.restTph[lev];
  return trip > 0 ? (n * 3600) / trip : 0;
}

/** The headway the weekends' row at level `lev` shows with `n` trains, to a whole minute (seconds); 0 none. */
function shownRestHeadway(id: LineId, lev: number, n: number): number {
  const st = world.value?.lineStats[id];
  if (!st || !(n > 0)) return 0;
  const tph = restTphShown(id, st, lev, n);
  const h = tph > 0 ? 3600 / tph : 0;
  return h >= 60 ? Math.round(h / 60) * 60 : Math.round(h);
}

/** Set a line's weekend trains at a level to `to` and the headway they run (seconds, 0 none): the
 * weekends then run that headway (T-3557). False when that changes nothing. */
function setWeekendHeadway(id: LineId, demand: Demand, to: number, h: number, most: number): boolean {
  const cur = weekendShown(id);
  if (!cur) return false;
  const i = DEMAND_INDEX[demand], n = Math.max(0, Math.min(most, to)), sec = n > 0 ? Math.round(h) : 0;
  if (cur.custom && n === cur.counts[i] && sec === (cur.headways?.[i] ?? 0)) return false;
  const counts = [...cur.counts] as [number, number, number];
  counts[i] = n;
  // a level left alone keeps the headway it shows, in whole minutes, not the count's round trip share (T-3588)
  const headways = [0, 1, 2].map((k) => cur.headways?.[k] || shownRestHeadway(id, k, cur.counts[k])) as [number, number, number];
  headways[i] = sec;
  sendWeekend(id, { share: cur.share, custom: true, counts, headways }, { op: "weekendCounts", line: num(id), counts, headways });
  return true;
}

/** Step a line's weekend trains at a level by `by` steps (`trainStep`). */
export function stepWeekendCount(id: LineId, demand: Demand, by: number, most: number): boolean {
  const n = weekendShown(id)?.counts[DEMAND_INDEX[demand]];
  return n !== undefined && setWeekendCount(id, demand, n + by * trainStep(id), most);
}

/** Step the headway a line's weekend count gives at a level (`dir` 1 longer, -1 shorter), by the
 * weekdays' headway steps (`nextHeadway`), to the count that keeps it at the weekdays' round trip
 * `trip` (`autoTrainsFor`), and sends the headway with them so the weekends run it (T-3557). */
export function stepWeekendHeadway(id: LineId, demand: Demand, dir: number, trip: number, limitS: number, loop: boolean, most: number, shown: number): boolean {
  const n = weekendShown(id)?.counts[DEMAND_INDEX[demand]];
  const step = n === undefined ? null : restHeadwayStep(n, dir, trip, limitS, loop, shown);
  return step !== null && setWeekendHeadway(id, demand, step.to, step.h, most);
}

/** Where a weekend headway step (`stepWeekendHeadway`) goes from `n` trains: the count and the
 * headway it was stepped to (0 none); null where it cannot step. `dir`'s size is minutes a step, as
 * the weekdays' (Shift, Ctrl). It steps from `shown`, the headway the row shows (the rate settled
 * for weekends, which a full track or held trains make longer than the round trip over the count):
 * from the shorter one, + could land below what is shown and read backwards. */
export function restHeadwayStep(n: number, dir: number, trip: number, limitS: number, loop: boolean, shown: number): { to: number; h: number } | null {
  if (trip <= 0 || dir === 0) return null;
  const up = dir > 0;
  const now = shown > 0 ? shown : n > 0 ? trip / n : 0;
  // past the longest headway is no service; a step shorter than the shortest is none
  if (up && now >= HEADWAY_MAX_S) return n > 0 ? { to: 0, h: 0 } : null;
  const h = nextHeadway(now, Math.sign(dir), Math.abs(dir));
  if (h <= 0 || h === now) return null;
  return { to: autoTrainsFor(trip, h, limitS, loop), h };
}

/** A weekend headway typed in minutes at a level: it runs that headway (to 5 s, as the weekdays'), with the
 * count it needs; 0 or less is no service. */
export function typeWeekendHeadway(id: LineId, demand: Demand, minutes: number, trip: number, limitS: number, loop: boolean, most: number): boolean {
  const h = headwayFromMinutes(minutes);
  return setWeekendHeadway(id, demand, h > 0 ? autoTrainsFor(trip, h, limitS, loop) : 0, h, most);
}

/** Run every line's weekends at `share` percent of its weekdays, counts set by hand dropped: one
 * undoable edit. */
export function setWeekendShareAll(share: number): void {
  const s = Math.max(WEEKEND_SHARE_MIN, Math.min(WEEKEND_SHARE_MAX, Math.round(share)));
  void edit({ op: "weekendShareAll", share: s });
}

/** The most vehicles a count is raised to per level, when the track sets no lower limit (T-3152). */
export const MAX_TYPED_VEHICLES = 1000;

/** T-3165: the most a hand-set count can be at a level: the whole trains the track lets run (the
 * round trip over its shortest headway `limitS`, a loop's even, one at least) plus one step, so one
 * train is held and the held messages and red headway show. 1000 with no limit. */
export function trainCap(st: Pick<LineStats, "loop" | "oneWay">, tripS: number, limitS: number | null): number {
  const k = loopHalves(st) ? 2 : 1;
  if (!limitS || !(tripS > 0)) return MAX_TYPED_VEHICLES;
  const room = Math.floor(tripS / limitS + 0.01);
  return Math.min(MAX_TYPED_VEHICLES, Math.max(k, room - (room % k)) + k);
}

/** The step of a line's train count: 2 on a loop run both ways, whose trains are half each way (T-798). */
export function trainStep(id: LineId): number {
  const st = world.value?.lineStats[id];
  // T-3333: a line on both pairs runs as many on each
  return st && (loopHalves(st) || st.pairs === 1) ? 2 : 1;
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
  const w = world.peek(), st = w?.lineStats[id], had = st && pairStats(w!, st).trainsNeeded;
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
    say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains", lineVehicle(id)), true); // i18n-ignore
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
  const short = checkCash && st && to > st.cars ? carBuyCheck?.(id, (to - st.cars) * fleetOf(st)) : null;
  if (short) {
    say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains", lineVehicle(id)), true); // i18n-ignore
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
  const w = world.value, st = w?.lineStats[id];
  return st ? pairStats(w!, st).autoS : null;
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

/** All three levels' headways at once, as one edit (a line on counts is switched to Headway by it). */
export function setHeadways(id: LineId, secs: [number, number, number]) {
  queueHeadway(id, [...secs]);
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
      say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains", lineVehicle(id)), true); // i18n-ignore
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
  if (!on) {
    queueHeadway(id, null);
    return;
  }
  const seconds = headwaysOfTrains(id);
  if (seconds) queueHeadway(id, seconds);
}

/** The headways a line's trains give now, so a switch to Headway changes no train (T-3588 shares it). */
export function headwaysOfTrains(id: LineId): [number, number, number] | null {
  const st = world.value?.lineStats[id];
  if (!st) return null;
  const n = trainsShown(id) ?? st.trainsNeeded;
  return n.map((k, lev) => headwayOfTrains(st, st.roundTripByLevel[lev], k)) as [number, number, number];
}

/** The headway `k` trains give over round trip `trip`, rounded to the coarsest of 1 minute, 30, 10 or
 * 5 seconds that keeps them; 0 for none. */
function headwayOfTrains(st: LineStats, trip: number, k: number): number {
  if (!(k > 0) || !(trip > 0)) return 0;
  const exact = Math.ceil(trip / k);
  for (const step of [60, 30, 10, 5]) {
    const h = Math.ceil(exact / step) * step;
    if (autoTrainsFor(trip, h, 0, loopHalves(st)) === k) return h;
  }
  return exact;
}

// ---- T-3542: rows. A line sets its trains in 3, 4 or 5 rows (game/periods.ts `ROWS_OF`). One of
// three edits through the functions above (its rows are high, medium and low); one of four or five
// through the row edits (`setRowTrains`, `setRowHeadways`, `rowWeekend`), every value an array in the
// day's order. The functions below take a row index and pick the way by the line.

/** A line's busiest row's trains (its own, without a hidden twin's): the fleet it runs. */
export const fleetOf = (st: LineStats): number => Math.max(...(st.rows?.trains ?? st.trainsNeeded));

/** A line's rows as its panels show them (3, 4 or 5). */
export const rowCount = (st: LineStats | undefined): number => st?.periods ?? 3;

/** A line of four or five rows' figures with its hidden twin's trains added (T-3333, as `pairStats`;
 * the sim's row headways and weekends are whole-line already); undefined for a line of three. */
export function pairRows(w: WorldView, st: LineStats): RowFigs | undefined {
  const r = st.rows, tw = r && st.twin ? w.lineStats[st.twin]?.rows : undefined;
  if (!r || !tw) return r;
  return { ...r, trains: r.trains.map((n, i) => n + (tw.trains[i] ?? 0)), tph: r.tph.map((x, i) => x + (tw.tph[i] ?? 0)) };
}

const pendingRowTrains = signal<ReadonlyMap<LineId, number[]>>(new Map());
const pendingRowHeadway = signal<ReadonlyMap<LineId, number[] | null>>(new Map());
const pendingRowWeekend = signal<ReadonlyMap<LineId, WeekendRows>>(new Map());

/** A line's trains per row as the player last set them (`trainsShown` for a line of three). */
export function rowTrainsShown(id: LineId): number[] | null {
  const w = world.value, st = w?.lineStats[id];
  if (!st?.rows) return trainsShown(id);
  const p = pendingRowTrains.value.get(id);
  return p && p.length === st.periods ? p : pairRows(w!, st)!.trains;
}

/** A line's headways per row as last set, or null on counts (`headwayShown` for a line of three). */
export function rowHeadwayShown(id: LineId): number[] | null {
  const w = world.value, st = w?.lineStats[id];
  if (!st?.rows) return headwayShown(id);
  const p = pendingRowHeadway.value;
  if (p.has(id)) {
    const v = p.get(id) ?? null;
    if (!v || v.length === st.periods) return v;
  }
  return st.rows.auto;
}

const sendRowTrains = folded(pendingRowTrains, async (id, trains) => {
  const had = world.peek()?.lineStats[id]?.rows?.trains;
  const a = await edit({ op: "setRowTrains", line: num(id), trains }, false, true);
  noteAnswer(id, a, trains, had);
  return a.ok;
});
const sendRowHeadway = folded(pendingRowHeadway, async (id, secs) => {
  const had = world.peek()?.lineStats[id]?.rows?.auto;
  const a = await edit(secs ? { op: "setRowHeadways", line: num(id), seconds: secs } : { op: "setHeadway", line: num(id), seconds: null }, false, true);
  if (secs) noteAnswer(id, a, secs, had);
  return a.ok;
});

/** A line of four or five rows' new counts, as `setTrainCounts`: refused here when the cash cannot cover them. */
function setRowCounts(id: LineId, trains: number[]) {
  const was = rowTrainsShown(id);
  const short = trainBuyCheck?.(id, trains);
  if (short) {
    refusedForCash.value = { id, levs: trains.map((v, i) => v !== was?.[i]), at: performance.now(), need: short.need };
    say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains", lineVehicle(id)), true); // i18n-ignore
    return;
  }
  if (!was || trains.some((n, i) => n !== was[i])) clickSound({ op: "setTrainCount" });
  sendRowTrains(id, trains);
}

function changeRow(id: LineId, row: number, to: (now: number, step: number) => number): boolean {
  const base = rowTrainsShown(id);
  if (!base) return true;
  const t = [...base], k = trainStep(id);
  t[row] = to(t[row], k);
  if (t[row] === base[row]) return false;
  setRowCounts(id, t);
  return true;
}

/** Step a line's trains in row `row` (`stepTrainCount`). */
export function stepRowCount(id: LineId, row: number, by: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return stepTrainCount(id, DEMANDS[row], by);
  return changeRow(id, row, (now, k) => Math.max(0, Math.round(now / k + by) * k));
}
/** Set a line's trains in row `row` (`setTrainCount`). */
export function setRowCount(id: LineId, row: number, trains: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return setTrainCount(id, DEMANDS[row], trains);
  return changeRow(id, row, (_, k) => Math.max(0, Math.ceil(Math.round(trains) / k) * k));
}

/** The trains a row headway edit leaves each row with, as `headwayTrains`, or null when the panel cannot tell. */
function rowHeadwayTrains(id: LineId, secs: readonly number[], was: readonly number[] | null): number[] | null {
  const st = world.peek()?.lineStats[id], r = st?.rows, lim = headwayLimits.get(id);
  if (!st || !r || st.bus || !lim) return null;
  const out: number[] = [];
  for (let i = 0; i < secs.length; i++) {
    if (secs[i] === was?.[i]) out.push(r.trains[i]);
    else if (secs[i] * AUTO_SLACK < lim[i]) return null;
    else out.push(autoTrainsFor(r.roundTrip[i], secs[i], lim[i], loopHalves(st)));
  }
  return out;
}
function queueRowHeadway(id: LineId, secs: number[] | null) {
  const was = rowHeadwayShown(id);
  const trains = secs && trainBuyCheck ? rowHeadwayTrains(id, secs, was) : null;
  if (trains) {
    const short = trainBuyCheck!(id, trains);
    if (short) {
      refusedForCash.value = { id, levs: secs!.map((v, i) => v !== was?.[i]), at: performance.now(), need: short.need };
      say(describeIssues([`Money:${short.need.toFixed(1)}:${short.have.toFixed(1)}`], "trains", lineVehicle(id)), true); // i18n-ignore
      return;
    }
  }
  if (JSON.stringify(secs) !== JSON.stringify(was)) clickSound({ op: "setHeadway" });
  sendRowHeadway(id, secs);
}

/** Set a line's headway in row `row` (`setHeadway`). */
export function setRowHeadway(id: LineId, row: number, seconds: number) {
  if (!world.value?.lineStats[id]?.rows) return setHeadway(id, DEMANDS[row], seconds);
  const base = rowHeadwayShown(id);
  if (!base) return;
  const h = [...base];
  h[row] = seconds;
  queueRowHeadway(id, h);
}
/** Step a line's headway in row `row` (`stepHeadway`). */
export function stepRowHeadway(id: LineId, row: number, dir: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return stepHeadway(id, DEMANDS[row], dir);
  const base = rowHeadwayShown(id);
  if (!base) return true;
  const now = base[row], to = nextHeadway(now, Math.sign(dir), Math.abs(dir));
  if (to === now) return false;
  setRowHeadway(id, row, to);
  return true;
}
/** Every row's headway at once, any number of rows (`setHeadways`), as one edit (T-3767: a default's). */
export function setRowHeadways(id: LineId, secs: readonly number[]) {
  if (!world.value?.lineStats[id]?.rows) return setHeadways(id, [secs[0], secs[1], secs[2]]);
  queueRowHeadway(id, [...secs]);
}
/** The headways per row a line's trains give now (`headwaysOfTrains`, any number of rows). */
export function rowHeadwaysOfTrains(id: LineId): number[] | null {
  const st = world.value?.lineStats[id];
  if (!st?.rows) return headwaysOfTrains(id);
  const n = rowTrainsShown(id) ?? st.rows.trains;
  return n.map((k, i) => headwayOfTrains(st, st.rows!.roundTrip[i], k));
}
/** Switch a line between counts and a headway (`setAutoTrains`), any number of rows. */
export function setRowAuto(id: LineId, on: boolean) {
  const st = world.value?.lineStats[id];
  if (!st?.rows) return setAutoTrains(id, on);
  if (!on) return queueRowHeadway(id, null);
  queueRowHeadway(id, rowHeadwaysOfTrains(id)!);
}

/** A line's weekends per row as they show (`WeekendShown` for any number of rows). */
export interface WeekendRows { share: number; custom: boolean; counts: number[]; headways?: number[] }

/** A line's weekends as they show, per row (`weekendShown` for a line of three). */
export function rowWeekendShown(id: LineId): WeekendRows | null {
  const st = world.value?.lineStats[id];
  if (!st?.rows) return weekendShown(id);
  const p = pendingRowWeekend.value.get(id);
  if (p && p.counts.length === st.periods) return p;
  const n = st.periods, x = st.rows.weekend, wk = rowTrainsShown(id) ?? st.rows.trains;
  const share = x[0] ?? 100, custom = x[1] === 1, counts = x.slice(2, 2 + n), hw = x.slice(2 + n, 2 + 2 * n);
  if (custom || st.rows.auto || !pendingRowTrains.value.has(id)) return { share, custom, counts, headways: hw.some(Boolean) ? hw : undefined };
  return { share, custom, counts: wk.map((k) => weekendCount(k, share)) };
}

function sendRowWeekend(id: LineId, next: WeekendRows, op: EditOp) {
  pendingRowWeekend.value = new Map(pendingRowWeekend.value).set(id, next);
  void edit(op, false, true).finally(() => {
    if (pendingRowWeekend.peek().get(id) !== next) return;
    const m = new Map(pendingRowWeekend.peek());
    m.delete(id);
    pendingRowWeekend.value = m;
  });
}

/** Run a line's weekends at `share` percent (`setWeekendShare`, any number of rows). */
export function setRowWeekendShare(id: LineId, share: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return setWeekendShare(id, share);
  const cur = rowWeekendShown(id), wk = rowTrainsShown(id);
  if (!cur || !wk) return false;
  const s = Math.max(WEEKEND_SHARE_MIN, Math.min(WEEKEND_SHARE_MAX, Math.round(share)));
  if (s === cur.share && !cur.custom) return false;
  sendRowWeekend(id, { share: s, custom: false, counts: wk.map((n) => weekendCount(n, s)) }, { op: "weekendShare", line: num(id), share: s });
  return true;
}

/** Set a line's weekend trains in row `row` (`setWeekendCount`). */
export function setRowWeekendCount(id: LineId, row: number, want: number, most: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return setWeekendCount(id, DEMANDS[row], want, most);
  const cur = rowWeekendShown(id);
  if (!cur) return false;
  const to = Math.max(0, Math.min(most, want));
  if (to === cur.counts[row]) return false;
  const counts = [...cur.counts];
  counts[row] = to;
  const headways = cur.headways?.map((h, k) => (k === row ? 0 : h));
  sendRowWeekend(id, { share: cur.share, custom: true, counts, headways: headways?.some(Boolean) ? headways : undefined }, { op: "rowWeekend", line: num(id), counts, headways: [] });
  return true;
}
/** Step a line's weekend trains in row `row` (`stepWeekendCount`). */
export function stepRowWeekendCount(id: LineId, row: number, by: number, most: number): boolean {
  const n = rowWeekendShown(id)?.counts[row];
  return n !== undefined && setRowWeekendCount(id, row, n + by * trainStep(id), most);
}

/** The trains an hour a line's weekend row `row` shows with `n` trains (`restTphShown`). */
export function restRowTphShown(id: LineId, st: LineStats, row: number, n: number): number {
  const r = st.rows;
  if (!r) return restTphShown(id, st, row, n);
  const trip = r.roundTrip[row], wk = rowWeekendShown(id), auto = rowHeadwayShown(id);
  if (n > 0 && wk) {
    const given = wk.custom ? wk.headways?.[row] ?? 0 : auto ? weekendHeadwayAt(auto[row], wk.share) : 0;
    if (given > 0) return 3600 / given;
  }
  const ran = r.weekend[2 + row];
  if (!pendingRowWeekend.value.has(id) && n === ran) return r.restTph[row];
  return trip > 0 ? (n * 3600) / trip : 0;
}

/** Set a line's weekend trains in row `row` to `to` with the headway they run (`setWeekendHeadway`). */
function setRowWeekendHeadway(id: LineId, row: number, to: number, h: number, most: number): boolean {
  const st = world.value?.lineStats[id], cur = rowWeekendShown(id);
  if (!st || !cur) return false;
  const n = Math.max(0, Math.min(most, to)), sec = n > 0 ? Math.round(h) : 0;
  if (cur.custom && n === cur.counts[row] && sec === (cur.headways?.[row] ?? 0)) return false;
  const counts = [...cur.counts];
  counts[row] = n;
  const shown = (k: number, c: number) => { const tph = c > 0 ? restRowTphShown(id, st, k, c) : 0, s = tph > 0 ? 3600 / tph : 0; return s >= 60 ? Math.round(s / 60) * 60 : Math.round(s); };
  const headways = cur.counts.map((c, k) => cur.headways?.[k] || shown(k, c));
  headways[row] = sec;
  sendRowWeekend(id, { share: cur.share, custom: true, counts, headways }, { op: "rowWeekend", line: num(id), counts, headways });
  return true;
}
/** Step a line's weekend headway in row `row` (`stepWeekendHeadway`). */
export function stepRowWeekendHeadway(id: LineId, row: number, dir: number, trip: number, limitS: number, loop: boolean, most: number, shown: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return stepWeekendHeadway(id, DEMANDS[row], dir, trip, limitS, loop, most, shown);
  const n = rowWeekendShown(id)?.counts[row];
  const step = n === undefined ? null : restHeadwayStep(n, dir, trip, limitS, loop, shown);
  return step !== null && setRowWeekendHeadway(id, row, step.to, step.h, most);
}
/** A weekend headway typed in row `row` (`typeWeekendHeadway`). */
export function typeRowWeekendHeadway(id: LineId, row: number, minutes: number, trip: number, limitS: number, loop: boolean, most: number): boolean {
  if (!world.value?.lineStats[id]?.rows) return typeWeekendHeadway(id, DEMANDS[row], minutes, trip, limitS, loop, most);
  const h = headwayFromMinutes(minutes);
  return setRowWeekendHeadway(id, row, h > 0 ? autoTrainsFor(trip, h, limitS, loop) : 0, h, most);
}

/** Set the game's rows (`line` undefined) or line `line`'s own (0: the game's). */
export function setPeriods(rows: number, line?: LineId) {
  void edit(line === undefined ? { op: "setPeriods", rows } : { op: "setPeriods", line: num(line), rows });
}

/** T-3542: set the game's hours (`line` undefined) or line `line`'s own (five, game/periods.ts `Cuts`). */
export function setHours(cuts: number[], line?: LineId) {
  void edit(line === undefined ? { op: "setHours", cuts } : { op: "setHours", line: num(line), cuts });
}

/** The fare curve for the whole network (T-883, game/fareCurve.ts): its nodes, km and US$. */
export function setFares(nodes: FareNode[]) {
  return edit({ op: "fares", nodes: tidyNodes(nodes) });
}
/** T-1277: a country's fare scale, null for its income's. */
export function setFareScale(country: string, scale: number | null) {
  return edit({ op: "fareScale", country, scale });
}
/** T-3766: a city's fare scale (zone id), null for its default. */
export function setFareCityScale(city: string, scale: number | null) {
  return edit({ op: "fareCityScale", city, scale });
}
/** T-3766: fares by city (true) or by country. */
export function setFareByCity(on: boolean) {
  return edit({ op: "fareByCity", on });
}
/** T-2679: the reference country, whose fares the Money tab's multiples are of. */
export function setFareRef(country: string) {
  return edit({ op: "fareRef", country });
}
/** T-1626: the fare multiplier on high-speed lines. */
export function setFareHsr(mul: number) {
  return edit({ op: "fareHsr", mul });
}
/** T-2194: the fare multiplier on ferry lines; T-3892: on those of one boat size (Medium unless given). */
export function setFareFerry(mul: number, boat: 0 | 1 | 2 = 1) {
  return edit({ op: "fareFerry", mul, ...(boat !== 1 ? { boat } : {}) });
}
/** T-3028: the fare multiplier on bus lines; T-3915: on BRT lines with `brt`. */
export function setFareBus(mul: number, brt = false) {
  return edit({ op: "fareBus", mul, ...(brt ? { brt: true as const } : {}) });
}
/** T-3567: line `line`'s own fare multiplier. */
export function setLineFare(line: number, mul: number) {
  return edit({ op: "fareLine", line, mul });
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
export const stationsById = (w: WorldView) => lookup(w, "stations", (s) => { const m = new Map<string, StationInput>(); for (const x of s.stations) m.set(x.id, x); return m; });
const stopMemo = new WeakMap<WorldView, { byId: Map<string, StationInput>; names: Map<string, string>; bus: Map<string, StationInput>; custom: Map<string, string> }>();
/** T-1464: the stations and a bus line's other stops (no station, no name of their own) by id: a stop's
 * place for the figures, its words for the panels. A bus stop is named for the station within 1 km
 * (the nearest), else "Bus stop". `stationsById` stays stations only. */
const BUS_NAME_M = 1000;
/** T-4125: named stations in squares of this many world metres, per stations list (kept while no station changed) */
const NAMED_CELL = 2048;
const namedGrids = new WeakMap<readonly StationInput[], Map<number, number[]>>();
const cellKey = (cx: number, cy: number) => cx * 0x200000 + cy;
/** The named stations within `r` world metres of (x, y) on each axis, perhaps with others, in the list's order. */
function namedNear(stations: readonly StationInput[], x: number, y: number, r: number): StationInput[] {
  let g = namedGrids.get(stations);
  if (!g) {
    g = new Map();
    stations.forEach((s, i) => {
      if (!s.name) return;
      const k = cellKey(Math.floor(s.x / NAMED_CELL), Math.floor(s.y / NAMED_CELL));
      (g!.get(k) ?? g!.set(k, []).get(k)!).push(i);
    });
    namedGrids.set(stations, g);
  }
  const out: number[] = [];
  for (let cx = Math.floor((x - r) / NAMED_CELL); cx <= Math.floor((x + r) / NAMED_CELL); cx++)
    for (let cy = Math.floor((y - r) / NAMED_CELL); cy <= Math.floor((y + r) / NAMED_CELL); cy++) out.push(...(g.get(cellKey(cx, cy)) ?? []));
  return out.sort((a, b) => a - b).map((i) => stations[i]);
}
/** A bus line's stops that are no station, each with its name, and the box (world m) a station must be in to change one */
type BusNamed = { bus: NonNullable<LineStats["bus"]>; plain: string; keeps: Save["keeps"]; box: [number, number, number, number]; stops: { id: string; stop: StationInput; name: string; custom: boolean }[] };
/** T-4080: a bus stop takes the name of the nearest named kept stop this near (`bus::KEEP_M` in the sim), ground m */
const KEEP_M = 50;
function namedBus(w: WorldView, b: NonNullable<LineStats["bus"]>, plain: string): BusNamed {
  const stops: BusNamed["stops"] = [];
  const box: BusNamed["box"] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const p of b.stops) {
    if (p.station >= 0) continue;
    const id = String(p.id);
    const stop: StationInput = { id, num: p.id, name: "", lng: p.lng, lat: p.lat, x: p.x, y: p.y, level: 0, length: 0, built: b.built, heading: 0 };
    const r = BUS_NAME_M * groundScale(p.y) * 1.1;
    let kept = "", away = KEEP_M;
    if (w.save.keeps) {
      const rk = KEEP_M * groundScale(p.y) * 1.1;
      for (const k of w.save.keeps) {
        if (Math.abs(k.x - p.x) > rk || Math.abs(k.y - p.y) > rk) continue;
        const d = groundDist(k.x, k.y, p.x, p.y);
        if (d <= away) { away = d; kept = k.name; }
      }
    }
    let near = "", best = 1000;
    if (!kept) for (const s of namedNear(w.save.stations, p.x, p.y, r)) {
      const d = groundDist(s.x, s.y, p.x, p.y);
      if (d < best) { best = d; near = s.name; }
    }
    stops.push({ id, stop, name: kept || (near ? t("world.busStopNear", { name: near }) : plain), custom: !!kept });
    box[0] = Math.min(box[0], p.x - r); box[1] = Math.min(box[1], p.y - r); box[2] = Math.max(box[2], p.x + r); box[3] = Math.max(box[3], p.y + r);
  }
  return { bus: b, plain, keeps: w.save.keeps, box, stops };
}
/** T-4150: the last world's bus stops by line, kept for a line while its route is the same and no station changed in its box */
let lastBusNamed: { w: WorldView; lines: Map<string, BusNamed> } | null = null;

function stopsOf(w: WorldView) {
  let m = stopMemo.get(w);
  if (m) return m;
  const byId = new Map<string, StationInput>(), names = new Map<string, string>();
  for (const x of w.save.stations) { byId.set(x.id, x); names.set(x.id, x.name); }
  const bus = new Map<string, StationInput>(), custom = new Map<string, string>();
  const was = freshNow() ? null : lastBusNamed, d = was ? worldDiff(was.w, w) : null;
  // where the stations that changed stand, before and after
  const moved: [number, number][] = [];
  if (d) for (const pair of d.stations.values()) for (const s of pair) if (s) moved.push([s.x, s.y]);
  const plain = t("world.busStop"), lines = new Map<string, BusNamed>();
  for (const [lid, st] of Object.entries(w.lineStats)) {
    if (!st.bus) continue;
    let e = d ? was!.lines.get(lid) : undefined;
    if (!e || e.bus !== st.bus || e.plain !== plain || e.keeps !== w.save.keeps || moved.some(([x, y]) => x >= e!.box[0] && x <= e!.box[2] && y >= e!.box[1] && y <= e!.box[3])) e = namedBus(w, st.bus, plain);
    lines.set(lid, e);
    for (const s of e.stops) {
      if (byId.has(s.id)) continue;
      byId.set(s.id, s.stop);
      bus.set(s.id, s.stop);
      names.set(s.id, s.name);
      if (s.custom) custom.set(s.id, s.name);
    }
  }
  // T-4080: a stop with no name of its own, where another stop of its place has one, shows that one
  if (custom.size) {
    const p = busPlaces(w), at = new Map<number, string>();
    for (const [id, name] of custom) {
      const k = p.placeOf.get(Number(id));
      if (k !== undefined && !at.has(k)) at.set(k, name);
    }
    for (const [k, name] of at) for (const s of p.places[k].stops) if (s.station < 0 && bus.has(String(s.id)) && !custom.has(String(s.id))) { names.set(String(s.id), name); custom.set(String(s.id), name); }
  }
  if (!freshNow()) lastBusNamed = { w, lines };
  stopMemo.set(w, (m = { byId, names, bus, custom }));
  if (checkPatches() && d && !sameBuilt(m, freshly(() => stopsOf({ ...w })))) console.error("T-4150 check: the bus stops kept differ from a whole build");
  return m;
}
export const stopsById = (w: WorldView) => stopsOf(w).byId;
/** Only the bus stops that are no station. */
export const busStopsById = (w: WorldView) => stopsOf(w).bus;
/** T-4080: the bus stops that show a name the player gave, by id (their own, or their place's) */
export const namedBusStops = (w: WorldView) => stopsOf(w).custom;
/** Every stop's name by id, a bus stop's included. */
export const stationNames = (w: WorldView) => stopsOf(w).names;
export const stationsByNum = (w: WorldView) => lookup(w, "stationNums", (s) => { const m = new Map<number, StationInput>(); for (const x of s.stations) m.set(x.num, x); return m; });
export const linesById = (w: WorldView) => lookup(w, "lines", (s) => new Map(s.lines.map((l) => [l.id, l])));
export const linesByNum = (w: WorldView) => lookup(w, "lineNums", (s) => new Map(s.lines.map((l) => [l.num, l])));
/** T-3333: the lines a player sees listed: every line but the hidden twins of lines run on both pairs of
 * quad track (which every sum still counts). */
export const listedLines = (w: WorldView) => lookup(w, "listed", (s) => (s.lines.some((l) => l.twinOf) ? s.lines.filter((l) => !l.twinOf) : s.lines));
/** The lines that stop at each station, in line order; at a twin's stations, the line it runs for (T-3333). */
export const linesAtMap = (w: WorldView) => lookup(w, "linesAt", (s) => {
  const at = new Map<string, LineInput[]>();
  const byId = new Map(s.lines.map((l) => [l.id, l]));
  for (const t of s.lines) {
    const l = t.twinOf ? byId.get(t.twinOf) : t;
    if (!l) continue;
    for (const st of new Set(t.stops)) {
      const v = at.get(st) ?? at.set(st, []).get(st)!;
      if (!v.includes(l)) v.push(l);
    }
  }
  return at;
});

/** T-4125: the lines stopping at each station, as indices into the save's lines, ascending and once each (twins as
 * themselves) */
export const lineIndicesAt = (w: WorldView) => lookup(w, "lineIndicesAt", (s) => {
  const at = new Map<string, number[]>();
  s.lines.forEach((l, i) => {
    for (const st of l.stops) {
      const v = at.get(st) ?? at.set(st, []).get(st)!;
      if (v[v.length - 1] !== i) v.push(i);
    }
  });
  return at;
});

export const lineById = (id: LineId) => (world.value ? linesById(world.value).get(id) : undefined);
export const linesAt = (station: string) => (world.value ? linesAtMap(world.value).get(station) : undefined) ?? [];

const paired = new WeakMap<LineStats, { twin: LineStats; v: LineStats }>();
/** T-3333: a line run on both pairs of quad track as its panel shows it: its hidden twin's trains added
 * to its own and the headway the two keep together (each pair runs twice as long). Its edits take the
 * same whole-line figures (`TrackApi.set_train_counts`, `set_auto_headway`). Other lines as they are. */
export function pairStats(w: WorldView, st: LineStats): LineStats {
  const tw = st.twin ? w.lineStats[st.twin] : undefined;
  if (!tw) return st;
  const had = paired.get(st);
  if (had && had.twin === tw) return had.v;
  const v: LineStats = { ...st, trainsNeeded: [0, 1, 2].map((k) => st.trainsNeeded[k] + tw.trainsNeeded[k]) as [number, number, number], autoS: st.autoS ? st.autoS.map((h) => h / 2) as [number, number, number] : null };
  paired.set(st, { twin: tw, v });
  return v;
}
