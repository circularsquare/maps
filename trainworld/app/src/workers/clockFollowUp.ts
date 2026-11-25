// An edit's follow-up queue, and the previews the main thread no longer wants (T-2374).

import { toMerc01 } from "../map/geo";
import type { ClockState, EditOp, ToClock } from "./protocol";
import { forget, game, localOf, sendRows, sendState, shiftFrom, stateSteps } from "./clockMain";
import { isPanic, MODEL_RESTORED, recover } from "./clockPanic";
import { diagramOn, postDiagram, posted } from "./clockPost";
import { api, messages, packLngLat, putMessages, putTiming, send, timing } from "./clockState";
import { loadsOwed, scheduleSnapshot } from "./clockDemand";
import { gameNow } from "./clockMoney";

// T-1039: previews the main thread no longer wants (`previewStale`, sent outside the queue) are
// answered `stale` without being worked out, or while they wait for masks
export let previewStaleSeq = 0;

export const stale = (seq: number) => seq <= previewStaleSeq;
type PreviewMsg = Extract<ToClock, { kind: "preview" | "previewEdge" | "previewNode" | "previewNodes" }>;
export const isPreview = (m: ToClock): m is PreviewMsg => m.kind === "preview" || m.kind === "previewEdge" || m.kind === "previewNode" || m.kind === "previewNodes";
export function sendStale(seq: number) {
  send({ kind: "preview", seq, stale: true, ok: false, cost: 0, lengthM: 0, mult: 0, issues: [], verts: new Float64Array(0), pts: new Float32Array(0) });
}

// ---------------------------------------------------------------- an edit's follow-up (T-1407)
//
// An edit's state, trips and boarding hour (and, unless it was answered `early`, its answer) wait
// until the previews asked while it was applied are answered, so the ghost waits for the model's
// change only. Any other message, and any trips sent meanwhile, run it first: nothing but the
// previews sees the model ahead of its state. T-3803: its state is built a piece a task, and an
// edit that comes between the pieces goes ahead (`goesAhead`), the two edits sharing one state.

/** messages answered ahead of an edit's follow-up: they read the model and change nothing (T-4214: and the delete
 * question's trial of a removal, which a delete clicked right after an edit waited a whole state for) */
export const jumpsAhead = (m: ToClock) => isPreview(m) || m.kind === "previewStation" || m.kind === "previewCrossover" || (m.kind === "query" && m.q.what === "removal");
/** how long an `early` edit's state waits for the ghost the track tool asks on its answer, ms */
export const EARLY_WAIT_MS = 30;
/** T-2404: an early edit's follow-up is held while the track tool keeps asking ghosts (the pointer
 * moving), until none has been asked for REST_MS, HOLD_MS after the first edit it holds, or any
 * message but a preview, a line query or a later early edit (which joins it) comes (`release` is
 * the tool's own ask) */
export const REST_MS = 220;
export const HOLD_MS = 5000;
/** T-2766: a stop click's follow-up (network, trips, boarding, demand's snapshot) is held until no stop click has
 * come for this long, ms, so quick clicks share one (players click stops 0.2 to 0.6 s apart); its rows went at once */
const STOP_REST_MS = 500;
/** an early edit's follow-up held: until when, and whether it is stop clicks' (T-2766), which only stop clicks join */
interface Held { until: number; stops: boolean }
/** What a follow-up's state is, so two edits' can be one: `net`, the renderer's network built again (not for a train
 * count or a name, T-229, T-3803); `name`, a name edit's (the schematic's graph sent again; a full state after all if
 * the drawn pieces are to move to a new anchor); `service`, a train count's (its lead is timed from `at`, when the edit
 * began); `effectiveAt`, the moment it takes effect (T-1071); `trips` sent with it; `early`, taken after the next
 * paint (`ghostFirst`, T-1407) */
interface StateAsk { net: boolean; name: boolean; service: boolean; effectiveAt?: number; trips: boolean; early: boolean; at: number }
/** T-3803: an edit in a follow-up that still waits for a world holding it (its answer, or the page's split of an early
 * answer's money, which it reads off the next state it takes), and what to do once one went */
export interface Waiting { owes: () => boolean; worldSent: () => void }
/** `after`: the edits' work after the state (each answers its edit once). `step`: work that can go ahead of the state
 * a few ms a task (the boarding account's lines), true while more is left; previews asked meanwhile are answered
 * between the steps. `held`: held for the pointer's rest (T-2404) or the next stop click (T-2766). `job`: the state
 * being built a piece a task (T-3803), `rows` the rows it has built. `carry`: an edit came before the state went, so
 * the next edit's follow-up takes this one in (one state for both). */
interface FollowUp { ask: StateAsk; after: () => void; fail: () => void; waiting: Waiting[]; step?: () => boolean; held?: Held; job?: Generator<void, void, void>; rows?: ClockState; carry?: boolean }
export let followUp: FollowUp | null = null;
export let followTimer = 0;
/** `hold`: held for the pointer's rest (T-2404); with `stops`, as a stop click's (T-2766) */
export function later(ask: StateAsk, after: () => void, fail: () => void, waiting: Waiting | null, waitMs: number, step?: () => boolean, hold = false, stops = false) {
  const prev = followUp;
  // T-3803: an edit was just applied, so nothing built before it may go out: a state begun is begun again
  if (prev) { prev.job = undefined; prev.rows = undefined; }
  let until = performance.now() + HOLD_MS;
  let f: FollowUp = { ask, after, fail, waiting: waiting ? [waiting] : [], step };
  if (hold && prev?.held && prev.held.stops === stops) {
    // one state for both edits, the earlier one's other work first
    f = joined(prev, f);
    until = prev.held.until;
  } else if (prev?.carry) {
    // T-3803: one state for both, not held: the earlier edit's network does not wait for this one's pointer to rest
    f = joined(prev, f);
    hold = false;
  } else runFollowUp();
  clearTimeout(followTimer);
  followUp = f;
  if (hold) {
    f.held = { until, stops };
    return holdOn();
  }
  followTimer = setTimeout(() => stepThenRun(f), Math.max(1, waitMs)) as unknown as number;
}
/** two edits' follow-ups as one, the earlier's work first; the state the later one's, but built in full if either needs
 * it and with the trips if either sends them */
function joined(a: FollowUp, b: FollowUp): FollowUp {
  const x = a.ask, y = b.ask;
  return {
    ask: { net: x.net || y.net, name: x.name || y.name, service: x.service || y.service, effectiveAt: y.effectiveAt ?? x.effectiveAt, trips: x.trips || y.trips, early: y.early, at: Math.min(x.at, y.at) },
    after: () => { a.after(); b.after(); },
    fail: () => { a.fail(); b.fail(); },
    waiting: [...a.waiting, ...b.waiting],
    step: b.step ?? a.step,
  };
}
/** the follow-up's state, its trips, and its edits' work after it, a piece per `next` */
function* followSteps(f: FollowUp): Generator<void, void, void> {
  putTiming({});
  // (from its first edit's apply to its state begun: the queue and the held wait)
  timing.fromApply = performance.now() - f.ask.at;
  // T-4272: the services stations and deletes answered from what they touch owe, a piece of their own before the state
  // reads them (an edit coming after it goes ahead of the rest)
  if (settleOwed(f)) yield;
  const a = f.ask, light = !a.net && !(a.name && shiftFrom);
  yield* stateSteps(light, a.effectiveAt, a.trips, a.early, true, (s) => { f.rows = s; }, a.service ? a.at : undefined);
  // the schematic reads its names off the world when a graph comes, so the same graph again
  if (light && a.name && diagramOn) postDiagram();
  if (a.service && !loadsOwed()) serviceEditMs = performance.now() - a.at;
  f.after();
}
/** T-4272: the services answered edits owe worked out (`settle_owed`), or what a refused edit, an undo with nothing to take
 * or a trial worked out of them first, taken in as an edit's would be: the drawn lines they moved forgotten (the edit's
 * own `forget` ran before them), and trips with the state where they moved a line for demand (they moved no running
 * line's trains, so no rider's trip moves otherwise). Whether there were any. */
function settleOwed(f?: FollowUp): boolean {
  const a = api!;
  if (a.owes()) a.set_now(gameNow());
  if (!a.settle_owed()) return false;
  forget();
  a.take_service_moved();
  if (a.take_settled_demand().length) {
    if (f) f.ask.trips = true;
    scheduleSnapshot();
  }
  return true;
}
/** T-4272: what answered edits owe worked out now, for a save, which then holds no edit without its services (autosave
 * waits for them; a game left while an edit waits for its ground does not) */
export function settleOwedNow() {
  if (api?.owes()) settleOwed(followUp ?? undefined);
}
/** a follow-up's steps a task each, then its state a piece a task (T-3803), outside the message queue: a message
 * that reads the model runs what is left of it first, and an edit may come between the pieces (`standAside`) */
export function stepThenRun(f: FollowUp) {
  if (followUp !== f) return;
  try {
    if (f.step?.()) { followTimer = setTimeout(() => stepThenRun(f), 0) as unknown as number; return; }
  } catch (err) {
    // a step that throws ends the stepping, never the follow-up: its run (which meets a panic
    // again) still answers the edit
    if (!isPanic(err)) send({ kind: "error", message: String(err) });
  }
  f.step = undefined;
  f.job ??= followSteps(f);
  pump(f);
}
/** one piece of the follow-up's state, the next a task later */
function pump(f: FollowUp) {
  if (followUp !== f || !f.job) return;
  // (so nothing in the piece runs it again: the trips at its end run a waiting follow-up first)
  followUp = null;
  let done: boolean | undefined;
  try { done = f.job.next().done; } catch (err) {
    // the whole follow-up again in the message queue, which meets a panic again and recovers there
    f.job = undefined;
    if (!followUp) followUp = f;
    if (!isPanic(err)) send({ kind: "error", message: String(err) });
    return queueFollowUp();
  }
  if (done) return stateWent();
  if (followUp) return;
  followUp = f;
  followTimer = setTimeout(() => pump(f), 0) as unknown as number;
}
/** T-3803: whether edit `m` goes ahead of follow-up `f`'s state, not after it: an edit of the main path that changes no
 * service (a train count's lead is sized to a train count's state, T-1071) when `f` builds the network and its rows can
 * go as a rows state (a network posted, no anchor move pending) */
export function goesAhead(m: ToClock, f: FollowUp): boolean {
  // T-4214: a station's name after its edit joins that edit's state, as a rename's rows would
  if (m.kind === "stationName") return f.ask.net && !!posted.net && !shiftFrom && m.game === game;
  if (m.kind !== "edit" || !f.ask.net || !posted.net || shiftFrom) return false;
  if (m.game !== undefined && m.game !== game) return false;
  return !OWN_STATE_OPS.has(m.op.op) && !isServiceOp(m.op);
}
/** edits that post their own state or money and answer at once, after any follow-up */
const OWN_STATE_OPS = new Set<EditOp["op"]>(["fares", "fareScale", "fareCityScale", "fareByCity", "fareHsr", "fareFerry", "fareBus", "fareTransfer", "fareLine", "fareZones", "fareRef", "borrow", "repay", "link", "addBuilding", "reshapeBuilding", "renameBuilding", "removeBuilding"]);
/** T-1071, T-1073: an edit that changes only the service (a train count takes effect a moment ahead) */
export function isServiceOp(op: EditOp): boolean {
  return op.op === "setTrainCount" || op.op === "setHeadway" || op.op === "weekendShare" || op.op === "weekendCounts" || op.op === "weekendShareAll" || op.op === "setPeriods" || op.op === "setHours" || op.op === "setRowTrains" || op.op === "setRowHeadways" || op.op === "rowWeekend"
    || ((op.op === "undo" || op.op === "redo") && api!.step_is_service(op.op === "redo"));
}
/** T-3803: an edit came before follow-up `f`'s state went: the rows it built (or new ones) go as a rows state when an
 * edit in it waits for a world holding it, that edit's answer after them, and the rest waits for the new edit's
 * follow-up to take it in (`carry`; on its own again if none does) */
export function standAside(f: FollowUp) {
  clearTimeout(followTimer);
  if (f.waiting.some((w) => w.owes())) {
    // T-4272: these rows answer an edit, so they hold what answered edits owe (one whose map had no patch waits for its
    // world); rows already built came after them
    if (!f.rows) settleOwed(f);
    sendRows(f.rows);
    for (const w of f.waiting) w.worldSent();
  }
  f.waiting = [];
  f.job = undefined;
  f.rows = undefined;
  f.held = undefined;
  f.carry = true;
}
/** the follow-up's steps and state from the next task, not waiting out its hold */
export function startNow(f: FollowUp) {
  clearTimeout(followTimer);
  followTimer = setTimeout(() => stepThenRun(f), 0) as unknown as number;
}
/** T-2404: the held follow-up's wait for the pointer's rest, started again (a ghost was just asked); T-2766: a stop
 * click's for the next click. Not once its state has started. */
export function holdOn() {
  const f = followUp;
  if (!f?.held || f.job) return;
  clearTimeout(followTimer);
  const rest = f.held.stops ? STOP_REST_MS : REST_MS;
  followTimer = setTimeout(() => stepThenRun(f), Math.max(1, Math.min(rest, f.held.until - performance.now()))) as unknown as number;
}
/** T-4272: messages that go at once, a follow-up waiting or not: the page's money figures from the world it holds, a paid
 * edit's split it read off it, and the clock. None reads what a follow-up changes or sends (T-2766's stop clicks and
 * T-2404's holds passed them already), and each ran a waiting follow-up whole in one block, which the next click waited
 * behind (0.4-2.5 s on her Asia save, notes/T-4267-measure.md) */
export const passesFollowUp = (m: ToClock) => STOP_KINDS.has(m.kind);
/** T-4272: demand's results, which may not reach the model ahead of its state (T-1407) and which nothing waits on: kept,
 * in order, until no follow-up waits (`keepUntilState`), instead of running it whole */
export const waitsFollowUp = (m: ToClock) => m.kind === "riders" || m.kind === "transfers";
/** each with the game it came in, as a game opened or rebuilt since (queued ahead of them) clears what they set */
type Kept = { m: ToClock; game: number };
let keptMsgs: Kept[] = [];
let onStateWent: ((msgs: Kept[]) => void) | null = null;
export function keepUntilState(m: ToClock) { keptMsgs.push({ m, game }); }
/** `cb` takes the kept messages once no follow-up waits (into the message queue, in order, each dropped there if its
 * game is gone) */
export function whenStateWent(cb: (msgs: Kept[]) => void) { onStateWent = cb; }
function stateWent() {
  if (followUp || !keptMsgs.length) return;
  const msgs = keptMsgs;
  keptMsgs = [];
  onStateWent?.(msgs);
}
/** T-2404: messages a held follow-up stays held for: they read nothing it would change (blueprint
 * track changes no line's run), and a later early edit joins it. T-2766: a stop click's, the next stop click and the
 * line tool's reads of the track model's lines, which the click changed already */
export const passesHeld = (m: ToClock, h: Held) => h.stops
  ? (m.kind === "edit" && !!m.rows) || (m.kind === "query" && STOP_QUERIES.has(m.q.what)) || STOP_KINDS.has(m.kind)
  : (m.kind === "edit" && !!m.early) || (m.kind === "query" && HELD_QUERIES.has(m.q.what)) || HELD_KINDS.has(m.kind);
const STOP_QUERIES = new Set(["stopTiers", "stopGap", "insertStop", "clickPath", "passedStations"]);
/** the page's money figures from the world each click's rows gave it, and the clock: none reads what the follow-up sends */
const STOP_KINDS = new Set<ToClock["kind"]>(["income", "points", "regionShares", "chargeShares", "clock"]);
/** demand's results and the clock: money, riders and trips, none of which a blueprint changes */
const HELD_KINDS = new Set<ToClock["kind"]>(["income", "points", "regionShares", "riders", "transfers", "epoch", "clock"]);
const HELD_QUERIES = new Set(["today", "lineAccount", "linePeriods", "linesPeriods", "lineWaits", "lineTrips", "stationDeps", "lineHeld"]);
/** T-2404: the blueprint edges the last edit made or changed, local x, y, z each, for the client's
 * stand-ins while their state is held (read from the track model, not kept: one may be gone) */
export function heldEdges(): Float32Array[] {
  const out: Float32Array[] = [];
  for (const id of api!.dirty_edges()) {
    if (posted.edges.get(id)?.built) continue;
    const r = api!.edge_render(id);
    if (r.length < 6) continue;
    const m = new Float64Array(r.length);
    for (let k = 0; k < r.length; k += 3) {
      const [mx, my] = toMerc01(...packLngLat(r[k], r[k + 1]));
      m[k] = mx;
      m[k + 1] = my;
      m[k + 2] = r[k + 2];
    }
    out.push(localOf(m, 3));
  }
  return out;
}
/** the waiting follow-up, now (what is left of its state, when it has started); one that throws answers its edits as
 * failed */
export function runFollowUp() {
  clearTimeout(followTimer);
  const f = followUp;
  followUp = null;
  if (!f) return;
  try {
    const g = (f.job ??= followSteps(f));
    while (!g.next().done);
  } catch (err) { f.fail(); throw err; } finally { stateWent(); }
}
/** the follow-up into the message queue; a failure there never leaves the queue rejected, which
 * would skip the next message (and an edit's answer with it) */
function queueFollowUp() {
  putMessages(messages.then(guardedFollowUp).catch((err) => send({ kind: "error", message: String(err) })));
}
export async function guardedFollowUp() {
  try { runFollowUp(); } catch (err) {
    if (isPanic(err)) await recover(err, "an edit's state", MODEL_RESTORED, performance.now()); // i18n-ignore
    else send({ kind: "error", message: String(err) });
  }
}
/** A full state sent outside the message queue (ground or streets fetched, a quote that moved). T-3803: as for trips,
 * a waiting follow-up goes first, so an edit's own state and answer come before it (the page splits a paid edit's
 * charge between the world before its state and the one with it); a held one stays held. A state begun is begun
 * again, so none goes out after this one built from rows older than it. */
export function sendStateOutside() {
  const f = followUp;
  if (f) {
    const begun = !!f.job;
    f.job = undefined;
    f.rows = undefined;
    // (its own pieces' timer would find no state to build)
    if (!f.held) void guardedFollowUp();
    else if (begun) startNow(f);
  }
  sendState();
}
/** a model rebuilt: the follow-up is dropped (the rebuild posts its own state), its edit answered as failed */
export function dropFollowUp() {
  clearTimeout(followTimer);
  const f = followUp;
  followUp = null;
  f?.fail();
  stateWent();
}

/** Wall ms the last train count or headway edit took here, from applying it to its trips sent with their loads. */
export let serviceEditMs = 100;
export function putServiceEditMs(v: typeof serviceEditMs) { serviceEditMs = v; }
export function putPreviewStaleSeq(v: typeof previewStaleSeq) { previewStaleSeq = v; }
