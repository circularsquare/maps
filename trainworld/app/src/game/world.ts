// The newest world snapshot, and the edits the UI can ask for.
//
// `world` is read-only for the UI: components read it, never mutate it. It is replaced each time
// the clock worker sends a new state (workers/clockClient.ts). Edits go to the worker as ops
// (workers/protocol.ts); a refused edit puts its reasons, in plain words, in `notice`.

import { signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import type { EditAnswer } from "../workers/clockClient";
import { describeIssues } from "./issues";
import { tidyNodes, type FareNode } from "./fareCurve";
import type { Demand, LineId, WorldView } from "./types";
import { DEMAND_INDEX } from "./types";

export const world = signal<WorldView | null>(null);
/** A short message for the player: why an edit was refused, what a construct cost. */
export const notice = signal<{ text: string; bad: boolean; at: number } | null>(null);

let sink: ((op: EditOp) => Promise<EditAnswer>) | null = null;
/** workers/clockClient.ts hands over its edit function. */
export function setEditSink(f: (op: EditOp) => Promise<EditAnswer>) {
  sink = f;
}

export function say(text: string, bad = false) {
  notice.value = { text, bad, at: performance.now() };
}

/** Send an edit; a refusal is explained in `notice`. Resolves with the answer either way. */
export async function edit(op: EditOp, quiet = false): Promise<EditAnswer> {
  if (!sink) return { ok: false, issues: [], charge: 0 };
  const a = await sink(op);
  if (!a.ok && !quiet && a.issues.length) say(describeIssues(a.issues), true);
  return a;
}

const num = (id: LineId) => Number(id);

/** Train counts asked for but not yet in `world` (T-229): the stepper shows them at once, and a
 * second click before the first edit lands counts from them, not from the old figure. */
export const pendingTrains = signal<ReadonlyMap<LineId, [number, number, number]>>(new Map());
/** Per line: the edit in flight, and whether a newer count waits to go when it lands. */
const trainEdits = new Map<LineId, { busy: boolean; next: [number, number, number] | null }>();

/** A line's trains on the entire line per demand level, as the player last set them. */
export function trainsShown(id: LineId): [number, number, number] | null {
  return pendingTrains.value.get(id) ?? (world.value?.lineStats[id]?.trainsNeeded as [number, number, number] | undefined) ?? null;
}

/** Step a line's trains for one demand level, `by` from what it shows now (T-229). Clicks while an
 * edit is in flight are folded into one edit sent when it lands, so fast clicking never queues a
 * recompute per click. */
export function stepTrainCount(id: LineId, demand: Demand, by: number) {
  const base = trainsShown(id);
  if (!base) return;
  const t: [number, number, number] = [...base] as [number, number, number];
  // a loop's trains are half each way, so its count moves by two (T-798)
  const k = trainStep(id);
  t[DEMAND_INDEX[demand]] = Math.max(0, Math.round(t[DEMAND_INDEX[demand]] / k + by) * k);
  setTrainCounts(id, t);
}

export function setTrainCount(id: LineId, demand: Demand, trains: number) {
  const base = trainsShown(id);
  if (!base) return;
  const t: [number, number, number] = [...base] as [number, number, number];
  const k = trainStep(id);
  t[DEMAND_INDEX[demand]] = Math.max(0, Math.ceil(Math.round(trains) / k) * k);
  setTrainCounts(id, t);
}

/** The step of a line's train count: 2 on a loop, whose trains are half each way (T-798). */
export function trainStep(id: LineId): number {
  return world.value?.lineStats[id]?.loop ? 2 : 1;
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

function setTrainCounts(id: LineId, t: [number, number, number]) {
  const p = new Map(pendingTrains.peek());
  p.set(id, t);
  pendingTrains.value = p;
  const job = trainEdits.get(id) ?? { busy: false, next: null };
  trainEdits.set(id, job);
  job.next = t;
  if (job.busy) return;
  const send = async () => {
    job.busy = true;
    while (job.next) {
      const trains = job.next;
      job.next = null;
      const had = world.peek()?.lineStats[id]?.trainsNeeded;
      const a = await edit({ op: "setTrainCount", line: num(id), trains });
      noteAnswer(id, a, trains, had);
      if (!a.ok) job.next = null; // refused (money): the stepper goes back to the world's count
    }
    job.busy = false;
    trainEdits.delete(id);
    // the world now holds the last count sent (or, refused, the one before it)
    const q = new Map(pendingTrains.peek());
    q.delete(id);
    pendingTrains.value = q;
  };
  void send();
}

/** T-625: a line whose trains follow a headway. The headways asked for but not yet in `world`,
 * folded like the train counts above; null is a switch back to counts set by hand (T-747). */
export const pendingHeadway = signal<ReadonlyMap<LineId, [number, number, number] | null>>(new Map());
const headwayEdits = new Map<LineId, { busy: boolean; next: [number, number, number] | null | undefined }>();

/** Seconds per level the line keeps, as last set, or null when its trains are counted by hand. */
export function headwayShown(id: LineId): [number, number, number] | null {
  const p = pendingHeadway.value;
  if (p.has(id)) return p.get(id) ?? null;
  return world.value?.lineStats[id]?.autoS ?? null;
}

/** The headway limits (0 is no service) and the step from one headway to the next: 30 s under 5
 * minutes, 1 minute under 15, 5 minutes above; from no service the first step is 15 minutes. */
export const HEADWAY_MIN_S = 60;
export const HEADWAY_MAX_S = 7200;
export function nextHeadway(s: number, dir: number): number {
  if (dir > 0) return s <= 0 ? 900 :Math.min(HEADWAY_MAX_S, s + (s < 300 ? 30 : s < 900 ? 60 : 300));
  return s <= HEADWAY_MIN_S ? 0 : Math.max(HEADWAY_MIN_S, s - (s <= 300 ? 30 : s <= 900 ? 60 : 300));
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

/** Show `h` at once and send it, folded with any edit to the line's headways still in flight. */
function queueHeadway(id: LineId, h: [number, number, number] | null) {
  const p = new Map(pendingHeadway.peek());
  p.set(id, h);
  pendingHeadway.value = p;
  const job = headwayEdits.get(id) ?? { busy: false, next: undefined };
  headwayEdits.set(id, job);
  job.next = h;
  if (job.busy) return;
  const send = async () => {
    job.busy = true;
    while (job.next !== undefined) {
      const secs = job.next;
      job.next = undefined;
      const had = world.peek()?.lineStats[id]?.autoS;
      const a = await edit({ op: "setHeadway", line: num(id), seconds: secs });
      if (secs) noteAnswer(id, a, secs, had);
      if (!a.ok) job.next = undefined; // refused (money): the line goes back to the world's headways
    }
    job.busy = false;
    headwayEdits.delete(id);
    const q = new Map(pendingHeadway.peek());
    q.delete(id);
    pendingHeadway.value = q;
  };
  void send();
}

export function stepHeadway(id: LineId, demand: Demand, dir: number) {
  const base = headwayShown(id);
  if (base) setHeadway(id, demand, nextHeadway(base[DEMAND_INDEX[demand]], dir));
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
      if (autoTrainsFor(trip, h, 0, !!st.loop) === k) return h;
    }
    return exact;
  }) as [number, number, number];
  queueHeadway(id, seconds);
}

/** The fare curve for the whole network (T-883, game/fareCurve.ts): its nodes, km and US$. */
export function setFares(nodes: FareNode[]) {
  return edit({ op: "fares", nodes: tidyNodes(nodes) });
}

export function renameLine(id: LineId, name: string) {
  const n = name.trim();
  const l = lineById(id);
  if (n && l) void edit({ op: "lineLook", line: num(id), name: n, colour: l.colour });
}

export function setLineColour(id: LineId, colour: string) {
  const l = lineById(id);
  if (l) void edit({ op: "lineLook", line: num(id), name: l.name, colour });
}

export const undo = () => edit({ op: "undo" }, true);
export const redo = () => edit({ op: "redo" }, true);

export const lineById = (id: LineId) => world.value?.save.lines.find((l) => l.id === id);
export const stationById = (id: string) => world.value?.save.stations.find((s) => s.id === id);
export const linesAt = (station: string) => world.value?.save.lines.filter((l) => l.stops.includes(station)) ?? [];
