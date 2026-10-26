// The newest world snapshot, and the edits the UI can ask for.
//
// `world` is read-only for the UI: components read it, never mutate it. It is replaced each time
// the clock worker sends a new state (workers/clockClient.ts). Edits go to the worker as ops
// (workers/protocol.ts); a refused edit puts its reasons, in plain words, in `notice`.

import { signal, type Signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import type { EditAnswer } from "../workers/clockClient";
import { describeIssues } from "./issues";
import { tidyNodes, type FareNode } from "./fareCurve";
import { groundDist } from "./coords";
import type { Demand, LineId, LineInput, Save, StationInput, WorldView } from "./types";
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

/** Edits sent other than fares, undo and redo: a fare change is the last action while this holds
 * still (T-1130, game/fareUndo.ts). */
export const editsSent = signal(0);

/** Told of every answered edit the player made (map/buildSound.ts, T-1282); checks that send edits
 * to the client directly are not. */
let afterEdit: ((op: EditOp, a: EditAnswer) => void) | null = null;
export function setAfterEdit(f: (op: EditOp, a: EditAnswer) => void) {
  afterEdit = f;
}

/** Send an edit; a refusal is explained in `notice`. Resolves with the answer either way. */
export async function edit(op: EditOp, quiet = false): Promise<EditAnswer> {
  if (!sink) return { ok: false, issues: [], charge: 0 };
  if (op.op !== "fares" && op.op !== "undo" && op.op !== "redo") editsSent.value++;
  const a = await sink(op);
  if (!a.ok && !quiet && a.issues.length) say(describeIssues(a.issues), true);
  afterEdit?.(op, a);
  return a;
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
export function stepTrainCount(id: LineId, demand: Demand, by: number) {
  changeTrains(id, demand, (now, k) => Math.max(0, Math.round(now / k + by) * k));
}

export function setTrainCount(id: LineId, demand: Demand, trains: number) {
  changeTrains(id, demand, (_, k) => Math.max(0, Math.ceil(Math.round(trains) / k) * k));
}

function changeTrains(id: LineId, demand: Demand, to: (now: number, step: number) => number) {
  const base = trainsShown(id);
  if (!base) return;
  const t: [number, number, number] = [...base] as [number, number, number];
  const k = trainStep(id);
  t[DEMAND_INDEX[demand]] = to(t[DEMAND_INDEX[demand]], k);
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

const setTrainCounts = folded(pendingTrains, async (id, trains) => {
  const had = world.peek()?.lineStats[id]?.trainsNeeded;
  const a = await edit({ op: "setTrainCount", line: num(id), trains });
  noteAnswer(id, a, trains, had);
  return a.ok;
});

/** T-625: a line whose trains follow a headway. The headways asked for but not yet in `world`,
 * folded like the train counts above; null is a switch back to counts set by hand (T-747). */
export const pendingHeadway = signal<ReadonlyMap<LineId, [number, number, number] | null>>(new Map());

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

const queueHeadway = folded(pendingHeadway, async (id, secs) => {
  const had = world.peek()?.lineStats[id]?.autoS;
  const a = await edit({ op: "setHeadway", line: num(id), seconds: secs });
  if (secs) noteAnswer(id, a, secs, had);
  return a.ok;
});

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
/** T-1277: a country's fare scale, null for its income's. */
export function setFareScale(country: string, scale: number | null) {
  return edit({ op: "fareScale", country, scale });
}

export function renameLine(id: LineId, name: string) {
  const n = name.trim();
  const l = lineById(id);
  if (n && l) void edit({ op: "lineLook", line: num(id), name: n, colour: l.colour });
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
      names.set(id, near ? `Bus stop near ${near}` : "Bus stop");
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
