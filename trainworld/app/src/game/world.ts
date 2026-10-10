// The newest world snapshot, and the edits the UI can ask for.
//
// `world` is read-only for the UI: components read it, never mutate it. It is replaced each time
// the clock worker sends a new state (workers/clockClient.ts). Edits go to the worker as ops
// (workers/protocol.ts); a refused edit puts its reasons, in plain words, in `notice`.

import { signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import type { EditAnswer } from "../workers/clockClient";
import { describeIssues } from "./issues";
import { LINE_PALETTE } from "./palette";
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

export function setTrainCount(id: LineId, demand: Demand, trains: number) {
  const st = world.value?.lineStats[id];
  if (!st) return;
  const t: [number, number, number] = [...st.trainsNeeded] as [number, number, number];
  t[DEMAND_INDEX[demand]] = Math.max(0, Math.round(trains));
  void edit({ op: "setTrainCount", line: num(id), trains: t });
}

/** The fare curve for the whole network (T-028): US$ a ride plus US$ a km. */
export function setFares(base: number, perKm: number) {
  const r = (v: number) => Math.max(0, Math.round(v * 100) / 100);
  void edit({ op: "fares", base: r(base), perKm: r(perKm) });
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

/** The next default colour: the first palette colour no line uses, else round again. */
export function nextFreeColour(): string {
  const used = new Set(world.value?.save.lines.map((l) => l.colour) ?? []);
  return LINE_PALETTE.find((c) => !used.has(c)) ?? LINE_PALETTE[(world.value?.save.lines.length ?? 0) % LINE_PALETTE.length];
}

/** "Line 3": the lowest number not taken. */
export function nextLineName(): string {
  const taken = new Set(world.value?.save.lines.map((l) => l.name) ?? []);
  for (let k = 1; ; k++) if (!taken.has(`Line ${k}`)) return `Line ${k}`;
}

export const undo = () => edit({ op: "undo" }, true);
export const redo = () => edit({ op: "redo" }, true);

export const lineById = (id: LineId) => world.value?.save.lines.find((l) => l.id === id);
export const stationById = (id: string) => world.value?.save.stations.find((s) => s.id === id);
export const linesAt = (station: string) => world.value?.save.lines.filter((l) => l.stops.includes(station)) ?? [];
