// T-4145: one word from the worker after every accepted edit, before its rows, state and answer: what the page can
// draw and show of the edit until the first network (or rows) after it (game/pending.ts, notes/T-4145.md). Each kind
// of edit has its part in a file of its own; this file runs them and merges their parts.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { posted } from "./clockPost";
import type { EditOp, EditWord } from "./protocol";
import * as stations from "./clockWordStations";
import * as lines from "./clockWordLines";
import * as pieces from "./clockWordPieces";

export type Posted = typeof posted;
type Kind = { word: (api: TrackApi, op: EditOp, answer: Applied, posted: Posted, before?: unknown) => Partial<EditWord> | undefined; before?: (api: TrackApi, op: EditOp, posted: Posted) => unknown };
/** (a kind file's optional `<name>Before` is read by name, so a stream adds one without touching this file) */
const kind = (m: Record<string, unknown>, name: string): Kind => ({ word: m[name] as Kind["word"], before: m[name + "Before"] as Kind["before"] });
const KINDS: Kind[] = [
  { word: pathWord, before: pathBefore },
  kind(stations, "stationWord"),
  kind(lines, "lineWord"),
  kind(pieces, "pieceWord"),
];

/** T-4114: the ops that can change which edges a line's strokes draw */
const LINE_PATH_OPS = new Set<EditOp["op"]>(["toggleStop", "insertStop", "oneWayStop", "setStops", "dropStop", "endLineAt", "stopPassed", "orderStops", "sortStops", "turnEnd", "openLoop", "removeLine"]);

/** the edges a rail line's strokes draw: its plan's path and its return run's */
export function lineEdges(api: TrackApi, line: number): Uint32Array {
  const out = new Set<number>();
  for (const p of api.line_path(line)) out.add(p >> 1);
  for (const p of api.line_back_path(line)) out.add(p >> 1);
  return Uint32Array.from(out);
}

/** T-4114: the rail line a path op names (a bus or ferry line has no strokes); read before the apply, a deleted line's too */
function pathBefore(api: TrackApi, op: EditOp): number | undefined {
  const line = LINE_PATH_OPS.has(op.op) ? (op as { line?: number }).line : undefined;
  return line !== undefined && !api.is_bus(line) && !api.is_ferry(line) ? line : undefined;
}
function pathWord(api: TrackApi, op: EditOp, _answer: Applied, _posted: Posted, before?: unknown): Partial<EditWord> | undefined {
  if (typeof before !== "number") return undefined;
  return { lines: [{ line: before, edges: op.op === "removeLine" ? null : lineEdges(api, before) }] };
}

/** Each kind's read before the edit is applied, to hand to `wordAfter`. */
export function wordBefore(api: TrackApi, op: EditOp, p: Posted): unknown[] {
  return KINDS.map((k) => k.before?.(api, op, p));
}

/** The word for an accepted edit: every kind's part, arrays joined (a kind that throws gives none, and says so). */
export function wordAfter(api: TrackApi, op: EditOp, answer: Applied, p: Posted, before: unknown[], charge: number): EditWord {
  const out: EditWord = charge > 0 ? { charge } : {};
  KINDS.forEach((k, i) => {
    let part: Partial<EditWord> | undefined;
    try { part = k.word(api, op, answer, p, before[i]); } catch (err) { console.error("T-4145: an edit's word part failed", err); }
    if (!part) return;
    for (const key of Object.keys(part) as (keyof EditWord)[]) {
      const v = part[key];
      if (v === undefined) continue;
      if (key === "charge") out.charge = v as number;
      else (out as Record<string, unknown[]>)[key] = [...((out as Record<string, unknown[]>)[key] ?? []), ...(v as unknown[])];
    }
  });
  return out;
}
