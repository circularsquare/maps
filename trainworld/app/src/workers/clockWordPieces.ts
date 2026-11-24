// T-4148: Construct and the select tool's part of an applied edit's word (workers/clockWord.ts): the pieces it built,
// switched off, raised, made single or double, or removed, read from the track model's edge table before and after.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { EdgeLook, EditOp, EditWord } from "./protocol";
import type { Posted } from "./clockWord";

/** the ops this part reads the edge table for */
const READS = new Set<EditOp["op"]>(["construct", "constructAll", "constructLine", "edgesLevel", "edgesTracks", "blueprintOff", "removePieces"]);

/** `edges_info`: per edge id, tracks, constructed, lowest and highest level */
type Row = [number, number, number, number];
const STRIDE = 10;
function table(info: Float64Array): Map<number, Row> {
  const out = new Map<number, Row>();
  for (let i = 0; i + STRIDE <= info.length; i += STRIDE) out.set(info[i], [info[i + 3], info[i + 4], info[i + 8], info[i + 9]]);
  return out;
}

/** The edge table before the apply (a copy: the model hands out its own memory). */
export function pieceWordBefore(api: TrackApi, op: EditOp, _posted: Posted): unknown {
  return READS.has(op.op) ? Float64Array.from(api.edges_info()) : undefined;
}

export function pieceWord(api: TrackApi, op: EditOp, _answer: Applied, _posted: Posted, before?: unknown): Partial<EditWord> | undefined {
  if (!(before instanceof Float64Array)) return undefined;
  const was = table(before), now = table(api.edges_info());
  if (op.op === "removePieces") {
    const gone: number[] = [];
    for (const id of was.keys()) if (!now.has(id)) gone.push(id);
    return { hideEdges: gone, hideStations: op.stations };
  }
  const edges: NonNullable<EditWord["edges"]> = [];
  if (op.op === "blueprintOff") {
    for (const id of op.edges) if (now.get(id)?.[1] === 0) edges.push({ edge: id, off: op.off });
    return { edges };
  }
  for (const [id, a] of now) {
    const b = was.get(id);
    if (!b) continue;
    const look: { edge: number } & EdgeLook = { edge: id };
    if (a[1] && !b[1]) look.built = true;
    if (a[0] !== b[0]) look.tracks = a[0] === 1 ? 1 : 2;
    if ((a[2] !== b[2] || a[3] !== b[3]) && a[2] === a[3]) look.level = [a[2], a[3]];
    if (Object.keys(look).length > 1) edges.push(look);
  }
  return { edges };
}
