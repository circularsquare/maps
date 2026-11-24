// T-4148: Construct and the select tool's part of an applied edit's word (workers/clockWord.ts): the pieces it built,
// switched off, raised, made single or double, or removed, read from the track model's edge table before and after.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { EdgeLook, EditOp, EditWord } from "./protocol";
import type { Posted } from "./clockWord";
import { drawnEdgeNow } from "./clockMain";

/** the ops this part reads the edge table for (each read is two `edges_info`, tens of ms on a big save: an op whose
 * show is exact from its op alone, and a type change or one way's, which read their pieces only, do not) */
const READS = new Set<EditOp["op"]>(["construct", "constructAll", "constructLine", "edgesLevel", "edgesTracks", "blueprintOff", "removePieces", "doubleUp"]);

/** `edges_info`: per edge id, tracks, constructed, lowest and highest level */
type Row = [number, number, number, number];
const STRIDE = 10;
function table(info: Float64Array): Map<number, Row> {
  const out = new Map<number, Row>();
  for (let i = 0; i + STRIDE <= info.length; i += STRIDE) out.set(info[i], [info[i + 3], info[i + 4], info[i + 8], info[i + 9]]);
  return out;
}

/** what is read before the apply: the edge table (a copy: the model hands out its own memory), and the track type of the
 * edges a type change names */
interface Before { info: Float64Array; kinds: Map<number, number> }

export function pieceWordBefore(api: TrackApi, op: EditOp, _posted: Posted): unknown {
  const kinds = new Map<number, number>();
  if (op.op === "edgesKind") for (const id of op.edges) kinds.set(id, api.edge_kind(id));
  if (!READS.has(op.op)) return kinds.size ? ({ info: new Float64Array(0), kinds } satisfies Before) : undefined;
  return { info: Float64Array.from(api.edges_info()), kinds } satisfies Before;
}

export function pieceWord(api: TrackApi, op: EditOp, _answer: Applied, _posted: Posted, before?: unknown): Partial<EditWord> | undefined {
  if (op.op === "oneWay" || op.op === "oneWays") {
    // (the page has the end nodes from its guess; a double track's way is 0, which draws nothing)
    const edges: NonNullable<EditWord["edges"]> = [];
    for (const id of op.op === "oneWay" ? [op.edge] : op.edges) edges.push({ edge: id, oneWay: api.edge_way(id) as 0 | 1 | 2 });
    return { edges };
  }
  const pre = before as Before | undefined;
  if (!pre?.info) return undefined;
  if (op.op === "edgesKind") {
    const edges: NonNullable<EditWord["edges"]> = [];
    for (const [id, k0] of pre.kinds) { const k1 = api.edge_kind(id); if (k1 !== k0) edges.push({ edge: id, kind: k1 as 0 | 1 | 2 | 3 | 4 }); }
    return { edges };
  }
  const was = table(pre.info), now = table(api.edges_info());
  if (op.op === "doubleUp") {
    // the pair beside the track: drawn as blueprint from the model's own points (the cache holds none for an id new now)
    const made: NonNullable<EditWord["pieces"]> = [];
    for (const [id, a] of now) if (!was.has(id)) made.push({ edge: id, pts: drawnEdgeNow(api, id), tracks: a[0] === 1 ? 1 : 2, kind: api.edge_kind(id) });
    return made.length ? { pieces: made } : undefined;
  }
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
