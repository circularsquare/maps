// T-4146: what a station or delete edit changed, said right after the track model applied it (stream B of T-4144),
// so the page can draw it before the network holding it is built. Edge ids change under these edits (a station on an
// edge splits it, keeping the old id for the first half; removing a station joins the two pieces at its node; removing
// track past a station keeps a stub under the old id), so everything here is in the edges the page draws now
// (`posted.net`): an edge the edit made or reshaped is left out, and the page keeps drawing the old one in its place.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { EditOp, EditWord } from "./protocol";
import type { Posted } from "./clockWord";

/** The word for an accepted `op`, from the model as it left it and the rows and network the page holds (`posted`, not yet
 * holding the edit); undefined for an edit of another kind. A station placed, moved or changed needs no word: the page
 * draws its box from the ghost or the drawn station, and the track under it is drawn as it was. */
export function stationWord(api: TrackApi, op: EditOp, _answer: Applied, posted: Posted): Partial<EditWord> | undefined {
  const drawn = posted.net?.pts;
  if (!drawn) return undefined;
  if (op.op === "removeEdge") {
    const e = posted.edges.get(op.edge);
    // a crossover goes as a whole, both diagonals (T-338)
    const gone = e?.crossover && e.partner >= 0 ? [op.edge, e.partner] : [op.edge];
    const users = [...posted.lines.values()].filter((l) => [...(l.planPath ?? l.path), ...(l.back ?? [])].some((p) => gone.includes(p >> 1)));
    return { lines: users.flatMap((l) => said(api, l.id, drawn, new Set(gone), [])), hideEdges: gone.filter((x) => drawn.has(x)) };
  }
  if (op.op === "removeStation") {
    const n = op.node;
    // the two pieces at the node, which the removal may join into one under either id (T-1299)
    const at = [...posted.edges.values()].filter((e) => e.a === n || e.b === n).map((e) => e.id);
    const users = [...posted.lines.values()].filter((l) => [...(l.planStops ?? l.stops), ...(l.backStops ?? [])].includes(n));
    return { lines: users.flatMap((l) => said(api, l.id, drawn, new Set(), at.length === 2 ? at : [])), hideStations: [n] };
  }
  return undefined;
}

/** line `line`'s drawn edges after the edit (its plan's path and its return run's, as clockWord's line paths): those the
 * page draws and the edit left as they were, and the other half of a joined `pair` beside the half it kept */
function said(api: TrackApi, line: number, drawn: ReadonlyMap<number, unknown>, gone: ReadonlySet<number>, pair: number[]): { line: number; edges: Uint32Array }[] {
  // (a bus or ferry line has no strokes)
  if (api.is_bus(line) || api.is_ferry(line)) return [];
  const out = new Set<number>();
  for (const p of api.line_path(line)) out.add(p >> 1);
  for (const p of api.line_back_path(line)) out.add(p >> 1);
  if (pair.length === 2 && (out.has(pair[0]) || out.has(pair[1]))) for (const e of pair) out.add(e);
  for (const e of out) if (gone.has(e) || !drawn.has(e)) out.delete(e);
  return [{ line, edges: Uint32Array.from(out) }];
}
