// T-4214 stage C1: right after an edit is applied, the records the edits since the last state read touched, read as the
// next state will read them, go to the page with the edit's word (workers/clock.worker.ts), a few ms and before the full
// state, so the page patches the map its tools click on at once (game/mapPatch.ts). None when the page could not take
// them exactly: a line's row would change (line rows are not patched), more records than PATCH_MAX, everything touched
// (a load), or the page's network or draw anchor not settled. notes/T-4214.md "Stage C".

import type { TrackApi } from "../wasm/track/sim";
import { anchor } from "../map/geo";
import type { MapPatch } from "./protocol";
import { drawnEdgeNow, rowEnv, rowsCanGo, stateRows } from "./clockMain";
import { posted } from "./clockPost";

/** past this many records an edit's patch is not sent: the page waits for the network as before (a big Rebuild) */
export const PATCH_MAX = 2000;
const ALL = 0xffffffff;

/** The patch for the model as it is now, or undefined (why: `patchMiss`). */
export function mapPatch(api: TrackApi): MapPatch | undefined {
  patchMiss = "";
  if (!rowsCanGo()) return void (patchMiss = "no network posted, or the anchor moving");
  const t = api.peek_touched();
  if (t[0] === ALL) return void (patchMiss = "everything touched");
  const ne = t[0];
  if (t.length - 1 > PATCH_MAX) return void (patchMiss = `${t.length - 1} records`);
  // (a line whose path moved: which lines run over which edges is the state's to send; a line's other figures, a timetable
  // a settle moved, stay the last state's in the page until its state, as before)
  const moved = pathMoved(api);
  if (moved) return void (patchMiss = moved);
  const edgeIds = t.subarray(1, 1 + ne), nodeIds = t.subarray(1 + ne);
  const { edges, nodes } = stateRows.touchedRows(api, rowEnv, edgeIds, nodeIds);
  const liveE = new Set(edges.map((e) => e.id)), liveN = new Set(nodes.map((n) => n.id));
  return {
    edges, nodes,
    goneEdges: Array.from(edgeIds).filter((id) => !liveE.has(id)),
    goneNodes: Array.from(nodeIds).filter((id) => !liveN.has(id)),
    pts: edges.map((e) => drawnEdgeNow(api, e.id).slice()),
    anchor: anchor().version,
  };
}

/** Why the lines' paths may differ from those the page holds (`posted.lines`: its plan's path and its return run's), or "":
 * a line noted since the last state read (`peek_changed_lines`) that is new, gone or runs over other edges. */
function pathMoved(api: TrackApi): string {
  const noted = api.peek_changed_lines();
  if (noted[0] === ALL) return "every line noted";
  for (const id of noted) {
    const row = posted.lines.get(id);
    if (!row) return `line ${id} new`;
    if (!same(api.line_path(id), row.planPath ?? row.path) || !same(api.line_back_path(id), row.back ?? [])) return `line ${id}'s path`;
  }
  return "";
}
const same = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** why the last `mapPatch` gave none ("" when it gave one), for the edit's answer */
export let patchMiss = "";
