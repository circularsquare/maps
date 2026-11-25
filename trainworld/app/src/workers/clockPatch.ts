// T-4214 stage C1: right after an edit is applied, the records the edits since the last state read touched, read as the
// next state will read them, go to the page with the edit's word (workers/clock.worker.ts), a few ms and before the full
// state, so the page patches the map its tools click on at once (map/clickMap.ts), with any station whose shown name they
// moved though none touched it (a complex joined or split, stateRows.ts `patchNames`). A line whose path the edits moved (a
// station on its track splits an edge, track it runs over goes) has its paths in it, which say what runs on the touched
// track; its other figures wait for the state. None when the page could not take them exactly: a line new, gone or
// starting or stopping to run, more records than PATCH_MAX, everything touched (a load), or the page's network or draw
// anchor not settled. notes/T-4214.md "Stage C" and "Stage A".

import type { TrackApi } from "../wasm/track/sim";
import { anchor } from "../map/geo";
import type { MapPatch, PatchLine } from "./protocol";
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
  // (a line's other figures, a timetable a settle moved, stay the last state's in the page until its state, as before)
  const lines = movedPaths(api);
  if (typeof lines === "string") return void (patchMiss = lines);
  const edgeIds = t.subarray(1, 1 + ne), nodeIds = t.subarray(1 + ne);
  const { edges, nodes } = stateRows.touchedRows(api, rowEnv, edgeIds, nodeIds);
  const liveE = new Set(edges.map((e) => e.id)), liveN = new Set(nodes.map((n) => n.id));
  return {
    edges, nodes,
    goneEdges: Array.from(edgeIds).filter((id) => !liveE.has(id)),
    goneNodes: Array.from(nodeIds).filter((id) => !liveN.has(id)),
    pts: edges.map((e) => drawnEdgeNow(api, e.id).slice()),
    anchor: anchor().version,
    ...(lines.length ? { lines } : {}),
  };
}

/** The paths of the lines noted since the last state read (`peek_changed_lines`) that differ from those the page holds
 * (`posted.lines`), made as the state's rows make them (stateRows.ts `heavy`); why not, when a line is new, gone, starts
 * or stops counting (its whole track is read again then), changes between one path and a plan's and a run's, or stops
 * elsewhere (a tool reads the stops: the line tool's next click works from them), or what its owed services would say
 * of it can differ (broken or mended, a partly built line's run, its return run's stops). A partly built line's run is its own
 * path: constructing its plan's track moves the run and leaves the plan. */
function movedPaths(api: TrackApi): PatchLine[] | string {
  const noted = api.peek_changed_lines();
  if (noted[0] === ALL) return "every line noted";
  if (!noted.length) return [];
  const out: PatchLine[] = [];
  let info: Map<number, [number, number]> | null = null;
  // T-4272: while answered edits owe their services, what those say of a line (ok and running, a partly built line's run,
  // its return run's stops) is from before the edits; what the track says (its path, stops and broken flag) is now
  const owed = api.owes();
  for (const id of noted) {
    const row = posted.lines.get(id);
    if (!row) return `line ${id} new`;
    if (owed && (api.line_broken(id) !== row.broken || row.planPath || row.backStops?.length)) return `line ${id}'s services owed`;
    const plan = api.line_path(id), back = api.line_back_path(id), at = api.line_part(id), part = at.length === 4;
    const run = part ? api.line_run_path(id) : plan;
    if (part === !!row.planPath && same(run, row.path) && (!part || same(plan, row.planPath!)) && same(back, row.back ?? [])) continue;
    if (!info) { info = new Map(); const li = api.lines_info(); for (let i = 0; i < li.length; i += 22) info.set(li[i], [li[i + 2], li[i + 3]]); }
    const now = info.get(id);
    if (!now) return `line ${id} gone`;
    if ((now[0] === 1) !== row.ok || (now[1] === 1) !== row.running) return `line ${id} starts or stops running`;
    if (part !== !!row.planPath) return `line ${id}'s plan`;
    if (!same(api.line_stops(id), row.planStops ?? row.stops) || !same(api.line_back_stops(id), row.backStops ?? [])) return `line ${id}'s stops`;
    // (the page reads a partly built line's run stops and where they start among its plan's, `toWorld`)
    if (part && (!same(api.line_run_stops(id), row.stops) || at[0] !== row.stopStart)) return `line ${id}'s stops`;
    out.push({ id, path: [...run], ...(part ? { planPath: [...plan] } : {}), ...(back.length ? { back: [...back] } : {}) });
  }
  return out;
}
const same = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** why the last `mapPatch` gave none ("" when it gave one), for the edit's answer */
export let patchMiss = "";
