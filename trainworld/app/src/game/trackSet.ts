// T-3634: built track Shift+clicked together with the select tool, shown in one track panel to rebuild at once
// (ui/inspector/TrackInspector.tsx `TracksInspector`). It is the selection itself, apart from the Select tool's
// pick (game/blueprintPick.ts), whose Shift+click adds the path between. Main thread only, not saved.

import { effect } from "@preact/signals";
import type { TrackEdge } from "./types";
import { selection } from "./ui";
import { world } from "./world";

/** Pieces a set may hold: those whose own panel offers Rebuild. */
export const settable = (e: TrackEdge | undefined): e is TrackEdge => !!e && e.built && !e.crossover;

/** The selected built pieces when the selection is one such piece or a set; null otherwise. */
export function selectedPieces(): number[] | null {
  const s = selection.peek(), w = world.peek();
  if (s?.kind === "tracks") return s.edges;
  if (s?.kind === "track" && settable(w?.edges.find((e) => e.id === s.edge))) return [s.edge];
  return null;
}

/** The selection made of `edges`: a set, one piece, or nothing. */
const selectionOf = (edges: number[]) => (edges.length > 1 ? { kind: "tracks" as const, edges } : edges.length ? { kind: "track" as const, edge: edges[0] } : null);

/** A Shift+click on piece `edge` with built track selected: added to the selection, or taken out if in it. A
 * piece no set may hold leaves it as it is. False when the selection is no built track (the click goes on as plain). */
export function shiftTrack(edge: number): boolean {
  const cur = selectedPieces();
  if (!cur) return false;
  if (!settable(world.peek()?.edges.find((e) => e.id === edge))) return true;
  selection.value = selectionOf(cur.includes(edge) ? cur.filter((e) => e !== edge) : [...cur, edge]);
  return true;
}

// pieces gone (removed, or replaced by an edit) drop out of a set
effect(() => {
  const w = world.value, s = selection.peek();
  if (!w || s?.kind !== "tracks") return;
  const ok = new Set(w.edges.filter(settable).map((e) => e.id));
  const edges = s.edges.filter((e) => ok.has(e));
  if (edges.length !== s.edges.length) selection.value = selectionOf(edges);
});
