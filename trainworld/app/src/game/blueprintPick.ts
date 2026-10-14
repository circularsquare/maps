// The blueprint pieces picked with the blueprint tool's drag box (T-333), and switching them off or
// on. Main thread only, not saved; the switch itself lives on the pieces in the track model
// (`off`, saved with the game) and Construct and the blueprint quote skip what is off.

import { computed, effect, signal } from "@preact/signals";
import { tool } from "./ui";
import { edit, world } from "./world";

export interface BpPicked {
  /** blueprint track pieces (edge ids) and blueprint stations (node ids) */
  edges: number[];
  stations: number[];
}

const NONE: BpPicked = { edges: [], stations: [] };
export const bpPicked = signal<BpPicked>(NONE);

export const clearBpPicked = () => {
  if (bpPicked.peek().edges.length || bpPicked.peek().stations.length) bpPicked.value = NONE;
};

/** Merge two picks (shift adds). */
export function unionPicked(a: BpPicked, b: BpPicked): BpPicked {
  return { edges: [...new Set([...a.edges, ...b.edges])], stations: [...new Set([...a.stations, ...b.stations])] };
}

/** What the picked pieces are now: how many are switched off. */
export const pickedState = computed(() => {
  const w = world.value, p = bpPicked.value;
  if (!w) return { edges: 0, stations: 0, off: 0, on: 0 };
  const edges = new Map(w.edges.map((e) => [e.id, e]));
  const stations = new Map(w.save.stations.map((s) => [s.num, s]));
  let off = 0, on = 0;
  for (const id of p.edges) (edges.get(id)?.off ? off++ : on++);
  for (const id of p.stations) (stations.get(id)?.off ? off++ : on++);
  return { edges: p.edges.length, stations: p.stations.length, off, on };
});

/** Every blueprint piece switched off, for the Build tab. */
export const offTotals = computed(() => {
  const w = world.value;
  if (!w) return { edges: 0, stations: 0 };
  return { edges: w.edges.filter((e) => !e.built && e.off).length, stations: w.save.stations.filter((s) => !s.built && s.off).length };
});

/** Switch the picked pieces off (Construct and the quote leave them out) or on: one undoable edit. */
export async function switchPicked(off: boolean) {
  const p = bpPicked.peek();
  if (!p.edges.length && !p.stations.length) return;
  await edit({ op: "blueprintOff", edges: p.edges, stations: p.stations, off }, true);
}

// Pieces that were constructed or removed drop out of the pick; leaving the tool clears it.
effect(() => {
  const w = world.value;
  const p = bpPicked.peek();
  if (!w || (!p.edges.length && !p.stations.length)) return;
  const edges = new Set(w.edges.filter((e) => !e.built).map((e) => e.id));
  const stations = new Set(w.save.stations.filter((s) => !s.built).map((s) => s.num));
  const next = { edges: p.edges.filter((e) => edges.has(e)), stations: p.stations.filter((s) => stations.has(s)) };
  if (next.edges.length !== p.edges.length || next.stations.length !== p.stations.length) bpPicked.value = next;
});
effect(() => {
  if (tool.value !== "blueprint") clearBpPicked();
});
