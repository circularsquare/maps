// The pieces picked with the Select tool's drag box (T-333): blueprint, or constructed track when
// the pick holds no blueprint at all (T-829). Switching blueprint off or on, its elevation (T-815)
// and removing the lot (T-829). Main thread only, not saved; the off switch itself lives on the
// pieces in the track model (`off`, saved with the game) and Construct and the quote skip what is off.

import { computed, effect, signal } from "@preact/signals";
import { distance, levelLabel } from "./format";
import { lineEffects } from "./remove";
import type { Level } from "./types";
import { ask, tool } from "./ui";
import { edit, say, stationsByNum, world } from "./world";

export interface BpPicked {
  /** track pieces (edge ids) and stations (node ids) */
  edges: number[];
  stations: number[];
  /** constructed pieces (a pick with no blueprint in it); else blueprint only */
  built?: boolean;
}

const NONE: BpPicked = { edges: [], stations: [] };
export const bpPicked = signal<BpPicked>(NONE);

export const clearBpPicked = () => {
  if (bpPicked.peek().edges.length || bpPicked.peek().stations.length) bpPicked.value = NONE;
};

/** One kind only (Anita, T-829): any blueprint in it keeps only the blueprint, else constructed. */
export function onlyOneKind(p: BpPicked): BpPicked {
  const w = world.peek();
  if (!w) return p;
  const builtEdge = new Set(w.edges.filter((e) => e.built).map((e) => e.id));
  const builtStation = new Set(w.save.stations.filter((s) => s.built).map((s) => s.num));
  const bp = p.edges.some((e) => !builtEdge.has(e)) || p.stations.some((s) => !builtStation.has(s));
  return bp
    ? { edges: p.edges.filter((e) => !builtEdge.has(e)), stations: p.stations.filter((s) => !builtStation.has(s)) }
    : { edges: p.edges, stations: p.stations, built: p.edges.length + p.stations.length > 0 };
}

/** Merge two picks (shift adds), one kind kept. */
export function unionPicked(a: BpPicked, b: BpPicked): BpPicked {
  return onlyOneKind({ edges: [...new Set([...a.edges, ...b.edges])], stations: [...new Set([...a.stations, ...b.stations])] });
}

/** What the picked pieces are now: how many are switched off, their track's length, and the one
 * elevation all of it is on (null when mixed or ramped). */
export const pickedState = computed(() => {
  const w = world.value, p = bpPicked.value;
  if (!w) return { edges: 0, stations: 0, off: 0, on: 0, built: false, lengthM: 0, level: null as Level | null };
  const edges = new Map(w.edges.map((e) => [e.id, e]));
  const stations = stationsByNum(w);
  let off = 0, on = 0, lengthM = 0;
  let level: Level | null | undefined;
  for (const id of p.edges) {
    const e = edges.get(id);
    if (!e) continue;
    e.off ? off++ : on++;
    lengthM += e.lengthM;
    const l = e.levelMin === e.levelMax ? e.levelMin : null;
    level = level === undefined || level === l ? l : null;
  }
  for (const id of p.stations) (stations.get(id)?.off ? off++ : on++);
  return { edges: p.edges.length, stations: p.stations.length, off, on, built: !!p.built, lengthM, level: level ?? null };
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
  if (p.built || (!p.edges.length && !p.stations.length)) return;
  await edit({ op: "blueprintOff", edges: p.edges, stations: p.stations, off }, true);
}

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** T-815: the picked blueprint track on one elevation, one undoable edit; pieces that cannot
 * change (a flat crossing would form, no corners to raise, a ramp with no room) are left as they
 * are and counted. */
export async function levelPicked(level: Level) {
  const p = bpPicked.peek();
  if (p.built || !p.edges.length) return;
  const a = await edit({ op: "edgesLevel", edges: p.edges, level }, true);
  const n = p.edges.length, skipped = a.skipped ?? 0, to = `elevation ${levelLabel(level)}`;
  if (!a.ok) say(n === 1 ? `This track cannot go to ${to}.` : `None of the ${n} pieces can go to ${to}.`, true);
  else if (skipped) say(`${n - skipped} of ${n} pieces set to ${to}. ${skipped} could not change.`);
  else say(`${count(n, "piece", "pieces")} set to ${to}.`);
}

/** T-829: remove every picked piece in one edit: blueprint at once (one undo), constructed or
 * changing a line after a yes, as for one piece (game/remove.ts). */
export async function removePicked() {
  const p = bpPicked.peek(), s = pickedState.peek();
  if (!p.edges.length && !p.stations.length) return;
  const what = [s.edges ? `${distance(s.lengthM)} of ${p.built ? "constructed " : ""}track` : "", s.stations ? count(s.stations, p.built ? "constructed station" : "station", p.built ? "constructed stations" : "stations") : ""].filter(Boolean).join(" and ");
  const go = async () => {
    ask.value = null;
    const a = await edit({ op: "removePieces", edges: p.edges, stations: p.stations });
    if (a.ok) say(`Removed ${what}.`);
  };
  const lines = await lineEffects(p.edges, p.stations);
  if (!p.built && !lines.length) return void go();
  ask.value = { text: `Remove ${what}?${p.built ? " Nothing is refunded." : ""}`, yes: "Remove", no: "Keep", run: () => void go(), lines };
}

// Pieces removed, or no longer of the pick's kind (blueprint constructed), drop out of the pick;
// leaving the tool clears it.
effect(() => {
  const w = world.value;
  const p = bpPicked.peek();
  if (!w || (!p.edges.length && !p.stations.length)) return;
  const edges = new Set(w.edges.filter((e) => e.built === !!p.built).map((e) => e.id));
  const stations = new Set(w.save.stations.filter((s) => s.built === !!p.built).map((s) => s.num));
  const next = { edges: p.edges.filter((e) => edges.has(e)), stations: p.stations.filter((s) => stations.has(s)), built: p.built };
  if (next.edges.length !== p.edges.length || next.stations.length !== p.stations.length) bpPicked.value = next;
});
effect(() => {
  if (tool.value !== "blueprint") clearBpPicked();
});
