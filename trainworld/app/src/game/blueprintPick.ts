// The pieces picked with the Select tool's drag box (T-333): blueprint, or constructed track when
// the pick holds no blueprint at all (T-829). Switching blueprint off or on, its elevation (T-815)
// and removing the lot (T-829). Main thread only, not saved; the off switch itself lives on the
// pieces in the track model (`off`, saved with the game) and Construct and the quote skip what is off.

import { computed, effect, signal } from "@preact/signals";
import { bigDistance, distance, levelLabel } from "./format";
import { describeIssues } from "./issues";
import { isBigDelete, lineEffects, removeQuestion } from "./remove";
import type { Level, TrackEdge } from "./types";
import { ask, selection, tool } from "./ui";
import { edit, say, stationsByNum, world } from "./world";

export interface BpPicked {
  /** track pieces (edge ids) and stations (node ids) */
  edges: number[];
  stations: number[];
  /** constructed pieces (a pick with no blueprint in it); else blueprint only */
  built?: boolean;
  /** T-1898: the nodes a drag of the pick moves, where a box picked only those inside it; absent,
   * every end of the picked track and every picked station (`dragNodes`) */
  nodes?: number[];
}

const NONE: BpPicked = { edges: [], stations: [] };

/** The nodes a drag of the pick moves (some may turn out not movable: map/blueprintBox.ts). */
export function dragNodes(p: BpPicked): number[] {
  if (p.nodes) return p.nodes;
  const w = world.peek();
  const ids = new Set(p.stations);
  if (w && p.edges.length) {
    const picked = new Set(p.edges);
    for (const e of w.edges) if (picked.has(e.id)) ids.add(e.a), ids.add(e.b);
  }
  return [...ids];
}

/** `p` with its `nodes` cut to those still on its track or stations. */
function trimNodes(p: BpPicked): BpPicked {
  if (!p.nodes) return p;
  const on = new Set(dragNodes({ edges: p.edges, stations: p.stations }));
  const nodes = p.nodes.filter((n) => on.has(n));
  return nodes.length === p.nodes.length ? p : { ...p, nodes };
}

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
  const nodes = p.nodes ? { nodes: p.nodes } : {};
  return trimNodes(bp
    ? { edges: p.edges.filter((e) => !builtEdge.has(e)), stations: p.stations.filter((s) => !builtStation.has(s)), ...nodes }
    : { edges: p.edges, stations: p.stations, built: p.edges.length + p.stations.length > 0, ...nodes });
}

/** Merge two picks (shift adds), one kind kept. */
export function unionPicked(a: BpPicked, b: BpPicked): BpPicked {
  const nodes = a.nodes || b.nodes ? { nodes: [...new Set([...dragNodes(a), ...dragNodes(b)])] } : {};
  return onlyOneKind({ edges: [...new Set([...a.edges, ...b.edges])], stations: [...new Set([...a.stations, ...b.stations])], ...nodes });
}

/** `p` less the pieces in `out` (a Shift+click on a picked piece). */
export function withoutPicked(p: BpPicked, out: BpPicked): BpPicked {
  return trimNodes({ edges: p.edges.filter((v) => !out.edges.includes(v)), stations: p.stations.filter((v) => !out.stations.includes(v)), built: p.built, ...(p.nodes ? { nodes: p.nodes } : {}) });
}

type Adj = Map<number, TrackEdge[]>;
function adjacency(edges: TrackEdge[]): Adj {
  const adj: Adj = new Map();
  for (const e of edges) for (const n of [e.a, e.b]) {
    const l = adj.get(n);
    l ? l.push(e) : adj.set(n, [e]);
  }
  return adj;
}

/** T-1680: the run a piece is on, out each way to the next junction, track end, or track of the
 * other kind (built or blueprint): every piece through nodes where exactly two pieces meet. */
export function runOf(edge: number): number[] {
  const w = world.peek();
  const start = w?.edges.find((e) => e.id === edge);
  if (!w || !start) return [];
  const adj = adjacency(w.edges);
  const out = new Set([edge]);
  for (let node of [start.a, start.b]) {
    let cur = start;
    for (;;) {
      const at = adj.get(node) ?? [];
      const next = at.length === 2 ? at.find((e) => e !== cur) : undefined;
      if (!next || next.built !== start.built || out.has(next.id)) break;
      out.add(next.id);
      cur = next;
      node = next.a === node ? next.b : next.a;
    }
  }
  return [...out];
}

/** T-1680: the pieces on the shortest track path from the pick to `edge` (that piece included),
 * through track of the pick's kind; just the piece when no such path joins them. */
export function pathTo(p: BpPicked, edge: number): number[] {
  const w = world.peek();
  const target = w?.edges.find((e) => e.id === edge);
  if (!w || !target || !p.edges.length) return [edge];
  const built = p.built ?? target.built;
  const usable = w.edges.filter((e) => e.built === built);
  const adj = adjacency(usable);
  const goal = new Set<number>();
  const picked = new Set(p.edges);
  for (const e of usable) if (picked.has(e.id)) goal.add(e.a), goal.add(e.b);
  // Dijkstra from the clicked piece's ends; `via` is the piece each node was reached through
  const dist = new Map<number, number>([[target.a, 0], [target.b, 0]]);
  const via = new Map<number, TrackEdge>();
  const heap: [number, number][] = [[0, target.a], [0, target.b]];
  const pop = () => {
    let k = 0;
    for (let i = 1; i < heap.length; i++) if (heap[i][0] < heap[k][0]) k = i;
    const v = heap[k];
    heap[k] = heap[heap.length - 1];
    heap.pop();
    return v;
  };
  let found = -1;
  while (heap.length) {
    const [d, n] = pop();
    if (d > (dist.get(n) ?? Infinity)) continue;
    if (goal.has(n)) { found = n; break; }
    for (const e of adj.get(n) ?? []) {
      if (e.id === edge) continue;
      const m = e.a === n ? e.b : e.a, dm = d + e.lengthM;
      if (dm < (dist.get(m) ?? Infinity)) dist.set(m, dm), via.set(m, e), heap.push([dm, m]);
    }
  }
  const out = [edge];
  for (let n = found; n >= 0 && via.has(n);) {
    const e = via.get(n)!;
    if (!picked.has(e.id)) out.push(e.id);
    n = e.a === n ? e.b : e.a;
  }
  return out;
}

/** T-1680: one picked piece of double track for each run Double up would cover (`TrackApi::double_up`
 * stops at junctions, track ends, single, high-speed track and crossovers). */
export function doubleUpPieces(p: BpPicked): number[] {
  const w = world.peek();
  if (!w) return [];
  const picked = new Set(p.edges);
  const ok = (e: TrackEdge) => e.tracks === 2 && !e.hsr && !e.crossover;
  const adj = adjacency(w.edges);
  const seen = new Set<number>(), out: number[] = [];
  for (const e0 of w.edges) {
    if (!picked.has(e0.id) || !ok(e0) || seen.has(e0.id)) continue;
    out.push(e0.id);
    const stack = [e0];
    seen.add(e0.id);
    while (stack.length) {
      const e = stack.pop()!;
      for (const n of [e.a, e.b]) {
        const at = adj.get(n) ?? [];
        const o = at.length === 2 ? at.find((x) => x !== e) : undefined;
        if (o && picked.has(o.id) && ok(o) && !seen.has(o.id)) seen.add(o.id), stack.push(o);
      }
    }
  }
  return out;
}

/** T-1680: Double up on every run of the pick's double track, one undoable edit a run; a run whose
 * piece went in an earlier run's edit (split by a station's crossover) is skipped. */
export async function doubleUpPicked(pieces: number[]) {
  let done = 0, why = "";
  for (const edge of pieces) {
    if (!world.peek()?.edges.some((e) => e.id === edge)) continue;
    const a = await edit({ op: "doubleUp", edge }, true);
    if (a.ok) done++;
    else why ||= describeIssues(a.issues);
  }
  if (!done) say(why || "This track cannot be doubled up.", true);
  else if (done < pieces.length) say(`${done} of ${pieces.length} runs doubled up. ${why}`.trim());
}

/** What the picked pieces are now: how many are switched off, their track's length, and the one
 * elevation all of it is on (null when mixed or ramped). */
export const pickedState = computed(() => {
  const w = world.value, p = bpPicked.value;
  if (!w) return { edges: 0, stations: 0, off: 0, on: 0, built: false, lengthM: 0, level: null as Level | null, hsr: 0, regular: 0 };
  const edges = new Map(w.edges.map((e) => [e.id, e]));
  const stations = stationsByNum(w);
  let off = 0, on = 0, lengthM = 0, hsr = 0, regular = 0;
  let level: Level | null | undefined;
  for (const id of p.edges) {
    const e = edges.get(id);
    if (!e) continue;
    e.off ? off++ : on++;
    lengthM += e.lengthM;
    const l = e.levelMin === e.levelMax ? e.levelMin : null;
    level = level === undefined || level === l ? l : null;
    if (canBeHsr(e)) e.hsr ? hsr++ : regular++;
  }
  for (const id of p.stations) (stations.get(id)?.off ? off++ : on++);
  return { edges: p.edges.length, stations: p.stations.length, off, on, built: !!p.built, lengthM, level: level ?? null, hsr, regular };
});

/** T-1898: blueprint track that can be high-speed or regular (`TrackApi::set_edges_hsr`): double, no crossover. */
const canBeHsr = (e: TrackEdge) => !e.built && e.tracks === 2 && !e.crossover;

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

/** T-1420: switch every disabled blueprint piece on again, picked or not: one undoable edit. */
export async function enableAll() {
  const w = world.peek();
  if (!w) return;
  const edges = w.edges.filter((e) => !e.built && e.off).map((e) => e.id);
  const stations = w.save.stations.filter((s) => !s.built && s.off).map((s) => s.num);
  if (edges.length + stations.length) await edit({ op: "blueprintOff", edges, stations, off: false }, true);
}

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** T-815: the picked blueprint track on one elevation, one undoable edit; pieces that cannot
 * change (a flat crossing would form, no corners to raise, a ramp with no room) are left as they
 * are and counted. */
export async function levelPicked(level: Level) {
  const p = bpPicked.peek();
  if (p.built || !p.edges.length) return;
  const a = await edit({ op: "edgesLevel", edges: p.edges, level }, true);
  const n = p.edges.length, skipped = a.skipped ?? 0, to = `level ${levelLabel(level)}`;
  if (!a.ok) say(n === 1 ? `This track cannot go to ${to}.` : `None of the ${n} pieces can go to ${to}.`, true);
  else if (skipped) say(`${n - skipped} of ${n} pieces set to ${to}. ${skipped} could not change.`);
  else say(`${count(n, "piece", "pieces")} set to ${to}.`);
}

/** T-1898: the picked blueprint double track made high-speed or regular, one undoable edit; single
 * track and crossovers stay as they are, counted in the notice as `levelPicked` does. */
export async function hsrPicked(hsr: boolean) {
  const p = bpPicked.peek();
  if (p.built || !p.edges.length) return;
  const a = await edit({ op: "edgesHsr", edges: p.edges, hsr }, true);
  if (!a.ok) return;
  const n = p.edges.length, skipped = a.skipped ?? 0, to = hsr ? "high-speed track" : "regular track";
  if (skipped) say(`${n - skipped} of ${n} pieces set to ${to}. ${skipped} could not change.`);
  else say(`${count(n, "piece", "pieces")} set to ${to}.`);
}

/** T-829: remove every picked piece in one edit: blueprint at once (one undo), constructed or
 * changing a line after a yes, as for one piece (game/remove.ts). */
export async function removePicked() {
  const p = bpPicked.peek(), s = pickedState.peek();
  if (!p.edges.length && !p.stations.length) return;
  const big = isBigDelete(s.lengthM);
  const what = [s.edges ? `${(big ? bigDistance : distance)(s.lengthM)} of ${p.built ? "built " : ""}track` : "", s.stations ? count(s.stations, p.built ? "built station" : "station", p.built ? "built stations" : "stations") : ""].filter(Boolean).join(" and ");
  const go = async () => {
    ask.value = null;
    const a = await edit({ op: "removePieces", edges: p.edges, stations: p.stations });
    if (a.ok) say(`Removed ${what}.`);
  };
  const lines = await lineEffects(p.edges, p.stations);
  if (!p.built && !lines.length && !big) return void go();
  ask.value = { ...removeQuestion(what, !!p.built, big), run: () => void go(), lines };
}

// Pieces removed, or no longer of the pick's kind (blueprint constructed), drop out of the pick;
// leaving the tool clears it.
effect(() => {
  const w = world.value;
  const p = bpPicked.peek();
  if (!w || (!p.edges.length && !p.stations.length)) return;
  const edges = new Set(w.edges.filter((e) => e.built === !!p.built).map((e) => e.id));
  const stations = new Set(w.save.stations.filter((s) => s.built === !!p.built).map((s) => s.num));
  const next = trimNodes({ edges: p.edges.filter((e) => edges.has(e)), stations: p.stations.filter((s) => stations.has(s)), built: p.built, ...(p.nodes ? { nodes: p.nodes } : {}) });
  if (next.edges.length !== p.edges.length || next.stations.length !== p.stations.length || next.nodes?.length !== p.nodes?.length) bpPicked.value = next;
});
// T-2283: a plain click on blueprint track makes that piece the pick too, so the selection bar shows
// for it; anything else selected clears it.
effect(() => {
  const t = tool.value, sel = selection.value;
  if (t === "blueprint") return;
  const e = t === "select" && sel?.kind === "track" ? world.peek()?.edges.find((x) => x.id === sel.edge) : undefined;
  if (!e || e.built || e.crossover) return clearBpPicked();
  const p = bpPicked.peek();
  if (p.edges.length !== 1 || p.edges[0] !== e.id || p.stations.length) bpPicked.value = { edges: [e.id], stations: [] };
});
