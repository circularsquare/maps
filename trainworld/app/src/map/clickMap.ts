// T-4214 stage C2: the map the build tools pick on, network and world. The network is the drawn one with the edges of
// the edits patched in since it was set (workers/clockPatch.ts sends them with each edit's word; workers/clockClient.ts
// takes them), so a click right after an edit acts on the map that edit left, before its network is built and drawn. A
// network set drops the patches that arrived before its state; later ones are laid on it again. Only the edges' points
// (and their snap grid) differ from the drawn network: its strokes, stations and marks are the drawn ones.
// The world is the last state's with the patched rows (`clickWorld`). `world` itself waits for the state: the panels,
// layers and memos on it would otherwise all run twice per edit (79 ms for a station on New York's save).
// Pairs: whatever reads `clickNet` (`Tools.pickNet`) reads `clickWorld`; whatever reads the renderer's network reads
// `world`. notes/T-4214.md "Stage C".

import type { NetworkBuffers } from "./network";
import type { WorldView } from "../game/types";
import { world } from "../game/world";
import { carryEdgeGrid, edgesNear } from "./snapIndex";

let patchedWorld: WorldView | null = null;

/** The world the tools pick on: the last state's with the edits patched in since, else `world`. */
export const clickWorld = (): WorldView | null => patchedWorld ?? world.peek();

/** workers/clockClient.ts: the patched world, or null when the state taken holds every patch. */
export function setClickWorld(w: WorldView | null) {
  patchedWorld = w;
}

interface Patch { order: number; set: Map<number, Float32Array>; gone: number[] }
let patches: Patch[] = [];
/** the drawn network the view was made on, and the view */
let on: NetworkBuffers | null = null;
let view: NetworkBuffers | null = null;
/** every edge a patch still laid on sets or takes out */
let touched = new Set<number>();

/** The network the tools pick on: `drawn` with the patches since it was set. */
export function clickNet(drawn: NetworkBuffers | null): NetworkBuffers | null {
  return drawn && view && on === drawn ? view : drawn;
}

/** Whether `edge` is set or taken out by a patch the drawn network does not hold yet. */
export const clickPatched = (edge: number) => !!view && touched.has(edge);

/** Lay a patch on: `set` the edges (local units, at the drawn network's anchor) and `gone` those taken out; `order` its
 * place in message arrival (`ClockClient.orderOf`). */
export function patchClickNet(drawn: NetworkBuffers, order: number, set: Map<number, Float32Array>, gone: number[]) {
  patches.push({ order, set, gone });
  rebuild(drawn, [patches[patches.length - 1]]);
}

/** A network was set that arrived as the `order`th message: the patches it holds go, the rest are laid on it. */
export function clickNetIn(drawn: NetworkBuffers | null, order: number) {
  if (!patches.length && !view) return;
  patches = patches.filter((p) => p.order > order);
  on = view = null;
  touched = new Set();
  if (drawn && patches.length) rebuild(drawn, patches);
}

/** Nothing laid on (a load, a worker started again). */
export function clickNetClear() {
  patchedWorld = null;
  patches = [];
  on = view = null;
  touched = new Set();
}

/** each edge map's highest id, once found */
const lastIds = new WeakMap<Map<number, Float32Array>, number>();
function lastId(m: Map<number, Float32Array>): number {
  let id = lastIds.get(m);
  if (id === undefined) {
    id = -Infinity;
    for (const k of m.keys()) if (k > id) id = k;
    lastIds.set(m, id);
  }
  return id;
}

/** the view with `more` laid on the current one (or on `drawn`), the edges in id order as the network lists them */
function rebuild(drawn: NetworkBuffers, more: Patch[]) {
  const from = on === drawn && view ? view : drawn;
  const set = new Map<number, Float32Array>(), gone = new Set<number>();
  for (const p of more) {
    for (const id of p.gone) { gone.add(id); set.delete(id); touched.add(id); }
    for (const [id, pts] of p.set) { set.set(id, pts); gone.delete(id); touched.add(id); }
  }
  const fresh = [...set.keys()].filter((id) => !from.edgePts.has(id)).sort((a, b) => a - b);
  // T-4284: a copy with the patch's edges set and taken out, when its new edges come after the rest in id order (the
  // model gives an edge a new id above every other), else made in id order
  let pts: Map<number, Float32Array>;
  if (!fresh.length || fresh[0] > lastId(from.edgePts)) {
    pts = new Map(from.edgePts);
    for (const id of gone) pts.delete(id);
    for (const [id, p] of set) if (from.edgePts.has(id)) pts.set(id, p);
    for (const id of fresh) pts.set(id, set.get(id)!);
  } else {
    pts = new Map<number, Float32Array>();
    let f = 0;
    for (const [id, p] of from.edgePts) {
      while (f < fresh.length && fresh[f] < id) { pts.set(fresh[f], set.get(fresh[f])!); f++; }
      if (gone.has(id)) continue;
      pts.set(id, set.get(id) ?? p);
    }
    for (; f < fresh.length; f++) pts.set(fresh[f], set.get(fresh[f])!);
  }
  if (fresh.length) lastIds.set(pts, Math.max(lastId(from.edgePts), fresh[fresh.length - 1]));
  else if (lastIds.has(from.edgePts)) lastIds.set(pts, lastIds.get(from.edgePts)!);
  const v: NetworkBuffers = { ...from, edgePts: pts };
  // (the snap grid laid over by the edges the patches set or took out: worked out on first use)
  edgesNear(from, 0, 0, 1);
  carryEdgeGrid(from, v, false, [...set.keys(), ...gone]);
  on = drawn;
  view = v;
}
