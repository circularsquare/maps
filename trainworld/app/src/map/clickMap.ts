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
import { kAtY } from "./geo";

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
/** the points of the edges a patch still laid on took out, as they were before it */
let gonePts = new Map<number, Float32Array>();

/** The network the tools pick on: `drawn` with the patches since it was set. */
export function clickNet(drawn: NetworkBuffers | null): NetworkBuffers | null {
  return drawn && view && on === drawn ? view : drawn;
}

/** Whether `edge` is set or taken out by a patch the drawn network does not hold yet. */
export const clickPatched = (edge: number) => !!view && touched.has(edge);

/** Lay a patch on: `set` the edges (local units, at the drawn network's anchor) and `gone` those taken out; `order` its
 * place in message arrival (`ClockClient.orderOf`). */
export function patchClickNet(drawn: NetworkBuffers, order: number, set: Map<number, Float32Array>, gone: number[]) {
  const had = clickNet(drawn) ?? drawn;
  fresh = { order, edges: new Map([...set].filter(([id]) => !had.edgePts.has(id))) };
  patches.push({ order, set, gone });
  rebuild(drawn, [patches[patches.length - 1]]);
}

/** the edges the newest patch laid on added to the map (their ids and points), and its arrival order */
let fresh: { order: number; edges: Map<number, Float32Array> } | null = null;
/** T-4214 stage E: the edges the patch that arrived as the `order`th message added to the map the tools pick on, by id with
 * their points, when it is the one laid last (a segment's, read at its word, right after its patch); null otherwise */
export const patchFresh = (order: number) => (fresh?.order === order ? fresh.edges : null);

/** A network was set that arrived as the `order`th message: the patches it holds go, the rest are laid on it. */
export function clickNetIn(drawn: NetworkBuffers | null, order: number) {
  if (!patches.length && !view) return;
  patches = patches.filter((p) => p.order > order);
  on = view = null;
  touched = new Set();
  gonePts = new Map();
  if (drawn && patches.length) rebuild(drawn, patches);
}

/** Nothing laid on (a load, a worker started again). */
export function clickNetClear() {
  patchedWorld = null;
  fresh = null;
  patches = [];
  on = view = null;
  touched = new Set();
  gonePts = new Map();
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
    for (const id of p.gone) {
      const was = set.get(id) ?? from.edgePts.get(id);
      if (was) gonePts.set(id, was);
      gone.add(id); set.delete(id); touched.add(id);
    }
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

/** T-4214 stage E: how far a piece of the tools' map may lie from the drawn track it came from, m (the drawn points are
 * single floats, and a piece is laid out again as it is cut) */
const UNDER_M = 1.5;
/** and in height, m */
const UNDER_Z_M = 3;
/** points along an edge tried for the drawn track under it */
const UNDER_SAMPLES = 9;

/** T-4214 stage E: what is drawn ahead of the network at a click is the tools' map's. `pieces` are edges of that map
 * drawn as stand-ins (a segment's, from its patch), `opened` those a later edit hides, `hidden` the edges hidden ahead.
 * A hidden edge the drawn network does not hold as the tools' map does (the far half of an edge a station just split, a
 * piece joined or cut since) opens the drawn edge or stand-in it lies on, as does a stand-in hidden or gone from the map:
 * that goes, and the edges of the tools' map on it that stay are drawn in its place. The drawn edges to hide and the
 * stand-ins to draw; null when the drawn network has no patch laid on it (nothing to read). */
export function openUnder(drawn: NetworkBuffers | null, hidden: ReadonlySet<number>, pieces: ReadonlyMap<number, Float32Array>, opened: ReadonlyMap<number, Float32Array>): { hide: number[]; pieces: Map<number, Float32Array> } | null {
  const v = drawn && view && on === drawn ? view : null;
  if (!drawn || !v) return null;
  const out = new Map(pieces), underD = new Set<number>(), underP = new Map(opened);
  for (const [id, pts] of pieces) if (!v.edgePts.has(id)) { out.delete(id); underP.set(id, pts); }
  for (const x of hidden) {
    // (a piece laid again under an id hidden before it is drawn, as the edits came)
    if (!touched.has(x) || out.has(x)) continue;
    const pts = v.edgePts.get(x) ?? gonePts.get(x) ?? opened.get(x);
    if (!pts) continue;
    const d = drawn.edgePts.get(x);
    // (drawn as the tools' map has it: hidden as before, the pieces since laid on it kept)
    if (d && samePts(d, pts)) { underD.add(x); continue; }
    for (const u of lyingOn(drawn, out, pts, UNDER_SAMPLES)) {
      if (u.drawn) underD.add(u.id);
      else { underP.set(u.id, out.get(u.id)!); out.delete(u.id); }
    }
  }
  if (!underD.size && !underP.size) return null;
  for (const y of touched) {
    const pts = v.edgePts.get(y);
    if (!pts || out.has(y) || hidden.has(y)) continue;
    const d = drawn.edgePts.get(y);
    if (d && !underD.has(y) && samePts(d, pts)) continue;
    const [x0, y0, z0] = alongAt(pts, 0.5);
    let on = false;
    for (const id of underD) if (!on && offPolyline(drawn.edgePts.get(id) ?? EMPTY, x0, y0, z0) < UNDER_M * kAtY(y0)) on = true;
    for (const p of underP.values()) if (!on && offPolyline(p, x0, y0, z0) < UNDER_M * kAtY(y0)) on = true;
    if (on) out.set(y, pts);
  }
  return { hide: [...underD], pieces: out };
}
const EMPTY = new Float32Array(0);

/** the drawn edges a patch changed and the stand-ins `pieces` that `pts` (local x, y, z) lies on, tried at `samples` points
 * along it */
function lyingOn(drawn: NetworkBuffers, pieces: ReadonlyMap<number, Float32Array>, pts: Float32Array, samples: number): { id: number; drawn: boolean }[] {
  const out = new Map<number, boolean>();
  if (pts.length < 6) return [];
  for (let s = 0; s < samples; s++) {
    const [x, y, z] = alongAt(pts, (s + 1) / (samples + 1));
    let best = UNDER_M * kAtY(y), hit: { id: number; drawn: boolean } | null = null;
    for (const id of touched) {
      const d = drawn.edgePts.get(id);
      if (!d) continue;
      const g = offPolyline(d, x, y, z);
      if (g < best) [best, hit] = [g, { id, drawn: true }];
    }
    for (const [id, p] of pieces) {
      const g = offPolyline(p, x, y, z);
      if (g < best) [best, hit] = [g, { id, drawn: false }];
    }
    if (hit) out.set(hit.id, hit.drawn);
  }
  return [...out].map(([id, d]) => ({ id, drawn: d }));
}

/** the point (local x, y, then z) fraction `f` of the way along polyline `pts` */
function alongAt(pts: Float32Array, f: number): [number, number, number] {
  const n = pts.length / 3;
  let total = 0;
  for (let k = 1; k < n; k++) total += Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
  let at = total * f;
  for (let k = 1; k < n; k++) {
    const l = Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
    if (at <= l || k === n - 1) {
      const t = l > 0 ? Math.min(1, at / l) : 0;
      return [pts[k * 3 - 3] + t * (pts[k * 3] - pts[k * 3 - 3]), pts[k * 3 - 2] + t * (pts[k * 3 + 1] - pts[k * 3 - 2]), pts[k * 3 - 1] + t * (pts[k * 3 + 2] - pts[k * 3 - 1])];
    }
    at -= l;
  }
  return [pts[0], pts[1], pts[2]];
}

/** how far (x, y) is from polyline `p` (local units), Infinity where the height there is off by more than UNDER_Z_M */
function offPolyline(p: Float32Array, x: number, y: number, z: number): number {
  let best = Infinity;
  for (let k = 0; k + 5 < p.length; k += 3) {
    const x0 = p[k], y0 = p[k + 1], dx = p[k + 3] - x0, dy = p[k + 4] - y0, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / l2)) : 0;
    const d = Math.hypot(x0 + t * dx - x, y0 + t * dy - y);
    if (d < best && Math.abs(p[k + 2] + t * (p[k + 5] - p[k + 2]) - z) <= UNDER_Z_M) best = d;
  }
  return best;
}

const samePts = (a: Float32Array, b: Float32Array) => a === b || (a.length === b.length && a.every((v, i) => v === b[i]));
