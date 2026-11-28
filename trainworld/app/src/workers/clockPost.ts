// The network diagram's graph, and posting to the main thread only what changed (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { anchor, EARTH_C, makeAnchor, toMerc01, type Anchor } from "../map/geo";
import { type NetworkBuffers } from "../map/network";
import { besideSamples } from "../map/corridors";
import { sameCorridor } from "../map/lineBands";
import { bandEdges, diagramGraph, type DiagramGraph } from "../diagram/graph";
import { nodeLngLat, regionGroups } from "../diagram/regions";
import { busGraph, stopsIn } from "../diagram/busGraph";
import { cutLines } from "../diagram/cut";
import { StrokeSender, strokesOfPieces, type StrokeKind, type StrokeMode } from "../map/strokePost";
import { NO_CITIES } from "./saveFile";
import { StateRows, weekendsOf } from "./stateRows";
import type { ClockEdge, ClockLine, ClockNode, ClockState, EditTouched, MoneyPatch, MoneyState } from "./protocol";
import { diffValue } from "./valuePatch";
import { complexOfNodes } from "./clockDemand";
import { diagramIn, drawnAllMoved, edgeMerc, pathsAllMoved, pathsMoved, pathsPosted, rowEnv, stateRows } from "./clockMain";
import { diagramCities, diagramInside, frame, send, timing } from "./clockState";

// ---------------------------------------------------------------- the network diagram's graph (T-2159)

/** whether the page shows the diagram, and the last network's lines, edges and polylines */
export let diagramOn = false;
/** T-3409: while cities are chosen, post the whole network's graphs too (a share sends the whole network) */
export let diagramWhole = false;
export type DiagramIn = { lines: ClockLine[]; edges: ClockEdge[]; pts: ReadonlyMap<number, Float32Array> };

/** T-2261: the graph's points are float32 local units, so they are made from the kept float64 mercator points, from an
 * origin of the network's own (its box's middle on a 1 degree grid), never the draw anchor: the same network gives the
 * same graph whatever the camera has done. Each edge's converted points are kept while its mercator ones and the origin
 * are, so the graph's arc lengths stay cached (per origin: a network of several regions has one each, T-3448). */
let diagramPts = new Map<number, { m: Float64Array; d: Map<string, Float32Array> }>();
/** a region's graph also has the track (not on its lines) within this of its stations' box, m */
const REGION_TRACK_M = 50_000;
function diagramFrame(pts: ReadonlyMap<number, Float32Array>, keep: Map<number, { m: Float64Array; d: Map<string, Float32Array> }>): { pts: Map<number, Float32Array>; at: Anchor } {
  const a = anchor();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const merc = new Map<number, Float64Array>();
  for (const [id, p] of pts) {
    let m = edgeMerc.get(id);
    if (!m || m.length !== p.length) {
      // not kept (a move or a clear since): from the local points, float32 rounded
      m = new Float64Array(p.length);
      for (let k = 0; k < p.length; k += 3) (m[k] = a.merc[0] + p[k] / EARTH_C), (m[k + 1] = a.merc[1] + p[k + 1] / EARTH_C), (m[k + 2] = p[k + 2]);
    }
    merc.set(id, m);
    for (let k = 0; k < m.length; k += 3) (x0 = Math.min(x0, m[k])), (x1 = Math.max(x1, m[k])), (y0 = Math.min(y0, m[k + 1])), (y1 = Math.max(y1, m[k + 1]));
  }
  const lng = merc.size ? Math.round(((x0 + x1) / 2) * 360 - 180) : a.lng;
  const lat = merc.size ? Math.round((Math.asin(Math.tanh(2 * Math.PI * (0.5 - (y0 + y1) / 2))) * 180) / Math.PI) : a.lat;
  const at = makeAnchor(lng, lat, 0), from = `${lng},${lat}`;
  const out = new Map<number, Float32Array>();
  for (const [id, m] of merc) {
    const had = diagramPts.get(id);
    let c = keep.get(id);
    if (!c || c.m !== m) keep.set(id, (c = { m, d: new Map() }));
    let d = c.d.get(from) ?? (had?.m === m ? had.d.get(from) : undefined);
    if (!d) {
      d = new Float32Array(m.length);
      for (let k = 0; k < m.length; k += 3) (d[k] = (m[k] - at.merc[0]) * EARTH_C), (d[k + 1] = (m[k + 1] - at.merc[1]) * EARTH_C), (d[k + 2] = m[k + 2]);
    }
    c.d.set(from, d);
    out.set(id, d);
  }
  return { pts: out, at };
}
/** the graph, after the network state it is of; the beside table is the one the bands just built. T-3448: a network of
 * several regions (diagram/regions.ts) gets a graph per region, each from an origin of its own, so a city far from the
 * rest gets the graph it gets alone (float32 points thousands of km from the origin move stations and passes). */
export function postDiagram() {
  if (!diagramIn) return;
  const t0 = performance.now(), keep: typeof diagramPts = new Map(), ends = new Map(diagramIn.edges.map((e) => [e.id, e]));
  // T-3409: chosen cities: their lines cut to the runs inside them, and the graphs built from those runs and their track
  // alone; none chosen, or none of their stations on a line, the whole network
  const none = diagramCities.includes(NO_CITIES);
  const c = diagramCities.length && !none ? cutLines(diagramIn.lines, ends, diagramIn.pts, complexOfNodes(), diagramInside.stations, diagramInside.stops) : null;
  const graphs = none ? [] : diagramGraphs(c ? { ...diagramIn, lines: c.lines, pts: c.pts } : diagramIn, c ? c.complexOf : complexOfNodes(), ends, keep);
  const whole = c && diagramWhole ? diagramGraphs(diagramIn, complexOfNodes(), ends, keep) : undefined;
  diagramPts = keep;
  send({ kind: "diagramGraph", graphs, ms: performance.now() - t0, ...(c || none ? { cut: true } : {}), ...(whole ? { whole } : {}) }, [...graphs, ...(whole ?? [])].flatMap((g) => [g.key.buffer, g.x.buffer, g.y.buffer, g.ea.buffer, g.eb.buffer, g.first.buffer, g.on.buffer]));
}
/** the graphs of the network `diagramIn`, stations by `cx` (T-3409: a cut network's has those outside made no station) */
function diagramGraphs(diagramIn: DiagramIn, cx: Uint32Array, ends: Map<number, ClockEdge>, keep: typeof diagramPts): DiagramGraph[] {
  const { lines, edges } = diagramIn;
  const build = (ls: ClockLine[], pts0: ReadonlyMap<number, Float32Array>) => {
    const { pts, at } = diagramFrame(pts0, keep);
    return diagramGraph(ls, edges, pts, cx, besideSamples(bandEdges(ls), pts, ends), at);
  };
  const whole = build(lines, diagramIn.pts), groups = regionGroups(whole);
  let graphs = [whole];
  if (groups.length >= 1) {
    // each region (a lone one too, so track with no line far away never moves it): its lines, their track, and other track
    // within REGION_TRACK_M of its stations' box (as a network of that region alone has), so its origin is the one it has alone
    const { lng, lat } = nodeLngLat(whole), a = anchor(), pad = REGION_TRACK_M / EARTH_C / Math.cos((whole.origin[1] * Math.PI) / 180);
    const regionOf = new Int32Array(whole.key.length), boxes = groups.map((nodes, r) => {
      const b = [Infinity, Infinity, -Infinity, -Infinity];
      for (const i of nodes) {
        regionOf[i] = r;
        const [x, y] = toMerc01(lng[i], lat[i]);
        (b[0] = Math.min(b[0], x - pad)), (b[1] = Math.min(b[1], y - pad)), (b[2] = Math.max(b[2], x + pad)), (b[3] = Math.max(b[3], y + pad));
      }
      return b;
    });
    const lineRegion = new Map<number | string, number>();
    for (const l of whole.lines) if (l.chain.length) lineRegion.set(runKey(l), regionOf[l.chain[0]]);
    const ls: ClockLine[][] = groups.map(() => []), pts = groups.map(() => new Map<number, Float32Array>());
    for (const l of lines) {
      const r = lineRegion.get(runKey(l));
      if (r === undefined) continue;
      ls[r].push(l);
      for (const p of l.path) {
        const q = diagramIn.pts.get(p >> 1);
        if (q) pts[r].set(p >> 1, q);
      }
    }
    for (const [id, q] of diagramIn.pts) {
      if (q.length < 3) continue;
      const m = edgeMerc.get(id), x = m ? m[0] : a.merc[0] + q[0] / EARTH_C, y = m ? m[1] : a.merc[1] + q[1] / EARTH_C;
      boxes.forEach((b, r) => { if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] && !pts[r].has(id)) pts[r].set(id, q); });
    }
    graphs = groups.map((_, r) => build(ls[r], pts[r]));
  }
  // T-3360: bus and ferry lines with no stop near a rail region's stations are a network of their own, laid out as rail is;
  // the rest are drawn thin into the rail layout by the page (diagram/thin.ts) and change no graph
  const buses = lines.filter((l) => l.ok && l.bus && l.bus.stops.length >= 2);
  if (buses.length) {
    const { lng, lat } = nodeLngLat(whole), boxes = (whole.key.length ? groups : []).map((nodes) => {
      const b = [Infinity, Infinity, -Infinity, -Infinity];
      for (const i of nodes) (b[0] = Math.min(b[0], lng[i])), (b[1] = Math.min(b[1], lat[i])), (b[2] = Math.max(b[2], lng[i])), (b[3] = Math.max(b[3], lat[i]));
      return b;
    });
    const lone = buses.filter((l) => !stopsIn(l, boxes));
    if (lone.length) {
      const frameOf = (ls: ClockLine[]) => {
        let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
        for (const l of ls) for (const p of l.bus!.stops) (w = Math.min(w, p.lng)), (e = Math.max(e, p.lng)), (s = Math.min(s, p.lat)), (n = Math.max(n, p.lat));
        return makeAnchor(Math.round((w + e) / 2), Math.round((s + n) / 2), 0);
      };
      const all = busGraph(lone, cx, frameOf(lone)), byId = new Map(lone.map((l) => [runKey(l), l]));
      const bg = regionGroups(all);
      for (const nodes of bg.length > 1 ? bg : [null]) {
        if (!nodes) { graphs.push(all); break; }
        const inIt = new Set(nodes), ls = all.lines.filter((l) => l.chain.length && inIt.has(l.chain[0])).map((l) => byId.get(runKey(l))!);
        graphs.push(busGraph(ls, cx, frameOf(ls)));
      }
    }
  }
  return graphs;
}
/** T-3409: a line's key among the graph's lines, the runs of a cut line told apart by their first stop */
const runKey = (l: { id: number; from?: number }): number | string => (l.from === undefined ? l.id : `${l.id}/${l.from}`);

// ---------------------------------------------------------------- posting only what changed (T-1029)

/** `?patch=` of the page: "0" posts every state and network whole, "check" also builds each network
 * from scratch and compares (the client compares the merged state too); otherwise changes only */
export let patchMode = "1";
/** the network's typed arrays that, when as last posted, are left out of the message (`NetPatch.same`) */
export const SAME_FIELDS = ["phases", "meta", "samples", "lineTable", "sampleBend", "bends", "lineCars", "lineColours", "sampleLane", "sampleBand", "samplePass", "passing", "corridor", "sampleStretch", "sampleStroke", "stations", "stationAxis", "stationShape", "stationLine", "stationLength", "stationLinkIdx", "stationLinks", "lots", "junctions", "marks"] as const;
type SameField = (typeof SAME_FIELDS)[number];
/** What the client holds: the last network posted (its serial, draw anchor, a copy of each
 * `SAME_FIELDS` array and its edges' points) and the rows of the last state, by id. Cleared to post
 * the next state whole. */
export const posted = {
  net: null as null | { serial: number; anchor: number; pts: Map<number, Float32Array> },
  arrays: new Map<string, ArrayBufferView>(),
  edges: new Map<number, ClockEdge>(), nodes: new Map<number, ClockNode>(), lines: new Map<number, ClockLine>(),
  edgeIds: [] as number[], nodeIds: [] as number[], lineIds: [] as number[],
  /** T-4291: the first rows read (workers/rowLog.ts) the client has not had, once a state was posted */
  read: undefined as number | undefined,
  /** T-4363: the states posted: the page holds the patches sent since the last (clockPatch.ts `laid`) */
  posts: 0,
};
export const sender = new StrokeSender();
export function postWhole() {
  posted.net = null;
  sender.forget();
  posted.arrays.clear();
  posted.edges.clear(); posted.nodes.clear(); posted.lines.clear();
  posted.edgeIds = []; posted.nodeIds = []; posted.lineIds = [];
  posted.read = undefined;
  moneyHeld = null;
}

/** T-3803: the money record as last posted (a copy of its own, so nothing the worker changes in place reaches it) and
 * its serial. The record goes with every state and "money" message as its changes from that one: whole, it was about
 * 2 MB on a 2,647-station save (the regions' ledgers), 35-65 ms of structured clone a post there. */
let moneyHeld: MoneyState | null = null;
let moneySerial = 0;
/** T-4149: the record as held for the next diff. Its ledgers (`ledger`, `hours`, `regions`, `lineDays`) are held as they
 * are: arrays made anew for each record, of entries never changed once made (`moneyState`'s kept copies), so the
 * next diff passes the unchanged ones by identity. The rest is copied: it can share parts the worker changes in place. */
const HELD_AS_THEY_ARE = new Set(["ledger", "hours", "regions", "lineDays"]);
function heldMoney(m: MoneyState): MoneyState {
  const rest: Record<string, unknown> = {}, src = m as unknown as Record<string, unknown>;
  for (const k of Object.keys(src)) if (!HELD_AS_THEY_ARE.has(k)) rest[k] = src[k];
  const copy = structuredClone(rest), out: Record<string, unknown> = {};
  // (in the record's own key order, so the next diff finds the same keys)
  for (const k of Object.keys(src)) out[k] = HELD_AS_THEY_ARE.has(k) ? src[k] : copy[k];
  return out as unknown as MoneyState;
}
export function moneyPost(m: MoneyState): { money: MoneyState; moneySerial: number; checkMoney?: undefined } | { money: undefined; moneyPatch: MoneyPatch; checkMoney?: MoneyState } {
  if (!moneyHeld || patchMode === "0") {
    moneyHeld = heldMoney(m);
    return { money: m, moneySerial: ++moneySerial };
  }
  const patch = diffValue(moneyHeld, m);
  if (patch) moneyHeld = heldMoney(m);
  const base = moneySerial;
  return { money: undefined, moneyPatch: { base, serial: ++moneySerial, patch }, ...(patchMode === "check" ? { checkMoney: m } : {}) };
}

const bytesOf = (v: ArrayBufferView) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
export function sameBytes(a: ArrayBufferView | undefined, b: ArrayBufferView): boolean {
  if (a === b) return true;
  if (!a || a.byteLength !== b.byteLength || a.constructor !== b.constructor) return false;
  if (a.byteLength % 4 === 0 && a.byteOffset % 4 === 0 && b.byteOffset % 4 === 0) {
    const x = new Int32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  }
  const x = bytesOf(a), y = bytesOf(b);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}
export function keepPosted(k: string, v: ArrayBufferView) {
  posted.arrays.set(k, (v as Float32Array).slice());
}
/** arrays kept by reference and posted as a copy: their makers keep them as they are, never changed in place, and hand
 * out the same array while unchanged: the rows' phase tables (`StateRows.phases`), the bands' Traffic table (T-4187,
 * `LineBands.corridor`, which the bands read again) and the train samples' tables (T-4153 stage 10, `NetCache.tables`,
 * which the next build patches) */
const BY_REFERENCE: ReadonlySet<SameField> = new Set(["phases", "corridor", "sampleStroke", "sampleLane", "sampleBand", "sampleStretch", "samplePass"]);
function keepPostedField(out: NetworkBuffers, k: SameField, v: ArrayBufferView) {
  if (!BY_REFERENCE.has(k)) return keepPosted(k, v);
  posted.arrays.set(k, v);
  (out as Record<SameField, unknown>)[k] = (v as Float32Array).slice();
}
/** the phase tables of a state that does not use them (the network's are posted as changes) */
export const NO_PHASES = new Float32Array(0);

/** The network as it is posted: whole (a copy of its edges' points in one buffer), or with
 * `patch` and the fields the client already has left out. */
export function netForPost(n: NetworkBuffers, scratch?: NetworkBuffers): NetworkBuffers {
  const anchorNow = anchor().version;
  const pts = n.edgePts;
  const whole = patchMode === "0" || !n.patch || !posted.net || posted.net.serial !== n.patch.base || posted.net.anchor !== anchorNow;
  const pieces = n.strokePieces!;
  const out: NetworkBuffers = { ...n };
  delete out.strokePieces;
  const t0 = performance.now();
  const modes = {} as Record<StrokeKind, StrokeMode>, ranges = {} as Record<StrokeKind, Int32Array>, moves = {} as Record<StrokeKind, Int32Array>;
  for (const kind of ["track", "lines"] as const) {
    const r = sender.post(kind, pieces[kind], n[kind].count, whole ? null : n.patch![kind], whole ? 0 : kind === "track" ? n.patch!.oldTrack : n.patch!.oldLines);
    out[kind] = r.strokes;
    modes[kind] = r.mode;
    ranges[kind] = r.ranges;
    moves[kind] = r.moves;
  }
  timing.strokes = performance.now() - t0;
  timing.modes = (["whole", "same", "patch"] as const).indexOf(modes.track) * 10 + (["whole", "same", "patch"] as const).indexOf(modes.lines);
  if (scratch) {
    const full = { track: strokesOfPieces("track", pieces.track), lines: strokesOfPieces("lines", pieces.lines) };
    checkNet({ ...n, ...full }, scratch);
    out.checkStrokes = full;
  }
  if (whole) {
    delete out.patch;
    let total = 0;
    for (const p of pts.values()) total += p.length;
    const all = new Float32Array(total), mine = new Map<number, Float32Array>();
    let at = 0;
    for (const [id, p] of pts) {
      all.set(p, at);
      mine.set(id, all.subarray(at, at + p.length));
      at += p.length;
    }
    out.edgePts = mine;
    for (const k of SAME_FIELDS) { const v = n[k]; if (v) keepPostedField(out, k, v); }
  } else {
    const patch = { ...n.patch!, same: [] as string[], modes, ranges, moves };
    const ts0 = performance.now();
    for (const k of SAME_FIELDS) {
      const v = n[k];
      if (!v) continue;
      // the phase tables are one array until a table changes (`StateRows.phases`)
      if (k === "phases" ? posted.arrays.get(k) === v : sameBytes(posted.arrays.get(k), v)) {
        patch.same.push(k);
        (out as Record<SameField, unknown>)[k] = undefined;
        // (T-4187: the maker's new array alike, held in place of the last, which it no longer keeps)
        if (BY_REFERENCE.has(k)) posted.arrays.set(k, v);
      } else keepPostedField(out, k, v);
    }
    timing.pSame = performance.now() - ts0;
    const changed = new Map<number, Float32Array>();
    const was = posted.net!.pts;
    for (const [id, p] of pts) if (was.get(id) !== p) changed.set(id, p.slice());
    out.edgePts = changed;
    out.patch = patch;
  }
  posted.net = { serial: n.serial!, anchor: anchorNow, pts };
  return out;
}

/** T-1170's check (`?patch=shiftcheck`): a build from pieces moved to a new anchor against one from
 * scratch: the same segments, positions near the anchor and distances along within a few
 * centimetres (logged), anything else an error. */
export function checkShift(n: NetworkBuffers, full: NetworkBuffers) {
  /** the most off anywhere, within 100 km or so of the anchor, and along strokes (`dist`) */
  let most = 0, near = 0, along = 0;
  const bad: string[] = [];
  for (const kind of ["track", "lines"] as const) {
    const a = strokesOfPieces(kind, n.strokePieces![kind]), b = full[kind];
    if (a.count !== b.count) { bad.push(`${kind} count`); continue; }
    for (let i = 0; i < a.seg.length; i++) {
      const d = Math.abs(a.seg[i] - b.seg[i]);
      most = Math.max(most, d);
      if (Math.abs(b.seg[i - (i % 2)]) < 1e5 && Math.abs(b.seg[i - (i % 2) + 1]) < 1e5) near = Math.max(near, d);
    }
    for (let i = 0; i < a.dist.length; i++) along = Math.max(along, Math.abs(a.dist[i] - b.dist[i]));
    for (const f of ["colour", "edge", "level", "flags"] as const) if (!sameBytes(a[f], b[f])) bad.push(`${kind}.${f}`);
  }
  if (!sameBytes(n.stations, full.stations)) bad.push("stations");
  console.log(`T-1170 check: pieces moved to the new anchor are off by up to ${most.toFixed(4)} local units, ${near.toFixed(4)} near it, distances along by ${along.toFixed(4)}`);
  if (bad.length || near > 0.2 || along > 0.5) console.error(`T-1170 check: moved pieces differ in ${bad.join(", ") || "position"} (near ${near.toFixed(3)}, along ${along.toFixed(3)})`);
}

/** T-1029's check (`?patch=check`): the network built from the cache against one built from scratch. */
function checkNet(n: NetworkBuffers, full: NetworkBuffers) {
  const bad: string[] = [];
  const cmp = (k: string, a: unknown, b: unknown) => {
    if (ArrayBuffer.isView(a) && ArrayBuffer.isView(b)) { if (!sameBytes(a, b)) bad.push(k); }
    else if (JSON.stringify(a) !== JSON.stringify(b)) bad.push(k);
  };
  for (const k of Object.keys(full) as (keyof NetworkBuffers & string)[]) {
    if (k === "serial" || k === "patch" || k === "anchor") continue;
    const a = n[k], b = full[k];
    // (T-4153: the Traffic table numbers its ribbons and recipes as they came, a build from scratch as the lines use them)
    if (k === "corridor") {
      if (!(a instanceof Float32Array) || !(b instanceof Float32Array) || !sameCorridor(a, b)) bad.push(k);
      continue;
    }
    if (k === "edgePts") {
      const x = a as Map<number, Float32Array>, y = b as Map<number, Float32Array>;
      if (x.size !== y.size || [...x.keys()].join() !== [...y.keys()].join() || [...y].some(([id, p]) => !sameBytes(x.get(id), p))) bad.push(k);
    } else if (b && typeof b === "object" && "seg" in b) for (const f of Object.keys(b)) cmp(`${k}.${f}`, (a as any)?.[f], (b as any)[f]);
    else cmp(k, a, b);
  }
  // the patch's runs rebuild the strokes from the last build's
  if (bad.length) console.error(`T-1029 check: the network built from the cache differs from a full build in ${bad.join(", ")}`);
}

/** `?patch=check`: the rows just read against a fresh read of the whole model (T-1382). The fresh
 * read leaves the kept rows alone and its bus rows are the same ones. Pricing a planned line (an ok
 * line not running) runs a trial that moves headway lines a little, so with one the lines are not
 * compared here; checks/t1382-rows.mjs compares them with two models. */
export function checkRows(got: { edges: ClockEdge[]; nodes: ClockNode[]; lines: ClockLine[] }, a: TrackApi, ups: Map<number, number>) {
  const fresh = new StateRows(), full = fresh.read(a, rowEnv, frame, ups);
  const planned = got.lines.some((l) => l.ok && !l.running);
  const keys: ("edges" | "nodes" | "lines")[] = planned ? ["edges", "nodes"] : ["edges", "nodes", "lines"];
  const bad = keys.filter((k) => !sameValue(got[k], full[k]));
  if (bad.length) console.error(`T-1382 check: the ${bad.join(", ")} read from changes differ from a full read`);
  // T-4300: the complexes the names were folded by, and what renames read of them, against the whole read's
  const cx = stateRows.complexesDiffer(a, rowEnv), names = stateRows.namesDiffer(fresh);
  if (cx) console.error(`T-4300 check: the station rows' complexes kept from changes differ from a whole build (${cx})`);
  if (names) console.error(`T-4300 check: the complexes' names kept from changes differ from a full read's (${names})`);
  // T-4149: the weekends read for many lines at once, the track limits shared, against each line's own reads
  const ids = got.lines.map((l) => l.id), many = weekendsOf(a, ids);
  const off = ids.filter((id) => { const [w, r] = many.get(id)!; return !sameBytes(w, a.line_weekend(id)) || !sameBytes(r, a.line_row_weekend(id)); });
  if (off.length) console.error(`T-4149 check: the weekends of lines ${off.slice(0, 8).join(", ")} read together differ from each line's own`);
}

/** Rows field for field, arrays and typed arrays by value. */
function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return Number.isNaN(a) && Number.isNaN(b);
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!sameValue(a[i], b[i])) return false;
    return true;
  }
  if (ArrayBuffer.isView(a)) return ArrayBuffer.isView(b) && sameBytes(a, b);
  const x = a as Record<string, unknown>, y = b as Record<string, unknown>;
  const kx = Object.keys(x);
  if (kx.length !== Object.keys(y).length) return false;
  for (const k of kx) if (!sameValue(x[k], y[k])) return false;
  return true;
}

/** The rows that differ from those last posted, with every row's id in order (null when the ids
 * are as last posted), and the ids of those gone; `posted` takes the new rows. */
function rowsDelta<T extends { id: number }>(rows: T[], by: Map<number, T>, ids: number[]): { ids: number[] | null; rows: T[]; gone: number[] } {
  const changed: T[] = [];
  let sameIds = rows.length === ids.length;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (sameIds && ids[i] !== r.id) sameIds = false;
    // T-3803: the rows keep an unchanged row's object (stateRows.ts), so most pass with one lookup; `by` is written only
    // where a row changed, not cleared and refilled
    const was = by.get(r.id);
    if (was === r) continue;
    if (!sameValue(was, r)) changed.push(r);
    by.set(r.id, r);
  }
  // `by` holds every row now, and only those gone besides: there are some exactly when it holds more
  const gone: number[] = [];
  if (by.size > rows.length) {
    const now = new Set<number>();
    for (const r of rows) now.add(r.id);
    for (const id of by.keys()) if (!now.has(id)) { by.delete(id); gone.push(id); }
  }
  return { ids: sameIds ? null : rows.map((r) => r.id), rows: changed, gone };
}

/** T-4291: `rowsDelta` from the rows reads since the client's last (`RowLog.since`): only the rows they changed are looked at
 * (`now`: a row as the last read handed it out), and the ids in order are sent only where some came, went or moved */
function loggedDelta<T extends { id: number }>(rows: T[], by: Map<number, T>, ids: ReadonlySet<number>, order: boolean, now: (id: number) => T | undefined): { ids: number[] | null; rows: T[]; gone: number[] } {
  const changed: T[] = [], gone: number[] = [];
  for (const id of ids) {
    const r = now(id), was = by.get(id);
    if (r) {
      if (r === was) continue;
      if (!sameValue(was, r)) changed.push(r);
      by.set(id, r);
    } else if (was) { by.delete(id); gone.push(id); }
  }
  return { ids: order ? rows.map((r) => r.id) : null, rows: changed, gone };
}

/** T-4291's check (`?patch=check`): after a delta from the log, what the client holds (`posted`) is the state's rows */
function checkLogged(s: ClockState, d: { ids: number[] | null }[]) {
  const bad: string[] = [];
  (["edges", "nodes", "lines"] as const).forEach((k, j) => {
    const rows = s[k] as { id: number }[], by = posted[k] as Map<number, unknown>, ids = d[j].ids ?? posted[k === "edges" ? "edgeIds" : k === "nodes" ? "nodeIds" : "lineIds"];
    if (by.size !== rows.length || ids.length !== rows.length || rows.some((r, i) => by.get(r.id) !== r || ids[i] !== r.id)) bad.push(k);
  });
  if (bad.length) console.error(`T-4291 check: the ${bad.join(", ")} posted from the rows' log differ from the state's`);
}

/** T-4149: a state's `touched` from its rows' deltas and the lines whose runs moved; the drawn runs' list starts again
 * once a network is posted, the phase tables' (T-4280) once a state carrying them is */
function touchedOf(s: ClockState, ...d: { rows: { id: number }[]; gone: number[] }[]): EditTouched {
  // (a line whose row is gone has no drawn runs)
  for (const id of d[2].gone) pathsMoved.add(id);
  const [edges, nodes, lines] = d.map((x) => [...x.rows.map((r) => r.id), ...x.gone]);
  const every = (kept: Set<number>) => [...new Set([...kept, ...s.lines.map((l) => l.id)])];
  const paths = pathsAllMoved ? every(pathsMoved) : [...pathsMoved];
  const phases = stateRows.phasesAll ? every(stateRows.phasesMoved) : [...stateRows.phasesMoved];
  const t: EditTouched = drawnAllMoved ? touchedAll() : { edges, nodes, lines, paths, phases };
  listsPosted(s);
  return t;
}
const touchedAll = (): EditTouched => ({ all: true, edges: [], nodes: [], lines: [], paths: [], phases: [] });
/** the lists `touched` is made from start again with what state `s` posts */
function listsPosted(s: ClockState) {
  if (s.net) pathsPosted();
  if (s.net || s.service) stateRows.phasesPosted();
}

/** Post a state, its edges, nodes and lines as only those that changed since the last state posted
 * (T-648, T-1029: copying them to the main thread and back into objects was most of a state's time
 * there on a big network). */
export function postState(s: ClockState, transfer: Transferable[]) {
  posted.posts++;
  const read = s.rowsRead;
  delete s.rowsRead;
  if (patchMode === "0" || !posted.edgeIds.length && !posted.nodeIds.length && !posted.lineIds.length) {
    rowsDelta(s.edges, posted.edges, posted.edgeIds);
    rowsDelta(s.nodes, posted.nodes, posted.nodeIds);
    rowsDelta(s.lines, posted.lines, posted.lineIds);
    posted.edgeIds = s.edges.map((e) => e.id); posted.nodeIds = s.nodes.map((n) => n.id); posted.lineIds = s.lines.map((l) => l.id);
    posted.read = read === undefined ? undefined : read + 1;
    listsPosted(s);
    send({ kind: "state", state: { ...s, ...moneyPost(s.money!), touched: touchedAll() } }, transfer);
    return;
  }
  const t0 = performance.now();
  // T-4291: the rows the reads since the client's last changed, when this state holds the last read's lists; else every row
  // compared (a state built before a later read, a whole read among them, `?patch=walk`)
  const since = read !== undefined && read === stateRows.lastRead && posted.read !== undefined && patchMode !== "walk" ? stateRows.log.since(posted.read) : null;
  const [de, dn, dl] = since
    ? [loggedDelta(s.edges, posted.edges, since.ids[0], since.order[0], (id) => stateRows.edgeNow(id)), loggedDelta(s.nodes, posted.nodes, since.ids[1], since.order[1], (id) => stateRows.nodeNow(id)),
      loggedDelta(s.lines, posted.lines, since.ids[2], since.order[2], (id) => stateRows.lineNow(id))]
    : [rowsDelta(s.edges, posted.edges, posted.edgeIds), rowsDelta(s.nodes, posted.nodes, posted.nodeIds), rowsDelta(s.lines, posted.lines, posted.lineIds)];
  if (patchMode === "check" && since) checkLogged(s, [de, dn, dl]);
  posted.read = read === undefined ? undefined : read + 1;
  if (de.ids) posted.edgeIds = de.ids;
  if (dn.ids) posted.nodeIds = dn.ids;
  if (dl.ids) posted.lineIds = dl.ids;
  const full = patchMode === "check" ? { edges: s.edges, nodes: s.nodes, lines: s.lines } : undefined;
  const touched = touchedOf(s, de, dn, dl);
  const t1 = performance.now(), money = moneyPost(s.money!), t2 = performance.now();
  // T-4149: the post's parts, for the edit's answer
  timing.pRows = t1 - t0; timing.pMoney = t2 - t1;
  const delta = { edges: { ids: de.ids, rows: de.rows }, nodes: { ids: dn.ids, rows: dn.rows }, lines: { ids: dl.ids, rows: dl.rows } };
  send({ kind: "state", state: { ...s, ...money, edges: [], nodes: [], lines: [], delta, touched, ...(full ? { checkRows: full } : {}) }, ...(import.meta.env.DEV ? { timing: { ...timing } } : {}) }, transfer);
  timing.pSend = performance.now() - t2;
}
export function putDiagramOn(v: typeof diagramOn) { diagramOn = v; }
export function putDiagramWhole(v: typeof diagramWhole) { diagramWhole = v; }
export function putPatchMode(v: typeof patchMode) { patchMode = v; }
