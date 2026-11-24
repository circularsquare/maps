// T-4148: Construct and the select tool's edits drawn at the click (game/pending.ts): the pieces built, switched off,
// raised, made single or double, or removed, and the track panel's speed limit and one way. Guessed from the op and the
// world when it is sent (game/editGuess.ts); the worker's word (workers/clockWordPieces.ts) replaces the guess with what
// the model did.

import { live } from "../map/live";
import type { NetworkBuffers, Strokes } from "../map/network";
import { drawnEdgesOf, type RawStations } from "../map/lineAhead";
import { PLATFORM_POINTS } from "../map/stationGeometry";
import type { EdgeLook, EditOp } from "../workers/protocol";
import { ahead, shown, type Show } from "./pending";
import { blueprintTotalNow, lineQuoteNow } from "./quotes";
import { LIMIT_MIN } from "./pieceLimit";
import { TRACK_SPECS } from "./trackKinds";
import { planStopsOf, type TrackEdge, type WorldView } from "./types";
import { linesByNum, stationsById, world } from "./world";

/** the station buffers' blueprint flag (map/network.ts flags a station: 2 blueprint) */
const ST_BLUEPRINT = 2;

/** an edit's track type (`edgesKind`, `rebuild`: 1 regular, 2 high speed, 3 tram) as the track model numbers it */
const KIND_OF_OP = { 1: 0, 2: 1, 3: 3 } as const;
/** the track model's number for an edge's type */
const kindOf = (e: TrackEdge): 0 | 1 | 2 | 3 | 4 => (e.hsr ? 1 : ((e.kind ?? 0) as 0 | 2 | 3 | 4));

/** `kmh` as the piece keeps it (the sim's `piece_limit`): none at or over its top speed, never under the floor */
function keptLimit(e: TrackEdge, kmh: number): number {
  const top = e.kind ? TRACK_SPECS[e.kind === 4 ? "busway" : e.kind === 2 ? "streetTram" : "tram"].vCapKmh : e.hsr ? 320 : 160;
  return kmh <= 0 || kmh >= top ? 0 : Math.max(LIMIT_MIN, kmh);
}

/** the station flag for a station switched off (network.ts) */
const ST_OFF = 8;

/** Blueprint stations `ids` of the network, in the buffers' form with their flags changed by `flag` (built: the
 * blueprint flag off; switched off or on). Null where none of them is drawn as blueprint. */
function restyledStations(n: NetworkBuffers, ids: Iterable<number>, flag: (f: number) => number): RawStations | null {
  const at = new Map(n.stationIds.map((s, i) => [s, i]));
  const idx: number[] = [];
  for (const id of ids) { const i = at.get(String(id)); if (i !== undefined && ((n.stations[i * 3 + 2] | 0) & ST_BLUEPRINT)) idx.push(i); }
  if (!idx.length) return null;
  const P2 = 2 * PLATFORM_POINTS, m = idx.length;
  const out: RawStations = { st: new Float32Array(m * 3), axis: new Float32Array(m * 2), shape: new Float32Array(m * 4), line: new Float32Array(m * P2), count: m };
  idx.forEach((i, k) => {
    out.st.set(n.stations.subarray(i * 3, i * 3 + 3), k * 3);
    out.st[k * 3 + 2] = flag(n.stations[i * 3 + 2] | 0);
    if (n.stationAxis) out.axis.set(n.stationAxis.subarray(i * 2, i * 2 + 2), k * 2);
    if (n.stationShape) out.shape.set(n.stationShape.subarray(i * 4, i * 4 + 4), k * 4);
    if (n.stationLine && n.stationLine.length >= (i + 1) * P2) out.line.set(n.stationLine.subarray(i * P2, (i + 1) * P2), k * P2);
  });
  return out;
}

/** blueprint pieces a construct would build: not built, not switched off, a crossover's partner with it */
function toBuild(w: WorldView, edges: Iterable<number> | null, stations: Iterable<number> | null) {
  const byId = new Map(w.edges.map((e) => [e.id, e]));
  const es = new Set<number>(), ss = new Set<number>();
  for (const id of edges ?? w.edges.map((e) => e.id)) {
    const e = byId.get(id);
    if (!e || e.built || e.off) continue;
    es.add(id);
    if (e.crossover && e.partner !== undefined) es.add(e.partner);
  }
  const byNum = new Map(w.save.stations.map((s) => [s.num, s]));
  for (const num of stations ?? w.save.stations.map((s) => s.num)) { const s = byNum.get(num); if (s && !s.built && !s.off) ss.add(num); }
  return { es, ss, track: [...es].reduce((a, id) => a + (byId.get(id)?.cost ?? 0), 0) };
}

/** `quote`: the worker's price for exactly this construct (US$), when the page has it; else only its track's is known,
 * which the construct costs at least (stations, upgrades, junctions and trains come on top) */
function constructShow(w: WorldView, edges: Iterable<number> | null, stations: Iterable<number> | null, quote?: number): Show | null {
  const { es, ss, track } = toBuild(w, edges, stations);
  if (!es.size && !ss.size) return null;
  const show: Show = { edges: new Map([...es].map((e) => [e, { built: true }])), ...(quote !== undefined ? { cost: quote / 1e6 } : { least: track / 1e6 }) };
  const n = live.overlay?.renderer.net;
  const raw = n && ss.size ? restyledStations(n, ss, (f) => f & ~ST_BLUEPRINT) : null;
  if (raw) { show.hideStations = [...ss]; show.rawStations = [raw]; }
  return show;
}

/** What an edit shows at its click (game/editGuess.ts), or null for an edit this does not draw ahead. */
export function pieceGuess(op: EditOp): Show | null {
  const w = world.peek();
  if (!w) return null;
  const byId = () => new Map(w.edges.map((e) => [e.id, e]));
  switch (op.op) {
    case "construct": return constructShow(w, op.edges, op.stations);
    case "constructAll": return constructShow(w, null, null, blueprintTotalNow(w.version));
    case "constructLine": {
      const l = linesByNum(w).get(op.line), n = live.overlay?.renderer.net;
      if (!l || !n || w.lineStats[l.id]?.bus) return null;
      // (its trains are bought once it runs: a quote with trains to buy is not what Construct charges alone)
      const q = lineQuoteNow(op.line, w.version);
      const stops = new Set(planStopsOf(l)), byStation = stationsById(w);
      return constructShow(w, drawnEdgesOf(n, op.line), [...stops].map((s) => byStation.get(s)?.num).filter((x): x is number => x !== undefined), q && !q.trains ? q.build : undefined);
    }
    case "blueprintOff": {
      const b = byId(), edges = new Map<number, { off: boolean }>();
      for (const id of op.edges) if (b.get(id) && !b.get(id)!.built) edges.set(id, { off: op.off });
      const n = live.overlay?.renderer.net, raw = n && op.stations.length ? restyledStations(n, op.stations, (f) => (op.off ? f | ST_OFF : f & ~ST_OFF)) : null;
      const show: Show = { ...(edges.size ? { edges } : {}), ...(raw ? { hideStations: op.stations, rawStations: [raw] } : {}) };
      return edges.size || raw ? show : null;
    }
    case "edgesKind": {
      const b = byId(), kind = KIND_OF_OP[op.kind], edges = new Map<number, EdgeLook>();
      for (const id of op.edges) { const e = b.get(id); if (e && !e.built && !e.crossover && e.tracks === 2 && e.kind !== 4 && kindOf(e) !== kind) edges.set(id, { kind }); }
      return edges.size ? { edges } : null;
    }
    case "rebuild": {
      const b = byId(), edges = new Map<number, EdgeLook>();
      for (const id of op.edges) {
        const e = b.get(id);
        if (!e || !e.built || e.crossover) continue;
        const look: EdgeLook = {};
        if (op.level !== null) look.level = [op.level, op.level];
        if (op.kind) look.kind = KIND_OF_OP[op.kind];
        if (op.tracks) look.tracks = op.tracks;
        if (Object.keys(look).length) edges.set(id, look);
      }
      return edges.size ? { edges } : null;
    }
    case "edgesLevel": {
      const b = byId(), edges = new Map<number, { level: [number, number] }>();
      for (const id of op.edges) { const e = b.get(id); if (e && !e.built && !e.crossover) edges.set(id, { level: [op.level, op.level] }); }
      return edges.size ? { edges } : null;
    }
    case "edgesTracks": {
      const b = byId(), edges = new Map<number, { tracks: 1 | 2 }>();
      for (const id of op.edges) { const e = b.get(id); if (e && !e.built && !e.crossover) edges.set(id, { tracks: op.tracks }); }
      return edges.size ? { edges } : null;
    }
    case "removePieces": {
      const b = byId(), hide = new Set<number>();
      for (const id of op.edges) { const e = b.get(id); if (!e) continue; hide.add(id); if (e.crossover && e.partner !== undefined) hide.add(e.partner); }
      return hide.size || op.stations.length ? { hideEdges: hide, hideStations: op.stations } : null;
    }
    case "pieceLimit": {
      const b = byId(), fields = new Map<string, Record<string, unknown>>();
      for (const id of op.edges) { const e = b.get(id); if (e) fields.set("edge:" + id, { limit: keptLimit(e, op.kmh) || undefined }); }
      return fields.size ? { fields } : null;
    }
    case "oneWay": case "oneWays": {
      const edges = op.op === "oneWay" ? [op.edge] : op.edges, dirs = op.op === "oneWay" ? [op.dir] : op.dirs;
      const fields = new Map<string, Record<string, unknown>>(), looks = new Map<number, EdgeLook>(), b = byId();
      edges.forEach((id, k) => {
        fields.set("edge:" + id, { oneWay: dirs[k] === 1 || dirs[k] === 2 ? dirs[k] : undefined });
        const e = b.get(id);
        if (e && e.tracks === 1 && !e.crossover) looks.set(id, { oneWay: dirs[k], ends: [e.a, e.b] });
      });
      return { fields, ...(looks.size ? { edges: looks } : {}) };
    }
    default: return null;
  }
}

/** A blueprint drag let go: the track it previewed is drawn as it will be, and the track it moves hidden, until the
 * network holds the move (`strokes` null: nothing to draw, a refused preview). */
export function dragAhead<T>(strokes: Strokes | null | undefined, hide: Iterable<number>, send: () => Promise<T>): Promise<T> {
  if (!strokes?.count) return send();
  return ahead({ track: [{ ...strokes, marks: undefined, chevrons: undefined }], hideEdges: hide }).send(send);
}

/** The picked pieces with the panel fields edits not yet in the world give (the speed limit and one way). */
export function withShown<T extends { id: number; limit?: number; oneWay?: 1 | 2 }>(edges: readonly T[]): T[] {
  return edges.map((e) => {
    const limit = shown("edge:" + e.id, "limit", e.limit), oneWay = shown("edge:" + e.id, "oneWay", e.oneWay);
    return limit === e.limit && oneWay === e.oneWay ? e : { ...e, limit, oneWay };
  });
}
