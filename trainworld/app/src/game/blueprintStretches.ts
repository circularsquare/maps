// T-2269: the blueprint as the Build tab's cost breakdown groups it: parts joined by blueprint
// track, each cut into stretches at stations, junctions and track ends, every row priced from the
// track model's pieces (`blueprint_piece_costs`, game/quotes.ts). Switching a row off is the pieces'
// own `off` (T-333): saved with the game, left out of the quote and of Construct.

import { signal } from "@preact/signals";
import type { Level, TrackEdge, WorldView } from "./types";
import { edit } from "./world";

export interface BpRow {
  kind: "stretch" | "station";
  label: string;
  edges: number[];
  /** blueprint station node ids */
  stations: number[];
  /** nodes the camera fits to */
  nodes: number[];
  lengthM: number;
  /** every piece's price, US$, and the part of it the quote counts now */
  cost: number;
  onCost: number;
  /** whether any piece is on, and any off */
  on: boolean;
  off: boolean;
  levelMin: Level;
  levelMax: Level;
  /** flat crossings on the stretch: how many, US$ */
  crossings: number;
  crossingCost: number;
  /** its station ends (a stretch between two stations, or the station itself), else empty: quad
   * track's twin rows are merged by them */
  ends: number[];
}

export interface BpPart extends Omit<BpRow, "kind" | "crossings" | "crossingCost" | "ends"> {
  rows: BpRow[];
  /** built and blueprint stations its track reaches */
  stationCount: number;
  /** junction switches, flyovers and crossovers, not switched on their own */
  junctions: number;
  junctionCost: number;
  junctionOnCost: number;
}

/** The blueprint track the breakdown's pointer is over, highlighted on the map (map/blueprintBox.ts). */
export const bpHover = signal<number[]>([]);

/** a stretch end this close to a station (pack metres, about half the longest platform) is named by it */
const NEAR_STATION_M = 250;
const END_WORDS ={ junction: "a junction", end: "a track end", built: "built track" } as const;

/** The parts of the blueprint, biggest first, from world `w` and the pieces' prices. */
export function blueprintParts(w: WorldView, pieces: ArrayLike<number>): BpPart[] {
  const edgeById = new Map(w.edges.map((e) => [e.id, e]));
  const stationByNum = new Map(w.save.stations.map((s) => [s.num, s]));
  const edgeCost = new Map<number, [number, boolean]>(), stationCost = new Map<number, [number, boolean]>();
  const junctionCost = new Map<number, [number, boolean]>();
  const crossings: [number, number, number, boolean][] = [], passages: [number, number, number, boolean][] = [];
  for (let i = 0; i + 4 < pieces.length; i += 5) {
    const kind = pieces[i], a = pieces[i + 1], b = pieces[i + 2], c = pieces[i + 3] * 1e6, on = pieces[i + 4] === 1;
    if (kind === 0) edgeCost.set(a, [c, on]);
    else if (kind === 1) stationCost.set(a, [c, on]);
    else if (kind === 2) junctionCost.set(a, [c, on]);
    else if (kind === 3) crossings.push([a, b, c, on]);
    else if (kind === 4) passages.push([a, b, c, on]);
  }
  const isStation = (n: number) => { const t = w.nodes.get(n); return !!t && t.platform > 0 && !t.bus && !t.port; };
  const bp = w.edges.filter((e) => !e.built);
  // every edge end at a node, crossovers apart, and the blueprint ones
  const allAt = new Map<number, TrackEdge[]>(), bpAt = new Map<number, TrackEdge[]>();
  const push = (m: Map<number, TrackEdge[]>, n: number, e: TrackEdge) => { const l = m.get(n); l ? l.push(e) : m.set(n, [e]); };
  for (const e of w.edges) if (!e.crossover) {
    push(allAt, e.a, e), push(allAt, e.b, e);
    if (!e.built) push(bpAt, e.a, e), push(bpAt, e.b, e);
  }
  // parts: nodes joined by blueprint track, crossovers included
  const up = new Map<number, number>();
  const find = (n: number): number => { let r = n; while (up.has(r) && up.get(r) !== r) r = up.get(r)!; up.set(n, r); return r; };
  for (const e of bp) { const ra = find(e.a), rb = find(e.b); if (ra !== rb) up.set(ra, rb); }
  const byRoot = new Map<number, TrackEdge[]>();
  for (const e of bp) push(byRoot, find(e.a), e);
  const splits = (n: number) => isStation(n) || (bpAt.get(n)?.length ?? 0) !== 2 || (allAt.get(n)?.length ?? 0) >= 3;
  const endName = (n: number): { name: string; station: boolean } => {
    if (isStation(n)) return { name: stationByNum.get(n)?.name || "Station", station: true };
    // track drawn from a station starts at its platform's end, so a station that close names the end
    const at = w.nodes.get(n);
    let best: string | null = null, bd = NEAR_STATION_M;
    if (at) for (const s of w.save.stations) {
      const sn = w.nodes.get(s.num);
      const d = sn ? Math.hypot(sn.x - at.x, sn.y - at.y) : Infinity;
      if (d < bd && !s.port && s.name) bd = d, best = s.name;
    }
    if (best) return { name: best, station: true };
    const all = allAt.get(n)?.length ?? 0;
    return { name: all >= 3 ? END_WORDS.junction : all <= 1 ? END_WORDS.end : END_WORDS.built, station: false };
  };
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  const stretchLabel = (a: number, b: number) => {
    let x = endName(a), y = endName(b);
    if (!x.station && y.station) [x, y] = [y, x];
    return `${x.station ? x.name : cap(x.name)} to ${y.name}`;
  };
  const used = new Set<number>(); // stations in a part's rows
  const parts: BpPart[] = [];
  const stationRow = (n: number): BpRow => {
    const [c, on] = stationCost.get(n) ?? [0, false];
    const s = stationByNum.get(n);
    const level = (s?.level ?? 0) as Level;
    return { kind: "station", label: s?.name || "Station", edges: (allAt.get(n) ?? []).map((e) => e.id), stations: [n], nodes: [n], lengthM: 0, cost: c, onCost: on ? c : 0, on: !s?.off, off: !!s?.off, levelMin: level, levelMax: level, crossings: 0, crossingCost: 0, ends: [n] };
  };
  for (const edges of byRoot.values()) {
    const seen = new Set<number>();
    const rows: BpRow[] = [];
    const named: number[] = []; // station nodes in walking order, for the part's name
    let junctions = 0, jCost = 0, jOn = 0;
    const nodesOf = new Set<number>();
    for (const e of edges) nodesOf.add(e.a), nodesOf.add(e.b);
    for (const e of edges) if (e.crossover) {
      const [c, on] = edgeCost.get(e.id) ?? [0, false];
      seen.add(e.id), junctions++, jCost += c, jOn += on ? c : 0;
    }
    for (const n of nodesOf) {
      const j = junctionCost.get(n);
      if (j) junctions++, jCost += j[0], jOn += j[1] ? j[0] : 0;
    }
    const visit = (n: number) => {
      if (isStation(n) && !named.includes(n)) {
        named.push(n);
        const s = stationByNum.get(n);
        if (s && !s.built && stationCost.has(n)) { rows.push(stationRow(n)); used.add(n); }
      }
      for (const e0 of bpAt.get(n) ?? []) {
        if (seen.has(e0.id)) continue;
        const ids: number[] = [];
        let at = n, e: TrackEdge | undefined = e0;
        while (e && !seen.has(e.id)) {
          seen.add(e.id), ids.push(e.id);
          at = e.a === at ? e.b : e.a;
          if (splits(at)) break;
          e = bpAt.get(at)!.find((x) => !seen.has(x.id));
        }
        rows.push(stretchRow(ids, n, at));
        visit(at);
      }
    };
    const stretchRow = (ids: number[], a: number, b: number): BpRow => {
      let lengthM = 0, cost = 0, onCost = 0, on = false, off = false, lo = 3 as Level, hi = -3 as Level;
      const nodes = new Set<number>();
      for (const id of ids) {
        const e = edgeById.get(id)!, [c, isOn] = edgeCost.get(id) ?? [0, false];
        nodes.add(e.a), nodes.add(e.b);
        lengthM += e.lengthM, cost += c, onCost += isOn ? c : 0;
        e.off ? (off = true) : (on = true);
        if (e.levelMin < lo) lo = e.levelMin;
        if (e.levelMax > hi) hi = e.levelMax;
      }
      return { kind: "stretch", label: stretchLabel(a, b), edges: ids, stations: [], nodes: [...nodes], lengthM, cost, onCost, on, off, levelMin: lo, levelMax: hi, crossings: 0, crossingCost: 0, ends: isStation(a) && isStation(b) ? [a, b] : [] };
    };
    // start at an end, a station end first, so a line reads from one end to the other
    const ends = [...nodesOf].filter((n) => (bpAt.get(n)?.length ?? 0) === 1);
    const start = ends.find(isStation) ?? ends[0] ?? [...nodesOf].find(splits) ?? edges[0].a;
    visit(start);
    for (const e of edges) if (!seen.has(e.id)) visit(e.a); // a loop the walk did not reach
    parts.push(partOf(rows, named, junctions, jCost, jOn, edges.filter((e) => e.crossover).map((e) => e.id)));
  }
  // blueprint stations on built track only
  for (const n of stationCost.keys()) {
    if (used.has(n) || !isStation(n) || stationByNum.get(n)?.built !== false) continue;
    if ((bpAt.get(n)?.length ?? 0) > 0) continue;
    parts.push(partOf([stationRow(n)], [n], 0, 0, 0, []));
  }
  // flat crossings go with a blueprint stretch they are on, passages with a blueprint station
  const rowOfEdge = new Map<number, BpRow>(), rowOfStation = new Map<number, BpRow>(), partOfRow = new Map<BpRow, BpPart>();
  for (const p of parts) for (const r of p.rows) {
    partOfRow.set(r, p);
    if (r.kind === "stretch") for (const e of r.edges) rowOfEdge.set(e, r);
    else rowOfStation.set(r.stations[0], r);
  }
  const addTo = (r: BpRow | undefined, c: number, on: boolean, crossing: boolean) => {
    if (!r) return;
    r.cost += c, r.onCost += on ? c : 0;
    if (crossing) r.crossings++, r.crossingCost += c;
    const p = partOfRow.get(r)!;
    p.cost += c, p.onCost += on ? c : 0;
  };
  for (const [e1, e2, c, on] of crossings) addTo(rowOfEdge.get(e1) ?? rowOfEdge.get(e2), c, on, true);
  for (const [a, b, c, on] of passages) addTo(rowOfStation.get(a) ?? rowOfStation.get(b), c, on, false);
  const near = (a: number, b: number) => {
    const x = w.nodes.get(a), y = w.nodes.get(b);
    return !!x && !!y && Math.hypot(x.x - y.x, x.y - y.y) < TWIN_M && stationByNum.get(a)?.name === stationByNum.get(b)?.name;
  };
  for (const p of parts) mergeTwins(p, near);
  return parts.sort((a, b) => b.cost - a.cost);

  /** `crossovers`: the crossover diagonals, which have no row and switch with the part */
  function partOf(rows: BpRow[], named: number[], junctions: number, jCost: number, jOn: number, crossovers: number[]): BpPart {
    const first = named[0], last = named[named.length - 1];
    const name = (n: number) => stationByNum.get(n)?.name || "Station";
    const label = !named.length ? "Track" : name(first) === name(last) ? name(first) : `${name(first)} to ${name(last)}`;
    const p: BpPart = {
      label, rows, edges: [], stations: [], nodes: [], lengthM: 0, cost: jCost, onCost: jOn, on: false, off: false,
      levelMin: 3, levelMax: -3, stationCount: new Set(named.map(name)).size, junctions, junctionCost: jCost, junctionOnCost: jOn,
    };
    for (const r of rows) {
      p.edges.push(...(r.kind === "stretch" ? r.edges : [])), p.stations.push(...r.stations), p.nodes.push(...r.nodes);
      p.lengthM += r.lengthM, p.cost += r.cost, p.onCost += r.onCost;
      p.on ||= r.on, p.off ||= r.off;
      if (r.levelMin < p.levelMin) p.levelMin = r.levelMin;
      if (r.levelMax > p.levelMax) p.levelMax = r.levelMax;
    }
    p.edges.push(...crossovers);
    return p;
  }
}

/** How far apart twin stations of quad track may be, pack metres */
const TWIN_M = 80;

/** Quad track's twin stations, and the stretches between twins, are one row each. */
function mergeTwins(p: BpPart, near: (a: number, b: number) => boolean) {
  const out: BpRow[] = [];
  const twin = (r: BpRow, o: BpRow) => r.kind === o.kind && r.ends.length > 0 && r.ends.length === o.ends.length
    && r.ends.every((a) => o.ends.some((b) => near(a, b)));
  for (const r of p.rows) {
    const was = out.find((o) => twin(r, o));
    if (!was) {
      out.push(r);
      continue;
    }
    was.edges.push(...r.edges), was.stations.push(...r.stations), was.nodes.push(...r.nodes);
    was.lengthM = Math.max(was.lengthM, r.lengthM);
    was.cost += r.cost, was.onCost += r.onCost, was.crossings += r.crossings, was.crossingCost += r.crossingCost;
    was.on ||= r.on, was.off ||= r.off;
    if (r.levelMin < was.levelMin) was.levelMin = r.levelMin;
    if (r.levelMax > was.levelMax) was.levelMax = r.levelMax;
  }
  p.rows = out;
}

/** Switch a row's or a part's pieces off (all of them, when any is on) or back on: one undoable edit. */
export async function switchRow(r: Pick<BpRow, "edges" | "stations" | "on">) {
  await edit({ op: "blueprintOff", edges: r.edges, stations: r.stations, off: r.on }, true);
}
