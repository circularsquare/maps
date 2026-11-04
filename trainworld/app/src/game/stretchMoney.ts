// T-2420 (design notes/T-2411.md): money by stretch of track, for the Profit track display. On the day
// shown, each running rail line's money is placed on the edges of its path, then summed over the lines
// on each edge:
// - fares: the per-km part (and demand's bent-curve extra) by the paid passenger-km on each stretch
//   between stops, both directions, as `incomeFor` charges it; the line's base fares (its T-2104 share)
//   half by those passenger-km, half by boardings at the stretch's two stops (a through stop's half to
//   each side, an end stop's all to its one stretch);
// - running trains: the line's train-km cost a day (as `runningToday`, over a whole day) by length;
// - upkeep: each edge's split between the lines on it by trains a day, as `line_upkeep` does; each
//   station's split so between the lines stopping there, then half to the stretch on each side.
// So a line's edges sum to its fares on that day and its upkeep share. Track no running line uses keeps
// its own upkeep. Construction and trains bought are left out, as in the Cities tab. Bus and ferry lines
// are not placed yet (their share of a station's upkeep is left off the track).
//
// A track stretch, what the colour and the hover cover, runs between stations, junctions and track
// ends; every edge on it takes the stretch's profit over its km.

import { computed } from "@preact/signals";
import { demandView, type DemandView } from "./demand";
import { faresOf, lineFareMul, type Fares } from "./fareCurve";
import { lineFares } from "./fareInEffect";
import { trainKind } from "./format";
import { baseTrips, lineGeo, moneyState, moneyTerms, moneyWorld, scalesOf, type MoneyTerms } from "./money";
import { PERIODS } from "./periods";
import { edgeUpkeep, stationUpkeep } from "./regionMoney";
import { stopsById, world } from "./world";
import type { LineInput, WorldView } from "./types";

/** Hours a day at high, medium and low demand, as the clock worker's `LEVEL_HOURS`. */
const LEVEL_HOURS = (["high", "medium", "low"] as const).map((lev) => PERIODS.reduce((s, p) => s + (p.demand === lev ? p.to - p.from : 0), 0));

/** US$ a day. */
export interface Money3 { fares: number; running: number; upkeep: number }
export interface StretchFigures extends Money3 {
  /** its km */
  km: number;
  /** no running line uses it */
  bare: boolean;
}
export interface StretchMoney {
  /** by edge id: the money placed on it, each line's part by line id */
  edges: Map<number, { all: Money3; lines: Map<string, Money3> }>;
  /** by edge id: its track stretch's figures (every edge of a stretch shares the object) */
  stretchOf: Map<number, StretchFigures>;
  /** by edge id, its stretch's profit per km a day; `NO_PROFIT` for blueprint track and ids with no edge */
  perKm: Float32Array;
}
/** What `perKm` holds where there is no figure. */
export const NO_PROFIT = 1e30;

const zero = (): Money3 => ({ fares: 0, running: 0, upkeep: 0 });

let memo: { shape: WorldView; t: MoneyTerms; dv: DemandView; run: string; out: StretchMoney } | null = null;
/** Money by edge and stretch on the day shown; null until demand has an answer. Redone when the
 * network (`moneyWorld`), the fares, the day's demand or a line's trains change, not on every state. */
export const stretchMoney = computed<StretchMoney | null>(() => {
  const shape = moneyWorld.value, w = world.value, t = moneyTerms.value, dv = demandView.value, m = moneyState.peek();
  if (!shape || !w || !t || !dv || !m) return null;
  let run = [m.runCostTrainKm, m.runCostCarKm, m.hsRunCostTrainKm, m.hsRunCostCarKm, m.economy].join();
  for (const l of w.save.lines) run += `|${l.tph.high},${l.tph.medium},${l.tph.low},${w.lineStats[l.id]?.cars}`;
  for (const e of w.edges) if (e.built) run += `,${e.id}`;
  if (memo && memo.shape === shape && memo.t === t && memo.dv === dv && memo.run === run) return memo.out;
  const out = stretchMoneyOf(w, t, dv, shape);
  memo = { shape, t, dv, run, out };
  return out;
});

/** `shape`: the same network as `moneyWorld` holds it, for the fare figures' caches (`lineGeo`). */
export function stretchMoneyOf(w: WorldView, t: MoneyTerms, dv: DemandView, shape = w): StretchMoney {
  const m = moneyState.peek()!;
  const E = t.economy;
  const byId = stopsById(shape);
  const edgeById = new Map(w.edges.map((e) => [e.id, e]));
  const edges: StretchMoney["edges"] = new Map();
  const part = (edge: number, line: string): Money3 => {
    let e = edges.get(edge);
    if (!e) edges.set(edge, (e = { all: zero(), lines: new Map() }));
    let p = e.lines.get(line);
    if (!p) e.lines.set(line, (p = zero()));
    return p;
  };
  // trains a day of each running line, the weight `line_upkeep` splits shared upkeep by
  const weight = new Map<string, number>();
  for (const l of w.save.lines) {
    if (w.lineStats[l.id]?.status !== "running") continue;
    weight.set(l.id, l.tph.high * LEVEL_HOURS[0] + l.tph.medium * LEVEL_HOURS[1] + l.tph.low * LEVEL_HOURS[2]);
  }
  const based = baseTrips(shape, dv, t.fares, byId);
  const curves = new Map<Fares, Fares>();
  /** per rail line placed: each stretch between stops, its edges and their metres in it */
  const placed = new Map<string, [number, number][][]>();
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], n = l.stops.length;
    if (!st || st.status !== "running" || st.bus || n < 2 || !st.path?.length) continue;
    const pieces = stretchEdges(l, st.path, st.stopS[0], edgeById);
    if (!pieces) continue;
    placed.set(l.id, pieces);
    // running trains by length along the path
    const k = trainKind(m, st.fast);
    const runDay = (k.runCostTrainKm + st.cars * k.runCostCarKm) * E * (weight.get(l.id) ?? 0) * 2 * st.km;
    let pathM = 0;
    for (const p of st.path) pathM += edgeById.get(p >> 1)?.lengthM ?? 0;
    if (pathM > 0) for (const p of st.path) part(p >> 1, l.id).running += runDay * (edgeById.get(p >> 1)?.lengthM ?? 0) / pathM;
    // fares, as `incomeFor` charges them, by stretch
    const dm = dv.lines.get(l.id);
    if (!dm || st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
    const sl = shape.save.lines.find((x) => x.id === l.id) ?? l;
    const g = lineGeo(shape, sl, byId), km = g.km;
    const lf = lineFares(dm, t.fares);
    let f = curves.get(lf);
    if (!f) curves.set(lf, (f = faresOf(lf, lf)));
    const { sc, from } = scalesOf(g, sl, byId, lf);
    const pk = new Array<number>(n - 1).fill(0);
    for (const seg of dm.loads) {
      if (seg.length !== km.length) continue;
      // run 1's stretch j covers run 0's stretch n - 2 - j
      for (let j = 0; j < seg.length; j++) pk[j < n - 1 ? j : 2 * n - 3 - j] += seg[j] * km[j] * from[j];
    }
    let pkAll = 0;
    for (const v of pk) pkAll += v;
    if (!(pkAll > 0)) continue;
    const fx = dm.fareX?.reduce((a, b) => a + b, 0) ?? 0;
    const mul = lineFareMul({ fast: st.fast, ferry: false }, lf);
    const base = E * f.base * (based.get(l.id) ?? 0);
    const bw = new Array<number>(n - 1).fill(0);
    let bwAll = 0;
    dm.boardings.forEach((v, s) => {
      const b = v * (sc[s] ?? 1), sides = s === 0 || s === n - 1 ? 1 : 2;
      if (s > 0) bw[s - 1] += b / sides;
      if (s < n - 1) bw[s] += b / sides;
      bwAll += b;
    });
    for (let s = 0; s < n - 1; s++) {
      const share = pk[s] / pkAll;
      const fare = E * (mul * f.perKm * pk[s] + fx * share) + base / 2 * share + base / 2 * (bwAll > 0 ? bw[s] / bwAll : share);
      spread(pieces[s], (edge, sh) => (part(edge, l.id).fares += fare * sh));
    }
  }
  // upkeep: each built edge between the running lines on it by trains a day
  const bareUpkeep = new Map<number, number>();
  for (const e of w.edges) {
    const u = edgeUpkeep(e, t);
    if (!(u > 0)) continue;
    let total = 0;
    for (const id of e.lines) total += weight.get(id) ?? 0;
    if (total > 0) {
      for (const id of e.lines) {
        const wt = weight.get(id) ?? 0;
        if (wt > 0) part(e.id, id).upkeep += u * wt / total;
      }
    } else bareUpkeep.set(e.id, u);
  }
  // each built station between the running lines stopping there, a rail line's part beside it
  const stopping = new Map<string, { l: LineInput; k: number }[]>();
  for (const l of w.save.lines) {
    if (!(weight.get(l.id)! > 0)) continue;
    l.stops.forEach((s, k) => {
      const v = stopping.get(s);
      if (v) v.push({ l, k });
      else stopping.set(s, [{ l, k }]);
    });
  }
  for (const s of w.save.stations) {
    if (!s.built) continue;
    const on = stopping.get(s.id);
    if (!on) continue;
    const u = s.port ? (t.upkeepPort ?? 0) * E : stationUpkeep(t);
    let total = 0;
    for (const { l } of on) total += weight.get(l.id)!;
    for (const { l, k } of on) {
      const pieces = placed.get(l.id);
      if (!pieces) continue;
      const mine = u * weight.get(l.id)! / total;
      const sides = [k - 1, k].filter((x) => x >= 0 && x < pieces.length);
      for (const x of sides) spread(pieces[x], (edge, sh) => (part(edge, l.id).upkeep += mine / sides.length * sh));
    }
  }
  for (const v of edges.values()) {
    for (const p of v.lines.values()) { v.all.fares += p.fares; v.all.running += p.running; v.all.upkeep += p.upkeep; }
  }
  // track stretches, and each edge's profit per km
  const stretchOf = new Map<number, StretchFigures>();
  let maxId = 0;
  for (const e of w.edges) maxId = Math.max(maxId, e.id);
  const perKm = new Float32Array(maxId + 1).fill(NO_PROFIT);
  for (const chain of trackStretches(w)) {
    const fig: StretchFigures = { fares: 0, running: 0, upkeep: 0, km: 0, bare: true };
    for (const e of chain) {
      const a = edges.get(e.id)?.all;
      if (a) { fig.fares += a.fares; fig.running += a.running; fig.upkeep += a.upkeep; }
      fig.upkeep += bareUpkeep.get(e.id) ?? 0;
      if (e.lines.some((id) => (weight.get(id) ?? 0) > 0)) fig.bare = false;
      fig.km += e.lengthM / 1000;
    }
    const v = fig.km > 0 ? (fig.fares - fig.running - fig.upkeep) / fig.km : 0;
    for (const e of chain) { stretchOf.set(e.id, fig); perKm[e.id] = v; }
  }
  return { edges, stretchOf, perKm };
}

/** A line's stretches between stops along its path: each one's edges and the metres of each inside
 * it, by where the stops sit along run 0 (`stopS[0]`); null when an edge is missing. */
function stretchEdges(l: LineInput, path: number[], s0: number[], edgeById: Map<number, { lengthM: number }>): [number, number][][] | null {
  const n = l.stops.length;
  if (s0.length !== n) return null;
  const out: [number, number][][] = Array.from({ length: n - 1 }, () => []);
  let at = 0, k = 0;
  for (const p of path) {
    const e = edgeById.get(p >> 1);
    if (!e) return null;
    const x = at, y = at + e.lengthM;
    at = y;
    while (k < n - 1 && s0[k + 1] <= x) k++;
    for (let j = k; j < n - 1 && s0[j] < y; j++) {
      const o = Math.min(s0[j + 1], y) - Math.max(s0[j], x);
      if (o > 1e-6) out[j].push([p >> 1, o]);
    }
  }
  return out;
}

/** `f` for each edge of a stretch with its share of the stretch's metres. */
function spread(piece: [number, number][], f: (edge: number, share: number) => void) {
  let total = 0;
  for (const [, mm] of piece) total += mm;
  if (total > 0) for (const [edge, mm] of piece) f(edge, mm / total);
}

/** Built track in runs between stations, junctions and track ends (a crossover's diagonal is a run
 * of its own, between the junctions at its ends). */
function trackStretches(w: WorldView): WorldView["edges"][] {
  const at = new Map<number, WorldView["edges"]>();
  const built = w.edges.filter((e) => e.built);
  for (const e of built) for (const nd of [e.a, e.b]) {
    const v = at.get(nd);
    if (v) v.push(e);
    else at.set(nd, [e]);
  }
  const through = (nd: number) => {
    const node = w.nodes.get(nd);
    return !!node && !node.platform && at.get(nd)?.length === 2;
  };
  const seen = new Set<number>(), out: WorldView["edges"][] = [];
  for (const e0 of built) {
    if (seen.has(e0.id)) continue;
    seen.add(e0.id);
    const chain = [e0];
    for (const start of [e0.a, e0.b]) {
      let nd = start, e = e0;
      while (through(nd)) {
        const next = at.get(nd)!.find((x) => x !== e)!;
        if (seen.has(next.id)) break;
        seen.add(next.id);
        chain.push(next);
        nd = next.a === nd ? next.b : next.a;
        e = next;
      }
    }
    out.push(chain);
  }
  return out;
}
