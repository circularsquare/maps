// T-3360: a network of buses and ferries with no rail near it (docs/ui-diagram.md "Buses and ferries"). Near rail, buses
// and ferries are drawn thin into the rail lines' layout and pull on nothing (thin.ts); with no rail near, nothing would
// lay them out, so here they make a graph of their own and are laid out as rail lines are, drawn thin. Its nodes are
// their stops (a station's stops one node per complex), each line its stops in order.

import { kAtY, toLocal, type Anchor } from "../map/geo";
import { simplifyLine, type DiagramGraph, type DiagramLine } from "./graph";

/** a bus or ferry stop within this many degrees of a rail region's stations' box goes with that region (thin.ts) */
export const THIN_BOX_PAD = 0.05;

/** a bus or ferry line as the clock worker has it (protocol.ts `ClockLine` with its `bus` row) */
export interface BusLineIn {
  id: number;
  colour: string;
  bus?: { routeLL: number[]; stops: { id: number; lng: number; lat: number; station: number }[]; ferry?: unknown };
}

const NO_COMPLEX = 0xffffffff;
/** a hop's route kept to within this of the ground's, metres */
const HOP_TOL_M = 15;

/** whether any of a line's stops lies within a box (west, south, east, north, degrees) padded by THIN_BOX_PAD */
export function stopsIn(l: BusLineIn, boxes: readonly number[][]): boolean {
  for (const s of l.bus?.stops ?? []) for (const b of boxes) {
    if (s.lng >= b[0] - THIN_BOX_PAD && s.lng <= b[2] + THIN_BOX_PAD && s.lat >= b[1] - THIN_BOX_PAD && s.lat <= b[3] + THIN_BOX_PAD) return true;
  }
  return false;
}

/** The graph of bus and ferry lines `lines`, places from anchor `at`; `complexOf` each node's complex (stations). */
export function busGraph(lines: readonly BusLineIn[], complexOf: Uint32Array, at: Anchor): DiagramGraph {
  const idOf = new Map<string, number>(), keys: number[] = [], sx: number[] = [], sy: number[] = [], cnt: number[] = [];
  const nodeOf = (s: { id: number; lng: number; lat: number; station: number }): number => {
    const st = s.station, c = st >= 0 && st < complexOf.length ? complexOf[st] : NO_COMPLEX;
    const name = st >= 0 ? (c !== NO_COMPLEX ? `c${c}` : `s${st}`) : `b${s.id}`, key = st >= 0 ? st : s.id;
    let g = idOf.get(name);
    if (g === undefined) idOf.set(name, (g = keys.length)), keys.push(key), sx.push(0), sy.push(0), cnt.push(0);
    else if (key < keys[g]) keys[g] = key;
    const [x, y] = toLocal(s.lng, s.lat, at);
    (sx[g] += x), (sy[g] += y), cnt[g]++;
    return g;
  };
  type Out = { l: BusLineIn; chain: number[]; cuts: number[] };
  const outs: Out[] = [];
  for (const l of [...lines].sort((a, b) => a.id - b.id)) {
    const b = l.bus;
    if (!b || b.stops.length < 2) continue;
    // each stop's place along the route (the vertex nearest it, found from the last one's on)
    const chain: number[] = [], cuts: number[] = [];
    let from = 0;
    for (const s of b.stops) {
      const g = nodeOf(s);
      let best = Infinity, bi = from;
      for (let i = from; i + 1 < b.routeLL.length; i += 2) {
        const d = (b.routeLL[i] - s.lng) ** 2 + (b.routeLL[i + 1] - s.lat) ** 2;
        if (d < best) (best = d), (bi = i);
      }
      from = bi;
      if (chain.length && chain[chain.length - 1] === g) continue;
      chain.push(g), cuts.push(bi);
    }
    if (chain.length >= 2) outs.push({ l, chain, cuts });
  }
  const n = keys.length;
  let mx = 0, my = 0;
  const px = new Float64Array(n), py = new Float64Array(n);
  for (let i = 0; i < n; i++) (px[i] = sx[i] / cnt[i]), (py[i] = sy[i] / cnt[i]), (mx += px[i]), (my += py[i]);
  mx /= Math.max(1, n);
  my /= Math.max(1, n);
  const k = kAtY(my, at);
  // nodes numbered in order of their keys, as the rail graph's are
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => keys[a] - keys[b]), renum = new Int32Array(n);
  order.forEach((g, i) => (renum[g] = i));
  const key = new Int32Array(n), x = new Float64Array(n), y = new Float64Array(n);
  order.forEach((g, i) => ((key[i] = keys[g]), (x[i] = (px[g] - mx) / k), (y[i] = -(py[g] - my) / k)));
  const dl: DiagramLine[] = outs.map(({ l, chain }) => ({ id: l.id, colour: l.colour.toLowerCase(), ...(l.bus!.ferry ? { ferry: true } : { bus: true }), chain: Int32Array.from(chain, (g) => renum[g]), stop: new Uint8Array(chain.length).fill(1) }));
  // edges, each line its own stroke; each edge's route as the first line on it runs it
  const em = new Map<number, number[]>(), hops = new Map<number, number[]>();
  dl.forEach((l, li) => {
    const { l: src, cuts } = outs[li], ll = src.bus!.routeLL;
    for (let i = 1; i < l.chain.length; i++) {
      const a = Math.min(l.chain[i - 1], l.chain[i]), b = Math.max(l.chain[i - 1], l.chain[i]), kk = a * n + b;
      let v = em.get(kk);
      if (!v) em.set(kk, (v = []));
      if (v[v.length - 1] !== li) v.push(li);
      if (hops.has(kk)) continue;
      const p: number[] = [];
      const i0 = Math.min(cuts[i - 1], cuts[i]), i1 = Math.max(cuts[i - 1], cuts[i]);
      for (let q = i0; q <= i1; q += 2) {
        const [qx, qy] = toLocal(ll[q], ll[q + 1], at);
        p.push((qx - mx) / k, -(qy - my) / k);
      }
      if (p.length < 4) continue;
      const fwd = l.chain[i - 1] === a ? cuts[i - 1] <= cuts[i] : cuts[i - 1] > cuts[i];
      if (!fwd) for (let s = 0, t = p.length - 2; s < t; s += 2, t -= 2) [p[s], p[s + 1], p[t], p[t + 1]] = [p[t], p[t + 1], p[s], p[s + 1]];
      hops.set(kk, simplifyLine(p, HOP_TOL_M));
    }
  });
  const ek = [...em.keys()].sort((a, b) => a - b);
  const ea = new Int32Array(ek.length), eb = new Int32Array(ek.length), first = new Int32Array(ek.length + 1), on: number[] = [];
  const hopAt = new Int32Array(ek.length + 1), hp: number[] = [];
  ek.forEach((kk, e) => {
    (ea[e] = Math.floor(kk / n)), (eb[e] = kk % n), (first[e] = on.length);
    on.push(...em.get(kk)!);
    const h = hops.get(kk);
    if (h) hp.push(...h);
    hopAt[e + 1] = hp.length / 2;
  });
  first[ek.length] = on.length;
  return { key, x, y, lines: dl, ea, eb, first, on: Int32Array.from(on), stroke: Int32Array.from(on), mid: [mx, my], origin: [at.lng, at.lat], k, hop: Float32Array.from(hp), hopAt };
}
