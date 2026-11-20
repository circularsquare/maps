// T-3360: bus and ferry lines on the network diagram, drawn last as thin lines fitted into the layout the rail lines
// set (docs/ui-diagram.md "Buses and ferries"). They pull on nothing: each route is carried through the map the sheet
// carried with the stations (sheetLines.ts `carryMap`), moved so it meets the drawn pill of each station it stops at,
// and fitted with runs at 0, 45 and 90 degrees between those stations. Runs in the layout worker. Pure: no DOM.

import { fitOcti, rounded } from "./borders";
import { simplifyLine } from "./graph";
import type { CarryMap } from "./sheetThin";
import type { DiagramLayout } from "./layout";
import { STUB, stubEnd, type Drawing, type V } from "./draw";
import type { DiagramWater } from "./water";
import { toCoast, waterGrid, waterPath, wetAt } from "./ferryWater";

/** a bus or ferry line as the page sends it, in its region's graph metres */
export interface ThinIn {
  id: number;
  colour: string;
  ferry: boolean;
  /** its route, x, y pairs */
  route: Float64Array;
  /** its stops in run order, x, y pairs, and per stop the graph node of the station it stops at (-1 for none) */
  stops: Float64Array;
  node: Int32Array;
  /** T-3409: cut at the chosen cities' edge, the way it goes on there (east, north), drawn as a stub (draw.ts `stubEnd`) */
  stubA?: [number, number];
  stubB?: [number, number];
}

/** a bus or ferry line drawn, layout units */
export interface ThinDrawn {
  id: number;
  colour: string;
  ferry: boolean;
  /** its path as commands (draw.ts `toPath`); the Path2D made on the page when first painted */
  cmds: Float64Array;
  path?: Path2D;
  /** its fitted vertices, x, y pairs (picks) */
  ribbon: Float64Array;
  /** its stops that are not at a station drawn on the diagram, x, y pairs */
  stops: Float64Array;
}

/** the route kept to within this of the ground before it is carried, metres */
const ROUTE_TOL_M = 15;
/** a fitted run strays at most this many pitches from the carried route, a ferry's (whose water the carry moves freely)
 * this many */
const FIT_TOL = 2, FIT_TOL_FERRY = 4;
/** corners rounded with this radius, pitches */
const CORNER = 2;

/** T-3824: a ferry over the drawn water within this many pitches of its carried route; a stop that is no station moved
 * onto the drawn coast where its route first meets the water, else the coast within PORT_REACH pitches; the way fitted
 * within FIT_TOL_WATER pitches of the way found, over no more land than it */
const WATER_PAD = 12, PORT_REACH = 6, FIT_TOL_WATER = 3;

/** T-3824: a ferry's way over the drawn water (`water`, before the thin lines): each stop that is no station moved onto
 * the drawn coast, then from stop to stop the cheapest way over a grid of the drawn water (land far dearer). The way
 * (layout units), the indices of its stops in it, each moved port by its route vertex, and a fit's guard that no run
 * crosses more land than the way found did. */
function overWater(water: DiagramWater, X: Float64Array, Y: Float64Array, anchors: [number, number, number, number][], pitch: number) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < X.length; i++) (x0 = Math.min(x0, X[i])), (x1 = Math.max(x1, X[i])), (y0 = Math.min(y0, Y[i])), (y1 = Math.max(y1, Y[i]));
  const pad = WATER_PAD * pitch, span = Math.max(x1 - x0, y1 - y0) + 2 * pad;
  const G = waterGrid(water, [x0 - pad, y0 - pad, x1 + pad, y1 + pad], Math.max(0.5 * pitch, span / 500));
  const at = new Map<number, [number, number]>();
  /** a port on land: where its route, toward the next stop (the last stop's toward the one before), first meets the
   * drawn water */
  const alongRoute = (q: number): [number, number] | null => {
    const v = anchors[q][0], w = anchors[q + 1]?.[0] ?? anchors[q - 1][0], dir = w > v ? 1 : -1, step = G.cell / 3;
    for (let i = v; i !== w; i += dir) {
      const n = Math.max(1, Math.ceil(Math.hypot(X[i + dir] - X[i], Y[i + dir] - Y[i]) / step));
      for (let t = 1; t <= n; t++) { const x = X[i] + ((X[i + dir] - X[i]) * t) / n, y = Y[i] + ((Y[i + dir] - Y[i]) * t) / n; if (wetAt(G, x, y)) return [x, y]; }
    }
    return null;
  };
  const ends = anchors.map(([v, , , station], q): [number, number] => {
    if (station) return [X[v], Y[v]];
    const c = (!wetAt(G, X[v], Y[v]) && alongRoute(q)) || toCoast(G, X[v], Y[v], PORT_REACH * pitch) || [X[v], Y[v]];
    at.set(v, c);
    return c;
  });
  // the route before the first stop and after the last (to the chosen cities' edge, where a stub goes on) kept as
  // carried
  const px: number[] = [], py: number[] = [], cuts: number[] = [];
  const v0 = anchors[0][0], vN = anchors[anchors.length - 1][0];
  for (let i = 0; i < v0; i++) px.push(X[i]), py.push(Y[i]);
  cuts.push(px.length);
  for (let a = 0; a + 1 < ends.length; a++) {
    const p = waterPath(G, ends[a][0], ends[a][1], ends[a + 1][0], ends[a + 1][1]) ?? [...ends[a], ...ends[a + 1]];
    // steps going on the same way joined, so the fit has few points to try
    for (let q = a ? 2 : 0; q < p.length; q += 2) {
      const n = px.length;
      if (n >= 2 && q + 2 < p.length && Math.abs((px[n - 1] - px[n - 2]) * (p[q + 1] - py[n - 1]) - (py[n - 1] - py[n - 2]) * (p[q] - px[n - 1])) < 1e-9 * G.cell * G.cell && cuts[cuts.length - 1] < n - 1) (px[n - 1] = p[q]), (py[n - 1] = p[q + 1]);
      else px.push(p[q]), py.push(p[q + 1]);
    }
    cuts.push(px.length - 1);
  }
  for (let i = vN + 1; i < X.length; i++) px.push(X[i]), py.push(Y[i]);
  if (px.length - 1 > cuts[cuts.length - 1]) cuts.push(px.length - 1);
  if (cuts[0] > 0) cuts.unshift(0);
  // land along the way found, summed from its start (layout units)
  const step = G.cell / 2, landTo = new Float64Array(px.length);
  const landOn = (ax: number, ay: number, bx: number, by: number) => {
    const l = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.ceil(l / step));
    let land = 0;
    for (let t = 0; t < n; t++) if (!wetAt(G, ax + ((bx - ax) * (t + 0.5)) / n, ay + ((by - ay) * (t + 0.5)) / n)) land += l / n;
    return land;
  };
  for (let k = 1; k < px.length; k++) landTo[k] = landTo[k - 1] + landOn(px[k - 1], py[k - 1], px[k], py[k]);
  /** for the piece from way point i0 on: a run from its point i to j (bent at bx, by) crosses no more land than the way */
  const keeps = (i0: number) => (i: number, j: number, bx: number, by: number) => {
    const a = i0 + i, b = i0 + j, land = bx === bx ? landOn(px[a], py[a], bx, by) + landOn(bx, by, px[b], py[b]) : landOn(px[a], py[a], px[b], py[b]);
    return land <= landTo[b] - landTo[a] + G.cell;
  };
  return { x: Float64Array.from(px), y: Float64Array.from(py), cuts, at, keeps };
}

/** where along polyline `p` (x, y pairs) point (x, y) is nearest, searching segments from `from`: segment and share */
function nearestOn(p: ArrayLike<number>, x: number, y: number, from: number): [number, number, number] {
  let best = Infinity, bi = from, bt = 0;
  for (let i = from; i + 3 < p.length; i += 2) {
    const ax = p[i], ay = p[i + 1], dx = p[i + 2] - ax, dy = p[i + 3] - ay, l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    const d = (ax + dx * t - x) ** 2 + (ay + dy * t - y) ** 2;
    if (d < best) (best = d), (bi = i), (bt = t);
  }
  return [bi, bt, best];
}

/** The lines `ins` drawn on a layout: `map` the carried map (pitches per graph metre place), `K` layout units a pitch. */
export function thinLines(ins: readonly ThinIn[], map: CarryMap, K: number, layout: DiagramLayout, d: Drawing): ThinDrawn[] {
  const pillsOf = new Map<number, number[]>();
  d.pillOf.forEach((k, i) => { let v = pillsOf.get(k); if (!v) pillsOf.set(k, (v = [])); v.push(i); });
  /** the point of node k's pills nearest (x, y), else its place */
  const atNode = (k: number, x: number, y: number): [number, number] => {
    let best = Infinity, bx = layout.x[k], by = layout.y[k];
    for (const i of pillsOf.get(k) ?? []) {
      const p = d.pills[i], dx = p[2] - p[0], dy = p[3] - p[1], l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - p[0]) * dx + (y - p[1]) * dy) / l2)) : 0;
      const qx = p[0] + dx * t, qy = p[1] + dy * t, e = (qx - x) ** 2 + (qy - y) ** 2;
      if (e < best) (best = e), (bx = qx), (by = qy);
    }
    return [bx, by];
  };
  const pitch = d.pitch, out: ThinDrawn[] = [];
  for (const L of ins) {
    const route = L.route.length >= 4 ? L.route : L.stops;
    const ns = L.stops.length / 2;
    if (route.length < 4) continue;
    // the stops put into the route as vertices, in run order (each found from the last one's segment on)
    const seg = new Int32Array(ns), share = new Float64Array(ns);
    for (let s = 0, from = 0; s < ns; s++) {
      const [i, t] = nearestOn(route, L.stops[2 * s], L.stops[2 * s + 1], from);
      (seg[s] = i), (share[s] = t), (from = i);
    }
    // chunks between stops, each kept to ROUTE_TOL_M; vertex list with the stops' indices
    const vx: number[] = [], stopV = new Int32Array(ns);
    let chunk: number[] = [route[0], route[1]];
    const flush = (end: boolean) => {
      const kept = simplifyLine(chunk, ROUTE_TOL_M);
      for (let q = vx.length ? 2 : 0; q < kept.length; q += 2) vx.push(kept[q], kept[q + 1]);
      if (!end) chunk = [kept[kept.length - 2], kept[kept.length - 1]];
    };
    let s = 0;
    for (let i = 0; i + 3 < route.length; i += 2) {
      for (; s < ns && seg[s] === i; s++) {
        const x = route[i] + (route[i + 2] - route[i]) * share[s], y = route[i + 1] + (route[i + 3] - route[i + 1]) * share[s];
        const last = chunk.length - 2;
        if (Math.hypot(chunk[last] - x, chunk[last + 1] - y) > 1e-6) chunk.push(x, y);
        flush(false);
        stopV[s] = vx.length / 2 - 1;
      }
      chunk.push(route[i + 2], route[i + 3]);
    }
    flush(true);
    for (; s < ns; s++) stopV[s] = vx.length / 2 - 1;
    // carried, then moved so each stop meets its place: the moves between two stops blended by arc
    const m = vx.length / 2, X = new Float64Array(m), Y = new Float64Array(m), arc = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const [a, b] = map.at(vx[2 * i], vx[2 * i + 1]);
      (X[i] = a * K), (Y[i] = b * K);
      if (i) arc[i] = arc[i - 1] + Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]);
    }
    // each stop meets its own carried place (a ferry's route ends at its berth, the port is on land), a station stop its pill
    /** per stop vertex: the vertex, its move, 1 at a station drawn here */
    const anchors: [number, number, number, number][] = [];
    for (let q = 0; q < ns; q++) {
      const k = L.node[q], v = stopV[q];
      const [tx, ty] = k >= 0 ? atNode(k, X[v], Y[v]) : map.at(L.stops[2 * q], L.stops[2 * q + 1]).map((c) => c * K);
      if (anchors.length && anchors[anchors.length - 1][0] === v) {
        if (k >= 0) anchors[anchors.length - 1] = [v, tx - X[v], ty - Y[v], 1];
        continue;
      }
      anchors.push([v, tx - X[v], ty - Y[v], k >= 0 ? 1 : 0]);
    }
    if (anchors.length) {
      let a = 0;
      for (let i = 0; i < m; i++) {
        while (a + 1 < anchors.length && anchors[a + 1][0] <= i) a++;
        const A = anchors[a], B = anchors[a + 1];
        let dx = A[1], dy = A[2];
        if (i > A[0] && B) {
          const t = (arc[i] - arc[A[0]]) / Math.max(1e-9, arc[B[0]] - arc[A[0]]);
          (dx = A[1] + (B[1] - A[1]) * t), (dy = A[2] + (B[2] - A[2]) * t);
        }
        (X[i] += dx), (Y[i] += dy);
      }
    }
    // T-3824: a ferry over the drawn water, its ports on the drawn coast
    const sea = L.ferry && d.water && anchors.length > 1 ? overWater(d.water, X, Y, anchors, pitch) : null;
    const PX = sea ? sea.x : X, PY = sea ? sea.y : Y, pm = PX.length;
    // fitted piece by piece between the station stops, each piece's ends kept
    const cuts = sea ? sea.cuts : [0, ...anchors.filter((a) => a[3]).map((a) => a[0]).filter((v) => v > 0 && v < m - 1), m - 1];
    const fit: number[] = [], tol = (sea ? FIT_TOL_WATER : L.ferry ? FIT_TOL_FERRY : FIT_TOL) * pitch;
    for (let c = 0; c + 1 < cuts.length; c++) {
      const i0 = cuts[c], i1 = cuts[c + 1];
      if (i1 <= i0 || i1 >= pm) continue;
      const px = Array.from(PX.subarray(i0, i1 + 1)), py = Array.from(PY.subarray(i0, i1 + 1));
      const f = fitOcti(px, py, px.map(() => tol), sea ? sea.keeps(i0) : undefined);
      for (let q = fit.length ? 2 : 0; q < f.length; q++) fit.push(f[q]);
    }
    if (fit.length < 4) continue;
    if (L.stubA || L.stubB) {
      const V: V[] = [];
      for (let q = 0; q + 1 < fit.length; q += 2) V.push([fit[q], fit[q + 1], 0, 0, -1]);
      if (L.stubA) { const s = stubEnd(V, 0, L.stubA, STUB * pitch); fit.unshift(s[0], s[1]); }
      if (L.stubB) { const s = stubEnd(V, 1, L.stubB, STUB * pitch); fit.push(s[0], s[1]); }
    }
    const ribbon = Float64Array.from(fit), cmd: number[] = [];
    rounded(ribbon, false, CORNER * pitch, cmd);
    // the stops not at a station drawn here: on the fitted line, nearest their carried place
    const stops: number[] = [], drawn = new Set(anchors.filter((a) => a[3]).map((a) => a[0]));
    for (let q = 0; q < ns; q++) {
      const v = stopV[q];
      if (L.node[q] >= 0 || drawn.has(v)) continue;
      // a port moved onto the coast sits where the way over the water starts
      const at = sea?.at.get(v), [i, t] = nearestOn(ribbon, at ? at[0] : X[v], at ? at[1] : Y[v], 0);
      stops.push(ribbon[i] + (ribbon[i + 2] - ribbon[i]) * t, ribbon[i + 1] + (ribbon[i + 3] - ribbon[i + 1]) * t);
    }
    out.push({ id: L.id, colour: L.colour, ferry: L.ferry, cmds: Float64Array.from(cmd), ribbon, stops: Float64Array.from(stops) });
  }
  return out;
}
