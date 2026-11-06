// The rubber sheet's map, before its coasts are traced (sheetLines.ts `sheetMap`; T-2580; docs/ui-diagram.md "The
// rubber sheet"). Narrow rivers a hop crosses are drawn as bands at least `riverMin` pitches wide, and every thin
// piece of water or land (a river, a strait, a creek, a spit, a pier) is drawn again as a band: one width along a middle
// fitted with runs at 0, 45 and 90 degrees. Two shores fitted apart can only follow a thin piece's own wiggles, so its
// edges came out as stair steps (Anita, 2026-11-06: "these peninsulas are very wiggly"); a band's edges are straight, and
// the coast's fit then follows them. Pure: no DOM, no fetch.

import { fitOcti, rounded, segD } from "./borders";
import { chamfer, type RealWater } from "./waterReal";
import type { DiagramGraph } from "./graph";
import type { DiagramLayout } from "./layout";

/** the drawn grid: per cell its label (0 water, OUT off the carried map, NOZ land in no zone, else a zone), `cell`
 * pitches a side from (x0, y0); `pill` cells are a station's own land, `band` cells drawn as a band */
export interface MapGrid { D: Uint8Array | Uint16Array; w: number; h: number; cell: number; x0: number; y0: number; OUT: number; NOZ: number; pill: Uint8Array; band: Uint8Array }

export interface ThinOptions {
  /** water or land narrower than this, pitches, is drawn as a band */
  thinW: number;
  /** a band's middle may move this far at most from the piece's own, pitches */
  tolMax: number;
  /** the least width of a band, pitches; of a river a hop crosses */
  minW: number;
  riverMin: number;
  /** the coast's and a band's least tolerance, cells (a straight edge drawn on the grid steps a cell) */
  tolCell: number;
}

export const THIN: ThinOptions = { thinW: 8, tolMax: 6, minW: 1.5, riverMin: 3.5, tolCell: 0.85 };

export interface ThinStats { pieces: number; bands: number; skipped: number; ms: number }

/** water (or land) as a mask of the grid */
const classMask = (G: MapGrid, water: boolean) => {
  const M = new Uint8Array(G.w * G.h);
  for (let c = 0; c < M.length; c++) M[c] = +((G.D[c] === 0 || G.D[c] === G.OUT) === water);
  return M;
};
const isZone = (G: MapGrid, v: number) => v > 0 && v < G.OUT && v !== G.NOZ;

/** One pass over the thin pieces of water (or land): each drawn again as bands, its middle straying at most 0.4 of the
 * way to the next piece of its kind (and on the water pass clear of the pills). */
export function thinPass(G: MapGrid, water: boolean, o: ThinOptions, stats: ThinStats): void {
  const t0 = performance.now();
  const { w: W, h: H, cell } = G, N = W * H, CAP = 60000;
  const M = classMask(G, water);
  const dIn = chamfer(M, W, H, false, CAP);
  const r3 = Math.round((3 * o.thinW) / 2 / cell);
  const core = new Uint8Array(N);
  for (let c = 0; c < N; c++) core[c] = +(M[c] === 1 && dIn[c] > r3);
  const dCore = chamfer(core, W, H, true, CAP);
  const thin = new Uint8Array(N);
  for (let c = 0; c < N; c++) thin[c] = +(M[c] === 1 && dCore[c] > r3);
  const dPill = water ? chamfer(G.pill, W, H, true, CAP) : null;
  const can = (c: number) => G.D[c] !== G.OUT && !G.pill[c];
  const comp = new Int32Array(N).fill(-1);
  const bands: { segs: number[]; w: number; label: number; me: number; ox: number; oy: number; ow: number; oh: number; dOther: Uint16Array }[] = [];
  const cleared: number[] = [];
  const stack: number[] = [];
  let id = 0;
  for (let s0 = 0; s0 < N; s0++) {
    if (!thin[s0] || comp[s0] >= 0) continue;
    // the piece: thin cells joined by sides or corners
    const cells: number[] = [];
    let i0 = W, i1 = -1, j0 = H, j1 = -1, out = false, onPill = false;
    stack.length = 0;
    stack.push(s0), (comp[s0] = id);
    while (stack.length) {
      const c = stack.pop()!, i = c % W, j = (c - i) / W;
      cells.push(c);
      (i0 = Math.min(i0, i)), (i1 = Math.max(i1, i)), (j0 = Math.min(j0, j)), (j1 = Math.max(j1, j));
      if (G.D[c] === G.OUT) out = true;
      if (G.pill[c]) onPill = true;
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        const ii = i + a, jj = j + b;
        if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
        const u = jj * W + ii;
        if (thin[u] && comp[u] < 0) (comp[u] = id), stack.push(u);
      }
    }
    const me = id++;
    stats.pieces++;
    // a piece off the map's edge, and (on the land pass) one with a station on it, stay as they are
    if (out || onPill) {
      stats.skipped++;
      if (onPill) for (const c of cells) G.band[c] = 1;
      continue;
    }
    // a piece under 3 pitches every way (a corner the opening cut off) stays as it is
    if (Math.max(i1 - i0 + 1, j1 - j0 + 1) * cell < 3) { stats.skipped++; continue; }
    // the piece's middle: its cells thinned to lines one cell wide
    const bw = i1 - i0 + 3, bh = j1 - j0 + 3, img = new Uint8Array(bw * bh);
    for (const c of cells) { const i = c % W, j = (c - i) / W; img[(j - j0 + 1) * bw + (i - i0 + 1)] = 1; }
    skeleton(img, bw, bh);
    const chains = chainsOf(img, bw, bh);
    const toG = (p: number) => { const i = (p % bw) - 1 + i0, j = Math.floor(p / bw) - 1 + j0; return j * W + i; };
    const wOf = (c: number) => Math.max(0.5, (2 * dIn[c]) / 3 - 0.5) * cell;
    let len = 0;
    const allW: number[] = [];
    for (const ch of chains) { len += ch.px.length; for (const p of ch.px) allW.push(wOf(toG(p))); }
    allW.sort((a, b) => a - b);
    const wMed = allW.length ? allW[allW.length >> 1] : 0;
    // pieces short against their width (a corner the opening cut off) stay as they are
    if (len * cell < Math.max(3, 1.5 * wMed)) { stats.skipped++; continue; }
    // what else of this kind is near: distances from it, on the piece's box and a margin
    const pad = Math.ceil((o.tolMax + o.thinW) / cell) + 2;
    const ox = Math.max(0, i0 - pad), oy = Math.max(0, j0 - pad), ow = Math.min(W - 1, i1 + pad) - ox + 1, oh = Math.min(H - 1, j1 + pad) - oy + 1;
    const other = new Uint8Array(ow * oh);
    for (let j = 0; j < oh; j++) for (let i = 0; i < ow; i++) { const c = (j + oy) * W + i + ox; other[j * ow + i] = +(M[c] === 1 && comp[c] !== me); }
    const dOther = chamfer(other, ow, oh, true, CAP);
    // its label, where land: the zone most of its cells are in
    let label = 0;
    if (!water) {
      const cnt = new Map<number, number>();
      for (const c of cells) if (isZone(G, G.D[c])) cnt.set(G.D[c], (cnt.get(G.D[c]) ?? 0) + 1);
      let best = -1;
      label = G.NOZ;
      for (const [v, k] of cnt) if (k > best) (best = k), (label = v);
    }
    // spurs (a chain from a free end to a junction, short against the width) go
    const near = (p: number, r: number) => {
      const c = toG(p), i = c % W, j = (c - i) / W;
      for (let b = -r; b <= r; b++) for (let a = -r; a <= r; a++) { const ii = i + a, jj = j + b; if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue; const u = jj * W + ii; if (M[u] === 1 && comp[u] !== me) return true; }
      return false;
    };
    let drawn = 0;
    // where a band meets the body it leaves from: the piece is kept there as it was, so no notch opens at the joint
    const joints: number[] = [];
    for (const ch of chains) {
      const n = ch.px.length;
      const endA = ch.a === 1 ? (near(ch.px[0], 2) ? "body" : "free") : ch.a === 0 ? "loop" : "join";
      const endB = ch.b === 1 ? (near(ch.px[n - 1], 2) ? "body" : "free") : ch.b === 0 ? "loop" : "join";
      const ws = ch.px.map((p) => wOf(toG(p))).sort((a, b) => a - b);
      const wc = Math.round(Math.max(o.minW, ws[ws.length >> 1]) * 4) / 4;
      if (((endA === "free" && endB === "join") || (endB === "free" && endA === "join")) && n * cell < Math.max(3, 2 * wc)) continue;
      const px: number[] = [], py: number[] = [], tol: number[] = [];
      for (const p of ch.px) {
        const c = toG(p), i = c % W, j = (c - i) / W;
        px.push(G.x0 + (i + 0.5) * cell), py.push(G.y0 + (j + 0.5) * cell);
        let t = Math.min(o.tolMax, Math.max(o.tolCell * cell, 0.4 * ((dOther[(j - oy) * ow + i - ox] / 3) * cell - wc / 2)));
        if (dPill) t = Math.min(t, Math.max(0.3 * cell, (dPill[c] / 3) * cell - wc / 2 - 0.5));
        tol.push(t);
      }
      if (ch.a === 0) (px.push(px[0]), py.push(py[0]), tol.push(tol[0]));
      const fit = px.length > 1 ? fitOcti(px, py, tol) : Float64Array.of(px[0], py[0], px[0] + 1e-6, py[0]);
      const ext = (e: string) => (e === "body" ? wc / 2 + 2 * cell : e === "join" ? wc / 2 + cell : e === "loop" ? wc / 2 : wc / 2);
      bands.push({ segs: [...fit, ext(endA), ext(endB)], w: wc, label, me, ox, oy, ow, oh, dOther });
      if (endA === "body") joints.push(px[0], py[0], wc / 2 + 2 * cell);
      if (endB === "body") joints.push(px[px.length - 1], py[py.length - 1], wc / 2 + 2 * cell);
      drawn++;
    }
    if (!drawn && chains.length) { stats.skipped++; continue; }
    stats.bands += drawn;
    for (const c of cells) {
      if (!can(c)) continue;
      const x = G.x0 + ((c % W) + 0.5) * cell, y = G.y0 + (Math.floor(c / W) + 0.5) * cell;
      let atJoint = false;
      for (let q = 0; q < joints.length && !atJoint; q += 3) atJoint = Math.hypot(x - joints[q], y - joints[q + 1]) <= joints[q + 2];
      if (!atJoint) cleared.push(c);
    }
  }
  // cleared first, then every band drawn
  const fresh = new Uint8Array(N);
  for (const c of cleared) { G.D[c] = water ? G.NOZ : 0; if (water) fresh[c] = 1; }
  for (const b of bands) {
    const v = b.segs, m = (v.length - 2) / 2, eA = v[v.length - 2], eB = v[v.length - 1], hw = b.w / 2;
    for (let k = 0; k + 1 < m; k++) {
      const ax = v[k * 2], ay = v[k * 2 + 1], bx = v[k * 2 + 2], by = v[k * 2 + 3], l = Math.hypot(bx - ax, by - ay);
      const ux = l > 1e-9 ? (bx - ax) / l : 1, uy = l > 1e-9 ? (by - ay) / l : 0;
      const s0 = -(k === 0 ? eA : hw), s1 = l + (k + 2 === m ? eB : hw);
      const xs = [ax + ux * s0 - uy * hw, ax + ux * s0 + uy * hw, ax + ux * s1 - uy * hw, ax + ux * s1 + uy * hw], ys = [ay + uy * s0 + ux * hw, ay + uy * s0 - ux * hw, ay + uy * s1 + ux * hw, ay + uy * s1 - ux * hw];
      const ci0 = Math.max(0, Math.floor((Math.min(...xs) - G.x0) / cell)), ci1 = Math.min(W - 1, Math.floor((Math.max(...xs) - G.x0) / cell));
      const cj0 = Math.max(0, Math.floor((Math.min(...ys) - G.y0) / cell)), cj1 = Math.min(H - 1, Math.floor((Math.max(...ys) - G.y0) / cell));
      for (let j = cj0; j <= cj1; j++) for (let i = ci0; i <= ci1; i++) {
        const x = G.x0 + (i + 0.5) * cell - ax, y = G.y0 + (j + 0.5) * cell - ay, al = x * ux + y * uy, ac = -x * uy + y * ux;
        if (al < s0 || al > s1 || Math.abs(ac) > hw) continue;
        const c = j * W + i;
        if (!can(c)) continue;
        // never nearer another piece of its kind than GUARD, outside its own piece: so no strait closes, no island
        // joins another, no isthmus is cut
        if (comp[c] !== b.me) {
          const li = i - b.ox, lj = j - b.oy;
          if (li >= 0 && lj >= 0 && li < b.ow && lj < b.oh && b.dOther[lj * b.ow + li] < GUARD) continue;
        }
        G.D[c] = water ? 0 : b.label;
        fresh[c] = 0;
        G.band[c] = 1;
      }
    }
  }
  // land given back by the water's bands takes the zone beside it
  if (water) fillZones(G, fresh);
  stats.ms += performance.now() - t0;
}

/** Nibs and notches a cell or two across, left where bands meet what they leave from and on the grid's own edges, go:
 * land opened by a 3 by 3 square (what does not hold one goes to water), then water likewise; never a pill's or a
 * band's cells, and only a cell whose eight neighbours hold one run of each kind, so no shape is cut in two or joined to
 * another. A straight edge at 0, 45 or 90 degrees is kept as it is. */
export function smoothEdges(G: MapGrid) {
  const { w: W, h: H } = G, N = W * H;
  // a 3 by 3 square's min (or max) as a row pass then a column pass
  const box3 = (A: Uint8Array, max: boolean) => {
    const R = new Uint8Array(N), O = new Uint8Array(N);
    for (let j = 0; j < H; j++) for (let i = 0, c = j * W; i < W; i++, c++) {
      const l = i > 0 ? A[c - 1] : A[c], r = i + 1 < W ? A[c + 1] : A[c];
      R[c] = max ? l | A[c] | r : l & A[c] & r;
    }
    for (let c = 0; c < N; c++) {
      const u = c >= W ? R[c - W] : R[c], d = c + W < N ? R[c + W] : R[c];
      O[c] = max ? u | R[c] | d : u & R[c] & d;
    }
    return O;
  };
  for (const water of [false, true]) {
    const M = classMask(G, water), held = box3(box3(M, false), true);
    const fresh = new Uint8Array(N), wet = (c: number) => G.D[c] === 0 || G.D[c] === G.OUT, RING = [-W - 1, -W, -W + 1, 1, W + 1, W, W - 1, -1];
    for (let j = 1; j + 1 < H; j++) for (let i = 1; i + 1 < W; i++) {
      const c = j * W + i;
      if (!M[c] || held[c] || G.D[c] === G.OUT || G.pill[c] || G.band[c]) continue;
      // only where it changes no shape's connections: round its eight neighbours, one run of each kind
      let runs = 0;
      for (let q = 0; q < 8; q++) if (wet(c + RING[q]) !== wet(c + RING[(q + 1) % 8])) runs++;
      if (runs !== 2) continue;
      if (water) (G.D[c] = G.NOZ), (fresh[c] = 1);
      else G.D[c] = 0;
    }
    if (water) fillZones(G, fresh);
  }
}

/** cells marked `fresh` (land with no zone yet) take the zone of the nearest zoned land, by steps of a side */
function fillZones(G: MapGrid, fresh: Uint8Array) {
  const { w: W, h: H } = G, q: number[] = [];
  for (let c = 0; c < W * H; c++) if (fresh[c] && G.D[c] === G.NOZ) {
    const i = c % W;
    for (const u of [i > 0 ? c - 1 : -1, i < W - 1 ? c + 1 : -1, c - W, c + W]) if (u >= 0 && u < W * H && !fresh[u] && isZone(G, G.D[u])) { q.push(c); break; }
  }
  for (let k = 0; k < q.length; k++) {
    const c = q[k];
    if (!fresh[c]) continue;
    const i = c % W;
    let z = G.NOZ;
    for (const u of [i > 0 ? c - 1 : -1, i < W - 1 ? c + 1 : -1, c - W, c + W]) if (u >= 0 && u < W * H && !fresh[u] && isZone(G, G.D[u])) { z = G.D[u]; break; }
    if (z === G.NOZ) continue;
    (G.D[c] = z), (fresh[c] = 0);
    for (const u of [i > 0 ? c - 1 : -1, i < W - 1 ? c + 1 : -1, c - W, c + W]) if (u >= 0 && u < W * H && fresh[u] && G.D[u] === G.NOZ) q.push(u);
  }
}

/** Zhang and Suen's thinning (1984) of a 0/1 image with a blank border, in place */
function skeleton(img: Uint8Array, bw: number, bh: number) {
  const del: number[] = [];
  for (let changed = true; changed; ) {
    changed = false;
    for (let step = 0; step < 2; step++) {
      del.length = 0;
      for (let j = 1; j < bh - 1; j++) for (let i = 1; i < bw - 1; i++) {
        const c = j * bw + i;
        if (!img[c]) continue;
        const p2 = img[c - bw], p3 = img[c - bw + 1], p4 = img[c + 1], p5 = img[c + bw + 1], p6 = img[c + bw], p7 = img[c + bw - 1], p8 = img[c - 1], p9 = img[c - bw - 1];
        const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        if (B < 2 || B > 6) continue;
        const A = +(!p2 && !!p3) + +(!p3 && !!p4) + +(!p4 && !!p5) + +(!p5 && !!p6) + +(!p6 && !!p7) + +(!p7 && !!p8) + +(!p8 && !!p9) + +(!p9 && !!p2);
        if (A !== 1) continue;
        if (step === 0 ? (p2 && p4 && p6) || (p4 && p6 && p8) : (p2 && p4 && p8) || (p2 && p6 && p8)) continue;
        del.push(c);
      }
      for (const c of del) img[c] = 0;
      if (del.length) changed = true;
    }
  }
}

/** a skeleton's lines between its ends and junctions: per chain its pixels and what each end is (1 an end, 3 a
 * junction, 0 a closed loop) */
function chainsOf(img: Uint8Array, bw: number, bh: number): { px: number[]; a: number; b: number }[] {
  // neighbours: by sides, and by corners where neither side between is set (so a step is not a junction)
  const nb = (c: number) => {
    const out: number[] = [];
    const up = img[c - bw], dn = img[c + bw], lf = img[c - 1], rt = img[c + 1];
    if (up) out.push(c - bw);
    if (dn) out.push(c + bw);
    if (lf) out.push(c - 1);
    if (rt) out.push(c + 1);
    if (img[c - bw - 1] && !up && !lf) out.push(c - bw - 1);
    if (img[c - bw + 1] && !up && !rt) out.push(c - bw + 1);
    if (img[c + bw - 1] && !dn && !lf) out.push(c + bw - 1);
    if (img[c + bw + 1] && !dn && !rt) out.push(c + bw + 1);
    return out;
  };
  const deg = new Map<number, number[]>();
  for (let c = 0; c < bw * bh; c++) if (img[c]) deg.set(c, nb(c));
  const used = new Set<string>(), ek = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
  const chains: { px: number[]; a: number; b: number }[] = [];
  const walk = (s: number, nx: number) => {
    const px = [s];
    let a = s, b = nx;
    for (;;) {
      used.add(ek(a, b)), px.push(b);
      const ns = deg.get(b)!;
      if (ns.length !== 2 || b === s) break;
      const c = ns[0] === a ? ns[1] : ns[0];
      if (used.has(ek(b, c))) break;
      (a = b), (b = c);
    }
    return px;
  };
  const kind = (c: number) => { const k = deg.get(c)!.length; return k === 2 ? 2 : k <= 1 ? 1 : 3; };
  for (const [c, ns] of deg) {
    if (ns.length === 2) continue;
    if (!ns.length) { chains.push({ px: [c], a: 1, b: 1 }); continue; }
    for (const u of ns) if (!used.has(ek(c, u))) { const px = walk(c, u); chains.push({ px, a: kind(c), b: kind(px[px.length - 1]) }); }
  }
  for (const [c, ns] of deg) for (const u of ns) if (!used.has(ek(c, u))) { const px = walk(c, u); chains.push({ px: px[0] === px[px.length - 1] ? px.slice(0, -1) : px, a: 0, b: 0 }); }
  return chains;
}

/** the carried map (sheetLines.ts `Carry`'s lattice) at a place on the ground, pitches */
export interface CarryMap { at(x: number, y: number): [number, number] }

/** narrow rivers as bands: per width (pitches) its middles as path commands and its ends' stubs (layout units), and
 * each band as fitted (pitches; checks) */
export interface RiverBands { cmds: Map<number, number[]>; ends: Map<number, number[]>; fits: { fit: Float64Array; w: number; arc: number }[]; arcs: number; crossings: number }

/** Narrow rivers a hop crosses (the river finder's, waterReal.ts, every arc of such a river), drawn as bands over the
 * map: each river's middle carried into the drawing, each crossing put on its drawn hop at the hop's share and the river
 * round it moved with it (out to CORR_M on the ground); `riverMin` pitches wide, pushed off stations and narrower where
 * they leave too little room (to `minW`); drawn only where the grid's water there is narrower than the band, and
 * stopping either side of a station it cannot pass (as at a bridge); its middle fitted with runs at 0, 45 and 90
 * degrees, straying up to `tolMax` but keeping clear of the stations. The grid's own water under and beside a band, where
 * narrower than it, goes to land, so the band's edges are the only ones there. */
export function riverBands(G: MapGrid, g: DiagramGraph, layout: DiagramLayout, k: number, real: RealWater, map: CarryMap, o: ThinOptions): RiverBands {
  const { w: W, h: H, cell } = G, n = g.key.length;
  const SX = Float64Array.from(layout.x, (v) => v / k), SY = Float64Array.from(layout.y, (v) => v / k);
  // each hop as drawn, from its piece, pitches
  const hopPoly = new Map<number, { pts: number[]; from: number }>();
  for (const pc of layout.pieces) for (let q = 0; q + 1 < pc.nodes.length; q++) {
    const a = pc.nodes[q], b = pc.nodes[q + 1], pts: number[] = [];
    for (let v = pc.at[q]; v <= pc.at[q + 1]; v++) pts.push(pc.verts[v * 2] / k, pc.verts[v * 2 + 1] / k);
    hopPoly.set(Math.min(a, b) * n + Math.max(a, b), { pts, from: a });
  }
  const along = (p: ArrayLike<number>, s: number): [number, number] => {
    for (let q = 0; q + 3 < p.length; q += 2) {
      const l = Math.hypot(p[q + 2] - p[q], p[q + 3] - p[q + 1]);
      if (s <= l || q + 4 >= p.length) { const f = l ? Math.min(1, s / l) : 0; return [p[q] + f * (p[q + 2] - p[q]), p[q + 1] + f * (p[q + 3] - p[q + 1])]; }
      s -= l;
    }
    return [p[0], p[1]];
  };
  const lengthOf = (p: ArrayLike<number>) => { let t = 0; for (let q = 0; q + 3 < p.length; q += 2) t += Math.hypot(p[q + 2] - p[q], p[q + 3] - p[q + 1]); return t; };
  /** whether polylines a and b cross */
  const crosses = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    for (let i = 0; i + 3 < a.length; i += 2) for (let j = 0; j + 3 < b.length; j += 2) {
      const ax = a[i], ay = a[i + 1], bx = a[i + 2], by = a[i + 3], cx = b[j], cy = b[j + 1], dx = b[j + 2], dy = b[j + 3];
      const d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax), d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
      const d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx), d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
      if (d1 * d2 < 0 && d3 * d4 < 0) return true;
    }
    return false;
  };
  const crossed = new Set<number>();
  const corr = new Map<number, number[]>();
  let crossings = 0;
  for (const A of real.arcs) for (const c of A.cross) {
    crossed.add(A.river);
    const key = Math.min(g.ea[c.e], g.eb[c.e]) * n + Math.max(g.ea[c.e], g.eb[c.e]), hp = hopPoly.get(key);
    if (!hp) continue;
    crossings++;
    const [tx, ty] = along(hp.pts, (hp.from === g.ea[c.e] ? c.t : 1 - c.t) * lengthOf(hp.pts)), [rx, ry] = along(A.pts, c.s), [px, py] = map.at(rx, ry);
    let l = corr.get(A.river);
    if (!l) corr.set(A.river, (l = []));
    l.push(rx, ry, tx - px, ty - py);
  }
  const corrAt = (river: number, x: number, y: number): [number, number] => {
    let sx = 0, sy = 0, sw = 1 / (CORR_M * CORR_M);
    const l = corr.get(river) ?? [];
    for (let q = 0; q < l.length; q += 4) {
      const d2 = (l[q] - x) ** 2 + (l[q + 1] - y) ** 2;
      if (d2 > CORR_M * CORR_M) continue;
      const wt = 1 / (d2 + 1) - 1 / (CORR_M * CORR_M);
      (sx += wt * l[q + 2]), (sy += wt * l[q + 3]), (sw += wt);
    }
    return [sx / sw, sy / sw];
  };
  // stations by 8-pitch squares
  const SH = new Map<number, number[]>(), sk = (i: number, j: number) => i * 100003 + j;
  for (let s = 0; s < n; s++) { const q = sk(Math.floor(SX[s] / 8), Math.floor(SY[s] / 8)); let l = SH.get(q); if (!l) SH.set(q, (l = [])); l.push(s); }
  const stNear = (x: number, y: number, r: number) => {
    const out: number[] = [], i = Math.floor(x / 8), j = Math.floor(y / 8), m = Math.ceil(r / 8);
    for (let a = i - m; a <= i + m; a++) for (let b = j - m; b <= j + m; b++) for (const s of SH.get(sk(a, b)) ?? []) if (Math.hypot(SX[s] - x, SY[s] - y) < r) out.push(s);
    return out;
  };
  const nearestSt = (x: number, y: number, r: number) => { let best = r; for (const s of stNear(x, y, r)) best = Math.min(best, Math.hypot(SX[s] - x, SY[s] - y)); return best; };
  const wetAt = (x: number, y: number) => { const i = Math.floor((x - G.x0) / cell), j = Math.floor((y - G.y0) / cell); return i < 0 || j < 0 || i >= W || j >= H || G.D[j * W + i] === 0 || G.D[j * W + i] === G.OUT; };
  const PR = Math.max(1.5, 0.5 + cell);
  const out: RiverBands = { cmds: new Map(), ends: new Map(), fits: [], arcs: 0, crossings }, raw: { fit: Float64Array; w: number }[] = [];
  real.arcs.forEach((A, ai) => {
    if (!crossed.has(A.river) || A.pts.length < 4) return;
    out.arcs++;
    // its middle carried (RX, RY: where the grid has the river) and moved onto the crossings (X, Y)
    const X: number[] = [], Y: number[] = [], RX: number[] = [], RY: number[] = [];
    for (let q = 0; q + 3 < A.pts.length; q += 2) {
      const l = Math.hypot(A.pts[q + 2] - A.pts[q], A.pts[q + 3] - A.pts[q + 1]), m = Math.max(1, Math.ceil(l / DENSE_M));
      for (let t = q ? 1 : 0; t <= m; t++) {
        const x = A.pts[q] + ((A.pts[q + 2] - A.pts[q]) * t) / m, y = A.pts[q + 1] + ((A.pts[q + 3] - A.pts[q + 1]) * t) / m;
        const [px, py] = map.at(x, y), [cx, cy] = corrAt(A.river, x, y);
        X.push(px + cx), Y.push(py + cy), RX.push(px), RY.push(py);
      }
    }
    // where the river is wider than the band the grid already holds it at its width (the ground carried), so the band is
    // the least width; pushed off the stations nearer than half its width and a pill, narrower where they leave too
    // little room
    let wr = o.riverMin;
    for (let tries = 0; tries < 8; tries++) {
      const R = wr / 2 + PR;
      let worst = Infinity;
      for (let pass = 0; pass < 6; pass++) for (let q = 0; q < X.length; q++) for (const s of stNear(X[q], Y[q], R)) {
        const dx = X[q] - SX[s], dy = Y[q] - SY[s], d = Math.hypot(dx, dy);
        const ux = d > 1e-6 ? dx / d : 1, uy = d > 1e-6 ? dy / d : 0;
        (X[q] = SX[s] + ux * R), (Y[q] = SY[s] + uy * R);
      }
      for (let q = 0; q < X.length; q++) for (const s of stNear(X[q], Y[q], R)) worst = Math.min(worst, Math.hypot(X[q] - SX[s], Y[q] - SY[s]));
      if (worst >= R - 0.3 || wr <= o.minW) break;
      wr = Math.max(o.minW, wr - 0.5);
    }
    const hw = wr / 2, R = hw + PR - 0.3;
    // covered: the grid's water is there at least as wide as the band (every cell within 0.45 of its width)
    const rc = 0.45 * wr;
    const covered = (x: number, y: number) => {
      for (let b = -rc; b <= rc + 1e-9; b += Math.min(rc, 0.5 * cell) || 1) for (let a = -rc; a <= rc + 1e-9; a += Math.min(rc, 0.5 * cell) || 1) if (a * a + b * b <= rc * rc && !wetAt(x + a, y + b)) return false;
      return true;
    };
    // pieces: where not covered (a point of overlap each end, so the band meets the water), split at stations
    const pieces: number[][] = [];
    let cur: number[] = [];
    const flush = () => { if (cur.length > 1) pieces.push(cur); cur = []; };
    for (let q = 0; q < X.length; q++) {
      if (stNear(X[q], Y[q], R).length) { flush(); continue; }
      if (!covered(X[q], Y[q])) { if (!cur.length && q > 0 && !stNear(X[q - 1], Y[q - 1], R).length) cur.push(q - 1); cur.push(q); }
      else if (cur.length) (cur.push(q), flush());
    }
    flush();
    for (let pi = 0; pi < pieces.length && pi < 400; pi++) {
      const pc = pieces[pi], XX = pc.map((q) => X[q]), YY = pc.map((q) => Y[q]);
      if (lengthOf(XX.flatMap((x, q) => [x, YY[q]])) < 2) continue;
      // straying up to tolMax, less near a station but at least BAND_TOL (a band the fit takes over a station is split
      // round it below)
      const tol = XX.map((x, q) => Math.max(BAND_TOL, Math.min(o.tolMax, nearestSt(x, YY[q], o.tolMax + R + 1) - R)));
      const fit = fitOcti(XX, YY, tol);
      // a fitted band still over a station splits the piece round it
      const hit = new Set<number>();
      for (let q = 0; q + 3 < fit.length; q += 2) {
        const l = Math.hypot(fit[q + 2] - fit[q], fit[q + 3] - fit[q + 1]), m = Math.max(1, Math.ceil(l / 4));
        for (let t = 0; t <= m; t++) for (const s of stNear(fit[q] + ((fit[q + 2] - fit[q]) * t) / m, fit[q + 1] + ((fit[q + 3] - fit[q + 1]) * t) / m, hw + 4)) if (segD(SX[s], SY[s], fit[q], fit[q + 1], fit[q + 2], fit[q + 3]) < hw + 0.3) hit.add(s);
      }
      if (hit.size) {
        let part: number[] = [];
        for (const q of pc) {
          if ([...hit].some((s) => Math.hypot(X[q] - SX[s], Y[q] - SY[s]) < R + 1.5)) { if (part.length > 1) pieces.push(part); part = []; continue; }
          part.push(q);
        }
        if (part.length > 1 && part.length < pc.length) pieces.push(part);
        continue;
      }
      // a stub (shorter than twice its width) that crosses none of its arc's hops is not drawn
      if (lengthOf(fit) < 2 * wr && !A.cross.some((c) => { const hp = hopPoly.get(Math.min(g.ea[c.e], g.eb[c.e]) * n + Math.max(g.ea[c.e], g.eb[c.e])); return !!hp && crosses(fit, hp.pts); })) continue;
      out.fits.push({ fit, w: wr, arc: ai });
      raw.push({ fit: Float64Array.from(pc.flatMap((q) => [RX[q], RY[q]])), w: wr });
      const cmd = out.cmds.get(wr) ?? out.cmds.set(wr, []).get(wr)!, fk = Float64Array.from(fit, (v) => v * k);
      rounded(fk, false, BEND_R * k, cmd);
      // round ends: stubs a hundredth of a pitch long
      const e = out.ends.get(wr) ?? out.ends.set(wr, []).get(wr)!, m = fk.length;
      for (const [px, py, qx, qy] of [[fk[0], fk[1], fk[2], fk[3]], [fk[m - 2], fk[m - 1], fk[m - 4], fk[m - 3]]]) {
        const l = Math.hypot(qx - px, qy - py) || 1;
        e.push(0, px, py, 1, px + ((qx - px) / l) * 0.01 * k, py + ((qy - py) / l) * 0.01 * k);
      }
    }
  });
  // the grid's water within a band and a cell and a half of it, or of the river's own middle carried, where narrower
  // than the band (no disc as wide reaches it), goes to land
  if (out.fits.length) {
    const M = classMask(G, true), dIn = chamfer(M, W, H, false, 60000), fresh = new Uint8Array(W * H);
    const r3 = Math.floor((3 * o.riverMin) / 2 / cell), core = new Uint8Array(W * H);
    for (let c = 0; c < W * H; c++) core[c] = +(M[c] === 1 && dIn[c] > r3);
    const dCore = chamfer(core, W, H, true, 60000);
    for (const { fit, w } of [...out.fits, ...raw]) {
      const r = w / 2 + 1.5 * cell;
      for (let q = 0; q + 3 < fit.length; q += 2) {
        const ax = fit[q], ay = fit[q + 1], bx = fit[q + 2], by = fit[q + 3];
        const ci0 = Math.max(0, Math.floor((Math.min(ax, bx) - r - G.x0) / cell)), ci1 = Math.min(W - 1, Math.floor((Math.max(ax, bx) + r - G.x0) / cell));
        const cj0 = Math.max(0, Math.floor((Math.min(ay, by) - r - G.y0) / cell)), cj1 = Math.min(H - 1, Math.floor((Math.max(ay, by) + r - G.y0) / cell));
        for (let j = cj0; j <= cj1; j++) for (let i = ci0; i <= ci1; i++) {
          const c = j * W + i;
          if (G.D[c] !== 0 || dCore[c] <= r3) continue;
          if (segD(G.x0 + (i + 0.5) * cell, G.y0 + (j + 0.5) * cell, ax, ay, bx, by) > r) continue;
          (G.D[c] = G.NOZ), (fresh[c] = 1);
        }
      }
    }
    fillZones(G, fresh);
  }
  return out;
}

/** a band drawn outside its own piece keeps this far (chamfer units, 3 a cell) from another piece of its kind */
const GUARD = 5;
/** a river's course sampled this often on the ground, and how far round a crossing it moves with it, metres */
const DENSE_M = 60, CORR_M = 3000;
/** a band's corners rounded with this radius, pitches (the flag-off rivers', water.ts RADIUS) */
const BEND_R = 2.5;
/** a band's middle may always stray this far, pitches */
const BAND_TOL = 1.5;
