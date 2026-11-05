// City names on the network diagram zoomed far out (T-2438; docs/ui-diagram.md "City names"): one name per city
// block (blocks.ts), its biggest city zone's, put above, below or beside the block's box where it covers the least
// of the lines, or in its emptiest corner. Below CITY_SWITCH the city names are drawn and the station names not.
// Placed once per zoom level, as the station names are (names.ts); between levels a name only shrinks toward its
// anchor, which lies on the block's box edge, so it never creeps over the block.

import { BoxGrid } from "../map/labelLayout";
import type { Blocks } from "./blocks";
import type { Drawing } from "./draw";
import { pow } from "./fmath";
import { NAME_LINE } from "./names";

/** a zone as the page describes it: its shown name, the name's width in CSS px, its local trips a day (its size)
 * and whether it is a city zone */
export interface ZoneLabel { name: string; width: number; trips: number; city: boolean; towns?: Town[] }
/** a place of the zone's place list, for naming a block the zone's own name is taken from (T-2447): its name and
 * width, its place in the graph's metres, and whether it is a city or town (a village only where none is left) */
export interface Town { name: string; width: number; x: number; y: number; big: boolean }

/** the levels, a pitch in CSS px: 7 times STEP to the powers LO to HI (0.16 to 1.24 px); the station names' faint
 * levels start at the next (names.ts FAINT_LO), so below it the city names are drawn and the station names not */
const LEVEL_STEP = pow(2, 0.25), LEVEL_LO = -22, LEVEL_HI = -10;
export const CITY_SWITCH = 7 * pow(LEVEL_STEP, LEVEL_HI + 1);
/** px between the box and its name, and the halo kept clear of other names */
const GAP_PX = 4, SPACE_PX = 6;
/** each place's own cost, in px of line covered: above the box first, then below, beside, inside */
const PLACE_COST = { above: 0, aboveFlush: 4, below: 6, belowFlush: 10, side: 14, inside: 18 };

export interface CityNames {
  /** per block, its name ("" for none) and the name's width in CSS px */
  text: string[];
  width: Float32Array;
  fontPx: number;
  /** a pitch in CSS px each level is placed for, rising, all below CITY_SWITCH */
  levels: number[];
  /** ids of the zones with blocks that wait for the zone's place list to be named (T-2447) */
  want: string[];
  /** per level, per name placed: its block, its anchor (layout units) and where the anchor lies on the name's box
   * (hx, hy: 0 its left or bottom, 0.5 its middle, 1 its right or top) */
  at: { block: Int32Array; x: Float64Array; y: Float64Array; hx: Float32Array; hy: Float32Array }[];
  ms: number;
}

/** Each block's zone (an index into `zones`, -1 for none): the zone holding most of its stations, ties to a city
 * zone, then to more trips. `zoneOf` per node indexes `zones`. */
export function blockNames(blocks: Blocks, zoneOf: ArrayLike<number>, zones: (ZoneLabel | undefined)[]): number[] {
  const nb = blocks.list.length, counts: Map<number, number>[] = Array.from({ length: nb }, () => new Map());
  for (let i = 0; i < blocks.of.length; i++) {
    const b = blocks.of[i], z = zoneOf[i];
    if (b >= 0 && z >= 0 && zones[z]) counts[b].set(z, (counts[b].get(z) ?? 0) + 1);
  }
  return counts.map((c) => {
    let best = -1, most = -1;
    for (const [z, k] of c) {
      const l = zones[z]!, o = zones[best];
      if (best < 0 || k > most || (k === most && (+l.city > +o!.city || (l.city === o!.city && (l.trips > o!.trips || (l.trips === o!.trips && z < best)))))) (best = z), (most = k);
    }
    return best;
  });
}

/** Each block's name and its width: its zone's name for the biggest block of that zone, and for each smaller one
 * (T-2447) the nearest city or town of its zone's place list (a village only when no city or town is left) to the
 * middle of its stations, that no block has yet. No name twice. `want` lists the zones (indices into `zones`) that
 * have such blocks and no place list yet; a block with no place left has "". */
export function nameBlocks(g: { x: ArrayLike<number>; y: ArrayLike<number> }, blocks: Blocks, pick: number[], zones: (ZoneLabel | undefined)[]): { text: string[]; width: Float32Array; want: number[] } {
  const nb = blocks.list.length, size = new Int32Array(nb), sx = new Float64Array(nb), sy = new Float64Array(nb), core = new Int32Array(nb);
  for (let i = 0; i < blocks.of.length; i++) {
    const b = blocks.of[i];
    if (b < 0) continue;
    size[b]++;
    if (blocks.core[i]) (sx[b] += g.x[i]), (sy[b] += g.y[i]), core[b]++;
  }
  const text: string[] = Array(nb).fill(""), width = new Float32Array(nb), want = new Set<number>(), seen = new Set<string>();
  const order = [...Array(nb).keys()].sort((p, q) => size[q] - size[p] || blocks.list[p].key - blocks.list[q].key);
  const later: number[] = [];
  for (const b of order) {
    const z = pick[b];
    if (z < 0) continue;
    const l = zones[z]!;
    if (seen.has(l.name)) later.push(b);
    else seen.add(l.name), (text[b] = l.name), (width[b] = l.width);
  }
  for (const b of later) {
    const l = zones[pick[b]]!;
    if (!l.towns) {
      want.add(pick[b]);
      continue;
    }
    const mx = core[b] ? sx[b] / core[b] : 0, my = core[b] ? sy[b] / core[b] : 0;
    let best: Town | null = null, bd = Infinity;
    for (const t of l.towns) {
      if (seen.has(t.name)) continue;
      const dd = Math.hypot(t.x - mx, t.y - my) + (t.big ? 0 : 1e9);
      if (dd < bd) (bd = dd), (best = t);
    }
    if (best) seen.add(best.name), (text[b] = best.name), (width[b] = best.width);
  }
  return { text, width, want: [...want] };
}

/** The city names at every far level, for the drawing `d`, graph `g` and its blocks. */
export function placeCityNames(d: Drawing, g: { x: ArrayLike<number>; y: ArrayLike<number> }, blocks: Blocks, zoneOf: ArrayLike<number>, zones: (ZoneLabel | undefined)[], fontPx: number, zoneIds: string[] = []): CityNames {
  const t0 = performance.now();
  const nb = blocks.list.length, pitch = d.pitch, h = fontPx * NAME_LINE;
  const pick = blockNames(blocks, zoneOf, zones);
  const { text, width, want } = nameBlocks(g, blocks, pick, zones);
  const size = new Int32Array(nb);
  for (const b of blocks.of) if (b >= 0) size[b]++;
  // biggest block first
  const order = [...Array(nb).keys()].filter((b) => text[b]).sort((p, q) => size[q] - size[p] || blocks.list[p].key - blocks.list[q].key);
  // the lines' segments
  const seg: number[] = [];
  for (const r of d.ribbons) for (let j = 0; j + 3 < r.length; j += 2) seg.push(r[j], r[j + 1], r[j + 2], r[j + 3]);
  const ns = seg.length / 4;
  const levels: number[] = [], at: CityNames["at"] = [];
  for (let lv = LEVEL_LO; lv <= LEVEL_HI; lv++) {
    const P = 7 * pow(LEVEL_STEP, lv), s = P / pitch;
    // segments filed by cells about a name's height
    const C = (4 * h) / s, cells = new Map<number, number[]>();
    for (let i = 0; i < ns; i++) {
      const x0 = seg[i * 4], y0 = seg[i * 4 + 1], x1 = seg[i * 4 + 2], y1 = seg[i * 4 + 3];
      for (let gx = Math.floor(Math.min(x0, x1) / C); gx <= Math.floor(Math.max(x0, x1) / C); gx++)
        for (let gy = Math.floor(Math.min(y0, y1) / C); gy <= Math.floor(Math.max(y0, y1) / C); gy++) {
          const k = gx * 1048576 + gy, v = cells.get(k);
          if (v) v.push(i);
          else cells.set(k, [i]);
        }
    }
    const stamp = new Uint32Array(ns);
    let turn = 0;
    /** px of line inside box b (layout units) */
    const covered = (b: number[]): number => {
      turn++;
      let sum = 0;
      for (let gx = Math.floor(b[0] / C); gx <= Math.floor(b[2] / C); gx++)
        for (let gy = Math.floor(b[1] / C); gy <= Math.floor(b[3] / C); gy++)
          for (const i of cells.get(gx * 1048576 + gy) ?? []) {
            if (stamp[i] === turn) continue;
            stamp[i] = turn;
            sum += clipped(seg[i * 4], seg[i * 4 + 1], seg[i * 4 + 2], seg[i * 4 + 3], b);
          }
      return sum * s;
    };
    const names = new BoxGrid(), g = GAP_PX / s, sp = SPACE_PX / s;
    const out = { block: [] as number[], x: [] as number[], y: [] as number[], hx: [] as number[], hy: [] as number[] };
    for (const b of order) {
      const [x0, y0, x1, y1] = blocks.list[b].box, W = width[b] / s, H = h / s, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const cands: [number, number, number, number, number][] = [
        [mx, y1 + g, 0.5, 0, PLACE_COST.above], [x0, y1 + g, 0, 0, PLACE_COST.aboveFlush], [x1, y1 + g, 1, 0, PLACE_COST.aboveFlush],
        [mx, y0 - g, 0.5, 1, PLACE_COST.below], [x0, y0 - g, 0, 1, PLACE_COST.belowFlush], [x1, y0 - g, 1, 1, PLACE_COST.belowFlush],
        [x1 + g, my, 0, 0.5, PLACE_COST.side], [x0 - g, my, 1, 0.5, PLACE_COST.side],
      ];
      if (x1 - x0 >= W + 2 * g && y1 - y0 >= H + 2 * g)
        cands.push([x0 + g, y1 - g, 0, 1, PLACE_COST.inside], [x1 - g, y1 - g, 1, 1, PLACE_COST.inside], [x0 + g, y0 + g, 0, 0, PLACE_COST.inside], [x1 - g, y0 + g, 1, 0, PLACE_COST.inside]);
      let best: [number, number, number, number] | null = null, bc = Infinity, bpx: number[] = [];
      for (const [ax, ay, hx, hy, cost] of cands) {
        const nx0 = ax - hx * W, ny0 = ay - hy * H, box = [nx0, ny0, nx0 + W, ny0 + H];
        const px = [(box[0] - sp) * s, -(box[3] + sp) * s, (box[2] + sp) * s, -(box[1] - sp) * s];
        if (names.hits(px)) continue;
        const c = cost + covered([box[0] - g / 2, box[1] - g / 2, box[2] + g / 2, box[3] + g / 2]);
        if (c < bc) (bc = c), (best = [ax, ay, hx, hy]), (bpx = px);
      }
      if (!best) continue;
      names.add(bpx);
      out.block.push(b), out.x.push(best[0]), out.y.push(best[1]), out.hx.push(best[2]), out.hy.push(best[3]);
    }
    levels.push(P);
    at.push({ block: Int32Array.from(out.block), x: Float64Array.from(out.x), y: Float64Array.from(out.y), hx: Float32Array.from(out.hx), hy: Float32Array.from(out.hy) });
  }
  return { text, width, fontPx, levels, at, want: want.map((z) => zoneIds?.[z] ?? ""), ms: performance.now() - t0 };
}

/** the length of segment (x0, y0)-(x1, y1) inside box b (Liang-Barsky) */
function clipped(x0: number, y0: number, x1: number, y1: number, b: number[]): number {
  const dx = x1 - x0, dy = y1 - y0;
  let t0 = 0, t1 = 1;
  const p = [-dx, dx, -dy, dy], q = [x0 - b[0], b[2] - x0, y0 - b[1], b[3] - y0];
  for (let k = 0; k < 4; k++) {
    if (p[k] === 0) {
      if (q[k] < 0) return 0;
      continue;
    }
    const r = q[k] / p[k];
    if (p[k] < 0) t0 = Math.max(t0, r);
    else t1 = Math.min(t1, r);
    if (t0 > t1) return 0;
  }
  const l2 = dx * dx + dy * dy;
  return (t1 - t0) * Math.sqrt(l2);
}

/** the city level drawn at a pitch of `pitchPx` CSS px: the last placed for at most that, -1 for none (and none
 * from CITY_SWITCH up) */
export function cityLevelAt(c: CityNames, pitchPx: number): number {
  if (pitchPx >= CITY_SWITCH * (1 - 1e-9)) return -1;
  let i = -1;
  while (i + 1 < c.levels.length && c.levels[i + 1] <= pitchPx * (1 + 1e-9)) i++;
  return i;
}

/** a placed name's box (layout units, y north) at `s` CSS px a layout unit */
export function cityBox(c: CityNames, li: number, i: number, s: number): number[] {
  const L = c.at[li], b = L.block[i], W = c.width[b] / s, H = (c.fontPx * NAME_LINE) / s;
  const x0 = L.x[i] - L.hx[i] * W, y0 = L.y[i] - L.hy[i] * H;
  return [x0, y0, x0 + W, y0 + H];
}

/** the city names over a painted drawing, at `scale` device px a layout unit, (cx, cy) in the middle */
export function paintCityNames(ctx: CanvasRenderingContext2D, c: CityNames, pitch: number, w: number, h: number, cx: number, cy: number, scale: number, dpr: number, theme: { ink: string; halo: string; font: string }): void {
  const s = scale / dpr, li = cityLevelAt(c, pitch * s);
  if (li < 0) return;
  const L = c.at[li], f = c.fontPx * dpr;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.font = `${f}px ${theme.font}`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3 * dpr;
  const shown: [number, number, string][] = [];
  for (let i = 0; i < L.block.length; i++) {
    const b = cityBox(c, li, i, s);
    const x0 = w / 2 + (b[0] - cx) * scale, x1 = w / 2 + (b[2] - cx) * scale, y0 = h / 2 - (b[3] - cy) * scale, y1 = h / 2 - (b[1] - cy) * scale;
    if (x1 < 0 || x0 > w || y1 < 0 || y0 > h) continue;
    shown.push([(x0 + x1) / 2, (y0 + y1) / 2, c.text[L.block[i]]]);
  }
  ctx.strokeStyle = theme.halo;
  for (const [x, y, t] of shown) ctx.strokeText(t, x, y);
  ctx.fillStyle = theme.ink;
  for (const [x, y, t] of shown) ctx.fillText(t, x, y);
}
