// City names on the network diagram zoomed far out (T-2438, T-2521, T-2581; docs/ui-diagram.md "City names"): one name
// per city block (blocks.ts), centred on its named zone's own stations as drawn, in the map's place grey at half
// opacity, painted under the lines. Below CITY_SWITCH the city names are drawn and the station names not. A name stays
// near one screen size, BIG_MIN to BIG_MAX times the station names' size; where two names meet the bigger block keeps
// its size and the other shrinks toward BIG_MIN, then goes (paintCityNames).

import type { Blocks } from "./blocks";
import { pow } from "./fmath";
import { NAME_LINE } from "./names";

/** a zone as the page describes it: its shown name, the name's width in CSS px, its local trips a day (its size)
 * and whether it is a city zone */
export interface ZoneLabel { name: string; width: number; trips: number; city: boolean; towns?: Town[] }
/** a place of the zone's place list, for naming a block the zone's own name is taken from (T-2447): its name and
 * width, its place in the graph's metres, and whether it is a city or town (a village only where none is left) */
export interface Town { name: string; width: number; x: number; y: number; big: boolean }

/** below this pitch in CSS px (7 times the step to the power -9: 1.47 px) the city names are drawn and the station
 * names not; the station names' faint levels start there (names.ts FAINT_LO) */
export const CITY_SWITCH = 7 * pow(pow(2, 0.25), -9);
/** a name's size in station-name sizes: BIG_AT where its stations span SPAN_PX on screen, times that span's ratio to
 * the power GROW, within BIG_MIN to BIG_MAX; so zooming out 15 times shrinks it by a third at most (T-2581) */
const BIG_MIN = 1.25, BIG_MAX = 2.5, BIG_AT = 1.5, SPAN_PX = 64, GROW = 0.15;
/** a name's stations: the share of them left out at each side of its box, so a few far ones do not move it */
const TRIM = 0.1;
/** px kept clear between two names */
const SPACE_PX = 6;
/** a name's opacity over the background and what lies under it */
const NAME_ALPHA = 0.5;

export interface CityNames {
  /** per block, its name ("" for none) and the name's width in CSS px at fontPx */
  text: string[];
  width: Float32Array;
  fontPx: number;
  /** per block, the middle of its name's stations as drawn and their span, in layout units (`anchors`) */
  mx: Float64Array; my: Float64Array; bw: Float64Array; bh: Float64Array;
  /** the named blocks, biggest first (by stations) */
  order: Int32Array;
  /** ids of the zones with blocks that wait for the zone's place list to be named (T-2447) */
  want: string[];
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

/** Where each block's name goes (T-2581): the stations of the block's city (`core`) in its named zone `pick[b]`, else
 * its city's stations, else all of its stations; the box of their drawn places (`x`, `y`) with TRIM of them left out
 * at each side, its middle and size. A block also holds the lines out of its city, which may run far (Washington's
 * held the stations to Atlanta), so its own box can lie well off the city. */
export function anchors(x: ArrayLike<number>, y: ArrayLike<number>, blocks: Blocks, zoneOf: ArrayLike<number>, pick: number[]) {
  const nb = blocks.list.length, rank = new Int32Array(blocks.of.length), best = new Int32Array(nb);
  for (let i = 0; i < blocks.of.length; i++) {
    const b = blocks.of[i];
    if (b < 0) continue;
    rank[i] = 1 + blocks.core[i] + (pick[b] >= 0 && zoneOf[i] === pick[b] && blocks.core[i] ? 1 : 0);
    best[b] = Math.max(best[b], rank[i]);
  }
  const xs: number[][] = Array.from({ length: nb }, () => []), ys: number[][] = Array.from({ length: nb }, () => []);
  for (let i = 0; i < blocks.of.length; i++) {
    const b = blocks.of[i];
    if (b >= 0 && rank[i] === best[b]) xs[b].push(x[i]), ys[b].push(y[i]);
  }
  const mx = new Float64Array(nb), my = new Float64Array(nb), bw = new Float64Array(nb), bh = new Float64Array(nb);
  const span = (v: number[]): [number, number] => {
    v.sort((p, q) => p - q);
    const cut = Math.floor((v.length - 1) * TRIM);
    return [v[cut], v[v.length - 1 - cut]];
  };
  for (let b = 0; b < nb; b++) {
    if (!xs[b].length) continue;
    const [x0, x1] = span(xs[b]), [y0, y1] = span(ys[b]);
    (mx[b] = (x0 + x1) / 2), (my[b] = (y0 + y1) / 2), (bw[b] = x1 - x0), (bh[b] = y1 - y0);
  }
  return { mx, my, bw, bh };
}

/** The city names for the graph `g` (places in its metres) laid out at `drawn` and its blocks. */
export function placeCityNames(g: { x: ArrayLike<number>; y: ArrayLike<number> }, drawn: { x: ArrayLike<number>; y: ArrayLike<number> }, blocks: Blocks, zoneOf: ArrayLike<number>, zones: (ZoneLabel | undefined)[], fontPx: number, zoneIds: string[] = []): CityNames {
  const nb = blocks.list.length;
  const pick = blockNames(blocks, zoneOf, zones);
  const { text, width, want } = nameBlocks(g, blocks, pick, zones);
  const size = new Int32Array(nb);
  for (const b of blocks.of) if (b >= 0) size[b]++;
  const order = Int32Array.from([...Array(nb).keys()].filter((b) => text[b]).sort((p, q) => size[q] - size[p] || blocks.list[p].key - blocks.list[q].key));
  return { text, width, fontPx, ...anchors(drawn.x, drawn.y, blocks, zoneOf, pick), order, want: want.map((z) => zoneIds?.[z] ?? "") };
}

/** the city names over a painted drawing, at `scale` device px a layout unit, (cx, cy) in the middle. A name's size
 * grows slowly with its stations' span on screen (BIG_AT, SPAN_PX, GROW), within BIG_MIN to BIG_MAX station-name
 * sizes; biggest block first, a name that would meet one already placed shrinks toward BIG_MIN and is left out if it
 * still meets one. */
export function paintCityNames(ctx: CanvasRenderingContext2D, c: CityNames, pitch: number, w: number, h: number, cx: number, cy: number, scale: number, dpr: number, theme: { ink: string; font: string }): void {
  const s = scale / dpr;
  if (pitch * s >= CITY_SWITCH * (1 - 1e-9)) return;
  const placed: number[][] = [], shown: [number, number, number, string][] = [], sp = SPACE_PX * dpr;
  for (const b of c.order) {
    const x = w / 2 + (c.mx[b] - cx) * scale, y = h / 2 - (c.my[b] - cy) * scale;
    const per = c.width[b] / c.fontPx, spanPx = Math.max(c.bw[b], c.bh[b], 1e-9) * s;
    let f = c.fontPx * Math.min(BIG_MAX, Math.max(BIG_MIN, BIG_AT * pow(spanPx / SPAN_PX, GROW)));
    for (;;) {
      const hw = (per * f * dpr) / 2 + sp, hh = (f * NAME_LINE * dpr) / 2 + sp;
      const r = [x - hw, y - hh, x + hw, y + hh];
      if (r[2] < 0 || r[0] > w || r[3] < 0 || r[1] > h) break;
      if (!placed.some((o) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1])) {
        placed.push(r);
        shown.push([x, y, f * dpr, c.text[b]]);
        break;
      }
      if (f <= BIG_MIN * c.fontPx) break;
      f = Math.max(BIG_MIN * c.fontPx, f * 0.9);
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = NAME_ALPHA;
  ctx.fillStyle = theme.ink;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  for (const [x, y, f, t] of shown) {
    ctx.font = `${f}px ${theme.font}`;
    ctx.fillText(t, x, y);
  }
  ctx.globalAlpha = 1;
}
