// City and zone names on the network diagram (T-2438, T-2521, T-2581, T-2640; docs/ui-diagram.md "City names"): one name
// per city block (blocks.ts), centred on its named zone's own stations as drawn, and one per zone that holds stations
// whose name no block shows, in the map's place grey at half opacity, painted under the lines. A name stays near one screen
// size, BIG_MIN to BIG_MAX times the station names' size; where two names meet the bigger one keeps its size and the
// other shrinks toward BIG_MIN, then goes. How many are shown depends on the screen (paintCityNames): about one per
// NAME_AREA_PX of it, at least FEW_NAMES where that many zones are in view, busiest first. Below CITY_SWITCH the
// station names are not drawn; above it the names keep clear of the station names and, where room allows, the pills.

import type { Blocks } from "./blocks";
import type { Drawing } from "./draw";
import { pillSize } from "./draw";
import { pow } from "./fmath";
import { levelAt, NAME_LINE, placedBox, type NamePlan } from "./names";

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
/** how many names a screen takes: one per NAME_AREA_PX CSS px squared of it, within MIN_NAMES to MAX_NAMES (T-2640) */
export const NAME_AREA_PX = 150000, MIN_NAMES = 4, MAX_NAMES = 40;
/** at least this many names are shown when that many have a place in view, over pills if need be */
export const FEW_NAMES = 3;
/** the power of a name's stations in the score that spreads the names (T-2674): 1 is stations times distance, 0 is distance alone; 0.5 keeps a big city first without letting its neighbours crowd out Denver */
const SPREAD_RANK = 0.5;
/** where the names keep clear of the station names, a name is tried at up to this many steps of its half size off its
 * middle, across and up and down */
const STEPS_X = 2, STEPS_Y = 3;

export interface CityNames {
  /** per entry (the blocks, then the zones with stations and no named block, T-2640), its name ("" for none) and the
   * name's width in CSS px at fontPx */
  text: string[];
  width: Float32Array;
  fontPx: number;
  /** per entry, the middle of its name's stations as drawn and their span, in layout units (`anchors`) */
  mx: Float64Array; my: Float64Array; bw: Float64Array; bh: Float64Array;
  /** per entry, its stations (what the order goes by) */
  rank: Float64Array;
  /** the named entries, busiest first (by stations) */
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
  for (let b = 0; b < nb; b++) {
    if (!xs[b].length) continue;
    const [x0, x1] = span(xs[b]), [y0, y1] = span(ys[b]);
    (mx[b] = (x0 + x1) / 2), (my[b] = (y0 + y1) / 2), (bw[b] = x1 - x0), (bh[b] = y1 - y0);
  }
  return { mx, my, bw, bh };
}

/** the range of `v` with TRIM of it left out at each side (sorts `v`) */
function span(v: number[]): [number, number] {
  v.sort((p, q) => p - q);
  const cut = Math.floor((v.length - 1) * TRIM);
  return [v[cut], v[v.length - 1 - cut]];
}

/** The city names for the graph `g` (places in its metres) laid out at `drawn` and its blocks. */
export function placeCityNames(g: { x: ArrayLike<number>; y: ArrayLike<number> }, drawn: { x: ArrayLike<number>; y: ArrayLike<number> }, blocks: Blocks, zoneOf: ArrayLike<number>, zones: (ZoneLabel | undefined)[], fontPx: number, zoneIds: string[] = []): CityNames {
  const nb = blocks.list.length;
  const pick = blockNames(blocks, zoneOf, zones);
  const { text, width, want } = nameBlocks(g, blocks, pick, zones);
  const size = new Int32Array(nb);
  for (const b of blocks.of) if (b >= 0) size[b]++;
  const a = anchors(drawn.x, drawn.y, blocks, zoneOf, pick);
  // T-2640: a zone with stations whose name no block shows has its own name, at the middle of its stations
  const used = new Set(text.filter(Boolean)), left = new Map<number, number[]>();
  for (let i = 0; i < blocks.of.length; i++) {
    const z = zoneOf[i];
    if (z < 0 || !zones[z] || used.has(zones[z]!.name)) continue;
    (left.get(z) ?? left.set(z, []).get(z)!).push(i);
  }
  const extra = [...left.keys()].sort((p, q) => p - q).filter((z) => (used.has(zones[z]!.name) ? false : (used.add(zones[z]!.name), true)));
  const ne = extra.length, n = nb + ne;
  const mx = new Float64Array(n), my = new Float64Array(n), bw = new Float64Array(n), bh = new Float64Array(n), rank = new Float64Array(n), w = new Float32Array(n);
  mx.set(a.mx), my.set(a.my), bw.set(a.bw), bh.set(a.bh), rank.set(size), w.set(width);
  const all = text.concat(Array(ne).fill(""));
  extra.forEach((z, e) => {
    const v = left.get(z)!, xs = v.map((i) => drawn.x[i]), ys = v.map((i) => drawn.y[i]);
    const [x0, x1] = span(xs), [y0, y1] = span(ys), k = nb + e;
    (mx[k] = (x0 + x1) / 2), (my[k] = (y0 + y1) / 2), (bw[k] = x1 - x0), (bh[k] = y1 - y0), (rank[k] = v.length);
    (all[k] = zones[z]!.name), (w[k] = zones[z]!.width);
  });
  const order = Int32Array.from([...Array(n).keys()].filter((b) => all[b]).sort((p, q) => rank[q] - rank[p] || (p < nb && q < nb ? blocks.list[p].key - blocks.list[q].key : p - q)));
  return { text: all, width: w, fontPx, mx, my, bw, bh, rank, order, want: want.map((z) => zoneIds?.[z] ?? "") };
}

/** what the names keep clear of where the station names are drawn: the drawing and its station names */
export interface Over { d: Drawing; plan: NamePlan }

/** the pills and the station names drawn at this view, as device px boxes (x0, y0, x1, y1), those near the screen */
function drawnBoxes(o: Over, w: number, h: number, cx: number, cy: number, scale: number, dpr: number): { pills: number[][]; names: number[][] } {
  const { d, plan } = o, s = scale / dpr, pills: number[][] = [], names: number[][] = [];
  const sx = (x: number) => w / 2 + (x - cx) * scale, sy = (y: number) => h / 2 - (y - cy) * scale;
  const keep = (b: number[]) => b[2] >= 0 && b[0] <= w && b[3] >= 0 && b[1] <= h;
  d.pills.forEach((p, i) => {
    const [r, ol] = pillSize(d, d.pillOf[i]), e = r + ol;
    const b = [sx(Math.min(p[0], p[2]) - e), sy(Math.max(p[1], p[3]) + e), sx(Math.max(p[0], p[2]) + e), sy(Math.min(p[1], p[3]) - e)];
    if (keep(b)) pills.push(b);
  });
  const li = levelAt(plan, d.pitch * s);
  if (li >= 0) {
    for (let i = 0; i < plan.at[li].node.length; i++) {
      const q = placedBox(plan, li, i, s), b = [sx(q[0]) - 3 * dpr, sy(q[3]) - 3 * dpr, sx(q[2]) + 3 * dpr, sy(q[1]) + 3 * dpr];
      if (keep(b)) names.push(b);
    }
  }
  return { pills, names };
}

const meets = (r: number[], o: number[]) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1];

/** the names over a painted drawing, at `scale` device px a layout unit, (cx, cy) in the middle. A name's size grows
 * slowly with its stations' span on screen (BIG_AT, SPAN_PX, GROW), within BIG_MIN to BIG_MAX station-name sizes.
 * Busiest first among those with a place in view, up to a count the screen's area sets (NAME_AREA_PX); a name that
 * would meet one already placed shrinks toward BIG_MIN and is left out if it still meets one. A name whose middle is
 * off screen but whose stations are partly in view is moved into view. Where station names are drawn (`over`, from
 * CITY_SWITCH up) a name keeps clear of them and of the pills, tried round its middle; for the first FEW_NAMES it may
 * lie over pills if no place is clear of them. */
export function paintCityNames(ctx: CanvasRenderingContext2D, c: CityNames, pitch: number, w: number, h: number, cx: number, cy: number, scale: number, dpr: number, theme: { ink: string; font: string }, over?: Over): void {
  const s = scale / dpr, near = pitch * s >= CITY_SWITCH * (1 - 1e-9);
  const cap = Math.min(MAX_NAMES, Math.max(MIN_NAMES, Math.round((w * h) / (dpr * dpr) / NAME_AREA_PX)));
  const clear = near && over ? drawnBoxes(over, w, h, cx, cy, scale, dpr) : null;
  const placed: number[][] = [], shown: [number, number, number, string][] = [], sp = SPACE_PX * dpr;
  const tryPlace = (b: number): number[] | null => {
    const x = w / 2 + (c.mx[b] - cx) * scale, y = h / 2 - (c.my[b] - cy) * scale, bwp = c.bw[b] * scale, bhp = c.bh[b] * scale;
    const per = c.width[b] / c.fontPx, spanPx = Math.max(c.bw[b], c.bh[b], 1e-9) * s;
    let f = c.fontPx * Math.min(BIG_MAX, Math.max(BIG_MIN, BIG_AT * pow(spanPx / SPAN_PX, GROW)));
    for (;;) {
      const hw = (per * f * dpr) / 2 + sp, hh = (f * NAME_LINE * dpr) / 2 + sp;
      // a middle off screen moves into the part of the stations' box that is in view
      let bx = x, by = y;
      if (x < 0 || x > w || y < 0 || y > h) {
        const x0 = Math.max(x - bwp / 2, 0), x1 = Math.min(x + bwp / 2, w), y0 = Math.max(y - bhp / 2, 0), y1 = Math.min(y + bhp / 2, h);
        if (x0 > x1 || y0 > y1 || (x0 === x1 && y0 === y1)) return null;
        (bx = (x0 + x1) / 2), (by = (y0 + y1) / 2);
      }
      const spots: [number, number][] = [[bx, by]];
      if (clear) {
        for (let i = -STEPS_X; i <= STEPS_X; i++) for (let j = -STEPS_Y; j <= STEPS_Y; j++) {
          const px = bx + i * hw, py = by + j * hh;
          if ((i || j) && px >= 0 && px <= w && py >= 0 && py <= h && Math.abs(px - x) <= bwp / 2 + hw && Math.abs(py - y) <= bhp / 2 + hh) spots.push([px, py]);
        }
        spots.sort((p, q) => Math.hypot(p[0] - bx, (p[1] - by) * 2) - Math.hypot(q[0] - bx, (q[1] - by) * 2));
      }
      let got: number[] | null = null, over0: number[] | null = null;
      for (const [px, py] of spots) {
        const r = [px - hw, py - hh, px + hw, py + hh];
        if (r[2] < 0 || r[0] > w || r[3] < 0 || r[1] > h || placed.some((o) => meets(r, o))) continue;
        if (clear?.names.some((o) => meets(r, o))) continue;
        if (!clear || !clear.pills.some((o) => meets(r, o))) {
          got = [px, py, ...r];
          break;
        }
        over0 ??= [px, py, ...r];
      }
      if (!got && over0 && shown.length < FEW_NAMES) got = over0;
      if (got) {
        placed.push(got.slice(2));
        shown.push([got[0], got[1], f * dpr, c.text[b]]);
        return [got[0], got[1]];
      }
      if (f <= BIG_MIN * c.fontPx) return null;
      f = Math.max(BIG_MIN * c.fontPx, f * 0.9);
    }
  };
  // T-2674: the next name tried is the one with the highest square root of its stations times its distance on screen to the nearest name already
  // shown (the busiest first, while none is), so the names spread over the view; one that does not fit is dropped
  const left = [...c.order], near2 = new Float64Array(c.order.length).fill(Infinity);
  while (shown.length < cap && left.length) {
    let bi = 0, bs = -1;
    for (let i = 0; i < left.length; i++) {
      const sc = pow(c.rank[left[i]], SPREAD_RANK) * (near2[i] === Infinity ? 1e9 : Math.sqrt(near2[i]));
      if (sc > bs) (bs = sc), (bi = i);
    }
    const b = left[bi], at = tryPlace(b);
    left.splice(bi, 1), near2.copyWithin(bi, bi + 1);
    if (!at) continue;
    for (let i = 0; i < left.length; i++) {
      const o = left[i], ox = Math.min(w, Math.max(0, w / 2 + (c.mx[o] - cx) * scale)), oy = Math.min(h, Math.max(0, h / 2 - (c.my[o] - cy) * scale));
      near2[i] = Math.min(near2[i], (ox - at[0]) ** 2 + (oy - at[1]) ** 2);
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

