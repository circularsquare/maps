// Station names on the network diagram (T-2162, docs/ui-diagram.md "Names"). A name goes at one of
// eight places round one of its station's pills, off the pill's ends first, at two distances, kept off
// the lines, the pills and the names placed before it (`BoxGrid`): interchanges first, then line ends,
// then the rest by lines stopping. Along a straight run the names keep to one side, the side with fewer
// other stations. Placed once per zoom level (a pitch in CSS px); a level's names are drawn at every
// zoom up to the next. Text keeps its size in px while the drawing scales, so past its level a name only
// shrinks toward its pill, inside the box that was tested: what was clear at the level stays clear.
// No DOM here: the layout worker runs it, and checks/t2162-diagram-names.mjs in Node.

import { BoxGrid } from "../map/labelLayout";
import type { DiagramGraph } from "./graph";
import type { DiagramLayout } from "./layout";
import { PILL_OUTLINE, PILL_R, type Drawing } from "./draw";

/** px between a pill's outline and its name */
const GAP_PX = 3;
/** the second distance: this many pitches further out, over a line beside the pill */
const LIFT = 1.5;
/** an interchange may also go this many times LIFT out, where lines meet all round it */
const LIFT_HUB = 3;
/** but those never more than this many CSS px out at the level's pitch */
const LIFT_MAX_PX = 32;
/** a name keeps this many pitches, plus a px, from a line's middle (the casing's half width) */
const CLEAR = 0.5;
/** the zoom levels, a pitch in CSS px: BASE times STEP to the powers LO to HI */
const LEVEL_BASE = 7, LEVEL_STEP = 2 ** 0.25, LEVEL_LO = -3, LEVEL_HI = 12;
/** a name's box is this many times the font size tall, and a second line adds this many */
export const NAME_LINE = 1.2, NAME_LINE2 = 1.05;
/** a name over two lines costs this much more than on one, and the second distance this much more */
const WRAP_COST = 2, LIFT_COST = 3;
/** a name slid along beside its pill, flush with its anchor, costs this much more than centred */
const SLIDE_COST = 0.3;
/** names at least this many characters long may go over two lines */
const WRAP_FROM = 12;
/** stations within this many pitches of a run, beside it, count against that side */
const SIDE_REACH = 6;
/** the last level's place for a name counts this much cheaper */
const STAY = 1.2;

/** the eight places, counter-clockwise from east */
const DIR8: [number, number][] = Array.from({ length: 8 }, (_, k) => [Math.round(Math.cos((k * Math.PI) / 4) * 1e9) / 1e9, Math.round(Math.sin((k * Math.PI) / 4) * 1e9) / 1e9]);

export interface LevelNames {
  /** per name placed: its node, its pill's anchor (layout units), its place (DIR8) and lift (layout units) */
  node: Int32Array;
  x: Float64Array;
  y: Float64Array;
  dir: Uint8Array;
  /** east, west, north and south: slid along (`nameBox`) */
  slide: Int8Array;
  lift: Float64Array;
  /** 1 where it goes over two lines (`splitName`) */
  wrap: Uint8Array;
}

export interface NamePlan {
  /** a pitch in CSS px each level is placed for, rising */
  levels: number[];
  at: LevelNames[];
  /** per graph node its name ("" none), the name's width in CSS px, and over two lines (0 where it
   * is not split) */
  text: string[];
  width: Float32Array;
  width2: Float32Array;
  fontPx: number;
  ms: number;
}

/** a long name's two lines: broken at the space or hyphen nearest its middle, the hyphen kept on the
 * first; null for a short name or one with nowhere to break */
export function splitName(name: string): [string, string] | null {
  if ([...name].length < WRAP_FROM) return null;
  let best = -1, bd = Infinity;
  for (let i = 1; i < name.length - 1; i++) {
    if (name[i] !== " " && name[i] !== "-") continue;
    const d = Math.abs(i + 0.5 - name.length / 2);
    if (d < bd) (bd = d), (best = i);
  }
  if (best < 0) return null;
  return name[best] === " " ? [name.slice(0, best), name.slice(best + 1)] : [name.slice(0, best + 1), name.slice(best + 1)];
}

/** a station's name from its complex's members: the name with the most lines stopping, cut at the first "/" */
export function pickName(members: readonly { name: string; lines: number }[]): string {
  const score = new Map<string, number>();
  for (const m of members) score.set(m.name, (score.get(m.name) ?? 0) + m.lines + 1e-3);
  let best = "", bs = -1;
  for (const [n, s] of score) if (s > bs) (best = n), (bs = s);
  return best.split("/")[0].trim();
}

/** a name's width in px at `fontPx`, guessed from its characters (Node checks; the page measures) */
export function guessWidth(name: string, fontPx: number): number {
  let w = 0;
  for (const ch of name) {
    const c = ch.codePointAt(0)!;
    w += c >= 0x2e80 ? 1 : /[A-Z]/.test(ch) ? 0.68 : /[0-9]/.test(ch) ? 0.6 : ch === " " ? 0.27 : /[a-z]/.test(ch) ? 0.53 : 0.35;
  }
  return w * fontPx;
}

/** a name's box (x0, y0, x1, y1, layout units, y north) at `s` CSS px a layout unit, `w` by `h` px,
 * placed `dir` of its anchor, east, west, north and south of it slid along by `slide` (-1 flush with
 * the anchor's left or bottom, 1 its right or top, 0 centred); and the box tested, which runs back to
 * the anchor lifted */
export function nameBox(ax: number, ay: number, dir: number, slide: number, lift: number, w: number, h: number, s: number, tested?: number[]): number[] {
  const [cx, cy] = DIR8[dir];
  const o = lift + GAP_PX / s, rx = ax + cx * o, ry = ay + cy * o, W = w / s, H = h / s;
  const x0 = cx > 0.3 ? rx : cx < -0.3 ? rx - W : rx - W / 2 + (slide * W) / 2;
  const y0 = cy > 0.3 ? ry : cy < -0.3 ? ry - H : ry - H / 2 + (slide * H) / 2;
  if (tested) {
    const lx = ax + cx * lift, ly = ay + cy * lift;
    tested[0] = Math.min(x0, lx), tested[1] = Math.min(y0, ly), tested[2] = Math.max(x0 + W, lx), tested[3] = Math.max(y0 + H, ly);
  }
  return [x0, y0, x0 + W, y0 + H];
}

/** segment (x0, y0)-(x1, y1) against box b: whether they meet (separating axes) */
function segHits(x0: number, y0: number, x1: number, y1: number, b: ArrayLike<number>): boolean {
  if (Math.max(x0, x1) < b[0] || Math.min(x0, x1) > b[2] || Math.max(y0, y1) < b[1] || Math.min(y0, y1) > b[3]) return false;
  const nx = y0 - y1, ny = x1 - x0, c = nx * x0 + ny * y0;
  const s1 = nx * b[0] + ny * b[1] - c, s2 = nx * b[2] + ny * b[1] - c, s3 = nx * b[0] + ny * b[3] - c, s4 = nx * b[2] + ny * b[3] - c;
  return !((s1 > 0 && s2 > 0 && s3 > 0 && s4 > 0) || (s1 < 0 && s2 < 0 && s3 < 0 && s4 < 0));
}
const ptSeg = (px: number, py: number, x0: number, y0: number, x1: number, y1: number) => {
  const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
  return Math.hypot(x0 + dx * t - px, y0 + dy * t - py);
};
const ptBox = (px: number, py: number, b: ArrayLike<number>) => Math.hypot(Math.max(b[0] - px, 0, px - b[2]), Math.max(b[1] - py, 0, py - b[3]));
/** the distance from a segment to box b */
export function segBoxDist(x0: number, y0: number, x1: number, y1: number, b: ArrayLike<number>): number {
  if (segHits(x0, y0, x1, y1, b)) return 0;
  return Math.min(ptBox(x0, y0, b), ptBox(x1, y1, b), ptSeg(b[0], b[1], x0, y0, x1, y1), ptSeg(b[2], b[1], x0, y0, x1, y1), ptSeg(b[0], b[3], x0, y0, x1, y1), ptSeg(b[2], b[3], x0, y0, x1, y1));
}

/** segments filed by grid cell: the lines' ribbons (kind 0), pills (1) and passages (2), each with a
 * half width and its station (-1 for a line) */
class SegGrid {
  seg: number[] = [];
  r: number[] = [];
  kind: number[] = [];
  owner: number[] = [];
  private cells = new Map<number, number[]>();
  private stamp: Uint32Array | null = null;
  private turn = 0;
  constructor(private cell: number) {}
  add(x0: number, y0: number, x1: number, y1: number, r: number, kind: number, owner: number) {
    const i = this.r.length;
    this.seg.push(x0, y0, x1, y1);
    this.r.push(r), this.kind.push(kind), this.owner.push(owner);
    const C = this.cell;
    for (let gx = Math.floor((Math.min(x0, x1) - r) / C); gx <= Math.floor((Math.max(x0, x1) + r) / C); gx++)
      for (let gy = Math.floor((Math.min(y0, y1) - r) / C); gy <= Math.floor((Math.max(y0, y1) + r) / C); gy++) {
        const k = gx * 1048576 + gy, v = this.cells.get(k);
        if (v) v.push(i);
        else this.cells.set(k, [i]);
      }
  }
  /** each segment filed near box b grown by `pad`, once; stops when `f` returns true */
  some(b: ArrayLike<number>, pad: number, f: (i: number) => boolean): boolean {
    if (!this.stamp || this.stamp.length < this.r.length) this.stamp = new Uint32Array(this.r.length);
    const turn = ++this.turn, C = this.cell;
    for (let gx = Math.floor((b[0] - pad) / C); gx <= Math.floor((b[2] + pad) / C); gx++)
      for (let gy = Math.floor((b[1] - pad) / C); gy <= Math.floor((b[3] + pad) / C); gy++) {
        const v = this.cells.get(gx * 1048576 + gy);
        if (!v) continue;
        for (const i of v) {
          if (this.stamp[i] === turn) continue;
          this.stamp[i] = turn;
          if (f(i)) return true;
        }
      }
    return false;
  }
}

/** The names placed at every level. `text`, `width` and `width2` (CSS px at `fontPx`, on one line and
 * the wider of two, 0 for none) are per graph node. */
export function placeNames(g: DiagramGraph, lay: DiagramLayout, d: Drawing, text: string[], width: ArrayLike<number>, width2: ArrayLike<number>, fontPx: number, why?: (level: number, k: number, dir: number, lifted: number, what: string) => void): NamePlan {
  const t0 = performance.now();
  const n = g.key.length, pitch = d.pitch, h = fontPx * NAME_LINE, h2 = fontPx * (NAME_LINE + NAME_LINE2);
  const R = (PILL_R + PILL_OUTLINE) * d.stroke, rPass = (PILL_R * 0.45 + PILL_OUTLINE) * d.stroke;
  const grid = new SegGrid(4 * pitch);
  for (const p of d.ribbons) for (let j = 0; j + 3 < p.length; j += 2) grid.add(p[j], p[j + 1], p[j + 2], p[j + 3], 0, 0, -1);
  const pillsOf: number[][] = Array.from({ length: n }, () => []);
  d.pills.forEach((p, i) => {
    grid.add(p[0], p[1], p[2], p[3], R + p[4], 1, d.pillOf[i]);
    pillsOf[d.pillOf[i]].push(i);
  });
  // a passage belongs to the station whose pill it starts at
  for (const q of d.passages) {
    let own = -1;
    grid.some([q[0], q[1], q[0], q[1]], 0, (i) => grid.kind[i] === 1 && ptSeg(q[0], q[1], grid.seg[i * 4], grid.seg[i * 4 + 1], grid.seg[i * 4 + 2], grid.seg[i * 4 + 3]) < 1e-6 * pitch + 1e-9 && ((own = grid.owner[i]), true));
    grid.add(q[0], q[1], q[2], q[3], rPass, 2, own);
  }

  // the order: interchanges (two colours or more stopping) by lines stopping, then line ends, then the rest
  const lines = new Uint16Array(n), ends = new Uint8Array(n), colours: Set<string>[] = Array.from({ length: n }, () => new Set());
  for (const ln of g.lines) {
    ln.chain.forEach((k, i) => {
      if (!ln.stop[i]) return;
      lines[k]++;
      colours[k].add(ln.colour);
    });
    ends[ln.chain[0]] = ends[ln.chain[ln.chain.length - 1]] = 1;
  }
  const order: number[] = [];
  for (let k = 0; k < n; k++) if (pillsOf[k].length && text[k]) order.push(k);
  const tier = (k: number) => (colours[k].size >= 2 ? 0 : ends[k] ? 1 : 2);
  order.sort((a, b) => tier(a) - tier(b) || lines[b] - lines[a] || g.key[a] - g.key[b]);

  // each station on a straight run: the run's sides, and which has fewer other stations near
  const normal = new Float64Array(n * 2), prefer = new Int8Array(n);
  lay.pieces.forEach((pc) => {
    const at = pc.at, v = pc.verts, nv = v.length / 2;
    const isNode = new Uint8Array(nv);
    for (const a of at) isNode[a] = 1;
    // legs: runs of vertices between the piece's ends and its bend
    const legs: number[][] = [[]];
    for (let j = 0; j < nv; j++) {
      if (!isNode[j]) { legs.push([]); continue; }
      legs[legs.length - 1].push(j);
    }
    const nodeAt = new Map<number, number>();
    at.forEach((a, i) => nodeAt.set(a, pc.nodes[i]));
    const bend = isNode.indexOf(0);
    legs.forEach((leg, li) => {
      // the leg's direction: toward the bend, or end to end
      const a = li === 0 ? 0 : bend, b = li === 0 ? (bend >= 0 ? bend : nv - 1) : nv - 1;
      let dx = v[b * 2] - v[a * 2], dy = v[b * 2 + 1] - v[a * 2 + 1];
      const l = Math.hypot(dx, dy);
      if (l < 1e-9) return;
      (dx /= l), (dy /= l);
      const nx = -dy, ny = dx;
      const inner = leg.filter((j) => j !== 0 && j !== nv - 1).map((j) => nodeAt.get(j)!);
      if (!inner.length) return;
      const mine = new Set(inner);
      let left = 0, right = 0;
      const ox = v[a * 2], oy = v[a * 2 + 1], reach = SIDE_REACH * pitch, hw = Math.max(...inner.map((k) => lay.hw[k]));
      const box = [Math.min(v[a * 2], v[b * 2]) - reach, Math.min(v[a * 2 + 1], v[b * 2 + 1]) - reach, Math.max(v[a * 2], v[b * 2]) + reach, Math.max(v[a * 2 + 1], v[b * 2 + 1]) + reach];
      grid.some(box, 0, (i) => {
        if (grid.kind[i] !== 1 || mine.has(grid.owner[i])) return false;
        const px = (grid.seg[i * 4] + grid.seg[i * 4 + 2]) / 2 - ox, py = (grid.seg[i * 4 + 1] + grid.seg[i * 4 + 3]) / 2 - oy;
        const t = px * dx + py * dy, u = px * nx + py * ny;
        if (t < -reach / 2 || t > l + reach / 2 || Math.abs(u) <= hw + 0.5 * pitch || Math.abs(u) > reach) return false;
        if (u > 0) left++;
        else right++;
        return false;
      });
      // even: east, else north
      const side = left !== right ? (left < right ? 1 : -1) : nx > 1e-6 || (Math.abs(nx) <= 1e-6 && ny > 0) ? 1 : -1;
      for (const k of inner) (normal[k * 2] = nx), (normal[k * 2 + 1] = ny), (prefer[k] = side);
    });
  });

  // per station its candidates, cheapest first apart from the last level's choice: [pill, dir, lifted, cost]
  type Cand = { pill: number; dir: number; slide: number; lifted: number; wrap: 0 | 1; cost: number; ax: number; ay: number };
  const cands: Cand[][] = Array.from({ length: n }, () => []);
  for (const k of order) {
    for (const pi of pillsOf[k]) {
      const p = d.pills[pi], r = R + p[4], ex = p[2] - p[0], ey = p[3] - p[1], el = Math.hypot(ex, ey);
      const axx = el > 1e-9 ? ex / el : 0, axy = el > 1e-9 ? ey / el : 0;
      for (let dir = 0; dir < 8; dir++) {
        const [cx, cy] = DIR8[dir];
        const along = cx * ex + cy * ey;
        // the outline's farthest point that way
        const bx = Math.abs(along) < 1e-9 ? (p[0] + p[2]) / 2 : along > 0 ? p[2] : p[0], by = Math.abs(along) < 1e-9 ? (p[1] + p[3]) / 2 : along > 0 ? p[3] : p[1];
        const ax = bx + cx * r, ay = by + cy * r;
        // off the pill's ends first (a round stop has none)
        const endness = el > 1e-9 ? Math.abs(cx * axx + cy * axy) : 1;
        let cost = endness > 0.9 ? 0 : endness > 0.5 ? 0.6 : 1.5;
        if (prefer[k]) {
          const u = cx * normal[k * 2] + cy * normal[k * 2 + 1];
          cost += Math.abs(u) < 0.3 ? 0.8 : Math.sign(u) === prefer[k] ? 0 : 1;
        }
        cost += dir === 0 ? 0 : dir === 4 ? 0.05 : 0.1 + dir * 0.001;
        cost += pi === pillsOf[k][0] ? 0 : 0.2;
        for (const wrap of width2[k] > 0 ? ([0, 1] as const) : ([0] as const))
          for (let lifted = 0; lifted <= (tier(k) ? 1 : LIFT_HUB); lifted++)
            for (const slide of dir % 2 ? [0] : [0, -1, 1])
              cands[k].push({ pill: pi, dir, slide, lifted, wrap, cost: cost + lifted * LIFT_COST + wrap * WRAP_COST + (slide ? SLIDE_COST : 0), ax, ay });
      }
    }
    cands[k].sort((a, b) => a.cost - b.cost);
  }

  const levels: number[] = [], out: LevelNames[] = [];
  let prev = new Map<number, Cand>();
  const T = [0, 0, 0, 0];
  for (let lv = LEVEL_LO; lv <= LEVEL_HI; lv++) {
    const P = LEVEL_BASE * LEVEL_STEP ** lv, s = P / pitch, boxes = new BoxGrid();
    const clear = CLEAR * pitch + 1 / s;
    const now = new Map<number, Cand>();
    const node: number[] = [], xs: number[] = [], ys: number[] = [], dirs: number[] = [], slides: number[] = [], lifts: number[] = [], wraps: number[] = [];
    for (const k of order) {
      const was = prev.get(k);
      // the last level's place is favoured, so names move little as the zoom changes
      const eff = (c: Cand) => c.cost - (c === was ? STAY : 0);
      const list = was ? [...cands[k]].sort((a, b) => eff(a) - eff(b)) : cands[k];
      for (const c of list) {
        const lift = c.lifted * LIFT * pitch;
        // zoomed in, a hub's lift is many px: a name that far off reads as no station's
        if (c.lifted > 1 && lift * s > LIFT_MAX_PX) continue;
        const B = nameBox(c.ax, c.ay, c.dir, c.slide, lift, c.wrap ? width2[k] : width[k], c.wrap ? h2 : h, s, T);
        // a lifted name's box runs back to its pill for the names and other pills, not for lines
        const Tn = lift ? [Math.min(T[0], c.ax), Math.min(T[1], c.ay), Math.max(T[2], c.ax), Math.max(T[3], c.ay)] : T;
        const px = [Tn[0] * s, Tn[1] * s, Tn[2] * s, Tn[3] * s];
        if (boxes.hits(px)) {
          why?.(P, k, c.dir, c.lifted, "name");
          continue;
        }
        // how far its own pills are from the name
        let own = Infinity;
        for (const pi of pillsOf[k]) {
          const p = d.pills[pi];
          own = Math.min(own, segBoxDist(p[0], p[1], p[2], p[3], B) - R - p[4]);
        }
        const blocked = grid.some(Tn, Math.max(clear, R + own + pitch), (i) => {
          const x0 = grid.seg[i * 4], y0 = grid.seg[i * 4 + 1], x1 = grid.seg[i * 4 + 2], y1 = grid.seg[i * 4 + 3];
          const kind = grid.kind[i];
          let hit: boolean;
          if (kind === 0) hit = segBoxDist(x0, y0, x1, y1, T) < clear;
          // its own pills and passages keep off the name itself; another station's off what was tested,
          // and its pills farther than its own
          else if (grid.owner[i] === k) hit = segBoxDist(x0, y0, x1, y1, B) < grid.r[i] - 1e-9;
          else {
            const dist = segBoxDist(x0, y0, x1, y1, Tn) - grid.r[i];
            hit = kind === 1 ? dist <= own : dist < 0;
          }
          if (hit) why?.(P, k, c.dir, c.lifted, kind === 0 ? "line" : grid.owner[i] === k ? "own" : `${kind === 1 ? "pill" : "passage"} of ${grid.owner[i]}`);
          return hit;
        });
        if (blocked) continue;
        boxes.add(px);
        now.set(k, c);
        node.push(k), xs.push(c.ax), ys.push(c.ay), dirs.push(c.dir), slides.push(c.slide), lifts.push(lift), wraps.push(c.wrap);
        break;
      }
    }
    levels.push(P);
    out.push({ node: Int32Array.from(node), x: Float64Array.from(xs), y: Float64Array.from(ys), dir: Uint8Array.from(dirs), slide: Int8Array.from(slides), lift: Float64Array.from(lifts), wrap: Uint8Array.from(wraps) });
    prev = now;
  }
  return { levels, at: out, text, width: Float32Array.from(width), width2: Float32Array.from(width2), fontPx, ms: performance.now() - t0 };
}

/** a name placed at level `li`, its `i`th, as a box in layout units at `s` CSS px a layout unit */
export function placedBox(plan: NamePlan, li: number, i: number, s: number): number[] {
  const L = plan.at[li], k = L.node[i], wrap = L.wrap[i];
  return nameBox(L.x[i], L.y[i], L.dir[i], L.slide[i], L.lift[i], wrap ? plan.width2[k] : plan.width[k], plan.fontPx * (wrap ? NAME_LINE + NAME_LINE2 : NAME_LINE), s);
}

/** the level drawn at a pitch of `pitchPx` CSS px: the last placed for at most that, -1 for none */
export function levelAt(plan: NamePlan, pitchPx: number): number {
  let i = -1;
  while (i + 1 < plan.levels.length && plan.levels[i + 1] <= pitchPx * (1 + 1e-9)) i++;
  return i;
}

export interface NameTheme { ink: string; halo: string; font: string }

/** the names over a painted drawing, at `scale` device px a layout unit, (cx, cy) in the middle */
export function paintNames(ctx: CanvasRenderingContext2D, plan: NamePlan, d: Drawing, w: number, h: number, cx: number, cy: number, scale: number, dpr: number, theme: NameTheme): void {
  const s = scale / dpr, li = levelAt(plan, d.pitch * s);
  if (li < 0) return;
  const L = plan.at[li], f = plan.fontPx * dpr;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.font = `${f}px ${theme.font}`;
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3 * dpr;
  const sx = (x: number) => w / 2 + (x - cx) * scale, sy = (y: number) => h / 2 - (y - cy) * scale;
  // per line of text: where, aligned which way
  const shown: [number, number, string, CanvasTextAlign][] = [];
  for (let i = 0; i < L.node.length; i++) {
    const k = L.node[i], b = placedBox(plan, li, i, s);
    const x0 = sx(b[0]), x1 = sx(b[2]), y0 = sy(b[3]), y1 = sy(b[1]);
    if (x1 < 0 || x0 > w || y1 < 0 || y0 > h) continue;
    // a name left of its pill is set flush right, above or below it centred
    const dx = DIR8[L.dir[i]][0], sl = Math.abs(dx) > 0.3 ? 0 : L.slide[i];
    const align: CanvasTextAlign = dx > 0.3 || sl > 0 ? "left" : dx < -0.3 || sl < 0 ? "right" : "center";
    const x = align === "left" ? x0 : align === "right" ? x1 : (x0 + x1) / 2;
    const parts = L.wrap[i] ? splitName(plan.text[k]) : null;
    if (parts) {
      const lead = (NAME_LINE / 2) * f;
      shown.push([x, y0 + lead, parts[0], align], [x, y0 + lead + NAME_LINE2 * f, parts[1], align]);
    } else shown.push([x, (y0 + y1) / 2, plan.text[k], align]);
  }
  ctx.strokeStyle = theme.halo;
  for (const [x, y, t, a] of shown) (ctx.textAlign = a), ctx.strokeText(t, x, y);
  ctx.fillStyle = theme.ink;
  for (const [x, y, t, a] of shown) (ctx.textAlign = a), ctx.fillText(t, x, y);
}
