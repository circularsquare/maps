// Commuter bubbles (T-097): per-cell counts summed into H3 parent areas, and the discs that draw
// them. Pure functions, used by the bubble worker for whole-city views and on the main thread for
// the small sets (a selection's own bubbles and far end).

import { bubbleColour, purposeColour3, purposeSchemeOn } from "../game/palette";
import { DISC, type Grouping } from "../workers/commutersProtocol";
import { kAtY, type Anchor } from "./geo";

export interface Groups {
  /** group index per entry */
  group: Uint32Array;
  x: Float32Array;
  y: Float32Array;
  train: Float32Array;
  walk: Float32Array;
  drive: Float32Array;
  /** T-2215: trips no way serves (No route), all 0 where none was given */
  none: Float32Array;
}

/**
 * Sum cells into their groups: commuters by mode, and the commuter-weighted centre (so a bubble
 * sits where its people are, not at the hexagon's middle). `cells` and the value arrays are
 * parallel (a subset of cells), or null for every cell with values per cell. Groups with nobody
 * are left out.
 */
export function aggregate(g: Grouping, cellX: Float32Array, cellY: Float32Array, cells: ArrayLike<number> | null, train: ArrayLike<number>, walk: ArrayLike<number>, drive: ArrayLike<number>, none: ArrayLike<number> | null = null): Groups {
  const n = g.n;
  const t = new Float64Array(n), w = new Float64Array(n), d = new Float64Array(n), u = new Float64Array(n), sx = new Float64Array(n), sy = new Float64Array(n);
  const count = cells ? cells.length : cellX.length;
  for (let k = 0; k < count; k++) {
    const c = cells ? cells[k] : k;
    const a = train[k], b = walk[k], e = drive[k], f = none ? none[k] : 0;
    const all = a + b + e + f;
    if (!(all > 0)) continue;
    const q = g.groupOf[c];
    t[q] += a;
    w[q] += b;
    d[q] += e;
    u[q] += f;
    sx[q] += cellX[c] * all;
    sy[q] += cellY[c] * all;
  }
  let m = 0;
  for (let q = 0; q < n; q++) if (t[q] + w[q] + d[q] + u[q] > 0) m++;
  const out: Groups = { group: new Uint32Array(m), x: new Float32Array(m), y: new Float32Array(m), train: new Float32Array(m), walk: new Float32Array(m), drive: new Float32Array(m), none: new Float32Array(m) };
  let i = 0;
  for (let q = 0; q < n; q++) {
    const all = t[q] + w[q] + d[q] + u[q];
    if (!(all > 0)) continue;
    out.group[i] = q;
    out.x[i] = sx[q] / all;
    out.y[i] = sy[q] / all;
    out.train[i] = t[q];
    out.walk[i] = w[q];
    out.drive[i] = d[q];
    out.none[i] = u[q];
    i++;
  }
  return out;
}

/** T-3505: trip legs a day under which a bubble is not drawn: its trips go to a neighbour (`foldSmall`). */
export const BUBBLE_MIN = 1;
/** In a fold, a group whose trips go nowhere (the whole set is under `BUBBLE_MIN`). */
export const FOLD_DROP = 0xffffffff;

/**
 * T-3505: groups of `g` under `min` legs fold into the nearest group at or over it, which keeps its
 * place (a farmhouse's 0.3 trips a day join the hamlet's bubble beside it); with none at or over
 * it, into the busiest group, which is dropped too if it is still under. Returns the groups left, in
 * `g`'s order, and the fold as pairs of group indices (from, into; into `FOLD_DROP` for dropped
 * trips), null when nothing folds (and `g` itself is returned).
 */
export function foldSmall(g: Groups, min = BUBBLE_MIN): { groups: Groups; fold: Uint32Array | null } {
  const n = g.group.length;
  const tot = new Float64Array(n);
  let small = 0;
  for (let i = 0; i < n; i++) if ((tot[i] = g.train[i] + g.walk[i] + g.drive[i] + g.none[i]) < min) small++;
  if (!small) return { groups: g, fold: null };
  const big: number[] = [];
  for (let i = 0; i < n; i++) if (tot[i] >= min) big.push(i);
  let fold: Uint32Array, k = 0;
  if (!big.length) {
    let host = 0, sum = 0;
    for (let i = 0; i < n; i++) (sum += tot[i]), tot[i] > tot[host] && (host = i);
    const to = sum < min ? FOLD_DROP : g.group[host];
    fold = new Uint32Array(2 * (sum < min ? n : n - 1));
    for (let i = 0; i < n; i++) if (sum < min || i !== host) (fold[k++] = g.group[i]), (fold[k++] = to);
  } else {
    fold = new Uint32Array(2 * small);
    const near = nearestOf(g.x, g.y, big);
    for (let i = 0; i < n; i++) if (tot[i] < min) (fold[k++] = g.group[i]), (fold[k++] = g.group[near(g.x[i], g.y[i])]);
  }
  return { groups: applyFold(g, fold), fold };
}

/** The nearest of the points `pts` (indices into x, y) to a place: a grid of about one point a square,
 * searched ring by ring outward (lowest index on a tie). */
function nearestOf(x: Float32Array, y: Float32Array, pts: number[]): (px: number, py: number) => number {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of pts) (x0 = Math.min(x0, x[i])), (x1 = Math.max(x1, x[i])), (y0 = Math.min(y0, y[i])), (y1 = Math.max(y1, y[i]));
  const side = Math.max(Math.sqrt(((x1 - x0) * (y1 - y0)) / pts.length), Math.max(x1 - x0, y1 - y0) / 1024, 1e-6);
  const nx = Math.floor((x1 - x0) / side) + 1, ny = Math.floor((y1 - y0) / side) + 1;
  const cellOf = (v: number, v0: number, m: number) => Math.min(m - 1, Math.max(0, Math.floor((v - v0) / side)));
  const at = new Map<number, number[]>();
  for (const i of pts) {
    const key = cellOf(x[i], x0, nx) * ny + cellOf(y[i], y0, ny);
    const l = at.get(key);
    if (l) l.push(i);
    else at.set(key, [i]);
  }
  return (px, py) => {
    // a place outside the grid starts from the nearest square in it
    const cx = cellOf(px, x0, nx), cy = cellOf(py, y0, ny);
    let best = -1, bestD = Infinity;
    for (let r = 0; ; r++) {
      // a point in ring r or beyond is at least r - 1 squares away (more for a place outside the grid)
      if (best >= 0 && bestD <= ((r - 1) * side) ** 2) break;
      if (r > 0 && cx - r + 1 <= 0 && cx + r - 1 >= nx - 1 && cy - r + 1 <= 0 && cy + r - 1 >= ny - 1) break; // the last ring covered the grid
      for (let gx = cx - r; gx <= cx + r; gx++) {
        if (gx < 0 || gx >= nx) continue;
        const edge = gx === cx - r || gx === cx + r;
        for (let gy = cy - r; gy <= cy + r; gy += edge ? 1 : 2 * r || 1) {
          if (gy < 0 || gy >= ny) continue;
          const l = at.get(gx * ny + gy);
          if (!l) continue;
          for (const i of l) {
            const d = (x[i] - px) ** 2 + (y[i] - py) ** 2;
            if (d < bestD || (d === bestD && i < best)) (bestD = d), (best = i);
          }
        }
      }
    }
    return best;
  };
}

/** `g` with a fold (`foldSmall`'s pairs, by group index) applied: a folded group's figures added to
 * the group it goes into, which keeps its place; groups it names that `g` lacks are passed over.
 * Without places (a built level's figures) x and y come out 0. */
export function applyFold(g: Groups | Omit<Groups, "x" | "y">, fold: Uint32Array | null | undefined): Groups {
  if (!fold?.length) return "x" in g ? g : { ...g, x: new Float32Array(g.group.length), y: new Float32Array(g.group.length) };
  const gx = "x" in g ? g.x : null, gy = "y" in g ? g.y : null;
  const into = new Map<number, number>();
  for (let k = 0; k < fold.length; k += 2) into.set(fold[k], fold[k + 1]);
  const at = new Map<number, number>();
  const n = g.group.length;
  let m = 0;
  for (let i = 0; i < n; i++) if (!into.has(g.group[i])) at.set(g.group[i], m++);
  const out: Groups = { group: new Uint32Array(m), x: new Float32Array(m), y: new Float32Array(m), train: new Float32Array(m), walk: new Float32Array(m), drive: new Float32Array(m), none: new Float32Array(m) };
  for (let i = 0; i < n; i++) {
    const j = at.get(g.group[i]);
    if (j === undefined) continue;
    out.group[j] = g.group[i];
    if (gx && gy) (out.x[j] = gx[i]), (out.y[j] = gy[i]);
    (out.train[j] += g.train[i]), (out.walk[j] += g.walk[i]), (out.drive[j] += g.drive[i]), (out.none[j] += g.none[i]);
  }
  for (let i = 0; i < n; i++) {
    const to = into.get(g.group[i]);
    const j = to === undefined ? undefined : at.get(to);
    if (j === undefined) continue;
    (out.train[j] += g.train[i]), (out.walk[j] += g.walk[i]), (out.drive[j] += g.drive[i]), (out.none[j] += g.none[i]);
  }
  return out;
}

/** The groups folded into each group (`foldSmall`'s pairs turned around), for a selection to take them too. */
export function foldedInto(fold: Uint32Array | null | undefined): Map<number, number[]> {
  const m = new Map<number, number[]>();
  for (let k = 0; fold && k < fold.length; k += 2) {
    if (fold[k + 1] === FOLD_DROP) continue;
    const l = m.get(fold[k + 1]);
    if (l) l.push(fold[k]);
    else m.set(fold[k + 1], [fold[k]]);
  }
  return m;
}

/** Order of groups biggest first (smaller bubbles draw on top of bigger ones). */
export function biggestFirst(g: Groups): Uint32Array {
  const idx = Uint32Array.from({ length: g.group.length }, (_, i) => i);
  const tot = (i: number) => g.train[i] + g.walk[i] + g.drive[i] + g.none[i];
  return idx.sort((a, b) => tot(b) - tot(a));
}

/** Keep groups in the given order (in place of `g`'s own). */
export function reorder(g: Groups, order: Uint32Array): Groups {
  const pick = <T extends Float32Array | Uint32Array>(a: T): T => {
    const b = new (a.constructor as any)(order.length) as T;
    for (let i = 0; i < order.length; i++) b[i] = a[order[i]];
    return b;
  };
  return { group: pick(g.group), x: pick(g.x), y: pick(g.y), train: pick(g.train), walk: pick(g.walk), drive: pick(g.drive), none: pick(g.none) };
}

type Modes = { train: Float32Array; walk: Float32Array; drive: Float32Array; none?: Float32Array };
/** A level's trips by purpose per disc, in the disc order (T-2886): work, errands and leisure, everything else. */
type Purposes = { work: Float32Array; errands: Float32Array; other: Float32Array };
/** Colour built discs again under the scheme now chosen (T-804), in place: `levels` carry each
 * disc's trips by mode in the discs' order, and their smoothed `shade` where there is one (T-900). Purposes are
 * painted as they are, unsmoothed, so a bubble's colour says what its hover says (T-2911). */
export function recolour(levels: (Modes & { discs: Float32Array; shade?: Modes; purp?: Purposes })[]) {
  const byPurpose = purposeSchemeOn();
  for (const l of levels) {
    const m = l.shade ?? l;
    const p = byPurpose ? l.purp : undefined;
    const n = m.train.length;
    for (let i = 0; i < n; i++) {
      const c = p ? purposeColour3(p.work[i], p.other[i], p.errands[i]) : bubbleColour(m.train[i], m.walk[i], m.drive[i], m.none?.[i] ?? 0);
      l.discs[i * DISC + 3] = c[0] / 255;
      l.discs[i * DISC + 4] = c[1] / 255;
      l.discs[i * DISC + 5] = c[2] / 255;
    }
  }
}

/**
 * Discs for groups, in their order: area `perCommuter` m² for each commuter (no minimum: nobody,
 * no disc), coloured by the chosen scheme (`bubbleColour`), or by `colour` when given. `a` is the anchor the
 * groups' positions are relative to (T-317): a disc's radius is in metres of ground times k where
 * the disc is.
 */
export function discs(g: Groups, perCommuter: number, alpha: number, edge: number, a: Anchor, colour?: [number, number, number]): Float32Array {
  const n = g.group.length;
  const out = new Float32Array(n * DISC);
  for (let i = 0; i < n; i++) {
    const all = g.train[i] + g.walk[i] + g.drive[i] + g.none[i];
    const c = colour ?? bubbleColour(g.train[i], g.walk[i], g.drive[i], g.none[i]);
    out[i * DISC] = g.x[i];
    out[i * DISC + 1] = g.y[i];
    out[i * DISC + 2] = Math.sqrt((all * perCommuter) / Math.PI) * kAtY(g.y[i], a);
    out[i * DISC + 3] = c[0] / 255;
    out[i * DISC + 4] = c[1] / 255;
    out[i * DISC + 5] = c[2] / 255;
    out[i * DISC + 6] = alpha;
    out[i * DISC + 7] = edge;
  }
  return out;
}
