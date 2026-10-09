// Commuter bubbles (T-097): per-cell counts summed into H3 parent areas, and the discs that draw
// them. Pure functions, used by the bubble worker for whole-city views and on the main thread for
// the small sets (a selection's far end, a station's catchment).

import { modeMix } from "../game/palette";
import { DISC, type Grouping } from "../workers/commutersProtocol";
import { MERC_K } from "./geo";

export interface Groups {
  /** group index per entry */
  group: Uint32Array;
  x: Float32Array;
  y: Float32Array;
  train: Float32Array;
  walk: Float32Array;
  drive: Float32Array;
}

/**
 * Sum cells into their groups: commuters by mode, and the commuter-weighted centre (so a bubble
 * sits where its people are, not at the hexagon's middle). `cells` and the three value arrays are
 * parallel (a subset of cells), or null for every cell with values per cell. Groups with nobody
 * are left out.
 */
export function aggregate(g: Grouping, cellX: Float32Array, cellY: Float32Array, cells: ArrayLike<number> | null, train: ArrayLike<number>, walk: ArrayLike<number>, drive: ArrayLike<number>): Groups {
  const n = g.n;
  const t = new Float64Array(n), w = new Float64Array(n), d = new Float64Array(n), sx = new Float64Array(n), sy = new Float64Array(n);
  const count = cells ? cells.length : cellX.length;
  for (let k = 0; k < count; k++) {
    const c = cells ? cells[k] : k;
    const a = train[k], b = walk[k], e = drive[k];
    const all = a + b + e;
    if (!(all > 0)) continue;
    const q = g.groupOf[c];
    t[q] += a;
    w[q] += b;
    d[q] += e;
    sx[q] += cellX[c] * all;
    sy[q] += cellY[c] * all;
  }
  let m = 0;
  for (let q = 0; q < n; q++) if (t[q] + w[q] + d[q] > 0) m++;
  const out: Groups = { group: new Uint32Array(m), x: new Float32Array(m), y: new Float32Array(m), train: new Float32Array(m), walk: new Float32Array(m), drive: new Float32Array(m) };
  let i = 0;
  for (let q = 0; q < n; q++) {
    const all = t[q] + w[q] + d[q];
    if (!(all > 0)) continue;
    out.group[i] = q;
    out.x[i] = sx[q] / all;
    out.y[i] = sy[q] / all;
    out.train[i] = t[q];
    out.walk[i] = w[q];
    out.drive[i] = d[q];
    i++;
  }
  return out;
}

/** Order of groups biggest first (smaller bubbles draw on top of bigger ones). */
export function biggestFirst(g: Groups): Uint32Array {
  const idx = Uint32Array.from({ length: g.group.length }, (_, i) => i);
  const tot = (i: number) => g.train[i] + g.walk[i] + g.drive[i];
  return idx.sort((a, b) => tot(b) - tot(a));
}

/** Keep groups in the given order (in place of `g`'s own). */
export function reorder(g: Groups, order: Uint32Array): Groups {
  const pick = <T extends Float32Array | Uint32Array>(a: T): T => {
    const b = new (a.constructor as any)(order.length) as T;
    for (let i = 0; i < order.length; i++) b[i] = a[order[i]];
    return b;
  };
  return { group: pick(g.group), x: pick(g.x), y: pick(g.y), train: pick(g.train), walk: pick(g.walk), drive: pick(g.drive) };
}

/**
 * Discs for groups, in their order: area `perCommuter` m² for each commuter (no minimum: nobody,
 * no disc), coloured by the mode mix (`modeMix`), or by `colour` when given.
 */
export function discs(g: Groups, perCommuter: number, alpha: number, edge: number, colour?: [number, number, number]): Float32Array {
  const n = g.group.length;
  const out = new Float32Array(n * DISC);
  for (let i = 0; i < n; i++) {
    const all = g.train[i] + g.walk[i] + g.drive[i];
    const c = colour ?? modeMix(g.train[i], g.walk[i], g.drive[i]);
    out[i * DISC] = g.x[i];
    out[i * DISC + 1] = g.y[i];
    out[i * DISC + 2] = Math.sqrt((all * perCommuter) / Math.PI) * MERC_K;
    out[i * DISC + 3] = c[0] / 255;
    out[i * DISC + 4] = c[1] / 255;
    out[i * DISC + 5] = c[2] / 255;
    out[i * DISC + 6] = alpha;
    out[i * DISC + 7] = edge;
  }
  return out;
}
