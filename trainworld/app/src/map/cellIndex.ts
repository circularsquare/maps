// A grid over boxes for pointer queries (T-724): picking, the build tools' snap. Every edit makes a
// new network, so it is built in two passes over typed arrays (a hash of the occupied cells, then
// each cell's items) and in idle time: 5-10 ms for the real New York network's 100,000 segments.

/** cell sides, local units: a query uses the smallest whose side covers its width */
export const CELL_SIDES = [256, 2048, 16384, 131072];
/** an item covering more cells than this is kept apart and handed to every query */
const BIG_CELLS = 16;

const hash = (cx: number, cy: number) => Math.imul(cx, 0x9e3779b1) ^ Math.imul(cy, 0x85ebca77);

/** The items, by cell, of `box`: four numbers an item, two corners in any order. */
export class CellIndex {
  private cx: Int32Array;
  private cy: Int32Array;
  private used: Uint8Array;
  private start: Int32Array;
  private items: Int32Array;
  private big: Int32Array;
  private mask: number;

  constructor(box: ArrayLike<number>, count: number, readonly side: number) {
    // pass one: occupied cells and how many items each holds
    let entries = 0;
    const range = new Int32Array(count * 4), isBig = new Uint8Array(count);
    let bigs = 0;
    for (let i = 0; i < count; i++) {
      const ax = box[i * 4], ay = box[i * 4 + 1], bx = box[i * 4 + 2], by = box[i * 4 + 3];
      const x0 = Math.floor(Math.min(ax, bx) / side), x1 = Math.floor(Math.max(ax, bx) / side);
      const y0 = Math.floor(Math.min(ay, by) / side), y1 = Math.floor(Math.max(ay, by) / side);
      const cells = (x1 - x0 + 1) * (y1 - y0 + 1);
      if (!(cells <= BIG_CELLS)) {
        isBig[i] = 1;
        bigs++;
        continue;
      }
      range[i * 4] = x0, range[i * 4 + 1] = y0, range[i * 4 + 2] = x1, range[i * 4 + 3] = y1;
      entries += cells;
    }
    let size = 16;
    while (size < entries * 1.5) size *= 2;
    this.mask = size - 1;
    this.cx = new Int32Array(size);
    this.cy = new Int32Array(size);
    this.used = new Uint8Array(size);
    const counts = new Int32Array(size + 1);
    const slotOf = new Int32Array(entries);
    let e = 0;
    for (let i = 0; i < count; i++) {
      if (isBig[i]) continue;
      for (let x = range[i * 4]; x <= range[i * 4 + 2]; x++)
        for (let y = range[i * 4 + 1]; y <= range[i * 4 + 3]; y++) {
          const s = this.insert(x, y);
          counts[s + 1]++;
          slotOf[e++] = s;
        }
    }
    // pass two: each cell's items, in index order
    for (let s = 0; s < size; s++) counts[s + 1] += counts[s];
    this.start = counts;
    const fill = counts.slice(0, size);
    this.items = new Int32Array(entries);
    e = 0;
    for (let i = 0; i < count; i++) {
      if (isBig[i]) continue;
      const cells = (range[i * 4 + 2] - range[i * 4] + 1) * (range[i * 4 + 3] - range[i * 4 + 1] + 1);
      for (let c = 0; c < cells; c++) this.items[fill[slotOf[e++]]++] = i;
    }
    this.big = new Int32Array(bigs);
    for (let i = 0, k = 0; i < count; i++) if (isBig[i]) this.big[k++] = i;
  }

  private insert(x: number, y: number): number {
    for (let s = hash(x, y) & this.mask; ; s = (s + 1) & this.mask) {
      if (!this.used[s]) {
        this.used[s] = 1;
        this.cx[s] = x;
        this.cy[s] = y;
        return s;
      }
      if (this.cx[s] === x && this.cy[s] === y) return s;
    }
  }

  private find(x: number, y: number): number {
    for (let s = hash(x, y) & this.mask; this.used[s]; s = (s + 1) & this.mask) if (this.cx[s] === x && this.cy[s] === y) return s;
    return -1;
  }

  /** Calls `take` with every item whose box comes within `reach` of (x, y), and perhaps others;
   * an item may come more than once. */
  near(x: number, y: number, reach: number, take: (i: number) => void) {
    for (let k = 0; k < this.big.length; k++) take(this.big[k]);
    const side = this.side;
    for (let cx = Math.floor((x - reach) / side); cx <= Math.floor((x + reach) / side); cx++)
      for (let cy = Math.floor((y - reach) / side); cy <= Math.floor((y + reach) / side); cy++) {
        const s = this.find(cx, cy);
        if (s < 0) continue;
        for (let k = this.start[s]; k < this.start[s + 1]; k++) take(this.items[k]);
      }
  }
}

/** Runs `job` when the main thread is idle: a build takes 5-20 ms, which a pointer move never waits for. */
export function whenIdle(job: () => void) {
  if (typeof requestIdleCallback === "function") requestIdleCallback(job, { timeout: 1000 });
  else setTimeout(job, 50);
}

/** The CellIndex levels of one set of boxes (`box()` gives them), each built in idle time once a
 * query has wanted it. */
export class CellLevels {
  private levels: (CellIndex | "asked" | undefined)[] = [];
  private boxes: ArrayLike<number> | null = null;
  constructor(private box: () => ArrayLike<number>, readonly count: number) {}
  /** The level for a query of this reach, or null to scan everything: when the reach is too wide
   * for any level, or the level is not built yet (it is then built when the thread is idle). */
  for(reach: number): CellIndex | null {
    const L = CELL_SIDES.findIndex((side) => 2 * reach <= side);
    if (L < 0 || !(reach >= 0)) return null;
    const had = this.levels[L];
    if (had instanceof CellIndex) return had;
    if (!had) {
      this.levels[L] = "asked";
      whenIdle(() => (this.levels[L] = new CellIndex((this.boxes ??= this.box()), this.count, CELL_SIDES[L])));
    }
    return null;
  }
}
