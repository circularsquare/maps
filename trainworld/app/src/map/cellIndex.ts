// A grid over boxes for pointer queries (T-724): picking, the build tools' snap. Every edit makes a
// new network, so it is built in two passes over typed arrays (a hash of the occupied cells, then
// each cell's items), in idle time for small sets and at once for big ones: 5-10 ms for the real
// New York network's 100,000 segments.

/** cell sides, local units: a query uses the smallest whose side covers its width */
const CELL_SIDES = [256, 2048, 16384, 131072];
/** an item covering more cells than this is kept apart and handed to every query */
const BIG_CELLS = 16;
/** items from which a level wanted is built at once rather than in idle time (`CellLevels.for`) */
const BUILD_NOW = 4096;

const hash = (cx: number, cy: number) => Math.imul(cx, 0x9e3779b1) ^ Math.imul(cy, 0x85ebca77);

/** The items, by cell, of `box`: four numbers an item, two corners in any order. */
class CellIndex {
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

/** T-4284: a renumbering of items as runs, (from, count, to) triples ascending by `from`: item `from + k` is now
 * `to + k`, an item in no run gone. An edit's costs its runs, never a pass over every item. */
type RunMap = Int32Array;
/** item `i`'s number through `m`, -1 when it went */
function mapItem(m: RunMap, i: number): number {
  let lo = 0, hi = m.length / 3;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (m[mid * 3] <= i) lo = mid + 1; else hi = mid; }
  if (!lo) return -1;
  const k = (lo - 1) * 3;
  return i < m[k] + m[k + 1] ? m[k + 2] + (i - m[k]) : -1;
}
/** `a` then `b`: an item's number through both */
function composeRuns(a: RunMap, b: RunMap): RunMap {
  const out: number[] = [], nb = b.length / 3;
  for (let k = 0; k < a.length; k += 3) {
    const from = a[k], n = a[k + 1], mid = a[k + 2];
    // the runs of `b` over [mid, mid + n): the first that ends past mid, then on while they start before its end
    let lo = 0, hi = nb;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b[m * 3] + b[m * 3 + 1] <= mid) lo = m + 1; else hi = m; }
    for (let j = lo; j < nb && b[j * 3] < mid + n; j++) {
      const s = Math.max(mid, b[j * 3]), e = Math.min(mid + n, b[j * 3] + b[j * 3 + 1]);
      if (s >= e) continue;
      const f = from + (s - mid), t = b[j * 3 + 2] + (s - b[j * 3]), c = e - s, q = out.length - 3;
      // (a run that goes on where the last ended, both ways, extends it)
      if (q >= 0 && out[q] + out[q + 1] === f && out[q + 2] + out[q + 1] === t) out[q + 1] += c;
      else out.push(f, c, t);
    }
  }
  return Int32Array.from(out);
}

/** A grid carried over edits (T-1030): an older grid with its items renumbered (`map`, T-4284 as runs), and a small
 * grid of the items added since (`extra`, their numbers in `ids`). */
class PatchedIndex {
  constructor(readonly base: CellIndex, readonly map: RunMap, readonly extra: CellIndex | null, readonly ids: Int32Array) {}
  near(x: number, y: number, reach: number, take: (i: number) => void) {
    const m = this.map, ids = this.ids;
    this.base.near(x, y, reach, (i) => {
      const j = mapItem(m, i);
      if (j >= 0) take(j);
    });
    this.extra?.near(x, y, reach, (k) => take(ids[k]));
  }
}
/** a grid carried past this many runs is built anew instead */
const RUNS_MAX = 4096;
type Grid = CellIndex | PatchedIndex;

/** Runs `job` when the main thread is idle: a build takes 5-20 ms, which a pointer move never waits for. */
export function whenIdle(job: () => void) {
  if (typeof requestIdleCallback === "function") requestIdleCallback(job, { timeout: 1000 });
  else setTimeout(job, 50);
}

/** How a set of items came from an older one over an edit (T-1030), for `CellLevels.carried`. */
interface CarriedFrom {
  old: CellLevels;
  runs: Int32Array;
  itemBox?: (i: number, out: Float64Array, at: number) => void;
  /** the old items' new numbers (`RunMap`), and the new items, worked out when first needed */
  step?: RunMap;
  fresh?: number[];
}

/** The CellIndex levels of one set of boxes (`box()` gives them), each built in idle time once a
 * query has wanted it, or carried over from the set before an edit (`carried`). */
export class CellLevels {
  private levels: (Grid | "asked" | undefined)[] = [];
  private boxes: ArrayLike<number> | null = null;
  private from: CarriedFrom | null = null;
  /** `now`: each level wanted is built at once, whatever the count (a small set laid over a big one, map/snapIndex.ts) */
  constructor(private box: () => ArrayLike<number>, readonly count: number, private now = false) {}
  /** The level for a query of this reach, or null to scan everything: when the reach is too wide
   * for any level, or the level is not built yet (it is then built when the thread is idle). */
  for(reach: number): Grid | null {
    const L = CELL_SIDES.findIndex((side) => 2 * reach <= side);
    if (L < 0 || !(reach >= 0)) return null;
    const had = this.level(L);
    if (had) return had;
    // T-2404: a big set's scan costs about what building its grid does, and a pointer move scanning
    // it each time leaves the thread no idle time to build in (T-1861): built now
    if (this.count >= BUILD_NOW || this.now) return (this.levels[L] = new CellIndex((this.boxes ??= this.box()), this.count, CELL_SIDES[L]));
    if (!this.levels[L]) {
      this.levels[L] = "asked";
      whenIdle(() => (this.levels[L] = new CellIndex((this.boxes ??= this.box()), this.count, CELL_SIDES[L])));
    }
    return null;
  }

  /** Level L if built or carried over (carrying it now if the older set has it), else null. */
  private level(L: number): Grid | null {
    const had = this.levels[L];
    if (had && had !== "asked") return had;
    if (had || !this.from) return null;
    const g = this.carry(L);
    if (g) this.levels[L] = g;
    return g;
  }

  /**
   * T-1030: these items' grids carried over from `old`'s, the set before an edit, instead of built
   * again: `runs` (start, count, old start or -1, as `NetPatch`) make the new items from the old;
   * `itemBox(i)` writes new item i's box where `box()` would build all of them. Each level is carried
   * when a query first asks for it; one whose added items pass a share of the whole is built anew.
   */
  static carried(old: CellLevels, runs: Int32Array, box: () => ArrayLike<number>, count: number, itemBox?: (i: number, out: Float64Array, at: number) => void): CellLevels {
    const out = new CellLevels(box, count);
    // edits with no query between: the older set's levels are carried now, so only one older set
    // (and its boxes) is ever held
    if (old.from) {
      for (let L = 0; L < CELL_SIDES.length; L++) if (old.from.old.levels[L] instanceof CellIndex || old.from.old.levels[L] instanceof PatchedIndex) old.level(L);
      old.from = null;
    }
    if (old.levels.some((g) => g && g !== "asked")) out.from = { old, runs, itemBox };
    return out;
  }

  private carry(L: number): Grid | null {
    const f = this.from!;
    const g = f.old.levels[L];
    if (!g || g === "asked") return null;
    if (!f.step) {
      const kept: [number, number, number][] = [], fresh: number[] = [], r = f.runs;
      let added = 0;
      for (let k = 0; k < r.length; k += 3) {
        const start = r[k], n = r[k + 1], from = r[k + 2];
        if (from >= 0) kept.push([from, n, start]);
        else if ((added += n) <= Math.max(4096, this.count / 8)) for (let j = 0; j < n; j++) fresh.push(start + j);
      }
      kept.sort((a, b) => a[0] - b[0]);
      f.step = Int32Array.from(kept.flat());
      // (more new items than a carry takes: no level is carried)
      f.fresh = added <= Math.max(4096, this.count / 8) ? fresh : undefined;
    }
    const step = f.step, fresh = f.fresh;
    if (!fresh) return null;
    const base = g instanceof PatchedIndex ? g.base : g;
    const ids: number[] = [];
    if (g instanceof PatchedIndex) for (const i of g.ids) { const j = mapItem(step, i); if (j >= 0) ids.push(j); }
    for (const i of fresh) ids.push(i);
    if (ids.length > Math.max(4096, this.count / 8)) return null;
    const map = g instanceof PatchedIndex ? composeRuns(g.map, step) : step;
    if (map.length / 3 > RUNS_MAX) return null;
    const all = f.itemBox ? null : (this.boxes ??= this.box());
    const b = new Float64Array(ids.length * 4);
    ids.forEach((i, k) => {
      if (all) for (let c = 0; c < 4; c++) b[k * 4 + c] = all[i * 4 + c];
      else f.itemBox!(i, b, k * 4);
    });
    return new PatchedIndex(base, map, ids.length ? new CellIndex(b, ids.length, CELL_SIDES[L]) : null, Int32Array.from(ids));
  }

  /** `?patch=check`: every level that can be carried over, carried, answering as grids built from
   * scratch would at the boxes of up to `samples` items spread over all; the levels that do not. */
  verify(samples = 64): string[] {
    const bad: string[] = [];
    for (let L = 0; L < CELL_SIDES.length; L++) {
      const g = this.level(L);
      if (!(g instanceof PatchedIndex)) continue;
      const all = (this.boxes ??= this.box());
      const fresh = new CellIndex(all, this.count, CELL_SIDES[L]);
      for (let s = 0; s < samples && this.count; s++) {
        const i = Math.floor((s * this.count) / samples);
        const x = (all[i * 4] + all[i * 4 + 2]) / 2, y = (all[i * 4 + 1] + all[i * 4 + 3]) / 2, reach = CELL_SIDES[L] / 4;
        const a = new Set<number>(), b = new Set<number>();
        // the same cells hold the same items, so the answers match exactly
        g.near(x, y, reach, (j) => a.add(j));
        fresh.near(x, y, reach, (j) => b.add(j));
        if (a.size !== b.size || [...b].some((j) => !a.has(j))) { bad.push(`level ${L}`); break; }
      }
    }
    return bad;
  }
}
