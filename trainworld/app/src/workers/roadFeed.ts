// The roads track crosses (T-1929, notes/T-1929.md): the clock worker fetches the basemap's zoom 14
// road tiles under the track being drawn and the blueprint, and puts each in the track model, which
// counts the crossings of every edge reaching into it and prices those at ground level. Like terrain,
// a preview never waits for them: it is priced on the roads in hand and asked again when more land.
// Construct waits for the tiles under the blueprint, so nothing is paid on a guess.

import { crossingWeight, decodeRoads } from "../map/roadTiles";

/** Tiles across the world at zoom 14, and tile units across a tile (sim `roads.rs`). */
const TILES = 1 << 14;
const EXTENT = 4096;
const PARALLEL = 6;
/** a tile download given up after this long, ms (tried again by the next call that needs it) */
const TILE_TIMEOUT_MS = 20_000;
/** A walk along a polyline looks at a tile every this many tile units (about 120 m in New York). */
const WALK = 256;

const worldX = (lng: number) => ((lng + 180) / 360) * TILES * EXTENT;
const worldY = (lat: number) => {
  const s = Math.min(0.9999, Math.max(-0.9999, Math.sin((lat * Math.PI) / 180)));
  return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * TILES * EXTENT;
};

/** The track model's side: adds one tile's lines, true if it was new. */
export type AddTile = (tx: number, ty: number, extent: number, pts: Int32Array, starts: Uint32Array, weights: Uint8Array) => boolean;

export class RoadFeed {
  private template: string | null = null;
  /** tiles in the model (an empty one where the server has none) */
  private inModel = new Set<number>();
  private fetches = new Map<number, Promise<void>>();
  private queue: (() => void)[] = [];
  private active = 0;
  private flushTimer = 0;
  private added = 0;

  /** `landed` runs once a burst of tiles is in, with how many. */
  constructor(private add: () => AddTile | null, private landed: (n: number) => void) {}

  setTemplate(t: string) {
    this.template = t;
  }

  /** A new track model: none of the tiles is in it. */
  reset() {
    this.inModel.clear();
  }

  /** The tiles a polyline of longitude, latitude pairs passes through, by key (sim `Roads::key`). */
  static tilesOf(ll: ArrayLike<number>): number[] {
    const keys = new Set<number>();
    const put = (x: number, y: number) => {
      const tx = Math.floor(x / EXTENT), ty = Math.floor(y / EXTENT);
      if (tx >= 0 && tx < TILES && ty >= 0 && ty < TILES) keys.add(tx * TILES + ty);
    };
    let px = NaN, py = NaN;
    for (let k = 0; k + 1 < ll.length; k += 2) {
      const x = worldX(ll[k]), y = worldY(ll[k + 1]);
      if (Number.isFinite(px)) {
        const n = Math.ceil(Math.hypot(x - px, y - py) / WALK);
        for (let j = 1; j < n; j++) put(px + ((x - px) * j) / n, py + ((y - py) * j) / n);
      }
      put(x, y);
      px = x;
      py = y;
    }
    return [...keys];
  }

  /** Fetch and put in the tiles of `keys` not in yet; resolves once each is in or has failed (a
   * failed one is tried again by the next call). Nothing to fetch from before the template comes. */
  ensure(keys: number[]): Promise<void> {
    const tpl = this.template;
    if (!tpl) return Promise.resolve();
    const waits: Promise<void>[] = [];
    for (const key of keys) {
      if (this.inModel.has(key)) continue;
      let p = this.fetches.get(key);
      if (!p) {
        p = this.slot().then(() => this.load(tpl, key)).finally(() => {
          this.fetches.delete(key);
          this.active--;
          this.queue.shift()?.();
        });
        this.fetches.set(key, p);
      }
      waits.push(p);
    }
    return Promise.all(waits).then(() => {});
  }

  /** How many of `keys` are not in the model (none before the template comes: there is nothing to price on). */
  missing(keys: number[]): number {
    return this.template ? keys.filter((k) => !this.inModel.has(k)).length : 0;
  }

  private slot(): Promise<void> {
    if (this.active < PARALLEL) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise((r) => this.queue.push(() => { this.active++; r(); }));
  }

  private async load(tpl: string, key: number) {
    const tx = Math.floor(key / TILES), ty = key % TILES;
    let buf: ArrayBuffer | null = null;
    try {
      // T-3196: a tile that never comes would hold its slot, and Construct waiting on it, for ever
      const r = await fetch(tpl.replace("{z}", "14").replace("{x}", String(tx)).replace("{y}", String(ty)), { signal: AbortSignal.timeout(TILE_TIMEOUT_MS) });
      if (r.status === 200) buf = await r.arrayBuffer();
      else if (r.status !== 404 && r.status !== 204) return;
    } catch {
      return;
    }
    const rl = buf ? decodeRoads(buf, crossingWeight) : { extent: EXTENT, lines: [], weights: [] };
    let n = 0;
    for (const l of rl.lines) n += l.length;
    const pts = new Int32Array(n), starts = new Uint32Array(rl.lines.length), weights = Uint8Array.from(rl.weights ?? []);
    let at = 0;
    rl.lines.forEach((l, i) => {
      starts[i] = at / 2;
      pts.set(l, at);
      at += l.length;
    });
    const add = this.add();
    if (!add) return;
    this.inModel.add(key);
    if (add(tx, ty, rl.extent, pts, starts, weights)) this.added++;
    clearTimeout(this.flushTimer);
    this.flushTimer = setTimeout(() => {
      const n = this.added;
      this.added = 0;
      if (n) this.landed(n);
    }, 30) as unknown as number;
  }
}
