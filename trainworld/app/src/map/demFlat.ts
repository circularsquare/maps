// T-1881: the hillshade drawn flat where the track is priced flat. The public Terrain Tiles are a
// surface model in much of the world (towers, raised roads and SRTM's speckle read as hills in dense
// centres); the terrain classes (game/terrain.ts) are the ground the price reads. A height tile from
// zoom FLAT_MIN_Z up, where it meets loaded class grids, goes to workers/demFlat.worker.ts with how
// flat each pixel's ground is, and comes back with the relief over flat ground taken out. Where no
// grid is loaded the tile is drawn as it comes.

import type { DemFlatIn, DemFlatOut } from "../workers/demFlat.worker";
import { zonesTouching } from "../game/cities";
import { zoneShown } from "../game/realms";
import { terrainGrids, terrainVersion, classAt, type ClassGrid } from "../game/terrain";

/** below this tile zoom (pixels 150 m and more) a tower block is a pixel or two and the classes are not
 * loaded for the whole view */
export const FLAT_MIN_Z = 10;
/** how long a tile waits for the class grids of the zones under it (ms); a zone with none never comes,
 * and one that comes later has the tile drawn again (`onRedraw`) */
const GRID_WAIT_MS = 1500;
const TILE_CACHE = 400;
/** `?rawdem=1`: the tiles as they come, for comparing */
const RAW = new URLSearchParams(location.search).has("rawdem");
/** tiles drawn flattened, and as they came (by reason), for checks */
export const demFlatStats = { flattened: 0, lowZoom: 0, noZone: 0, noGrid: 0, noFlat: 0, failed: 0 };
(globalThis as { twDemFlat?: typeof demFlatStats }).twDemFlat = demFlatStats;

const tileBytes = new Map<string, Promise<ArrayBuffer | null>>();

/** A Terrain Tiles PNG by key "z/x/y", cached (each is wanted again as a neighbour); null for none. */
export function demBytes(url: (z: number, x: number, y: number) => string, z: number, x: number, y: number): Promise<ArrayBuffer | null> {
  const n = 2 ** z;
  if (y < 0 || y >= n) return Promise.resolve(null);
  x = ((x % n) + n) % n;
  const key = `${z}/${x}/${y}`;
  let p = tileBytes.get(key);
  if (p) {
    tileBytes.delete(key);
    tileBytes.set(key, p);
    return p;
  }
  p = fetch(url(z, x, y)).then((r) => {
    if (!r.ok) throw new Error(`terrain tile ${r.status}`);
    return r.arrayBuffer();
  });
  p.catch(() => tileBytes.delete(key));
  tileBytes.set(key, p);
  if (tileBytes.size > TILE_CACHE) tileBytes.delete(tileBytes.keys().next().value!);
  return p;
}

const mercLat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;

function tileBox(z: number, x: number, y: number): [number, number, number, number] {
  const n = 2 ** z;
  return [x / n * 360 - 180, mercLat((y + 1) / n), (x + 1) / n * 360 - 180, mercLat(y / n)];
}

/** Resolves once every zone in `ids` has a grid, or after GRID_WAIT_MS, or on abort. */
function gridsFor(ids: string[], signal: AbortSignal): Promise<void> {
  const ready = () => ids.every((id) => terrainGrids.has(id));
  if (ready()) return Promise.resolve();
  return new Promise((resolve) => {
    let stop = () => {};
    const done = () => { stop(); clearTimeout(timer); signal.removeEventListener("abort", done); resolve(); };
    const timer = setTimeout(done, GRID_WAIT_MS);
    signal.addEventListener("abort", done);
    stop = terrainVersion.subscribe(() => { if (ready()) done(); });
  });
}

/** cells either side whose flatness a cell's is the mean of: a lone hilly cell in flat ground is mostly
 * drawn flat, and the edge of flat ground fades over about half a kilometre */
const MASK_REACH = 2;

/** How flat each pixel of the tile is (0-255): bilinear between the class cells' centres, each cell the
 * share of flat cells round it (MASK_REACH; cells no grid covers count as not flat); null when none of
 * the tile is flat. A function of the cells alone, so tiles agree at their edges. */
function flatMask(z: number, x: number, y: number, grids: ClassGrid[]): Uint8Array | null {
  const N = 256, n = 2 ** z;
  const out = new Uint8Array(N * N);
  let any = false;
  const isFlat = (c: number, r: number) => {
    for (const g of grids) {
      const cc = c - g.col0, rr = r - g.row0;
      if (cc >= 0 && rr >= 0 && cc < g.w && rr < g.h) return classAt(g, rr * g.w + cc) === 0 ? 1 : 0;
    }
    return 0;
  };
  const seen = new Map<number, number>();
  const side = 2 * MASK_REACH + 1;
  const flatAt = (c: number, r: number) => {
    const key = r * 1e7 + c;
    let v = seen.get(key);
    if (v === undefined) {
      v = 0;
      for (let dr = -MASK_REACH; dr <= MASK_REACH; dr++) for (let dc = -MASK_REACH; dc <= MASK_REACH; dc++) v += isFlat(c + dc, r + dr);
      v /= side * side;
      seen.set(key, v);
    }
    return v;
  };
  const per = grids[0].perDeg;
  const cols = new Float64Array(N);
  for (let i = 0; i < N; i++) cols[i] = ((x + (i + 0.5) / N) / n * 360) * per - 0.5; // from 180 W, cell centres at .5
  for (let j = 0; j < N; j++) {
    const fr = (mercLat((y + (j + 0.5) / N) / n) + 90) * per - 0.5;
    const r0 = Math.floor(fr), tr = fr - r0;
    let c0 = -1, a = 0, b = 0, c = 0, d = 0;
    for (let i = 0; i < N; i++) {
      const fc = cols[i], cc = Math.floor(fc), tc = fc - cc;
      if (cc !== c0) {
        c0 = cc;
        a = flatAt(cc, r0), b = flatAt(cc + 1, r0), c = flatAt(cc, r0 + 1), d = flatAt(cc + 1, r0 + 1);
      }
      const v = (a * (1 - tc) + b * tc) * (1 - tr) + (c * (1 - tc) + d * tc) * tr;
      if (v > 0) any = true;
      out[j * N + i] = Math.round(v * 255);
    }
  }
  return any ? out : null;
}

/** tiles drawn before every zone under them had its grid: "z/x/y" to the zones missing */
const provisional = new Map<string, string[]>();
let redraw: ((tiles: { z: number; x: number; y: number }[]) => void) | null = null;
let redrawSoon = 0;

/** `f` is called with the tiles to load again once a grid they waited for has come. */
export function onRedraw(f: (tiles: { z: number; x: number; y: number }[]) => void) {
  redraw = f;
}

terrainVersion.subscribe(() => {
  if (!provisional.size) return;
  window.clearTimeout(redrawSoon);
  redrawSoon = window.setTimeout(() => {
    const again: { z: number; x: number; y: number }[] = [];
    for (const [key, missing] of provisional)
      if (missing.some((id) => terrainGrids.has(id))) {
        provisional.delete(key);
        const [z, x, y] = key.split("/").map(Number);
        again.push({ z, x, y });
      }
    if (again.length) redraw?.(again);
  }, 200);
});

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, (r: DemFlatOut) => void>();

function post(m: DemFlatIn): Promise<DemFlatOut> {
  if (!worker) {
    worker = new Worker(new URL("../workers/demFlat.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<DemFlatOut>) => {
      const f = pending.get(e.data.id);
      pending.delete(e.data.id);
      f?.(e.data);
    };
  }
  return new Promise((resolve) => {
    pending.set(m.id, resolve);
    worker!.postMessage(m);
  });
}

/** The tile's PNG bytes as they come, or an ImageBitmap of its heights with the relief over flat ground
 * taken out. `view`: whether the clock worker is loading the class grids of the zones in view. */
export async function flattenDem(url: (z: number, x: number, y: number) => string, z: number, x: number, y: number,
  own: ArrayBuffer, signal: AbortSignal, view: boolean): Promise<ArrayBuffer | ImageBitmap> {
  if (z < FLAT_MIN_Z || RAW) return demFlatStats.lowZoom++, own;
  const [w, s, e, nn] = tileBox(z, x, y);
  // a zone outside the game's realms has no grid loaded (and no track)
  const ids = zonesTouching(w, s, e, nn).filter((c) => zoneShown(c.id)).map((c) => c.id as string);
  if (!ids.length) return demFlatStats.noZone++, own;
  if (view) await gridsFor(ids, signal);
  const missing = ids.filter((id) => !terrainGrids.has(id));
  if (missing.length && view) {
    provisional.set(`${z}/${x}/${y}`, missing);
    if (provisional.size > 2000) provisional.delete(provisional.keys().next().value!);
  } else provisional.delete(`${z}/${x}/${y}`);
  const grids = ids.map((id) => terrainGrids.get(id)).filter((g): g is ClassGrid => !!g);
  if (!grids.length || signal.aborted) return demFlatStats.noGrid++, own;
  const mask = flatMask(z, x, y, grids);
  if (!mask) return demFlatStats.noFlat++, own;
  const keys: string[] = [], bytes: (ArrayBuffer | null)[] = [];
  const n = 2 ** z;
  const got = await Promise.all([-1, 0, 1].flatMap((dy) => [-1, 0, 1].map((dx) => (dx || dy ? demBytes(url, z, x + dx, y + dy).catch(() => null) : Promise.resolve(own)))));
  for (let k = 0; k < 9; k++) {
    const dx = (k % 3) - 1, dy = Math.floor(k / 3) - 1;
    keys.push(`${z}/${(((x + dx) % n) + n) % n}/${y + dy}`);
    bytes.push(got[k]);
  }
  if (signal.aborted) return own;
  const lat = (s + nn) / 2;
  const pxM = 40075016.686 / n / 256 * Math.cos((lat * Math.PI) / 180);
  const r = await post({ id: nextId++, keys, bytes, mask, pxM });
  if (r.bitmap) demFlatStats.flattened++;
  else demFlatStats.failed++, console.warn(`terrain tile ${z}/${x}/${y} not flattened: ${r.error}`);
  return r.bitmap ?? own;
}
