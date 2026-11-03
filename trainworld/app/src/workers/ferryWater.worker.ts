// The ferry tool's water (T-2192, notes/T-2177.md 3.3): fetches the basemap's zoom 12 tiles itself, as
// `map/busRoads.ts` fetches roads, keeps them decoded, and answers port snaps and line routes with
// `ferry/route.ts`. A line request names its legs' ends; each leg's search is kept, so moving one point
// searches only the legs it touches. Requests are taken in order; a line request with the `key` of a newer
// one still waiting is dropped unanswered but for `dropped`.

import { EMPTY_TILE, decodeWater, type WaterTile } from "../ferry/waterTiles";
import {
  assembleLine, buildRaster, probe, routeLeg, snapPort, tilesOf,
  type Box, type LegPart, type LegRoute, type LegStats, type LineRoute, type LinePoint, type LonLat, type PortSnap, type Raster, type TileLoad,
} from "../ferry/route";

export type FerryIn =
  | { kind: "template"; url: string }
  | { kind: "port"; id: number; at: LonLat }
  | { kind: "probe"; id: number; at: LonLat }
  | { kind: "line"; id: number; points: LinePoint[]; key?: string };

export type FerryOut =
  | { kind: "port"; id: number; snap: PortSnap }
  | { kind: "probe"; id: number; probe: { water: boolean; shore: boolean } | null }
  /** `stats`: each leg searched for this answer (cached legs have none); `ms` in all, tiles' fetch included */
  | { kind: "line"; id: number; route: LineRoute; stats: (LegStats | null)[]; ms: number }
  | { kind: "dropped"; id: number };

const PARALLEL = 8;
const KEEP_TILES = 160;
const KEEP_LEGS = 200;
/** rasters kept for rounding lines again (a long leg's is about 7 MB, its ends' 3 MB each) */
const KEEP_RASTERS = 9;
const RETRY_MS = 15000;

let template: string | null = null;
let waitTemplate: (() => void)[] = [];
const tiles = new Map<string, { tile: WaterTile | null; at: number } | Promise<WaterTile | null>>();

async function fetchTile(x: number, y: number, z: number): Promise<WaterTile | null> {
  if (!template) await new Promise<void>((res) => waitTemplate.push(res));
  try {
    const r = await fetch(template!.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y)));
    if (r.status === 200) return decodeWater(await r.arrayBuffer());
    if (r.status === 204 || r.status === 404) return EMPTY_TILE;
  } catch {
    /* counts as failed */
  }
  return null;
}

const load: TileLoad = async (want) => {
  const got = new Map<string, WaterTile>();
  let i = 0, failed = false;
  const one = async (): Promise<void> => {
    while (i < want.length) {
      const [x, y, z] = want[i++];
      const key = `${z}/${x}/${y}`;
      let t = tiles.get(key);
      if (t && !(t instanceof Promise) && !t.tile && Date.now() - t.at > RETRY_MS) t = undefined;
      if (!t) {
        const p = fetchTile(x, y, z);
        tiles.set(key, p);
        const tile = await p;
        tiles.set(key, { tile, at: Date.now() });
        t = tiles.get(key);
      } else if (t instanceof Promise) {
        await t;
        t = tiles.get(key);
      } else {
        // used again: newest in the map's order
        tiles.delete(key);
        tiles.set(key, t);
      }
      const tile = t && !(t instanceof Promise) ? t.tile : null;
      if (tile) got.set(key, tile);
      else failed = true;
    }
  };
  await Promise.all(Array.from({ length: Math.min(PARALLEL, want.length) }, one));
  for (const k of tiles.keys()) {
    if (tiles.size <= KEEP_TILES) break;
    if (!(tiles.get(k) instanceof Promise) && !got.has(k)) tiles.delete(k);
  }
  if (failed) return null;
  return (x, y, z) => got.get(`${z}/${x}/${y}`) ?? EMPTY_TILE;
};

type Leg = Exclude<LegRoute, { ok: true }> | { ok: true; pieces: { ll: number[]; box: Box }[] };
const legs = new Map<string, Leg>();
const rasters = new Map<string, Raster>();
const legKey = (a: LonLat, b: LonLat) => `${a[0]},${a[1]}|${b[0]},${b[1]}`;

function keepRaster(key: string, r: Raster) {
  rasters.delete(key);
  rasters.set(key, r);
  while (rasters.size > KEEP_RASTERS) rasters.delete(rasters.keys().next().value!);
}

async function line(points: LinePoint[]): Promise<{ route: LineRoute; stats: (LegStats | null)[] }> {
  const parts: LegPart[] = [], stats: (LegStats | null)[] = [];
  for (let k = 0; k + 1 < points.length; k++) {
    const a = points[k].at, b = points[k + 1].at, key = legKey(a, b);
    let leg = legs.get(key);
    let st: LegStats | null = null;
    if (!leg) {
      const r = await routeLeg(load, a, b);
      st = r.stats ?? null;
      if (r.ok) {
        leg = { ok: true, pieces: r.pieces.map((p) => ({ ll: p.ll, box: p.box })) };
        r.pieces.forEach((p, j) => keepRaster(`${key}#${j}`, p.raster));
      } else leg = r;
      // a leg whose tiles failed is searched again next time
      if (leg.ok || leg.reason !== "tiles") {
        legs.set(key, leg);
        if (legs.size > KEEP_LEGS) legs.delete(legs.keys().next().value!);
      }
    }
    stats.push(st);
    if (!leg.ok) {
      parts.push({ ok: false, reason: leg.reason });
      continue;
    }
    const pieces: { ll: number[]; raster: Raster }[] = [];
    for (const [j, p] of leg.pieces.entries()) {
      let raster = rasters.get(`${key}#${j}`);
      if (!raster) {
        const get = await load(tilesOf(p.box));
        if (!get) break;
        raster = buildRaster(get, p.box);
      }
      keepRaster(`${key}#${j}`, raster);
      pieces.push({ ll: p.ll, raster });
    }
    parts.push(pieces.length === leg.pieces.length ? { ok: true, pieces } : { ok: false, reason: "tiles" });
  }
  return { route: assembleLine(points, parts), stats };
}

const queue: FerryIn[] = [];
let running = false;
const post = (m: FerryOut) => (self as unknown as Worker).postMessage(m);

async function run() {
  if (running) return;
  running = true;
  // let every message already sent arrive first, so superseded ones can be dropped
  await new Promise((res) => setTimeout(res, 0));
  while (queue.length) {
    const m = queue.shift()!;
    if (m.kind === "template") continue;
    try {
      if (m.kind === "line") {
        if (m.key && queue.some((q) => q.kind === "line" && q.key === m.key)) {
          post({ kind: "dropped", id: m.id });
          continue;
        }
        const t0 = performance.now();
        const { route, stats } = await line(m.points);
        post({ kind: "line", id: m.id, route, stats, ms: performance.now() - t0 });
      } else if (m.kind === "port") post({ kind: "port", id: m.id, snap: await snapPort(load, m.at) });
      else if (m.kind === "probe") post({ kind: "probe", id: m.id, probe: await probe(load, m.at) });
    } catch (e) {
      console.error("ferry water", e);
      post({ kind: "dropped", id: m.id });
    }
  }
  running = false;
}

self.onmessage = (e: MessageEvent<FerryIn>) => {
  const m = e.data;
  if (m.kind === "template") {
    template = m.url;
    for (const w of waitTemplate) w();
    waitTemplate = [];
    return;
  }
  queue.push(m);
  void run();
};
