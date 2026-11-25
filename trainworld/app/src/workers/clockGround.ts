// The ground the model needs, fetched on demand: water masks, terrain classes, street speeds, the world's water, road crossings, and the masks an edit or a load waits for (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { NEW_GAME_FRAME, zoneBox, zonesTouching } from "../game/cities";
import { blockFile, indexHas, parseBlockIndex, type BlockIndex } from "../game/terrainBlocks";
import { M_PER_DEG, makeFrame } from "../game/frame";
import { fetchRange } from "./rangeFetch";
import { trackFrameKind, SAVE_DAMAGED } from "./saveFile";
import { RoadFeed } from "./roadFeed";
import { scheduleSnapshot } from "./clockDemand";
import { sendStateOutside, stale } from "./clockFollowUp";
import { api, emptyTrack, frame, midLoad, putFrame, send, syncFrame } from "./clockState";

// ---------------------------------------------------------------- water masks on demand (T-302)
//
// A zone's mask is fetched only when the view comes near its box or track is drawn or loaded within
// it (docs/zones.md 6). The rule that keeps building exact: every zone whose box meets existing track
// has its mask installed. An edit or preview that would make new track is first worked out, and
// if its geometry meets a zone whose mask is not in yet, it waits for the mask and is worked out
// again, so ground over water is never accepted because a mask was late.

/** the packs' folder (the init message's `pack` names one pack in it) */
export let packBase = "";
export interface WaterRaster { cell: number; x0: number; y0: number; w: number; h: number; rows: Uint32Array; runs: Uint16Array; lon: number; lat: number }
const waterFetches = new Map<string, Promise<WaterRaster | null>>();
/** zones whose masks are in the track model (a zone with no mask counts once its head said so) */
export const masksIn = new Set<string>();
/** a point of track this near a zone's box needs its mask: the sampled geometry's spacing to spare */
const TRACK_MARGIN_M = 2_000;
/** the view this near a zone's box fetches its mask ahead, when the view is under VIEW_SPAN_M wide */
export const VIEW_MARGIN_M = 20_000;
export const VIEW_SPAN_M = 400_000;

/** masks fetched at once at most: a game over a continent meets hundreds of zones, and that many fetches at once fail
 * in the browser (ERR_INSUFFICIENT_RESOURCES, T-3463, as the diagram's masks did in T-3448) */
const WATER_FETCHES = 32;
let waterBusy = 0;
const waterQueue: (() => void)[] = [];
async function inWaterSlot<T>(run: () => Promise<T>): Promise<T> {
  if (waterBusy >= WATER_FETCHES) await new Promise<void>((r) => waterQueue.push(r)); // a finished fetch hands its slot over
  else waterBusy++;
  try {
    return await run();
  } finally {
    const next = waterQueue.shift();
    if (next) next(); else waterBusy--;
  }
}

export function fetchWater(id: string): Promise<WaterRaster | null> {
  let p = waterFetches.get(id);
  if (!p) {
    const url = new URL(id, packBase).href;
    const once = async (): Promise<WaterRaster | null> => {
      const res = await fetch(url + ".json", { signal: AbortSignal.timeout(GROUND_TIMEOUT_MS) });
      if (res.status === 404) return null;
      const head = await res.json();
      const row = head.arrays.find((a: any) => a.name === "water_row");
      const xs = head.arrays.find((a: any) => a.name === "water_x");
      if (!head.water || !row || !xs) return null;
      const base = row.offset;
      const buf = await fetchRange(url + ".bin", base, xs.offset + xs.count * 2);
      const w = head.water;
      return {
        cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1],
        rows: new Uint32Array(buf, row.offset - base, row.count), runs: new Uint16Array(buf, xs.offset - base, xs.count),
        lon: head.origin.lon, lat: head.origin.lat,
      };
    };
    // one more try a second later, in the same slot
    p = inWaterSlot(() => once().catch(() => new Promise<void>((r) => setTimeout(r, 1000)).then(once)));
    waterFetches.set(id, p);
    watchGround("w:" + id, p);
    // a failed fetch is tried again by the next edit or view that needs it, or by `retryGround`
    p.catch((err) => {
      if (waterFetches.get(id) === p) waterFetches.delete(id);
      send({ kind: "error", message: `water mask for ${id} not loaded: ${err}` }); // i18n-ignore
    });
  }
  return p;
}

/** Install the masks of `ids` (fetching those not fetched); resolves once each is in or failed. The
 * raster is laid out in the pack's own frame; the track model puts it in the game's (T-125). `streets`: the zones'
 * street speeds come too (T-3686: not for a load's masks, which are every zone the track meets, hundreds on a
 * continent at 8 fetches each beside the packs; the loaded network's buses take theirs in `masksForLoaded`). */
export async function ensureMasks(ids: Iterable<string>, streets = true): Promise<number> {
  const all = [...new Set(ids)];
  if (streets) void ensureStreets(all);
  const todo = all.filter((id) => !masksIn.has(id));
  if (!todo.length) return 0;
  const got = await Promise.all(todo.map((id) => fetchWater(id).then((r) => ({ r }), () => null)));
  let added = 0;
  todo.forEach((id, k) => {
    const g = got[k];
    if (!g || masksIn.has(id)) return;
    masksIn.add(id);
    const r = g.r;
    if (r) { api!.add_pack_water(r.cell, r.x0, r.y0, r.w, r.h, r.rows, r.runs, r.lon, r.lat); added++; }
  });
  return added;
}

// ---------------------------------------------------------------- terrain classes (T-1270, T-2901)
//
// One file per degree block of land (`terrain-world/<la>_<lo>.terrain`, pipeline/world_terrain.py), listed
// in the folder's `index.bin`: a block not listed is sea, known flat, with nothing to fetch. A listed block
// that cannot be had is a failure (retried, the status tag says so), never flat ground. Blocks are fetched
// under what the track model prices (the blocks it asks for, `take_terrain_wanted`) and what the camera
// reaches. Terrain only prices track, so no preview or edit waits for it: track drawn before a block
// lands is priced on flat ground and priced again when it does. Construct waits for the blocks under what
// it builds, and is refused while one of them could not be had (T-2613), so nothing is paid on a guess.

/** where the blocks are (the init message's, game/dataBase.ts) */
export let terrainBase = "";
/** the tint's copies live on the page; it keeps at most this many blocks, the camera's newest */
const TINT_BLOCKS = 200;
/** the blocks kept in the track model beyond those an edge stands on or the view has (230 KB each) */
const MODEL_BLOCKS = 300;
/** a game opens without waiting longer than this for its terrain; what is late reprices when it lands */
const LOAD_TERRAIN_WAIT_MS = 15_000;

let terrainIndexFetch: Promise<ArrayBuffer> | null = null;
/** the current track model has the index */
export let terrainIndexIn = false;
let terrainIndex: BlockIndex | null = null;
const blockFetches = new Map<number, Promise<ArrayBuffer>>();
/** blocks in the track model, oldest wanted first */
export const blocksIn = new Set<number>();
/** blocks the page has for the tint, oldest wanted first */
export const tintSent = new Set<number>();
/** the blocks the view reaches (the camera's rule, as for masks: only near its centre when it is wide) */
export let viewBlocks = new Set<number>();

const blockListed = (key: number) => !!terrainIndex && indexHas(terrainIndex, key);

export function terrainIndexIntoModel(): Promise<void> {
  if (terrainIndexIn) return Promise.resolve();
  if (!terrainIndexFetch) {
    const p = (async () => {
      const res = await fetch(new URL("index.bin", terrainBase).href, { signal: AbortSignal.timeout(GROUND_TIMEOUT_MS) });
      if (!res.ok) throw new Error(`${res.status}`);
      return await res.arrayBuffer();
    })();
    terrainIndexFetch = p;
    watchGround("index", p);
    // tried again by the next need for it or by `retryGround`; Construct waits for it
    p.catch(() => {
      if (terrainIndexFetch === p) terrainIndexFetch = null;
    });
  }
  return terrainIndexFetch.then((b) => {
    if (terrainIndexIn || !api) return;
    const bytes = new Uint8Array(b);
    if (!api.set_terrain_index(bytes)) return;
    terrainIndex = parseBlockIndex(bytes);
    terrainIndexIn = true;
    send({ kind: "terrainIndex", bytes: b.slice(0) });
    void ensureBlocks(viewBlocks);
    pumpBlocks();
  }, () => {});
}

function fetchBlock(key: number): Promise<ArrayBuffer> {
  let p = blockFetches.get(key);
  if (!p) {
    p = (async () => {
      const res = await fetch(new URL(blockFile(key), terrainBase).href, { signal: AbortSignal.timeout(GROUND_TIMEOUT_MS) });
      // only listed blocks are asked for, so a missing file is a fault, never flat ground
      if (!res.ok) throw new Error(`${res.status}`);
      const b = await res.arrayBuffer();
      // T-1854: the files are gzip (a server may already have unzipped them on the way)
      const gz = new Uint8Array(b, 0, Math.min(2, b.byteLength));
      if (gz[0] !== 0x1f || gz[1] !== 0x8b) return b;
      return await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    })();
    blockFetches.set(key, p);
    watchGround("t:" + key, p);
    // the bytes are not kept; a failed fetch is tried again by what needs it, or by `retryGround`
    const done = () => { if (blockFetches.get(key) === p) blockFetches.delete(key); };
    p.then(done, done);
  }
  return p;
}

/** Install these blocks (fetching those not fetched) and give the tint the ones the view reaches;
 * resolves once each is in or failed. A block that reprices the blueprint sends the new state, so the
 * quote follows. */
export async function ensureBlocks(keys: Iterable<number>): Promise<void> {
  if (!api || !terrainIndex) return;
  const todo = [...new Set(keys)].filter((k) => blockListed(k) && (!blocksIn.has(k) || (viewBlocks.has(k) && !tintSent.has(k))));
  if (!todo.length) return;
  const got = await Promise.all(todo.map((k) => fetchBlock(k).then((b) => ({ b }), () => null)));
  const a = api;
  if (!a) return;
  const before = a.blueprint_cost();
  todo.forEach((k, n) => {
    const g = got[n];
    if (!g) return;
    if (!blocksIn.has(k)) {
      // a file that is not a block (a server's page, a damaged upload) is a failure as well
      if (!a.add_terrain(new Uint8Array(g.b))) {
        groundFailed.add("t:" + k);
        retryGroundLater();
        sendGround();
        return;
      }
      blocksIn.add(k);
    }
    if (viewBlocks.has(k) && !tintSent.has(k)) {
      tintSent.add(k);
      send({ kind: "terrain", key: k, bytes: g.b }, [g.b]);
    }
  });
  trimBlocks();
  if (Math.abs(a.blueprint_cost() - before) > 1e-9) sendStateOutside();
}

/** Keep the blocks in memory in hand: a roaming player would otherwise fill WASM's memory block by block. */
function trimBlocks() {
  for (const k of blocksIn) {
    if (blocksIn.size <= MODEL_BLOCKS) break;
    if (!viewBlocks.has(k) && api!.drop_terrain(k)) blocksIn.delete(k);
  }
  if (tintSent.size > TINT_BLOCKS) {
    const keys: number[] = [];
    for (const k of tintSent) {
      if (tintSent.size - keys.length <= TINT_BLOCKS) break;
      if (!viewBlocks.has(k)) keys.push(k);
    }
    for (const k of keys) tintSent.delete(k);
    if (keys.length) send({ kind: "terrainDrop", keys });
  }
}

/** The blocks the track model has asked for since last time, on their way. */
export function pumpBlocks() {
  if (!api || !terrainIndexIn) return;
  const w = api.take_terrain_wanted();
  if (w.length) void ensureBlocks(w);
}

/** After a network is opened: the blocks it stands on, waited for (within a limit) so its Built figure
 * is right from the first state. */
async function terrainForLoaded() {
  if (!api || !terrainIndexIn) return;
  const w = api.take_terrain_wanted();
  if (w.length) await Promise.race([ensureBlocks(w), new Promise((r) => setTimeout(r, LOAD_TERRAIN_WAIT_MS))]);
}

// ---------------------------------------------------------------- ground downloads (T-2613)
//
// The terrain blocks, the zones' water masks and the world's water price track and say where it may
// go. The page is told how many are on their way and how many failed (the status tag), and a failed
// one still needed by the view or the blueprint is fetched again, 2 s after the failure doubling to 30 s.

/** a download given up after this long (a stalled connection would hold Construct for minutes) */
const GROUND_TIMEOUT_MS = 60_000;
const GROUND_RETRY_MS = [2_000, 30_000];
/** keys ("t:<block>", "index", "w:<zone>", "world") on their way, and those whose last fetch failed */
const groundPending = new Set<string>(), groundFailed = new Set<string>();
let groundSent = "";
let groundTimer: ReturnType<typeof setTimeout> | 0 = 0;
let groundWait = 0;

function watchGround(key: string, p: Promise<unknown>) {
  groundPending.add(key);
  sendGround();
  void p.then(() => { groundFailed.delete(key); groundWhy.delete(key); }, (err) => { groundFailed.add(key); groundWhy.set(key, String(err)); }).then(() => {
    groundPending.delete(key);
    if (groundFailed.size) retryGroundLater();
    else groundWait = 0;
    sendGround();
  });
}

/** the error text of each failed key, for a feedback report */
const groundWhy = new Map<string, string>();

function sendGround() {
  const why = [...groundFailed].slice(0, 5).map((k) => `${k} ${(groundWhy.get(k) ?? "bad file").slice(0, 80)}`);
  const k = `${groundPending.size} ${groundFailed.size} ${why.join("|")}`;
  if (k === groundSent) return;
  groundSent = k;
  send({ kind: "ground", pending: groundPending.size, failed: groundFailed.size, why });
}

/** T-3479: the status tag clicked: what failed is fetched again now, and the wait before the next try starts over */
export function retryGroundNow() {
  if (groundTimer) clearTimeout(groundTimer);
  groundTimer = 0;
  groundWait = 0;
  retryGround();
}

function retryGroundLater() {
  if (groundTimer) return;
  groundWait = Math.min(GROUND_RETRY_MS[1], groundWait ? groundWait * 2 : GROUND_RETRY_MS[0]);
  groundTimer = setTimeout(() => { groundTimer = 0; retryGround(); }, groundWait);
}

/** Fetch again what failed and is still needed; the rest is forgotten until something needs it. */
function retryGround() {
  if (!api) return;
  // (T-4072: and the loaded track's, whose masks a load no longer waits on for long)
  const wantedZones = new Set([...viewZones, ...loadZones, ...zonesOfPoints(blueprintPolylines().flat())]);
  // a block under any live edge is still needed, not just the blueprint's: its Built figure waits for it
  const wantedBlocks = new Set<number>([...viewBlocks, ...api.terrain_blocks_under(false)]);
  const needed = (k: string) => (k[0] === "t" ? wantedBlocks.has(Number(k.slice(2))) : k[0] === "w" && k[1] === ":" ? wantedZones.has(k.slice(2)) : true);
  for (const k of [...groundFailed]) if (!needed(k)) groundFailed.delete(k);
  sendGround();
  if (!groundFailed.size) { groundWait = 0; return; }
  if (groundFailed.has("world")) void worldWaterIntoModel();
  if (groundFailed.has("index")) void terrainIndexIntoModel();
  const failedBlocks = [...groundFailed].filter((k) => k[0] === "t").map((k) => Number(k.slice(2)));
  if (failedBlocks.length) void ensureBlocks(failedBlocks);
  const zones = [...new Set([...groundFailed].filter((k) => k[0] === "w" && k[1] === ":").map((k) => k.slice(2)))];
  // (a mask under loaded track reprices it: the state follows)
  if (zones.length) void ensureMasks(zones).then((added) => { if (added && api && !midLoad && zones.some((id) => loadZones.has(id))) sendStateOutside(); });
}

// ---------------------------------------------------------------- street speeds (T-1577)
//
// A bus runs at its share of the car's street speed where it is, so it is slower in a dense centre
// (sim track/streets.rs). The speed comes from a zone's pack: its cells' people and jobs, the cell's
// zone and the zones' squares and land, fetched as ranges of the pack's .bin with the mask. Like
// terrain, nothing waits for it: a route in a zone whose speeds are not in yet runs at the top speed
// and is timed again when they land. A pack without the arrays leaves its zone at the top speed.

interface StreetData { zoneM: number; x0: number; y0: number; w: number; h: number; pop: Float32Array; jobs: Float32Array; zone: Uint32Array; gx: Uint16Array; gy: Uint16Array; size: Uint8Array; cells9: Uint32Array; lon: number; lat: number }
const streetFetches = new Map<string, Promise<StreetData | null>>();
/** zones whose street speeds are in the track model (or that have none) */
export const streetsIn = new Set<string>();
const DTYPE_BYTES: Record<string, number> = { f32: 4, u32: 4, u16: 2, u8: 1 };

function fetchStreets(id: string): Promise<StreetData | null> {
  let p = streetFetches.get(id);
  if (!p) {
    const url = new URL(id, packBase).href;
    p = (async () => {
      const head = await (await fetch(url + ".json", { signal: AbortSignal.timeout(GROUND_TIMEOUT_MS) })).json();
      const g = head.gravity;
      const arr = (name: string) => head.arrays.find((a: any) => a.name === name);
      const need = ["pop", "jobs", "zone", "zone_gx", "zone_gy", "zone_cells9"].map(arr);
      if (!g?.grid_origin_m || !g.grid || !g.zone_m || need.some((a) => !a)) return null;
      const get = (a: any): Promise<ArrayBuffer> => fetchRange(url + ".bin", a.offset, a.offset + a.count * DTYPE_BYTES[a.dtype]);
      const size = arr("zone_size");
      const [pop, jobs, zone, gx, gy, cells9, sz] = await Promise.all([...need.map(get), size ? get(size) : Promise.resolve(new ArrayBuffer(0))]);
      return {
        zoneM: g.zone_m, x0: g.grid_origin_m[0], y0: g.grid_origin_m[1], w: g.grid[0], h: g.grid[1],
        pop: new Float32Array(pop), jobs: new Float32Array(jobs), zone: new Uint32Array(zone), gx: new Uint16Array(gx), gy: new Uint16Array(gy),
        size: new Uint8Array(sz), cells9: new Uint32Array(cells9), lon: head.origin.lon, lat: head.origin.lat,
      };
    })();
    streetFetches.set(id, p);
    // a failed fetch is tried again by the next view or edit that needs it
    p.catch(() => {
      if (streetFetches.get(id) === p) streetFetches.delete(id);
    });
  }
  return p;
}

/** Install the street speeds of `ids` (fetching those not fetched); resolves once each is in or failed.
 * The bus lines whose route meets them are timed again, and only then do the state and the snapshot follow. */
export async function ensureStreets(ids: Iterable<string>): Promise<void> {
  const todo = [...new Set(ids)].filter((id) => !streetsIn.has(id));
  if (!todo.length || !api) return;
  const got = await Promise.all(todo.map((id) => fetchStreets(id).then((d) => ({ d }), () => null)));
  const a = api;
  if (!a) return;
  let retimed = false;
  todo.forEach((id, k) => {
    const g = got[k];
    if (!g || streetsIn.has(id)) return;
    streetsIn.add(id);
    const d = g.d;
    const lines = d ? a.add_pack_streets(d.zoneM, d.x0, d.y0, d.w, d.h, d.pop, d.jobs, d.zone, d.gx, d.gy, d.size, d.cells9, d.lon, d.lat) : -1;
    if (lines >= 0) {
      // T-4155: a state only when a bus line was timed again; speeds where no bus runs change nothing drawn or counted
      if (lines > 0) retimed = true;
      // T-3913: where BRT runs faster, for the bus tool's road hint
      const outline = a.last_town_outline();
      if (outline.length > 1) send({ kind: "town", id, outline }, [outline.buffer]);
    }
  });
  if (retimed) {
    sendStateOutside();
    scheduleSnapshot();
  }
}

// ---------------------------------------------------------------- the world's coarse water (T-1285)
//
// One file for the whole world (`world.water` among the packs, pipeline/world_water.py, about 1 MB),
// deciding water wherever no zone's mask does: the open sea between zones. Fetched once at start;
// previews, edits and loads wait for it as for a zone's mask. A server without it (a 404) leaves the
// sea outside the zones as land.

let worldWaterFetch: Promise<ArrayBuffer | null> | null = null;
/** the current track model has the world's water (or there is none to have) */
export let worldWaterIn = false;

export function worldWaterIntoModel(): Promise<void> {
  if (worldWaterIn) return Promise.resolve();
  if (!worldWaterFetch) {
    const p = (async () => {
      const res = await fetch(new URL("world.water", packBase).href, { signal: AbortSignal.timeout(GROUND_TIMEOUT_MS) });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`${res.status}`);
      return await res.arrayBuffer();
    })();
    worldWaterFetch = p;
    watchGround("world", p);
    // a failed fetch is tried again by the next edit or `retryGround`; the sea outside the zones is
    // land until then (Construct waits for it)
    p.catch((err) => {
      if (worldWaterFetch === p) worldWaterFetch = null;
      send({ kind: "error", message: `world water mask not loaded: ${err}` }); // i18n-ignore
    });
  }
  return worldWaterFetch.then((b) => {
    if (worldWaterIn || !api) return;
    // a server that answers a missing file with a page gives no mask either
    if (b) api.add_world_water(new Uint8Array(b));
    worldWaterIn = true;
  }, () => {});
}

/** World x, y of every blueprint piece's drawn polyline, one array per edge. */
export function blueprintPolylines(): number[][] {
  const a = api!;
  const out: number[][] = [];
  const ei = a.edges_info();
  for (let i = 0; i < ei.length; i += 10) {
    if (ei[i + 4] === 1) continue;
    const r = a.edge_render(ei[i]);
    const xy: number[] = [];
    for (let k = 0; k + 2 < r.length; k += 3) xy.push(r[k], r[k + 1]);
    out.push(xy);
  }
  return out;
}

/** T-3196: Construct waits at most this long for the ground under the blueprint, ms; what has not landed by
 * then refuses it (the downloads go on, for the next try). Every other message waits behind it. */
const GROUND_WAIT_MS = 20_000;
/** `p` settled, or `until` (performance.now() ms) passed */
const within = (p: Promise<unknown>, until: number) =>
  Promise.race([p.then(() => {}), new Promise<void>((r) => setTimeout(r, Math.max(0, until - performance.now())))]);

/** Before Construct: the terrain, water and roads (T-1929) under every blueprint piece; the issue that
 * refuses it when one of them could not be had (T-2613), or null. */
export async function groundForBlueprint(): Promise<string | null> {
  const until = performance.now() + GROUND_WAIT_MS;
  const lines = blueprintPolylines();
  const zones = zonesOfPoints(lines.flat());
  const need = zones.filter((id) => !masksIn.has(id));
  const roads = roadKeys(lines);
  await within(Promise.all([worldWaterIntoModel(), terrainIndexIntoModel(), need.length ? ensureMasks(need) : null, roadFeed.ensure(roads)]), until);
  if (!worldWaterIn || !terrainIndexIn || zones.some((id) => !masksIn.has(id))) return "TerrainMissing";
  // walked by the track model along each edge at its own sample spacing, so a long straight piece counts
  // every block it crosses, after the index has made listed blocks known
  const blocks = api!.terrain_blocks_under(true);
  await within(ensureBlocks(blocks), until);
  if (blocks.some((k) => !blocksIn.has(k))) return "TerrainMissing";
  return roadFeed.missing(roads) ? "RoadsMissing" : null;
}

// ---------------------------------------------------------------- road crossings (T-1929)

/** the blueprint's price before the road tiles landing now (`roadFeed`'s burst) */
let roadsBefore: number | null = null;
export const roadFeed = new RoadFeed(() => {
  const a = api;
  if (!a) return null;
  return (tx, ty, extent, pts, starts, weights) => {
    roadsBefore ??= a.blueprint_cost();
    return a.add_road_tile(tx, ty, extent, pts, starts, weights);
  };
}, () => {
  const before = roadsBefore;
  roadsBefore = null;
  if (api && before !== null && Math.abs(api.blueprint_cost() - before) > 1e-9) sendStateOutside();
  // the ghost asks again
  send({ kind: "roads" });
});

/** Fetch the road tiles under these polylines of world x, y (not waited for by previews). */
export function roadsUnder(...lines: ArrayLike<number>[]): Promise<void> {
  const keys = roadKeys(lines);
  return keys.length ? roadFeed.ensure(keys) : Promise.resolve();
}
/** the road tiles under these polylines, by key */
function roadKeys(lines: ArrayLike<number>[]): number[] {
  const keys = new Set<number>();
  for (const xy of lines) {
    const ll: number[] = [];
    for (let k = 0; k + 1 < xy.length; k += 2) ll.push(...frame.toLonLat(xy[k], xy[k + 1]));
    for (const key of RoadFeed.tilesOf(ll)) keys.add(key);
  }
  return [...keys];
}

/** The zones within TRACK_MARGIN_M of any of these world-metre points (x, y every `stride`). */
export function zonesOfPoints(xy: ArrayLike<number>, stride = 2): string[] {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (let k = 0; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    if (lon < w) w = lon; if (lon > e) e = lon;
    if (lat < s) s = lat; if (lat > n) n = lat;
  }
  if (!(e >= w)) return [];
  const near = zonesTouching(w, s, e, n, TRACK_MARGIN_M);
  // one box over points far apart (a long route, a whole save) meets zones none of them is near:
  // test those point by point
  if (near.length <= 1) return near.map((c) => c.id);
  const boxes = near.map((c) => {
    const [bw, bs, be, bn] = zoneBox(c);
    const dy = TRACK_MARGIN_M / M_PER_DEG, dx = dy / Math.cos(Math.min(85, Math.max(Math.abs(bs), Math.abs(bn))) * Math.PI / 180);
    return { id: c.id, w: bw - dx, s: bs - dy, e: be + dx, n: bn + dy };
  });
  const hit = new Set<string>();
  for (let k = 0; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    for (const b of boxes) if (lon >= b.w && lon <= b.e && lat >= b.s && lat <= b.n) hit.add(b.id);
  }
  return [...hit];
}

export let wakeStale: (() => void) | null = null;

/** whether the last `withMasks` ended with every mask its geometry meets in */
export let masksComplete = false;

/** Work out a preview; while its geometry meets zones whose masks are not in, install them and work it
 * out again. Returns the last result, or null once preview `seq` is no longer wanted (the masks
 * still come in). */
export async function withMasks(run: () => Float64Array, points: (out: Float64Array) => number[], seq?: number): Promise<Float64Array | null> {
  masksComplete = false;
  await worldWaterIntoModel();
  for (;;) {
    const out = run();
    const need = zonesOfPoints(points(out)).filter((id) => !masksIn.has(id));
    if (!need.length) {
      masksComplete = true;
      return out;
    }
    if (seq === undefined) await ensureMasks(need);
    else {
      const gone = await Promise.race([ensureMasks(need).then(() => false), new Promise<boolean>((r) => (wakeStale = () => stale(seq) && r(true)))]);
      wakeStale = null;
      if (gone) return null;
    }
    // a mask that could not be fetched (reported) leaves the edit as it was without it
    if (need.every((id) => !masksIn.has(id))) return run();
  }
}

/** world x, y of a route or edge preview's polyline (`preview_route`'s layout) */
export function routePoints(out: Float64Array): number[] {
  const nv = out[3], p0 = 5 + nv * 4, xy: number[] = [];
  for (let k = p0; k + 2 < out.length; k += 3) xy.push(out[k], out[k + 1]);
  return xy;
}
/** the same for a node drag's preview (`preview_move_node`) */
export function nodePoints(out: Float64Array): number[] {
  const n = out[2], xy: number[] = [];
  let i = 3;
  for (let e = 0; e < n; e++) {
    i += out[i] * 4 + 1;
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) xy.push(out[i], out[i + 1]);
  }
  return xy;
}

/** T-4072: a game opens without waiting longer than this for its water masks (a stalled connection held a load
 * for over 10 minutes: each mask is up to four reads of 60 s, tried twice, 32 at a time). A mask that lands later
 * is installed then, reworking only the track in its box (`add_water`), and the state follows; until then that
 * track is priced as on land (DECISIONS 2026-11-23). */
let loadMaskWaitMs = 20_000;
export function putLoadMaskWait(ms: number) { loadMaskWaitMs = ms; }
/** T-4072: the zones the loaded network's track meets; a failed mask among them is fetched again (`retryGround`) */
let loadZones = new Set<string>();
/** T-4072: the zones whose masks this load has asked for already: `masksForLoaded` does not wait on them twice */
let askedAtLoad = new Set<string>();

/** T-4072: the load's step, for a report on a load that never ends */
const loadStep = (step: string) => send({ kind: "loadStep", step });

/** The masks of `ids`, waited for at most loadMaskWaitMs; those later are installed when they land. */
async function loadMasks(ids: string[]) {
  loadStep(`water masks for ${ids.length} zones`); // i18n-ignore
  const p = ensureMasks(ids, false);
  let done = false;
  await within(p.then(() => { done = true; }), performance.now() + loadMaskWaitMs);
  if (done) return;
  const late = ids.filter((id) => !masksIn.has(id)).length;
  loadStep(`water masks: ${late} of ${ids.length} late, the load goes on`); // i18n-ignore
  send({ kind: "error", message: `water masks: ${late} of ${ids.length} not in after ${loadMaskWaitMs / 1000} s; the game opens without them` }); // i18n-ignore
  // installed on the loaded network when they land (a load still under way sends its own state)
  void p.then((added) => { if (added && api && !midLoad) sendStateOutside(); });
}

/** Before a load (T-737): the masks of the zones the save's nodes and curve points fall in, so the
 * network is derived once (a load is about 3 s in WASM for Japan's 709 lines), and (T-4193) the street speeds of
 * the zones its bus routes meet, so the load times its buses on them and settles once (each zone's speeds landing
 * after it settle the whole network again: 13 minutes on a save with 92 bus lines in 19 zones, T-4188).
 * `masksForLoaded` still catches a zone only the drawn curves reach. */
async function masksBeforeLoad(net: Uint8Array) {
  askedAtLoad = new Set();
  if (trackFrameKind(net) !== NEW_GAME_FRAME.kind) return;
  const pts = TrackApi.save_points(net);
  if (!pts.length) return; // undecodable: the load says so
  const shown = frame;
  putFrame(makeFrame(NEW_GAME_FRAME));
  const missing = zonesOfPoints(pts).filter((id) => !masksIn.has(id));
  const streets = zonesOfPoints(TrackApi.save_bus_points(net)).filter((id) => !streetsIn.has(id));
  putFrame(shown);
  if (!missing.length && !streets.length) return;
  for (const id of missing) askedAtLoad.add(id);
  // installed in the last game's network, they would rework its track and time its buses again
  api!.load(emptyTrack!);
  const speeds = ensureStreets(streets);
  if (missing.length) await loadMasks(missing);
  if (streets.length) {
    loadStep(`street speeds for ${streets.length} zones`); // i18n-ignore
    await speeds;
  }
}

/** Put a network (in the world frame) in the track model, with the masks of the zones it meets. `level`: the
 * save level a player's game was written at (T-3647, `TrackApi.set_load_level`), absent for this build's. */
export async function openNetwork(net: Uint8Array, level?: number) {
  loadStep("world water"); // i18n-ignore
  await worldWaterIntoModel();
  // T-3478: a stalled index holds the open no longer than its blocks may; the blocks follow when it lands
  loadStep("terrain index"); // i18n-ignore
  await Promise.race([terrainIndexIntoModel(), new Promise((r) => setTimeout(r, LOAD_TERRAIN_WAIT_MS))]);
  await masksBeforeLoad(net);
  loadStep("network into the track model"); // i18n-ignore
  if (level !== undefined) api!.set_load_level(level);
  if (!api!.load(net)) throw new Error(SAVE_DAMAGED);
  await masksForLoaded(net, level);
  loadStep("terrain under the network"); // i18n-ignore
  await terrainForLoaded();
  // T-1929: the roads under the blueprint, so its quote has them before Construct waits for them
  void roadsUnder(...blueprintPolylines());
  loadStep("network open"); // i18n-ignore
}

/** After a network is loaded: the masks of every zone its track meets, then the network opened again
 * over them if any was missing that the load had not asked for yet (T-4072: a mask asked before the load and still
 * on its way, or failed, is not waited for again: it is installed when it lands, or fetched again by `retryGround`). */
async function masksForLoaded(net: Uint8Array, level?: number) {
  const a = api!;
  syncFrame();
  const xy: number[] = [];
  const ni = a.nodes_info();
  for (let i = 0; i < ni.length; i += 10) xy.push(ni[i + 1], ni[i + 2]);
  const ei = a.edges_info();
  for (let i = 0; i < ei.length; i += 10) {
    const r = a.edge_render(ei[i]);
    for (let k = 0; k + 2 < r.length; k += 3) xy.push(r[k], r[k + 1]);
  }
  // T-1577: the street speeds under the bus routes (their points are no track), in before the load unless a fetch
  // failed or the save was in another frame; T-2499: waited for, so the load's network is the save's alone (its
  // buses timed on them), not whichever fetch won
  const bus: number[] = [];
  for (const id of a.bus_lines()) {
    const r = a.bus_route(id);
    for (let k = 1; k + 2 < r.length; k += 3) bus.push(r[k], r[k + 1]);
  }
  const busZones = bus.length ? zonesOfPoints(bus).filter((id) => !streetsIn.has(id)) : [];
  if (busZones.length) {
    loadStep(`street speeds for ${busZones.length} zones`); // i18n-ignore
    await ensureStreets(busZones);
  }
  const zones = zonesOfPoints(xy);
  loadZones = new Set(zones);
  const missing = zones.filter((id) => !masksIn.has(id) && !askedAtLoad.has(id));
  if (!missing.length) return;
  a.load(emptyTrack!);
  await loadMasks(missing);
  if (level !== undefined) a.set_load_level(level);
  if (!a.load(net)) throw new Error(SAVE_DAMAGED);
}

/** the zones whose masks the view fetches ahead (T-302) */
export let viewZones: string[] = [];
export function putPackBase(v: typeof packBase) { packBase = v; }
export function putTerrainBase(v: typeof terrainBase) { terrainBase = v; }
export function putTerrainIndexIn(v: typeof terrainIndexIn) { terrainIndexIn = v; }
export function putViewBlocks(v: typeof viewBlocks) { viewBlocks = v; }
export function putViewZones(v: typeof viewZones) { viewZones = v; }
export function putWorldWaterIn(v: typeof worldWaterIn) { worldWaterIn = v; }
