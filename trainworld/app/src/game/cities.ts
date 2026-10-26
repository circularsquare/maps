// The city registry (T-125) and the conversions between the world frame and each demand pack's
// own frame. Everything that names a city, a pack, a frame or the regional market reads
// registry.json: adding a region is a registry entry plus its pack (pipeline/build_city.py), not
// code in the clock worker, the demand workers, the save header or the UI.
//
// Every conversion goes through longitude and latitude, so a pack keeps its own local origin and
// the world frame can be either of the two frames (game/frame.ts).
import { signal } from "@preact/signals";
import registryJson from "./registry.json";
import { M_PER_DEG, makeFrame, type Frame, type FrameSpec } from "./frame";
import { zoneAliases, zoneName } from "./zoneNames";
import { fareWeight } from "./income";

export type CityId = string;
/** A zone of the world (T-276, docs/zones.md): a city zone, or a non-city zone (`kind` "zone").
 * `name` is the shown name and `aliases` every other name search finds it by (T-825, game/zoneNames.ts).
 * `localTrips`: its local trip legs a day with no rail (pipeline/zones.py), for the far-zoomed
 * bubble of a zone not opened.
 * `market`: the city's place in the regional market, or -1 for one outside it (Toronto, T-277).
 * `place`: its place in the world market (every zone has one; `market` is the Northeast joint
 * commute gravity's). Its market row is in market-rows.json, fetched once a market pack opens
 * (T-776, `marketWorld`).
 * `utcOffset`: its whole-hour standard time offset from UTC, no daylight saving (T-309).
 * `box`: [west, south, east, north] in degrees round its boundary and water mask
 * (pipeline/zone_boxes.py, T-302), so where a zone is is known without fetching its files.
 * `country`: ISO 3166 alpha-2. `realm`: the realm it shows with (T-975, game/realms.ts;
 * pipeline/realms.py). */
export interface City { id: CityId; name: string; aliases: string[]; kind?: "city" | "zone"; country?: string; realm?: string; origin: { lon: number; lat: number }; center: [number, number]; zoom: number; utcOffset?: number; market: number; place: number; localTrips?: number; box?: [number, number, number, number] }
interface Registry {
  world: string;
  /** T-309: the UTC offset the game clock counts world time in (game/localTime.ts) */
  clockUtcOffset: number;
  frame: FrameSpec;
  legacyFrame: FrameSpec;
  market: { frame: FrameSpec; rule: unknown; places: { name: string; regionName: string }[] };
  cities: Omit<City, "aliases">[];
}
export const REGISTRY = registryJson as unknown as Registry;
const fold = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
export const CITIES: City[] = REGISTRY.cities.map((c) => ({ ...c, name: zoneName(c.id, c.name), aliases: zoneAliases(c.id, c.name, fold) }));
/** The identity of the shared world in save headers and slot lists. */
export const WORLD_ID = REGISTRY.world;
/** The city a save names in its header (`city`): the first. */
export const HOME_CITY: CityId = CITIES[0].id;
/** The frame of new games, and of every save made before T-125. */
export const NEW_GAME_FRAME: FrameSpec = REGISTRY.frame;
export const LEGACY_FRAME: FrameSpec = REGISTRY.legacyFrame;
/** The frame the market reports positions in. */
export const MARKET_FRAME = makeFrame({ ...REGISTRY.market.frame, kind: "equirect" });
/** T-377: the world market's frame, rule, places and every registered zone's row in registry order,
 * as `MarketPacks` reads it; `rowsText` is market-rows.json ({zone id: row}, T-776). */
export function marketWorld(rowsText: string): string {
  const rows = JSON.parse(rowsText) as Record<CityId, { country?: string } | undefined>;
  // T-1277: a plane's fare weighs the departing country's minutes a dollar
  const withFw = (r: { country?: string }) => ({ ...r, fw: fareWeight(r.country) });
  return JSON.stringify({ frame: REGISTRY.market.frame, rule: REGISTRY.market.rule, places: REGISTRY.market.places.length, rows: CITIES.flatMap((c) => { const r = rows[c.id]; return r ? [withFw(r)] : []; }) });
}

export const activeCity = signal<CityId>(HOME_CITY);
const citiesById = new Map(CITIES.map((c) => [c.id, c] as const));
export const cityById = (id: CityId) => citiesById.get(id)!;
/** Whether the registry has a city or zone of this id. */
export const isCity = (id: string | null | undefined): boolean => !!id && citiesById.has(id);
const packFrames = new Map(CITIES.map(c => [c.id, makeFrame({ kind: "equirect", ...c.origin })] as const));
const packFrame = (id: CityId): Frame => packFrames.get(id)!;

// ---------------------------------------------------------------- the frame the game is in

let world: Frame = makeFrame(LEGACY_FRAME);
/** Bumped when the frame changes, for caches kept in world metres. */
export const worldFrameEpoch = signal(0);

/** The loaded game's frame (the clock worker reports it on every load). */
export function setWorldFrame(spec: FrameSpec) {
  const next = makeFrame(spec);
  if (next === world) return false;
  world = next;
  worldFrameEpoch.value++;
  return true;
}
export const getWorldFrame = (): Frame => world;

// ---------------------------------------------------------------- conversions

function convert(x: number, y: number, from: Frame, to: Frame): { x: number; y: number } {
  if (from === to) return { x, y };
  const [lon, lat] = from.toLonLat(x, y);
  const [px, py] = to.fromLonLat(lon, lat);
  return { x: px, y: py };
}

/** World metres to a city pack's local metres. */
export function worldToCity(x: number, y: number, id: CityId): { x: number; y: number } {
  return convert(x, y, world, packFrame(id));
}
/** A city pack's local metres to world metres. */
export function cityToWorld(x: number, y: number, id: CityId): { x: number; y: number } {
  return convert(x, y, packFrame(id), world);
}
/** World metres to the regional market pack's metres (its sites and 600 m squares live there). */
export function worldToMarket(x: number, y: number): { x: number; y: number } {
  return convert(x, y, world, MARKET_FRAME);
}
/** The regional market pack's metres to world metres. */
export function marketToWorld(x: number, y: number): { x: number; y: number } {
  return convert(x, y, MARKET_FRAME, world);
}
/** A city pack's metres to its market pack's (`<id>.market.json`): the Northeast's sites are in
 * the market frame, as the joint commute gravity names them (T-276), every other zone's in its
 * own pack frame (T-377). */
export function cityToMarket(x: number, y: number, id: CityId): { x: number; y: number } {
  return cityById(id).market >= 0 ? convert(x, y, packFrame(id), MARKET_FRAME) : { x, y };
}
/** Longitude and latitude to a city pack's local metres. */
export function lonLatToCity(lon: number, lat: number, id: CityId): { x: number; y: number } {
  const [x, y] = packFrame(id).fromLonLat(lon, lat);
  return { x, y };
}
/** A city pack's local metres to longitude and latitude. */
export function cityToLonLat(x: number, y: number, id: CityId): [number, number] {
  return packFrame(id).toLonLat(x, y);
}
/** World metres to longitude and latitude. */
export function worldToLonLat(x: number, y: number): [number, number] {
  return world.toLonLat(x, y);
}

/** The nearest registered city or zone by centre, at any distance (a lookup that needs some zone,
 * not a name for the place: for that see `cityWithin` and `placeAt` in game/zones.ts). */
export function cityNear(lon: number, lat: number): CityId {
  const cosLat = Math.cos(lat * Math.PI / 180);
  let best = 0, bd = Infinity;
  const at = (i: number) => {
    const c = CITIES[i].center, d = Math.hypot((lon - c[0]) * cosLat, lat - c[1]);
    if (d < bd || (d === bd && i < best)) { bd = d; best = i; }
  };
  // T-1008: the centres by whole degree, searched ring by ring out from the place (every station
  // of a save asks, thousands at a load); a place off the grid scans them all
  const x = Math.floor(lon) + 180, y = Math.floor(lat) + 90;
  if (!(x >= 0 && x < GRID_W && y >= 0 && y < GRID_H)) {
    for (let i = 0; i < CITIES.length; i++) at(i);
    return CITIES[best].id;
  }
  const g = centreGrid();
  for (const i of g.off) at(i);
  // a centre r rings out is at least r - 1 whole degrees away in latitude or longitude
  const step = Math.min(1, cosLat) * (1 - 1e-9);
  for (let r = 0; r < Math.max(GRID_W, GRID_H); r++) {
    if (Math.max(0, r - 1) * step > bd) break;
    // far from every zone (mid-ocean): more cells than zones, so the plain scan is quicker
    if ((2 * r + 1) ** 2 > CITIES.length) { for (let i = 0; i < CITIES.length; i++) at(i); break; }
    for (let yy = y - r; yy <= y + r; yy++) {
      if (yy < 0 || yy >= GRID_H) continue;
      const edge = yy === y - r || yy === y + r;
      for (let xx = x - r; xx <= x + r; xx += edge ? 1 : 2 * r || 1) {
        if (xx < 0 || xx >= GRID_W) continue;
        const cell = yy * GRID_W + xx;
        for (let q = g.start[cell]; q < g.start[cell + 1]; q++) at(g.ids[q]);
      }
    }
  }
  return CITIES[best].id;
}

const GRID_W = 360, GRID_H = 180;
/** CITIES indices by cell (`start` offsets into `ids`, cell = (lat + 90) * 360 + lon + 180 floored) */
interface Grid { start: Uint32Array; ids: Uint32Array }
function makeGrid(cells: (i: number) => number[]): Grid {
  const lists = CITIES.map((_, i) => cells(i));
  const start = new Uint32Array(GRID_W * GRID_H + 1);
  for (const l of lists) for (const c of l) start[c + 1]++;
  for (let c = 0; c < GRID_W * GRID_H; c++) start[c + 1] += start[c];
  const ids = new Uint32Array(start[GRID_W * GRID_H]), fill = start.slice(0, -1);
  lists.forEach((l, i) => { for (const c of l) ids[fill[c]++] = i; });
  return { start, ids };
}
let centres: Grid & { off: number[] } | null = null;
const centreGrid = () => {
  if (centres) return centres;
  const off: number[] = [];
  return (centres = { ...makeGrid((i) => {
    const [lon, lat] = CITIES[i].center, x = Math.floor(lon) + 180, y = Math.floor(lat) + 90;
    if (x >= 0 && x < GRID_W && y >= 0 && y < GRID_H) return [y * GRID_W + x];
    off.push(i);
    return [];
  }), off });
};

/** The cells a box [w, s, e, n] in degrees covers; null for one the grid cannot hold (crossing the
 * antimeridian, or not a number). */
function boxCells(w: number, s: number, e: number, n: number): number[] | null {
  if (!(w <= e && s <= n)) return null;
  // clamped, not cut, so two boxes that meet off the grid still share its edge cells
  const cx = (v: number) => Math.min(GRID_W - 1, Math.max(0, Math.floor(v) + 180)), cy = (v: number) => Math.min(GRID_H - 1, Math.max(0, Math.floor(v) + 90));
  const x0 = cx(w), x1 = cx(e), y0 = cy(s), y1 = cy(n);
  const out: number[] = [];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) out.push(y * GRID_W + x);
  return out;
}
let boxes: { grid: Grid; always: number[] } | null = null;
/** T-1008: the zones whose `zoneBox` may meet the box [w, s, e, n] (a superset), ascending
 * CITIES indices: those by the grid's cells, and every zone the grid cannot hold. */
export function zonesNearBox(w: number, s: number, e: number, n: number): number[] {
  if (!boxes) {
    const always: number[] = [];
    const grid = makeGrid((i) => { const c = boxCells(...zoneBox(CITIES[i])); if (!c) always.push(i); return c ?? []; });
    boxes = { grid, always };
  }
  const cells = boxCells(w, s, e, n);
  if (!cells) return CITIES.map((_, i) => i);
  const out = new Set(boxes.always);
  for (const c of cells) for (let q = boxes.grid.start[c]; q < boxes.grid.start[c + 1]; q++) out.add(boxes.grid.ids[q]);
  return [...out].sort((a, b) => a - b);
}

/** T-318: how far from a zone's centre a place still counts as in it while its boundary is not
 * loaded: 120 km, past the farthest edge of the biggest metro here (New York reaches Trenton and
 * Pike County, about 100 km out). Once the boundary is in, `ZONE_REACH_M` from it is used instead. */
export const CENTRE_REACH_M = 120_000;
/** T-318: how far outside a zone's boundary a place still gets its name and loads its pack: 40 km,
 * about the market's onset (25-50 km, docs/zones.md 5) and the ring T-300 would add round a city.
 * Past it the view is over dead space. */
export const ZONE_REACH_M = 40_000;

/** T-1265: the zone a place is in by the coarse outlines, set by map/coarseOutlines.ts in each thread
 * that has read them. */
let byOutline: ((lon: number, lat: number) => CityId | null) | null = null;
/** Bumped when the coarse outlines arrive, so answers kept from before them are worked out again. */
export const outlinesIn = signal(0);
export function setOutlineLookup(f: (lon: number, lat: number) => CityId | null) {
  byOutline = f;
  within.clear();
  withinSize = 0;
  outlinesIn.value++;
}

/** cityWithin's answers by lon then lat, so the stations and nodes several modules place on each
 * state are looked up once (T-1209); dropped when the outlines arrive or it grows past WITHIN_MAX */
const within = new Map<number, Map<number, CityId | null>>();
let withinSize = 0;
const WITHIN_MAX = 50_000;

/** The zone whose coarse outline holds (lon, lat) or is within `ZONE_REACH_M` of it; before the
 * outlines are in, or where none is near, the nearest zone whose centre is within `CENTRE_REACH_M`;
 * else null (dead space). A nearest centre alone named Shenzhen for Yuen Long (T-1265). */
export function cityWithin(lon: number, lat: number): CityId | null {
  let row = within.get(lon);
  const got = row?.get(lat);
  if (got !== undefined) return got;
  const id = byOutline?.(lon, lat) ?? centreWithin(lon, lat);
  if (withinSize >= WITHIN_MAX) { within.clear(); withinSize = 0; row = undefined; }
  if (!row) within.set(lon, (row = new Map()));
  row.set(lat, id);
  withinSize++;
  return id;
}

/** The nearest zone whose centre is within `CENTRE_REACH_M`, else null: the track model's rule for a
 * place's local time (sim/src/track/localtime.rs), which has no outlines. */
export function centreWithin(lon: number, lat: number): CityId | null {
  const id = cityNear(lon, lat), c = cityById(id);
  const kx = M_PER_DEG * Math.cos(lat * Math.PI / 180);
  return Math.hypot((lon - c.center[0]) * kx, (lat - c.center[1]) * M_PER_DEG) <= CENTRE_REACH_M ? id : null;
}

/** A zone's `box`, or for one without it a square `CENTRE_REACH_M` round its centre. */
export function zoneBox(c: City): [number, number, number, number] {
  if (c.box) return c.box;
  const dy = CENTRE_REACH_M / M_PER_DEG, dx = dy / Math.cos(c.center[1] * Math.PI / 180);
  return [c.center[0] - dx, c.center[1] - dy, c.center[0] + dx, c.center[1] + dy];
}

/** A zone's box [w, s, e, n] widened by `marginM` metres: the margin in degrees of longitude is as
 * at the box's highest latitude, at most 85. */
export function widenBox([w, s, e, n]: readonly number[], marginM: number): [number, number, number, number] {
  const dy = marginM / M_PER_DEG, dx = dy / Math.cos(Math.min(85, Math.max(Math.abs(s), Math.abs(n))) * Math.PI / 180);
  return [w - dx, s - dy, e + dx, n + dy];
}

/** Every zone's box widened by one margin, and the grid of those boxes (T-1348: every station of a
 * load asked with `ZONE_REACH_M`, whose candidates by the unwidened grid were most of a region). */
interface Widened { box: Float64Array; grid: Grid; always: number[] }
const widened = new Map<number, Widened>();
function widenedBy(marginM: number): Widened {
  let wb = widened.get(marginM);
  if (wb) return wb;
  const box = new Float64Array(4 * CITIES.length);
  CITIES.forEach((c, i) => box.set(widenBox(zoneBox(c), marginM), 4 * i));
  const always: number[] = [];
  const grid = makeGrid((i) => { const c = boxCells(box[4 * i], box[4 * i + 1], box[4 * i + 2], box[4 * i + 3]); if (!c) always.push(i); return c ?? []; });
  widened.set(marginM, (wb = { box, grid, always }));
  return wb;
}

/** The zones whose boxes, widened by `marginM` metres, meet the box [w, s, e, n] in degrees, in
 * registry order. */
export function zonesTouching(w: number, s: number, e: number, n: number, marginM = 0): City[] {
  const { box, grid, always } = widenedBy(marginM);
  const meets = (i: number) => w <= box[4 * i + 2] && e >= box[4 * i] && s <= box[4 * i + 3] && n >= box[4 * i + 1];
  const cells = boxCells(w, s, e, n);
  if (!cells) return CITIES.filter((_, i) => meets(i));
  let ids: number[];
  if (cells.length === 1 && !always.length) {
    const c = cells[0];
    ids = Array.from(grid.ids.subarray(grid.start[c], grid.start[c + 1]));
  } else {
    const set = new Set(always);
    for (const c of cells) for (let q = grid.start[c]; q < grid.start[c + 1]; q++) set.add(grid.ids[q]);
    ids = [...set];
  }
  ids.sort((a, b) => a - b);
  const out: City[] = [];
  for (const i of ids) if (meets(i)) out.push(CITIES[i]);
  return out;
}

/** The view in the URL hash (`#zoom/lat/lng`), before MapLibre reads it. */
export function hashView(): { lng: number; lat: number; zoom: number } | null {
  const m = /^#(-?[\d.]+)\/(-?[\d.]+)\/(-?[\d.]+)/.exec(location.hash);
  const zoom = m ? Number(m[1]) : NaN, lat = m ? Number(m[2]) : NaN, lng = m ? Number(m[3]) : NaN;
  return Number.isFinite(zoom) && Number.isFinite(lat) && Number.isFinite(lng) ? { lng, lat, zoom } : null;
}

/** T-318: the zone the view is in or within reach of; null over dead space. It decides the top bar's
 * name and whether anything loads for the view (workers/demandClient.ts). Set by map/zoneFocus.ts
 * after each pan; from the start view until the map exists (the URL hash, else the default view). */
export const viewedZone = signal<CityId | null>((() => { const v = hashView(); return v ? cityWithin(v.lng, v.lat) : cityWithin(-73.96, 40.75); })());
// T-396: the focus starts at the zone in view, so a reload at Toronto never opens New York (and the market)
if (viewedZone.peek()) activeCity.value = viewedZone.peek()!;
