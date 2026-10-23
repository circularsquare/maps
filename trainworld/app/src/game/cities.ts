// The city registry (T-125) and the conversions between the world frame and each demand pack's
// own frame. Everything that names a city, a pack, a frame or the regional market reads
// registry.json: adding a region is a registry entry plus its pack (pipeline/build_city.py), not
// code in the clock worker, the demand workers, the save header or the UI.
//
// Every conversion goes through longitude and latitude, so a pack keeps its own local origin and
// the world frame can be either of the two frames (game/frame.ts).
import { signal } from "@preact/signals";
import registryJson from "./registry.json";
import { makeFrame, type Frame, type FrameSpec } from "./frame";
import { zoneAliases, zoneName } from "./zoneNames";

export type CityId = string;
/** A zone of the world (T-276, docs/zones.md): a city zone, or a non-city zone (`kind` "zone");
 * `localTrips`: its local trip legs a day with no rail (pipeline/zones.py), for the far-zoomed
 * bubble of a zone not opened. */
/** `market`: the city's place in the regional market, or -1 for one outside it (Toronto, T-277).
 * `utcOffset`: its whole-hour standard time offset from UTC, no daylight saving (T-309). */
/** `box`: [west, south, east, north] in degrees round its boundary and water mask
 * (pipeline/zone_boxes.py, T-302), so where a zone is is known without fetching its files. */
/** T-377: `place`, the zone's place in the world market (every zone has one; `market` is the
 * Northeast joint commute gravity's, -1 outside it). Its market row is not here but in
 * market-rows.json, fetched once a market pack opens (T-776, `marketWorld`). `country`: ISO 3166 alpha-2.
 * `realm`: the realm it shows with (T-975, game/realms.ts; pipeline/realms.py). */
/** `name` is the shown name and `aliases` every other name search finds it by (T-825, game/zoneNames.ts). */
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
  const rows = JSON.parse(rowsText) as Record<CityId, unknown>;
  return JSON.stringify({ frame: REGISTRY.market.frame, rule: REGISTRY.market.rule, places: REGISTRY.market.places.length, rows: CITIES.flatMap((c) => (rows[c.id] ? [rows[c.id]] : [])) });
}

export const activeCity = signal<CityId>(HOME_CITY);
const citiesById = new Map(CITIES.map((c) => [c.id, c] as const));
export const cityById = (id: CityId) => citiesById.get(id)!;
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

/** The nearest zone whose centre is within `CENTRE_REACH_M`, else null (dead space). */
export function cityWithin(lon: number, lat: number): CityId | null {
  const id = cityNear(lon, lat), c = cityById(id);
  const kx = 111_320 * Math.cos(lat * Math.PI / 180);
  return Math.hypot((lon - c.center[0]) * kx, (lat - c.center[1]) * 111_320) <= CENTRE_REACH_M ? id : null;
}

/** A zone's `box`, or for one without it a square `CENTRE_REACH_M` round its centre. */
export function zoneBox(c: City): [number, number, number, number] {
  if (c.box) return c.box;
  const dy = CENTRE_REACH_M / 111_320, dx = dy / Math.cos(c.center[1] * Math.PI / 180);
  return [c.center[0] - dx, c.center[1] - dy, c.center[0] + dx, c.center[1] + dy];
}

/** The zones whose boxes, widened by `marginM` metres, meet the box [w, s, e, n] in degrees. */
export function zonesTouching(w: number, s: number, e: number, n: number, marginM = 0): City[] {
  // the candidates by the box widened by the most any zone's margin can be (at 85 degrees)
  const my = marginM / 111_320, mx = my / Math.cos(85 * Math.PI / 180);
  return zonesNearBox(w - mx, s - my, e + mx, n + my).map((i) => CITIES[i]).filter((c) => {
    const [bw, bs, be, bn] = zoneBox(c);
    const dy = marginM / 111_320, dx = dy / Math.cos(Math.min(85, Math.max(Math.abs(bs), Math.abs(bn))) * Math.PI / 180);
    return w <= be + dx && e >= bw - dx && s <= bn + dy && n >= bs - dy;
  });
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
