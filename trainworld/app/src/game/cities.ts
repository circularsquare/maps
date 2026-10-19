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

export type CityId = string;
/** A zone of the world (T-276, docs/zones.md): a city zone, or a non-city zone (`kind` "zone");
 * `localTrips`: its local trip legs a day with no rail (pipeline/zones.py), for the far-zoomed
 * bubble of a zone not opened. */
/** `market`: the city's place in the regional market, or -1 for one outside it (Toronto, T-277).
 * `utcOffset`: its whole-hour standard time offset from UTC, no daylight saving (T-309). */
/** `box`: [west, south, east, north] in degrees round its boundary and water mask
 * (pipeline/zone_boxes.py, T-302), so where a zone is is known without fetching its files. */
/** T-377: `place`, the zone's place in the world market (every zone has one; `market` is the
 * Northeast joint commute gravity's, -1 outside it); `marketRow`, its row (pipeline/market_zones.py). */
export interface City { id: CityId; name: string; kind?: "city" | "zone"; origin: { lon: number; lat: number }; center: [number, number]; zoom: number; utcOffset?: number; market: number; place: number; marketRow?: unknown; localTrips?: number; box?: [number, number, number, number] }
interface Registry {
  world: string;
  /** T-309: the UTC offset the game clock counts world time in (game/localTime.ts) */
  clockUtcOffset: number;
  frame: FrameSpec;
  legacyFrame: FrameSpec;
  market: { frame: FrameSpec; rule: unknown; places: { name: string; regionName: string }[] };
  cities: City[];
}
export const REGISTRY = registryJson as unknown as Registry;
export const CITIES: City[] = REGISTRY.cities;
/** The identity of the shared world in save headers and slot lists. */
export const WORLD_ID = REGISTRY.world;
/** The city a save names in its header (`city`): the first. */
export const HOME_CITY: CityId = CITIES[0].id;
/** The frame of new games, and of every save made before T-125. */
export const NEW_GAME_FRAME: FrameSpec = REGISTRY.frame;
export const LEGACY_FRAME: FrameSpec = REGISTRY.legacyFrame;
/** The frame the market reports positions in. */
export const MARKET_FRAME = makeFrame({ ...REGISTRY.market.frame, kind: "equirect" });
/** T-377: the world market's frame, rule, places and every zone's row, as `MarketPacks` reads it. */
export const MARKET_WORLD = JSON.stringify({ frame: REGISTRY.market.frame, rule: REGISTRY.market.rule, places: REGISTRY.market.places.length, rows: CITIES.flatMap((c) => (c.marketRow ? [c.marketRow] : [])) });

export const activeCity = signal<CityId>(HOME_CITY);
export const cityById = (id: CityId) => CITIES.find(c => c.id === id)!;
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
  // a scan, not a sort: it runs on every map move (localTime's shiftAt) over every registered zone
  let best = CITIES[0], bd = Infinity;
  for (const c of CITIES) {
    const d = Math.hypot((lon - c.center[0]) * cosLat, lat - c.center[1]);
    if (d < bd) { bd = d; best = c; }
  }
  return best.id;
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
  return CITIES.filter((c) => {
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
