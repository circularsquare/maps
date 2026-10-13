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
/** `market`: the city's place in the regional market, or -1 for one outside it (Toronto, T-277). */
export interface City { id: CityId; name: string; kind?: "city" | "zone"; origin: { lon: number; lat: number }; center: [number, number]; zoom: number; market: number; localTrips?: number }
interface Registry {
  world: string;
  frame: FrameSpec;
  legacyFrame: FrameSpec;
  market: { pack: string; frame: FrameSpec; places: { name: string; regionName: string }[] };
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
/** The pack the regional market keeps its sites in. */
export const MARKET_PACK = REGISTRY.market.pack;
export const MARKET_FRAME = makeFrame({ ...REGISTRY.market.frame, kind: "equirect" });

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
/** A city pack's metres to the regional market pack's. */
export function cityToMarket(x: number, y: number, id: CityId): { x: number; y: number } {
  return convert(x, y, packFrame(id), MARKET_FRAME);
}

/** Follow the nearer city when panning; navigation buttons select it immediately. */
export function cityNear(lon: number, lat: number): CityId {
  const cosLat = Math.cos(lat * Math.PI / 180);
  const dist = (c: City) => Math.hypot((lon - c.center[0]) * cosLat, lat - c.center[1]);
  return [...CITIES].sort((a, b) => dist(a) - dist(b))[0].id;
}
