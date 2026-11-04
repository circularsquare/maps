// Moving the camera to a station or a line (T-569, and search's picks, T-148): the Stations and
// Lines tabs' double click and the inspectors' locate button. A jump, never a flight (T-182). It
// only moves the camera; selecting is the caller's.

import { effect, untracked } from "@preact/signals";
import { activeCity, CITIES, viewedZone } from "../game/cities";
import { makeFrame, type FrameSpec } from "../game/frame";
import { home } from "../game/home";
import { fitView } from "../game/loading";
import { WORLD_ZOOM } from "../game/worldTotals";
import { WORLD_HOME } from "../workers/saveFile";
import { openedZones } from "../game/openedZones";
import { display } from "../game/ui";
import { stopsById, world } from "../game/world";
import { placeAt, zoneAt } from "../game/zones";
import { live } from "../map/live";
import { STATION_BOX_ZOOM } from "../map/stationGeometry";

/** The map at a station: its zoom if that is already closer than where platform boxes show. */
export function locateStation(lng: number, lat: number) {
  const map = live.map;
  if (!map) return;
  map.jumpTo({ center: [lng, lat], zoom: Math.max(map.getZoom(), STATION_BOX_ZOOM) });
  activeCity.value = zoneAt(lng, lat);
}

/** A start on the world map: the Americas, Europe and East Asia in view. */
const WORLD_VIEW = { center: [20, 30] as [number, number], zoom: 1.8 };

/** A start from the player's location (T-1132): where the camera goes, taken by the next `viewStart`. */
let startAt: { center: [number, number]; zoom: number } | null = null;
export function setStartAt(at: { center: [number, number]; zoom: number } | null) {
  startAt = at;
}

/** Move the camera to a new game's start: the city's registry view, or the world map; a start from
 * the player's location goes to that location instead. */
export function viewStart(start?: string) {
  const at = startAt;
  startAt = null;
  const c = start ? CITIES.find((x) => x.id === start) : undefined;
  if (!at && c && viewedZone.peek() === c.id && live.map && live.map.getZoom() > 9) return; // already there
  if (c) {
    activeCity.value = c.id;
    viewedZone.value = c.id;
  }
  live.map?.jumpTo(at ?? (c ? { center: c.center, zoom: c.zoom } : WORLD_VIEW));
  // T-801: a start on the map is picked from the zones' zoomed-out bubbles, so Destinations goes on (T-1441)
  if (!c) display.value = { ...display.peek(), commuters: true, commuterEnd: "work" };
}

/** Furthest in a saved game opens: a district, so a lone station is not shown at street level (T-895). */
const OPEN_MAX_ZOOM = 12;

/** T-895, T-1872: a game the player opened (start menu, saves panel, imported file) opens on the city
 * view of the zone with the most built stations (most of all stations, if none is built); with no
 * stations, the zone it was started in, else the world map. Runs when the game's first state arrives
 * (`fitView`), which is before its demand. */
export function startOpenFit() {
  // T-977: untracked, since the camera move runs the map's listeners in this effect (zoomed out, the
  // city view's tags read the bubble scheme), and anything they read would make the fit run again
  effect(() => { if (fitView.value) untracked(fitOpened); });
}

function fitOpened() {
  const w = world.peek();
  const map = live.map;
  if (!w || !map) return;
  const all = w.save.stations;
  const built = all.filter((s) => s.built);
  const at = built.length ? built : all;
  if (atHome(home.peek(), at.map((s) => s.lng), at.map((s) => s.lat))) return;
  if (!at.length) return viewStart([...(openedZones.peek() ?? [])][0]);
  openAt(at.map((s) => s.lng), at.map((s) => s.lat));
}

/** T-1539: the same fit from a loading save's built stations (`stationsAhead`, x, y in `frame`), as
 * soon as they are read: the game's figures can show before its first state. `homeAt`: the save's home. */
export function fitAhead(frame: FrameSpec, xy: Float64Array, homeAt?: string) {
  const f = makeFrame(frame), lng: number[] = [], lat: number[] = [];
  for (let k = 0; k + 1 < xy.length; k += 2) {
    const [x, y] = f.toLonLat(xy[k], xy[k + 1]);
    lng.push(x);
    lat.push(y);
  }
  if (atHome(homeAt ?? null, lng, lat)) return;
  if (lng.length) openAt(lng, lat);
}

/** T-2395: the camera at the game's home, if it has one: a zone's registry view, as the Cities tab's
 * double click, or for `WORLD_HOME` the stations at `lng`, `lat` fitted zoomed out far enough that
 * the top bar names the world (the world map with none). False: no home, so the caller fits. */
function atHome(at: string | null, lng: number[], lat: number[]): boolean {
  if (at === WORLD_HOME) {
    if (!lng.length) live.map?.jumpTo(WORLD_VIEW);
    else live.map?.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 70, maxZoom: HOME_WORLD_ZOOM, duration: 0 });
    return true;
  }
  const c = at ? CITIES.find((x) => x.id === at) : undefined;
  if (!c) return false;
  activeCity.value = c.id;
  viewedZone.value = c.id;
  live.map?.jumpTo({ center: c.center, zoom: c.zoom });
  return true;
}

/** Closest the `WORLD_HOME` start goes: just under where the top bar turns from the world to a city. */
const HOME_WORLD_ZOOM = WORLD_ZOOM - 0.25;

/** T-2395: the all cities view's go-to: every built station (all of them if none is built) in view,
 * zoomed out as a `WORLD_HOME` start is. */
export function locateAllCities() {
  const all = world.peek()?.save.stations ?? [];
  const built = all.filter((s) => s.built);
  const at = built.length ? built : all;
  atHome(WORLD_HOME, at.map((s) => s.lng), at.map((s) => s.lat));
}

/** The city view (as the Cities tab's pick) of the zone most of the points are in; all of them in view when none is in a zone. */
function openAt(lng: number[], lat: number[]) {
  const count = new Map<string, number>();
  for (let i = 0; i < lng.length; i++) {
    const z = placeAt(lng[i], lat[i]);
    if (z) count.set(z, (count.get(z) ?? 0) + 1);
  }
  let best: string | null = null, most = 0;
  for (const [z, n] of count) if (n > most) [best, most] = [z, n];
  const c = best ? CITIES.find((x) => x.id === best) : undefined;
  if (!c) return fitAll(lng, lat);
  activeCity.value = c.id;
  viewedZone.value = c.id;
  live.map?.jumpTo({ center: c.center, zoom: c.zoom });
}

function fitAll(lng: number[], lat: number[]) {
  live.map?.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 70, maxZoom: OPEN_MAX_ZOOM, duration: 0 });
}

/** T-2269: the map fitted to track nodes (a blueprint stretch's ends); one node as a station. */
export function locateNodes(ids: readonly number[]) {
  const map = live.map, w = world.peek();
  const at = ids.flatMap((id) => w?.nodes.get(id) ?? []);
  if (!map || !at.length) return;
  const lng = at.map((n) => n.lng), lat = at.map((n) => n.lat);
  if (Math.max(...lng) - Math.min(...lng) + Math.max(...lat) - Math.min(...lat) < 1e-4) return locateStation(lng[0], lat[0]);
  map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 60, maxZoom: 15, duration: 0 });
  const c = map.getCenter();
  activeCity.value = zoneAt(c.lng, c.lat);
}

/** The map fitted to a line's stops. */
export function locateLine(stopIds: readonly string[]) {
  const map = live.map;
  const w = world.peek(), at = w ? stopsById(w) : null;
  const stops = stopIds.flatMap((id) => at?.get(id) ?? []);
  if (!map || !stops.length) return;
  if (stops.length === 1) return locateStation(stops[0].lng, stops[0].lat);
  const lng = stops.map((s) => s.lng), lat = stops.map((s) => s.lat);
  map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 60, maxZoom: 15, duration: 0 });
  const c = map.getCenter();
  activeCity.value = zoneAt(c.lng, c.lat);
}
