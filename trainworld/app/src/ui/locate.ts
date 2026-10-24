// Moving the camera to a station or a line (T-569, and search's picks, T-148): the Stations and
// Lines tabs' double click and the inspectors' locate button. A jump, never a flight (T-182). It
// only moves the camera; selecting is the caller's.

import { effect, untracked } from "@preact/signals";
import { activeCity, CITIES, viewedZone } from "../game/cities";
import { fitView } from "../game/loading";
import { openedZones } from "../game/openedZones";
import { display } from "../game/ui";
import { world } from "../game/world";
import { zoneAt } from "../game/zones";
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
  // T-801: a start on the map is picked from the zones' zoomed-out bubbles, so Homes goes on
  if (!c) display.value = { ...display.peek(), commuters: true, commuterEnd: "home" };
}

/** Furthest in a saved game opens: a district, so a lone station is not shown at street level (T-895). */
const OPEN_MAX_ZOOM = 12;

/** T-895: a game the player opened (start menu, saves panel, imported file) has its built stations all
 * in view (every station, if none is built); with no stations, the zone it was started in, else the
 * world map. Runs when the game's first state arrives (`fitView`), which is before its demand. */
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
  if (!at.length) return viewStart([...(openedZones.peek() ?? [])][0]);
  const lng = at.map((s) => s.lng), lat = at.map((s) => s.lat);
  map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 70, maxZoom: OPEN_MAX_ZOOM, duration: 0 });
}

/** The map fitted to a line's stops. */
export function locateLine(stopIds: readonly string[]) {
  const map = live.map;
  const stops = (world.peek()?.save.stations ?? []).filter((s) => stopIds.includes(s.id));
  if (!map || !stops.length) return;
  if (stops.length === 1) return locateStation(stops[0].lng, stops[0].lat);
  const lng = stops.map((s) => s.lng), lat = stops.map((s) => s.lat);
  map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 60, maxZoom: 15, duration: 0 });
  const c = map.getCenter();
  activeCity.value = zoneAt(c.lng, c.lat);
}
