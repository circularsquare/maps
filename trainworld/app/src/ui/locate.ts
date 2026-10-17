// Moving the camera to a station or a line (T-569, and search's picks, T-148): the Stations and
// Lines tabs' double click and the inspectors' locate button. A jump, never a flight (T-182). It
// only moves the camera; selecting is the caller's.

import { activeCity } from "../game/cities";
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
