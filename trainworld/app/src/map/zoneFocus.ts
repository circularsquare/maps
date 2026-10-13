// The focus follows the zone under the view (T-276, docs/zones.md 6): after a pan the focused zone
// (`activeCity`) is the one whose boundary holds the view's centre, the nearest boundary over dead
// space, except while a bubble, station or trip selection holds the focus (its cell ids belong to
// one zone, and a change of focus would clear it). The zones beside it whose boundaries are on
// screen (`nearbyZones`, at most three, nearest first) show their bubbles too, while Homes or
// Destinations is on and the map is not zoomed out to the one-bubble-per-city overview.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity } from "../game/cities";
import { display, selection } from "../game/ui";
import { loadZones, nearbyZones, zoneAt, zonesInBox, zonesLoaded } from "../game/zones";
import { OVERVIEW_ZOOM } from "./intercityView";

export function startZoneFocus(map: MlMap) {
  const place = (moved: boolean) => {
    const c = map.getCenter();
    const k = selection.peek()?.kind;
    if (moved && k !== "commuters" && k !== "station" && k !== "trip") activeCity.value = zoneAt(c.lng, c.lat);
    nearby();
  };
  const nearby = () => {
    const focus = activeCity.peek();
    let next: string[] = [];
    if (display.peek().commuters && map.getZoom() > OVERVIEW_ZOOM) {
      const c = map.getCenter(), b = map.getBounds();
      next = zonesInBox(focus, c.lng, c.lat, b.getWest(), b.getSouth(), b.getEast(), b.getNorth());
    }
    const prev = nearbyZones.peek();
    if (next.length !== prev.length || next.some((id, i) => id !== prev[i])) nearbyZones.value = next;
  };
  map.on("moveend", () => place(true));
  // boundaries arriving place the view again; Homes or Destinations turned on or off, or a focus
  // set elsewhere (search, a nearby bubble's click), recompute the nearby zones
  effect(() => {
    if (zonesLoaded.value) place(true);
  });
  effect(() => {
    display.value.commuters;
    activeCity.value;
    nearby();
  });
  void loadZones();
}
