// The focus follows the zone under the view (T-276, docs/zones.md 6): after a pan the focused zone
// (`activeCity`) is the one whose boundary holds the view's centre, or lies within reach of it;
// over dead space (T-318, `placeAt`) the focus stays and `viewedZone` is null, except while a bubble, station or trip selection holds the focus (its cell ids belong to
// one zone, and a change of focus would clear it). The zones beside it whose boundaries are on
// screen (`nearbyZones`, at most three, nearest first) show their bubbles too, while Homes or
// Destinations is on and the map is not zoomed out to the one-bubble-per-city overview.

import { batch, effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, viewedZone, ZONE_REACH_M, zonesTouching } from "../game/cities";
import { display, selection } from "../game/ui";
import { viewCentre } from "../game/listScope";
import { nearbyZones, placeAt, ZONE_LOAD_ZOOM, zonesInBox, zonesLoaded, zonesLoadForView } from "../game/zones";
import { OVERVIEW_ZOOM } from "./intercityView";
import { shiftAt, viewShift } from "../game/localTime";
import { WORLD_ZOOM, worldView } from "../game/worldTotals";
import { shownRealms, zoneShown } from "../game/realms";

export function startZoneFocus(map: MlMap) {
  const place = (moved: boolean) => {
    const c = map.getCenter();
    const k = selection.peek()?.kind;
    // T-318: over dead space the focus stays where it was and nothing is viewed (no name, no load);
    // T-975: a realm not shown is dead space too, asked before any boundary there is fetched
    const hidden = shownRealms.peek() && !zonesTouching(c.lng, c.lat, c.lng, c.lat, ZONE_REACH_M).some((z) => zoneShown(z.id));
    const at = hidden ? null : placeAt(c.lng, c.lat);
    const here = at && zoneShown(at) ? at : null;
    batch(() => {
      viewCentre.value = [c.lng, c.lat]; // T-332: the Lines and Stations lists sort by it
      if (here && moved && k !== "commuters" && k !== "station" && k !== "trip") activeCity.value = here;
      viewedZone.value = here;
      zonesLoadForView.value = map.getZoom() > ZONE_LOAD_ZOOM;
    });
    nearby();
  };
  const nearby = () => {
    const focus = activeCity.peek();
    let next: string[] = [];
    if (display.peek().commuters && map.getZoom() > OVERVIEW_ZOOM) {
      const c = map.getCenter(), b = map.getBounds();
      next = zonesInBox(focus, c.lng, c.lat, b.getWest(), b.getSouth(), b.getEast(), b.getNorth(), zoneShown);
    }
    const prev = nearbyZones.peek();
    if (next.length !== prev.length || next.some((id, i) => id !== prev[i])) nearbyZones.value = next;
  };
  map.on("moveend", () => place(true));
  { const c = map.getCenter(); viewCentre.value = [c.lng, c.lat]; }
  // T-309: the clock shows the local time in view, switching as the view crosses a time zone
  const local = () => {
    const c = map.getCenter(), s = shiftAt(c.lng, c.lat);
    if (s !== viewShift.peek()) viewShift.value = s;
    const far = map.getZoom() < WORLD_ZOOM; // T-1110: the top bar names the world
    if (far !== worldView.peek()) worldView.value = far;
  };
  map.on("move", local);
  local();
  // boundaries arriving place the view again; Homes or Destinations turned on or off, or a focus
  // set elsewhere (search, a nearby bubble's click), recompute the nearby zones
  effect(() => {
    if (zonesLoaded.value) place(true);
  });
  // T-975: a realm shown or hidden places the view again
  let realms = shownRealms.peek();
  effect(() => {
    if (shownRealms.value !== realms) { realms = shownRealms.peek(); place(true); }
  });
  effect(() => {
    display.value.commuters;
    activeCity.value;
    nearby();
  });
  place(false); // fetches the boundaries near the view (T-404)
}
