// The MapLibre map. Its basemap style (Positron, faint, labels in Zen Maru Gothic) is in
// map/basemap.ts. Station names are DOM labels above the overlay (map/stationLabels.ts).

import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 looks for its worker beside its own module, which Vite bundles away; hand it the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { loadBasemap } from "./basemap";
import { styleUp } from "./styleReady";
import { cityById, cityWithin, hashView } from "../game/cities";
import { display } from "../game/ui";
import { effect } from "@preact/signals";

maplibregl.setWorkerUrl(workerUrl);

/** T-310: a reload does not restore the exact view in the URL hash. If the hash holds a view, the
 * map starts at the registry centre and zoom of the city or zone it is in; with no hash (a new
 * game, a first visit) it starts at the default. A hash over dead space (T-318) is kept as it is.
 * The hash is cleared so MapLibre writes the new one. */
function startView(): { center: [number, number]; zoom: number } {
  const v = hashView();
  if (!v) return { center: [-73.96, 40.75], zoom: 11.2 };
  try { history.replaceState(null, "", location.pathname + location.search); } catch { /* the hash stays; MapLibre then restores it */ }
  const id = cityWithin(v.lng, v.lat);
  if (!id) return { center: [v.lng, v.lat], zoom: v.zoom };
  const c = cityById(id);
  return { center: c.center, zoom: c.zoom };
}

export function createMap(container: HTMLElement): maplibregl.Map {
  const map = new maplibregl.Map({
    container,
    ...startView(),
    hash: true,
    // T-924: GISCO's terms want the EuroGeographics notice on the map itself (European zone units)
    attributionControl: {
      compact: true,
      customAttribution: [
        '<a href="https://openfreemap.org">OpenFreeMap</a> <a href="https://www.openmaptiles.org/">© OpenMapTiles</a> Data from <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        "© EuroGeographics for the administrative boundaries",
      ],
    },
    dragRotate: false,
    pitchWithRotate: false,
  });
  map.touchZoomRotate.disableRotation();
  wheelGapsByEventTime(map);
  loadBasemap(map);
  map.on("load", () => {
    document.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
  });
  followProjection(map);
  return map;
}

/** T-033: the Settings choice of the flat map or the globe, MapLibre's "globe" (a globe out to zoom
 * 11, blending into the flat map by 12). A new style (a lost context) takes it again. */
function followProjection(map: maplibregl.Map) {
  let styled = false;
  const apply = () => {
    if (!styled || !styleUp(map)) return;
    const want = display.peek().projection === "globe" ? "globe" : "mercator";
    if ((map.getProjection()?.type ?? "mercator") !== want) map.setProjection({ type: want });
  };
  map.on("style.load", () => {
    styled = true;
    apply();
  });
  effect(() => {
    void display.value.projection;
    apply();
  });
}

/** T-675: MapLibre's scroll zoom tells a mouse wheel from a trackpad by the time between wheel
 * events, measured when each is handled (`performance.now()`), not when it happened. A main-thread
 * stall as a scroll starts (a demand publish, 70-110 ms) queues its first two events, they are
 * handled under a millisecond apart, and the mouse wheel is taken for a trackpad for the rest of
 * the scroll: each frame then zooms straight to its target at 4.5 times the wheel's rate, about
 * 3x as far and with no easing. A stall over 400 ms mid-scroll likewise reads as a new scroll.
 * The handler is handed the gap between the events' own timestamps instead: it stores the last
 * event's handling time in `_lastWheelEventTime` (MapLibre 6.13) and takes the gap from it, so
 * that is set back by the real gap before each event. checks/t675-wheel-lag.mjs. */
function wheelGapsByEventTime(map: maplibregl.Map) {
  const h = map.scrollZoom as unknown as { wheel(e: WheelEvent): void; isEnabled(): boolean; _lastWheelEventTime?: number };
  const wheel = h.wheel;
  let lastStamp: number | null = null;
  h.wheel = function (e: WheelEvent) {
    if (h.isEnabled()) {
      if (lastStamp !== null && typeof h._lastWheelEventTime === "number") {
        h._lastWheelEventTime = performance.now() - Math.max(0, e.timeStamp - lastStamp);
      }
      lastStamp = e.timeStamp;
    }
    wheel.call(h, e);
  };
}
