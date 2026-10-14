// The MapLibre map. Its basemap style (Positron, faint, labels in Zen Maru Gothic) is in
// map/basemap.ts. Station names are DOM labels above the overlay (map/stationLabels.ts).

import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 looks for its worker beside its own module, which Vite bundles away; hand it the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { loadBasemap } from "./basemap";
import { cityById, cityWithin, hashView } from "../game/cities";

maplibregl.setWorkerUrl(workerUrl);

export { setBasemapLabels } from "./basemap";

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
    attributionControl: { compact: true },
    dragRotate: false,
    pitchWithRotate: false,
  });
  map.touchZoomRotate.disableRotation();
  loadBasemap(map);
  map.on("load", () => {
    document.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
  });
  return map;
}

export { maplibregl };
