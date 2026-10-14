// The MapLibre map. Its basemap style (Positron, faint, labels in Zen Maru Gothic) is in
// map/basemap.ts. Station names are DOM labels above the overlay (map/stationLabels.ts).

import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 looks for its worker beside its own module, which Vite bundles away; hand it the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { loadBasemap } from "./basemap";
import { cityById, cityNear } from "../game/cities";

maplibregl.setWorkerUrl(workerUrl);

export { setBasemapLabels } from "./basemap";

/** T-310: a reload does not restore the exact view in the URL hash. If the hash holds a view, the
 * map starts at the registry centre and zoom of the city or zone nearest to it; with no hash (a new
 * game, a first visit) it starts at the default. The hash is cleared so MapLibre writes the new one. */
function startView(): { center: [number, number]; zoom: number } {
  const m = /^#(-?[\d.]+)\/(-?[\d.]+)\/(-?[\d.]+)/.exec(location.hash);
  const zoom = m ? Number(m[1]) : NaN, lat = m ? Number(m[2]) : NaN, lng = m ? Number(m[3]) : NaN;
  if (!Number.isFinite(zoom) || !Number.isFinite(lat) || !Number.isFinite(lng)) return { center: [-73.96, 40.75], zoom: 11.2 };
  try { history.replaceState(null, "", location.pathname + location.search); } catch { /* the hash stays; MapLibre then restores it */ }
  const c = cityById(cityNear(lng, lat));
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
