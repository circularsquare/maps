// The MapLibre map. Its basemap style (Positron, faint, labels in Zen Maru Gothic) is in
// map/basemap.ts. Station names are DOM labels above the overlay (map/stationLabels.ts).

import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 looks for its worker beside its own module, which Vite bundles away; hand it the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { loadBasemap } from "./basemap";

maplibregl.setWorkerUrl(workerUrl);

export { setBasemapLabels } from "./basemap";

export function createMap(container: HTMLElement): maplibregl.Map {
  const map = new maplibregl.Map({
    container,
    center: [-73.96, 40.75],
    zoom: 11.2,
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
