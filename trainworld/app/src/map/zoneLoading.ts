// T-343: a zone whose demand has not arrived is drawn grey inside its boundary, so it reads as
// "nothing known here yet" rather than "no demand", with a small bar over it filling as it loads
// (hover: "Loading Toronto, 40%"). Grey: every zone being loaded (game/loading.ts `zoneLoads`), and
// a nearby zone whose bubbles would show but has not been opened yet (it opens after the startup
// load). The shade goes in one step when the zone's figures arrive. A zone re-solving after an
// edit keeps its figures and is never grey. The bars hide while the startup card is up, which says
// the same thing in the middle of the map.

import { effect } from "@preact/signals";
import { h, render } from "preact";
import type { FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { cityBoundary } from "../game/boundaries";
import { cityById, type CityId } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { loadFailure, zoneLoads } from "../game/loading";
import { nearbyZones } from "../game/zones";
import { LoadBar, loadingCardShown } from "../ui/LoadingScreen";
import { cityGeometry } from "./boundaryGeometry";

const SHADE = "tw-zone-loading";
/** how far a bar keeps from the map's edges, px */
const EDGE = 48;
/** how far under the zone's centre a bar sits, px */
const BELOW = 24;

type Box = [number, number, number, number];

export function startZoneLoading(map: MlMap): () => void {
  let alive = true;
  const shapes = new Map<CityId, { geometry: MultiPolygon; box: Box }>();
  const asked = new Set<CityId>();
  let grey: CityId[] = [];
  let loads: ReadonlyMap<CityId, number> = new Map();

  const layer = document.createElement("div");
  layer.className = "zone-loads";
  map.getContainer().appendChild(layer);
  const bars = new Map<CityId, HTMLDivElement>();

  const shape = (id: CityId) => {
    const s = shapes.get(id);
    if (s || asked.has(id)) return s;
    asked.add(id);
    cityBoundary(id).then((fc) => {
      if (!alive) return;
      const geometry = cityGeometry(fc);
      const box: Box = [Infinity, Infinity, -Infinity, -Infinity];
      for (const poly of geometry.coordinates) for (const [x, y] of poly[0]) {
        box[0] = Math.min(box[0], x); box[1] = Math.min(box[1], y);
        box[2] = Math.max(box[2], x); box[3] = Math.max(box[3], y);
      }
      shapes.set(id, { geometry, box });
      draw();
    }, () => asked.delete(id));
    return undefined;
  };

  /** Each bar at its zone's centre, held inside the part of the zone's extent that is on screen. */
  const place = () => {
    const c = map.getContainer(), w = c.clientWidth, ht = c.clientHeight;
    for (const [id, el] of bars) {
      const s = shapes.get(id);
      if (!s) { el.style.display = "none"; continue; }
      const [x0, y0, x1, y1] = s.box;
      const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([x, y]) => map.project([x, y]));
      const left = Math.max(EDGE, Math.min(...corners.map((p) => p.x)));
      const right = Math.min(w - EDGE, Math.max(...corners.map((p) => p.x)));
      const top = Math.max(EDGE, Math.min(...corners.map((p) => p.y)));
      const bottom = Math.min(ht - EDGE, Math.max(...corners.map((p) => p.y)));
      if (left > right || top > bottom) { el.style.display = "none"; continue; }
      // a little under the centre, clear of the basemap's name for the place
      const p = map.project(cityById(id).center);
      const x = Math.min(right, Math.max(left, p.x)), y = Math.min(bottom, Math.max(top, p.y + BELOW));
      el.style.display = "";
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
  };

  const draw = () => {
    const fc: FeatureCollection<MultiPolygon> = { type: "FeatureCollection", features: [] };
    for (const id of grey) {
      const s = shape(id);
      if (s) fc.features.push({ type: "Feature", properties: { id }, geometry: s.geometry });
    }
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(fc);
    const cardUp = loadingCardShown();
    for (const [id, el] of bars) if (!loads.has(id) || cardUp) { render(null, el); el.remove(); bars.delete(id); }
    if (!cardUp) for (const [id, f] of loads) {
      shape(id);
      let el = bars.get(id);
      if (!el) {
        el = document.createElement("div");
        el.className = "zone-load";
        layer.appendChild(el);
        bars.set(id, el);
      }
      const tip = `Loading ${cityById(id).name}, ${Math.floor(f * 100)}%`;
      render(h(LoadBar, { fraction: f, step: tip, thin: true, tip }), el);
    }
    place();
  };

  const add = () => {
    if (!map.getSource(SHADE)) map.addSource(SHADE, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    if (!map.getLayer(SHADE)) {
      // over the basemap and the outside shade, under its labels and the network
      const before = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
      map.addLayer({
        id: SHADE, type: "fill", source: SHADE,
        paint: { "fill-color": "#8a8f99", "fill-opacity": 0.32, "fill-antialias": false },
      }, before);
    }
    draw();
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  map.on("move", place);
  map.on("resize", place);

  const dispose = effect(() => {
    loads = loadFailure.value ? new Map() : zoneLoads.value;
    const shown = cityDemandViews.value;
    const ids = new Set(loads.keys());
    if (!loadFailure.value) for (const id of nearbyZones.value) if (!shown.has(id)) ids.add(id);
    loadingCardShown(); // the bars follow the card
    grey = [...ids];
    draw();
  });

  return () => {
    alive = false;
    dispose();
    map.off("style.load", add);
    map.off("move", place);
    map.off("resize", place);
    for (const el of bars.values()) render(null, el);
    layer.remove();
    if (map.getLayer(SHADE)) map.removeLayer(SHADE);
    if (map.getSource(SHADE)) map.removeSource(SHADE);
  };
}
