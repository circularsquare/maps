// Exact pack outlines and a subtle shade outside the focused metropolitan area. Basemap-only
// layers sit beneath labels, track and demand; they never intercept a map click.
import { effect } from "@preact/signals";
import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { activeCity, CITIES } from "../game/cities";
import { cityGeometry, outsideCity } from "./boundaryGeometry";

const SHADE = "tw-city-outside";
const FOCUS = "tw-city-focus";
const INNER = "tw-city-focus-inner";
const OUTER = "tw-city-focus-outer";
const empty: FeatureCollection = { type: "FeatureCollection", features: [] };

/** Install the registered packs' outlines, including after a basemap style replacement. */
export function startCityBoundaries(map: MlMap): () => void {
  let alive = true;
  let request = 0;
  const geometries = new Map<string, Promise<MultiPolygon>>();
  const geometry = (id: string) => {
    let p = geometries.get(id);
    if (!p) {
      p = fetch(new URL(`packs/${id}.boundary.geojson`, document.baseURI))
        .then(async r => {
          if (!r.ok) throw new Error(`City boundary ${id}: ${r.status}`);
          return cityGeometry(await r.json() as FeatureCollection<Polygon | MultiPolygon>);
        });
      geometries.set(id, p);
      void p.catch(() => geometries.delete(id));
    }
    return p;
  };
  const focus = async (id: string) => {
    const mine = ++request;
    // Never shade against the previous city while the new footprint is loading.
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(empty);
    (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(empty);
    for (const city of CITIES) {
      const layer = `tw-city-boundary-${city.id}`;
      if (map.getLayer(layer)) map.setLayoutProperty(layer, "visibility", city.id === id ? "none" : "visible");
    }
    try {
      const city = await geometry(id);
      if (!alive || mine !== request) return;
      (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(outsideCity(city));
      (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(city);
    } catch (error) {
      if (alive && mine === request) console.warn("City focus boundary unavailable", error);
    }
  };
  const add = () => {
    const before = map.getStyle().layers.find((layer) => layer.type === "symbol")?.id;
    if (!map.getSource(SHADE)) map.addSource(SHADE, { type: "geojson", data: empty });
    if (!map.getLayer(SHADE)) map.addLayer({
      id: SHADE, type: "fill", source: SHADE,
      paint: { "fill-color": "#182536", "fill-opacity": 0.075, "fill-antialias": false },
    }, before);
    for (const city of CITIES) {
      const id = `tw-city-boundary-${city.id}`;
      if (!map.getSource(id)) map.addSource(id, {
        type: "geojson",
        data: new URL(`packs/${city.id}.boundary.geojson`, document.baseURI).href,
      });
      if (!map.getLayer(id)) map.addLayer({
        id,
        type: "line",
        source: id,
        maxzoom: 10.5,
        paint: {
          "line-color": "#73798b",
          "line-width": 0.8,
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 8.5, 0.5, 10.5, 0],
        },
      }, before);
    }
    if (!map.getSource(FOCUS)) map.addSource(FOCUS, { type: "geojson", data: empty });
    // MapLibre polygon strokes use positive offsets for insets and negative for outsets.
    for (const [id, offset, opacity] of [[INNER, 0.45, 0.18], [OUTER, -0.45, 0.48]] as const) {
      if (!map.getLayer(id)) map.addLayer({
        id, type: "line", source: FOCUS,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": "#657082", "line-width": 0.75, "line-offset": offset,
          "line-opacity": opacity,
        },
      }, before);
    }
    void focus(activeCity.peek());
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  const dispose = effect(() => {
    const id = activeCity.value;
    if (map.getSource(SHADE)) void focus(id);
  });
  return () => {
    alive = false;
    request++;
    dispose();
    map.off("style.load", add);
    const ids = [INNER, OUTER, FOCUS, SHADE, ...CITIES.map(c => `tw-city-boundary-${c.id}`)];
    for (const id of ids) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of ids) if (map.getSource(id)) map.removeSource(id);
  };
}
