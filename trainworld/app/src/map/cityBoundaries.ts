// Exact pack outlines and a subtle shade outside the focused metropolitan area. Basemap-only
// layers sit beneath labels, track and demand; they never intercept a map click.
import { effect } from "@preact/signals";
import type { FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { cityBoundary } from "../game/boundaries";
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
      p = cityBoundary(id).then(cityGeometry);
      geometries.set(id, p);
      void p.catch(() => geometries.delete(id));
    }
    return p;
  };
  // The footprints that have arrived with their shade shapes: a change of focus swaps one for the
  // other in a single setData, so the veil is never empty between them (T-310: the empty step drew
  // a frame without the veil each time a pan ended on another zone).
  const ready = new Map<string, { city: MultiPolygon; outside: MultiPolygon }>();
  const show = (id: string, city: MultiPolygon) => {
    let r = ready.get(id);
    if (!r) ready.set(id, (r = { city, outside: outsideCity(city) }));
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(r.outside);
    (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(r.city);
  };
  const focus = async (id: string) => {
    const mine = ++request;
    for (const city of CITIES) {
      const layer = `tw-city-boundary-${city.id}`;
      if (map.getLayer(layer)) map.setLayoutProperty(layer, "visibility", city.id === id ? "none" : "visible");
    }
    const known = ready.get(id);
    if (known) return show(id, known.city);
    // Never shade against the previous city while the new footprint is loading.
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(empty);
    (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(empty);
    try {
      const city = await geometry(id);
      if (!alive || mine !== request) return;
      show(id, city);
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
      if (!map.getSource(id)) {
        // the same fetch as the demand pool's reach test (game/boundaries.ts), not one of its own
        map.addSource(id, { type: "geojson", data: empty });
        cityBoundary(city.id).then((fc) => {
          if (alive) (map.getSource(id) as GeoJSONSource | undefined)?.setData(fc);
        }, (error) => console.warn("City outline unavailable", error));
      }
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
    // every footprint made ready now, so no later change of focus has to wait for one
    for (const city of CITIES) {
      void geometry(city.id).then((g) => {
        if (alive && !ready.has(city.id)) ready.set(city.id, { city: g, outside: outsideCity(g) });
      }, () => {});
    }
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
