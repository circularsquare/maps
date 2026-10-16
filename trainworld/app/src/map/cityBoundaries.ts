// Exact pack outlines and a subtle shade outside the focused metropolitan area. Basemap-only
// layers sit beneath labels, track and demand; they never intercept a map click.
// T-400: one source and one line layer hold the outlines of the zones near the view, fetched as the
// view comes near them (their registry boxes say where they are); a zone's footprint and outside
// shade are built when it is focused or near the view, never for every registered zone.
import { effect } from "@preact/signals";
import type { Feature, FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { cityBoundary } from "../game/boundaries";
import { activeCity, zonesTouching } from "../game/cities";
import { outsidePaint, SHADES, shadeColours } from "../game/shades";
import { shade } from "../game/ui";
import { cityGeometry, outsideCity } from "./boundaryGeometry";

const SHADE = "tw-city-outside";
const FOCUS = "tw-city-focus";
const INNER = "tw-city-focus-inner";
const OUTER = "tw-city-focus-outer";
export const OUTLINES = "tw-city-boundaries";
/** outlines fade out by this zoom (the layer's maxzoom) */
const OUTLINE_MAX_ZOOM = 10.5;
/** below this zoom a view spans so many zones that their outlines are not fetched for it */
const OUTLINE_MIN_ZOOM = 5;
const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
/** The one shade of "not the zone in view": outside the focused zone, and (T-350) a zone still
 * loading; the theme's own (game/shades.ts, T-543). */
export const shadePaint = () => ({ ...outsidePaint(shadeColours()) });
/** Keeps layer `id`'s fill in the current theme's shade, from now on. */
export function followShade(map: MlMap, id: string): () => void {
  return effect(() => {
    const paint = outsidePaint(SHADES[shade.value]);
    if (!map.getLayer(id)) return;
    map.setPaintProperty(id, "fill-color", paint["fill-color"]);
    map.setPaintProperty(id, "fill-opacity", paint["fill-opacity"]);
  });
}

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
  const prepared = (id: string, city: MultiPolygon) => {
    let r = ready.get(id);
    if (!r) ready.set(id, (r = { city, outside: outsideCity(city) }));
    return r;
  };
  const show = (id: string, city: MultiPolygon) => {
    const r = prepared(id, city);
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(r.outside);
    (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(r.city);
  };
  const focus = async (id: string) => {
    const mine = ++request;
    if (map.getLayer(OUTLINES)) map.setFilter(OUTLINES, ["!=", ["get", "id"], id]);
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

  // the outlines drawn, by zone; the source is set again as each arrives
  const outlines = new Map<string, Feature<MultiPolygon>>();
  const asked = new Set<string>();
  const near = () => {
    const b = map.getBounds();
    return zonesTouching(b.getWest(), b.getSouth(), b.getEast(), b.getNorth()).map((c) => c.id);
  };
  const follow = () => {
    const z = map.getZoom();
    if (z < OUTLINE_MIN_ZOOM) return;
    for (const id of near()) {
      // a zone near the view has its shade ready, so focusing it does not wait
      void geometry(id).then((g) => { if (alive) prepared(id, g); }, () => {});
      if (z > OUTLINE_MAX_ZOOM || asked.has(id)) continue;
      asked.add(id);
      geometry(id).then((g) => {
        if (!alive) return;
        outlines.set(id, { type: "Feature", properties: { id }, geometry: g });
        (map.getSource(OUTLINES) as GeoJSONSource | undefined)?.setData({ type: "FeatureCollection", features: [...outlines.values()] });
      }, (error) => { asked.delete(id); console.warn("City outline unavailable", error); });
    }
  };

  const add = () => {
    const before = map.getStyle().layers.find((layer) => layer.type === "symbol")?.id;
    if (!map.getSource(SHADE)) map.addSource(SHADE, { type: "geojson", data: empty });
    if (!map.getLayer(SHADE)) map.addLayer({
      id: SHADE, type: "fill", source: SHADE,
      paint: shadePaint(),
    }, before);
    if (!map.getSource(OUTLINES)) map.addSource(OUTLINES, { type: "geojson", data: { type: "FeatureCollection", features: [...outlines.values()] } });
    if (!map.getLayer(OUTLINES)) map.addLayer({
      id: OUTLINES,
      type: "line",
      source: OUTLINES,
      maxzoom: OUTLINE_MAX_ZOOM,
      paint: {
        "line-color": "#73798b",
        "line-width": 0.8,
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 8.5, 0.5, 10.5, 0],
      },
    }, before);
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
    follow();
  };
  map.on("style.load", add);
  map.on("moveend", follow);
  if (map.isStyleLoaded()) add();
  const dispose = effect(() => {
    const id = activeCity.value;
    if (map.getSource(SHADE)) void focus(id);
  });
  const disposeShade = followShade(map, SHADE);
  return () => {
    alive = false;
    request++;
    dispose();
    disposeShade();
    map.off("style.load", add);
    map.off("moveend", follow);
    const ids = [INNER, OUTER, FOCUS, SHADE, OUTLINES];
    for (const id of ids) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of ids) if (map.getSource(id)) map.removeSource(id);
  };
}
