// Exact pack outlines and a subtle shade outside the focused metropolitan area. Basemap-only
// layers sit beneath labels, track and demand; they never intercept a map click.
// T-400: one source and one line layer hold the outlines of the zones near the view, fetched as the
// view comes near them (their registry boxes say where they are); a zone's footprint and outside
// shade are built when it is focused or near the view, never for every registered zone.
import { effect } from "@preact/signals";
import type { FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { cityBoundary } from "../game/boundaries";
import { activeCity, zonesTouching } from "../game/cities";
import { ZONE_LOAD_ZOOM } from "../game/zones";
import { perfOff } from "../perfFlags";
import { outsidePaint, SHADES, shadeColours } from "../game/shades";
import { shade } from "../game/ui";
import { cityGeometry, OutlineEdges, outsideCity } from "./boundaryGeometry";
import { coarseOutlines } from "./coarseOutlines";

const SHADE = "tw-city-outside";
const FOCUS = "tw-city-focus";
const INNER = "tw-city-focus-inner";
const OUTER = "tw-city-focus-outer";
export const OUTLINES = "tw-city-boundaries";
/** outlines fade out by this zoom (the layer's maxzoom) */
const OUTLINE_MAX_ZOOM = 10.5;
/** below this zoom a view spans so many zones that their exact outlines are not fetched for it; the
 * coarse ones stand (T-871: the zoom zones load at, game/zones.ts) */
const OUTLINE_MIN_ZOOM = Math.max(5, ZONE_LOAD_ZOOM);
const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
const everywhere = outsideCity({ type: "MultiPolygon", coordinates: [] });
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
    redraw();
    const known = ready.get(id);
    if (known) return show(id, known.city);
    // While the new footprint loads, the shade covers everything, the new zone too: never against the
    // previous city, and never empty (T-904: an empty shade flashed the whole map pale until it came).
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(everywhere);
    (map.getSource(FOCUS) as GeoJSONSource | undefined)?.setData(empty);
    try {
      const city = await geometry(id);
      if (!alive || mine !== request) return;
      show(id, city);
    } catch (error) {
      if (alive && mine === request) console.warn("City focus boundary unavailable", error);
    }
  };

  // the outlines drawn, shared borders once; the source is set again as each zone arrives
  const edges = new OutlineEdges();
  const asked = new Set<string>();
  // zones drawn from the coarse file (T-800) until their own boundary arrives
  const coarse = new Set<string>();
  // T-871: only the zones within a view's width of the screen are drawn, so a redraw costs what is
  // near, not every outline fetched on a long pan
  let around = new Set<string>();
  const aroundView = () => {
    const b = map.getBounds(), w = b.getWest(), s = b.getSouth(), e = b.getEast(), n = b.getNorth();
    return new Set(zonesTouching(w - (e - w), s - (n - s), e + (e - w), n + (n - s)).map((c) => c.id));
  };
  const drawn = () => edges.lines(activeCity.peek(), perfOff("zoneZoom") ? undefined : around);
  const redraw = () => {
    if (edges.size) (map.getSource(OUTLINES) as GeoJSONSource | undefined)?.setData(drawn());
  };
  // zones arrive in a burst while panning: one redraw for the lot
  let pending = 0;
  const redrawSoon = () => {
    if (!pending) pending = window.setTimeout(() => { pending = 0; redraw(); }, 60);
  };
  void coarseOutlines().then(async (zones) => {
    let n = 0;
    // in slices of about 30 ms, so startup never waits on one long task
    for (const [id, g] of zones) {
      if (!alive) return;
      if (!edges.has(id)) {
        edges.add(id, g);
        coarse.add(id);
      }
      if (++n % 40 === 0) await new Promise((done) => setTimeout(done));
    }
    redraw();
  });
  const near = () => {
    const b = map.getBounds();
    return zonesTouching(b.getWest(), b.getSouth(), b.getEast(), b.getNorth()).map((c) => c.id);
  };
  const follow = () => {
    const next = aroundView();
    if (next.size !== around.size || [...next].some((id) => !around.has(id))) {
      around = next;
      redrawSoon();
    }
    const z = map.getZoom();
    if (z <= OUTLINE_MIN_ZOOM) return;
    for (const id of near()) {
      // a zone near the view has its shade ready, so focusing it does not wait
      void geometry(id).then((g) => { if (alive) prepared(id, g); }, () => {});
      if (z > OUTLINE_MAX_ZOOM || asked.has(id)) continue;
      asked.add(id);
      geometry(id).then((g) => {
        if (!alive) return;
        if (coarse.delete(id)) edges.remove(id);
        edges.add(id, g);
        redrawSoon();
      }, (error) => { asked.delete(id); console.warn("City outline unavailable", error); });
    }
  };

  const add = () => {
    const before = map.getStyle().layers.find((layer) => layer.type === "symbol")?.id;
    if (!map.getSource(SHADE)) map.addSource(SHADE, { type: "geojson", data: everywhere });
    if (!map.getLayer(SHADE)) map.addLayer({
      id: SHADE, type: "fill", source: SHADE,
      paint: shadePaint(),
    }, before);
    if (!map.getSource(OUTLINES)) map.addSource(OUTLINES, { type: "geojson", data: drawn() });
    if (!map.getLayer(OUTLINES)) map.addLayer({
      id: OUTLINES,
      type: "line",
      source: OUTLINES,
      maxzoom: OUTLINE_MAX_ZOOM,
      layout: { "line-join": "round", "line-cap": "round" },
      paint: {
        "line-color": "#73798b",
        "line-width": 0.7,
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 8.5, 0.36, 10.5, 0],
      },
    }, before);
    if (!map.getSource(FOCUS)) map.addSource(FOCUS, { type: "geojson", data: empty });
    // MapLibre polygon strokes use positive offsets for insets and negative for outsets.
    for (const [id, offset, opacity] of [[INNER, 0.45, 0.14], [OUTER, -0.45, 0.36]] as const) {
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
    window.clearTimeout(pending);
    dispose();
    disposeShade();
    map.off("style.load", add);
    map.off("moveend", follow);
    const ids = [INNER, OUTER, FOCUS, SHADE, OUTLINES];
    for (const id of ids) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of ids) if (map.getSource(id)) map.removeSource(id);
  };
}
