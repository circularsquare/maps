// T-343: a zone whose demand has not arrived is drawn grey inside its boundary, so it reads as
// "nothing known here yet" rather than "no demand", with a small bar over it filling as it loads
// (hover: "Loading Toronto, 40%"). Grey: every zone being loaded (game/loading.ts `zoneLoads`), and
// a nearby zone whose bubbles would show but has not been opened yet (it opens after the startup
// load). T-350: the grey is the outside shade's own (`shadePaint`), filled only on the zone in view,
// since every other zone already lies under the outside shade. The shade goes in one step when the
// zone's figures arrive. A zone re-solving after an
// edit keeps its figures and is never grey. The bars hide while the startup card is up, which says
// the same thing in the middle of the map.

import { effect } from "@preact/signals";
import { h, render } from "preact";
import type { FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { cityBoundary } from "../game/boundaries";
import { activeCity, cityById, viewedZone, type CityId } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { loadFailure, mapBlank, zoneLoads } from "../game/loading";
import { nearbyZones } from "../game/zones";
import { zoneShown } from "../game/realms";
import { LoadBar, loadingCardShown } from "../ui/LoadingScreen";
import { cityGeometry } from "./boundaryGeometry";
import { firstSymbolId } from "./basemap";
import { screenBounds } from "./labelLayout";
import { followShade, shadePaint } from "./cityBoundaries";
import { hiddenOnGlobe, live } from "./live";
import { mapCamera, PanPlane } from "./moving";

const SHADE = "tw-zone-loading";
/** how far a bar keeps from the map's edges, px */
const EDGE = 48;
/** how far under the zone's centre a bar sits, px */
const BELOW = 24;
const MAX_BARS = 3;

type Box = [number, number, number, number];

export function startZoneLoading(map: MlMap): () => void {
  let alive = true;
  const shapes = new Map<CityId, { geometry: MultiPolygon; box: Box }>();
  const asked = new Set<CityId>();
  let grey: CityId[] = [];
  let loads: ReadonlyMap<CityId, number> = new Map();

  const box = document.createElement("div");
  box.className = "zone-loads";
  map.getContainer().appendChild(box);
  const layer = document.createElement("div");
  layer.className = "mark-plane";
  box.appendChild(layer);
  // T-1207: placed in full at most every MOVING_PLACE_MS while the camera moves
  const pan = new PanPlane(mapCamera(map, () => live.overlay?.moving ?? false), layer);
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
  const place = (force?: unknown) => {
    if (force !== true && !pan.due()) return;
    pan.placed();
    const c = map.getContainer(), w = c.clientWidth, ht = c.clientHeight;
    for (const [id, el] of bars) {
      const s = shapes.get(id);
      if (!s) { el.style.display = "none"; continue; }
      const { left, right, top, bottom } = screenBounds(map, s.box, w, ht, EDGE);
      if (left > right || top > bottom || hiddenOnGlobe(...cityById(id).center)) { el.style.display = "none"; continue; }
      // a little under the centre, clear of the basemap's name for the place
      const p = map.project(cityById(id).center);
      const x = Math.min(right, Math.max(left, p.x)), y = Math.min(bottom, Math.max(top, p.y + BELOW));
      el.style.display = "";
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
  };

  /** T-939: at most MAX_BARS bars: the zone in view, then the zones nearest the map's centre. */
  const barsShown = (): Map<CityId, number> => {
    if (loads.size <= MAX_BARS) return new Map(loads);
    const c = map.getContainer(), cx = c.clientWidth / 2, cy = c.clientHeight / 2;
    const focus = activeCity.value;
    const far = (id: CityId) => {
      if (id === focus) return -1;
      const p = map.project(cityById(id).center);
      return Math.hypot(p.x - cx, p.y - cy);
    };
    const picked = [...loads.keys()].sort((a, b) => far(a) - far(b)).slice(0, MAX_BARS);
    return new Map(picked.map((id) => [id, loads.get(id)!]));
  };

  const draw = () => {
    const fc: FeatureCollection<MultiPolygon> = { type: "FeatureCollection", features: [] };
    for (const id of grey) {
      const s = shape(id);
      if (s) fc.features.push({ type: "Feature", properties: { id }, geometry: s.geometry });
    }
    (map.getSource(SHADE) as GeoJSONSource | undefined)?.setData(fc);
    const cardUp = loadingCardShown();
    const shown = cardUp ? new Map<CityId, number>() : barsShown();
    for (const [id, el] of bars) if (!shown.has(id)) { render(null, el); el.remove(); bars.delete(id); }
    for (const [id, f] of shown) {
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
    place(true);
  };

  const add = () => {
    if (!map.getSource(SHADE)) map.addSource(SHADE, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    if (!map.getLayer(SHADE)) {
      // over the basemap and the outside shade, under its labels and the network
      const before = firstSymbolId(map);
      map.addLayer({
        id: SHADE, type: "fill", source: SHADE,
        paint: shadePaint(),
      }, before);
    }
    draw();
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  map.on("move", place);
  map.on("resize", place);
  const repick = () => { if (loads.size > MAX_BARS) draw(); else place(true); };
  map.on("moveend", repick);

  const dispose = effect(() => {
    // T-975: a realm not shown draws no loading grey or bar, though demand may be opening its packs
    loads = loadFailure.value ? new Map() : new Map([...zoneLoads.value].filter(([id]) => zoneShown(id)));
    const shown = cityDemandViews.value;
    const ids = new Set(loads.keys());
    if (!loadFailure.value) for (const id of nearbyZones.value) if (!shown.has(id) && zoneShown(id)) ids.add(id);
    loadingCardShown(); // the bars follow the card
    const blank = mapBlank.value; // T-1051: no grey either while a game loads
    // T-350: only the zone in view takes the fill: every other zone already lies under the outside
    // shade (map/cityBoundaries.ts), which is this same grey, and stacking would darken it
    const focus = activeCity.value;
    grey = blank ? [] : [...ids].filter((id) => id === focus);
    draw();
  });

  // ?debug=1 (T-1493's check): what says the zone in view is loading
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twZoneLoading = () => {
    const focus = activeCity.peek();
    return { focus, viewed: viewedZone.peek(), realmShown: zoneShown(focus), grey: grey.includes(focus), bar: bars.has(focus) && bars.get(focus)!.style.display !== "none", bars: bars.size, loading: loads.has(focus) };
  };

  const disposeShade = followShade(map, SHADE);
  return () => {
    alive = false;
    dispose();
    disposeShade();
    map.off("style.load", add);
    map.off("move", place);
    map.off("resize", place);
    map.off("moveend", repick);
    for (const el of bars.values()) render(null, el);
    box.remove();
    if (map.getLayer(SHADE)) map.removeLayer(SHADE);
    if (map.getSource(SHADE)) map.removeSource(SHADE);
  };
}
