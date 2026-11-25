// The basemap style (T-009 mock, T-056, T-676): OpenFreeMap Positron repainted in the palette of
// game/shadeTable.ts (warm paper, teal water, sage parks, faint labels) with roads cut to three
// weights and no casings, so the network reads first; the basemap's own railways, road shields and
// minor road names are hidden (Positron has no other POI layers), and every label in Zen
// Maru Gothic from our own glyph tiles. Applied to the style JSON before MapLibre commits it, so
// nothing draws in Positron's colours or fonts first.
//
// Glyphs: `public/fonts/Zen Maru Gothic Regular,Noto Sans Regular/{range}.pbf`, made by
// pipeline/glyphs.py (notes/T-056.md). MapLibre fetches one file per font stack and 256-codepoint
// range, so the Noto fallback for scripts Zen Maru lacks is merged into those files. Kana, kanji
// and hangul never come from them: MapLibre draws those in the browser (localIdeographFontFamily).

import * as maplibregl from "maplibre-gl";
import { signal } from "@preact/signals";
import { SHADES, type ShadeColours } from "../game/shades";
import { display, SAT_INK_FLIP, shade, type Shade } from "../game/ui";
import { styleUp } from "./styleReady";
import { busRoadFilter, fastRoadExpr, railRoadFilter } from "./busRoads";
import { townGeometry } from "./townArea";
import { demBytes, flattenDem, onRedraw } from "./demFlat";
import { canvasReadsTrue } from "./canvasRead";
import { OPENFREEMAP, TILEJSON_PROTOCOL, openMapTiles, routedTileJson } from "./openMapTiles";

const POSITRON = `${OPENFREEMAP}styles/positron`;
/** The one font stack every basemap label uses: Regular only, Zen Maru has no italic. */
const FONT = ["Zen Maru Gothic Regular", "Noto Sans Regular"];
const RTL_PLUGIN = "rtl/mapbox-gl-rtl-text.min.js";

function mix(a: string, b: string, t: number) {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

type Paint = Record<string, Record<string, unknown>>;

/** The basemap's label colour in a theme, a little stronger than the palette's label grey, which the halo used to
 * lift off water and shade; the network diagram's city names share it (T-2492). */
export const placeNameInk = (s: Shade) => (s === "dark" ? mix(SHADES.dark.label, "#ffffff", 0.3) : mix(SHADES.light.label, "#000000", 0.35));

/** The three road weights (T-676), line width in pixels by zoom, the same in both themes. Roads
 * are one colour with no casing, so only these tell a motorway from a lane. */
const zoomWidth = (...stops: number[]) => ["interpolate", ["exponential", 1.4], ["zoom"], ...stops];
const ROAD_WEIGHT = {
  motorway: zoomWidth(4, 1, 8, 1.6, 12, 3, 16, 8, 20, 28),
  main: zoomWidth(8, 0.6, 10, 1, 12, 2, 16, 6, 20, 20),
  other: zoomWidth(12, 0.7, 14, 1.4, 16, 3.5, 20, 14),
};
const ROADS: Record<string, keyof typeof ROAD_WEIGHT> = {
  highway_motorway_subtle: "motorway", highway_motorway_inner: "motorway", highway_motorway_bridge_inner: "motorway",
  tunnel_motorway_inner: "motorway",
  highway_major_subtle: "main", highway_major_inner: "main",
  highway_minor: "other", highway_path: "other",
};
/** Roads of the smaller classes appear later than Positron has them. */
const ROAD_MINZOOM: Record<string, number> = { highway_minor: 11, highway_path: 14 };
/** Positron layers that only add clutter: road casings, the airfield casing, minor road names. */
const HIDDEN = new Set([
  "highway_major_casing", "highway_motorway_casing", "highway_motorway_bridge_casing", "tunnel_motorway_casing",
  "aeroway-runway-casing", "highway-name-path", "highway-name-minor",
]);

/** T-1270: the optional hillshade (Settings, Terrain), from the Terrain Tiles on AWS Open Data
 * (Terrarium encoding, public, no key; Mapterhorn's hosted tiles have no stated terms for an app). For
 * a published game the fallback is our own Terrarium tiles on R2 from GEDTM30 (notes/T-1270.md). */
const DEM = "terrain-dem";
const HILLSHADE = "hillshade";
const WATER_OVER_SHADE = "water_over_shade";
const TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
/** T-1552, T-2628: the style names the terrain tiles through this protocol, which holds each tile back
 * until the basemap's vector tiles over the same ground have loaded, so the shade never shows where the
 * water is still coming. */
const DEM_PROTOCOL = "twdem";
/** T-3705: MapLibre reads the heights back through a canvas, or VideoFrame where a canvas read is made up; a
 * browser with neither (Firefox resisting fingerprinting) would shade noise, so no shade is drawn there. */
const demReadable = () => canvasReadsTrue() || typeof VideoFrame !== "undefined";
const VECTOR_SOURCE = "openmaptiles";
/** how often a held terrain tile asks whether the basemap has stopped loading (ms): its last tile failing fires no tile event */
const DEM_POLL_MS = 500;
const TERRAIN_CREDIT = '<a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank">Terrain Tiles</a>';

/** USGS answers 200 with a small blank tile for ground it has no imagery of (Canada and Mexico inside the box below);
 * this protocol turns those into a 404, so the Sentinel-2 layer under shows there. */
const USGS_PROTOCOL = "twusgs";
const USGS_TILES = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile";
const USGS_BLANK_BYTES = 3000;
let tileJsonProtocolAdded = false;
function addTileJsonProtocol() {
  if (tileJsonProtocolAdded) return;
  tileJsonProtocolAdded = true;
  maplibregl.addProtocol(TILEJSON_PROTOCOL, async (params) => ({ data: await openMapTiles(params.url) }));
}
let usgsProtocolAdded = false;
function addUsgsProtocol() {
  if (usgsProtocolAdded) return;
  usgsProtocolAdded = true;
  maplibregl.addProtocol(USGS_PROTOCOL, async (params, abort) => {
    const url = `${USGS_TILES}/${params.url.slice(USGS_PROTOCOL.length + 3)}`;
    const res = await fetch(url, { signal: abort.signal });
    if (!res.ok) throw new maplibregl.AJAXError(res.status, res.statusText, url, new Blob());
    const data = await res.arrayBuffer();
    if (data.byteLength < USGS_BLANK_BYTES) throw new maplibregl.AJAXError(404, "no imagery", url, new Blob());
    return { data };
  });
}
/** T-3249: the free satellite imagery, three raster sources stacked just above the background, each fetching only
 * inside its box: Sentinel-2 cloudless everywhere (10 m, soft past zoom 14), GSI's photos over Japan and USGS's NAIP
 * over the contiguous US on top of it. Each licence's credit is the source's attribution. The imagery's opacity is the
 * Base map slider, over the drawn map at its 0% look (T-3278). */
const SAT_SOURCES: { id: string; tiles: string; maxzoom: number; bounds?: [number, number, number, number]; attribution: string }[] = [
  { id: "sat-eox", tiles: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg", maxzoom: 14,
    attribution: 'Sentinel-2 cloudless by <a href="https://s2maps.eu" target="_blank">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2024)' },
  { id: "sat-gsi", tiles: "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg", maxzoom: 18, bounds: [122.5, 20.3, 154.5, 46.3],
    attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html" target="_blank">GSI Japan (地理院タイル)</a>' },
  { id: "sat-usgs", tiles: `${USGS_PROTOCOL}://{z}/{y}/{x}`, maxzoom: 16, bounds: [-125, 24.3, -66.9, 49.5],
    attribution: '<a href="https://www.usgs.gov/programs/national-geospatial-program/national-map" target="_blank">USGS National Map</a>' },
]; // i18n-ignore: attributions, as the data owners word them
const SAT_RASTER = SAT_SOURCES.map((s) => s.id);
const SAT_LABEL = "#f2f2f2";
/** the layers drawn over the imagery (main roads, boundaries, names), and the paint the satellite view overrides on
 * them, to put back when the imagery gives way */
let satKept: { id: string; kind: "road" | "bound" | "text"; weight?: number; was: Record<string, unknown> }[] = [];
let satOn = false;
/** the kept layers are in their imagery paint (the Base map slider at SAT_INK_FLIP or more) */
let satInked = false;
let satelliteWanted = false;

/** Positron's place layers that name a settlement, each with a black dot icon. */
const CITY_LABELS = new Set(["label_city", "label_city_capital", "label_town", "label_village"]);

/** T-1692: minor labels start later than Positron has them: suburbs and islands, villages, towns, small
 * rivers, river and strait names, main road names. City names keep Positron's. */
const LABEL_MINZOOM: Record<string, number> = {
  label_other: 12, label_village: 12, label_town: 9, waterway_line_label: 12, water_name_line_label: 8,
  "highway-name-major": 13.5, airport: 12,
};
/** T-1692: pixels kept clear around every basemap label. */
const TEXT_PADDING = 10;
/** T-1692: the same around city and town names, so only the larger of two close ones is placed. */
const PLACE_PADDING = 28;

/** Positron's layers repainted in one theme's colours. A layer or key the light theme leaves out
 * keeps Positron's own paint there. */
function paintFor(c: ShadeColours, dark: boolean): Paint {
  const { land, basemapMix: warm } = c;
  const air = mix(land, warm, dark ? 0.1 : 0.08);
  const p: Paint = {
    background: { "background-color": land },
    park: { "fill-color": c.park },
    landcover_wood: { "fill-color": c.park },
    water: { "fill-color": c.water },
    waterway: { "line-color": c.water },
    landuse_residential: { "fill-color": mix(land, warm, 0.03) },
    building: { "fill-color": mix(land, warm, 0.04), "fill-outline-color": mix(land, warm, 0.08) },
    road_area_pier: { "fill-color": land },
    road_pier: { "line-color": land },
    "aeroway-taxiway": { "line-color": air },
    "aeroway-runway": { "line-color": air },
    "aeroway-area": { "fill-color": air },
    landcover_ice_shelf: { "fill-color": mix(land, dark ? warm : "#ffffff", dark ? 0.08 : 0.55) },
  };
  p.landcover_glacier = p.landcover_ice_shelf;
  // T-1270: the hillshade, and the water drawn again over it so the sea floor's relief stays hidden
  p[HILLSHADE] = dark
    ? { "hillshade-shadow-color": "#000000", "hillshade-accent-color": "#000000", "hillshade-highlight-color": mix(land, "#ffffff", 0.12), "hillshade-exaggeration": 0.45 }
    : { "hillshade-shadow-color": "#5e4c33", "hillshade-accent-color": "#5e4c33", "hillshade-highlight-color": "#ffffff", "hillshade-exaggeration": 0.35 };
  p[WATER_OVER_SHADE] = { "fill-color": c.water };
  for (const [id, weight] of Object.entries(ROADS)) p[id] = { "line-color": c.road, "line-opacity": id === "tunnel_motorway_inner" ? 0.5 : 1, "line-width": ROAD_WEIGHT[weight] };
  for (const id of ["boundary_2", "boundary_3", "boundary_disputed"]) p[id] = { "line-color": mix(land, warm, 0.25) };
  return p;
}

/** Each restyled layer's paint values in either theme: the keys that differ, every plain colour the
 * contrast setting mutes (`mute`, T-1249), and every label opacity it fades (`fade`). */
type Fade = "mute" | "fade" | null;
let themed: { id: string; key: string; light: unknown; dark: unknown; how: Fade }[] = [];
/** Water is never muted (Anita, T-1249). */
const UNMUTED = new Set(["water", "waterway", "background", WATER_OVER_SHADE]);
const HEX = /^#[0-9a-f]{6}$/i;
/** A paint value at a contrast: a plain colour mixed toward the land by what the contrast lacks; a
 * label's opacity scaled by it, since a label mixed toward the land reads as a pale patch on water
 * (T-1266). */
const atContrast = (v: unknown, how: Fade, land: string, contrast: number) =>
  how === "fade" ? (typeof v === "number" ? v * contrast : v)
  : how === "mute" && contrast < 1 && typeof v === "string" && HEX.test(v) ? mix(v, land, 1 - contrast) : v;
/** Basemap symbol layers (not ours), whose visibility the "map labels" setting controls. */
let basemapSymbols: string[] = [];

/** The first symbol layer's id: layers added before it draw under the basemap's labels. */
export const firstSymbolId = (map: maplibregl.Map) => map.getStyle().layers.find((l) => l.type === "symbol")?.id;

// T-1552, T-2628: the basemap vector tiles loaded ("z/x/y"; one starting to load again is taken out). A
// terrain tile (256 px, so its zoom runs 1-2 above the 512 px vector tiles') is ready to show once every
// vector tile at the zoom the basemap draws now over its ground in view is in, or once the basemap has
// stopped loading (a vector tile that failed leaves bare ground, water or not, so the shade waits for
// nothing more). No time limit: on a slow connection the shade comes with the water, however late.
const vectorIn = new Set<string>();
const VECTOR_KEPT = 20_000;
const demWaiting: { z: number; x: number; y: number; done: () => void }[] = [];
let demPoll = 0;

/** the zoom of the basemap's vector tiles drawn now (512 px tiles, to the source's maxzoom) */
function vectorZoom(map: maplibregl.Map) {
  const max = (map.getSource(VECTOR_SOURCE) as { maxzoom?: number } | undefined)?.maxzoom ?? 14;
  return Math.max(0, Math.min(max, Math.floor(map.getZoom())));
}

function vectorReady(map: maplibregl.Map, z: number, x: number, y: number) {
  const v = vectorZoom(map);
  if (z >= v) return vectorIn.has(`${v}/${x >> (z - v)}/${y >> (z - v)}`);
  // a terrain tile wider than a vector tile (past its zoom 12): those of them in view
  const d = v - z, n = 2 ** v, b = map.getBounds();
  const tx = (lng: number) => Math.floor(((lng + 180) / 360) * n);
  const ty = (lat: number) => Math.floor(((1 - Math.log(Math.tan(Math.PI / 4 + (Math.max(-85, Math.min(85, lat)) * Math.PI) / 360)) / Math.PI) / 2) * n);
  const [x0, x1, y0, y1] = [tx(b.getWest()), tx(b.getEast()), Math.max(0, ty(b.getNorth())), Math.min(n - 1, ty(b.getSouth()))];
  for (let j = Math.max(y0, y << d); j <= Math.min(y1, ((y + 1) << d) - 1); j++)
    for (let i = x << d; i < (x + 1) << d; i++) {
      // the view's columns may run past the antimeridian
      const seen = [i - n, i, i + n].some((c) => c >= x0 && c <= x1);
      if (seen && !vectorIn.has(`${v}/${i}/${j}`)) return false;
    }
  return true;
}

const vectorSettled = (map: maplibregl.Map) => styleUp(map) && !!map.getSource(VECTOR_SOURCE) && map.isSourceLoaded(VECTOR_SOURCE);

function releaseDem(map: maplibregl.Map) {
  if (!demWaiting.length) return;
  const all = vectorSettled(map);
  for (const w of demWaiting.slice()) if (all || vectorReady(map, w.z, w.x, w.y)) w.done();
}

function vectorWait(map: maplibregl.Map, z: number, x: number, y: number, signal: AbortSignal): Promise<void> {
  // a macrotask on, so the basemap's own tile requests of the same pass have started
  return new Promise((resolve) => setTimeout(resolve, 0)).then(() => {
    if (signal.aborted || vectorReady(map, z, x, y) || vectorSettled(map)) return;
    return new Promise<void>((resolve) => {
      const w = {
        z, x, y,
        done: () => {
          signal.removeEventListener("abort", w.done);
          demWaiting.splice(demWaiting.indexOf(w), 1);
          if (!demWaiting.length) window.clearInterval(demPoll);
          resolve();
        },
      };
      signal.addEventListener("abort", w.done);
      if (!demWaiting.length) demPoll = window.setInterval(() => releaseDem(map), DEM_POLL_MS);
      demWaiting.push(w);
    });
  });
}

/** T-2870: the basemap's vector tiles (or its style) have been missing for BASEMAP_SLOW_MS, counted from when
 * the current wait began, so a pan whose tiles come quickly shows nothing. The status tag reads it. */
export const basemapSlow = signal(false);
const BASEMAP_SLOW_MS = 3000;
let basemapTimer = 0;
function checkBasemapWait(map: maplibregl.Map, polled = false) {
  const waiting = !styleUp(map) || !map.getSource(VECTOR_SOURCE) || !map.isSourceLoaded(VECTOR_SOURCE);
  if (!waiting) { clearTimeout(basemapTimer); basemapTimer = 0; if (basemapSlow.peek()) basemapSlow.value = false; }
  else if (polled) basemapSlow.value = true;
  else if (!basemapTimer && !basemapSlow.peek()) basemapTimer = window.setTimeout(() => { basemapTimer = 0; checkBasemapWait(map, true); }, BASEMAP_SLOW_MS);
}

let demProtocolAdded = false;
export const terrainUrl = (z: number, x: number, y: number) => TERRAIN_TILES.replace("{z}", `${z}`).replace("{x}", `${x}`).replace("{y}", `${y}`);
/** the clock worker loads the terrain blocks in view below this width (clock.worker.ts VIEW_SPAN_M) */
const GRID_VIEW_M = 400_000;
let demMap: maplibregl.Map | null = null;
const viewLoadsGrids = () => {
  if (!demMap) return false;
  const b = demMap.getBounds();
  return (b.getEast() - b.getWest()) * 111_320 * Math.cos((b.getCenter().lat * Math.PI) / 180) < GRID_VIEW_M;
};
function addDemProtocol() {
  if (demProtocolAdded) return;
  demProtocolAdded = true;
  maplibregl.addProtocol(DEM_PROTOCOL, async (params, abort) => {
    const m = /^[a-z]+:\/\/(\d+)\/(\d+)\/(\d+)/.exec(params.url);
    if (!m) throw new Error(`bad terrain tile url ${params.url}`);
    const [z, x, y] = [Number(m[1]), Number(m[2]), Number(m[3])];
    const bytes = demBytes(terrainUrl, z, x, y).then((b) => { if (!b) throw new Error("terrain tile missing"); return b; });
    await Promise.all([bytes, demMap && vectorWait(demMap, z, x, y, abort.signal)]);
    // T-1881: relief over ground priced flat taken out (towers and raised roads in the surface model)
    return { data: await flattenDem(terrainUrl, z, x, y, await bytes, abort.signal, viewLoadsGrids()) };
  });
}

/** Loads Positron into the map, restyled. Call once, right after creating the map. */
export function loadBasemap(map: maplibregl.Map) {
  demMap = map;
  addDemProtocol();
  addUsgsProtocol();
  addTileJsonProtocol();
  onRedraw((tiles) => { if (styleUp(map) && map.getSource(DEM)) map.refreshTiles(DEM, tiles); });
  const key = (c: { z: number; x: number; y: number }) => `${c.z}/${c.x}/${c.y}`;
  map.on("sourcedataloading", (e) => {
    if (e.sourceId !== VECTOR_SOURCE) return;
    if (e.coord) vectorIn.delete(key(e.coord.canonical));
    checkBasemapWait(map);
  });
  map.on("idle", () => checkBasemapWait(map));
  map.on("error", () => checkBasemapWait(map));
  map.on("sourcedata", (e) => {
    if (e.sourceId !== VECTOR_SOURCE) return;
    checkBasemapWait(map);
    const c = e.tile?.state === "loaded" ? e.coord?.canonical : null;
    if (c) {
      vectorIn.delete(key(c));
      vectorIn.add(key(c));
      if (vectorIn.size > VECTOR_KEPT) vectorIn.delete(vectorIn.values().next().value!);
    }
    releaseDem(map);
  });
  // Arabic and Hebrew names (world view) are drawn unjoined and reversed without it. Lazy: fetched
  // only once such a label is on screen (memory reference_maplibre_rtl_and_name_en).
  if (maplibregl.getRTLTextPluginStatus() === "unavailable") maplibregl.setRTLTextPlugin(new URL(RTL_PLUGIN, document.baseURI).href, true).catch(() => {});
  map.setStyle(POSITRON, { transformStyle: (_previous, next) => restyle(next) });
  checkBasemapWait(map);
  // T-957: changes that came while there was no style (a lost context) are painted when it is back
  map.on("style.load", () => {
    if (!missed) return;
    missed = false;
    satOn = false;
    satInked = false;
    setSatellite(map, satelliteWanted);
    setBasemapShade(map, ...shadeAs);
    setBasemapLabels(map, ...labelsAs);
    setHillshade(map, hillshadeOn);
    repaintRoadHints(map);
  });
}

/** set when a basemap change found no style; `restyle` paints from the current values, so it clears it */
let missed = false;
let shadeAs: [Shade, number] = ["light", 1];

function restyle(style: maplibregl.StyleSpecification): maplibregl.StyleSpecification {
  // Absolute, from the page's own URL, so it works under /trainworld/ and on any host.
  style.glyphs = new URL("fonts/", document.baseURI).href + "{fontstack}/{range}.pbf";
  // T-4275: the vector tiles by whichever route a tracker list leaves open (openMapTiles.ts)
  const vector = style.sources[VECTOR_SOURCE];
  if (vector?.type === "vector" && vector.url) vector.url = routedTileJson(vector.url);
  basemapSymbols = [];
  themed = [];
  missed = false;
  const light = paintFor(SHADES.light, false), dark = paintFor(SHADES.dark, true);
  // T-1270: the hillshade over the land cover and under the roads, off unless Settings has Terrain on
  const at = style.layers.findIndex((l) => l.id === "waterway");
  const water = style.layers.find((l) => l.id === "water");
  if (at >= 0 && water) {
    style.sources[DEM] = { type: "raster-dem", tiles: [`${DEM_PROTOCOL}://{z}/{x}/{y}`], encoding: "terrarium", tileSize: 256, maxzoom: 12, attribution: TERRAIN_CREDIT };
    const visibility = display.peek().hillshade ? "visible" : "none";
    style.layers.splice(at, 0,
      { id: HILLSHADE, type: "hillshade", source: DEM, layout: { visibility: demReadable() ? visibility : "none" }, paint: {} },
      { ...(JSON.parse(JSON.stringify(water)) as typeof water), id: WATER_OVER_SHADE, layout: { visibility } } as maplibregl.LayerSpecification);
  }
  for (const layer of style.layers) {
    const paint = ((layer as any).paint ??= {});
    const layout = ((layer as any).layout ??= {});
    // The player builds the rail: hide the basemap's own railways and road shields.
    if (/^railway|shield/.test(layer.id) || HIDDEN.has(layer.id)) {
      layout.visibility = "none";
      continue;
    }
    if (layer.id in ROAD_MINZOOM) layer.minzoom = ROAD_MINZOOM[layer.id];
    const L = { ...light[layer.id] }, D = { ...dark[layer.id] };
    if (layer.type === "symbol") {
      basemapSymbols.push(layer.id);
      // T-978: the black dot beside a place name reads as a gameplay dot; the name stays, centred on the place
      if (CITY_LABELS.has(layer.id)) {
        for (const k of ["icon-image", "icon-size", "icon-allow-overlap", "icon-optional"]) delete layout[k];
        layout["text-anchor"] = "center";
        layout["text-offset"] = [0, 0];
      }
      if (layout["text-field"] !== undefined) {
        layout["text-font"] = FONT;
        layout["text-field"] = guardEnglishName(layout["text-field"]);
        if ((layout["symbol-placement"] ?? "point") === "point") layout["text-field"] = guardWrapMarks(layout["text-field"]);
        // T-1692: plain one-colour text, no halo (a halo in the land colour shows as an outline over shade)
        paint["text-halo-width"] = 0;
        L["text-color"] = placeNameInk("light");
        D["text-color"] = placeNameInk("dark");
        // T-1692: fewer labels at once, by a wider collision box
        layout["text-padding"] = Math.max(Number(layout["text-padding"]) || 0, layer.id.startsWith("label_c") || layer.id === "label_town" ? PLACE_PADDING : TEXT_PADDING);
      }
      if (layer.id in LABEL_MINZOOM) layer.minzoom = LABEL_MINZOOM[layer.id];
      for (const k of ["text-opacity", "icon-opacity"]) if (paint[k] === undefined) paint[k] = 1;
    }
    const s = shade.peek(), land = SHADES[s].land, contrast = display.peek().basemapContrast;
    for (const key of new Set([...Object.keys(L), ...Object.keys(D), ...Object.keys(paint)])) {
      const l = key in L ? L[key] : paint[key], d = key in D ? D[key] : paint[key];
      const how: Fade = layer.type === "symbol" ? (key.endsWith("opacity") ? "fade" : null)
        : !UNMUTED.has(layer.id) && key.endsWith("color") ? "mute" : null;
      if (l !== d || how) themed.push({ id: layer.id, key, light: l, dark: d, how });
      const v = s === "dark" ? d : l;
      if (v !== undefined) paint[key] = atContrast(v, how, land, contrast);
    }
  }
  addSatellite(style);
  return style;
}

/** T-3249: the imagery's sources and layers (hidden until Satellite is on), and which basemap layers stay above it. */
function addSatellite(style: maplibregl.StyleSpecification) {
  satKept = [];
  for (const layer of style.layers) {
    if ((layer as any).layout?.visibility === "none") continue;
    const paint = (layer as any).paint ?? {};
    if (layer.type === "symbol") {
      if ((layer as any).layout?.["text-field"] !== undefined)
        satKept.push({ id: layer.id, kind: "text", was: { "text-halo-width": paint["text-halo-width"], "text-halo-color": paint["text-halo-color"], "text-halo-blur": paint["text-halo-blur"] } });
    } else if (layer.type === "background") continue;
    else if (layer.type === "line" && ROADS[layer.id] && ROADS[layer.id] !== "other") satKept.push({ id: layer.id, kind: "road", weight: ROADS[layer.id] === "motorway" ? 0.26 : 0.18, was: { "line-opacity": paint["line-opacity"] } });
    else if (layer.type === "line" && layer.id.startsWith("boundary_")) satKept.push({ id: layer.id, kind: "bound", was: { "line-opacity": paint["line-opacity"] } });
  }
  const at = style.layers.findIndex((l) => l.type === "background") + 1;
  const rasters: maplibregl.LayerSpecification[] = [];
  for (const s of SAT_SOURCES) {
    style.sources[s.id] = { type: "raster", tiles: [s.tiles], tileSize: 256, maxzoom: s.maxzoom, attribution: s.attribution, ...(s.bounds ? { bounds: s.bounds } : {}) };
    rasters.push({ id: s.id, type: "raster", source: s.id, layout: { visibility: "none" }, paint: {} });
  }
  style.layers.splice(at, 0, ...rasters);
}

/** Satellite on: the imagery over every fill, minor road and casing of the drawn map, and the kept roads and boundaries
 * over the imagery (just below the game's own layers). Off: each goes back before the layer it was before. */
let satBack: [string, string | undefined][] = [];
function orderSatellite(map: maplibregl.Map, on: boolean) {
  if (on) {
    const moved = new Set([...SAT_RASTER, ...satKept.filter((k) => k.kind !== "text").map((k) => k.id)]);
    const live = (map.getStyle()?.layers ?? []).map((l) => l.id);
    const next = (i: number) => live.slice(i + 1).find((id) => !moved.has(id));
    const lastLine = live.reduce((a, id, i) => (moved.has(id) && !SAT_RASTER.includes(id) ? i : a), -1);
    const above = next(lastLine);
    satBack = [];
    live.forEach((id, i) => { if (moved.has(id)) satBack.push([id, next(i)]); });
    for (const [id] of satBack) map.moveLayer(id, above);
  } else {
    for (const [id, before] of satBack) if (map.getLayer(id)) map.moveLayer(id, before && map.getLayer(before) ? before : undefined);
    satBack = [];
  }
}

/** Whether a basemap layer is drawn: the terrain setting takes the shading away. */
function layerVisible(id: string) {
  return id === HILLSHADE ? hillshadeOn && demReadable() : id === WATER_OVER_SHADE ? hillshadeOn : true;
}

/** The imagery's opacity is the Base map slider (0-1) over the drawn map at its 0% look, so 0% is Map mode's own
 * 0% and 100% is the imagery alone (T-3267, T-3278). */
function dimImagery(map: maplibregl.Map, contrast: number) {
  for (const id of SAT_RASTER) {
    if (!map.getLayer(id)) continue;
    map.setPaintProperty(id, "raster-opacity", contrast);
    map.setPaintProperty(id, "raster-saturation", 0);
    map.setPaintProperty(id, "raster-brightness-max", 1);
  }
}

/** The kept layers' paint for strong imagery: roads and boundaries faint white, names light with a dark halo (the
 * no-halo rule of T-1692 gives way over imagery). Below SAT_INK_FLIP they keep the drawn map's own paint, which sits
 * well on the faded image. Painted again after every basemap repaint, which writes theirs. */
function paintSatellite(map: maplibregl.Map) {
  dimImagery(map, shadeAs[1]);
  satInked = shadeAs[1] >= SAT_INK_FLIP;
  if (!satInked) return;
  for (const k of satKept) {
    if (!map.getLayer(k.id)) continue;
    if (k.kind === "text") {
      map.setPaintProperty(k.id, "text-color", SAT_LABEL);
      map.setPaintProperty(k.id, "text-opacity", 1);
      map.setPaintProperty(k.id, "text-halo-color", "rgba(0,0,0,0.8)");
      map.setPaintProperty(k.id, "text-halo-width", 1.4);
      map.setPaintProperty(k.id, "text-halo-blur", 0.5);
    } else {
      map.setPaintProperty(k.id, "line-color", "#ffffff");
      map.setPaintProperty(k.id, "line-opacity", k.kind === "road" ? k.weight! : 0.3);
    }
  }
}

/** T-3249: the satellite imagery on or off. */
export function setSatellite(map: maplibregl.Map, on: boolean) {
  satelliteWanted = on;
  if (!styleUp(map)) return void (missed = true);
  if (on === satOn) return;
  satOn = on;
  for (const id of SAT_RASTER) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  orderSatellite(map, on);
  setBasemapShade(map, ...shadeAs);
}

/** Repaints the basemap in the light or dark theme and at a contrast (T-1249), without reloading
 * the style or its tiles. */
export function setBasemapShade(map: maplibregl.Map, s: Shade, contrast: number) {
  shadeAs = [s, contrast];
  if (!styleUp(map)) return void (missed = true);
  // over imagery the drawn map underneath stays at its 0% look in the panels' own theme (names follow the slider as in
  // Map mode); the slider is the imagery's opacity
  const look = satOn ? shade.peek() : s;
  const land = SHADES[look].land;
  repaintRoadHints(map);
  // the kept layers go back to the drawn map's own paint before it is written again
  if (satInked && !(satOn && contrast >= SAT_INK_FLIP)) {
    for (const k of satKept) for (const [key, v] of Object.entries(k.was)) if (map.getLayer(k.id)) map.setPaintProperty(k.id, key as any, v);
    satInked = false;
  }
  if (satOn && map.getLayer("background")) map.setPaintProperty("background", "background-color", land);
  for (const t of themed) {
    if (!map.getLayer(t.id)) continue;
    const v = look === "dark" ? t.dark : t.light;
    map.setPaintProperty(t.id, t.key as any, atContrast(v, t.how, land, satOn && t.how !== "fade" ? 0 : contrast));
  }
  if (satOn) paintSatellite(map);
  // T-3913: the town covered in the land as now painted
  if (hintOn.bus) setTownFill(map, true);
}

/**
 * OpenFreeMap's tiles carry name_en = "T" for Türkiye, and Positron shows name_en first: fall back
 * to the Latin name when name_en is under three letters (memory reference_maplibre_rtl_and_name_en).
 */
function guardEnglishName(e: unknown): unknown {
  if (!Array.isArray(e)) return e;
  if (e.length === 2 && e[0] === "get" && e[1] === "name_en")
    return ["case", [">=", ["length", ["to-string", ["coalesce", ["get", "name_en"], ""]]], 3],
      ["get", "name_en"], ["coalesce", ["get", "name:latin"], ["get", "name"]]];
  return e.map(guardEnglishName);
}

/**
 * T-3842: spacing marks that MapLibre 6.13-6.19 joins to the letter before only when the text around them holds some
 * other mark (its `textCanContainGraphemeClusters` leaves them out, `canCombineGraphemes` takes every gc=Mc). A
 * wrapped label whose line holds these but no such other mark is re-split into more clusters than it had sections
 * for, and the worker throws ("Cannot use 'in' operator to search for 'imageName'"), losing the whole tile: roads,
 * water and all. Mostly Burmese (aa, visarga), Shan and Karen tones; Tai Tham, Myanmar Extended-A and Ahom too.
 * Myanmar's letters are in the list as well, so every Myanmar-script name goes the same way: Burmese is in nearly
 * every name with those marks, and villages with and without their Burmese line would read as a fault.
 */
const WRAP_MARKS = [
  [0x102b, 0x102c], [0x1038, 0x1038], [0x1062, 0x1064], [0x1067, 0x106d], [0x1083, 0x1083], [0x1087, 0x108c],
  [0x108f, 0x108f], [0x109a, 0x109c], [0x1a61, 0x1a61], [0x1a63, 0x1a64], [0xaa7b, 0xaa7b], [0xaa7d, 0xaa7d],
  [0x11720, 0x11721],
  // Myanmar letters: Burmese, Pali, Mon, Karen, Shan, Palaung
  [0x1000, 0x102a], [0x103f, 0x103f], [0x1050, 0x1055], [0x105a, 0x105d], [0x1061, 0x1061], [0x1065, 0x1066],
  [0x106e, 0x1070], [0x1075, 0x1081], [0x108e, 0x108e],
].flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, i) => String.fromCodePoint(a + i)));
const hasWrapMark = (s: unknown) => ["any", ...WRAP_MARKS.map((c) => ["in", c, s])];

/** T-3842: a point label whose place name holds one of those shows its Latin name alone, or no name where it has
 * none. Only point labels wrap; a line label is laid out as one line. Tested on the `name` property, not the label
 * text: a `var` is evaluated afresh at every use, so testing a bound label would build it once per character. */
function guardWrapMarks(field: unknown): unknown {
  const latin = ["coalesce", ["get", "name:latin"], ""];
  return ["case", hasWrapMark(["coalesce", ["get", "name"], ""]),
    ["case", hasWrapMark(latin), "", latin],
    field];
}

/** T-1270: the hillshade on or off (and the water drawn over it with it); the shade never on the lighter map. */
export function setHillshade(map: maplibregl.Map, on: boolean) {
  hillshadeOn = on;
  if (!styleUp(map)) return void (missed = true);
  if (map.getLayer(HILLSHADE)) map.setLayoutProperty(HILLSHADE, "visibility", layerVisible(HILLSHADE) ? "visible" : "none");
  // kept on the lighter map too: the terrain tint (map/terrainTint.ts) sits under it, and water is never charged
  if (map.getLayer(WATER_OVER_SHADE)) map.setLayoutProperty(WATER_OVER_SHADE, "visibility", layerVisible(WATER_OVER_SHADE) ? "visible" : "none");
}
let hillshadeOn = false;

/** T-2883: in bus placing mode, the roads a bus may drive are tinted faintly over the basemap's own
 * roads, and a one-way street gets arrows pointing the way traffic runs; T-2998: with Tram in the track
 * tool and the street snap on, the streets a tram may follow, in green. Colour light, dark, opacity light, dark. */
const ONEWAY_ARROW = "bus-oneway-arrow", ONEWAY_ARROW_FAST = "bus-oneway-arrow-fast";
const ONEWAY_LAYER = "bus_roads_oneway";
/** A chevron pointing right, the direction `symbol-placement: line` treats as forward along the road. */
function onewayArrow(colour: string): { width: number; height: number; data: Uint8ClampedArray } {
  const s = 32;
  if (!canvasReadsTrue()) return { width: s, height: s, data: chevronPixels(s, [1, 3, 5].map((i) => parseInt(colour.slice(i, i + 2), 16))) };
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = s;
  const g = canvas.getContext("2d", { willReadFrequently: true })!;
  g.clearRect(0, 0, s, s);
  g.strokeStyle = colour;
  g.lineWidth = 3.5;
  g.lineCap = "butt";
  g.lineJoin = "miter";
  g.beginPath();
  g.moveTo(11, 10);
  g.lineTo(19, 16);
  g.lineTo(11, 22);
  g.stroke();
  const pixels = g.getImageData(0, 0, s, s).data;
  return { width: s, height: s, data: new Uint8ClampedArray(pixels) };
}
/** T-3705: the same chevron worked out pixel by pixel, where a canvas read is made up (map/canvasRead.ts): each
 * pixel's cover by a 3.5 px stroke along both legs, butt ends and a mitred joint, edges smoothed over a pixel */
function chevronPixels(s: number, [r, g, b]: number[]): Uint8ClampedArray {
  const legs = [[11, 10, 19, 16], [19, 16, 11, 22]], half = 1.75;
  const px = new Uint8ClampedArray(s * s * 4);
  for (let y = 0; y < s; y++)
    for (let x = 0; x < s; x++) {
      const cx = x + 0.5, cy = y + 0.5;
      let d = Infinity;
      const off: number[] = [], at: number[] = [];
      for (const [x0, y0, x1, y1] of legs) {
        const vx = x1 - x0, vy = y1 - y0, t = ((cx - x0) * vx + (cy - y0) * vy) / (vx * vx + vy * vy);
        const e = Math.hypot(cx - x0 - t * vx, cy - y0 - t * vy);
        off.push(e);
        at.push(t);
        if (t >= 0 && t <= 1) d = Math.min(d, e);
      }
      // the mitre: where both legs' strokes, run on past the joint, overlap
      if (at[0] >= 0 && at[1] <= 1) d = Math.min(d, Math.max(off[0], off[1]));
      const a = Math.min(1, Math.max(0, half + 0.5 - d));
      if (a > 0) px.set([r, g, b, Math.round(a * 255)], (y * s + x) * 4);
    }
  return px;
}
/** the arrows in the hint's colours: its teal, and on a fast road its blue */
function ensureOnewayArrow(map: maplibregl.Map, dark: boolean) {
  for (const [id, colour] of [[ONEWAY_ARROW, ROAD_HINTS.bus.colour[+dark]], [ONEWAY_ARROW_FAST, BUS_FAST_COLOUR[+dark]]]) {
    const image = onewayArrow(colour);
    if (map.hasImage(id)) map.updateImage(id, image);
    else map.addImage(id, image);
  }
}
const ROAD_HINTS = {
  // T-3014: both a quarter wider and from a zoom further out (the road tiles carry the minor streets from 12, so
  // at 11 the hint is the major ones)
  bus: { id: "bus_roads_hint", filter: () => busRoadFilter(), colour: ["#1a9e94", "#3fd0c4"], opacity: [0.55, 0.38], minzoom: 11, width: 1.25 },
  tram: { id: "tram_roads_hint", filter: () => busRoadFilter(), colour: ["#3f9a1c", "#86d65a"], opacity: [0.55, 0.38], minzoom: 11, width: 1.25 },
  // T-3129: the main roads rail track snaps to
  rail: { id: "rail_roads_hint", filter: () => railRoadFilter(), colour: ["#c2661a", "#f0a050"], opacity: [0.6, 0.45], minzoom: 9, width: 1.5 },
} as const;
type RoadHint = keyof typeof ROAD_HINTS;
const hintOn: Record<RoadHint, boolean> = { bus: false, tram: false, rail: false };
/** Settings > Buses keep to one-way streets: the arrows are drawn only with it on */
let busOneWayOn = false;
export function setBusRoadHint(map: maplibregl.Map, on: boolean, oneWay = false) {
  busOneWayOn = oneWay;
  setRoadHint(map, "bus", on);
}
export function setTramRoadHint(map: maplibregl.Map, on: boolean) {
  setRoadHint(map, "tram", on);
}
export function setRailRoadHint(map: maplibregl.Map, on: boolean) {
  setRoadHint(map, "rail", on);
}
/** the hints that are on, painted again (a new style or shade) */
function repaintRoadHints(map: maplibregl.Map) {
  for (const k of Object.keys(hintOn) as RoadHint[]) if (hintOn[k]) setRoadHint(map, k, true);
}
/** T-3829 (Anita picked it from three): the bus hint draws the roads a bus runs fastest on (`fastRoadExpr`) in
 * blue, a little wider, the other roads in its teal. */
const BUS_FAST_COLOUR = ["#3a4fb0", "#a6bcff"], BUS_FAST_OPACITY = [0.85, 0.9], BUS_FAST_WIDTH = 1.2;
/** T-3901: the bus hint's colour of a fast or an ordinary road in the current shade, for a speed tag over it */
export function busRoadColour(fast: boolean): string {
  const dark = shadeAs[0] === "dark" ? 1 : 0;
  return fast ? BUS_FAST_COLOUR[dark] : ROAD_HINTS.bus.colour[dark];
}
/** the hint's width at each zoom, `mul` times (a number or a per-road expression) */
const hintWidth = (w: number, mul: unknown = 1) =>
  zoomWidth(11, 0.4 * w, 12, 0.5 * w, 14, w, 16, 2.5 * w, 20, 10 * w).map((s, i) => (i >= 4 && i % 2 === 0 && mul !== 1 ? ["*", s, mul] : s));
const byFast = (fast: unknown, slow: unknown) => ["case", fastRoadExpr(), fast, slow];
/** the hint's paint, with `mul` its width per road over the plain hint's (the one-way arrows follow it) */
function hintPaint(kind: RoadHint, dark: 0 | 1): { colour: unknown; opacity: unknown; width: unknown; mul: unknown } {
  const h = ROAD_HINTS[kind];
  if (kind !== "bus") return { colour: h.colour[dark], opacity: h.opacity[dark], width: hintWidth(h.width), mul: 1 };
  const mul = byFast(BUS_FAST_WIDTH, 1);
  return { colour: byFast(BUS_FAST_COLOUR[dark], h.colour[dark]), opacity: byFast(BUS_FAST_OPACITY[dark], h.opacity[dark]), width: hintWidth(h.width, mul), mul };
}
/** T-3913: the bus hint tints the ground by where a bus is in town for BRT (`townArea.ts`), under the streets: `town`
 * tints the town, `country` what is out of town (drawn over the whole world, then the town covered in the land's own
 * colour), each light and dark. `?town=<letter>` picks one and keeps it in this browser. Streets keep their colours:
 * splitting the street layer by `within` miscoloured whole tiles (notes/T-3913.md). */
interface Tint { colour: [string, string]; opacity: [number, number] }
const TOWN_GREEN: Tint = { colour: ["#22a43c", "#5fd070"], opacity: [0.044, 0.034] };
const TOWN_TEAL: Tint = { colour: ["#1a9e94", "#3fd0c4"], opacity: [0.14, 0.12] };
const TOWN_VIOLET: Tint = { colour: ["#7a4fc0", "#a98cf0"], opacity: [0.14, 0.12] };
const COUNTRY_KHAKI: Tint = { colour: ["#9a8a3a", "#c8b468"], opacity: [0.2, 0.14] };
const COUNTRY_ROSE: Tint = { colour: ["#c8505e", "#e88090"], opacity: [0.16, 0.12] };
// a: green town (landed); b: violet town; c: rose out of town; now: the first teal town; khaki: out of town khaki
const TOWN_LOOKS: Record<string, { town?: Tint; country?: Tint }> = {
  none: {},
  a: { town: TOWN_GREEN },
  b: { town: TOWN_VIOLET },
  c: { country: COUNTRY_ROSE },
  now: { town: TOWN_TEAL },
  khaki: { country: COUNTRY_KHAKI },
};
const TOWN_LOOK_DEFAULT = "a";
const townLook = (() => {
  let k = TOWN_LOOK_DEFAULT;
  try {
    const asked = new URLSearchParams(location.search).get("town");
    if (asked && asked in TOWN_LOOKS) localStorage.setItem("twTownLook", asked);
    const kept = localStorage.getItem("twTownLook");
    if (kept && kept in TOWN_LOOKS) k = kept;
  } catch {
    // no storage: the default
  }
  return TOWN_LOOKS[k];
})();
const TOWN_SOURCE = "bus_town", WORLD_SOURCE = "bus_world";
const TOWN_FILL = "bus_town_fill", COUNTRY_FILL = "bus_country_fill", TOWN_MASK = "bus_town_mask";
let townTimer: ReturnType<typeof setTimeout> | undefined;
/** T-3913: the town's polygons changed: drawn again a moment later, once a burst of packs is in */
export function townChanged(map: maplibregl.Map) {
  clearTimeout(townTimer);
  townTimer = setTimeout(() => {
    if (hintOn.bus) setTownFill(map, true);
  }, 300);
}
/** T-3913: the ground's tints under the bus hint, in the look's layers */
function setTownFill(map: maplibregl.Map, on: boolean) {
  const geometry = on ? townGeometry() : null;
  const look = townLook;
  const ids = [TOWN_FILL, COUNTRY_FILL, TOWN_MASK];
  if (!styleUp(map)) return;
  if (!on || !(look.town || look.country)) {
    for (const id of ids) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of [TOWN_SOURCE, WORLD_SOURCE]) if (map.getSource(id)) map.removeSource(id);
    return;
  }
  const dark = shadeAs[0] === "dark" ? 1 : 0;
  // out of town where no pack's street speeds are in, as in the sim
  const data = { type: "FeatureCollection" as const, features: geometry ? [{ type: "Feature" as const, properties: {}, geometry }] : [] };
  const src = map.getSource(TOWN_SOURCE) as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(data);
  else map.addSource(TOWN_SOURCE, { type: "geojson", data, tolerance: 0 });
  const put = (id: string, source: string, colour: string, opacity: number, before: string | undefined) => {
    if (map.getLayer(id)) {
      map.setPaintProperty(id, "fill-color", colour);
      map.setPaintProperty(id, "fill-opacity", opacity);
    } else map.addLayer({ id, type: "fill", source, minzoom: ROAD_HINTS.bus.minzoom, paint: { "fill-color": colour, "fill-opacity": opacity, "fill-antialias": false } }, before);
  };
  const layers = map.getStyle().layers;
  // the town's tint under the water and the roads: the ground of the town only
  if (look.town) put(TOWN_FILL, TOWN_SOURCE, look.town.colour[dark], look.town.opacity[dark], map.getLayer("water") ? "water" : layers.find((l) => l.type === "line")?.id);
  if (look.country) {
    if (!map.getSource(WORLD_SOURCE)) map.addSource(WORLD_SOURCE, { type: "geojson", data: { type: "Polygon", coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]] } });
    // right over the land, under imagery, parks and everything else
    const above = layers.slice(layers.findIndex((l) => l.type === "background") + 1).find((l) => !ids.includes(l.id))?.id;
    const land = map.getLayer("background") ? (map.getPaintProperty("background", "background-color") as string) : SHADES[shadeAs[0]].land;
    put(COUNTRY_FILL, WORLD_SOURCE, look.country.colour[dark], look.country.opacity[dark], above);
    put(TOWN_MASK, TOWN_SOURCE, land, 1, above);
  }
}
function setRoadHint(map: maplibregl.Map, kind: RoadHint, on: boolean) {
  hintOn[kind] = on;
  if (!styleUp(map)) return;
  const h = ROAD_HINTS[kind], dark = shadeAs[0] === "dark" ? 1 : 0;
  if (!on) {
    if (map.getLayer(h.id)) map.removeLayer(h.id);
    if (kind === "bus" && map.getLayer(ONEWAY_LAYER)) map.removeLayer(ONEWAY_LAYER);
    if (kind === "bus") setTownFill(map, false);
    return;
  }
  const style = map.getStyle();
  if (!style || !map.getLayer("highway_minor") || !map.getSource(VECTOR_SOURCE)) return;
  const p = hintPaint(kind, dark);
  const before = style.layers.find((l) => l.type === "symbol")?.id;
  if (map.getLayer(h.id)) {
    map.setPaintProperty(h.id, "line-color", p.colour as maplibregl.ExpressionSpecification);
    map.setPaintProperty(h.id, "line-opacity", p.opacity as maplibregl.ExpressionSpecification);
    map.setPaintProperty(h.id, "line-width", p.width as maplibregl.ExpressionSpecification);
  } else {
    map.addLayer({
      id: h.id, type: "line", source: VECTOR_SOURCE, "source-layer": "transportation", minzoom: h.minzoom,
      filter: h.filter() as maplibregl.FilterSpecification,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": p.colour as string, "line-opacity": p.opacity as number, "line-width": p.width as maplibregl.ExpressionSpecification },
    }, before);
  }
  if (kind === "bus") {
    setTownFill(map, true);
    if (busOneWayOn) setBusOnewayArrows(map, dark === 1, before, p.mul);
    else if (map.getLayer(ONEWAY_LAYER)) map.removeLayer(ONEWAY_LAYER);
  }
}

/** Arrows on the one-way streets a bus may use, in the hint's own colour, pointing the way the road
 * runs. `oneway` -1 is drawn against the line. They wait until the streets are close enough to draw
 * on, and go when the bus tool is off. */
const ONEWAY_MINZOOM = 15;
function setBusOnewayArrows(map: maplibregl.Map, dark: boolean, before: string | undefined, mul: unknown) {
  ensureOnewayArrow(map, dark);
  // the chevron's stroke is 3.5 px at size 1: sized to 1.2 x the road hint's own stroke at each zoom
  const size = ["interpolate", ["exponential", 1.4], ["zoom"], ...[14, 0.43, 16, 1.06, 20, 4.24].map((s, i) => (i % 2 && mul !== 1 ? ["*", s, mul] : s))] as maplibregl.ExpressionSpecification;
  if (map.getLayer(ONEWAY_LAYER)) {
    map.setLayerZoomRange(ONEWAY_LAYER, ONEWAY_MINZOOM, 24);
    map.setLayoutProperty(ONEWAY_LAYER, "icon-size", size);
    return;
  }
  map.addLayer({
    id: ONEWAY_LAYER, type: "symbol", source: VECTOR_SOURCE, "source-layer": "transportation", minzoom: ONEWAY_MINZOOM,
    filter: ["all", busRoadFilter(), ["any", ["==", ["to-number", ["get", "oneway"]], 1], ["==", ["to-number", ["get", "oneway"]], -1]]] as unknown as maplibregl.FilterSpecification,
    layout: {
      "symbol-placement": "line",
      "symbol-spacing": 90,
      "icon-image": byFast(ONEWAY_ARROW_FAST, ONEWAY_ARROW) as maplibregl.ExpressionSpecification,
      "icon-size": size,
      "icon-rotate": ["case", ["==", ["to-number", ["get", "oneway"]], -1], 180, 0] as maplibregl.ExpressionSpecification,
      "icon-rotation-alignment": "map",
      "icon-allow-overlap": true,
      "icon-ignore-placement": true,
      "icon-padding": 0,
    },
    paint: { "icon-opacity": 0.7 },
  }, before);
}

/** `noCities`: the names of cities, towns and villages hidden whatever `on` says (the start menu's map, T-974). */
export function setBasemapLabels(map: maplibregl.Map, on: boolean, noCities = false) {
  labelsAs = [on, noCities];
  if (!styleUp(map)) return void (missed = true);
  for (const id of basemapSymbols) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on && !(noCities && CITY_LABELS.has(id)) ? "visible" : "none");
}
let labelsAs: [boolean, boolean] = [true, false];

