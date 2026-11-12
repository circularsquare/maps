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
import { display, shade, type Shade } from "../game/ui";
import { styleUp } from "./styleReady";
import { busRoadFilter, railRoadFilter } from "./busRoads";
import { demBytes, flattenDem, onRedraw } from "./demFlat";

const POSITRON = "https://tiles.openfreemap.org/styles/positron";
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
const VECTOR_SOURCE = "openmaptiles";
/** how often a held terrain tile asks whether the basemap has stopped loading (ms): its last tile failing fires no tile event */
const DEM_POLL_MS = 500;
const TERRAIN_CREDIT = '<a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank">Terrain Tiles</a>';

/** USGS answers 200 with a small blank tile for ground it has no imagery of (Canada and Mexico inside the box below);
 * this protocol turns those into a 404, so the Sentinel-2 layer under shows there. */
const USGS_PROTOCOL = "twusgs";
const USGS_TILES = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile";
const USGS_BLANK_BYTES = 3000;
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
 * over the contiguous US on top of it. Each licence's credit is the source's attribution. The imagery is dimmed by
 * the Base map slider; the game's vector fills, shade and minor roads are hidden while it shows. */
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
/** the paint the satellite view overrides on the layers it keeps, to put back when it goes off */
let satKept: { id: string; kind: "road" | "bound" | "text"; weight?: number; was: Record<string, unknown> }[] = [];
/** the basemap layers hidden while the imagery shows: every fill and shade, minor roads, anything else drawn */
let satHide = new Set<string>();
let satOn = false;
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
    setSatellite(map, satelliteWanted);
    setBasemapShade(map, ...shadeAs);
    applyLite(map);
    repaintRoadHints(map);
  });
}

/** set when a basemap change found no style; `restyle` paints from the current values, so it clears it */
let missed = false;
let shadeAs: [Shade, number] = ["light", 1];

function restyle(style: maplibregl.StyleSpecification): maplibregl.StyleSpecification {
  // Absolute, from the page's own URL, so it works under /trainworld/ and on any host.
  style.glyphs = new URL("fonts/", document.baseURI).href + "{fontstack}/{range}.pbf";
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
      { id: HILLSHADE, type: "hillshade", source: DEM, layout: { visibility }, paint: {} },
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

/** T-3249: the imagery's sources and layers (hidden until Satellite is on), and which basemap layers give way to it. */
function addSatellite(style: maplibregl.StyleSpecification) {
  satKept = [];
  satHide = new Set();
  for (const layer of style.layers) {
    if ((layer as any).layout?.visibility === "none") continue;
    const paint = (layer as any).paint ?? {};
    if (layer.type === "symbol") {
      if ((layer as any).layout?.["text-field"] !== undefined)
        satKept.push({ id: layer.id, kind: "text", was: { "text-halo-width": paint["text-halo-width"], "text-halo-color": paint["text-halo-color"], "text-halo-blur": paint["text-halo-blur"] } });
    } else if (layer.type === "background") continue;
    else if (layer.type === "line" && ROADS[layer.id] && ROADS[layer.id] !== "other") satKept.push({ id: layer.id, kind: "road", weight: ROADS[layer.id] === "motorway" ? 0.26 : 0.18, was: { "line-opacity": paint["line-opacity"] } });
    else if (layer.type === "line" && layer.id.startsWith("boundary_")) satKept.push({ id: layer.id, kind: "bound", was: { "line-opacity": paint["line-opacity"] } });
    else satHide.add(layer.id);
  }
  satHide.add(HILLSHADE).add(WATER_OVER_SHADE);
  const at = style.layers.findIndex((l) => l.type === "background") + 1;
  const rasters: maplibregl.LayerSpecification[] = [];
  for (const s of SAT_SOURCES) {
    style.sources[s.id] = { type: "raster", tiles: [s.tiles], tileSize: 256, maxzoom: s.maxzoom, attribution: s.attribution, ...(s.bounds ? { bounds: s.bounds } : {}) };
    rasters.push({ id: s.id, type: "raster", source: s.id, layout: { visibility: "none" }, paint: {} });
  }
  style.layers.splice(at, 0, ...rasters);
}

/** Whether a basemap layer is drawn: the lighter map, the terrain setting and satellite each take layers away. */
function layerVisible(id: string) {
  if (satOn && satHide.has(id)) return false;
  if (id === HILLSHADE) return hillshadeOn && !lite;
  if (id === WATER_OVER_SHADE) return hillshadeOn;
  return !(lite && LITE_HIDDEN.has(id));
}

/** The imagery's dimming at a Base map contrast (0-1): lower shows more of the dark ground under it. */
function dimImagery(map: maplibregl.Map, contrast: number) {
  for (const id of SAT_RASTER) {
    if (!map.getLayer(id)) continue;
    map.setPaintProperty(id, "raster-opacity", 0.3 + 0.55 * contrast);
    map.setPaintProperty(id, "raster-saturation", -0.3);
    map.setPaintProperty(id, "raster-brightness-max", 0.85);
  }
}

/** The kept layers' paint for imagery: roads and boundaries faint white, names light with a dark halo (the
 * no-halo rule of T-1692 gives way over imagery). Painted again after every basemap repaint, which writes theirs. */
function paintSatellite(map: maplibregl.Map) {
  for (const k of satKept) {
    if (!map.getLayer(k.id)) continue;
    if (k.kind === "text") {
      map.setPaintProperty(k.id, "text-color", SAT_LABEL);
      map.setPaintProperty(k.id, "text-halo-color", "rgba(0,0,0,0.8)");
      map.setPaintProperty(k.id, "text-halo-width", 1.4);
      map.setPaintProperty(k.id, "text-halo-blur", 0.5);
    } else {
      map.setPaintProperty(k.id, "line-color", "#ffffff");
      map.setPaintProperty(k.id, "line-opacity", k.kind === "road" ? k.weight! : 0.3);
    }
  }
  dimImagery(map, shadeAs[1]);
}

/** T-3249: the satellite imagery on or off. */
export function setSatellite(map: maplibregl.Map, on: boolean) {
  satelliteWanted = on;
  if (!styleUp(map)) return void (missed = true);
  if (on === satOn) return;
  satOn = on;
  for (const id of SAT_RASTER) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  for (const id of satHide) if (map.getLayer(id) && !basemapSymbols.includes(id)) map.setLayoutProperty(id, "visibility", layerVisible(id) ? "visible" : "none");
  if (on) paintSatellite(map);
  else {
    for (const k of satKept) for (const [key, v] of Object.entries(k.was)) if (map.getLayer(k.id)) map.setPaintProperty(k.id, key as any, v);
    setBasemapShade(map, ...shadeAs);
  }
}

/** Repaints the basemap in the light or dark theme and at a contrast (T-1249), without reloading
 * the style or its tiles. */
export function setBasemapShade(map: maplibregl.Map, s: Shade, contrast: number) {
  shadeAs = [s, contrast];
  if (!styleUp(map)) return void (missed = true);
  const land = SHADES[s].land;
  repaintRoadHints(map);
  for (const t of themed) {
    if (!map.getLayer(t.id)) continue;
    const v = s === "dark" ? t.dark : t.light;
    map.setPaintProperty(t.id, t.key as any, atContrast(v, t.how, land, contrast));
  }
  if (satOn) paintSatellite(map);
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
 * roads; T-2998: with Tram in the track tool and the street snap on, the streets a tram may follow, in
 * green. Colour light, dark, opacity light, dark. */
const ROAD_HINTS = {
  // T-3014: both a quarter wider and from a zoom further out (the road tiles carry the minor streets from 12, so
  // at 11 the hint is the major ones)
  bus: { id: "bus_roads_hint", filter: () => busRoadFilter(), colour: ["#1a9e94", "#3fd0c4"], opacity: [0.55, 0.38], minzoom: 11, width: 1.25 },
  tram: { id: "tram_roads_hint", filter: () => busRoadFilter(false), colour: ["#3f9a1c", "#86d65a"], opacity: [0.55, 0.38], minzoom: 11, width: 1.25 },
  // T-3129: the main roads rail track snaps to
  rail: { id: "rail_roads_hint", filter: () => railRoadFilter(), colour: ["#c2661a", "#f0a050"], opacity: [0.6, 0.45], minzoom: 9, width: 1.5 },
} as const;
type RoadHint = keyof typeof ROAD_HINTS;
const hintOn: Record<RoadHint, boolean> = { bus: false, tram: false, rail: false };
export function setBusRoadHint(map: maplibregl.Map, on: boolean) {
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
function setRoadHint(map: maplibregl.Map, kind: RoadHint, on: boolean) {
  hintOn[kind] = on;
  if (!styleUp(map)) return;
  const h = ROAD_HINTS[kind], dark = shadeAs[0] === "dark" ? 1 : 0;
  if (!on) {
    if (map.getLayer(h.id)) map.removeLayer(h.id);
    return;
  }
  const style = map.getStyle();
  if (!style || !map.getLayer("highway_minor") || !map.getSource(VECTOR_SOURCE)) return;
  const colour = h.colour[dark], opacity = h.opacity[dark];
  if (map.getLayer(h.id)) {
    map.setPaintProperty(h.id, "line-color", colour);
    map.setPaintProperty(h.id, "line-opacity", opacity);
    return;
  }
  const before = style.layers.find((l) => l.type === "symbol")?.id;
  map.addLayer({
    id: h.id, type: "line", source: VECTOR_SOURCE, "source-layer": "transportation", minzoom: h.minzoom,
    filter: h.filter() as maplibregl.FilterSpecification,
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": colour, "line-opacity": opacity, "line-width": zoomWidth(11, 0.4 * h.width, 12, 0.5 * h.width, 14, h.width, 16, 2.5 * h.width, 20, 10 * h.width) as maplibregl.ExpressionSpecification },
  }, before);
}

/** `noCities`: the names of cities, towns and villages hidden whatever `on` says (the start menu's map, T-974). */
export function setBasemapLabels(map: maplibregl.Map, on: boolean, noCities = false) {
  labelsAs = [on, noCities];
  if (!styleUp(map)) return void (missed = true);
  for (const id of basemapSymbols) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on && !(noCities && CITY_LABELS.has(id)) && !(lite && LITE_HIDDEN.has(id)) ? "visible" : "none");
}
let labelsAs: [boolean, boolean] = [true, false];

/** T-1207: the Positron layers the lighter map leaves out: buildings, the residential tint, minor
 * roads and paths, airfields, small rivers' and minor places' and roads' names. */
const LITE_HIDDEN = new Set([
  "building", "landuse_residential", "highway_minor", "highway_path", "aeroway-taxiway", "aeroway-runway", "aeroway-area",
  "waterway", "waterway_line_label", "airport", "highway-name-major", "label_other", "label_village", "water_name_line_label",
]);
let lite = false;

/** T-1207: the lighter map's basemap on or off (the pixel ratio is map/lite.ts's). */
export function setBasemapLite(map: maplibregl.Map, on: boolean) {
  if (on === lite) return;
  lite = on;
  applyLite(map);
}

function applyLite(map: maplibregl.Map) {
  if (!styleUp(map)) return void (missed = true);
  for (const id of LITE_HIDDEN) {
    if (!map.getLayer(id) || basemapSymbols.includes(id)) continue;
    map.setLayoutProperty(id, "visibility", layerVisible(id) ? "visible" : "none");
  }
  setBasemapLabels(map, ...labelsAs);
  setHillshade(map, hillshadeOn);
}
