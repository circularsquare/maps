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
import { SHADES, type ShadeColours } from "../game/shades";
import { display, shade, type Shade } from "../game/ui";
import { styleUp } from "./styleReady";

const POSITRON = "https://tiles.openfreemap.org/styles/positron";
/** The one font stack every basemap label uses: Regular only, Zen Maru has no italic. */
const FONT = ["Zen Maru Gothic Regular", "Noto Sans Regular"];
const RTL_PLUGIN = "https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js";

function mix(a: string, b: string, t: number) {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

type Paint = Record<string, Record<string, unknown>>;

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
const TERRAIN_CREDIT = '<a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank">Terrain Tiles</a>';

/** Positron's place layers that name a settlement, each with a black dot icon. */
const CITY_LABELS = new Set(["label_city", "label_city_capital", "label_town", "label_village"]);

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

/** Loads Positron into the map, restyled. Call once, right after creating the map. */
/** The first symbol layer's id: layers added before it draw under the basemap's labels. */
export const firstSymbolId = (map: maplibregl.Map) => map.getStyle().layers.find((l) => l.type === "symbol")?.id;

export function loadBasemap(map: maplibregl.Map) {
  // Arabic and Hebrew names (world view) are drawn unjoined and reversed without it. Lazy: fetched
  // only once such a label is on screen (memory reference_maplibre_rtl_and_name_en).
  if (maplibregl.getRTLTextPluginStatus() === "unavailable") maplibregl.setRTLTextPlugin(RTL_PLUGIN, true).catch(() => {});
  map.setStyle(POSITRON, { transformStyle: (_previous, next) => restyle(next) });
  // T-957: changes that came while there was no style (a lost context) are painted when it is back
  map.on("style.load", () => {
    if (!missed) return;
    missed = false;
    setBasemapShade(map, ...shadeAs);
    applyLite(map);
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
    style.sources[DEM] = { type: "raster-dem", tiles: [TERRAIN_TILES], encoding: "terrarium", tileSize: 256, maxzoom: 12, attribution: TERRAIN_CREDIT };
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
        paint["text-halo-width"] = 1.2;
        L["text-color"] = SHADES.light.label, L["text-halo-color"] = SHADES.light.land;
        D["text-color"] = SHADES.dark.label, D["text-halo-color"] = SHADES.dark.land;
      }
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
  return style;
}

/** Repaints the basemap in the light or dark theme and at a contrast (T-1249), without reloading
 * the style or its tiles. */
export function setBasemapShade(map: maplibregl.Map, s: Shade, contrast: number) {
  shadeAs = [s, contrast];
  if (!styleUp(map)) return void (missed = true);
  const land = SHADES[s].land;
  for (const t of themed) {
    if (!map.getLayer(t.id)) continue;
    const v = s === "dark" ? t.dark : t.light;
    map.setPaintProperty(t.id, t.key as any, atContrast(v, t.how, land, contrast));
  }
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
  if (map.getLayer(HILLSHADE)) map.setLayoutProperty(HILLSHADE, "visibility", on && !lite ? "visible" : "none");
  // kept on the lighter map too: the terrain tint (map/terrainTint.ts) sits under it, and water is never charged
  if (map.getLayer(WATER_OVER_SHADE)) map.setLayoutProperty(WATER_OVER_SHADE, "visibility", on ? "visible" : "none");
}
let hillshadeOn = false;

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
    map.setLayoutProperty(id, "visibility", lite ? "none" : "visible");
  }
  setBasemapLabels(map, ...labelsAs);
  setHillshade(map, hillshadeOn);
}
