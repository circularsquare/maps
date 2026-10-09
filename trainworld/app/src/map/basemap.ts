// The basemap style (T-009 mock, T-056): OpenFreeMap Positron repainted neutral and faint so the
// network reads first, the basemap's own railways and road shields hidden, and every label in Zen
// Maru Gothic from our own glyph tiles. Applied to the style JSON before MapLibre commits it, so
// nothing draws in Positron's colours or fonts first.
//
// Glyphs: `public/fonts/Zen Maru Gothic Regular,Noto Sans Regular/{range}.pbf`, made by
// pipeline/glyphs.py (notes/T-056.md). MapLibre fetches one file per font stack and 256-codepoint
// range, so the Noto fallback for scripts Zen Maru lacks is merged into those files. Kana, kanji
// and hangul never come from them: MapLibre draws those in the browser (localIdeographFontFamily).

import * as maplibregl from "maplibre-gl";
import { NET_INK } from "../game/palette";

const POSITRON = "https://tiles.openfreemap.org/styles/positron";
/** The one font stack every basemap label uses: Regular only, Zen Maru has no italic. */
const FONT = ["Zen Maru Gothic Regular", "Noto Sans Regular"];
const RTL_PLUGIN = "https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js";

const LAND = "#f2f2ef", WATER = "#cbd8df", PARK = "#e3e8df", LABEL = "#a9abad";

function mix(a: string, b: string, t: number) {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

const PAINT: Record<string, Record<string, string>> = {
  background: { "background-color": LAND },
  park: { "fill-color": PARK },
  landcover_wood: { "fill-color": PARK },
  water: { "fill-color": WATER },
  waterway: { "line-color": WATER },
  landuse_residential: { "fill-color": mix(LAND, NET_INK, 0.03) },
  building: { "fill-color": mix(LAND, NET_INK, 0.04), "fill-outline-color": mix(LAND, NET_INK, 0.08) },
  road_area_pier: { "fill-color": LAND },
  road_pier: { "line-color": LAND },
  highway_minor: { "line-color": mix(LAND, NET_INK, 0.07) },
  highway_path: { "line-color": mix(LAND, NET_INK, 0.05) },
};
for (const id of ["highway_major_casing", "highway_motorway_casing", "highway_motorway_bridge_casing", "tunnel_motorway_casing"])
  PAINT[id] = { "line-color": mix(LAND, NET_INK, 0.13) };
for (const id of ["boundary_2", "boundary_3", "boundary_disputed"]) PAINT[id] = { "line-color": mix(LAND, NET_INK, 0.25) };

/** Basemap symbol layers (not ours), whose visibility the "map labels" setting controls. */
let basemapSymbols: string[] = [];

/** Loads Positron into the map, restyled. Call once, right after creating the map. */
export function loadBasemap(map: maplibregl.Map) {
  // Arabic and Hebrew names (world view) are drawn unjoined and reversed without it. Lazy: fetched
  // only once such a label is on screen (memory reference_maplibre_rtl_and_name_en).
  if (maplibregl.getRTLTextPluginStatus() === "unavailable") maplibregl.setRTLTextPlugin(RTL_PLUGIN, true).catch(() => {});
  map.setStyle(POSITRON, { transformStyle: (_previous, next) => restyle(next) });
}

function restyle(style: maplibregl.StyleSpecification): maplibregl.StyleSpecification {
  // Absolute, from the page's own URL, so it works under /trainworld/ and on any host.
  style.glyphs = new URL("fonts/", document.baseURI).href + "{fontstack}/{range}.pbf";
  basemapSymbols = [];
  for (const layer of style.layers) {
    const paint = ((layer as any).paint ??= {});
    const layout = ((layer as any).layout ??= {});
    Object.assign(paint, PAINT[layer.id]);
    // The player builds the rail: hide the basemap's own railways and road shields.
    if (/^railway|shield/.test(layer.id)) {
      layout.visibility = "none";
      continue;
    }
    if (layer.type !== "symbol") continue;
    basemapSymbols.push(layer.id);
    if (layout["text-field"] === undefined) continue;
    layout["text-font"] = FONT;
    layout["text-field"] = guardEnglishName(layout["text-field"]);
    paint["text-color"] = LABEL;
    paint["text-halo-color"] = LAND;
    paint["text-halo-width"] = 1.2;
  }
  return style;
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

export function setBasemapLabels(map: maplibregl.Map, on: boolean) {
  for (const id of basemapSymbols) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
}
