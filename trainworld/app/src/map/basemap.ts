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
import { SHADES, type ShadeColours } from "../game/shades";
import { shade, type Shade } from "../game/ui";

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

/** Positron's layers repainted in one theme's colours. A layer or key the light theme leaves out
 * keeps Positron's own paint there. */
function paintFor(c: ShadeColours, dark: boolean): Paint {
  const { land, basemapMix: ink } = c;
  const p: Paint = {
    background: { "background-color": land },
    park: { "fill-color": c.park },
    landcover_wood: { "fill-color": c.park },
    water: { "fill-color": c.water },
    waterway: { "line-color": c.water },
    landuse_residential: { "fill-color": mix(land, ink, 0.03) },
    building: { "fill-color": mix(land, ink, 0.04), "fill-outline-color": mix(land, ink, 0.08) },
    road_area_pier: { "fill-color": land },
    road_pier: { "line-color": land },
    highway_minor: { "line-color": mix(land, ink, 0.07) },
    highway_path: { "line-color": mix(land, ink, 0.05) },
  };
  for (const id of ["highway_major_casing", "highway_motorway_casing", "highway_motorway_bridge_casing", "tunnel_motorway_casing"])
    p[id] = { "line-color": mix(land, ink, dark ? 0.05 : 0.13) };
  for (const id of ["boundary_2", "boundary_3", "boundary_disputed"]) p[id] = { "line-color": mix(land, ink, 0.25) };
  if (dark) {
    // Positron's white roads, runways and ice, a little lighter than the land as in Dark Matter
    for (const id of ["highway_major_inner", "highway_motorway_inner", "highway_motorway_bridge_inner", "tunnel_motorway_inner"])
      p[id] = { "line-color": mix(land, ink, 0.1) };
    for (const id of ["highway_major_subtle", "highway_motorway_subtle", "aeroway-taxiway", "aeroway-runway-casing"])
      p[id] = { "line-color": mix(land, ink, 0.06) };
    p["aeroway-runway"] = { "line-color": mix(land, ink, 0.1) };
    p["aeroway-area"] = { "fill-color": mix(land, ink, 0.05) };
    p.landcover_ice_shelf = p.landcover_glacier = { "fill-color": mix(land, ink, 0.08) };
  }
  return p;
}

/** Each restyled layer's paint values in either theme, the keys that differ only. */
let themed: { id: string; key: string; light: unknown; dark: unknown }[] = [];
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
  themed = [];
  const light = paintFor(SHADES.light, false), dark = paintFor(SHADES.dark, true);
  for (const layer of style.layers) {
    const paint = ((layer as any).paint ??= {});
    const layout = ((layer as any).layout ??= {});
    // The player builds the rail: hide the basemap's own railways and road shields.
    if (/^railway|shield/.test(layer.id)) {
      layout.visibility = "none";
      continue;
    }
    const L = { ...light[layer.id] }, D = { ...dark[layer.id] };
    if (layer.type === "symbol") {
      basemapSymbols.push(layer.id);
      if (layout["text-field"] !== undefined) {
        layout["text-font"] = FONT;
        layout["text-field"] = guardEnglishName(layout["text-field"]);
        paint["text-halo-width"] = 1.2;
        L["text-color"] = SHADES.light.label, L["text-halo-color"] = SHADES.light.land;
        D["text-color"] = SHADES.dark.label, D["text-halo-color"] = SHADES.dark.land;
      }
    }
    for (const key of new Set([...Object.keys(L), ...Object.keys(D)])) {
      const l = key in L ? L[key] : paint[key], d = key in D ? D[key] : paint[key];
      if (l !== d) themed.push({ id: layer.id, key, light: l, dark: d });
      const v = shade.peek() === "dark" ? d : l;
      if (v !== undefined) paint[key] = v;
    }
  }
  return style;
}

/** Repaints the basemap in the light or dark theme, without reloading the style or its tiles. */
export function setBasemapShade(map: maplibregl.Map, s: Shade) {
  for (const t of themed) if (map.getLayer(t.id)) map.setPaintProperty(t.id, t.key as any, s ==="dark" ? t.dark : t.light);
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
