// The real-network reference layer (T-682 stage 1): where railways run in the real world, faint, under
// the game's own network and above the basemap's roads. View only; on by default at 0.15 opacity
// (Settings > Real network), at 0.4 while the track tool snaps to it (T-727).
//
// The lines come from noritetsu's per-region vector tiles (`noritetsu/dist/data/<region>.pmtiles`, one
// `track` layer: k kind, r rank 0 main or 1 other, p confirmed by a passenger route, n no line runs over
// it, c colour; nm and w only from zoom 10). The tiles carry no line type, so the classes are what they
// do carry: metro (subway), main line (rail, rank 0), other passenger rail (rank 1 with a route over
// it). Trams, light rail, monorails, funiculars, narrow gauge, heritage and track no line runs over are
// left out. A region's source is added when the view first touches it, from zoom 4 up.
//
// Where the files are: beside the game at `../noritetsu/` when that answers with noritetsu's
// regions.json (8800 serves it; the dev server proxies the path, vite.config.ts), else Cloudflare R2,
// the way noritetsu's own page decides.

import * as maplibregl from "maplibre-gl";
import { effect, signal } from "@preact/signals";
import { Protocol } from "pmtiles";
import { SHADES, type ShadeColours } from "../game/shadeTable";
import { display, shade, snapping } from "../game/ui";

const LOCAL_BASE = new URL("../noritetsu/", document.baseURI).href;
const R2_BASE = "https://data.anita.garden/noritetsu/";
const OSM_ATTRIBUTION = '<a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a> (ODbL)';
/** below this the view spans too many regions and the lines are too small to follow */
const MIN_ZOOM = 4;
/** regions added in one pass, nearest the middle of the view first */
const PER_PASS = 16;
const ID = "tw-real-";

const kind = (k: string) => ["==", ["get", "k"], k];
const METRO = kind("subway");
const MAIN = ["all", kind("rail"), ["==", ["get", "r"], 0]];
const SHOWN = ["all", ["!=", ["get", "n"], 1], ["any", METRO, ["all", kind("rail"), ["any", ["==", ["get", "r"], 0], ["==", ["get", "p"], 1]]]]];

/** line width (px) by zoom; main and metro the same, other rail thinner */
const width = (...stops: number[]) => ["interpolate", ["exponential", 1.4], ["zoom"], ...stops.flatMap((v, i) => [MIN_ZOOM + [0, 4, 7, 10, 13, 16][i], ["case", ["any", METRO, MAIN], v, v * 0.7]])];
const WIDTH = width(0.6, 1, 1.5, 2.4, 4, 8);
/** T-727: faint for reference, full while the track tool snaps to it */
const OPACITY = 0.15;
const OPACITY_SNAP = 0.4;

const colour = (c: ShadeColours) => ["case", METRO, c.realMetro, MAIN, c.realMain, c.realOther];

/** T-839: the real track a snapped route from the first click could reach, drawn over the layer in
 * the same colours at full strength and a little wider, the rest of the layer dimmed. Chain i's
 * lon, lat pairs are `xy[off[i] * 2]` to `xy[off[i + 1] * 2]`, its class `cls[i]` (0 main line,
 * 1 other rail, 2 metro); set by map/realSnap.ts. */
export const realReach = signal<{ xy: Float32Array; off: Uint32Array; cls: Uint8Array } | null>(null);
const REACH = "tw-reach";
const OPACITY_DIM = 0.25;
const REACH_OPACITY = 0.9;
const reachWidth = (...stops: number[]) => ["interpolate", ["exponential", 1.4], ["zoom"], ...stops.flatMap((v, i) => [MIN_ZOOM + [0, 4, 7, 10, 13, 16][i], ["case", ["==", ["get", "c"], 1], v * 0.7, v]])];
const REACH_WIDTH = reachWidth(1, 1.5, 2.2, 3.4, 5.5, 10);
const reachColour = (c: ShadeColours) => ["match", ["get", "c"], 2, c.realMetro, 0, c.realMain, c.realOther];

function reachData(r: { xy: Float32Array; off: Uint32Array; cls: Uint8Array } | null) {
  const lines: number[][][][] = [[], [], []];
  if (r)
    for (let i = 0; i + 1 < r.off.length; i++) {
      const l: number[][] = [];
      for (let k = r.off[i]; k < r.off[i + 1]; k++) l.push([r.xy[k * 2], r.xy[k * 2 + 1]]);
      if (l.length > 1) lines[r.cls[i]].push(l);
    }
  return {
    type: "FeatureCollection" as const,
    features: lines.map((coordinates, c) => ({ type: "Feature" as const, properties: { c }, geometry: { type: "MultiLineString" as const, coordinates } })),
  };
}

interface Region {
  cc: string;
  /** bounding boxes of the region's outline parts: west, south, east, north */
  parts: [number, number, number, number][];
}

export function startRealNetwork(map: maplibregl.Map) {
  try {
    maplibregl.addProtocol("pmtiles", new Protocol().tile as unknown as Parameters<typeof maplibregl.addProtocol>[1]);
  } catch {
    // already added
  }
  const added = new Set<string>();
  let regions: Region[] | null = null;
  let base = "";
  let index: Promise<void> | null = null;
  let on = false;
  let painted: "light" | "dark" | null = null;
  let opacity = OPACITY;

  /** regions.json (about 1.5 MB, outlines in it): which regions exist and where, and which base serves them */
  const load = () =>
    (index ??= (async () => {
      for (const b of [LOCAL_BASE, R2_BASE]) {
        try {
          const r = await fetch(b + "regions.json");
          const j = r.ok ? await r.json() : null;
          if (!j?.regions) continue;
          base = b;
          regions = Object.entries(j.regions as Record<string, { parts: [number, number][][] }>).map(([cc, r]) => ({
            cc,
            parts: r.parts.map((ring) => {
              let w = 180, s = 90, e = -180, n = -90;
              for (const [x, y] of ring) (w = Math.min(w, x)), (e = Math.max(e, x)), (s = Math.min(s, y)), (n = Math.max(n, y));
              return [w, s, e, n] as [number, number, number, number];
            }),
          }));
          return;
        } catch {
          // not there, or not JSON: try the next
        }
      }
      console.warn("Real network: noritetsu's data is not reachable");
    })());

  const layerIds = () => [...added].map((cc) => ID + cc);

  const setShade = () => {
    const s = shade.peek();
    if (s === painted) return;
    painted = s;
    for (const id of layerIds()) map.setPaintProperty(id, "line-color", colour(SHADES[s]) as never);
  };

  const ensure = async () => {
    if (!on || map.getZoom() < MIN_ZOOM) return;
    await load();
    if (!on || !regions) return;
    if (!map.isStyleLoaded()) return void map.once("idle", ensure);
    const b = map.getBounds(), c = map.getCenter();
    const dist = (r: Region) => Math.min(...r.parts.map((p) => Math.hypot(Math.max(0, p[0] - c.lng, c.lng - p[2]), Math.max(0, p[1] - c.lat, c.lat - p[3]))));
    const hits = regions
      .filter((r) => !added.has(r.cc) && r.parts.some((p) => p[0] <= b.getEast() && p[2] >= b.getWest() && p[1] <= b.getNorth() && p[3] >= b.getSouth()))
      .sort((x, y) => dist(x) - dist(y))
      .slice(0, PER_PASS);
    if (!hits.length) return;
    const before = map.getLayer(REACH) ? REACH : map.getStyle().layers.find((l) => l.type === "symbol")?.id;
    const colours = colour(SHADES[shade.peek()]);
    for (const { cc } of hits) {
      added.add(cc);
      // MapLibre's restored style after a lost context already has them (T-953)
      if (map.getLayer(ID + cc)) continue;
      if (!map.getSource(ID + cc)) map.addSource(ID + cc, { type: "vector", url: `pmtiles://${base}data/${cc}.pmtiles`, attribution: OSM_ATTRIBUTION });
      map.addLayer(
        {
          id: ID + cc, type: "line", source: ID + cc, "source-layer": "track", minzoom: MIN_ZOOM,
          filter: SHOWN as never,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": colours as never, "line-width": WIDTH as never, "line-opacity": opacity },
        },
        before,
      );
    }
    painted = shade.peek();
    if (hits.length === PER_PASS) map.once("idle", ensure);
  };

  const showReach = () => {
    const r = snapping.peek() ? realReach.peek() : null;
    // not isStyleLoaded(): that waits on every tile, which can be seconds
    if (!map.getStyle()) return void map.once("style.load", showReach);
    const src = map.getSource(REACH) as maplibregl.GeoJSONSource | undefined;
    if (!src) {
      if (!r) return;
      map.addSource(REACH, { type: "geojson", data: reachData(r), tolerance: 0.3 });
      map.addLayer(
        {
          id: REACH, type: "line", source: REACH,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": reachColour(SHADES[shade.peek()]) as never, "line-width": REACH_WIDTH as never, "line-opacity": REACH_OPACITY },
        },
        map.getStyle().layers.find((l) => l.type === "symbol")?.id,
      );
      return;
    }
    src.setData(reachData(r));
    map.setPaintProperty(REACH, "line-color", reachColour(SHADES[shade.peek()]) as never);
  };

  effect(() => {
    // snapping shows the layer even when Settings has it off: nothing unseen is snapped to
    const snap = snapping.value;
    const reach = snap && !!realReach.value;
    on = display.value.realNetwork || snap;
    opacity = snap ? (reach ? OPACITY_DIM : OPACITY_SNAP) : OPACITY;
    shade.value;
    if (on) setShade(), void ensure();
    for (const id of layerIds()) {
      map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
      map.setPaintProperty(id, "line-opacity", opacity);
    }
    showReach();
  });
  map.on("moveend", ensure);
  map.on("style.load", () => {
    added.clear();
    painted = null;
    void ensure();
    showReach();
  });
}
