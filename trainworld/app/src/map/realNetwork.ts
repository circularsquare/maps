// The real-network reference layer (T-682 stage 1): where railways run in the real world, faint, under
// the game's own network and above the basemap's roads. View only, off by default (Settings > Real
// network).
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
import { effect } from "@preact/signals";
import { Protocol } from "pmtiles";
import { SHADES, type ShadeColours } from "../game/shadeTable";
import { display, shade } from "../game/ui";

const LOCAL_BASE = new URL("../noritetsu/", document.baseURI).href;
const R2_BASE = "https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/noritetsu/";
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
const OPACITY = 0.8;

const colour = (c: ShadeColours) => ["case", METRO, c.realMetro, MAIN, c.realMain, c.realOther];

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
    const before = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
    const colours = colour(SHADES[shade.peek()]);
    for (const { cc } of hits) {
      added.add(cc);
      map.addSource(ID + cc, { type: "vector", url: `pmtiles://${base}data/${cc}.pmtiles`, attribution: OSM_ATTRIBUTION });
      map.addLayer(
        {
          id: ID + cc, type: "line", source: ID + cc, "source-layer": "track", minzoom: MIN_ZOOM,
          filter: SHOWN as never,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": colours as never, "line-width": WIDTH as never, "line-opacity": OPACITY },
        },
        before,
      );
    }
    painted = shade.peek();
    if (hits.length === PER_PASS) map.once("idle", ensure);
  };

  effect(() => {
    on = display.value.realNetwork;
    shade.value;
    if (on) setShade(), void ensure();
    for (const id of layerIds()) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  });
  map.on("moveend", ensure);
  map.on("style.load", () => {
    added.clear();
    painted = null;
    void ensure();
  });
}
