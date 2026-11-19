// Real stations (T-3570, T-3571): the stations of noritetsu's model, `data/<cc>/stations.json` beside its
// tiles (the base map/realNetwork.ts reads them from, so R2 for players), one region's file fetched the first
// time it is needed. Junction points (`j`) are not stations and are left out.
//
// Marks: soft purple dots (the real metro colour, low opacity) over the real network from MIN_ZOOM, and while the station tool is active purple rings with a pale halo over the game's own network too, when Settings > Real world network > Show stations
// is on (off by default) and the network itself is shown; the name only on hover. A station whose every line
// is a tram, funicular, monorail or narrow gauge one shows only where those lines do (All, or tram track
// snapping), which takes the region's lines.json too (only while the marks are on); as rings, tram stops
// show always, smaller (T-3802).
//
// Names: a station placed within NAME_M of a real one is first suggested its name (map/stationNames.ts),
// written as the game writes place names: Latin letters, the local name where it is in them, else the
// English one, else Cyrillic transliterated (pipeline/realauto.py `pick_name`); a station with none of those
// gives no name, and naming goes on as before. For a Japanese, Chinese or Korean player the local name comes
// first where it is in a script they read (i18n/nativeNames.ts, T-3631).

import type { Feature } from "geojson";
import * as maplibregl from "maplibre-gl";
import { effect } from "@preact/signals";
import { SHADES } from "../game/shadeTable";
import { display, mapShade, snapMode, snapping, tool } from "../game/ui";
import { inRing, ringBox } from "../game/rings";
import { M_PER_DEG, M_PER_DEG_LAT } from "../game/frame";
import { patientFetch } from "../workers/patientFetch";
import { regionsJson } from "./regionsJson";
import { shadeColours } from "../game/shades";
import { firstSymbolId } from "./basemap";
import { styleUp } from "./styleReady";
import { fitSparse } from "./gl";
import type { Overlay } from "./overlay";
import { addFinder } from "./demandHover";
import { hiddenOnGlobe } from "./live";
import { language } from "../i18n";
import { nativeLang, readable } from "../i18n/nativeNames";

/** A new station this close to a real one takes its name: half a long platform from the real station's
 * point, and under the gap between neighbouring stations of different names almost everywhere (the
 * nearest such neighbour is under 150 m for 3% of Japan's stations, 4% of Germany's; notes/T-3570.md). */
export const NAME_M = 150;
/** further out, a city's stations are a dust of dots */
const MIN_ZOOM = 10;
const HOVER_PX = 6;
/** the grid the lookups use, degrees */
const CELL = 0.01;
export const REAL_STATIONS_LAYER = "tw-realst";

interface Stations {
  lng: Float64Array;
  lat: Float64Array;
  /** the name the game gives it, in Latin letters, or null */
  name: (string | null)[];
  /** the name as the region writes it, for the hover where there is no Latin one */
  local: string[];
  /** line ids, for the tram test */
  lines: string[][];
  grid: Map<number, number[]>;
}

const LATIN = /^[\p{Script=Latin}\p{M}\d\s\p{P}\p{S}]*$/u;
/** every letter is a Latin one, and there is one */
const latin = (s: string) => /\p{L}/u.test(s) && LATIN.test(s);

const CYR: Record<string, string> = Object.fromEntries(
  [..."абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ"].map((c, i) =>
    [c, ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya", "i", "yi", "ye", "g"][i]]),
);
/** Cyrillic in Latin letters (BGN/PCGN, simplified, as pipeline/realauto.py), other letters as they are */
function translit(s: string): string {
  let out = "";
  for (const ch of s) {
    const lo = ch.toLowerCase(), v = CYR[lo];
    out += v === undefined ? ch : ch !== lo ? v.slice(0, 1).toUpperCase() + v.slice(1) : v;
  }
  return out;
}

/** Melbourne's tram stops are "Stop 3: Bourke Street Mall" in OSM; the stop number is no name */
const STOP_NO = /^Stop \d+[A-Za-z]?:\s*/;

export function shownName(n: string, e: string): string | null {
  if (latin(n)) return n.replace(STOP_NO, "");
  if (latin(e)) return e;
  const t = translit(n);
  return latin(t) ? t : null;
}

const cellKey = (lng: number, lat: number) => Math.floor((lat + 90) / CELL) * 40000 + Math.floor((lng + 180) / CELL);

interface Region { cc: string; rings: number[][]; boxes: [number, number, number, number][] }
let regionList: Promise<{ base: string; regions: Region[] } | null> | null = null;
const regions = () =>
  (regionList ??= regionsJson().then((j) => j && {
    base: j.base,
    regions: Object.entries(j.regions).map(([cc, r]) => {
      const rings = r.parts.map((ring) => ring.flat());
      return { cc, rings, boxes: rings.map((ring) => ringBox(ring)) };
    }),
  }));

const files = new Map<string, Promise<Stations | null>>();
const ready = new Map<string, Stations>();

function stationsOf(base: string, cc: string): Promise<Stations | null> {
  let p = files.get(cc);
  if (!p) {
    p = patientFetch(`${base}data/${cc}/stations.json`).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      const raw = (await r.json()) as { stations: Record<string, { n?: string; e?: string; x: number; y: number; l?: string[]; j?: number }> };
      const list = Object.values(raw.stations).filter((s) => !s.j);
      const st: Stations = { lng: new Float64Array(list.length), lat: new Float64Array(list.length), name: [], local: [], lines: [], grid: new Map() };
      list.forEach((s, i) => {
        st.lng[i] = s.x;
        st.lat[i] = s.y;
        st.name.push(shownName(s.n ?? "", s.e ?? ""));
        st.local.push(s.n || s.e || "");
        st.lines.push(s.l ?? []);
        const k = cellKey(s.x, s.y);
        (st.grid.get(k) ?? st.grid.set(k, []).get(k)!).push(i);
      });
      ready.set(cc, st);
      return st;
    });
    files.set(cc, p);
    // a failed fetch is forgotten, so the next caller tries again
    p.catch(() => { if (files.get(cc) === p) files.delete(cc); });
  }
  return p.catch(() => null);
}

const inBox = (b: [number, number, number, number], lng: number, lat: number) => lng >= b[0] && lng <= b[2] && lat >= b[1] && lat <= b[3];

/** The regions whose outline holds (lng, lat). */
async function regionsAt(lng: number, lat: number): Promise<{ base: string; ccs: string[] } | null> {
  const r = await regions();
  if (!r) return null;
  return { base: r.base, ccs: r.regions.filter((g) => g.rings.some((ring, i) => inBox(g.boxes[i], lng, lat) && inRing(ring, lng, lat))).map((g) => g.cc) };
}

/** Fetches the stations of the regions at (lng, lat), once (the station tool asks ahead of a click). */
export async function loadRealStationsAt(lng: number, lat: number): Promise<void> {
  const at = await regionsAt(lng, lat).catch(() => null);
  if (at) await Promise.all(at.ccs.map((cc) => stationsOf(at.base, cc)));
}

/** The loaded stations within `m` metres of (lng, lat), nearest first. */
function nearby(lng: number, lat: number, m: number): { st: Stations; i: number; d: number }[] {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), dx = m / kx, dy = m / M_PER_DEG_LAT;
  const out: { st: Stations; i: number; d: number }[] = [];
  const seen = new Set<string>();
  for (const st of ready.values())
    for (let y = Math.floor((lat - dy + 90) / CELL); y <= Math.floor((lat + dy + 90) / CELL); y++)
      for (let x = Math.floor((lng - dx + 180) / CELL); x <= Math.floor((lng + dx + 180) / CELL); x++)
        for (const i of st.grid.get(y * 40000 + x) ?? []) {
          const d = Math.hypot((st.lng[i] - lng) * kx, (st.lat[i] - lat) * M_PER_DEG_LAT);
          // a region's file and its neighbour's can both hold a border station
          const key = `${st.lng[i]},${st.lat[i]}`;
          if (d <= m && !seen.has(key)) seen.add(key), out.push({ st, i, d });
        }
  return out.sort((a, b) => a.d - b.d);
}

/** The name of the real station nearest (lng, lat) within NAME_M, unless it is taken or has no name the
 * game can show. `wait`: fetch the region's stations first; otherwise only what is loaded counts, so a
 * click never waits on a download. */
export async function realStationName(lng: number, lat: number, taken: Set<string>, wait: boolean): Promise<string | null> {
  const load = loadRealStationsAt(lng, lat);
  if (wait) await load;
  const near = nearby(lng, lat, NAME_M)[0];
  const lang = nativeLang(language.peek());
  const name = near ? (lang && readable(near.st.local[near.i], lang)) || near.st.name[near.i] : null;
  return name && !taken.has(name) ? name : null;
}

/** what the marks show: the passenger lines' stations, with the tram lines' too, or every station */
type Shown = "passenger" | "tram" | "all";
/** line kinds the Passenger setting leaves out (map/realNetwork.ts) */
const NOT_PASSENGER = new Set(["tram", "funicular", "monorail", "narrow_gauge"]);

let nameUnder: (lng: number, lat: number) => string | null = () => null;
/** The name the hover gives the real station mark under (lng, lat) as drawn now, or null (the station
 * tool's own tag shows it while placing, since that tag holds the demand hover back, T-3802). */
export const realStationUnder = (lng: number, lat: number) => nameUnder(lng, lat);

export function startRealStations(map: maplibregl.Map, overlay: Overlay) {
  /** line id -> kind, per region, read for the marks only */
  const kinds = new Map<string, Promise<Map<string, string> | null>>();
  const kindsOf = (base: string, cc: string) => {
    let p = kinds.get(cc);
    if (!p) {
      p = patientFetch(`${base}data/${cc}/lines.json`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((j: { lines: { id: string; kind?: string }[] }) => new Map(j.lines.map((l) => [l.id, l.kind ?? ""])))
        .catch(() => null);
      kinds.set(cc, p);
    }
    return p;
  };
  /** the regions drawn, with the kinds of their lines (null: unknown, every station drawn) */
  const drawn = new Map<string, { st: Stations; kinds: Map<string, string> | null }>();
  let on = false;
  let shown: Shown = "passenger";
  /** the station tool is active: rings on top; otherwise soft dots in a map layer under the network */
  let placing = false;
  let painted = false;

  const keep = (r: { st: Stations; kinds: Map<string, string> | null }, i: number) => {
    if (shown === "all" || !r.kinds) return true;
    const ls = r.st.lines[i];
    if (!ls.length) return true;
    return ls.some((l) => {
      const k = r.kinds!.get(l) ?? "";
      return !NOT_PASSENGER.has(k) || (shown === "tram" && k === "tram");
    });
  };
  /** every line is a tram (or the like) and one is a tram: a smaller ring, shown while placing whatever the setting */
  const tramStop = (r: { st: Stations; kinds: Map<string, string> | null }, i: number) => {
    const ls = r.st.lines[i], k = r.kinds;
    return !!k && ls.length > 0 && ls.every((l) => NOT_PASSENGER.has(k.get(l) ?? "")) && ls.some((l) => k.get(l) === "tram");
  };
  /** what a station is drawn as: 0 nothing, 1 a station, 2 a tram stop (rings only; the dots keep `keep`) */
  const mark = (r: { st: Stations; kinds: Map<string, string> | null }, i: number) => (placing && tramStop(r, i) ? 2 : keep(r, i) ? 1 : 0);

  const data = () => {
    const features: Feature[] = [];
    for (const r of drawn.values())
      for (let i = 0; i < r.st.lng.length; i++)
        if (keep(r, i)) features.push({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [r.st.lng[i], r.st.lat[i]] } });
    return { type: "FeatureCollection" as const, features };
  };
  let drawnKey = "";
  const paint = () => {
    const c = SHADES[mapShade.peek()];
    map.setPaintProperty(REAL_STATIONS_LAYER, "circle-color", c.realMetro);
    map.setPaintProperty(REAL_STATIONS_LAYER, "circle-opacity", c.realDotOpacity);
  };
  /** the soft dots, a map layer under the game's network, shown when the station tool is not active */
  const drawDots = () => {
    if (!styleUp(map)) return; // style.load runs it again
    const show = on && !placing;
    const src = map.getSource(REAL_STATIONS_LAYER) as maplibregl.GeoJSONSource | undefined;
    if (!src) {
      if (!show) return;
      map.addSource(REAL_STATIONS_LAYER, { type: "geojson", data: data() });
      drawnKey = `${shown}:${[...drawn.keys()].join()}`;
      map.addLayer(
        {
          id: REAL_STATIONS_LAYER, type: "circle", source: REAL_STATIONS_LAYER, minzoom: MIN_ZOOM,
          paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 2, 12, 2.6, 14, 3.8, 16, 5.2, 18, 7] },
        },
        map.getLayer("tw-reach") ? "tw-reach" : firstSymbolId(map),
      );
      paint();
      painted = true;
      return;
    }
    map.setLayoutProperty(REAL_STATIONS_LAYER, "visibility", show ? "visible" : "none");
    if (!painted) paint(), (painted = true);
    const key = `${shown}:${[...drawn.keys()].join()}`;
    if (show && key !== drawnKey) src.setData(data()), (drawnKey = key);
  };

  // Drawn on a 2D canvas of its own above the game's network canvas (a map layer sits under it), so the
  // rings show over blueprints, track and platforms; under the station names (z 2 DOM) and the stop marks (z 3).
  const canvas = document.createElement("canvas");
  canvas.className = "overlay realst";
  overlay.canvas.after(canvas);
  const ctx = canvas.getContext("2d")!;
  /** ring radius in CSS px by zoom */
  const RADIUS: [number, number][] = [[10, 3], [12, 3.8], [14, 5], [16, 6.5], [18, 8.5]];
  const radius = (z: number) => {
    if (z <= RADIUS[0][0]) return RADIUS[0][1];
    for (let k = 1; k < RADIUS.length; k++)
      if (z <= RADIUS[k][0]) return RADIUS[k - 1][1] + ((RADIUS[k][1] - RADIUS[k - 1][1]) * (z - RADIUS[k - 1][0])) / (RADIUS[k][0] - RADIUS[k - 1][0]);
    return RADIUS[RADIUS.length - 1][1];
  };
  /** a tram stop's ring, radius and strokes, to a station's */
  const TRAM_RING = 0.65;

  const draw = () => {
    const z = map.getZoom();
    const live = on && placing && z >= MIN_ZOOM && drawn.size > 0;
    fitSparse(canvas, map, live);
    if (!live) return;
    // device px per CSS px as the canvas really is: MapLibre draws at less than its pixel ratio past
    // maxCanvasSize, and does not resize when the ratio changes alone (a window moved to another screen)
    const k = canvas.width / Math.max(1, overlay.cssW);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(k, 0, 0, k, 0, 0);
    const b = map.getBounds(), r = radius(z), pad = 0.002;
    const w = b.getWest() - pad, e = b.getEast() + pad, s = b.getSouth() - pad, n = b.getNorth() + pad;
    const paths = [new Path2D(), new Path2D()];
    for (const g of drawn.values())
      for (let i = 0; i < g.st.lng.length; i++) {
        const lng = g.st.lng[i], lat = g.st.lat[i];
        if (lng < w || lng > e || lat < s || lat > n) continue;
        const m = mark(g, i);
        if (!m) continue;
        const p = map.project([lng, lat]), ri = m === 2 ? r * TRAM_RING : r;
        paths[m - 1].moveTo(p.x + ri, p.y);
        paths[m - 1].arc(p.x, p.y, ri, 0, 2 * Math.PI);
      }
    const c = SHADES[mapShade.peek()];
    paths.forEach((path, j) => {
      const thin = j === 1 ? TRAM_RING : 1;
      // a halo in the map's casing colour keeps the ring readable on track and on both themes
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 4 * thin;
      ctx.strokeStyle = shadeColours().casing;
      ctx.stroke(path);
      ctx.globalAlpha = c.realStationOpacity;
      ctx.lineWidth = 2 * thin;
      ctx.strokeStyle = c.realMetro;
      ctx.stroke(path);
    });
    ctx.globalAlpha = 1;
  };
  overlay.afterMapFrame.push(draw);

  /** fetches the stations (and lines) of the regions in view and draws them */
  const asked = new Set<string>();
  const ensure = async () => {
    if (!on || map.getZoom() < MIN_ZOOM) return;
    const r = await regions();
    if (!r || !on) return;
    const b = map.getBounds();
    const hits = r.regions.filter((g) => !drawn.has(g.cc) && !asked.has(g.cc) && g.boxes.some((p) => p[0] <= b.getEast() && p[2] >= b.getWest() && p[1] <= b.getNorth() && p[3] >= b.getSouth()));
    for (const g of hits) {
      asked.add(g.cc);
      void Promise.all([stationsOf(r.base, g.cc), kindsOf(r.base, g.cc)]).then(([st, k]) => {
        asked.delete(g.cc);
        if (!st) return;
        drawn.set(g.cc, { st, kinds: k });
        drawDots();
        map.triggerRepaint();
      });
    }
  };

  effect(() => {
    const snap = snapping.value, mode = snapMode.value, d = display.value;
    const real = d.realNetwork;
    // marks go with the real network: shown by Settings, or while the track tool snaps to it
    on = d.realStations && (real !== "none" || snap);
    shown = real === "all" ? "all" : snap && mode === "tram" ? "tram" : "passenger";
    placing = tool.value === "station";
    mapShade.value;
    painted = false;
    drawDots();
    map.triggerRepaint();
    void ensure();
  });
  map.on("moveend", () => void ensure());
  map.on("style.load", () => {
    painted = false;
    drawDots();
  });

  const escape = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
  // the hover: the name only; above the bubbles (10), under the airports (11)
  nameUnder = (lng, lat) => {
    if (!on || map.getZoom() < MIN_ZOOM || hiddenOnGlobe(lng, lat)) return null;
    const at = map.project([lng, lat]);
    const edge = map.unproject([at.x + HOVER_PX, at.y]);
    const m = Math.abs(edge.lng - lng) * M_PER_DEG * Math.cos((lat * Math.PI) / 180);
    for (const n of nearby(lng, lat, Math.max(m, 1))) {
      const r = [...drawn.values()].find((x) => x.st === n.st);
      if (!r || !mark(r, n.i)) continue;
      const s = map.project([n.st.lng[n.i], n.st.lat[n.i]]);
      if (Math.hypot(s.x - at.x, s.y - at.y) > HOVER_PX) continue;
      return n.st.name[n.i] ?? (n.st.local[n.i] || null);
    }
    return null;
  };
  addFinder(10.5, (p) => {
    const name = nameUnder(p.lng, p.lat);
    return name ? [escape(name)] : null;
  });
  if (location.search.includes("debug=1"))
    (window as any).twRealStations = (lng?: number, lat?: number, m = NAME_M) => ({
      loaded: [...ready.keys()], drawn: [...drawn].map(([cc, r]) => ({ cc, n: r.st.lng.length, kinds: !!r.kinds })), on,
      near: lng === undefined ? undefined : nearby(lng, lat!, m).map((n) => ({ name: n.st.name[n.i], d: Math.round(n.d), lng: n.st.lng[n.i], lat: n.st.lat[n.i] })),
    });
}
