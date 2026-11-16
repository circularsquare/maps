// The network diagram's layout worker (T-2160, diagram/layout.ts): a graph in, its layout, drawing
// (diagram/draw.ts) and names (diagram/names.ts) out. The page sends one graph at a time (diagram/view.ts keeps only
// the newest waiting), with its stations' names measured in the page's font. How far a build has got is posted for the
// loading bar (T-2235). The rubber sheet (diagram/sheet.ts, T-2568, T-2567, T-2580) is made first, from the water masks
// of the zones the page names, fetched here from the packs, and posted; the lines are laid out on it and the map carried
// with them (diagram/sheetLines.ts), whose water (with the page's `water`) and zone borders (diagram/borders.ts, T-2464)
// go with the drawing. The city names far out (diagram/cityNames.ts, T-2438) are made after the names from the zone
// outlines and names the page sends, each zone once; outlines arriving after a layout redo that layout's borders and city
// names alone. The carry is solved in the track module's WASM (T-2589; the page's `?perfOff=carryWasm` the same in
// JavaScript). The page sends one graph per region (diagram/regions.ts, T-3448); the layouts it still draws are kept here
// by seq, each given its borders and city names as outlines arrive.

import type { DiagramGraph } from "../diagram/graph";
import type { DiagramLayout } from "../diagram/layout";
import { drawing, type Drawing } from "../diagram/draw";
import { placeNames, type NamePlan } from "../diagram/names";
import type { DiagramBorders, ZoneShape } from "../diagram/borders";
import { placeCityNames, type CityNames, type ZoneLabel } from "../diagram/cityNames";
import { groundSea, realWater, type GroundSea, type RealWater, type WaterMask } from "../diagram/waterReal";
import type { DiagramWater } from "../diagram/water";
import { rubberSheet, sheetWarp, type Sheet } from "../diagram/sheet";
import { riverCellM, setCarrySolver, SHEET_LINES, sheetLines, sheetMap, type CarrySolverMaker, type SheetLines } from "../diagram/sheetLines";
import { wasmCarrySolver } from "../diagram/carryWasm";

/** per graph node: its name, and its width in CSS px on one line and on two (0 where not split) */
export type DiagramNames = { text: string[]; width: Float32Array; width2: Float32Array; fontPx: number };
/** zones' outlines as the page sends them: rings of mercator x, y (0 to 1), open; and the zone's name */
export type ZoneOutlines = { id: string; rings: Float64Array[]; label: ZoneLabel }[];
/** the graph's frame in mercator: graph metres are ((mx - frame[0]) * frame[2], -(my - frame[1]) * frame[2]) */
export type ZoneFrame = [number, number, number];
/** a zone's cities and towns (villages where it has none) as mercator x, y (0 to 1), with their widths */
export type ZoneTowns = { name: string; width: number; x: number; y: number; big: boolean }[];
export type ToDiagram =
  | { kind: "graph"; graph: DiagramGraph; names: DiagramNames; seq: number; zones?: ZoneOutlines; frame: ZoneFrame; water: boolean; sheet: { packs: string; zones: string[]; carryJs?: boolean }; keep?: number[] }
  | { kind: "zones"; zones: ZoneOutlines }
  | { kind: "towns"; towns: { id: string; list: ZoneTowns }[] };
export type FromDiagram =
  | { kind: "layout"; seq: number; graph: DiagramGraph; layout: DiagramLayout; drawing: Drawing; names: NamePlan; ms: { drawing: number; names: number; borders: number; cities: number } }
  | { kind: "borders"; seq: number; borders: DiagramBorders; cities?: CityNames }
  | { kind: "water"; seq: number; water: DiagramWater }
  | { kind: "progress"; seq: number; share: number }
  | { kind: "sheet"; seq: number; sheet: Sheet; ms: { masks: number; ground: number; sheet: number; lines: number; rivers: number; map: number }; lines: SheetLines["info"] & { map: Record<string, number> } };

const outlines = new Map<string, Float64Array[]>(), labels = new Map<string, ZoneLabel>();
/** zones' place lists as the page sends them (T-2447), places in mercator */
const towns = new Map<string, ZoneTowns>();
/** a layout on the sheet (T-2567): its lines and carried map, the graph with real places, whether the sea is drawn,
 * and the map from real places to the sheet's graph */
type OnSheet = { lines: SheetLines; real: DiagramGraph; water: boolean; rivers: RealWater | null; toSheet: (x: number, y: number) => [number, number] };
type Shown = { seq: number; graph: DiagramGraph; layout: DiagramLayout; d: Drawing; fontPx: number; frame: ZoneFrame; sheet: OnSheet };
/** the layouts the page still draws, by seq: one per region (T-3448), each given its borders and city names as outlines and
 * place lists arrive */
const shown = new Map<number, Shown>();

/** the zones the page has sent, in the graph's frame */
function shapesOf(frame: ZoneFrame): ZoneShape[] {
  const [fx, fy, k] = frame, shapes: ZoneShape[] = [];
  for (const [id, rings] of outlines) shapes.push({ id, rings: rings.map((r) => {
    const out = new Float64Array(r.length);
    for (let i = 0; i < r.length; i += 2) (out[i] = (r[i] - fx) * k), (out[i + 1] = -(r[i + 1] - fy) * k);
    return out;
  }) });
  return shapes;
}
/** the borders of the zones the page has sent; undefined without them and the frame; on the sheet, its map's (with its
 * water) */
function borders(d: Drawing, frame: ZoneFrame, sheet: OnSheet): DiagramBorders | undefined {
  if (!outlines.size) return undefined;
  const m = sheetMap(sheet.lines, sheet.real, shapesOf(frame), sheet.water, SHEET_LINES, false, sheet.rivers);
  if (m.water) d.water = m.water;
  return m.borders;
}

/** the city names, by the zones the borders found; a zone whose place list has arrived carries its towns, in the
 * graph's metres (on the sheet, its places), for blocks the zone's own name is taken from (T-2447) */
const cities = (graph: DiagramGraph, layout: DiagramLayout, b: DiagramBorders, fontPx: number, frame: ZoneFrame, sheet: OnSheet): CityNames => {
  const [fx, fy, k] = frame;
  const at = (x: number, y: number): [number, number] => sheet.toSheet((x - fx) * k, -(y - fy) * k);
  const zones = b.zones.map((id) => {
    const l = labels.get(id), t = towns.get(id);
    return l && t ? { ...l, towns: t.map((p) => { const [x, y] = at(p.x, p.y); return { name: p.name, width: p.width, x, y, big: p.big }; }) } : l;
  });
  return placeCityNames(graph, layout, layout.blocks, b.zoneOf, zones, fontPx, b.zones);
};

const keep = (zones: ZoneOutlines) => {
  for (const z of zones) outlines.set(z.id, z.rings), labels.set(z.id, z.label);
};

self.onmessage = async (e: MessageEvent<ToDiagram>) => {
  const m = e.data;
  if (m.kind === "zones" || m.kind === "towns") {
    if (m.kind === "zones") keep(m.zones);
    else for (const t of m.towns) towns.set(t.id, t.list);
    for (const s of shown.values()) {
      const b = m.kind === "towns" ? s.d.borders : borders(s.d, s.frame, s.sheet);
      if (b) (self as unknown as Worker).postMessage({ kind: "borders", seq: s.seq, borders: b, cities: cities(s.graph, s.layout, b, s.fontPx, s.frame, s.sheet) } satisfies FromDiagram);
      // the zones are part of the carried map, its water drawn again with them
      if (b && m.kind === "zones" && s.d.water) (self as unknown as Worker).postMessage({ kind: "water", seq: s.seq, water: s.d.water } satisfies FromDiagram);
    }
    return;
  }
  if (m.kind !== "graph") return;
  keep(m.zones ?? []);
  if (m.keep) for (const seq of [...shown.keys()]) if (!m.keep.includes(seq)) shown.delete(seq);
  return onSheet(m, m.sheet, m.frame);
};

/** The layout on the rubber sheet (T-2567, T-2580), its borders and
 * (with the page's `water`) its sea and narrow rivers from the map carried with the stations, posted with the layout; how
 * far it has got posted for the loading bar, in shares of the time (the lines and the carry most of it) */
async function onSheet(m: Extract<ToDiagram, { kind: "graph" }>, at: { packs: string; zones: string[]; carryJs?: boolean }, frame: ZoneFrame) {
  const t0 = performance.now();
  setCarrySolver(at.carryJs ? null : await carryWasm());
  let sent = -1;
  const progress = (share: number) => {
    if (share - sent < 0.01) return;
    sent = share;
    (self as unknown as Worker).postMessage({ kind: "progress", seq: m.seq, share } satisfies FromDiagram);
  };
  progress(0);
  const got = await masksFor(at.packs, at.zones);
  const t1 = performance.now(), key = groundKey(m.graph, frame, got.length);
  const sea = kept(seaKnown, key, () => groundSea(m.graph, got, frame));
  const t2 = performance.now();
  const sheet = rubberSheet(m.graph, { masks: got, frame, sea });
  const t3 = performance.now();
  progress(0.1);
  const lines = sheetLines(m.graph, sheet, SHEET_LINES, (s) => progress(0.1 + 0.7 * s)), g = lines.graph, layout = lines.layout, K = lines.k, warp = sheetWarp(sheet);
  const t4 = performance.now();
  const d = drawing(g, layout);
  const names = placeNames(g, layout, d, m.names.text, m.names.width, m.names.width2, m.names.fontPx);
  const t5 = performance.now();
  // the narrow rivers on the ground, with the water on (kept while the stations and tracks are the same)
  let rivers: RealWater | null = null;
  const cellM = riverCellM(m.graph);
  if (m.water && cellM) {
    rivers = kept(sheetRivers, key, () => realWater(m.graph, got, frame, sea, cellM));
  }
  const t5r = performance.now();
  progress(0.9);
  const on: OnSheet = { lines, real: m.graph, water: !!m.water, rivers, toSheet: (x, y) => { const [a, b] = warp(x, y); return [a * K, b * K]; } };
  const map = sheetMap(lines, m.graph, shapesOf(frame), on.water, SHEET_LINES, false, rivers);
  if (map.water) d.water = map.water;
  if (outlines.size) (d.borders = map.borders), (names.cities = cities(g, layout, map.borders, m.names.fontPx, frame, on));
  const t6 = performance.now();
  shown.set(m.seq, { seq: m.seq, graph: g, layout, d, fontPx: m.names.fontPx, frame, sheet: on });
  const { raster: _, ...posted } = sheet;
  (self as unknown as Worker).postMessage({ kind: "sheet", seq: m.seq, sheet: posted, ms: { masks: t1 - t0, ground: t2 - t1, sheet: t3 - t2, lines: t4 - t3, rivers: t5r - t5, map: t6 - t5r }, lines: { ...lines.info, map: map.info } } satisfies FromDiagram);
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout, drawing: d, names, ms: { drawing: t5 - t4, names: 0, borders: t6 - t5, cities: 0 } } satisfies FromDiagram);
}

/** the carry's solver in WASM (T-2589), from the track module, loaded with the first sheet; null if it did not load (the
 * JavaScript one gives the same, slower) */
let carryMaker: Promise<CarrySolverMaker | null> | null = null;
const carryWasm = () =>
  (carryMaker ??= import("../wasm/track/sim")
    .then(async (w) => {
      await w.default();
      return w.CarrySolver ? wasmCarrySolver(w.CarrySolver) : null;
    })
    .catch((e) => {
      console.warn("The diagram's carry solves in JavaScript: its WASM did not load", e);
      return null;
    }));

// ---- the water (T-2169): each zone's mask fetched once, as the clock worker fetches it (a range of the pack's .bin)
const masks = new Map<string, Promise<WaterMask | null>>();
function maskOf(packs: string, id: string): Promise<WaterMask | null> {
  let p = masks.get(id);
  if (!p) {
    const url = new URL(id, packs).href;
    p = (async () => {
      const head = await (await fetch(url + ".json")).json();
      const row = head.arrays.find((a: { name: string }) => a.name === "water_row"), xs = head.arrays.find((a: { name: string }) => a.name === "water_x");
      if (!head.water || !row || !xs) return null;
      const start = row.offset, end = xs.offset + xs.count * 2;
      const res = await fetch(url + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` }, cache: "no-store" });
      if (!res.ok) throw new Error(`${res.status} ${url}.bin`);
      const buf = await res.arrayBuffer(), base = res.status === 206 ? start : 0, w = head.water;
      return {
        cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1],
        rows: new Uint32Array(buf.slice(row.offset - base, row.offset - base + row.count * 4)), runs: new Uint16Array(buf.slice(xs.offset - base, xs.offset - base + xs.count * 2)),
        lon: head.origin.lon, lat: head.origin.lat,
      };
    })();
    // a mask that failed is fetched again by the next layout
    p.catch(() => masks.get(id) === p && masks.delete(id));
    masks.set(id, p);
  }
  return p;
}
/** the sea on the ground, kept while the stations' places and the hops' tracks are the same (a few, one per region) */
const seaKnown = new Map<string, GroundSea>();
/** on the sheet, the narrow rivers on the ground (a finer river finder than the water's, T-2580) */
const sheetRivers = new Map<string, RealWater>();
/** ground kept for this many graphs at most, the oldest dropped */
const GROUND_KEPT = 4;
function kept<T>(m: Map<string, T>, key: string, make: () => T): T {
  let v = m.get(key);
  if (v === undefined) {
    v = make();
    m.set(key, v);
    while (m.size > GROUND_KEPT) m.delete(m.keys().next().value!);
  }
  return v;
}
/** zones' masks fetched at once at most: a sheet over a continent meets hundreds of zones, and that many fetches at once
 * fail in the browser (ERR_INSUFFICIENT_RESOURCES), leaving random zones with no water (T-3448) */
const MASK_FETCHES = 32;
async function masksFor(packs: string, zones: string[]): Promise<WaterMask[]> {
  const got: (WaterMask | null)[] = new Array(zones.length).fill(null);
  let next = 0;
  const fetcher = async () => {
    while (next < zones.length) {
      const k = next++;
      // one more try a second later: a failed mask is dropped from the cache, so this fetches it again
      got[k] = await maskOf(packs, zones[k]).catch(() => new Promise<void>((r) => setTimeout(r, 1000)).then(() => maskOf(packs, zones[k]))).catch(() => null);
    }
  };
  await Promise.all(Array.from({ length: Math.min(MASK_FETCHES, zones.length) }, fetcher));
  return got.filter((w): w is WaterMask => !!w);
}
function groundKey(g: DiagramGraph, frame: ZoneFrame, masks: number): string {
  let sum = 0;
  for (let i = 0; i < g.key.length; i += 7) sum += g.x[i] * 3 + g.y[i];
  if (g.hop) for (let i = 0; i < g.hop.length; i += 13) sum += g.hop[i];
  return `${frame.join()}|${g.key.length}|${g.ea.length}|${sum}|${masks}`;
}
