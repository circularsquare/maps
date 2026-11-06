// The network diagram's layout worker (T-2160, diagram/layout.ts): a graph in, its layout, drawing
// (diagram/draw.ts) and names (diagram/names.ts) out. Each layout starts from the last one's places, so
// an edit moves little. The page sends one graph at a time (diagram/view.ts keeps only the newest
// waiting), with its stations' names measured in the page's font. A first layout posts how far it has
// got, for the loading bar (T-2235). The zones' borders (diagram/borders.ts, T-2464) and the city names
// far out (diagram/cityNames.ts, T-2438) are made after the names from the zone outlines and names the
// page sends, each zone once; outlines arriving after a layout redo that layout's borders and city names alone. The
// water (diagram/waterReal.ts, diagram/water.ts, T-2169) comes last, from the water masks of the zones the page names,
// fetched here from the packs, and is posted on its own; its sea part is found before a first layout, which keeps the
// sea open (T-2541). With the page's `sheet` (`?sheet=1`, T-2568, T-2567) the rubber sheet (diagram/sheet.ts) is made
// first, from the same masks, and posted; the lines are laid out on it and the map carried with them
// (diagram/sheetLines.ts), whose water (with the page's `water`) and zone borders replace the ones above.

import type { DiagramGraph } from "../diagram/graph";
import { dialled, LAYOUT, LAYOUT_R2, lastPlaces, layoutDiagram, type DiagramLayout, type LastPlaces } from "../diagram/layout";
import { drawing, type Drawing } from "../diagram/draw";
import { placeNames, type NamePlan } from "../diagram/names";
import { zoneBorders, type DiagramBorders, type ZoneShape } from "../diagram/borders";
import { placeCityNames, type CityNames, type ZoneLabel } from "../diagram/cityNames";
import { groundSea, realWater, type GroundSea, type RealWater, type WaterMask } from "../diagram/waterReal";
import { diagramWater, type DiagramWater } from "../diagram/water";
import { rubberSheet, sheetWarp, type Sheet } from "../diagram/sheet";
import { sheetLines, sheetMap, type SheetLines } from "../diagram/sheetLines";

/** per graph node: its name, and its width in CSS px on one line and on two (0 where not split) */
export type DiagramNames = { text: string[]; width: Float32Array; width2: Float32Array; fontPx: number };
/** zones' outlines as the page sends them: rings of mercator x, y (0 to 1), open; and the zone's name */
export type ZoneOutlines = { id: string; rings: Float64Array[]; label: ZoneLabel }[];
/** the graph's frame in mercator: graph metres are ((mx - frame[0]) * frame[2], -(my - frame[1]) * frame[2]) */
export type ZoneFrame = [number, number, number];
/** a zone's cities and towns (villages where it has none) as mercator x, y (0 to 1), with their widths */
export type ZoneTowns = { name: string; width: number; x: number; y: number; big: boolean }[];
export type ToDiagram =
  | { kind: "graph"; graph: DiagramGraph; names: DiagramNames; seq: number; options?: "r2"; distort?: number; zones?: ZoneOutlines; frame?: ZoneFrame; water?: { packs: string; zones: string[] }; sheet?: { packs: string; zones: string[] } }
  | { kind: "zones"; zones: ZoneOutlines }
  | { kind: "towns"; towns: { id: string; list: ZoneTowns }[] };
export type FromDiagram =
  | { kind: "layout"; seq: number; graph: DiagramGraph; layout: DiagramLayout; drawing: Drawing; names: NamePlan; ms: { drawing: number; names: number; borders: number; cities: number } }
  | { kind: "borders"; seq: number; borders: DiagramBorders; cities?: CityNames }
  | { kind: "water"; seq: number; water: DiagramWater; ms: { masks: number; ground: number; drawn: number } }
  | { kind: "progress"; seq: number; share: number }
  | { kind: "sheet"; seq: number; sheet: Sheet; ms: { masks: number; ground: number; sheet: number; lines: number; map: number }; lines: SheetLines["info"] & { map: Record<string, number> } };

/** the layout's share of a first build's time; the drawing and names take the rest */
const LAYOUT_SHARE = 0.85;

let last: LastPlaces | null = null;
let lastOptions = "";
const outlines = new Map<string, Float64Array[]>(), labels = new Map<string, ZoneLabel>();
/** zones' place lists as the page sends them (T-2447), places in mercator */
const towns = new Map<string, ZoneTowns>();
/** a layout on the sheet (T-2567): its lines and carried map, the graph with real places, whether the sea is drawn,
 * and the map from real places to the sheet's graph */
type OnSheet = { lines: SheetLines; real: DiagramGraph; water: boolean; toSheet: (x: number, y: number) => [number, number] };
let shown: { seq: number; graph: DiagramGraph; layout: DiagramLayout; d: Drawing; fontPx: number; frame?: ZoneFrame; sheet?: OnSheet } | null = null;

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
function borders(graph: DiagramGraph, layout: DiagramLayout, d: Drawing, frame?: ZoneFrame, sheet?: OnSheet): DiagramBorders | undefined {
  if (!frame || !outlines.size) return undefined;
  if (sheet) {
    const m = sheetMap(sheet.lines, sheet.real, shapesOf(frame), sheet.water);
    if (m.water) d.water = m.water;
    return m.borders;
  }
  return zoneBorders(graph, layout.x, layout.y, d, shapesOf(frame));
}

/** the city names, by the zones the borders found; a zone whose place list has arrived carries its towns, in the
 * graph's metres (on the sheet, its places), for blocks the zone's own name is taken from (T-2447) */
const cities = (graph: DiagramGraph, layout: DiagramLayout, d: Drawing, b: DiagramBorders, fontPx: number, frame?: ZoneFrame, sheet?: OnSheet): CityNames => {
  const [fx, fy, k] = frame ?? [0, 0, 1];
  const at = (x: number, y: number): [number, number] => (sheet ? sheet.toSheet((x - fx) * k, -(y - fy) * k) : [(x - fx) * k, -(y - fy) * k]);
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
    const s = shown, b = s && (m.kind === "towns" ? s.d.borders : borders(s.graph, s.layout, s.d, s.frame, s.sheet));
    if (b) (self as unknown as Worker).postMessage({ kind: "borders", seq: s!.seq, borders: b, cities: cities(s!.graph, s!.layout, s!.d, b, s!.fontPx, s!.frame, s!.sheet) } satisfies FromDiagram);
    // on the sheet the zones are part of the carried map, its water drawn again with them
    if (b && m.kind === "zones" && s!.sheet && s!.d.water) (self as unknown as Worker).postMessage({ kind: "water", seq: s!.seq, water: s!.d.water, ms: { masks: 0, ground: 0, drawn: 0 } } satisfies FromDiagram);
    return;
  }
  if (m.kind !== "graph") return;
  keep(m.zones ?? []);
  if (m.sheet && m.frame) return onSheet(m, m.sheet, m.frame);
  const opts = `${m.options}|${m.distort}`;
  if (opts !== lastOptions) last = null;
  lastOptions = opts;
  // the layout keeps the water on the ground open (T-2541): a first layout waits for the sea, a rebuild takes the last
  // found (its stations near where they were)
  const sea = m.water && m.frame ? (last && seaKnown ? seaKnown.sea : await seaOf(m.graph, m.water.packs, m.water.zones, m.frame)) : null;
  let sent = -1;
  const progress = last ? undefined : (share: number) => {
    if (share - sent < 0.01) return;
    sent = share;
    (self as unknown as Worker).postMessage({ kind: "progress", seq: m.seq, share } satisfies FromDiagram);
  };
  const layout = layoutDiagram(m.graph, m.options === "r2" ? LAYOUT_R2 : m.distort === undefined ? LAYOUT : dialled(m.distort), last, progress && ((s) => progress(s * LAYOUT_SHARE)), sea);
  last = lastPlaces(m.graph, layout);
  const t0 = performance.now();
  const d = drawing(m.graph, layout);
  const t1 = performance.now();
  const names = placeNames(m.graph, layout, d, m.names.text, m.names.width, m.names.width2, m.names.fontPx, undefined, progress && ((s) => progress(LAYOUT_SHARE + s * (1 - LAYOUT_SHARE))));
  const t2 = performance.now();
  d.borders = borders(m.graph, layout, d, m.frame);
  const t3 = performance.now();
  if (d.borders) names.cities = cities(m.graph, layout, d, d.borders, m.names.fontPx, m.frame);
  shown = { seq: m.seq, graph: m.graph, layout, d, fontPx: m.names.fontPx, frame: m.frame };
  const ms = { drawing: t1 - t0, names: t2 - t1, borders: t3 - t2, cities: performance.now() - t3 };
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout, drawing: d, names, ms } satisfies FromDiagram);
  if (m.water && m.frame) void water(m.seq, m.water.packs, m.water.zones, m.frame);
};

/** `?sheet=1` (T-2567): the layout on the rubber sheet, its borders and (with the page's `water`) its sea from the map
 * carried with the stations, posted with the layout */
async function onSheet(m: Extract<ToDiagram, { kind: "graph" }>, at: { packs: string; zones: string[] }, frame: ZoneFrame) {
  const t0 = performance.now();
  const got = await masksFor(at.packs, at.zones);
  const t1 = performance.now(), key = groundKey(m.graph, frame, got.length);
  if (seaKnown?.key !== key) seaKnown = { key, sea: groundSea(m.graph, got, frame) };
  const t2 = performance.now();
  const sheet = rubberSheet(m.graph, { masks: got, frame, sea: seaKnown.sea });
  const t3 = performance.now();
  const lines = sheetLines(m.graph, sheet), g = lines.graph, layout = lines.layout, K = lines.k, warp = sheetWarp(sheet);
  const t4 = performance.now();
  const d = drawing(g, layout);
  const names = placeNames(g, layout, d, m.names.text, m.names.width, m.names.width2, m.names.fontPx);
  const t5 = performance.now();
  const on: OnSheet = { lines, real: m.graph, water: !!m.water, toSheet: (x, y) => { const [a, b] = warp(x, y); return [a * K, b * K]; } };
  const map = sheetMap(lines, m.graph, shapesOf(frame), on.water);
  if (map.water) d.water = map.water;
  if (outlines.size) (d.borders = map.borders), (names.cities = cities(g, layout, d, map.borders, m.names.fontPx, frame, on));
  const t6 = performance.now();
  shown = { seq: m.seq, graph: g, layout, d, fontPx: m.names.fontPx, frame, sheet: on };
  const { raster: _, ...posted } = sheet;
  (self as unknown as Worker).postMessage({ kind: "sheet", seq: m.seq, sheet: posted, ms: { masks: t1 - t0, ground: t2 - t1, sheet: t3 - t2, lines: t4 - t3, map: t6 - t5 }, lines: { ...lines.info, map: map.info } } satisfies FromDiagram);
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout, drawing: d, names, ms: { drawing: t5 - t4, names: 0, borders: t6 - t5, cities: 0 } } satisfies FromDiagram);
}

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
      const res = await fetch(url + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` } });
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
/** the water on the ground, kept while the stations' places and the hops' tracks are the same; and its sea part, which
 * the layout reads first (T-2541) */
let ground: { key: string; real: RealWater } | null = null;
let seaKnown: { key: string; sea: GroundSea } | null = null;
const masksFor = async (packs: string, zones: string[]) => (await Promise.all(zones.map((id) => maskOf(packs, id).catch(() => null)))).filter((w): w is WaterMask => !!w);
function groundKey(g: DiagramGraph, frame: ZoneFrame, masks: number): string {
  let sum = 0;
  for (let i = 0; i < g.key.length; i += 7) sum += g.x[i] * 3 + g.y[i];
  if (g.hop) for (let i = 0; i < g.hop.length; i += 13) sum += g.hop[i];
  return `${frame.join()}|${g.key.length}|${g.ea.length}|${sum}|${masks}`;
}
async function seaOf(g: DiagramGraph, packs: string, zones: string[], frame: ZoneFrame): Promise<GroundSea> {
  const got = await masksFor(packs, zones), key = groundKey(g, frame, got.length);
  if (seaKnown?.key !== key) seaKnown = { key, sea: groundSea(g, got, frame) };
  return seaKnown.sea;
}
async function water(seq: number, packs: string, zones: string[], frame: ZoneFrame) {
  const t0 = performance.now();
  const got = await masksFor(packs, zones);
  // a newer layout came meanwhile: its own water follows
  const s = shown;
  if (!s || s.seq !== seq) return;
  const t1 = performance.now(), g = s.graph;
  const key = groundKey(g, frame, got.length);
  if (ground?.key !== key) ground = { key, real: realWater(g, got, frame, seaKnown?.key === key ? seaKnown.sea : undefined) };
  const t2 = performance.now();
  const w = diagramWater(g, s.layout.x, s.layout.y, s.d, ground.real);
  s.d.water = w;
  (self as unknown as Worker).postMessage({ kind: "water", seq, water: w, ms: { masks: t1 - t0, ground: t2 - t1, drawn: performance.now() - t2 } } satisfies FromDiagram);
}
