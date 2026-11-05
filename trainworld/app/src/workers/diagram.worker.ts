// The network diagram's layout worker (T-2160, diagram/layout.ts): a graph in, its layout, drawing
// (diagram/draw.ts) and names (diagram/names.ts) out. Each layout starts from the last one's places, so
// an edit moves little. The page sends one graph at a time (diagram/view.ts keeps only the newest
// waiting), with its stations' names measured in the page's font. A first layout posts how far it has
// got, for the loading bar (T-2235). The zones' borders (diagram/borders.ts, T-2464) and the city names
// far out (diagram/cityNames.ts, T-2438) are made after the names from the zone outlines and names the
// page sends, each zone once; outlines arriving after a layout redo that layout's borders and city names alone.

import type { DiagramGraph } from "../diagram/graph";
import { LAYOUT, LAYOUT_R2, lastPlaces, layoutDiagram, type DiagramLayout, type LastPlaces } from "../diagram/layout";
import { drawing, type Drawing } from "../diagram/draw";
import { placeNames, type NamePlan } from "../diagram/names";
import { zoneBorders, type DiagramBorders, type ZoneShape } from "../diagram/borders";
import { placeCityNames, type CityNames, type ZoneLabel } from "../diagram/cityNames";

/** per graph node: its name, and its width in CSS px on one line and on two (0 where not split) */
export type DiagramNames = { text: string[]; width: Float32Array; width2: Float32Array; fontPx: number };
/** zones' outlines as the page sends them: rings of mercator x, y (0 to 1), open; and the zone's name */
export type ZoneOutlines = { id: string; rings: Float64Array[]; label: ZoneLabel }[];
/** the graph's frame in mercator: graph metres are ((mx - frame[0]) * frame[2], -(my - frame[1]) * frame[2]) */
export type ZoneFrame = [number, number, number];
export type ToDiagram =
  | { kind: "graph"; graph: DiagramGraph; names: DiagramNames; seq: number; options?: "r2"; zones?: ZoneOutlines; frame?: ZoneFrame }
  | { kind: "zones"; zones: ZoneOutlines };
export type FromDiagram =
  | { kind: "layout"; seq: number; graph: DiagramGraph; layout: DiagramLayout; drawing: Drawing; names: NamePlan; ms: { drawing: number; names: number; borders: number; cities: number } }
  | { kind: "borders"; seq: number; borders: DiagramBorders; cities?: CityNames }
  | { kind: "progress"; seq: number; share: number };

/** the layout's share of a first build's time; the drawing and names take the rest */
const LAYOUT_SHARE = 0.85;

let last: LastPlaces | null = null;
let lastOptions: string | undefined;
const outlines = new Map<string, Float64Array[]>(), labels = new Map<string, ZoneLabel>();
let shown: { seq: number; graph: DiagramGraph; layout: DiagramLayout; d: Drawing; fontPx: number; frame?: ZoneFrame } | null = null;

/** the borders of the zones the page has sent, in the graph's frame; undefined without both */
function borders(graph: DiagramGraph, layout: DiagramLayout, d: Drawing, frame?: ZoneFrame): DiagramBorders | undefined {
  if (!frame || !outlines.size) return undefined;
  const [fx, fy, k] = frame, shapes: ZoneShape[] = [];
  for (const [id, rings] of outlines) shapes.push({ id, rings: rings.map((r) => {
    const out = new Float64Array(r.length);
    for (let i = 0; i < r.length; i += 2) (out[i] = (r[i] - fx) * k), (out[i + 1] = -(r[i + 1] - fy) * k);
    return out;
  }) });
  return zoneBorders(graph, layout.x, layout.y, d, shapes);
}

/** the city names, by the zones the borders found */
const cities = (layout: DiagramLayout, d: Drawing, b: DiagramBorders, fontPx: number): CityNames =>
  placeCityNames(d, layout.blocks, b.zoneOf, b.zones.map((id) => labels.get(id)), fontPx);

const keep = (zones: ZoneOutlines) => {
  for (const z of zones) outlines.set(z.id, z.rings), labels.set(z.id, z.label);
};

self.onmessage = (e: MessageEvent<ToDiagram>) => {
  const m = e.data;
  if (m.kind === "zones") {
    keep(m.zones);
    const s = shown, b = s && borders(s.graph, s.layout, s.d, s.frame);
    if (b) (self as unknown as Worker).postMessage({ kind: "borders", seq: s!.seq, borders: b, cities: cities(s!.layout, s!.d, b, s!.fontPx) } satisfies FromDiagram);
    return;
  }
  if (m.kind !== "graph") return;
  keep(m.zones ?? []);
  if (m.options !== lastOptions) last = null;
  lastOptions = m.options;
  let sent = -1;
  const progress = last ? undefined : (share: number) => {
    if (share - sent < 0.01) return;
    sent = share;
    (self as unknown as Worker).postMessage({ kind: "progress", seq: m.seq, share } satisfies FromDiagram);
  };
  const layout = layoutDiagram(m.graph, m.options === "r2" ? LAYOUT_R2 : LAYOUT, last, progress && ((s) => progress(s * LAYOUT_SHARE)));
  last = lastPlaces(m.graph, layout);
  const t0 = performance.now();
  const d = drawing(m.graph, layout);
  const t1 = performance.now();
  const names = placeNames(m.graph, layout, d, m.names.text, m.names.width, m.names.width2, m.names.fontPx, undefined, progress && ((s) => progress(LAYOUT_SHARE + s * (1 - LAYOUT_SHARE))));
  const t2 = performance.now();
  d.borders = borders(m.graph, layout, d, m.frame);
  const t3 = performance.now();
  if (d.borders) names.cities = cities(layout, d, d.borders, m.names.fontPx);
  shown = { seq: m.seq, graph: m.graph, layout, d, fontPx: m.names.fontPx, frame: m.frame };
  const ms = { drawing: t1 - t0, names: t2 - t1, borders: t3 - t2, cities: performance.now() - t3 };
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout, drawing: d, names, ms } satisfies FromDiagram);
};
