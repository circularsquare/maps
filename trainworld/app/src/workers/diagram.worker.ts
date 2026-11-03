// The network diagram's layout worker (T-2160, diagram/layout.ts): a graph in, its layout, drawing
// (diagram/draw.ts) and names (diagram/names.ts) out. Each layout starts from the last one's places, so
// an edit moves little. The page sends one graph at a time (diagram/view.ts keeps only the newest
// waiting), with its stations' names measured in the page's font. A first layout posts how far it has
// got, for the loading bar (T-2235).

import type { DiagramGraph } from "../diagram/graph";
import { LAYOUT, LAYOUT_R2, lastPlaces, layoutDiagram, type DiagramLayout, type LastPlaces } from "../diagram/layout";
import { drawing, type Drawing } from "../diagram/draw";
import { placeNames, type NamePlan } from "../diagram/names";

/** per graph node: its name, and its width in CSS px on one line and on two (0 where not split) */
export type DiagramNames = { text: string[]; width: Float32Array; width2: Float32Array; fontPx: number };
export type ToDiagram = { kind: "graph"; graph: DiagramGraph; names: DiagramNames; seq: number; options?: "r2" };
export type FromDiagram =
  | { kind: "layout"; seq: number; graph: DiagramGraph; layout: DiagramLayout; drawing: Drawing; names: NamePlan; ms: { drawing: number; names: number } }
  | { kind: "progress"; seq: number; share: number };

/** the layout's share of a first build's time; the drawing and names take the rest */
const LAYOUT_SHARE = 0.85;

let last: LastPlaces | null = null;
let lastOptions: string | undefined;

self.onmessage = (e: MessageEvent<ToDiagram>) => {
  const m = e.data;
  if (m.kind !== "graph") return;
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
  const ms = { drawing: t1 - t0, names: performance.now() - t1 };
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout, drawing: d, names, ms } satisfies FromDiagram);
};
