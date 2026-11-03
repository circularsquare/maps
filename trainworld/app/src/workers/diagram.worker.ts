// The network diagram's layout worker (T-2160, diagram/layout.ts): a graph in, its layout out. Each
// layout starts from the last one's B positions, so an edit moves little. The page sends one graph at
// a time (diagram/view.ts keeps only the newest waiting).

import type { DiagramGraph } from "../diagram/graph";
import { LAYOUT, LAYOUT_R2, layoutDiagram, type DiagramLayout, type LastPlaces } from "../diagram/layout";

export type ToDiagram = { kind: "graph"; graph: DiagramGraph; seq: number; options?: "r2" };
export type FromDiagram = { kind: "layout"; seq: number; graph: DiagramGraph; layout: DiagramLayout };

let last: LastPlaces | null = null;
let lastOptions: string | undefined;

self.onmessage = (e: MessageEvent<ToDiagram>) => {
  const m = e.data;
  if (m.kind !== "graph") return;
  if (m.options !== lastOptions) last = null;
  lastOptions = m.options;
  const layout = layoutDiagram(m.graph, m.options === "r2" ? LAYOUT_R2 : LAYOUT, last);
  last = new Map();
  for (let i = 0; i < m.graph.key.length; i++) last.set(m.graph.key[i], [layout.bx[i], layout.by[i]]);
  (self as unknown as Worker).postMessage({ kind: "layout", seq: m.seq, graph: m.graph, layout } satisfies FromDiagram);
};
