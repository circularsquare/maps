// The network diagram's view, rough (T-2160; T-2161 builds the real one): the button at the map's
// upper left and key M swap the map for a 2D canvas of the laid-out diagram (ribbons, rounded bends,
// pills, no names yet), drag to pan, wheel to zoom. While it is up the clock worker posts the graph
// after each network state; the layout worker lays it out, one at a time, the newest waiting next,
// and the last drawing stays up meanwhile. `?diagram=r2` lays out with round 2's options.

import { signal } from "@preact/signals";
import type { ClockClient } from "../workers/clockClient";
import type { DiagramGraph } from "./graph";
import type { FromDiagram, ToDiagram } from "../workers/diagram.worker";
import { drawing, paint, type Drawing } from "./draw";
import { keyIsTheirs } from "../ui/keys";
import { live } from "../map/live";
import { toLocal } from "../map/geo";

/** whether the diagram is up instead of the map */
export const diagramShown = signal(false);
export const DIAGRAM_KEY = "m";
/** a pitch this many CSS px when the diagram first opens */
const START_PITCH_PX = 7;

/** the graph node nearest a place on the ground */
function nearest(g: DiagramGraph, lng: number, lat: number): number {
  const [lx, ly] = toLocal(lng, lat), mx = (lx - g.mid[0]) / g.k, my = -(ly - g.mid[1]) / g.k;
  let best = 0, bd = Infinity;
  for (let i = 0; i < g.key.length; i++) {
    const d = (g.x[i] - mx) ** 2 + (g.y[i] - my) ** 2;
    if (d < bd) (bd = d), (best = i);
  }
  return best;
}

export function startDiagramView(client: ClockClient, mapEl: HTMLElement): void {
  const options = new URLSearchParams(location.search).get("diagram") === "r2" ? "r2" : undefined;
  const canvas = document.createElement("canvas");
  canvas.className = "diagram-canvas";
  Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", zIndex: "3", display: "none", cursor: "grab", touchAction: "none" });
  mapEl.appendChild(canvas);
  let worker: Worker | null = null;
  let busy = false, waiting: DiagramGraph | null = null, seq = 0;
  let shown: { graph: DiagramGraph; d: Drawing; x: Float64Array; y: Float64Array } | null = null;
  let cx = 0, cy = 0, scale = 0, placed = false;
  const send = (g: DiagramGraph) => {
    if (busy) {
      waiting = g;
      return;
    }
    busy = true;
    worker!.postMessage({ kind: "graph", graph: g, seq: ++seq, options } satisfies ToDiagram);
  };
  client.onDiagram = (g) => {
    if (!diagramShown.peek()) return;
    if (!worker) {
      worker = new Worker(new URL("../workers/diagram.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (e: MessageEvent<FromDiagram>) => {
        busy = false;
        const { graph, layout } = e.data;
        shown = { graph, d: drawing(graph, layout), x: layout.x, y: layout.y };
        // for checks and `?debug=1`: the last layout's figures and how to frame a station
        (window as unknown as { twDiagram: unknown }).twDiagram = { layouts: e.data.seq, ms: layout.ms, complexes: graph.key.length, pieces: layout.pieces.length, centre: (lng: number, lat: number, pitchPx: number) => { const i = nearest(graph, lng, lat); (cx = layout.x[i]), (cy = layout.y[i]), (scale = pitchPx / shown!.d.pitch), draw(); } };
        if (!placed) place(graph, layout.x, layout.y, shown.d.pitch);
        draw();
        const next = waiting;
        waiting = null;
        if (next) send(next);
      };
    }
    send(g);
  };
  /** first opened: the complex nearest the map's centre in the middle, a pitch START_PITCH_PX */
  const place = (g: DiagramGraph, x: Float64Array, y: Float64Array, pitch: number) => {
    placed = true;
    scale = START_PITCH_PX / pitch;
    const c = live.map?.getCenter();
    const best = c ? nearest(g, c.lng, c.lat) : 0;
    cx = x[best] ?? 0;
    cy = y[best] ?? 0;
  };
  const theme = () => {
    const s = getComputedStyle(document.body);
    const v = (k: string, d: string) => s.getPropertyValue(k).trim() || d;
    const dark = document.documentElement.dataset.shade === "dark";
    return { bg: v("--bg", "#fbf5ef"), ink: v("--net-ink", "#2b2b2b"), pill: dark ? v("--bg", "#2b2428") : "#ffffff" };
  };
  const draw = () => {
    if (!diagramShown.peek()) return;
    const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) (canvas.width = Math.round(w * dpr)), (canvas.height = Math.round(h * dpr));
    const ctx = canvas.getContext("2d")!;
    const t = theme();
    if (!shown) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }
    paint(ctx, shown.d, canvas.width, canvas.height, cx, cy, scale * dpr, t);
  };
  new ResizeObserver(() => draw()).observe(canvas);
  diagramShown.subscribe((on) => {
    canvas.style.display = on ? "block" : "none";
    client.post({ kind: "diagram", on });
    if (on) draw();
  });
  // drag to pan, wheel to zoom about the pointer
  let drag: { x: number; y: number } | null = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || !scale) return;
    cx -= (e.clientX - drag.x) / scale;
    cy += (e.clientY - drag.y) / scale;
    drag = { x: e.clientX, y: e.clientY };
    draw();
  });
  const up = () => {
    drag = null;
    canvas.style.cursor = "grab";
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (!scale) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left - r.width / 2, py = e.clientY - r.top - r.height / 2;
    const f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
    // the point under the pointer stays there
    cx += px / scale - px / (scale * f);
    cy -= py / scale - py / (scale * f);
    scale *= f;
    draw();
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (keyIsTheirs(e) || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    if (e.key.toLowerCase() !== DIAGRAM_KEY) return;
    e.preventDefault();
    diagramShown.value = !diagramShown.peek();
  });
}
