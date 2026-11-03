// The network diagram's view (T-2160, T-2161, T-2162): the button and key M swap the map for a 2D
// canvas of the laid-out diagram (ribbons, rounded bends, pills, names), drag to pan, wheel to zoom.
// While it is up the clock worker posts the graph after each network state; the page names its
// stations and measures the names, and the layout worker lays it out, draws it and places the names,
// one at a time, the newest waiting next, the last drawing staying up meanwhile. `?diagram=r2` lays
// out with round 2's options.

import { signal } from "@preact/signals";
import type { ClockClient } from "../workers/clockClient";
import type { DiagramGraph } from "./graph";
import type { DiagramNames, FromDiagram, ToDiagram } from "../workers/diagram.worker";
import { paint, type Drawing } from "./draw";
import { paintNames, pickName, splitName, type NamePlan } from "./names";
import { keyIsTheirs } from "../ui/keys";
import { live } from "../map/live";
import { toLocal } from "../map/geo";
import { hideBusy, showBusy } from "../game/busy";
import { world } from "../game/world";
import { complexes } from "../game/complexView";

/** station names are this many CSS px, as the map's, times the interface scale */
const NAME_PX = 12;

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

/** the names' font: the interface's family at NAME_PX times its scale */
function nameFont(): { px: number; family: string } {
  const ui = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ui")) || 1;
  return { px: NAME_PX * ui, family: getComputedStyle(document.body).fontFamily };
}

let measure: CanvasRenderingContext2D | null = null;
const widths = new Map<string, number>();
/** each graph node's name (its complex's, `pickName`) and widths in the names' font */
function namesOf(g: DiagramGraph): DiagramNames {
  const w = world.peek(), cx = complexes.peek(), f = nameFont();
  const byId = new Map((w?.save.stations ?? []).map((s) => [s.id, s]));
  const stops = new Map<string, number>();
  for (const l of w?.save.lines ?? []) for (const s of l.stops) stops.set(s, (stops.get(s) ?? 0) + 1);
  const text = Array.from(g.key, (key) => {
    const id = String(key), c = cx.of.get(id), members = c === undefined ? [id] : cx.members[c];
    return pickName(members.flatMap((m) => {
      const s = byId.get(m);
      return s ? [{ name: s.name, lines: stops.get(m) ?? 0 }] : [];
    }));
  });
  measure ??= document.createElement("canvas").getContext("2d");
  const font = `${f.px}px ${f.family}`;
  const wOf = (t: string) => {
    const k = font + "\u0000" + t;
    let v = widths.get(k);
    if (v === undefined) {
      measure!.font = font;
      widths.set(k, (v = measure!.measureText(t).width));
    }
    return v;
  };
  const width = new Float32Array(text.length), width2 = new Float32Array(text.length);
  text.forEach((t, i) => {
    width[i] = t ? wOf(t) : 0;
    const p = splitName(t);
    width2[i] = p ? Math.max(wOf(p[0]), wOf(p[1])) : 0;
  });
  return { text, width, width2, fontPx: f.px };
}

export function startDiagramView(client: ClockClient, mapEl: HTMLElement): void {
  const options = new URLSearchParams(location.search).get("diagram") === "r2" ? "r2" : undefined;
  const canvas = document.createElement("canvas");
  canvas.className = "diagram-canvas";
  Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", zIndex: "3", display: "none", cursor: "grab", touchAction: "none" });
  mapEl.appendChild(canvas);
  let worker: Worker | null = null;
  let busy = false, waiting: DiagramGraph | null = null, seq = 0;
  let shown: { graph: DiagramGraph; d: Drawing; names: NamePlan; x: Float64Array; y: Float64Array } | null = null;
  let cx = 0, cy = 0, scale = 0, placed = false;
  const send = (g: DiagramGraph) => {
    if (busy) {
      waiting = g;
      return;
    }
    busy = true;
    worker!.postMessage({ kind: "graph", graph: g, names: namesOf(g), seq: ++seq, options } satisfies ToDiagram);
  };
  client.onDiagram = (g) => {
    if (!diagramShown.peek()) return;
    if (!worker) {
      worker = new Worker(new URL("../workers/diagram.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (e: MessageEvent<FromDiagram>) => {
        busy = false;
        const { graph, layout, drawing: d, names } = e.data;
        shown = { graph, d, names, x: layout.x, y: layout.y };
        // for checks and `?debug=1`: the last layout's figures and how to frame a station
        (window as unknown as { twDiagram: unknown }).twDiagram = { layouts: e.data.seq, ms: { ...layout.ms, ...e.data.ms }, complexes: graph.key.length, pieces: layout.pieces.length, names: names.at.map((l, i) => [names.levels[i], l.node.length]), centreOn: (name: string, pitchPx: number) => { const stops = new Map<number, number>(); for (const l of graph.lines) l.chain.forEach((k, j) => l.stop[j] && names.text[k] === name && stops.set(k, (stops.get(k) ?? 0) + 1)); const i = [...stops].sort((a, b) => b[1] - a[1])[0]?.[0] ?? -1; if (i < 0) return false; (cx = layout.x[i]), (cy = layout.y[i]), (scale = pitchPx / shown!.d.pitch), draw(); return true; }, centre: (lng: number, lat: number, pitchPx: number) => { const i = nearest(graph, lng, lat); (cx = layout.x[i]), (cy = layout.y[i]), (scale = pitchPx / shown!.d.pitch), draw(); }, bench: (k: number) => { const t = performance.now(); for (let j = 0; j < k; j++) paintNow(); return (performance.now() - t) / k; } };
        if (busyToken >= 0) (hideBusy(busyToken), (busyToken = -1));
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
    // the panels' white pills with the ink outline, in both themes, as the line diagrams draw them
    return { bg: v("--bg", "#fbf5ef"), ink: v("--net-ink", "#2b2b2b"), pill: "#ffffff", text: v("--map-ink", "#2b2b2b"), font: s.fontFamily };
  };
  // pans and zooms draw once a frame
  let frame = 0;
  const draw = () => {
    if (!frame) frame = requestAnimationFrame(paintNow);
  };
  const paintNow = () => {
    frame = 0;
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
    // the map's station names: its ink with a halo of the background
    paintNames(ctx, shown.names, shown.d, canvas.width, canvas.height, cx, cy, scale * dpr, dpr, { ink: t.text, halo: t.bg, font: t.font });
  };
  new ResizeObserver(() => draw()).observe(canvas);
  // the first layout is a few seconds; the busy tag says so until it is up
  let busyToken = -1;
  diagramShown.subscribe((on) => {
    canvas.style.display = on ? "block" : "none";
    client.post({ kind: "diagram", on });
    if (on && !shown && busyToken < 0) busyToken = showBusy("Calculating diagram");
    if (!on && busyToken >= 0) (hideBusy(busyToken), (busyToken = -1));
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
