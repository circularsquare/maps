// Entry point: wires game state, the map and its overlay, the UI and the workers together.
//
//   game/     state the UI reads: the world snapshot (save + worker results), UI state, the clock
//   map/      MapLibre basemap, the overlay canvas and its WebGL renderer, picking, the tools
//   ui/       Preact components (dock, top bar, toolbar, inspector, settings)
//   workers/  the clock worker (owns the network, T-025) and the demand pool (T-026)
//
// URL parameters: ?debug=1 (performance readout, window.tw probes), ?paused=1, ?new=1 (a new
// game instead of the autosave, workers/clockClient.ts),
// ?synth=1&lines=500&trains=10000 (draw the synthetic stress network instead of the real one),
// ?perfOff=labels,shadow,hover,clock,overlay,markers (measurement only, perfFlags.ts).

import "./ui/style.css";
import { effect } from "@preact/signals";
import { h, render } from "preact";
import { clock, SPEEDS, startClock } from "./game/clock";
import { setPackOrigin } from "./game/coords";
import { ask, display, selection, tab, tool } from "./game/ui";
import { pickTool, TOOL_KEYS } from "./ui/tabs";
import { redo, undo, world } from "./game/world";
import { income, lastSaved, moneyState } from "./game/money";
import { live } from "./map/live";
import { createMap, setBasemapLabels } from "./map/mapView";
import { StationLabels } from "./map/stationLabels";
import { networkFromState, tripAt } from "./map/network";
import { Overlay } from "./map/overlay";
import { pick } from "./map/pick";
import { NetworkRenderer } from "./map/renderer";
import { makeSynth } from "./map/synth";
import { Tools } from "./map/tools";
import { EdgeEditor } from "./map/edgeEdit";
import { NodeDrag } from "./map/nodeDrag";
import { CapacityMarkers } from "./map/capacityMarkers";
import { perfOff, perfTry } from "./perfFlags";
import { DEMAND_INDEX, type Selection } from "./game/types";
import { MERC_K, ORIGIN_MERC } from "./map/geo";
import { Debug, perfText } from "./ui/Debug";
import { Dock } from "./ui/Dock";
import { Toolbar } from "./ui/Toolbar";
import { TopBar } from "./ui/TopBar";
import { ClockClient, clockWorkerStatus, onNetworkSnapshot, packOrigin } from "./workers/clockClient";
import { startDemand, submitNetwork } from "./workers/demandClient";
import { startDemandViews, viewStats } from "./map/demandViews";
import { MODE_COLOURS } from "./game/palette";

// The mode colours are the network's, not the theme's (T-084): the top bar's split uses them.
for (const [k, v] of Object.entries(MODE_COLOURS)) document.documentElement.style.setProperty(`--mode-${k}`, v);

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => {
  const v = Number(params.get(k));
  return Number.isFinite(v) && v > 0 ? v : d;
};
const DEBUG = params.get("debug") === "1";
const SYNTH = params.get("synth") === "1";
const HOVER_CURSOR = perfTry("hoverCursor");

// ---- the clock: a new game starts on day 1 at 07:00 (morning peak) ----
startClock(86400 + 7 * 3600);
if (params.get("paused") === "1") clock.setRunning(false);

// ---- map and overlay ----
const map = createMap(document.getElementById("map")!);
const renderer = new NetworkRenderer();
/** A click on the commuter bubbles (T-097; set below once the demand views start). */
let demandClick: ((x: number, y: number, add: boolean) => Selection | undefined) | null = null;
const overlay = new Overlay(map, renderer, () => {
  const sel = selection.peek();
  const d = display.peek();
  const n = renderer.net;
  return {
    showTrains: d.trains,
    byLevel: d.trackColour === "height",
    bySpeed: d.trackColour === "speed",
    selLine: sel?.kind === "line" || sel?.kind === "train" ? Number(sel.line) : -1,
    selEdge: sel?.kind === "track" ? sel.edge : -1,
    selStation: sel?.kind === "station" && n ? n.stationIds.indexOf(sel.station) : -1,
  };
});
live.map = map;
live.overlay = overlay;

// ---- the clock worker: owns the network; its states and trips go to the renderer ----
const client = new ClockClient();
effect(() => {
  const o = packOrigin.value;
  if (o) setPackOrigin(o);
});
if (SYNTH) {
  const s = makeSynth(Math.round(num("lines", 500)), Math.round(num("trains", 10000)));
  renderer.setNetwork(s);
  renderer.onEpoch = (e) => renderer.setTrips(e, 0, s.trips);
} else {
  // Demand views (T-084): network layers, the commuter dot map, a station's riders.
  const views = startDemandViews(overlay, map);
  demandClick = views.click;
  client.onState = (s) => {
    renderer.setNetwork(s.net ?? networkFromState(s)); // built in the worker (T-063)
    views.networkChanged();
    overlay.invalidate();
  };
  if (DEBUG) (window as any).twViews = { stats: viewStats, overlay: views.overlay, hover: views.hover, click: views.click };
  client.onTrips = (epoch, version, trips) => {
    renderer.setTrips(epoch, version, trips);
    overlay.invalidate();
  };
  renderer.onEpoch = (e) => client.setEpoch(e);
}
const tools = new Tools(map, overlay, client);
live.tools = tools;
// Reshaping a selected stretch of blueprint track (T-064).
const edgeEditor = new EdgeEditor(map, overlay, client);
live.edgeEditor = edgeEditor;
effect(() => {
  selection.value;
  tool.value;
  world.value;
  void edgeEditor.refresh();
});
// Dragging blueprint nodes: track ends, junctions, stations (T-093).
const nodeDrag = new NodeDrag(map, overlay, client);
live.nodeDrag = nodeDrag;

// Selection and display settings change what the overlay draws.
effect(() => {
  selection.value;
  display.value;
  overlay.invalidate();
});

const labels = new StationLabels(map.getContainer(), overlay, map);
effect(() => {
  const w = world.value;
  if (w && !SYNTH) labels.set(w);
});
effect(() => labels.setVisible(display.value.stationNames && !SYNTH));
const capacity = new CapacityMarkers(map.getContainer(), overlay, map);
effect(() => {
  const w = world.value;
  const level = DEMAND_INDEX[clock.demand.value];
  if (w && !SYNTH) capacity.set(w, level);
});
effect(() => capacity.setVisible(display.value.capacity && !SYNTH));
map.on("load", () => {
  effect(() => setBasemapLabels(map, display.value.basemapLabels));
});

// While a tool is on, double clicks finish a route instead of zooming.
effect(() => {
  if (tool.value === "select") map.doubleClickZoom.enable();
  else map.doubleClickZoom.disable();
  map.getCanvas().style.cursor = tool.value === "select" ? "" : "crosshair";
  if (tool.value !== "track") tools.reset();
});

// ---- picking and tools ----
map.on("click", (e) => {
  if (tool.peek() !== "select") tools.click(e);
  // commuter bubbles first while they are shown (T-097), then the network
  else selection.value = demandClick?.(e.point.x, e.point.y, e.originalEvent.shiftKey) ?? pick(overlay, e.point.x, e.point.y, display.peek().trains, selection.peek());
});
if (!SYNTH && !perfOff("hover")) // ?perfOff=hover: T-045 measurement only
  map.on("mousemove", (e) => {
    if (e.originalEvent.buttons) return; // dragging the map: no hover
    if (tool.peek() !== "select") return tools.move(e);
    const cur = nodeDrag.hit(e.point.x, e.point.y)?.movable ? "move" : pick(overlay, e.point.x, e.point.y, display.peek().trains) ? "pointer" : "";
    // ?perfTry=hoverCursor (T-045): write the cursor only when it changes
    if (!HOVER_CURSOR || map.getCanvas().style.cursor !== cur) map.getCanvas().style.cursor = cur;
  });
const typing = (t: EventTarget | null) =>
  t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || (t instanceof HTMLElement && t.isContentEditable);
window.addEventListener("keydown", (e) => {
  if (typing(e.target)) return;
  // Esc, one thing at a time (T-096): a question waiting above the bottom bar is answered no; a
  // node or corner being dragged goes back (T-093, T-064); a half-drawn route is dropped, and the
  // next Esc leaves the tool (tools.key); with no tool on, the selection clears (below).
  if (e.key === "Escape" && ask.peek()) {
    ask.value = null;
    e.preventDefault();
    return;
  }
  if (e.key === "Escape" && (nodeDrag.cancel() || edgeEditor.cancel())) {
    e.preventDefault();
    return;
  }
  // 1, 2, 3: draw track, place stations, delete, from any tab (T-096); the Build tab opens. The
  // key of the tool already on does nothing (Esc is the way out), so a half-drawn route survives.
  const k = TOOL_KEYS[e.key];
  if (k && !e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat) {
    e.preventDefault();
    nodeDrag.cancel();
    edgeEditor.cancel();
    if (tool.peek() !== k) pickTool(k);
    tab.value = "build";
    return;
  }
  if (tools.key(e)) {
    e.preventDefault();
    return;
  }
  // Undo and redo (blueprint edits only; constructing is final, SPEC 6.4).
  if ((e.ctrlKey || e.metaKey) && !e.altKey) {
    const k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) {
      e.preventDefault();
      void undo();
      return;
    }
    if (k === "y" || (k === "z" && e.shiftKey)) {
      e.preventDefault();
      void redo();
      return;
    }
  }
  if (e.key === "Escape") selection.value = null;
  // Space: pause, or carry on at the speed that was running (the clock keeps it while paused).
  if (e.key === " " && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    clock.setRunning(!clock.running.peek());
  }
});
// A focused button would also "click" on the space bar's keyup; the key is ours.
window.addEventListener("keyup", (e) => {
  if (e.key === " " && !typing(e.target)) e.preventDefault();
});

// ---- UI ----
render(h(Dock, null), document.getElementById("dock-root")!);
render(h(TopBar, null), document.getElementById("topbar-root")!);
render(h(Toolbar, null), document.getElementById("tools-root")!);

// ---- demand ----
// Demand pool (T-026): the clock worker's network snapshots go to `submitNetwork`
// (workers/demandClient.ts); `?demandTest=1` solves T-005's hand-made network instead.
startDemand();
onNetworkSnapshot(submitNetwork);

// ---- ?debug=1: readout and probes ----
if (DEBUG) {
  render(h(Debug, null), document.getElementById("debug-root")!);
  let mapMs = 0, mapN = 0, lastMapFrames = 0, lastAt = performance.now();
  // MapLibre has no public frame-start hook; this wraps the internal `_render`, and the readout
  // says "idle" if a release renames it.
  const internal = map as unknown as { _render?: (ts: number) => unknown };
  if (typeof internal._render === "function") {
    const orig = internal._render.bind(map);
    internal._render = (ts: number) => {
      const a = performance.now();
      const r = orig(ts);
      mapMs += performance.now() - a;
      mapN++;
      return r;
    };
  }
  setInterval(() => {
    const now = performance.now();
    const gap = overlay.frameGap.avg();
    const mapPerS = ((overlay.mapFrames - lastMapFrames) * 1000) / (now - lastAt);
    lastMapFrames = overlay.mapFrames;
    lastAt = now;
    perfText.value = [
      `${renderer.tripCount.toLocaleString()} trips this hour${SYNTH ? " (synthetic)" : ""}`,
      `Draw   ${overlay.drawMs.avg().toFixed(3)} ms avg, ${overlay.drawMs.max().toFixed(2)} max`,
      `Map    ${mapN ? (mapMs / mapN).toFixed(2) + " ms, " + mapPerS.toFixed(0) + " frames/s" : "idle"}`,
      `GPU    ${renderer.gpuMs === null ? "no timer" : renderer.gpuMs.toFixed(2) + " ms"}`,
      `Frame  ${gap ? gap.toFixed(1) + " ms, worst " + overlay.frameGap.max().toFixed(0) + " ms" : "idle"}`,
      `Clock  ${clock.ticking ? SPEEDS[clock.speed.peek()] + "x" : "stopped"}, epoch ${renderer.epoch}, ${renderer.rebases} re-bases`,
      `Worker ${clockWorkerStatus.value}`,
    ].join("\n");
    mapMs = mapN = 0;
  }, 250);

  // T-015 lag probe: after each frame, the offset between the basemap's camera and the overlay's.
  const lag = { on: false, samples: [] as number[] };
  const lagFrame = () => {
    if (!lag.on) return;
    const ch = new MessageChannel();
    ch.port1.onmessage = () => {
      const a = overlay.camMatrix, b = overlay.lastDrawnMatrix;
      if (a && b) {
        const c = map.getCenter();
        const px = (m: ArrayLike<number>) => {
          const x = (c.lng + 180) / 360, s = Math.sin((c.lat * Math.PI) / 180);
          const y = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
          const w = m[3] * x + m[7] * y + m[15];
          return [(m[0] * x + m[4] * y + m[12]) / w, (m[1] * x + m[5] * y + m[13]) / w];
        };
        const [ax, ay] = px(a), [bx, by] = px(b);
        lag.samples.push(Math.hypot(((ax - bx) / 2) * overlay.canvas.clientWidth, ((ay - by) / 2) * overlay.canvas.clientHeight));
      }
    };
    ch.port2.postMessage(0);
    requestAnimationFrame(lagFrame);
  };

  // GPU train positions against the CPU's float64 evaluation of the same keyframes (T-016, T-025),
  // on a context of its own.
  const checkPrecision = (opts: { t: number; frames?: number; dt?: number }) => {
    const n = renderer.net!;
    const gl = document.createElement("canvas").getContext("webgl2")!;
    const r = new NetworkRenderer();
    r.init(gl);
    r.setNetwork(n);
    const errs: number[] = [];
    for (let f = 0; f < (opts.frames ?? 60); f++) {
      const t = opts.t + f * (opts.dt ?? 1);
      const e = Math.floor(t / 3600) * 3600;
      const src = (renderer as any).tripsBy.get(e);
      if (!src) continue;
      r.setTrips(e, src.version, src.data);
      const gpu = r.probe(t);
      for (let i = 0; i < r.tripCount; i++) {
        const at = tripAt(n, src.data[i * 2], src.data[i * 2 + 1] + e, t);
        if (!at) continue;
        errs.push(Math.hypot(gpu[i * 2] - at.x, gpu[i * 2 + 1] - at.y) / MERC_K);
      }
    }
    errs.sort((a, b) => a - b);
    return { n: errs.length, p50M: errs[errs.length >> 1], maxM: errs[errs.length - 1] };
  };

  (window as any).tw = {
    map, overlay, renderer, clock, world, selection, display, tools, client, checkPrecision, moneyState, income, lastSaved, pick, tripAt,
    originMerc: ORIGIN_MERC,
    select: (s: unknown) => (selection.value = s as any),
    lagStart: () => {
      lag.samples = [];
      if (!lag.on) {
        lag.on = true;
        requestAnimationFrame(lagFrame);
      }
    },
    lagStop: () => {
      lag.on = false;
      const off = lag.samples.filter((v) => v > 0.01);
      return { frames: lag.samples.length, framesOff: off.length, maxPx: Math.max(0, ...lag.samples) };
    },
    stats: () => ({ draws: overlay.frames, mapFrames: overlay.mapFrames, drawMs: overlay.drawMs.avg(), gap: overlay.frameGap.avg() }),
    reset: () => {
      overlay.frames = overlay.mapFrames = 0;
      overlay.drawMs.clear();
      overlay.frameGap.clear();
    },
  };
}
