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
import { getWorldFrame } from "./game/cities";
import { packToLngLat, lngLatToPack } from "./game/coords";
import { ask, display, isBuildTool, selection, shade, stopEditing, tab, tool } from "./game/ui";
import { complexOf, dotGroups, inComplexView } from "./game/complexView";
import { cancelLink, linkPicked, linking } from "./game/links";
import { pickTool, TOOL_KEYS, TRIP_KEY } from "./ui/tabs";
import { followCapsLock, keyIsTheirs, releaseSelectsAfterMousePick } from "./ui/keys";
import { redo, undo, world } from "./game/world";
import { lastSaved, moneyState } from "./game/money";
import { live } from "./map/live";
import { createMap, setBasemapLabels, setBasemapShade } from "./map/mapView";
import { startCityBoundaries } from "./map/cityBoundaries";
import { startZoneLoading } from "./map/zoneLoading";
import { startZoneFocus } from "./map/zoneFocus";
import { StationLabels } from "./map/stationLabels";
import { networkFromState, tripAt } from "./map/network";
import { Overlay } from "./map/overlay";
import { pick } from "./map/pick";
import { NetworkRenderer } from "./map/renderer";
import { followTrain } from "./map/trainFollow";
import { makeSynth } from "./map/synth";
import { Tools } from "./map/tools";
import { StopMarkers } from "./map/stopMarkers";
import { EdgeEditor } from "./map/edgeEdit";
import { BlueprintBox } from "./map/blueprintBox";
import { bpPicked, clearBpPicked } from "./game/blueprintPick";
import { NodeDrag } from "./map/nodeDrag";
import { EndMarkers } from "./map/endMarkers";
import { hoverAbove } from "./map/demandHover";
import { CapacityMarkers } from "./map/capacityMarkers";
import { startTrackHover } from "./map/trackHover";
import { perfOff, perfTry } from "./perfFlags";
import { DEMAND_INDEX, type Selection } from "./game/types";
import { anchor, kAtY, toLngLat, toLocal } from "./map/geo";
import { Debug, perfText } from "./ui/Debug";
import { Dock } from "./ui/Dock";
import { Toolbar } from "./ui/Toolbar";
import { TopBar } from "./ui/TopBar";
import { MapTopLeft } from "./ui/DemandSelector";
import { BlueprintBar } from "./ui/Blueprint";
import { startSearch } from "./ui/Search";
import { startTips } from "./ui/tip";
import { startLoadingGate } from "./ui/LoadingScreen";
import { ClockClient, clockWorkerStatus, onNetworkSnapshot } from "./workers/clockClient";
import { startDemand, submitNetwork } from "./workers/demandClient";
import { startDemandViews, viewStats } from "./map/demandViews";
import { startTripView } from "./map/tripView";
import { MODE_COLOURS } from "./game/palette";

// The mode colours are the network's, not the theme's (T-084): the top bar's split uses them.
for (const [k, v] of Object.entries(MODE_COLOURS)) document.documentElement.style.setProperty(`--mode-${k}`, v);

/** Other lines' strength while a line's stops are edited (T-233). */
const STOP_EDIT_DIM = 0.3;

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => {
  const v = Number(params.get(k));
  return Number.isFinite(v) && v > 0 ? v : d;
};
const DEBUG = params.get("debug") === "1";
const SYNTH = params.get("synth") === "1";
const HOVER_CURSOR = perfTry("hoverCursor");

// T-210: the game's own hover tag for every panel and control, never the browser's
startTips();

// ---- the clock: a new game starts on day 1 at 07:00 (morning peak) ----
startClock(86400 + 7 * 3600);
if (params.get("paused") === "1") clock.setRunning(false);
// T-149, T-172: the clock is held until the viewed city's demand has its first estimate
startLoadingGate();
// Demand pool (T-026) first: its downloads and pack opening are the longest part of startup, so
// they start before the map is set up (T-172). The clock worker's network snapshots go to
// `submitNetwork` below (workers/demandClient.ts); `?demandTest=1` solves T-005's hand-made
// network instead.
startDemand();

// ---- map and overlay ----
const map = createMap(document.getElementById("map")!);
startCityBoundaries(map);
startZoneLoading(map); // T-343: zones whose demand has not arrived, grey with a loading bar
const demandSelectorRoot = document.createElement("div");
map.getContainer().appendChild(demandSelectorRoot);
render(h(MapTopLeft, null), demandSelectorRoot); // the demand buttons and the stations full alert (T-231)
const blueprintBarRoot = document.createElement("div");
map.getContainer().appendChild(blueprintBarRoot);
render(h(BlueprintBar, null), blueprintBarRoot); // T-353: the Select tool's Disable and Enable at the top centre
startSearch(map.getContainer()); // T-148: the search box at the map's upper right
const renderer = new NetworkRenderer();
/** A click on the commuter bubbles (T-097; set below once the demand views start). */
let demandClick: ((x: number, y: number, add: boolean) => Selection | undefined) | null = null;
let demandHover: ((x: number, y: number) => string[] | null) | null = null;
const overlay = new Overlay(map, renderer, () => {
  let sel = selection.peek();
  // a selected train that has turned at its terminal is followed onto its next trip (T-245)
  const followed = followTrain(renderer, sel, clock.now());
  if (followed !== sel) selection.value = sel = followed;
  const d = display.peek();
  const n = renderer.net;
  return {
    showTrains: d.trains,
    byLevel: d.trackColour === "height",
    bySpeed: d.trackColour === "speed",
    selLine: sel?.kind === "line" || sel?.kind === "train" ? Number(sel.line) : -1,
    selTrain: sel?.kind === "train" ? sel : null, // T-242: drawn over the others with a halo
    dimOthers: stopEditing.peek() ? STOP_EDIT_DIM : 1, // T-233: the other lines recede while stops are edited
    selEdge: sel?.kind === "track" ? sel.edge : -1,
    selStation: sel?.kind === "station" && n ? n.stationIds.indexOf(sel.station) : -1,
    // complex view: every member's box drawn selected (T-163)
    selStations: [
      ...(sel?.kind === "station" && n && inComplexView(sel.station) ? complexOf(sel.station).map((id) => n.stationIds.indexOf(id)) : []),
      // the blueprint stations picked with the blueprint tool's box (T-333)
      ...(n ? bpPicked.value.stations.map((id) => n.stationIds.indexOf(String(id))) : []),
    ],
    crowds: d.platformCrowds !== "off", // T-164, T-219
    groups: dotGroups.peek(), // zoomed out, the stations the distance rule groups are one dot (T-217, T-225)
  };
});
live.map = map;
live.overlay = overlay;
// Panning follows the zone under the view, except while a bubble, station or trip selection holds
// the focus; the zones beside it on screen show their bubbles too (T-276, map/zoneFocus.ts).
startZoneFocus(map);

// ---- the clock worker: owns the network; its states and trips go to the renderer ----
const client = new ClockClient();
if (SYNTH) {
  const s = makeSynth(Math.round(num("lines", 500)), Math.round(num("trains", 10000)));
  renderer.setNetwork(s);
  renderer.onEpoch = (e) => renderer.setTrips(e, 0, s.trips);
} else {
  // Demand views (T-084): network layers, the commuter dot map, a station's riders.
  const views = startDemandViews(overlay, map);
  demandClick = views.click;
  demandHover = views.hover;
  client.onState = (s) => {
    // a train count: the same buffers with the new run profiles (T-229)
    if (s.service && renderer.net) renderer.setService(s.render.phases, s.render.meta);
    else renderer.setNetwork(s.net ?? networkFromState(s)); // built in the worker (T-063)
    views.networkChanged();
    blueprintBox.refreshHighlight();
    overlay.invalidate();
  };
  if (DEBUG) (window as any).twViews = { stats: viewStats, overlay: views.overlay, hover: views.hover, click: views.click };
  client.onTrips = (epoch, version, trips, board, ends, nexts) => {
    renderer.setTrips(epoch, version, trips, board, ends, nexts);
    overlay.invalidate();
  };
  renderer.onEpoch = (e) => client.setEpoch(e);
  // The draw anchor follows the view (T-307, map/geo.ts): asked for when a move ends far from it;
  // once the network drawn at the new one is in, a MapLibre frame moves everything placed on it
  // (names, tags, the trip view) with the same camera.
  const bounds = () => { const b = map.getBounds(); client.followBounds(b.getWest(), b.getSouth(), b.getEast(), b.getNorth()); };
  map.on("moveend", () => {
    const c = map.getCenter();
    client.followView(c.lng, c.lat, map.getZoom());
    bounds();
  });
  bounds();
  client.onAnchor = () => {
    map.triggerRepaint();
    overlay.invalidate();
  };
}
const tools = new Tools(map, overlay, client);
// The blueprint tool: a drag box picks blueprint track and stations to switch off or on (T-333).
const blueprintBox = new BlueprintBox(map, overlay, client);
live.tools = tools;
// Stop editing (T-233): a toggle marker on every station; the page class hides the demand views
tools.stops = new StopMarkers(map, overlay, (station) => tools.toggleStop(station));
// The trip explorer (T-195): its clicks, its query and its drawing over the demand views.
const tripTool = startTripView(map, overlay);
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
// Markers on loose ends (track and station tools) and on draggable blueprint points (T-416).
const endMarks = new EndMarkers(map, overlay);
// T-486: clicks reach only what is drawn on top. The points' squares are drawn over the demand
// bubbles, so there neither the bubbles' clicks nor their hover tags answer, and the trains drawn
// under the squares are passed over too.
const onMark = (x: number, y: number) => endMarks.at(x, y);
const prevAbove = hoverAbove.at;
// T-526: with a build tool on, the demand views are drawn beneath the network, so the network and
// the tool's own tag come before a bubble's tag
const buildAbove = (x: number, y: number) => isBuildTool(tool.peek()) && (tools.tagShown || !!pick(overlay, x, y, display.peek().trains));
hoverAbove.at = (x, y) => onMark(x, y) || buildAbove(x, y) || !!prevAbove?.(x, y);
effect(() => {
  document.documentElement.classList.toggle("building", isBuildTool(tool.value));
});
const demandOnTop = (x: number, y: number, add: boolean) => (onMark(x, y) ? undefined : demandClick?.(x, y, add));

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
const capacity = new CapacityMarkers(map.getContainer(), overlay, map, labels); // tags keep off names (T-184)
capacity.demandClick = demandOnTop;
capacity.tripClick = (x, y) => tripTool.clickAt(x, y); // T-221
effect(() => {
  const w = world.value;
  const level = DEMAND_INDEX[clock.demand.value];
  if (w && !SYNTH) capacity.set(w, level);
});
effect(() => capacity.setVisible(display.value.capacity && !SYNTH));
// Resting the pointer on track says what the track display shows there (T-157).
startTrackHover(map, overlay);
map.on("load", () => {
  effect(() => setBasemapLabels(map, display.value.basemapLabels));
  effect(() => setBasemapShade(map, shade.value));
});
// T-039: the network's casing and inks come from game/shades.ts at each draw
effect(() => {
  shade.value;
  overlay.invalidate();
  map.triggerRepaint();
});

// While a tool is on, double clicks finish a route instead of zooming.
effect(() => {
  // a shift-click places a refused track point (T-410), which MapLibre's shift-drag zoom would eat
  if (tool.value === "select") map.doubleClickZoom.enable(), map.boxZoom.enable();
  else map.doubleClickZoom.disable(), map.boxZoom.disable();
  map.getCanvas().style.cursor = tool.value === "select" ? "" : "crosshair";
  if (tool.value !== "track") tools.reset();
});

// ---- picking and tools ----
map.on("click", (e) => {
  if (tool.peek() === "trip") tripTool.click(e); // T-195
  else if (tool.peek() === "blueprint") return; // its pointer handlers pick (map/blueprintBox.ts)
  else if (tool.peek() !== "select") tools.click(e);
  else selectAt(e.point.x, e.point.y, e.originalEvent.shiftKey);
});
/** The select tool's click at a map point (px): from the map, or from a capacity tag over it (T-256). */
function selectAt(x: number, y: number, add: boolean) {
  // linking a station into a complex (T-197): the station clicked is the other end
  if (linking.peek() !== null) void linkPicked(pick(overlay, x, y, false, selection.peek()));
  // commuter bubbles first while they are shown (T-097), then the network
  else selection.value = demandOnTop(x, y, add) ?? pick(overlay, x, y, display.peek().trains && !onMark(x, y), selection.peek());
}
capacity.selectClick = selectAt;
// Linking stops when its station is no longer the selection or a tool is picked.
effect(() => {
  const s = selection.value, from = linking.value;
  if (from !== null && (tool.value !== "select" || s?.kind !== "station" || s.station !== from)) linking.value = null;
});
if (!SYNTH && !perfOff("hover")) // ?perfOff=hover: T-045 measurement only
  map.on("mousemove", (e) => {
    if (e.originalEvent.buttons) return; // dragging the map: no hover
    if (tool.peek() === "trip") {
      // a station under the pointer is picked as the trip's end (T-221)
      const c = tripTool.stationAt(e.point.x, e.point.y) ? "pointer" : "crosshair";
      if (map.getCanvas().style.cursor !== c) map.getCanvas().style.cursor = c;
      return;
    }
    if (tool.peek() === "blueprint") return;
    if (tool.peek() !== "select") return tools.move(e);
    // a blueprint point drags even over a demand bubble (T-474), so its grab cursor comes first
    const cur = linking.peek() !== null ? (pick(overlay, e.point.x, e.point.y, false)?.kind === "station" ? "pointer" : "crosshair") : nodeDrag.hit(e.point.x, e.point.y)?.movable ? "grab" : demandHover?.(e.point.x, e.point.y) ? "pointer" : pick(overlay, e.point.x, e.point.y, display.peek().trains) ? "pointer" : "";
    // ?perfTry=hoverCursor (T-045): write the cursor only when it changes
    if (!HOVER_CURSOR || map.getCanvas().style.cursor !== cur) map.getCanvas().style.cursor = cur;
  });
// The focused control's own keys stay with it: all of them in a text field, the arrows, Enter and
// Space in a select (ui/keys.ts, T-183, the same rule as the search box's "/").
releaseSelectsAfterMousePick();
followCapsLock();
window.addEventListener("keydown", (e) => {
  if (keyIsTheirs(e)) return;
  // Esc, one thing at a time (T-096): a question waiting above the bottom bar is answered no; a
  // node or corner being dragged goes back (T-093, T-064); a half-drawn route is dropped, and the
  // next Esc leaves the tool (tools.key); with no tool on, the selection clears (below).
  if (e.key === "Escape" && ask.peek()) {
    ask.value = null;
    e.preventDefault();
    return;
  }
  if (e.key === "Escape" && (nodeDrag.cancel() || edgeEditor.cancel() || cancelLink() || blueprintBox.cancel())) {
    e.preventDefault();
    return;
  }
  // 1, 2, 3: draw track, place stations, delete, from any tab (T-096); the Build tab opens. The
  // key of the tool already on does nothing (Esc is the way out), so a half-drawn route survives.
  // T: the trip explorer (T-195, T-444), from any tab; the tab stays (the inspector shows the trip)
  if (e.key.toLowerCase() === TRIP_KEY &&!e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat) {
    e.preventDefault();
    nodeDrag.cancel();
    edgeEditor.cancel();
    if (tool.peek() !== "trip") pickTool("trip");
    return;
  }
  const k = TOOL_KEYS[e.key];
  if (k && !e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat) {
    e.preventDefault();
    nodeDrag.cancel();
    edgeEditor.cancel();
    if (tool.peek() !== k) pickTool(k);
    tab.value = "build";
    return;
  }
  // the blueprint tool: Esc clears the pick, then leaves the tool; D and E switch the pick off or on
  if (tool.peek() === "blueprint" && e.key === "Escape" && (bpPicked.peek().edges.length || bpPicked.peek().stations.length)) {
    e.preventDefault();
    clearBpPicked();
    return;
  }
  if (blueprintBox.key(e)) {
    e.preventDefault();
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
  if (e.key === " " && !keyIsTheirs(e)) e.preventDefault();
});

// ---- UI ----
render(h(Dock, null), document.getElementById("dock-root")!);
render(h(TopBar, null), document.getElementById("topbar-root")!);
render(h(Toolbar, null), document.getElementById("tools-root")!);

// ---- demand (the pool started at the top) ----
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
      r.setTrips(e, src.version, src.data, undefined, src.ends);
      const gpu = r.probe(t);
      for (let i = 0; i < r.tripCount; i++) {
        const at = tripAt(n, src.data[i * 2], src.data[i * 2 + 1] + e, t, src.ends ? src.ends[i] + e : -Infinity);
        if (!at) continue;
        errs.push(Math.hypot(gpu[i * 2] - at.x, gpu[i * 2 + 1] - at.y) / kAtY(at.y, n.anchor));
      }
    }
    errs.sort((a, b) => a - b);
    return { n: errs.length, p50M: errs[errs.length >> 1], maxM: errs[errs.length - 1] };
  };

  (window as any).tw = {
    map, overlay, renderer, clock, world, selection, display, shade, tools, client, checkPrecision, moneyState, lastSaved, pick, tripAt,
    // the current draw anchor (T-307), mercator 0..1, and conversions with it
    get originMerc() { return anchor().merc; },
    anchor, toLocal, toLngLat,
    frame: () => { const f = getWorldFrame(); return { kind: f.kind, lon: f.lon, lat: f.lat }; }, packToLngLat, lngLatToPack,
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
