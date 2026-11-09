// The network diagram's view (T-2160, T-2161, T-2162, T-2163): the button swaps the map for a
// 2D canvas of the laid-out diagram (ribbons, rounded bends, pills, names), drag to pan, wheel to zoom.
// While it is up the clock worker posts the graph after each network state; the page names its
// stations and measures the names, and the layout worker lays it out, draws it and places the names,
// one at a time, the newest waiting next, the last drawing staying up meanwhile. The layout is on the rubber sheet
// (diagram/sheet.ts, sheetLines.ts; T-2568, T-2567, T-2580), the sheet kept on `window.twSheet` for checks; `?sheet=0`
// lays out as before it, where `?diagram=r2` takes round 2's options and `?distort=0.5` that setting of the distortion
// dial (layout.ts `DISTORT`, T-2556).
//
// Clicks and hovers are the map's (T-2163): a station selects its complex, a line the line, with the
// map's tags; what is selected draws full and the rest faded. The camera goes with the selection, or
// the complex in the middle, between map and diagram; a build key goes back to the map.

import { effect, signal } from "@preact/signals";
import type { ClockClient } from "../workers/clockClient";
import { saveSlotsState } from "../workers/clockClient";
import type { DiagramGraph } from "./graph";
import type { DiagramNames, FromDiagram, ToDiagram, ZoneFrame, ZoneOutlines } from "../workers/diagram.worker";
import type { MultiPolygon } from "geojson";
import { coarseOutlines } from "../map/coarseOutlines";
import { zonesTouching } from "../game/cities";
import { sheetZoneBox } from "./sheet";
import { placesOfZone } from "../game/placeList";
import { EARTH_C, toMerc01 } from "../map/geo";
import { placeNameInk } from "../map/basemap";
import { SHADES } from "../game/shadeTable";
import { DATA_BASE } from "../game/dataBase";
import { tPlace } from "../i18n/places";
import { t } from "../i18n";
import { perfOff } from "../perfFlags";
import { FADE_ALPHA, paint, pickAt, WATER_SHARE, WIDTH_MAX, WIDTH_MIN, widened, type DiagramFocus, type Drawing } from "./draw";
import { clock } from "../game/clock";
import { mixOklab } from "../game/palette";
import { nameAt, paintNames, pickName, placedBox, splitName, type NamePlan } from "./names";
import { paintCityNames } from "./cityNames";
import { keyIsTheirs } from "../ui/keys";
import { live } from "../map/live";
import { toLocal } from "../map/geo";
import { afterPaint, hideBusy, showBusy } from "../game/busy";
import { world } from "../game/world";
import { complexes } from "../game/complexView";
import { purposeFilter, selection, shade, shownDayPhrase, theme as colourTheme, tool } from "../game/ui";
import type { LineInput, Selection } from "../game/types";
import { lineGroups } from "../game/lineGroups";
import { demandView } from "../game/demand";
import { purposeName } from "../game/purposes";
import { forecast } from "../game/format";
import { activeCity } from "../game/cities";
import { zoneAt } from "../game/zones";
import { complexTip, type TipLine } from "../map/networkTips";
import { dailyRiders } from "../map/trackHover";
import { factRow, lineBullet } from "../map/trackTagRows";
import { TAG_OFFSET_PX, tagLines } from "../map/labelLayout";
import { ACTION_KEYS, BUS_KEY, FERRY_KEY, TOOL_KEYS, TRIP_KEY } from "../ui/toolKeys";

/** station names are this many CSS px, as the map's, times the interface scale */
const NAME_PX = 12;

/** whether the diagram is up instead of the map */
export const diagramShown = signal(false);
/** while the first layout is worked out, how far it has got (0 to 1); null otherwise (T-2235) */
export const diagramProgress = signal<number | null>(null);
/** whether a drawing is up, to save as a picture */
export const diagramReady = signal(false);
/** T-2565: whether the water is worked out and drawn; experimental, so off until the player turns it on, and kept per browser */
const WATER_STORE = "anitabuilder-diagram-water";
export const diagramWater = signal((() => { try { return localStorage.getItem(WATER_STORE) === "1"; } catch { return false; } })());
/** T-2888: the lines' width as a multiple of the drawn one, with the stations' marks (draw.ts `widened`); kept per browser */
const WIDTH_STORE = "anitabuilder-diagram-width";
export const diagramWidth = signal((() => {
  try {
    const v = Number(localStorage.getItem(WIDTH_STORE));
    return Number.isFinite(v) && v >= WIDTH_MIN && v <= WIDTH_MAX ? v : WIDTH_MIN;
  } catch { return WIDTH_MIN; }
})());
/** a pitch this many CSS px when the diagram first opens */
const START_PITCH_PX = 7;
/** a pointer within this many CSS px of a pill or line is on it */
const PICK_SLOP_PX = 3;
/** a press that moves less than this many CSS px is a click */
const CLICK_PX = 4;
/** the saved picture: a pitch of at most this many px, smaller where the picture would pass these */
const PICTURE_PITCH_PX = 10, PICTURE_MAX_SIDE = 16000, PICTURE_MAX_AREA = 120e6, PICTURE_MARGIN_PX = 24;

/** keys that build, so go back to the map first (TOOL_KEYS, ACTION_KEYS and the bus, ferry and trip keys) */
const BUILD_KEYS = new Set([...Object.keys(TOOL_KEYS), ...Object.keys(ACTION_KEYS), BUS_KEY, FERRY_KEY, TRIP_KEY]);

let saver: (() => void) | null = null;
/** the whole diagram as a PNG, "<game name> diagram.png" */
export function saveDiagramPicture(): void {
  saver?.();
}

/** the graph node nearest a place on the ground, of `among` if given */
function nearest(g: DiagramGraph, lng: number, lat: number, among?: Iterable<number>): number {
  const [lx, ly] = toLocal(lng, lat), mx = (lx - g.mid[0]) / g.k, my = -(ly - g.mid[1]) / g.k;
  return nearestOf(g.x, g.y, mx, my, among ?? g.key.keys());
}

/** of `among`, the node whose place in `xs`, `ys` is nearest (x, y); -1 for none */
function nearestOf(xs: ArrayLike<number>, ys: ArrayLike<number>, x: number, y: number, among: Iterable<number>): number {
  let best = -1, bd = Infinity;
  for (const i of among) {
    const d = (xs[i] - x) ** 2 + (ys[i] - y) ** 2;
    if (d < bd) (bd = d), (best = i);
  }
  return best;
}

/** a CSS colour's lightness, 0 to 1 (#rgb, #rrggbb or rgb()), 1 when unread */
function lightness(c: string): number {
  let r: number, g: number, b: number;
  const m = c.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
  if (m) [r, g, b] = [+m[1], +m[2], +m[3]];
  else {
    const h = c.replace("#", ""), f = h.length === 3 ? h.split("").map((x) => x + x).join("") : h.slice(0, 6);
    if (!/^[0-9a-f]{6}$/i.test(f)) return 1;
    [r, g, b] = [0, 2, 4].map((i) => parseInt(f.slice(i, i + 2), 16));
  }
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

/** the names' font: the interface's family at NAME_PX times its scale */
function nameFont(): { px: number; family: string } {
  const ui = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ui")) || 1;
  return { px: NAME_PX * ui, family: getComputedStyle(document.body).fontFamily };
}

let measure: CanvasRenderingContext2D | null = null;
const widths = new Map<string, number>();
/** a text's width in CSS px in the names' font */
function widthOf(t: string, f = nameFont()): number {
  measure ??= document.createElement("canvas").getContext("2d");
  const font = `${f.px}px ${f.family}`, k = font + "\u0000" + t;
  let v = widths.get(k);
  if (v === undefined) {
    measure!.font = font;
    widths.set(k, (v = measure!.measureText(t).width));
  }
  return v;
}

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
      return s ? [{ name: s.name, lines: (stops.get(m) ?? 0) + (s.named ? 1e6 : 0) }] : []; // T-2683: a typed name outranks any line count
    }));
  });
  const wOf = (t: string) => widthOf(t, f);
  const width = new Float32Array(text.length), width2 = new Float32Array(text.length);
  text.forEach((t, i) => {
    width[i] = t ? wOf(t) : 0;
    const p = splitName(t);
    width2[i] = p ? Math.max(wOf(p[0]), wOf(p[1])) : 0;
  });
  return { text, width, width2, fontPx: f.px };
}

/** the zones' borders' colour, the map's own zone outline colour (map/cityBoundaries.ts), in both themes */
const BORDER_COLOUR = "#73798b";

/** T-2168: the frame from mercator to graph metres, by the stations' own places (so it never depends on the
 * anchor the graph was built at), and the box of their lng, lat; null without stations */
function zoneFrame(g: DiagramGraph): { frame: ZoneFrame; box: [number, number, number, number] } | null {
  const byId = new Map((world.peek()?.save.stations ?? []).map((s) => [s.id, s]));
  const k = g.k / EARTH_C;
  let fx = 0, fy = 0, c = 0, w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  g.key.forEach((key, i) => {
    const st = byId.get(String(key));
    if (!st) return;
    const [mx, my] = toMerc01(st.lng, st.lat);
    (fx += mx - g.x[i] * k), (fy += my + g.y[i] * k), c++;
    (w = Math.min(w, st.lng)), (e = Math.max(e, st.lng)), (s = Math.min(s, st.lat)), (n = Math.max(n, st.lat));
  });
  return c ? { frame: [fx / c, fy / c, 1 / k], box: [w, s, e, n] } : null;
}

/** a zone's coarse outline as mercator rings, open */
function mercRings(g: MultiPolygon): Float64Array[] {
  return g.coordinates.flatMap((poly) => poly.map((ring) => {
    const m = ring.length > 1 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1] ? ring.length - 1 : ring.length;
    const out = new Float64Array(m * 2);
    for (let i = 0; i < m; i++) [out[i * 2], out[i * 2 + 1]] = toMerc01(ring[i][0], ring[i][1]);
    return out;
  }));
}

/** A layout as shown, with what clicks and the focus look up in it. */
interface Shown {
  graph: DiagramGraph;
  d: Drawing;
  names: NamePlan;
  x: Float64Array;
  y: Float64Array;
  /** a line's number to its drawn lines */
  drawnOf: Map<number, number[]>;
  /** per node, the drawn lines stopping there */
  stopping: number[][];
  /** per drawn line, the nodes it stops at */
  stops: number[][];
  /** a station id to its node, every member of a complex */
  nodeOf: Map<string, number>;
}

function indexShown(graph: DiagramGraph, d: Drawing, names: NamePlan, x: Float64Array, y: Float64Array): Shown {
  const drawnOf = new Map<number, number[]>(), stopping: number[][] = Array.from(graph.key, () => []), stops: number[][] = [];
  d.lineOf.forEach((li, i) => {
    const l = graph.lines[li];
    drawnOf.set(l.id, [...(drawnOf.get(l.id) ?? []), i]);
    const at: number[] = [];
    l.chain.forEach((k, j) => {
      if (!l.stop[j]) return;
      at.push(k);
      if (!stopping[k].includes(i)) stopping[k].push(i);
    });
    stops.push(at);
  });
  const nodeOf = new Map<string, number>(), cx = complexes.peek();
  graph.key.forEach((key, i) => {
    const id = String(key), c = cx.of.get(id);
    for (const m of c === undefined ? [id] : cx.members[c]) nodeOf.set(m, i);
  });
  return { graph, d, names, x, y, drawnOf, stopping, stops, nodeOf };
}

/** the drawn lines a selection stands for, and its station's node; null where it has none here */
function selectedOn(s: Shown, sel: Selection): { lines: number[]; node: number } | null {
  if (!sel) return null;
  if (sel.kind === "station") {
    const k = s.nodeOf.get(sel.station);
    return k === undefined ? null : { lines: s.stopping[k], node: k };
  }
  let nums: number[] = [];
  if (sel.kind === "line" || sel.kind === "train") nums = [Number(sel.line)];
  else if (sel.kind === "group") nums = (lineGroups(world.peek()?.save.lines ?? []).get(sel.group) ?? []).map((l) => l.num);
  const lines = nums.flatMap((n) => s.drawnOf.get(n) ?? []);
  return lines.length ? { lines, node: -1 } : null;
}

/** the nodes a selection's lines stop at, or its station */
function selectedNodes(s: Shown, on: { lines: number[]; node: number }): number[] {
  return on.node >= 0 ? [on.node] : [...new Set(on.lines.flatMap((i) => s.stops[i]))];
}

/** A line's tag on the diagram: the map's for a line (map/trackHover.ts `lineTip`), on the stretch
 * between the stops either side of the pointer. */
function lineTag(l: LineInput, s: Shown, drawn: number, at: number): TipLine[] {
  const name = factRow(lineBullet(l), Object.assign(document.createElement("span"), { textContent: l.name }));
  name.style.gap = "5px";
  const status = world.peek()?.lineStats[l.id]?.status;
  if (status === "planned") return [name, t("trackHover.notRunningYet")];
  if (status === "broken") return [name, t("trackHover.notRunning")];
  const dv = demandView.peek();
  if (!dv) return [name];
  // the stretch: the stops the line has made up to the last node passed, less one
  const gl = s.graph.lines[s.d.lineOf[drawn]], c = s.d.chainAt[drawn][at] ?? 0;
  let k = -1;
  for (let j = 0; j <= c && j < gl.stop.length; j++) k += gl.stop[j];
  k = Math.max(0, Math.min(l.stops.length - 2, k));
  const purpose = purposeFilter.peek();
  const day = dailyRiders(dv, l.id, l.stops, k, purpose);
  if (day === null) return [name];
  const count = t("trackHover.riders", { n: day < 0.5 ? "0" : forecast(day), when: shownDayPhrase.peek() });
  return purpose === "all" ? [name, count] : [name, count, t("trackHover.tripsOnly", { purpose: purposeName(purpose) })];
}

export function startDiagramView(client: ClockClient, mapEl: HTMLElement): void {
  const q = new URLSearchParams(location.search), options = q.get("diagram") === "r2" ? "r2" : undefined;
  const dial = q.get("distort"), distort = dial !== null && dial !== "" && Number.isFinite(+dial) ? +dial : undefined;
  const sheetOn = q.get("sheet") !== "0";
  const canvas = document.createElement("canvas");
  canvas.className = "diagram-canvas";
  Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", zIndex: "3", display: "none", cursor: "grab", touchAction: "none" });
  mapEl.appendChild(canvas);
  const tag = document.createElement("div");
  tag.className = "tool-tag demand-tag track-tag";
  tag.style.display = "none";
  mapEl.appendChild(tag);
  let worker: Worker | null = null;
  let busy = false, waiting: DiagramGraph | null = null, seq = 0;
  let shown: Shown | null = null;
  let cx = 0, cy = 0, scale = 0, placed = false;
  /** the drawn line under the pointer, drawn over the rest */
  let hot = -1;
  /** whether the view was panned, zoomed or the selection changed since it opened: only then does
   * going back move the map */
  let moved = false;
  let tagShown = false;
  const hideTag = () => {
    if (tagShown) (tag.style.display = "none"), (tagShown = false);
    if (hot >= 0) (hot = -1), draw();
  };
  let lastGraph: DiagramGraph | null = null;
  const send = (g: DiagramGraph) => {
    lastGraph = g;
    if (busy) {
      waiting = g;
      return;
    }
    busy = true;
    const zf = zoneFrame(g);
    // the zones whose water masks the water is drawn from (T-2169)
    const water = zf && diagramWater.peek() ? { packs: DATA_BASE, zones: zonesTouching(...zf.box).map((c) => c.id) } : undefined;
    // the rubber sheet (T-2568, T-2580), from the water masks of the zones its box meets (T-2591); `?sheet=0` the layout
    // before it
    const sheet = sheetOn && zf ? { sheet: { packs: DATA_BASE, zones: zonesTouching(...sheetZoneBox(zf.box)).map((c) => c.id), ...(perfOff("carryWasm") ? { carryJs: true } : {}) } } : {};
    worker!.postMessage({ kind: "graph", graph: g, names: namesOf(g), seq: ++seq, options, distort, frame: zf?.frame, zones: zf ? newZones(zf.box) : undefined, water, ...sheet } satisfies ToDiagram);
  };
  // T-2168: the zones' coarse outlines, each sent to the worker once, for the zones near the network
  let outlineMap: ReadonlyMap<string, MultiPolygon> | null = null, lastBox: [number, number, number, number] | null = null;
  const zonesSent = new Set<string>();
  const newZones = (box: [number, number, number, number]): ZoneOutlines => {
    lastBox = box;
    const out: ZoneOutlines = [];
    if (!outlineMap) return out;
    for (const c of zonesTouching(...box)) {
      const g = outlineMap.get(c.id);
      if (!g || zonesSent.has(c.id)) continue;
      zonesSent.add(c.id);
      const shown = tPlace(c.id, c.name);
      // T-2438: its shown name, for the city names far out
      out.push({ id: c.id, rings: mercRings(g), label: { name: shown, width: widthOf(shown), trips: c.localTrips ?? 0, city: c.kind !== "zone" } });
    }
    return out;
  };
  void coarseOutlines().then((m) => {
    if (!m.size) return;
    outlineMap = m;
    // outlines in after a layout: that layout's borders alone
    if (worker && lastBox) {
      const zones = newZones(lastBox);
      if (zones.length) worker.postMessage({ kind: "zones", zones } satisfies ToDiagram);
    }
  });
  // T-2447: the place lists of the zones with blocks the zone's own name is taken from, each fetched once
  const townsAsked = new Set<string>();
  const wantTowns = (ids: string[] | undefined) => {
    const ask = (ids ?? []).filter((id) => id && !townsAsked.has(id));
    if (!ask.length || !worker) return;
    for (const id of ask) townsAsked.add(id);
    void Promise.all(ask.map(async (id) => {
      const ps = await placesOfZone(id), big = ps.filter((p) => p.kind === "city" || p.kind === "town");
      const villages = ps.filter((p) => p.kind === "village").sort((a, b) => b.pop - a.pop).slice(0, 30);
      return { id, list: [...big, ...villages].map((p) => {
        const [x, y] = toMerc01(p.lng, p.lat);
        return { name: p.name, width: widthOf(p.name), x, y, big: p.kind !== "village" };
      }) };
    })).then((towns) => worker?.postMessage({ kind: "towns", towns } satisfies ToDiagram));
  };
  let shownSeq = -1;
  let waterMs: unknown = null;
  client.onDiagram = (g) => {
    if (!diagramShown.peek()) return;
    if (!worker) {
      worker = new Worker(new URL("../workers/diagram.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (e: MessageEvent<FromDiagram>) => {
        if (e.data.kind === "progress") {
          if (!shown && diagramShown.peek()) diagramProgress.value = e.data.share;
          return;
        }
        if (e.data.kind === "water") {
          if (shown && e.data.seq === shownSeq && diagramWater.peek()) (shown.d.water = e.data.water), (waterMs = e.data.ms), draw();
          return;
        }
        if (e.data.kind === "sheet") {
          // for checks: the last sheet, its places and lattice (nothing draws it yet)
          (window as unknown as { twSheet: unknown }).twSheet = e.data;
          return;
        }
        if (e.data.kind === "borders") {
          if (shown && e.data.seq === shownSeq) (shown.d.borders = e.data.borders), (shown.names.cities = e.data.cities), draw();
          wantTowns(e.data.cities?.want);
          return;
        }
        busy = false;
        const { graph, layout, drawing: d, names } = e.data;
        shownSeq = e.data.seq;
        shown = indexShown(graph, d, names, layout.x, layout.y);
        focusKey = null;
        hot = -1;
        diagramProgress.value = null;
        diagramReady.value = true;
        // for checks and `?debug=1`: the last layout's figures and how to frame a station
        (window as unknown as { twDiagram: unknown }).twDiagram = { layouts: e.data.seq, ms: { ...layout.ms, ...e.data.ms }, complexes: graph.key.length, pieces: layout.pieces.length, names: names.at.map((l, i) => [names.levels[i], l.node.length]), cities: () => { const c = shown?.names.cities; return c ? [...c.order].map((b) => c.text[b]) : null; }, look: (fx: number, fy: number, pitchPx: number) => { const b = shown!.d.box; (cx = b[0] + (b[2] - b[0]) * fx), (cy = b[1] + (b[3] - b[1]) * fy), (scale = pitchPx / shown!.d.pitch), draw(); }, centreOn: (name: string, pitchPx: number) => { const i = nodeNamed(name); if (i < 0) return false; (cx = layout.x[i]), (cy = layout.y[i]), (scale = pitchPx / shown!.d.pitch), draw(); return true; }, centre: (lng: number, lat: number, pitchPx: number) => { const i = nearest(graph, lng, lat); (cx = layout.x[i]), (cy = layout.y[i]), (scale = pitchPx / shown!.d.pitch), draw(); }, borders: (on?: boolean) => { const b = shown?.d.borders; if (on === false && b) (b.cmds = new Float64Array(0)), (b.path = undefined), draw(); return b ? b.cmds.length : -1; }, bench: (k: number) => { const t = performance.now(); for (let j = 0; j < k; j++) paintNow(); return (performance.now() - t) / k; }, where: (name: string) => whereOf(nodeNamed(name)), linePoint, middle: () => { const s = shown!; const i = nearestOf(s.x, s.y, cx, cy, s.graph.key.keys()); return s.names.text[i]; }, pitchPx: () => scale * shown!.d.pitch, water: () => ({ ms: waterMs, stats: shown?.d.water?.stats, sea: shown?.d.water?.seaStats }) };
        if (!placed) place();
        draw();
        wantTowns(names.cities?.want);
        const next = waiting;
        waiting = null;
        if (next) send(next);
      };
    }
    send(g);
  };
  /** the node of the station with this name, the one most lines stop at; -1 for none */
  const nodeNamed = (name: string): number => {
    const s = shown;
    if (!s) return -1;
    let best = -1;
    s.names.text.forEach((t, i) => {
      if (t === name && (best < 0 || s.stopping[i].length > s.stopping[best].length)) best = i;
    });
    return best;
  };
  /** a node's first pill's middle, page px (for checks) */
  const whereOf = (k: number): [number, number] | null => {
    const s = shown;
    if (!s || k < 0) return null;
    const p = s.d.pills[s.d.pillOf.indexOf(k)], r = canvas.getBoundingClientRect();
    const x = p ? (p[0] + p[2]) / 2 : s.x[k], y = p ? (p[1] + p[3]) / 2 : s.y[k];
    return [r.left + r.width / 2 + (x - cx) * scale, r.top + r.height / 2 - (y - cy) * scale];
  };
  /** of line `num`'s ribbon, the point nearest the view's middle that nothing else is picked at, page px (for checks) */
  const linePoint = (num: number): [number, number] | null => {
    const s = shown, r = canvas.getBoundingClientRect();
    let best: [number, number] | null = null, bd = Infinity;
    for (const i of s?.drawnOf.get(num) ?? []) {
      const p = s!.d.ribbons[i];
      // midpoints: a ribbon's points are at its stations and bends
      for (let j = 0; j + 3 < p.length; j += 2) {
        const px = r.width / 2 + ((p[j] + p[j + 2]) / 2 - cx) * scale, py = r.height / 2 - ((p[j + 1] + p[j + 3]) / 2 - cy) * scale, dd = Math.hypot(px - r.width / 2, py - r.height / 2);
        if (dd >= bd || px < 0 || py < 0 || px > r.width || py > r.height) continue;
        const hit = hitAt(px, py);
        if (hit && "line" in hit && hit.line === i) (bd = dd), (best = [r.left + px, r.top + py]);
      }
    }
    return best;
  };
  /** centred on the selection's node nearest the map's centre, or the complex nearest the map's centre */
  const place = () => {
    const s = shown;
    if (!s) return;
    if (!placed) scale = START_PITCH_PX / s.d.pitch;
    placed = true;
    const c = live.map?.getCenter();
    const on = selectedOn(s, selection.peek());
    const among = on ? selectedNodes(s, on) : undefined;
    const best = c ? nearest(s.graph, c.lng, c.lat, among) : among?.[0] ?? 0;
    cx = s.x[best] ?? 0;
    cy = s.y[best] ?? 0;
    moved = false;
  };
  /** the map centred on the selection's node nearest the diagram's middle, or the complex nearest it */
  const back = () => {
    const s = shown, map = live.map;
    if (!s || !map || !moved) return;
    const on = selectedOn(s, selection.peek());
    const k = nearestOf(s.x, s.y, cx, cy, on ? selectedNodes(s, on) : s.graph.key.keys());
    const st = world.peek()?.save.stations.find((x) => x.id === String(s.graph.key[k]));
    if (!st) return;
    map.jumpTo({ center: [st.lng, st.lat] });
    activeCity.value = zoneAt(st.lng, st.lat);
  };
  const theme = () => {
    const s = getComputedStyle(document.body);
    const v = (k: string, d: string) => s.getPropertyValue(k).trim() || d;
    // the panels' white pills with the ink outline, in both themes, as the line diagrams draw them; on a
    // dark background a thin edge of the map's ink round the outline, so the ring shows (T-2272)
    const bg = v("--bg", "#fbf5ef"), text = v("--map-ink", "#2b2b2b");
    return { bg, ink: v("--net-ink", "#2b2b2b"), pill: "#ffffff", selected: v("--accent", "#d9788e"), text, place: placeNameInk(shade.peek()), font: s.fontFamily, edge: lightness(bg) < 0.5 ? text : undefined, border: BORDER_COLOUR, water: !diagramWater.peek() ? undefined : /^#[0-9a-f]{6}$/i.test(bg) ? mixOklab(bg, SHADES[shade.peek()].water, WATER_SHARE) : SHADES[shade.peek()].water };
  };
  /** the focus, rebuilt when the selection, the layout or the hot line changes */
  let focusKey: [Selection, number] | null = null, focus: DiagramFocus | undefined;
  const focusNow = (): DiagramFocus | undefined => {
    const s = shown, sel = selection.peek();
    if (!s) return undefined;
    if (focusKey && focusKey[0] === sel && focusKey[1] === hot) return focus;
    focusKey = [sel, hot];
    const on = selectedOn(s, sel);
    focus = on
      ? { lines: new Set(on.lines), stations: new Set(selectedNodes(s, on).concat(on.lines.flatMap((i) => s.stops[i]))), selected: new Set(on.node >= 0 ? [on.node] : []), hot }
      : hot >= 0 ? { lines: null, stations: null, selected: new Set(), hot } : undefined;
    return focus;
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
    const f = focusNow();
    const cn = shown.names.cities, d = widened(shown.d, diagramWidth.peek());
    // the city and zone names under the lines (T-2438, T-2521, T-2640)
    paint(ctx, d, canvas.width, canvas.height, cx, cy, scale * dpr, t, f, cn && ((c) => paintCityNames(c, cn, d.pitch, canvas.width, canvas.height, cx, cy, scale * dpr, dpr, { ink: t.place, font: t.font }, { d, plan: shown!.names })));
    // the map's station names: its ink with a halo of the background
    const st = f?.stations;
    paintNames(ctx, shown.names, d, canvas.width, canvas.height, cx, cy, scale * dpr, dpr, { ink: t.text, halo: t.bg, font: t.font, faded: st ? (k) => !st.has(k) : undefined, fadeAlpha: FADE_ALPHA });
  };
  new ResizeObserver(() => draw()).observe(canvas);
  effect(() => {
    void selection.value;
    if (diagramShown.peek() && shown) (moved = true), draw();
  });
  effect(() => {
    void shade.value;
    void colourTheme.value;
    if (diagramShown.peek()) draw();
  });
  // T-2565: the water switch; turning it on lays the diagram out again, with the water asked for
  effect(() => {
    const on = diagramWater.value;
    try { localStorage.setItem(WATER_STORE, on ? "1" : "0"); } catch { /* this visit only */ }
    if (!on) return void draw();
    if (shown && !shown.d.water && lastGraph) send(lastGraph);
  });
  effect(() => {
    const m = diagramWidth.value;
    try { localStorage.setItem(WIDTH_STORE, String(m)); } catch { /* this visit only */ }
    if (diagramShown.peek()) draw();
  });
  // T-2888: the game is paused while the diagram is up. Only a pause made here is undone on the way out, and
  // not if the player has already started the clock again; the speed is the clock's own and is untouched.
  let pausedHere = false;
  diagramShown.subscribe((on) => {
    if (on) {
      if (clock.running.peek()) (pausedHere = true), clock.setRunning(false);
    } else if (pausedHere) {
      pausedHere = false;
      if (!clock.running.peek()) clock.setRunning(true);
    }
    canvas.style.display = on ? "block" : "none";
    client.post({ kind: "diagram", on });
    if (on) {
      // last in the map's box, over the marks added since at the same z-index (attractions, zones loading)
      mapEl.appendChild(canvas);
      if (!shown) diagramProgress.value = 0;
      else place();
      draw();
    } else {
      diagramProgress.value = null;
      hideTag();
      back();
    }
  });
  // build tools work on the map
  effect(() => {
    if (tool.value !== "select" && diagramShown.peek()) diagramShown.value = false;
  });

  // ---- the pointer: drag to pan, wheel to zoom about the pointer, click and hover as on the map
  /** what is under a point, CSS px in the canvas */
  const hitAt = (px: number, py: number): { station: number } | { line: number; at: number } | null => {
    const s = shown;
    if (!s || !scale) return null;
    const ox = px - canvas.clientWidth / 2, oy = py - canvas.clientHeight / 2;
    const x = cx + ox / scale, y = cy - oy / scale;
    const hit = pickAt(widened(s.d, diagramWidth.peek()), x, y, scale, PICK_SLOP_PX);
    if (hit && "station" in hit) return hit;
    const named = nameAt(s.names, s.d, ox, oy, cx, cy, scale);
    return named >= 0 ? { station: named } : hit;
  };
  /** the station a node stands for: its key, the lowest member's id on the lines */
  const stationOf = (k: number) => String(shown!.graph.key[k]);
  const hover = (px: number, py: number) => {
    const s = shown, w = world.peek();
    const hit = hitAt(px, py);
    canvas.style.cursor = hit ? "pointer" : "grab";
    const newHot = hit && "line" in hit ? hit.line : -1;
    if (newHot !== hot) (hot = newHot), draw();
    let lines: TipLine[] | null = null;
    if (s && w && hit && "station" in hit) {
      const id = stationOf(hit.station), c = complexes.peek().of.get(id), ov = live.overlay;
      const ids = c === undefined ? [id] : complexes.peek().members[c];
      lines = ov ? complexTip(ov, w, ids, s.names.text[hit.station]) : [s.names.text[hit.station]];
    } else if (s && w && hit && "line" in hit) {
      const num = s.graph.lines[s.d.lineOf[hit.line]].id, l = w.save.lines.find((x) => x.num === num);
      if (l) lines = lineTag(l, s, hit.line, hit.at);
    }
    if (!lines) {
      if (tagShown) (tag.style.display = "none"), (tagShown = false);
      return;
    }
    tagLines(tag, lines);
    tag.style.display = "";
    tag.style.transform = `translate(${Math.round(px + TAG_OFFSET_PX)}px, ${Math.round(py + TAG_OFFSET_PX)}px)`;
    tagShown = true;
  };
  const click = (px: number, py: number) => {
    const s = shown;
    if (!s) return;
    const hit = hitAt(px, py);
    selection.value = !hit ? null
      : "station" in hit ? { kind: "station", station: stationOf(hit.station) }
      : { kind: "line", line: String(s.graph.lines[s.d.lineOf[hit.line]].id) };
  };
  let drag: { x: number; y: number; x0: number; y0: number; far: boolean } | null = null;
  const local = (e: MouseEvent): [number, number] => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, far: false };
    canvas.setPointerCapture(e.pointerId);
    hideTag();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) {
      hover(...local(e));
      return;
    }
    if (!scale) return;
    if (!drag.far && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < CLICK_PX) return;
    if (!drag.far) (drag.far = true), (canvas.style.cursor = "grabbing");
    cx -= (e.clientX - drag.x) / scale;
    cy += (e.clientY - drag.y) / scale;
    drag = { ...drag, x: e.clientX, y: e.clientY };
    moved = true;
    draw();
  });
  canvas.addEventListener("pointerup", (e) => {
    if (drag && !drag.far) click(...local(e));
    drag = null;
    hover(...local(e));
  });
  canvas.addEventListener("pointercancel", () => {
    drag = null;
    canvas.style.cursor = "grab";
  });
  canvas.addEventListener("pointerleave", () => drag || hideTag());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (!scale) return;
    hideTag();
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left - r.width / 2, py = e.clientY - r.top - r.height / 2;
    const f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
    // the point under the pointer stays there
    cx += px / scale - px / (scale * f);
    cy -= py / scale - py / (scale * f);
    scale *= f;
    moved = true;
    draw();
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (keyIsTheirs(e) || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    const k = e.key.toLowerCase();
    // a build key goes back to the map, where main.ts's keys do the rest
    if (diagramShown.peek() && BUILD_KEYS.has(k)) diagramShown.value = false;
  });

  saver = () => {
    const s = shown;
    if (!s) return;
    const token = showBusy(t("diagramView.savingPicture"));
    afterPaint(() => {
      let blobbing = false;
      try {
        const out = picture(s, theme());
        if (!out) return;
        blobbing = true;
        // for checks: the picture's size and pitch
        (window as unknown as { twDiagramPicture: unknown }).twDiagramPicture = { w: out.canvas.width, h: out.canvas.height, pitchPx: out.pitchPx, levels: s.names.levels };
        out.canvas.toBlob((blob) => {
          hideBusy(token);
          if (!blob) return;
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          const { slots, active } = saveSlotsState.peek();
          const name = (slots.find((x) => x.key === active)?.name ?? "").replace(/[\\/:*?"<>|]/g, "").trim();
          a.download = t("diagramView.pictureFile", { name: name || t("diagramView.unnamed") }) + ".png";
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
        }, "image/png");
      } finally {
        if (!blobbing) hideBusy(token);
      }
    });
  };
}

/** The whole diagram on a canvas of its own, unselected, at the largest names' level of at most
 * PICTURE_PITCH_PX a pitch that keeps it within PICTURE_MAX_SIDE and PICTURE_MAX_AREA. */
function picture(s: Shown, t: { bg: string; ink: string; pill: string; selected: string; text: string; place: string; font: string; edge?: string; border?: string }): { canvas: HTMLCanvasElement; pitchPx: number } | null {
  const d = widened(s.d, diagramWidth.peek()), levels = s.names.levels;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const r of d.ribbons) for (let j = 0; j + 1 < r.length; j += 2) (x0 = Math.min(x0, r[j])), (x1 = Math.max(x1, r[j])), (y0 = Math.min(y0, r[j + 1])), (y1 = Math.max(y1, r[j + 1]));
  if (!(x1 > x0) || !(y1 > y0)) return null;
  // a pill reaches a pitch past a line's end
  const pad = 2 * d.pitch;
  (x0 -= pad), (y0 -= pad), (x1 += pad), (y1 += pad);
  const fits = (w: number, h: number) => w <= PICTURE_MAX_SIDE && h <= PICTURE_MAX_SIDE && w * h <= PICTURE_MAX_AREA;
  const size = (sc: number, box: number[]) => [Math.ceil((box[2] - box[0]) * sc + 2 * PICTURE_MARGIN_PX), Math.ceil((box[3] - box[1]) * sc + 2 * PICTURE_MARGIN_PX)];
  let sc = 0, box = [x0, y0, x1, y1];
  for (let li = levels.length - 1; li >= 0 && !sc; li--) {
    if (levels[li] > PICTURE_PITCH_PX * (1 + 1e-9)) continue;
    const at = levels[li] / d.pitch, b = [x0, y0, x1, y1];
    // the names stick out past the lines
    for (let i = 0; i < s.names.at[li].node.length; i++) {
      const n = placedBox(s.names, li, i, at);
      (b[0] = Math.min(b[0], n[0])), (b[1] = Math.min(b[1], n[1])), (b[2] = Math.max(b[2], n[2])), (b[3] = Math.max(b[3], n[3]));
    }
    const [W, H] = size(at, b);
    if (fits(W, H)) (sc = at), (box = b);
  }
  // too big for station names at any level: the lines alone, as large as fits (the city names under them when the pitch is small enough)
  if (!sc) {
    const span = Math.max(x1 - x0, y1 - y0), area = (x1 - x0) * (y1 - y0);
    sc = Math.min((PICTURE_MAX_SIDE - 2 * PICTURE_MARGIN_PX) / span, Math.sqrt(PICTURE_MAX_AREA / area) * 0.95);
  }
  const [W, H] = size(sc, box);
  const out = document.createElement("canvas");
  out.width = W;
  out.height = H;
  const ctx = out.getContext("2d");
  if (!ctx) return null;
  const mx = (box[0] + box[2]) / 2, my = (box[1] + box[3]) / 2;
  const cn = s.names.cities;
  paint(ctx, d, W, H, mx, my, sc, t, undefined, cn && ((c) => paintCityNames(c, cn, d.pitch, W, H, mx, my, sc, 1, { ink: t.place, font: t.font }, { d, plan: s.names })));
  paintNames(ctx, s.names, d, W, H, mx, my, sc, 1, { ink: t.text, halo: t.bg, font: t.font });
  return { canvas: out, pitchPx: sc * d.pitch };
}
