import { edit, lineById, say, stationsById, world } from "./world";
import { lineDraft, newLine, renameLinePrompt, selection, tool } from "./ui";
import { live } from "../map/live";
import { busDraft } from "../map/busDraw";
import { ferryDraft } from "../map/ferryDraw";
import { money } from "./format";
import { viewedZone, type CityId } from "./cities";
import { stationZone } from "./listScope";
import { blueprintQuote } from "./quotes";
import { pickColour } from "./colourPick";
import { rememberPicked, settleNewLineColour } from "./lineColour";
import type { LineInput, StationInput, TrackEdge } from "./types";

/** The lines that share a new line's city: a line's city is its first stop's zone (the Lines tab's
 * City column); a line with no stops yet counts in the city being viewed. */
function lineMates(city: CityId | null): LineInput[] {
  const w = world.value;
  if (!w) return [];
  const at = stationsById(w);
  return w.save.lines.filter((l) => {
    const s = at.get(l.stops[0]);
    return (s ? stationZone(s) : viewedZone.peek()) === city;
  });
}

/** T-2526: the lines with any stop in the city's zone, whatever their home; a line with no stops
 * yet counts in the city being viewed. */
function linesStoppingIn(city: CityId | null): LineInput[] {
  const w = world.value;
  if (!w) return [];
  const at = stationsById(w);
  return w.save.lines.filter((l) => {
    if (!l.stops.length) return viewedZone.peek() === city;
    return l.stops.some((n) => { const s = at.get(n); return !!s && stationZone(s) === city; });
  });
}

/** T-813, T-2526: "Line m" with m the lowest number no line stopping in the city has as its name. */
export function nextLineName(city: CityId | null): string {
  const taken = new Set(linesStoppingIn(city).map((l) => l.name));
  for (let k = 1; ; k++) if (!taken.has(`Line ${k}`)) return `Line ${k}`;
}

/** T-1278: lines with a stop this near where a new line starts set its colour, m ... */
const NEAR_M = 15_000;
/** ... topped up with the nearest lines beyond to this many. */
const NEAR_MIN = 6;

/** T-1278: the palette colour farthest in OKLab from the colours of the lines near `at` (lng, lat;
 * none: every line), ties at random; `cityLines` the lines already in the new line's city. */
export function nextFreeColour(at: [number, number] | null, cityLines = Infinity): string {
  const w = world.peek();
  const lines = w?.save.lines ?? [];
  const pos = w ? stationsById(w) : new Map<string, StationInput>();
  const kx = at ? 111_320 * Math.cos((at[1] * Math.PI) / 180) : 0;
  const near = lines.map((l) => {
    let d = Infinity;
    if (at) for (const s of l.stops) {
      const p = pos.get(s);
      if (p) d = Math.min(d, Math.hypot((p.lng - at[0]) * kx, (p.lat - at[1]) * 111_320));
    }
    return { colour: l.colour, d: at ? d : 0 };
  }).sort((a, b) => a.d - b.d);
  return pickColour(near.filter((x, i) => x.d <= NEAR_M || i < NEAR_MIN).map((x) => x.colour), cityLines);
}

/** Where a new line starts: its first stop, else the middle of the map. */
function startOf(first: StationInput | undefined): [number, number] | null {
  if (first) return [first.lng, first.lat];
  const c = live.map?.getCenter();
  return c ? [c.lng, c.lat] : null;
}

/** Reveal the line before asking for stops, so even an empty line can be named and edited. Given a
 * track (its inspector's New line button), the line starts with the stations at the track's ends; if
 * the track model refuses that, it starts empty. */
export async function createLine(on?: Pick<TrackEdge, "a" | "b">) {
  const ends = on ? (world.value?.save.stations ?? []).filter((s) => s.num === on.a || s.num === on.b).sort((p, q) => (p.num === on!.a ? 0 : 1) - (q.num === on!.a ? 0 : 1)).map((s) => s.num) : [];
  const first = ends.length ? world.value?.save.stations.find((s) => s.num === ends[0]) : undefined;
  const city = first ? stationZone(first) : viewedZone.peek();
  const name = nextLineName(city), colour = nextFreeColour(startOf(first), lineMates(city).length);
  let a = ends.length ? await edit({ op: "addLine", stops: ends, name, colour, autoFast: true }, true) : null;
  if (!a?.ok) { ends.length = 0; a = await edit({ op: "addLine", stops: [], name, colour }); }
  if (!a.ok || a.line === undefined) return;
  live.tools?.exit();
  lineDraft.value = { line: a.line, stops: ends };
  renameLinePrompt.value = String(a.line);
  selection.value = { kind: "line", line: String(a.line) };
  tool.value = "line";
  newLine.value = { line: a.line, fast: !!world.peek()?.lineStats[String(a.line)]?.fast };
  rememberPicked(a.line, colour);
  void settleNewLineColour(a.line);
}

/** The N key: a new line, starting on the selected track when no line runs on it. */
export function newLineKey() {
  const s = selection.peek();
  const e = s?.kind === "track" ? world.peek()?.edges.find((x) => x.id === s.edge) : undefined;
  return createLine(e && e.lines.length === 0 ? e : undefined);
}

/** T-1464: draw a new bus line (the B key, the Lines tab's bus button, the Build tab's Bus tool):
 * click points, then finish (map/busDraw.ts). */
export function newBusLine() {
  const t = live.tools;
  if (!t || !world.peek()) return;
  t.exit();
  t.bus.begin(null);
}

/** T-1464: the bus line's Edit route button: the tool opens with the line's route to redraw;
 * pressed again it saves the route, as the tool's other ways to finish do (T-1514). Nothing for a
 * line that is no bus line. */
export function editBusRoute(l: Pick<LineInput, "id" | "num" | "colour">) {
  const t = live.tools, bus = world.peek()?.lineStats[l.id]?.bus, route = bus?.routeLL;
  if (!t || !route) return;
  if (tool.peek() === "bus" && busDraft.peek()?.line === l.num) return void t.bus.finish();
  t.exit();
  selection.value = { kind: "line", line: l.id };
  t.bus.begin(l.num, route, l.colour, bus.kinds);
}

/** T-2194: draw a new ferry line (the O key, the Lines tab's ferry button, the Build tab's Ferry tool):
 * place ports, then finish (map/ferryDraw.ts). */
export function newFerryLine() {
  const t = live.tools;
  if (!t || !world.peek()) return;
  t.exit();
  t.ferry.begin(null);
}

/** T-2194: the ferry line's Edit route button, as the bus line's. */
export function editFerryRoute(l: Pick<LineInput, "id" | "num" | "colour">) {
  const t = live.tools, ferry = world.peek()?.lineStats[l.id]?.bus?.ferry;
  if (!t || !ferry) return;
  if (tool.peek() === "ferry" && ferryDraft.peek()?.line === l.num) return void t.ferry.finish();
  t.exit();
  selection.value = { kind: "line", line: l.id };
  t.ferry.begin(l.num, l.colour);
}

/** The Edit stops button and its key: start adding and removing the line's stops, or finish. */
export function toggleEditStops(l: LineInput) {
  if (tool.peek() === "line" && lineDraft.peek().line === l.num) { live.tools?.exit(); return; }
  live.tools?.exit();
  lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
  tool.value = "line";
  selection.value = { kind: "line", line: l.id };
}

/** Edit stops on the selected line (the S key); nothing when no line is selected. */
export function toggleEditStopsSelected() {
  const s = selection.peek();
  const l = s && (s.kind === "line" || s.kind === "train") ? lineById(s.line) : undefined;
  if (l) toggleEditStops(l);
}

/** The G key (T-1417): every station the selected line's path passes becomes a stop; nothing when
 * no line panel is open or the line passes none. */
export async function addAllPassedSelected() {
  const s = selection.peek();
  const w = world.peek();
  const l = s?.kind === "line" ? lineById(s.line) : undefined;
  const passed = l && w?.lineStats[l.id]?.passed;
  if (!l || !passed?.length) return;
  const stops = l.stops.flatMap((x, i) => [Number(x), ...passed.filter((p) => p.gap === i).map((p) => p.node)]);
  const a = await edit({ op: "setStops", line: l.num, stops });
  if (a.ok && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops };
}

/** Construct every blueprint piece (the Build tab's button and the C key); nothing to build does nothing. */
export async function constructBlueprints() {
  const w = world.peek();
  const items = w ? await blueprintQuote(w.version, true) : null;
  if (!items?.some((r) => r.cost > 0)) return;
  const a = await edit({ op: "constructAll" });
  if (a.ok) say(`Constructed for ${money(a.charge * 1e6)}.`);
}
