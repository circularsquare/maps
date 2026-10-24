import { edit, lineById, say, stationsById, world } from "./world";
import { lineDraft, renameLinePrompt, selection, tool } from "./ui";
import { live } from "../map/live";
import { money } from "./format";
import { viewedZone, type CityId } from "./cities";
import { stationZone } from "./listScope";
import { LINE_PALETTE } from "./palette";
import type { LineInput, TrackEdge } from "./types";

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

/** T-813: "Line m" with m the lowest number no line in the city has as its name. */
export function nextLineName(city: CityId | null): string {
  const taken = new Set(lineMates(city).map((l) => l.name));
  for (let k = 1; ; k++) if (!taken.has(`Line ${k}`)) return `Line ${k}`;
}

/** T-813, T-856: a random palette colour no line in the city uses, else a random one of the least used. */
export function nextFreeColour(city: CityId | null): string {
  const uses = new Map<string, number>();
  for (const l of lineMates(city)) uses.set(l.colour.toLowerCase(), (uses.get(l.colour.toLowerCase()) ?? 0) + 1);
  const n = (c: string) => uses.get(c.toLowerCase()) ?? 0;
  const least = Math.min(...LINE_PALETTE.map(n));
  const pool = LINE_PALETTE.filter((c) => n(c) === least);
  return pool[Math.floor(Math.random() * pool.length)];
}

/** Reveal the line before asking for stops, so even an empty line can be named and edited. Given a
 * track (its inspector's New line button), the line starts with the stations at the track's ends; if
 * the track model refuses that, it starts empty. */
export async function createLine(on?: Pick<TrackEdge, "a" | "b">) {
  const ends = on ? (world.value?.save.stations ?? []).filter((s) => s.num === on.a || s.num === on.b).sort((p, q) => (p.num === on!.a ? 0 : 1) - (q.num === on!.a ? 0 : 1)).map((s) => s.num) : [];
  const first = ends.length ? world.value?.save.stations.find((s) => s.num === ends[0]) : undefined;
  const city = first ? stationZone(first) : viewedZone.peek();
  const name = nextLineName(city), colour = nextFreeColour(city);
  let a = ends.length ? await edit({ op: "addLine", stops: ends, name, colour }, true) : null;
  if (!a?.ok) { ends.length = 0; a = await edit({ op: "addLine", stops: [], name, colour }); }
  if (!a.ok || a.line === undefined) return;
  live.tools?.exit();
  lineDraft.value = { line: a.line, stops: ends };
  renameLinePrompt.value = String(a.line);
  selection.value = { kind: "line", line: String(a.line) };
  tool.value = "line";
}

/** The N key: a new line, starting on the selected track when no line runs on it. */
export function newLineKey() {
  const s = selection.peek();
  const e = s?.kind === "track" ? world.peek()?.edges.find((x) => x.id === s.edge) : undefined;
  return createLine(e && e.lines.length === 0 ? e : undefined);
}

/** The Edit stops button and its key: start adding and removing the line's stops, or finish. */
export function toggleEditStops(l: LineInput) {
  if (tool.peek() === "line" && lineDraft.peek().line === l.num) { live.tools?.exit(); return; }
  live.tools?.exit();
  lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
  tool.value = "line";
  selection.value = { kind: "line", line: l.id };
}

/** Edit stops on the selected line (the L key); nothing when no line is selected. */
export function toggleEditStopsSelected() {
  const s = selection.peek();
  const l = s && (s.kind === "line" || s.kind === "train") ? lineById(s.line) : undefined;
  if (l) toggleEditStops(l);
}

/** Construct every blueprint piece (the Build tab's button and the C key); nothing to build does nothing. */
export async function constructBlueprints() {
  const items = world.value?.money.blueprintItems ?? [];
  if (!items.some((r) => r.cost > 0)) return;
  const a = await edit({ op: "constructAll" });
  if (a.ok) say(`Constructed for ${money(a.charge * 1e6)}.`);
}
