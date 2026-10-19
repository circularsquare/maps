import { edit, lineById, nextFreeColour, say, world } from "./world";
import { lineDraft, renameLinePrompt, selection, tool } from "./ui";
import { live } from "../map/live";
import { money } from "./format";
import type { LineInput, TrackEdge } from "./types";

/** Reveal the line before asking for stops, so even an empty line can be named and edited. Given a
 * track (its inspector's New line button), the line starts with the stations at the track's ends; if
 * the track model refuses that, it starts empty. */
export async function createLine(on?: Pick<TrackEdge, "a" | "b">) {
  const ends = on ? (world.value?.save.stations ?? []).filter((s) => s.num === on.a || s.num === on.b).sort((p, q) => (p.num === on!.a ? 0 : 1) - (q.num === on!.a ? 0 : 1)).map((s) => s.num) : [];
  const colour = nextFreeColour();
  let a = ends.length ? await edit({ op: "addLine", stops: ends, name: "", colour }, true) : null;
  if (!a?.ok) { ends.length = 0; a = await edit({ op: "addLine", stops: [], name: "", colour }); }
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

/** Is the line tool adding stops to this line? */
export const editingStopsOf = (l: Pick<LineInput, "num">) => tool.value === "line" && lineDraft.value.line === l.num;

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
