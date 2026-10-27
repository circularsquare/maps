// Removing track, stations, flyovers and bus lines (T-096, T-1513): the delete tool and the inspectors'
// remove buttons. Blueprint goes at once (undoable); something constructed asks first in the centered
// modal (`ask`, ui/ToolNotes.tsx), because removing it is final and refunds nothing
// (SPEC 6.4). Train lines are removed from the Lines tab, not here; a bus line has no track to remove,
// so the delete tool takes its route and the line with it. A removal that changes a line asks
// too, built or not, naming each line and what happens to it (T-859).

import { queryClock } from "../workers/clockClient";
import type { EditOp } from "../workers/protocol";
import { andList, distance } from "./format";
import { ask, selection } from "./ui";
import { edit, linesByNum, say, stationsByNum, world } from "./world";

/** `bus`: a bus line by its number, route and line together. */
export type Removable = { kind: "track"; edge: number } | { kind: "station"; node: number } | { kind: "flyover"; node: number } | { kind: "bus"; line: number };

/** T-1161: the track a piece's removal keeps at its a and b ends, m. Past a station the removal
 * would leave at a bare end, half its platform and 4 m stay, so the station keeps its place; the
 * rule of the track model's `station_trim_ops` (sim/src/track/wasm/links.rs). */
export function stationTrim(edge: number): [number, number] {
  const w = world.peek();
  const e = w?.edges.find((x) => x.id === edge);
  if (!w || !e || e.crossover || e.a === e.b) return [0, 0];
  const keep = (id: number) => {
    const n = w.nodes.get(id);
    return n && n.platform > 0 && n.ports === 2 ? n.platform / 2 + 4 : 0;
  };
  const [ka, kb] = [keep(e.a), keep(e.b)];
  return ka + kb + 7 > e.lengthM ? [0, 0] : [ka, kb];
}

/** What would go, in words ("1.2 km of track", "W 42nd St station", "the flyover"), and whether
 * it is constructed; null if it is not there any more. */
export function describeRemovable(r: Removable): { what: string; built: boolean } | null {
  const w = world.peek();
  if (!w) return null;
  if (r.kind === "track") {
    const e = w.edges.find((x) => x.id === r.edge);
    // a crossover goes as a whole, both diagonals (T-338)
    if (e?.crossover) return { what: e.built ? "the built crossover" : "the crossover", built: e.built };
    const [ka, kb] = stationTrim(r.edge);
    return e ? { what: `${distance(e.lengthM - ka - kb)} of ${e.built ? "built " : ""}track`, built: e.built } : null;
  }
  if (r.kind === "bus") {
    const l = linesByNum(w).get(r.line), built = l ? w.lineStats[l.id]?.bus?.built : undefined;
    return l && built !== undefined ? { what: l.name, built } : null;
  }
  if (r.kind === "station") {
    const s = stationsByNum(w).get(r.node);
    return s ? { what: `${s.built ? "the built station " : ""}${s.name}${s.built ? "" : " station"}`, built: s.built } : null;
  }
  const n = w.nodes.get(r.node);
  const built = (n?.builtPorts ?? 0) >= 3;
  return n?.flying ? { what: built ? "the built flyover" : "the flyover", built } : null;
}

/** T-1464: the question before a line is deleted. A bus route goes with its line, and a constructed
 * one is not refunded (it has no track or stations to leave). */
export function deleteLineText(name: string, bus?: { built: boolean }): string {
  if (!bus) return `Delete ${name}? Track and stations stay.`;
  return `Delete ${name}?${bus.built ? " Nothing is refunded." : ""}`;
}

/** What removing these does to lines (T-859), from the track model's trial of it: a row per line
 * that loses a stop or stops running. */
export async function lineEffects(edges: number[], stations: number[]): Promise<{ line: number; text: string }[]> {
  const d = await queryClock({ what: "removal", edges, stations });
  const w = world.peek();
  if (!d || !w) return [];
  const gone = new Set(stations);
  const byNum = stationsByNum(w), lineOf = linesByNum(w);
  const out: { line: number; text: string }[] = [];
  for (let k = 0; k + 2 < d.length; k += 3) {
    const l = lineOf.get(d[k]);
    if (!l) continue;
    const lost = [...new Set(l.stops.map(Number).filter((s) => gone.has(s)))].map((s) => byNum.get(s)?.name ?? "a station");
    const loses = d[k + 1] && lost.length ? `lose its ${lost.length === 1 ? "stop" : "stops"} at ${andList(lost)}` : "";
    const stops = d[k + 2] ? "stop running until you fix its route" : "";
    out.push({ line: l.num, text: `${l.name} will ${[loses, stops].filter(Boolean).join(" and ")}.` });
  }
  return out;
}

function opOf(r: Removable): EditOp {
  if (r.kind === "track") return { op: "removeEdge", edge: r.edge };
  if (r.kind === "station") return { op: "removeStation", node: r.node };
  if (r.kind === "bus") return { op: "removeLine", line: r.line };
  return { op: "flying", node: r.node, flying: false };
}

/** The selection is the thing removed. */
function selected(r: Removable): boolean {
  const s = selection.peek();
  if (r.kind === "track") return s?.kind === "track" && s.edge === r.edge;
  if (r.kind === "station") return s?.kind === "station" && s.station === String(r.node);
  if (r.kind === "bus") return (s?.kind === "line" || s?.kind === "train") && linesByNum(world.peek()!).get(r.line)?.id === s.line;
  return false;
}

/** Remove it: at once if blueprint and no line changes, else after a modal confirmation. */
export async function removeThing(r: Removable) {
  const d = describeRemovable(r);
  if (!d) return;
  const go = async () => {
    ask.value = null;
    const wasSelected = selected(r);
    const a = await edit(opOf(r));
    if (!a.ok) return;
    if (wasSelected) selection.value = null;
    say(`Removed ${d.what}.`);
  };
  const lines = r.kind === "flyover" || r.kind === "bus" ? [] : await lineEffects(r.kind === "track" ? [r.edge] : [], r.kind === "station" ? [r.node] : []);
  if (!d.built && !lines.length) return go();
  ask.value = { text: `Remove ${d.what}?${d.built ? " Nothing is refunded." : ""}`, yes: "Remove", no: "Keep", run: () => void go(), lines };
}
