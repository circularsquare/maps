// Removing track, stations and flyovers (T-096): the delete tool and the inspectors' remove
// buttons. Blueprint goes at once (undoable); something constructed asks first in the centered
// modal (`ask`, ui/Toolbar.tsx), because removing it is final and refunds nothing
// (SPEC 6.4). Lines are removed from the Lines tab, not here. A removal that changes a line asks
// too, built or not, naming each line and what happens to it (T-859).

import { queryClock } from "../workers/clockClient";
import type { EditOp } from "../workers/protocol";
import { distance } from "./format";
import { ask, selection } from "./ui";
import { edit, say, world } from "./world";

export type Removable = { kind: "track"; edge: number } | { kind: "station"; node: number } | { kind: "flyover"; node: number };

/** What would go, in words ("1.2 km of track", "W 42nd St station", "the flyover"), and whether
 * it is constructed; null if it is not there any more. */
export function describeRemovable(r: Removable): { what: string; built: boolean } | null {
  const w = world.peek();
  if (!w) return null;
  if (r.kind === "track") {
    const e = w.edges.find((x) => x.id === r.edge);
    // a crossover goes as a whole, both diagonals (T-338)
    if (e?.crossover) return { what: e.built ? "the constructed crossover" : "the crossover", built: e.built };
    return e ? { what: `${distance(e.lengthM)} of ${e.built ? "constructed " : ""}track`, built: e.built } : null;
  }
  if (r.kind === "station") {
    const s = w.save.stations.find((x) => x.num === r.node);
    return s ? { what: `${s.built ? "the constructed station " : ""}${s.name}${s.built ? "" : " station"}`, built: s.built } : null;
  }
  const n = w.nodes.get(r.node);
  const built = (n?.builtPorts ?? 0) >= 3;
  return n?.flying ? { what: built ? "the constructed flyover" : "the flyover", built } : null;
}

const listed = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** What removing these does to lines (T-859), from the track model's trial of it: a row per line
 * that loses a stop or stops running. */
export async function lineEffects(edges: number[], stations: number[]): Promise<{ line: number; text: string }[]> {
  const d = await queryClock({ what: "removal", edges, stations });
  const w = world.peek();
  if (!d || !w) return [];
  const gone = new Set(stations);
  const names = new Map(w.save.stations.map((s) => [s.num, s.name]));
  const out: { line: number; text: string }[] = [];
  for (let k = 0; k + 2 < d.length; k += 3) {
    const l = w.save.lines.find((x) => x.num === d[k]);
    if (!l) continue;
    const lost = [...new Set(l.stops.map(Number).filter((s) => gone.has(s)))].map((s) => names.get(s) ?? "a station");
    const loses = d[k + 1] && lost.length ? `lose its ${lost.length === 1 ? "stop" : "stops"} at ${listed(lost)}` : "";
    const stops = d[k + 2] ? "stop running until you fix its route" : "";
    out.push({ line: l.num, text: `${l.name} will ${[loses, stops].filter(Boolean).join(" and ")}.` });
  }
  return out;
}

function opOf(r: Removable): EditOp {
  if (r.kind === "track") return { op: "removeEdge", edge: r.edge };
  if (r.kind === "station") return { op: "removeStation", node: r.node };
  return { op: "flying", node: r.node, flying: false };
}

/** The selection is the thing removed. */
function selected(r: Removable): boolean {
  const s = selection.peek();
  if (r.kind === "track") return s?.kind === "track" && s.edge === r.edge;
  if (r.kind === "station") return s?.kind === "station" && s.station === String(r.node);
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
  const lines = r.kind === "flyover" ? [] : await lineEffects(r.kind === "track" ? [r.edge] : [], r.kind === "station" ? [r.node] : []);
  if (!d.built && !lines.length) return go();
  ask.value = { text: `Remove ${d.what}?${d.built ? " Nothing is refunded." : ""}`, yes: "Remove", no: "Keep", run: () => void go(), lines };
}
