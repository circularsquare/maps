// Removing track, stations and flyovers (T-096): the delete tool and the inspectors' remove
// buttons. Blueprint goes at once (undoable); something constructed asks first in the hint area
// above the bottom bar (`ask`, ui/Toolbar.tsx), because removing it is final and refunds nothing
// (SPEC 6.4). Lines are removed from the Lines tab, not here.

import type { EditOp } from "../workers/protocol";
import { km } from "./format";
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
    return e ? { what: `${km(e.lengthM)} of ${e.built ? "constructed " : ""}track`, built: e.built } : null;
  }
  if (r.kind === "station") {
    const s = w.save.stations.find((x) => x.num === r.node);
    return s ? { what: `${s.built ? "the constructed station " : ""}${s.name}${s.built ? "" : " station"}`, built: s.built } : null;
  }
  const n = w.nodes.get(r.node);
  const built = (n?.builtPorts ?? 0) >= 3;
  return n?.flying ? { what: built ? "the constructed flyover" : "the flyover", built } : null;
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

/** Remove it: at once if blueprint, after a yes in the hint area if constructed. */
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
  if (!d.built) return go();
  ask.value = { text: `Remove ${d.what}? Nothing is refunded.`, yes: "Remove", no: "Keep", run: () => void go() };
}
