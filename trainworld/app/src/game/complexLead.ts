// The station a complex goes by (T-2683): where one name stands for a whole complex (a Stations tab
// row, a search hit, the diagram's label), it is the member whose name the player typed; with several
// of those, or none, the member with the most lines, then a constructed one, then the first in save
// order. Nothing else picks the name, so adding a station or a line never changes a typed name.
// T-3288, T-3750: the members nobody typed a name for all show one name already, the complex's
// (`nameSource`, applied where the worker reads its rows), so the lead's name is that one unless two
// members have typed names.

import type { StationInput } from "./types";
import type { EditOp } from "../workers/protocol";

/** The lead of `members` (not empty), `lines` the number of lines stopping at a member. */
export function leadOf<S extends { id: string; built?: boolean; named?: true }>(members: readonly S[], lines: (id: string) => number): S {
  let lead = members[0], leadLines = lines(lead.id);
  for (let i = 1; i < members.length; i++) {
    const m = members[i], n = lines(m.id);
    const better = !!m.named !== !!lead.named ? !!m.named : n !== leadLines ? n > leadLines : !!m.built && !lead.built;
    if (better) (lead = m), (leadLines = n);
  }
  return lead;
}

/** The member whose name is the complex's (T-3288, T-3750): the oldest (lowest `key`) member whose name
 * the player typed, else the oldest member. Every member with no typed name shows it, and so does a
 * station building nobody named; a member with a typed name shows its own. The one rule for the
 * complex's name: the worker's rows (`StateRows.foldNames`) and the renames go through it. */
export function nameSource<S>(members: readonly S[], key: (s: S) => number, typed: (s: S) => boolean): S | undefined {
  let best: S | undefined;
  for (const m of members) {
    if (best === undefined) best = m;
    else if (typed(m) !== typed(best)) { if (typed(m)) best = m; }
    else if (key(m) < key(best)) best = m;
  }
  return best;
}

/** The edit that renames a complex through its name source `head` (`nameSource`): a typed name stays
 * typed; an untyped one is set with no typed flag, so it stays the one the others follow (T-3288). */
export function complexRename(head: Pick<StationInput, "num" | "length" | "named">, name: string): EditOp {
  return head.named ? { op: "stationProps", node: head.num, platform: head.length, name } : { op: "stationName", node: head.num, name };
}

/** The lead of the complex holding `id`, for the checks and anything holding the lists: `members`
 * the ids of that complex. */
export function complexLead(stations: readonly StationInput[], lines: readonly { stops: readonly string[] }[], members: readonly string[]): StationInput | undefined {
  const byId = new Map(stations.map((s) => [s.id, s]));
  const of = members.map((id) => byId.get(id)).filter((s): s is StationInput => !!s);
  if (!of.length) return undefined;
  return leadOf(of, (id) => lines.reduce((n, l) => n + (l.stops.includes(id) ? 1 : 0), 0));
}
