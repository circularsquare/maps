// The station a complex goes by (T-2683): where one name stands for a whole complex (a Stations tab
// row, a search hit, the diagram's label), it is the member whose name the player typed; with several
// of those, or none, the member with the most lines, then a constructed one, then the first in save
// order. Nothing else picks the name, so adding a station or a line never changes a typed name.
// T-3288: the members nobody typed a name for all show one name already (`nameSource`, applied where
// the worker reads its rows), so with no typed name the lead's name is that one.

import type { StationInput } from "./types";

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

/** The member whose name the rest of a complex show (T-3288): the oldest (lowest `key`) of those no
 * one typed a name for. None when every member has a typed name, and then every member keeps its own. */
export function nameSource<S>(members: readonly S[], key: (s: S) => number, typed: (s: S) => boolean): S | undefined {
  let best: S | undefined;
  for (const m of members) if (!typed(m) && (best === undefined || key(m) < key(best))) best = m;
  return best;
}

/** The lead of the complex holding `id`, for the checks and anything holding the lists: `members`
 * the ids of that complex. */
export function complexLead(stations: readonly StationInput[], lines: readonly { stops: readonly string[] }[], members: readonly string[]): StationInput | undefined {
  const byId = new Map(stations.map((s) => [s.id, s]));
  const of = members.map((id) => byId.get(id)).filter((s): s is StationInput => !!s);
  if (!of.length) return undefined;
  return leadOf(of, (id) => lines.reduce((n, l) => n + (l.stops.includes(id) ? 1 : 0), 0));
}
