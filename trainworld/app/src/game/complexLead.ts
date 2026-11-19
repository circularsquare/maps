// The station a complex goes by (T-2683): where one name stands for a whole complex (a Stations tab
// row, a search hit, the diagram's label), it is the member whose name the player typed; with several
// of those, or none, the member with the most lines, then a constructed one, then the first in save
// order. Nothing else picks the name, so adding a station or a line never changes a typed name.
// T-3288, T-3750, T-3759: the bound members all show one name already, the complex's (`nameSource`,
// applied where the worker reads its rows); `complexTitle` gives it for any member, bound or not.

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

/** The member whose stored name is the complex's (T-3288, T-3750): the oldest (lowest `key`) member whose
 * name the player typed, else the oldest member, bound or not. Every bound member shows it, and so does a
 * station building nobody named; an unbound member shows its own (T-3759). The one rule for the
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

/** T-3759: the name of the complex station `s` is in, as the worker's rows give it: its own `name`
 * unless it is unbound with another (`hub`). Where one name stands for the complex: a Stations tab row,
 * the complex view's title. */
export const complexTitle = (s: Pick<StationInput, "name" | "hub">): string => s.hub ?? s.name;

/** The lead of the complex holding `id`, for the checks and anything holding the lists: `members`
 * the ids of that complex. */
export function complexLead(stations: readonly StationInput[], lines: readonly { stops: readonly string[] }[], members: readonly string[]): StationInput | undefined {
  const byId = new Map(stations.map((s) => [s.id, s]));
  const of = members.map((id) => byId.get(id)).filter((s): s is StationInput => !!s);
  if (!of.length) return undefined;
  return leadOf(of, (id) => lines.reduce((n, l) => n + (l.stops.includes(id) ? 1 : 0), 0));
}
