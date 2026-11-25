// T-4214 stage D: a track run's next segment goes at its own click, before the answer to the one before, starting from
// "the end of edit seq N" (`RouteEnd` kind "endOf"). Edits are applied in the order they arrive, so N is settled by then:
// its end is the node its route ended at. A segment from the end of a refused edit (or one this worker never applied) is
// refused with nothing to say (the refused one said why), so a refused segment refuses the rest of its run.

import type { RouteEnd } from "./protocol";

/** the node each route edit ended at, by seq, -1 when refused (the newest KEEP) */
const ends = new Map<number, number>();
const KEEP = 64;

/** Route edit `seq` was applied, ending at `node`, or refused (undefined). */
export function routeEnded(seq: number, node: number | undefined) {
  ends.set(seq, node ?? -1);
  if (ends.size > KEEP) ends.delete(ends.keys().next().value!);
}

function endOf(e: RouteEnd): RouteEnd | null {
  if (e.kind !== "endOf") return e;
  const n = ends.get(e.seq);
  return n === undefined || n < 0 ? null : { kind: "node", node: n };
}

/** `r` with its ends named by edit read as nodes; null when one names an edit refused or not applied. */
export function runEnds<T extends { from: RouteEnd; to: RouteEnd }>(r: T): T | null {
  const from = endOf(r.from), to = endOf(r.to);
  if (!from || !to) return null;
  return from === r.from && to === r.to ? r : { ...r, from, to };
}
