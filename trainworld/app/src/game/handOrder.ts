// T-1813: the Lines and Stations tabs' hand order, set by dragging rows. Saved with the game (the
// sim's `LOR1` block, outside the undo history); while a tab has one it shows no sort, and a click
// on a column header sorts again and drops it.

import { signal } from "@preact/signals";

export type ListKind = "lines" | "stations";

/** Line and station ids in the player's order, first first; empty for none. */
export const handOrder = signal<{ lines: readonly string[]; stations: readonly string[] }>({ lines: [], stations: [] });

let sink: ((kind: ListKind, ids: number[]) => void) | null = null;
/** Where a new order goes (the clock worker, for the save). */
export function setHandOrderSink(f: (kind: ListKind, ids: number[]) => void) { sink = f; }

export function setHandOrder(kind: ListKind, ids: readonly string[]) {
  handOrder.value = { ...handOrder.peek(), [kind]: [...ids] };
  sink?.(kind, ids.map(Number));
}

/** A loaded game's order, from the clock worker. */
export function takeHandOrder(lines: readonly number[], stations: readonly number[]) {
  handOrder.value = { lines: lines.map(String), stations: stations.map(String) };
}

/** `items` sorted by the hand order `ids`, each at its first id's place; items with none last, in
 * the order given. */
export function byHand<T>(items: readonly T[], idsOf: (t: T) => readonly string[], ids: readonly string[]): T[] {
  const at = new Map(ids.map((id, i) => [id, i]));
  const rank = (t: T) => idsOf(t).reduce((m, id) => Math.min(m, at.get(id) ?? Infinity), Infinity);
  return items.map((t, i) => ({ t, i, r: rank(t) })).sort((a, b) => (a.r === b.r ? a.i - b.i : a.r - b.r)).map((x) => x.t);
}

/** `order` with `moved` taken out and put back just before `before` (null: at the end). */
export function moveIn(order: readonly string[], moved: string, before: string | null): string[] {
  const out = order.filter((id) => id !== moved);
  const i = before === null ? -1 : out.indexOf(before);
  out.splice(i < 0 ? out.length : i, 0, moved);
  return out;
}
