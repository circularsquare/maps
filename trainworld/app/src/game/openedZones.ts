// The zones this game has opened (T-729, notes/T-728.md "Cities not yet opened"). A zone not on the
// list is a grey bubble (map/greyCities.ts) and is loaded for viewing like dead space (nothing); a
// click on its bubble or a station within reach adds it, for good. The clock worker keeps the list
// in the save's header (`opened`).

import { signal } from "@preact/signals";
import type { CityId } from "./cities";

/** `?opened=all`: every zone counts as opened and the game's list is left alone (checks that pan). */
export const OPENED_ALL = typeof location !== "undefined" && new URLSearchParams(location.search).get("opened") === "all";

/** The game's list, in the order opened; "all" with `?opened=all`; null until the clock worker has
 * sent the list of the game in hand. */
export const openedZones = signal<ReadonlySet<CityId> | "all" | null>(OPENED_ALL ? "all" : null);

let sink: ((zones: CityId[]) => void) | null = null;
/** Where an added zone is sent (the clock worker, for the save). */
export function setOpenedSink(f: (zones: CityId[]) => void) {
  sink = f;
}

export function isOpened(id: CityId, o = openedZones.peek()): boolean {
  return o === "all" || !!o?.has(id);
}

/** Add zones to the list; the whole list goes to the clock worker when it grew. */
export function addOpened(ids: Iterable<CityId>) {
  const o = openedZones.peek();
  if (o === "all" || o === null) return;
  let next: Set<CityId> | null = null;
  for (const id of ids) if (!o.has(id)) (next ??= new Set(o)).add(id);
  if (!next) return;
  openedZones.value = next;
  sink?.([...next]);
}

/** The list of the game just loaded or started; null for a save from before T-729, which starts with
 * the zone in view (`viewed`) and the zones its stations reach (added by the demand client once the
 * reach is known; Anita, 2026-10-19). */
export function setGameOpened(zones: CityId[] | null, viewed: CityId | null) {
  if (OPENED_ALL) return;
  openedZones.value = new Set(zones ?? []);
  if (zones) return;
  if (viewed) addOpened([viewed]);
  else sink?.([]);
}
