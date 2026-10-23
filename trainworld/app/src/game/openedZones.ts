// The zones this game has started in, a station has reached (T-729) or built track crosses (T-1047),
// kept in the save's header (`opened`) by the clock worker. Nothing reads it for loading (T-801):
// every zone loads for the view on pan.

import { signal } from "@preact/signals";
import type { CityId } from "./cities";

/** The game's list, in order; null until the clock worker has sent the list of the game in hand. */
export const openedZones = signal<ReadonlySet<CityId> | null>(null);

let sink: ((zones: CityId[]) => void) | null = null;
/** Where an added zone is sent (the clock worker, for the save). */
export function setOpenedSink(f: (zones: CityId[]) => void) {
  sink = f;
}

/** Add zones to the list; the whole list goes to the clock worker when it grew. */
export function addOpened(ids: Iterable<CityId>) {
  const o = openedZones.peek();
  if (o === null) return;
  let next: Set<CityId> | null = null;
  for (const id of ids) if (!o.has(id)) (next ??= new Set(o)).add(id);
  if (!next) return;
  openedZones.value = next;
  sink?.([...next]);
}

/** The list of the game just loaded or started; null for a save from before T-729, which starts with
 * the zone in view (`viewed`) and the zones its stations reach. */
export function setGameOpened(zones: CityId[] | null, viewed: CityId | null) {
  openedZones.value = new Set(zones ?? []);
  if (zones) return;
  if (viewed) addOpened([viewed]);
  else sink?.([]);
}
