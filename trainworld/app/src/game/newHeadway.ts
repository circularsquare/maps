// T-3543, T-3588: the headways every new line starts with, one set per kind of line (normal train, tram,
// high speed, bus, ferry; T-3703), set from the timetables window. Saved with the game (the header's `newHeadway`,
// kept by the clock worker, outside the undo history). A new game has none: a new rail line starts on train counts
// until the player sets a default for its kind, and a new bus or ferry line on its tool's headways (`NEW_LINE`).

import { signal } from "@preact/signals";
import type { NewHeadways, Headways } from "../workers/saveFile";
import { headwaysFor } from "./newLine";
import { world, headwayShown, headwaysOfTrains, setAutoTrains, setHeadways } from "./world";

export type { Headways, NewHeadways };
export { headwaysFor };
export type LineKind = keyof NewHeadways;

/** The defaults as the clock worker has them. */
export const newHeadway = signal<NewHeadways>({});

let sink: ((seconds: NewHeadways) => void) | null = null;
/** Where a change goes (the clock worker, for the save and for new lines). */
export function setNewHeadwaySink(f: (seconds: NewHeadways) => void) { sink = f; }

/** The defaults as the clock worker has them: on a load. */
export function takeNewHeadway(seconds: NewHeadways) { newHeadway.value = seconds; }

/** The kind of default line `id` takes. */
export function lineKind(id: string): LineKind | null {
  const st = world.value?.lineStats[id];
  if (!st) return null;
  if (st.bus) return st.bus.ferry ? "ferry" : "bus";
  return st.tram ? "tram" : st.fast ? "fast" : "rail";
}

/** Make line `id`'s headways the default for new lines of its kind. A line on counts is switched to
 * Headway first, at the headways its trains give, so it runs the same service; false when it has none
 * to give. */
export function makeDefaultHeadway(id: string): boolean {
  const kind = lineKind(id);
  if (!kind) return false;
  let h = headwayShown(id);
  if (!h) {
    h = headwaysOfTrains(id);
    if (!h) return false;
    setAutoTrains(id, true);
  }
  newHeadway.value = { ...newHeadway.value, [kind]: h };
  sink?.(newHeadway.value);
  return true;
}

/** Give line `id` the default headways of its kind, as the edit a player's own headway change is; false
 * when its kind has no default. */
export function applyDefaultHeadway(id: string): boolean {
  const kind = lineKind(id), h = kind && headwaysFor(kind, newHeadway.value);
  if (!h) return false;
  setHeadways(id, h);
  return true;
}
