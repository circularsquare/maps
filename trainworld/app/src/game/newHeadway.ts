// T-3543: the headways every new line starts with, set from the timetables window. Saved with the game
// (the header's `newHeadway`, kept by the clock worker, outside the undo history); null: a new line
// starts on the game's own train counts.

import { signal } from "@preact/signals";
import { headwayShown } from "./world";

export type Headways = [number, number, number];

/** The default as the clock worker has it. */
export const newHeadway = signal<Headways | null>(null);

let sink: ((seconds: Headways | null) => void) | null = null;
/** Where a change goes (the clock worker, for the save and for new lines). */
export function setNewHeadwaySink(f: (seconds: Headways | null) => void) { sink = f; }

/** The game's default as the clock worker has it: on a load. */
export function takeNewHeadway(seconds: Headways | null) { newHeadway.value = seconds; }

/** Make line `id`'s headways, as shown, the default for new lines; false when it runs on counts. */
export function makeDefaultHeadway(id: string): boolean {
  const h = headwayShown(id);
  if (!h) return false;
  newHeadway.value = h;
  sink?.(h);
  return true;
}

/** Back to the game's own counts for new lines. */
export function clearDefaultHeadway() {
  newHeadway.value = null;
  sink?.(null);
}
