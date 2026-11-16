// T-3420: lines the player hid on the map (the Lines tab's hide mode) and the track levels, buses and ferries
// hidden in Settings' Show. A view setting only: hidden lines still run, hidden track still carries trains.
// Hidden lines are saved with the game (the header's `hidden`, kept by the clock worker outside the undo
// history, as the hand order is); hidden levels, buses and ferries last while the page is open, so a reload never opens
// on a map missing a level. map/visibility.ts is what the renderer and the picks read.

import { signal } from "@preact/signals";
import type { Level } from "./types";

/** Line ids not drawn. */
export const hiddenLines = signal<ReadonlySet<string>>(new Set());
/** Track levels not drawn. */
export const hiddenLevels = signal<ReadonlySet<Level>>(new Set());
/** Bus and ferry lines not drawn (Settings' Show), while the page is open as the levels are. */
export const hideBuses = signal(false);
/** Hide mode: the Lines tab shows each row's box. Leaving it shows every line again, so lines stay hidden only while it is on. */
export const visToggles = signal(false);

let sink: ((lines: string[]) => void) | null = null;
/** Where a change goes (the clock worker, for the save). */
export function setHiddenLinesSink(f: (lines: string[]) => void) { sink = f; }

/** The game's hidden lines as the clock worker has them: on a load, and after a new line reuses an id. A game
 * with lines hidden opens in hide mode, the only place they can be shown again. */
export function takeHiddenLines(lines: readonly string[]) {
  hiddenLines.value = new Set(lines);
  if (lines.length) visToggles.value = true;
}

/** Hide or show lines `ids`. */
export function setLinesHidden(ids: readonly string[], hide: boolean) {
  const was = hiddenLines.peek(), next = new Set(was);
  for (const id of ids) {
    if (hide) next.add(id);
    else next.delete(id);
  }
  if (next.size === was.size && [...next].every((id) => was.has(id))) return;
  hiddenLines.value = next;
  sink?.([...next]);
}

/** Show every line. */
export function showAllLines() { setLinesHidden([...hiddenLines.peek()], false); }

/** Hide a shown level, show a hidden one. */
export function toggleLevel(l: Level) {
  const next = new Set(hiddenLevels.peek());
  if (next.has(l)) next.delete(l);
  else next.add(l);
  hiddenLevels.value = next;
}

/** `hiddenLevels` as the shaders' bits, one per level from -3. */
export const levelBits = (s: ReadonlySet<Level>) => [...s].reduce<number>((m, l) => m | (1 << (l + 3)), 0);
