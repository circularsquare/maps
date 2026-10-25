// Touch building (T-1260; docs/ui-building.md "Building by touch"). A finger has no hover, so on a
// touch screen the build tools take a tap as a preview and a second tap on the same spot, or the
// control bar's confirm button, as the click. This module says whether the last press was a finger
// and holds what the control bar offers; map/tools.ts fills it, ui/TouchBar.tsx draws it.

import { signal } from "@preact/signals";

/** The last press anywhere on the page was a finger (pen and mouse are not). Browsers follow a
 * tap with mouse events of their own; while this is true the map's mouse moves are those. */
export const touchInput = signal(false);

export function watchInput() {
  addEventListener("pointerdown", (e) => {
    const t = e.pointerType === "touch";
    if (touchInput.peek() !== t) touchInput.value = t;
  }, true);
  // a real mouse moving again (a laptop with a touch screen) takes the hover back
  addEventListener("pointermove", (e) => {
    if (e.pointerType === "mouse" && touchInput.peek()) touchInput.value = false;
  }, { capture: true, passive: true });
}

/** What the control bar shows: `confirm` null for no confirm button, false greyed; `back` drops
 * the last point; `stop` ends the run (track) or drops the placement; `level` and `track` the
 * elevation and track type buttons. */
export interface TouchCtl {
  confirm: boolean | null;
  confirmTip: string;
  back: boolean;
  stopTip: string;
  level: boolean;
  track: boolean;
}

export const touchCtl = signal<TouchCtl | null>(null);

/** the bar's buttons, and `drawn` after each draw of the bar (its size may have changed), set by map/tools.ts */
export const touchActions = { confirm: () => {}, back: () => {}, stop: () => {}, drawn: () => {} };

/** A tap within this of the point being previewed confirms it (px). */
export const SAME_TAP_PX = 24;
