// A thin mode bar for the demand hover tags (T-494): train blue, walking green (yellow under Train
// share bubbles, T-833), driving red from the bubbles' own colours (`modeColours`), widths by share, no numbers. A tag line is HTML
// (map/demandHover.ts), so this returns one; "" when nobody travels, which the tag leaves out.

import type { DemandView } from "../game/demand";
import { modeColours } from "../game/palette";

export function modeBar(train: number, walk: number, drive: number): string {
  const part = (v: number, colour: string) => (v > 0 ? `<i style="flex-grow:${v};background:${colour}"></i>` : "");
  const c = modeColours();
  const inner = part(train, c.train) + part(walk, c.walk) + part(drive, c.drive);
  return inner ? `<div class="mode-bar">${inner}</div>` : "";
}

/** An open zone's local trip legs a day by mode, from its modelled split (T-576): what its far-zoomed
 * bubble's bar, colour and inspector use. Null for a zone not opened, which has no split. */
export function localModes(v: DemandView | undefined): { train: number; walk: number; drive: number } | null {
  if (!v) return null;
  return { train: v.railTripsPerDay, walk: (v.tripsPerDay * v.split.walk) / 100, drive: (v.tripsPerDay * v.split.drive) / 100 };
}
