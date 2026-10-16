// A thin mode bar for the demand hover tags (T-494): train blue, walking green, driving red from
// the bubbles' own colours (`MODE_COLOURS`), widths by share, no numbers. A tag line is HTML
// (map/demandHover.ts), so this returns one; "" when nobody travels, which the tag leaves out.

import { MODE_COLOURS } from "../game/palette";

export function modeBar(train: number, walk: number, drive: number): string {
  const part = (v: number, colour: string) => (v > 0 ? `<i style="flex-grow:${v};background:${colour}"></i>` : "");
  const inner = part(train, MODE_COLOURS.train) + part(walk, MODE_COLOURS.walk) + part(drive, MODE_COLOURS.drive);
  return inner ? `<div class="mode-bar">${inner}</div>` : "";
}
