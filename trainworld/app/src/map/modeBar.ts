// A thin mode bar for the demand hover tags (T-494): train blue, walking green (yellow under Train
// share bubbles, T-833), driving red from the bubbles' own colours (`modeColours`), and No route grey
// (T-2215), widths by share, no numbers. A tag line is HTML (map/demandHover.ts), so this returns
// one; "" when nobody travels, which the tag leaves out.

import type { DemandView } from "../game/demand";
import { modeColours } from "../game/palette";

/** `width`: the bar's share of the tag's width (T-2271: a bubble's bar by its count, `logShare`). */
export function modeBar(train: number, walk: number, drive: number, none = 0, width = 1): string {
  const part = (v: number, colour: string) => (v > 0 ? `<i style="flex-grow:${v};background:${colour}"></i>` : "");
  const c = modeColours();
  const inner = part(train, c.train) + part(walk, c.walk) + part(drive, c.drive) + part(none, c.none);
  return inner ? `<div class="mode-bar"${width < 1 ? ` style="width:${(100 * width).toFixed(1)}%"` : ""}>${inner}</div>` : "";
}

/** T-2271: where `n` sits on a log scale from `lo` (0) to `hi` (1), held to that range. */
export const logShare = (n: number, lo: number, hi: number): number => (n > lo ? Math.min(1, Math.log(n / lo) / Math.log(hi / lo)) : 0);

/** An open zone's local trip legs a day by mode, from its modelled split (T-576): what its far-zoomed
 * bubble's bar, colour and inspector use. Null for a zone not opened, which has no split. */
export function localModes(v: DemandView | undefined): { train: number; walk: number; drive: number; none: number } | null {
  if (!v) return null;
  return { train: v.railTripsPerDay, walk: (v.tripsPerDay * v.split.walk) / 100, drive: (v.tripsPerDay * v.split.drive) / 100, none: (v.tripsPerDay * (v.split.none ?? 0)) / 100 };
}
