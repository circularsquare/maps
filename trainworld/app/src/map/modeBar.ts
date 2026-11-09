// A thin mode bar for the demand hover tags (T-494): train blue, walking green (yellow under Train
// share bubbles, T-833), driving red from the bubbles' own colours (`modeColours`), and No route grey
// (T-2215), widths by share, no numbers. A tag line is HTML (map/demandHover.ts), so this returns
// one; "" when nobody travels, which the tag leaves out.

import type { DemandView } from "../game/demand";
import { MODE_COLOURS, modeColours } from "../game/palette";
import { t, tNumber } from "../i18n";

/** `width`: the bar's share of the tag's width (T-2271: a bubble's bar by its count, `logShare`). */
export function modeBar(train: number, walk: number, drive: number, none = 0, width = 1): string {
  const part = (v: number, colour: string) => (v > 0 ? `<i style="flex-grow:${v};background:${colour}"></i>` : "");
  const c = modeColours();
  const inner = part(train, c.train) + part(walk, c.walk) + part(drive, c.drive) + part(none, c.none);
  return inner ? `<div class="mode-bar"${width < 1 ? ` style="width:${(100 * width).toFixed(1)}%"` : ""}>${inner}</div>` : "";
}

/** A bubble's trips under the Purpose scheme's three groups (`PURPOSE_CLASS`). */
export interface PurposeSplit { work: number; other: number; errands: number }

/** T-2908: the Purpose scheme's hover lines: the bar, then a row per group whose share rounds to at least 1%. Fixed colours,
 * the bubbles' own, whatever scheme the plain mode bars are in. */
export function purposeLines(p: PurposeSplit, width = 1): string[] {
  const all = p.work + p.errands + p.other;
  if (!(all > 0)) return [];
  const groups = [[p.work, MODE_COLOURS.train, "modeBar.purposeWork"], [p.errands, MODE_COLOURS.drive, "modeBar.purposeErrands"], [p.other, MODE_COLOURS.walk, "modeBar.purposeOther"]] as const;
  const shown = groups.filter(([v]) => v > 0);
  const bar = `<div class="mode-bar"${width < 1 ? ` style="width:${(100 * width).toFixed(1)}%"` : ""}>${shown.map(([v, c]) => `<i style="flex-grow:${v};background:${c}"></i>`).join("")}</div>`;
  const percent = (v: number) => tNumber(v, { style: "percent", maximumFractionDigits: 0 });
  return [bar, ...shown.filter(([v]) => Math.round((100 * v) / all) >= 1).map(([v, c, key]) =>
    `<span class="tag-fact purpose-row"><span class="tag-dot" style="background:${c}"></span>${t(key, { share: percent(v / all) })}</span>`)];
}

/** T-2271: where `n` sits on a log scale from `lo` (0) to `hi` (1), held to that range. */
export const logShare = (n: number, lo: number, hi: number): number => (n > lo ? Math.min(1, Math.log(n / lo) / Math.log(hi / lo)) : 0);

/** An open zone's local trip legs a day by mode, from its modelled split (T-576): what its far-zoomed
 * bubble's bar, colour and inspector use. Null for a zone not opened, which has no split. */
export function localModes(v: DemandView | undefined): { train: number; walk: number; drive: number; none: number } | null {
  if (!v) return null;
  return { train: v.railTripsPerDay, walk: (v.tripsPerDay * v.split.walk) / 100, drive: (v.tripsPerDay * v.split.drive) / 100, none: (v.tripsPerDay * (v.split.none ?? 0)) / 100 };
}
