// T-685, T-1437: a region's star and goals, kept apart from points.ts so a check can load them without a page.

import type { CityId } from "./cities";
import type { MoneyState } from "../workers/protocol";

/** T-685: the rail share (%) a region's star is earned at */
export const STAR_SHARE = 10;
/** T-1437: a region's goals, rail shares (%): the star's first, each kept once reached */
export const GOAL_SHARES = [STAR_SHARE, 20, 30, 40];

/** T-685, T-1437: the stars and goals `views` (each zone's view on the average day) earn now that `m` has not
 * recorded: from final figures only, coarse ones too (T-4117; T-4086 found every region's share within 0.01 points
 * of fine). `share` is left out for the star. */
export function goalsDue(views: ReadonlyMap<CityId, { final?: boolean; split: { train: number } }>, m: Pick<MoneyState, "stars" | "goals">): { zone: CityId; share?: number }[] {
  const out: { zone: CityId; share?: number }[] = [];
  for (const [id, v] of views) {
    if (!v.final) continue;
    if (v.split.train >= STAR_SHARE && m.stars?.[id] === undefined) out.push({ zone: id });
    for (const g of GOAL_SHARES) if (g !== STAR_SHARE && v.split.train >= g && m.goals?.[id]?.[g] === undefined) out.push({ zone: id, share: g });
  }
  return out;
}
