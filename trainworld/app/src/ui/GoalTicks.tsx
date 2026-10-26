// A region's goals (T-1437) as ticks over a mode split bar, at their train shares, the star's small
// star under its tick. Laid over the bar, so nothing around it moves. Used by the top bar and the
// city view; the bar's parent is `position: relative`.

import type { City } from "../game/cities";
import { clockTime, dayOf } from "../game/format";
import { weekdayOf } from "../game/days";
import { shiftAt } from "../game/localTime";
import { GOAL_SHARES, goalsReached, STAR_SHARE } from "../game/points";
import { Icon } from "./widgets";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/** "Monday, day 71, at 14:08" in the region's local time, as the top bar's clock names a day. */
export function goalWhen(city: City, at: number): string {
  const t = Math.floor((at + shiftAt(city.center[0], city.center[1])) / 60), d = dayOf(t);
  return `${WEEKDAY_LONG[weekdayOf(d)]}, day ${d}, at ${clockTime(t)}`;
}

export function goalTip(city: City, share: number, at: number | undefined): string {
  const what = share === STAR_SHARE ? `Star at ${share}% train share` : `Goal: ${share}% train share`;
  return at === undefined ? `${what} in ${city.name}` : `${what}, reached on ${goalWhen(city, at)}`;
}

export function GoalTicks({ city }: { city: City }) {
  const at = goalsReached(city.id);
  return (
    <>
      {GOAL_SHARES.map((g, k) => (
        <i class={"goal-tick" + (at[k] !== undefined ? " reached" : "")} style={{ left: g + "%" }} data-tip={goalTip(city, g, at[k])}>
          {g === STAR_SHARE && <span class="goal-star"><Icon.star /></span>}
        </i>
      ))}
    </>
  );
}
