// A region's goals (T-1437) as ticks over a mode split bar, at their train shares, the star's small
// star under its tick, and the real network's share where the city has one (T-2145). Laid over the
// bar, so nothing around it moves. Used by the city panel; the bar's parent is `position: relative`.

import type { City } from "../game/cities";
import { clockTime, dayOf } from "../game/format";
import { AVERAGE_DAY, weekdayOf } from "../game/days";
import { shiftAt } from "../game/localTime";
import { GOAL_SHARES, goalsReached, STAR_SHARE } from "../game/points";
import { realShare } from "../game/realShare";
import { shownDay, shownDayOn } from "../game/ui";
import { t, tNumber } from "../i18n";
import { tPlace } from "../i18n/places";
import { weekdayLong } from "../i18n/calendar";
import { Icon } from "./widgets";

/** "Monday, day 71, at 14:08" in the region's local time, as the top bar's clock names a day. */
export function goalWhen(city: City, at: number): string {
  const min = Math.floor((at + shiftAt(city.center[0], city.center[1])) / 60), d = dayOf(min);
  return t("goalTicks.when", { weekday: weekdayLong(weekdayOf(d)), day: tNumber(d), time: clockTime(min) });
}

export function goalTip(city: City, share: number, at: number | undefined): string {
  return t("goalTicks.tip", { star: share === STAR_SHARE, reached: at !== undefined, share: tNumber(share), city: tPlace(city.id, city.name), when: at === undefined ? "" : goalWhen(city, at) });
}

/** T-2145 (Anita: "the bars at 10%, 20%, etc are a bit much"): the goals drawn as ticks; the higher
 * ones are still reached and kept (`GOAL_SHARES`), listed in the city view's Goals reached row. */
const DRAWN = [STAR_SHARE, 20];

export function GoalTicks({ city }: { city: City }) {
  const at = goalsReached(city.id);
  const real = realShare(city.id, shownDay.value);
  const dayed = shownDay.value !== AVERAGE_DAY;
  return (
    <>
      {GOAL_SHARES.map((g, k) => DRAWN.includes(g) && (
        <i class={"goal-tick" + (at[k] !== undefined ? " reached" : "")} style={{ left: g + "%" }} data-tip={goalTip(city, g, at[k])}>
          {g === STAR_SHARE && <span class="goal-star"><Icon.star /></span>}
        </i>
      ))}
      {real !== null && (
        <i class="real-mark" style={{ left: real + "%" }} data-tip={t("goalTicks.real", { share: tNumber(real, { minimumFractionDigits: 1, maximumFractionDigits: 1 }), city: tPlace(city.id, city.name), dayed, day: shownDayOn.value })} />
      )}
    </>
  );
}
