// T-2266: riders pay the fares their demand was solved with. A fare edit shows in the fare controls
// at once, but a line's income keeps the fares its riders (its `LineDemand`) were solved with until
// riders solved with the new ones show. Per line, not per view: a cluster's final is merged into the
// shown view under the old version, and a view made again from an old solve (a holiday coming near,
// T-1373) holds old riders in new objects. The demand client stamps each line with its solve's fares
// (`solvedFares`, in `daysOf`, T-2286).

import type { LineDemand } from "./demand";
import type { Fares } from "./fareCurve";
import { perfOff } from "../perfFlags";

/** The fares to price a line's riders at: those they were solved with, else `current` (riders from a
 * cache, or none yet). */
export const lineFares = (dm: LineDemand | undefined, current: Fares): Fares => dm?.solvedFares ?? current;

/** `v` as JSON with every object's keys in order, so two fares built in another order compare equal. */
const canon = (v: unknown): string =>
  v === null || typeof v !== "object" ? JSON.stringify(v) ?? "null"
  : Array.isArray(v) ? `[${v.map(canon).join(",")}]`
  : `{${Object.keys(v).filter((k) => (v as Record<string, unknown>)[k] !== undefined).sort().map((k) => `${JSON.stringify(k)}:${canon((v as Record<string, unknown>)[k])}`).join(",")}}`;

/** Whether two lines' riders were solved at the same fares. */
export const sameFares = (a: Fares | undefined, b: Fares | undefined): boolean => a === b || (!!a && !!b && canon(a) === canon(b));

/** T-3303: a line's figures from a result that did not solve the regional market (a local result, T-1733, or a
 * cluster without the market) hold its longer trips as last solved, while the result stamps the whole line with its
 * own fares. After a fare edit that line would pay the new fare on riders who chose the old one (a fare a million
 * times higher kept every longer trip until the whole solve's final), so it takes the place of the shown line only
 * when both were solved at the same fares; a line with no shown figures takes it. */
export const takesMarketless = (shown: LineDemand | undefined, got: LineDemand): boolean => !shown || perfOff("fareHold") || sameFares(shown.solvedFares, got.solvedFares);
