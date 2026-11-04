// T-2266: riders pay the fares their demand was solved with. A fare edit shows in the fare controls
// at once, but a line's income keeps the fares its riders (its `LineDemand`) were solved with until
// riders solved with the new ones show. Per line, not per view: a cluster's final is merged into the
// shown view under the old version, and a view made again from an old solve (a holiday coming near,
// T-1373) holds old riders in new objects. main.ts marks each view the demand client makes with the
// fares of the solve it comes from (T-2286: demandClient's `daysOf` should do this itself).

import type { DayDemand, LineDemand } from "./demand";
import type { Fares } from "./fareCurve";

type Marked = LineDemand & { solvedFares?: Fares };

/** Mark every line of `days` with the fares of the solve they came from. */
export function markSolveFares(days: DayDemand[], fares: Fares | undefined) {
  if (!fares?.nodes) return;
  for (const d of days) if (d.view) for (const l of d.view.lines.values()) (l as Marked).solvedFares = fares;
}

/** The fares to price a line's riders at: those they were solved with, else `current` (riders from a
 * cache, or none yet). */
export const lineFares = (dm: LineDemand | undefined, current: Fares): Fares => (dm as Marked | undefined)?.solvedFares ?? current;
