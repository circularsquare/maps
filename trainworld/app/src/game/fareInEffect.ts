// T-2266: riders pay the fares their demand was solved with. A fare edit shows in the fare controls
// at once, but a line's income keeps the fares its riders (its `LineDemand`) were solved with until
// riders solved with the new ones show. Per line, not per view: a cluster's final is merged into the
// shown view under the old version, and a view made again from an old solve (a holiday coming near,
// T-1373) holds old riders in new objects. The demand client stamps each line with its solve's fares
// (`solvedFares`, in `daysOf`, T-2286).

import type { LineDemand } from "./demand";
import type { Fares } from "./fareCurve";

/** The fares to price a line's riders at: those they were solved with, else `current` (riders from a
 * cache, or none yet). */
export const lineFares = (dm: LineDemand | undefined, current: Fares): Fares => dm?.solvedFares ?? current;
