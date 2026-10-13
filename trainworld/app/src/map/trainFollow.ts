// Following a selected train onto its next trip (T-245). A trip ends at a terminal; the track
// model hands its train over to the next departure from there (T-206, `Service::hand_over`) and
// says which one (`nexts`, per trip in the renderer's buffer). Once the clock passes the end of
// the selected trip, the selection moves to that departure: the same train, turned round, drawn
// from the very spot it stood at. A train that leaves service there names none and stays selected,
// standing where it finished, as before.

import type { Selection } from "../game/types";
import type { NetworkRenderer } from "./renderer";

/** The selection after following the selected train through any hand-overs up to game time `t`;
 * the same object when nothing changed. */
export function followTrain(r: NetworkRenderer, sel: Selection, t: number): Selection {
  if (sel?.kind !== "train" || !r.tripEnds || !r.tripNexts) return sel;
  const epoch = r.tripsEpoch, trips = r.trips;
  let profile = sel.profile, dep = sel.dep;
  // a few hand-overs at once only when the clock jumped (a load, a long pause in a hidden tab)
  for (let hop = 0; hop < 4; hop++) {
    const i = r.tripIndex(profile, dep - epoch);
    if (i < 0) break;
    const end = r.tripEnds[i] + epoch, next = r.tripNexts[i];
    if (t < end || !Number.isFinite(next)) break;
    const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
    let j = -1;
    for (let k = 0; k < r.tripCount; k++) {
      const pk = trips[k * 2];
      if (Math.floor(pk / 6) === line && Math.floor(pk / 3) % 2 !== run && Math.abs(trips[k * 2 + 1] - next) < 0.01) { j = k; break; }
    }
    if (j < 0) break;
    profile = trips[j * 2];
    dep = trips[j * 2 + 1] + epoch;
  }
  if (profile === sel.profile && dep === sel.dep) return sel;
  return { kind: "train", line: sel.line, profile, dep };
}
