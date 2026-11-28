// T-4555 (T-4511 stage 13): one train picked in the schedule window's timetable or chart, the train the arrows above
// its column move (T-4556). Kept by its line, run and departure and found again in every new answer as the departure
// of that line and run nearest it, within half the gap to its neighbour that side: the sim's rule for which train a
// moved train's move goes to (`moves::secs`). So it stays the same train through a zoom, a move, an Offset step or a
// count step, and is let go when no train is that near (the train gone, or out of the table's day). notes/T-4555.md.

import { signal } from "@preact/signals";

/** Single trains (T-4511) stay off for players until T-4559 switches them on with the real-gap waits and the 90 s
 * gap; `?moves` in the address shows them for checks. */
export const MOVES_SHOWN = typeof location !== "undefined" && new URLSearchParams(location.search).has("moves");

/** A train as the timetable reads it: its line's number, its run and its departure from its first stop, game seconds
 * (`TrackApi.line_trips`'s, moves included). */
export interface TrainAt { line: number; run: number; dep: number }

/** The picked train, the departure as last found; null for none. */
export const pickedTrain = signal<TrainAt | null>(null);

/** The train of `trains` that `p` names now, or null for none. */
export function pickedOf<T extends TrainAt>(p: TrainAt, trains: readonly T[]): T | null {
  const deps: number[] = [];
  let best: T | null = null;
  for (const s of trains) {
    if (s.line !== p.line || s.run !== p.run) continue;
    deps.push(s.dep);
    if (!best || Math.abs(s.dep - p.dep) < Math.abs(best.dep - p.dep)) best = s;
  }
  if (!best) return null;
  deps.sort((a, b) => a - b);
  const i = deps.indexOf(best.dep), side = p.dep < best.dep ? -1 : 1;
  const nb = deps[i + side] ?? deps[i - side];
  return nb === undefined || Math.abs(best.dep - p.dep) < 0.5 * Math.abs(nb - best.dep) ? best : null;
}
