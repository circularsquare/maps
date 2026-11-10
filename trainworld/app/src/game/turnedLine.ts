// T-3025: a line turned round a loop of track (T-3021) lists its stops as the way out and then the
// stem again in reverse, so a station comes up twice. The line panel shows it as one row per
// station instead: the stem once, then the loop's stations, the loop drawn as a bend at the side.

import type { LineStats } from "./types";

/** One row of a turned line's stop list. */
export interface TurnRow {
  /** the stop index of the station's first pass, which its time and stretches belong to */
  i: number;
  /** every stop index of the station (the stem is passed twice, a loop's stations once) */
  occ: number[];
  first: boolean;
  last: boolean;
  /** a row of the loop's bend: the station the loop leaves, the loop's own stations, the last of them */
  arc?: "start" | "mid" | "end";
  /** the station the line turns at, where the turn opens again */
  turn?: boolean;
}

export interface TurnedLine {
  rows: TurnRow[];
  /** a loop past the first or last row, with no station on it */
  tipTop: boolean;
  tipBottom: boolean;
}

/**
 * The stops of a one-way closed line read as a stem and the loop at its end: the line goes out over
 * the stem, round the loop, and back over the stem, so its walk (the stops less the closing one,
 * taken round) turns straight back at the stem's end. Null for any other line (an open line, a
 * ring, a loop run both ways) and for a walk that does not read that way.
 */
export function turnedLine(stops: readonly string[], st: Pick<LineStats, "loop" | "oneWay">): TurnedLine | null {
  const n = stops.length;
  if (!st.loop || !st.oneWay || n < 4 || stops[0] !== stops[n - 1]) return null;
  const c = stops.slice(0, n - 1), m = c.length;
  const at = (k: number) => c[((k % m) + m) % m];
  const backs: number[] = [];
  for (let k = 0; k < m; k++) if (at(k - 1) === at(k + 1)) backs.push(k);
  if (!backs.length || backs.length > 2) return null;
  // the walk's closing stop is its first: a circle goes on round there, a balloon turns back
  const loopAt = (k: number) => k !== 0 || st.loop === "circle";
  const row = (k: number, first: boolean, last: boolean): TurnRow => {
    const i = ((k % m) + m) % m;
    return { i, occ: stops.flatMap((s, j) => (s === c[i] ? [j] : [])), first, last };
  };
  let rows: TurnRow[];
  let tipTop = false, tipBottom = false;
  if (backs.length === 2) {
    const [a, b] = backs;
    if (m % 2 || b - a !== m / 2) return null;
    for (let k = 1; k < m / 2; k++) if (at(a + k) !== at(a - k)) return null;
    rows = Array.from({ length: m / 2 + 1 }, (_, r) => row(a + r, r === 0, r === m / 2));
    tipTop = loopAt(a);
    tipBottom = loopAt(b);
  } else {
    const t = backs[0];
    let p = 0;
    while (2 * (p + 1) < m && at(t + p + 1) === at(t - p - 1)) p++;
    const q = m - (2 * p + 1);
    if (q < 2) return null;
    rows = Array.from({ length: p + 1 + q }, (_, r) => row(t + r, r === 0, r === p + q));
    rows[p].turn = true;
    rows[p].arc = "start";
    for (let r = p + 1; r < p + q; r++) rows[r].arc = "mid";
    rows[p + q].arc = "end";
    tipTop = loopAt(t);
  }
  if (new Set(rows.map((r) => c[r.i])).size !== rows.length) return null;
  if (backs.length === 2) rows[rows.length - 1].turn = true;
  return { rows, tipTop, tipBottom };
}
