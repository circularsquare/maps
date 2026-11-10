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
  /** the station is passed once (a loop's own station), so its dot is filled on the loop's side only */
  once: boolean;
  /** T-3050: a row listed before the walk's first stop (a loop at the top); its place along the line is a lap earlier */
  lead?: boolean;
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
  /** T-3050: the stop index the first row's walk starts at, when that is not the line's first stop (a loop at the top) */
  start?: number;
  /** T-3050: the loop past the stem's top has stations of its own (the rows above the stem's first station) */
  topLoop?: boolean;
}

/**
 * The stops of a one-way closed line read as a stem and the loop at its end: the line goes out over
 * the stem, round the loop, and back over the stem, so its walk (the stops less the closing one,
 * taken round) turns straight back at the stem's end. Null for any other line (an open line, a
 * ring, a loop run both ways) and for a walk that does not read that way.
 */
export function turnedLine(stops: readonly string[], st: Pick<LineStats, "loop" | "oneWay"> & Partial<Pick<LineStats, "reversals">>, topFirst?: (a: string, b: string) => boolean): TurnedLine | null {
  const n = stops.length;
  if (!st.loop || !st.oneWay || n < 4 || stops[0] !== stops[n - 1]) return null;
  const c = stops.slice(0, n - 1), m = c.length;
  // T-3099: the stops alone can't tell a loop past a station from a turn back there (an out-and-back
  // list in Exact edits reads as a loop at each end); a station where the trains reverse is never on a loop.
  // The walk's closing stop is its first: a circle goes on round there, a balloon turns back.
  const turnsAt = (i: number) => (i === 0 ? st.loop !== "circle" : !!st.reversals?.includes(i));
  const both = loopsBothEnds(stops, c, turnsAt, topFirst);
  if (both) return both;
  const at = (k: number) => c[((k % m) + m) % m];
  const backs: number[] = [];
  for (let k = 0; k < m; k++) if (at(k - 1) === at(k + 1)) backs.push(k);
  if (!backs.length || backs.length > 2) return null;
  const loopAt = (k: number) => !turnsAt(((k % m) + m) % m);
  const row = (k: number, first: boolean, last: boolean): TurnRow => {
    const i = ((k % m) + m) % m;
    const occ = stops.flatMap((s, j) => (s === c[i] ? [j] : []));
    return { i, occ, first, last, once: occ.length === 1 && !turnsAt(i) };
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
  if (backs.length === 2 && tipBottom) rows[rows.length - 1].turn = true;
  return { rows, tipTop, tipBottom };
}

/**
 * T-3050: a stem with a loop past each end, each loop having stations of its own: the walk goes down the
 * stem x1..xk, round the loop L2 and back to xk, up the stem again to x1, round the loop L1 and back to
 * x1. Drawn as the stem once with a bend above and a bend below, each loop's stations once in loop
 * order. `topFirst(a, b)` says which of the stem's two ends goes on top (true: a); without it the reading
 * found first, which starts where the line's stops do.
 */
function loopsBothEnds(stops: readonly string[], c: string[], turnsAt: (i: number) => boolean, topFirst?: (a: string, b: string) => boolean): TurnedLine | null {
  const m = c.length, n = stops.length;
  if (m < 8) return null;
  const count = new Map<string, number>();
  for (const s of c) count.set(s, (count.get(s) ?? 0) + 1);
  for (const v of count.values()) if (v > 2) return null;
  const found: TurnedLine[] = [];
  for (let p = 0; p < m; p++) {
    const d = (u: number) => c[(p + u) % m];
    if (count.get(d(0)) !== 2) continue;
    for (let k = 2; 2 * k + 2 <= m; k++) {
      const xk = d(k - 1);
      if (count.get(xk) !== 2) continue;
      // the loop past the stem's far end, then the stem back
      for (let q = 1; 2 * k + q + 1 <= m; q++) {
        if (d(k + q) !== xk) continue;
        let ok = true;
        for (let j = 0; j < k && ok; j++) ok = d(k + q + j) === d(k - 1 - j);
        if (!ok) continue;
        const r = m - 2 * k - q;
        const seen = new Set<string>();
        let fine = true;
        for (let u = 0; u < k + q && fine; u++) { const s = d(u); if (seen.has(s)) fine = false; seen.add(s); }
        for (let u = 2 * k + q; u < m && fine; u++) { const s = d(u); if (seen.has(s)) fine = false; seen.add(s); }
        if (!fine) continue;
        const s0 = (p + 2 * k + q) % m;
        const total = r + k + q;
        const wrapped = s0 + total - 1 >= m;
        const idx: number[] = [];
        for (let u = 0; u < r; u++) idx.push((s0 + u) % m);
        for (let u = 0; u < k + q; u++) idx.push((p + u) % m);
        if (idx.some((i, u) => (u < r || u >= r + k) && turnsAt(i))) continue;
        const rows = idx.map((i, u): TurnRow => {
          const stem = u >= r && u < r + k;
          const occ = stem || i !== 0 ? stops.flatMap((s, j) => (s === c[i] ? [j] : [])) : [0, n - 1];
          const row: TurnRow = { i, occ, first: u === 0, last: u === idx.length - 1, once: !stem };
          if (wrapped && s0 + u < m) row.lead = true;
          if (u < r) row.arc = u === 0 ? "start" : "mid";
          else if (u === r) row.arc = "end";
          else if (u === r + k - 1) { row.arc = "start"; }
          else if (u > r + k - 1) row.arc = u === idx.length - 1 ? "end" : "mid";
          return row;
        });
        found.push({ rows, tipTop: false, tipBottom: false, start: s0, topLoop: true });
      }
    }
  }
  if (!found.length) return null;
  if (topFirst) {
    const top = (t: TurnedLine) => c[t.rows.find((x) => x.arc === "end")!.i], bottom = (t: TurnedLine) => c[t.rows.find((x, u) => x.arc === "start" && u > 0 && !x.once)!.i];
    const pick = found.find((t) => topFirst(top(t), bottom(t)));
    if (pick) return pick;
  }
  return found[0];
}
