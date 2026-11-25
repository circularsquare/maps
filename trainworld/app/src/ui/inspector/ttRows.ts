// T-4276: the rows of the schedule panel's timetable. A line's are its stops. A group's are every station of its
// lines, in the group view's order (`groupRows`: the axis line's stops, each other line's own stations a block
// where it leaves and rejoins), laid out by one of three looks (notes/T-4276.md has the sheet of them):
// "a" each station once, every train of the group in it; "b" a band per line, its name over its own stops;
// "c" each station once, a row under it per line stopping there. B and C give each line rows of its own, so
// the table picks columns line by line there (`lanes`).

import { signal } from "@preact/signals";
import { stopKey } from "../../game/complexView";
import { groupRows, pathPlaces } from "../../game/groupLayout";
import type { LineInput, WorldView } from "../../game/types";

export type GroupLook = "a" | "b" | "c";
/** The group timetable's look: "a" is built; "b" and "c" are kept until she has picked from the sheet. */
export const groupLook = signal<GroupLook>("a");

/** A row: a station's (`station` its id), or a band's head (`station` null). `lane`: the line (an index into the
 * layout's lines) whose band or lane it is, -1 in a merged table. `first`: the first row of its station's lanes or
 * of its band, which carries the name. */
export interface TtRow { station: string | null; lane: number; first: boolean }

/** The table's rows in each direction (0: the axis line's run 0 way down), and where each line's trains go in them. */
export interface TtLayout {
  rows: [TtRow[], TtRow[]];
  /** per line, per direction, each of its stops' row (-1 none), in the line's own stop order */
  rowOf: [number[], number[]][];
  /** per line, per direction, the rows its trains run through without stopping ("|"); none: every row between
   * a train's first and last */
  fill: ([Uint8Array, Uint8Array] | null)[];
  /** per line: its run 0 goes up the table */
  flipped: boolean[];
  /** each line has rows of its own (looks b and c) */
  lanes: boolean;
}

/** A line's table: its stops down the side. */
export function lineRows(l: LineInput): TtLayout {
  const n = l.stops.length;
  const down = l.stops.map((s): TtRow => ({ station: s, lane: -1, first: true }));
  const idx = l.stops.map((_, i) => i);
  return { rows: [down, [...down].reverse()], rowOf: [[idx, idx.map((i) => n - 1 - i)]], fill: [null], flipped: [false], lanes: false };
}

/** A group's table: `lines` (the axis line first, whose run 0 way is down) in `look`. */
export function groupTable(w: WorldView, lines: readonly LineInput[], look: GroupLook): TtLayout {
  // each line's served stops, and the stations it runs through between them (`passed` counts in its plan's gaps)
  const ins = lines.map((l) => {
    const s0 = l.stopStart ?? 0, n = l.stops.length;
    const passed = (w.lineStats[l.id]?.passed ?? []).map((p) => ({ gap: p.gap - s0, node: p.node })).filter((p) => p.gap >= 0 && p.gap < n - 1);
    return { num: l.num, places: pathPlaces(l.stops, passed, stopKey) };
  });
  // the places some line stops at, top to bottom
  const pos = groupRows(ins, undefined, 0).filter((r) => r.cells.some((c) => c?.stop));
  const P = pos.length, L = lines.length;
  const stopsAt = (k: number, p: number) => !!pos[p].cells[k]?.stop;
  // each line's stops' places (a loop's last stop at its first's)
  const placeOf = lines.map((l, k) => {
    const out = l.stops.map(() => -1);
    pos.forEach((r, p) => { const c = r.cells[k]; if (c?.stop && c.index >= 0 && c.index < out.length) out[c.index] = p; });
    l.stops.forEach((s, i) => {
      if (out[i] >= 0) return;
      const key = stopKey(s);
      out[i] = pos.findIndex((r) => r.key === key && !!r.cells[k]?.stop);
    });
    return out;
  });
  const flipped = placeOf.map((ps) => { const on = ps.filter((p) => p >= 0); return on.length > 1 && on[on.length - 1] < on[0]; });
  const lanes = look !== "a";
  // the rows in direction 0 as [place, lane], and the order direction 1 puts them in: places the other way,
  // bands and lanes kept
  const ids: [number, number][] = [];
  if (look === "a") for (let p = 0; p < P; p++) ids.push([p, -1]);
  else if (look === "b") for (let k = 0; k < L; k++) { ids.push([-1, k]); for (let p = 0; p < P; p++) if (stopsAt(k, p)) ids.push([p, k]); }
  else for (let p = 0; p < P; p++) for (let k = 0; k < L; k++) if (stopsAt(k, p)) ids.push([p, k]);
  const key = (d: number) => ([p, k]: [number, number]) => (look === "b" ? k * (P + 1) + (p < 0 ? 0 : 1 + (d ? P - 1 - p : p)) : (d ? P - 1 - p : p) * (L + 1) + (k + 1));
  const orders = [0, 1].map((d) => ids.map((_, i) => i).sort((a, b) => key(d)(ids[a]) - key(d)(ids[b])));
  const nameOf = (p: number, k: number) => (k >= 0 ? pos[p].cells[k]!.station : (pos[p].cells[0]?.stop ? pos[p].cells[0]!.station : pos[p].cells.find((c) => c?.stop)!.station));
  const rows = orders.map((ord) => ord.map((i, j): TtRow => {
    const [p, k] = ids[i];
    if (p < 0) return { station: null, lane: k, first: true };
    const prev = j ? ids[ord[j - 1]] : null;
    return { station: nameOf(p, k), lane: k, first: look === "c" ? !prev || prev[0] !== p : look === "a" };
  })) as [TtRow[], TtRow[]];
  // each [place, lane]'s row in each direction
  const at = orders.map((ord) => { const m = new Map<number, number>(); ord.forEach((i, r) => m.set(ids[i][0] * (L + 1) + ids[i][1] + 1, r)); return m; });
  const rowAt = (d: number, p: number, k: number) => (p < 0 ? -1 : at[d].get(p * (L + 1) + (lanes ? k : -1) + 1) ?? -1);
  const rowOf = placeOf.map((ps, k) => [0, 1].map((d) => ps.map((p) => rowAt(d, p, k))) as [number[], number[]]);
  // where a line runs through without stopping: a station it passes, or one between two of its stops where another
  // line stopping at both stops too (a local beside an express); never another branch's stations
  const fill = lines.map((_, k): [Uint8Array, Uint8Array] => {
    // a band or lane holds only the line's own stops
    if (lanes) return [new Uint8Array(ids.length), new Uint8Array(ids.length)];
    const by = new Uint8Array(P);
    pos.forEach((r, p) => { if (r.cells[k] && !r.cells[k]!.stop) by[p] = 1; });
    const own = [...new Set(placeOf[k].filter((p) => p >= 0))].sort((a, b) => a - b);
    for (let i = 0; i + 1 < own.length; i++) {
      const a = own[i], b = own[i + 1];
      for (let j = 0; j < L; j++) {
        if (j === k || !stopsAt(j, a) || !stopsAt(j, b)) continue;
        for (let p = a + 1; p < b; p++) if (stopsAt(j, p)) by[p] = 1;
      }
    }
    const m = (d: number) => { const out = new Uint8Array(ids.length); by.forEach((v, p) => { if (v) out[rowAt(d, p, -1)] = 1; }); return out; };
    return [m(0), m(1)];
  });
  return { rows, rowOf, fill, flipped, lanes };
}
