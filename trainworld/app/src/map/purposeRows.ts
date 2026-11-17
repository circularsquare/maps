// T-3510: a bubble selection's purpose rows with its trips with other places (T-3503). The demand
// workers' rows (`Flows.purposes`) count the selection's local trips by purpose, period and mode;
// its total, as its bubbles, also counts the regional layer's trips at the end in the zone. Each
// purpose's cell legs (`cellModes(purpose)`) hold its own regional part (commutes across a zone line
// under Work, outings across one under their purposes), so a purpose's row takes them, spread over
// the periods as its local trips are (the model times both by the same tables); what all purposes'
// legs hold beyond every purpose's is the longer trips with other places, a row of its own at the
// longer trips' timing. The rows then add up to the selection's total under All purposes.

import type { CellModes, PurposeTrips } from "../game/demand";
import { longerShare } from "../game/marketRule";
import { PERIODS } from "../game/periods";
import type { PurposeKey } from "../game/purposes";

/** The shown purposes with their own cell legs, and the workers' row names each counts. */
export const ROW_PURPOSES: [PurposeKey, string[]][] = [
  ["work", ["Work"]], ["school", ["School", "University"]], ["shopping", ["Errands"]], ["social", ["Leisure"]], // i18n-ignore: row ids
  ["medical", ["Medical"]], ["airport", ["Airports"]], ["attractions", ["Attractions"]], // i18n-ignore: row ids
];
/** The longer trips' row id (shown through `purposeLabel`). */
export const LONGER_ROW = "Longer"; // i18n-ignore: a row id

const NQ = PERIODS.length;
const zeros = () => [0, 1, 2].map(() => new Array<number>(NQ).fill(0));

/** Legs by rail, walking and driving (No route with driving, as the rows count it) over `cells`. */
function legsOver(cells: ArrayLike<number>, ends: Float32Array[]): [number, number, number] {
  const [r, w, d, u] = ends;
  let a = 0, b = 0, c = 0;
  for (let k = 0; k < cells.length; k++) {
    const i = cells[k];
    (a += r[i]), (b += w[i]), (c += d[i] + (u?.[i] ?? 0));
  }
  return [a, b, c];
}

/** Each period's share of `rows`' legs, both directions apart, all modes together (null: none). */
function shape(rows: PurposeTrips[]): { out: number[]; back: number[] } | null {
  const out = new Array<number>(NQ).fill(0), back = new Array<number>(NQ).fill(0);
  let all = 0;
  for (const p of rows) for (let m = 0; m < 3; m++) for (let q = 0; q < NQ; q++) {
    (out[q] += p.outward[m][q]), (back[q] += p.returning[m][q]), (all += p.outward[m][q] + p.returning[m][q]);
  }
  if (!(all > 0)) return null;
  return { out: out.map((v) => v / all), back: back.map((v) => v / all) };
}

/**
 * The selection's rows with its regional trips. `local`: the workers' rows; `cells`: the selection's
 * cells; `ends(m)`: a cell legs' four blocks at the selection's end (rail, walking, driving, No
 * route); `all` and `byPurpose`: the legs of all purposes and of each of `ROW_PURPOSES`; `day`: the
 * demand day shown. A purpose without its legs keeps its local rows.
 */
export function purposeRows(local: PurposeTrips[], cells: ArrayLike<number>, ends: (m: CellModes) => Float32Array[], all: CellModes,
  byPurpose: Map<PurposeKey, CellModes>, day: number): PurposeTrips[] {
  const rows = local.map((p) => ({ name: p.name, outward: p.outward.map((v) => v.slice()), returning: p.returning.map((v) => v.slice()) }));
  const whole = shape(rows);
  const counted: [number, number, number] = [0, 0, 0];
  const known = new Set(ROW_PURPOSES.flatMap(([, names]) => names));
  for (const p of rows) if (!known.has(p.name)) for (let k = 0; k < 3; k++) counted[k] += p.outward[k].concat(p.returning[k]).reduce((a, b) => a + b, 0);
  for (const [key, names] of ROW_PURPOSES) {
    const m = byPurpose.get(key);
    let mine = rows.filter((p) => names.includes(p.name));
    if (!m) {
      for (const p of mine) for (let k = 0; k < 3; k++) counted[k] += p.outward[k].concat(p.returning[k]).reduce((a, b) => a + b, 0);
      continue;
    }
    const target = legsOver(cells, ends(m));
    for (let k = 0; k < 3; k++) {
      counted[k] += target[k];
      let have = 0;
      for (const p of mine) for (let q = 0; q < NQ; q++) have += p.outward[k][q] + p.returning[k][q];
      if (have > 0) {
        const f = target[k] / have;
        for (const p of mine) for (let q = 0; q < NQ; q++) (p.outward[k][q] *= f), (p.returning[k][q] *= f);
        continue;
      }
      if (!(target[k] > 0)) continue;
      // none of this mode here before: the purpose's own timing, else the selection's
      const s = shape(mine) ?? whole;
      if (!s) continue;
      if (!mine.length) rows.push(...(mine = [{ name: names[0], outward: zeros(), returning: zeros() }]));
      for (let q = 0; q < NQ; q++) (mine[0].outward[k][q] += target[k] * s.out[q]), (mine[0].returning[k][q] += target[k] * s.back[q]);
    }
  }
  // the longer trips: all purposes' legs beyond every purpose's, departing at their own timing
  const total = legsOver(cells, ends(all));
  const longer = total.map((v, k) => Math.max(0, v - counted[k]));
  if (longer[0] + longer[1] + longer[2] >= 0.5) {
    const sh = PERIODS.map((_, q) => longerShare(day, q)), s = sh.reduce((a, b) => a + b, 0);
    if (s > 0) rows.push({ name: LONGER_ROW, outward: longer.map((v) => sh.map((x) => (v * x) / s)), returning: zeros() });
  }
  return rows;
}
