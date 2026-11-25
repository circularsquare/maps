// T-3599: an edit's solve's time left, for the calculating trips hover. Pure, so a Node check can replay a
// recorded solve against it (checks/t3599-replay.mjs).
//
// Each solving worker plans its units (a period of a local solve, a round of a period of a cluster solved
// whole) with an expected time from its populations' last passes (demand.worker.ts `passPrior`), and says
// with each unit begun or ended how much expected time is left and how this solve's units ran against theirs.
// A worker's time left is that expected time at this solve's pace, stage by stage (a local solve runs far
// under its passes' time) and the whole solve's period by period; the solve's, the slowest worker's, plus
// the page's wait between the local and the whole solve and the stretch after the last unit until the
// figures show.

import type { WorkPart } from "./demand";

/** how far this solve's own pace counts against the prior one: as if the prior had run this many units
 * (after a load the passes of a period can be off by ten times, so a unit's own pace soon counts most) */
export const PACE_WEIGHT_UNITS = 1;
/** a pace outside this is a fluke, never the solve's */
const PACE_MIN = 0.01, PACE_MAX = 100;

export interface EditLeftIn {
  /** each stage's pace (ms run over ms expected) in the last edit's solves here */
  pace: [number, number];
  /** ms from a solve's last unit to its figures showing, as last measured */
  tailMs: number;
  /** ms from the local stage's last unit to the whole solve's first, as last measured */
  gapMs: number;
}

/** the time left, ms, at `now`, or null when it can't be told yet: to the solve's final, or (`view`) to the
 * zone in view's figures (where no worker has units before those, the final's) */
export function editLeft(parts: readonly WorkPart[], now: number, prior: EditLeftIn, part: "all" | "view" = "all"): number | null {
  return editLeftOf(parts, now, prior, part)?.ms ?? null;
}

/** T-4264 (Anita, 11-25: a bar past about half only on a time left we can bound): `editLeft`'s time left, and whether
 * this solve's own units bound it. They do when every unit still to run is timed at this solve's pace (a unit of its
 * stage has ended here) and none in hand has run past its time at that pace; not before (the passes' prior, a time of
 * minutes that turned into 8: T-4209), nor past an overrun (its time is then a floor, a fifth of its time so far; T-4359:
 * the wait for the next unit too, once longer than what is left), nor once only the stretch after the last unit is left. */
export function editLeftOf(parts: readonly WorkPart[], now: number, { pace, tailMs, gapMs }: EditLeftIn, part: "all" | "view" = "all"): { ms: number; bounded: boolean } | null {
  const sum = (v: [number, number]) => v[0] + v[1];
  if (part === "view" && !parts.some((p) => sum(p.viewLeft) > 0 || sum(p.viewBlind) > 0 || p.curView)) {
    // (the view's units all ended: its figures are on their way)
    if (parts.some((p) => p.done[0] + p.done[1] > 0)) return { ms: tailMs, bounded: false };
    part = "all";
  }
  let bounded = true, any = false;
  const view = part === "view";
  let worst = 0;
  // (a pace from `ran` over `exp` ms of `n` units, leaning on `prior` as if it had run PACE_WEIGHT_UNITS)
  const blend = (ran: number, exp: number, n: number, prior: number) => {
    const w = n > 0 ? (PACE_WEIGHT_UNITS * exp) / n : 0;
    return Math.min(PACE_MAX, Math.max(PACE_MIN, exp > 0 ? (ran + w * prior) / (exp + w) : prior));
  };
  for (const p of parts) {
    const k = [0, 1].map((s) => blend(p.ran[s], p.ranExp[s], p.ranN?.[s] ?? 1, pace[s]));
    // the whole solve's periods each at their own pace, leaning on the worker's
    const kq = (q: number) => blend(p.ranQ[q] ?? 0, p.expQ[q] ?? 0, p.nQ?.[q] ?? 1, k[1]);
    const left = view ? p.viewLeft : p.left, blind = view ? p.viewBlind : p.blind;
    const inHand = p.busy && (!view || p.curView);
    // the unit in hand: what is left of it at this pace, and once over, a fifth of its time so far again
    let hand = 0;
    if (inHand) {
      const el = Math.max(0, now - p.at);
      // (a unit with no expected time: its size at this solve's ms a population, none before a unit has ended)
      const rate = (p.curLate ? p.sizeMsLate?.[p.stage] || 0 : 0) || (p.sizeMs?.[p.stage] ?? 0);
      if (p.cur === null && !rate) return null;
      const exp = p.cur === null ? p.curSize * rate : (p.stage === 1 && p.curQ >= 0 ? kq(p.curQ) : k[p.stage]) * p.cur;
      hand = exp > 0 ? Math.max(exp - el, 0.2 * el) : 0;
      any = true;
      // (bounded while the unit is within its time at this solve's own pace, no prior: a pace half prior after one
      // unit ran the bar into a hold 10 s early on the real save)
      const ownK = (n: number, ran: number, exp: number) => (n > 0 && exp > 0 ? ran / exp : 0);
      const own = p.cur === null ? exp : (p.stage === 1 && p.curQ >= 0 && (p.nQ?.[p.curQ] ?? 0) > 0 ? ownK(p.nQ[p.curQ], p.ranQ[p.curQ] ?? 0, p.expQ[p.curQ] ?? 0) : ownK(p.ranN?.[p.stage] ?? 0, p.ran[p.stage], p.ranExp[p.stage])) * p.cur;
      if (!(own > 0 && el <= own)) bounded = false;
    }
    // the whole solve's units left by period, less the one in hand; and the pace of what is left, by its weight
    let byQ = 0, msQ = 0;
    (p.leftQ ?? []).forEach((ms, q) => {
      const x = Math.max(0, ms - (p.busy && p.curQ === q && p.cur !== null ? p.cur : 0));
      byQ += x;
      msQ += kq(q) * x;
    });
    const all1 = Math.max(0, p.left[1] - (p.busy && p.stage === 1 && p.cur !== null ? p.cur : 0));
    const k1 = all1 > 0 ? (msQ + k[1] * Math.max(0, all1 - byQ)) / all1 : k[1];
    let rest = 0, blindMs = 0;
    for (const s of [0, 1] as const) {
      const mine = inHand && p.stage === s;
      const ahead = Math.max(0, left[s] - (mine && p.cur !== null ? p.cur : 0));
      rest += (s === 1 ? k1 : k[0]) * ahead;
      if (ahead > 0) { any = true; if (!(p.ranN?.[s] ?? 0)) bounded = false; }
      // (`blind` is the size of the units with no expected time, timed as the one in hand is)
      // (the crowding rounds after the first by their own pace once one of their units has ended, T-4343)
      const lateAll = (view ? p.viewBlindLate : p.blindLate)?.[s] ?? 0;
      const inLate = mine && p.cur === null && p.curLate;
      const b = blind[s] - (mine && p.cur === null ? p.curSize : 0);
      const bLate = Math.min(b, Math.max(0, lateAll - (inLate ? p.curSize : 0)));
      if (b > 0) {
        const rate = p.sizeMs?.[s] ?? 0;
        if (!rate) return null;
        blindMs += (b - bLate) * rate + bLate * (p.sizeMsLate?.[s] || rate);
        any = true;
      }
    }
    // the whole solve waits a moment after the local results show, for the next edit
    const gap = !view && p.left[1] + p.blind[1] > 0 && (p.left[0] + p.blind[0] > 0 || (p.busy && p.stage === 0)) ? gapMs : 0;
    // T-4359: no unit in hand while units are still to run: the worker is between them (a cluster's set-up, the market's),
    // time no unit counts. Once that wait is longer than what is left (and the page's gap after a local stage), the next
    // unit is late: a fifth of the wait so far again and no bound, as for a unit in hand past its time (else the count
    // reaches 0 with units still to run, and the bar fills)
    let late = 0;
    if (!p.busy && rest + blindMs > 0) {
      const idle = Math.max(0, now - p.at);
      if (idle > rest + blindMs + (!view && p.stage === 0 ? gapMs : 0)) { bounded = false; late = 0.2 * idle; }
    }
    worst = Math.max(worst, hand + rest + blindMs + gap + late);
  }
  return { ms: worst + tailMs, bounded: bounded && any };
}

/** each stage's pace in this solve over every worker, or null for a stage too little ran in to tell */
export function solvePace(parts: readonly WorkPart[]): [number | null, number | null] {
  return [0, 1].map((s) => {
    let ran = 0, exp = 0;
    for (const p of parts) { ran += p.ran[s]; exp += p.ranExp[s]; }
    return exp >= 2000 ? Math.min(PACE_MAX, Math.max(PACE_MIN, ran / exp)) : null;
  }) as [number | null, number | null];
}

/** before any edit's solve has been timed here: each stage's pace (notes/T-3599.md: a local solve ran at
 * a tenth of its passes' time on the real save, the whole solve at 0.8), the tail (2 s on the real save,
 * 15 s on the Asia one) and the gap (the page's 8 s wait, `EDIT_SETTLE_MS`, and the whole solve's setup), ms */
export const PACE_GUESS: [number, number] = [0.2, 1];
export const TAIL_GUESS_MS = 5000;
export const GAP_GUESS_MS = 9000;
/** the last edit's solves here: their paces, tail and gap, the next one's prior */
export const editPrior: EditLeftIn = { pace: [...PACE_GUESS], tailMs: TAIL_GUESS_MS, gapMs: GAP_GUESS_MS };
/** an edit's solve's final shown, `tailMs` after its last unit ended; `gapMs` from its local stage's end to
 * its whole solve's start, null when it had no local stage */
export function noteEditSolve(parts: readonly WorkPart[], tailMs: number, gapMs: number | null) {
  const p = solvePace(parts);
  // (a whole solve leaves fresh passes behind, so a pace far off came of passes it has replaced: the next
  // solve's lean on it only within these)
  for (const s of [0, 1] as const) { const v = p[s]; if (v !== null) editPrior.pace[s] = s === 0 ? Math.min(2, Math.max(0.05, v)) : Math.min(1.25, Math.max(0.8, v)); }
  if (tailMs >= 0) editPrior.tailMs = tailMs;
  if (gapMs !== null && gapMs >= 0) editPrior.gapMs = gapMs;
}
