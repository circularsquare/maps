// T-1803: whether an edit's demand update holds the clock at DEMAND_CAP (game/lineFigures.ts). An edit
// is quiet when it cannot visibly change what runs now: a train count or headway changed at a level
// not in force on the line, or by less than half of what it was before the edits since demand last
// settled (from or to 0 is never small), or a stop edit on a line with no trains at any level. Every
// other edit is loud, and one loud edit holds the clock until demand settles.

import { signal } from "@preact/signals";
import type { EditOp } from "../workers/protocol";
import { demandAt, lineSeconds } from "./clock";
import { DEMAND_INDEX, type LineInput, type LineStats, type WorldView } from "./types";

/** Edits since demand last settled, all quiet: the update they start does not cap the clock */
export const quietEdits = signal(false);
/** bumped by every edit noted (lineFigures forgets a stale state after a wait) */
export const editStamp = signal(0);

let loud = false;
/** each line's trains and trains an hour per level as they were before its first edit since settling */
const baseline = new Map<number, { trains: readonly number[]; freq: readonly number[] }>();

/** Demand has settled (or nothing came of the edits): the next edit starts afresh. */
export function resetEdits() {
  loud = false;
  baseline.clear();
  quietEdits.value = false;
}

/** A change under half of the old value; from 0 or to 0 it is never small. */
const small = (was: number, now: number) => was > 0 && now > 0 && Math.abs(now - was) < 0.5 * was;
const same = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a));

function baselineOf(l: LineInput, st: LineStats) {
  let b = baseline.get(l.num);
  if (!b) {
    const freq = st.autoS ? st.autoS.map((s) => (s > 0 ? 3600 / s : 0)) : [l.tph.high, l.tph.medium, l.tph.low];
    baseline.set(l.num, (b = { trains: [...st.trainsNeeded], freq }));
  }
  return b;
}

/** Whether `op`, made on world `w`, leaves the clock alone. */
function isQuiet(op: EditOp, w: WorldView): boolean {
  switch (op.op) {
    case "setTrainCount":
    case "setHeadway": {
      const l = w.save.lines.find((x) => x.num === op.line);
      const st = l && w.lineStats[l.id];
      if (!l || !st) return false;
      if (op.op === "setHeadway" && !op.seconds) return true; // back to counts: no train comes or goes
      // a line that is not running carries nobody
      if (st.status !== "running") return true;
      const b = baselineOf(l, st), inForce = DEMAND_INDEX[demandAt(lineSeconds(st))];
      for (let lev = 0; lev < 3; lev++) {
        const was = op.op === "setTrainCount" ? b.trains[lev] : b.freq[lev];
        const now = op.op === "setTrainCount" ? op.trains[lev] : op.seconds![lev] > 0 ? 3600 / op.seconds![lev] : 0;
        if (same(was, now) || lev !== inForce) continue;
        if (!small(was, now)) return false;
      }
      return true;
    }
    case "setStops":
    case "insertStop":
    case "dropStop":
    case "endLineAt":
    case "stopPassed": {
      const l = w.save.lines.find((x) => x.num === op.line);
      if (!l) return false;
      const st = w.lineStats[l.id];
      return l.tph.high === 0 && l.tph.medium === 0 && l.tph.low === 0 && (st?.trainsNeeded ?? []).every((n) => n === 0);
    }
    default:
      return false;
  }
}

/** Ops that change nothing demand reads. */
const NEUTRAL = new Set(["lineLook", "lineGroup", "groupName", "borrow", "repay"]);

/** The player made an edit: called before it is sent. */
export function noteEdit(op: EditOp, w: WorldView | null) {
  if (NEUTRAL.has(op.op)) return;
  if (!w || !isQuiet(op, w)) loud = true;
  quietEdits.value = !loud;
  editStamp.value++;
}
