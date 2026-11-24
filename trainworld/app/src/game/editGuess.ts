// T-4170: what an edit shows at its click, read from its op and what the page holds (the drawn network, the world, the
// line tool's hover read), one part per kind of edit as the worker's word has its parts (workers/clockWord.ts).
// game/pending.ts asks it once for every edit sent, so every caller of `edit()` gets it the same way; a caller with more
// to show than its op says (the station tool's ghost, a price the panel quoted) gives that to `ahead()`, which joins the
// same entry.

import type { EditOp } from "../workers/protocol";
import { mergeShow, type Show } from "./pending";
import { pieceGuess } from "./pieceAhead";
import { linkGuess } from "./links";
import { stationGuess } from "../map/stationAhead";
import { lineGuess } from "../map/lineToolAhead";

const KINDS: ((op: EditOp) => Show | null)[] = [lineGuess, stationGuess, pieceGuess, linkGuess];

/** Every kind's part for `op` joined, or null when none has anything (a kind that throws gives none, and says so). */
export function editGuess(op: EditOp): Show | null {
  let out: Show | null = null;
  for (const k of KINDS) {
    let part: Show | null = null;
    try { part = k(op); } catch (err) { console.error("T-4170: an edit's guess part failed", op.op, err); }
    if (part) out = out ? mergeShow(out, part) : part;
  }
  return out;
}
