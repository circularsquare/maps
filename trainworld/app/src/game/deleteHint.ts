// T-3026: after four removals in a row with the Delete tool, a hint toast points at the Select tool.
// "In a row": four removals the Delete tool made, with no other edit and no other tool picked between them.
// Shown once ever in a browser (localStorage), and not while the tutorial card is up. The toast itself is
// ui/Achievements.tsx's, which registers `setHintShower`; this file stays light so game/world.ts can call it.

import { effect } from "@preact/signals";
import { guideVerbose, tool } from "./ui";

const ROW = 4;
const STORE = "anitabuilder-hint-select";
const seenBefore = () => { try { return localStorage.getItem(STORE) === "1"; } catch { return false; } };
const markSeen = () => { try { localStorage.setItem(STORE, "1"); } catch { /* this page load only */ } };

// T-3051: the same for height. Four height changes in a row, each to one piece picked alone, point at the
// Select tool's box drag (a hold then drag on touch), which picks many pieces to re-height at once
// (blueprintPick.ts `levelPicked`). Its own once-only flag.
const HEIGHT_STORE = "anitabuilder-hint-height";
const heightSeenBefore = () => { try { return localStorage.getItem(HEIGHT_STORE) === "1"; } catch { return false; } };

export type Hint = "delete" | "height";

let run = 0, heightRun = 0, shower: ((h: Hint) => void) | null = null;
let done = seenBefore(), heightDone = heightSeenBefore();

export function setHintShower(f: (h: Hint) => void) {
  shower = f;
}

/** Any edit but a removal or a height change ends that row (world.ts `edit`; the removal or the change counts itself after). */
export function breakDeleteRun() {
  run = 0;
  heightRun = 0;
}
effect(() => { if (tool.value !== "delete") run = 0; });

/** The row length before a removal's edit goes out, which breaks it. */
export const deleteRun = () => run;

/** A removal by the Delete tool landed; `before` is `deleteRun()` from just before its edit. */
export function noteDelete(before: number) {
  if (done || tool.peek() !== "delete") { run = 0; return; }
  run = before + 1;
  if (run < ROW || guideVerbose.peek()) return;
  done = true;
  run = 0;
  markSeen();
  shower?.("delete");
}

/** The height row's length before a height change's edit goes out, which breaks it. */
export const heightRunLength = () => heightRun;

/** A height change landed on `pieces` pieces; `before` is `heightRunLength()` from just before its edit. Several
 * pieces at once is the Select tool's job already, so it ends the row. */
export function noteHeight(before: number, pieces: number) {
  if (heightDone || pieces !== 1) { heightRun = 0; return; }
  heightRun = before + 1;
  if (heightRun < ROW || guideVerbose.peek()) return;
  heightDone = true;
  heightRun = 0;
  try { localStorage.setItem(HEIGHT_STORE, "1"); } catch { /* this page load only */ }
  shower?.("height");
}
