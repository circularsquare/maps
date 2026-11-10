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

let run = 0, shower: (() => void) | null = null;
let done = seenBefore();

export function setHintShower(f: () => void) {
  shower = f;
}

/** Any edit but a removal, or another tool picked, ends the row (world.ts `edit`; the removal counts itself after). */
export function breakDeleteRun() {
  run = 0;
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
  shower?.();
}
