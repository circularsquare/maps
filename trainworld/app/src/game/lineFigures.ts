// T-1665: what holds back while demand is worked out. The clock runs no faster than DEMAND_CAP while a
// solve or a load is going, and a line the player just changed is pending (Lines tab, line panel) until
// the result that counts it lands (T-1733: its cluster's local result, or the solve's end).

import { effect, signal } from "@preact/signals";
import { clock } from "./clock";
import { editStamp, quietEdits, resetEdits } from "./editCap";
import { demandUpdating, refiningZone, viewRecomputing } from "./demand";
import { demandStopped, loadProgress } from "./loading";
import { world } from "./world";

/** T-2392: "build" for a line that has just started running (its blueprint constructed, or new);
 * "change" for one already running whose stops or service changed */
export type PendingKind = "build" | "change";

/** Lines whose riders are being worked out again after an edit, and why. */
export const linesPending = signal<ReadonlyMap<string, PendingKind>>(new Map());

/** The label over a pending line's riders. */
export const pendingLabel = (k: PendingKind) => (k === "build" ? "constructing..." : "loading...");

/** T-1733: these lines' riders have landed (a cluster's result, local or whole) */
export function linesSettled(ids: Iterable<string>) {
  const set = new Map(linesPending.peek());
  let changed = false;
  for (const id of ids) changed = set.delete(id) || changed;
  if (changed) linesPending.value = set;
}

/** Wall ms to wait for a solve to start before deciding the edit needs none (a line with no riders, or
 * demand idle), so the label never sticks. */
const NO_SOLVE_MS = 8000;
/** Wall ms after an edit with no demand update under way, when the edits noted are forgotten */
const FORGET_EDITS_MS = 30000;

export function startLineFigures() {
  effect(() => {
    // T-1790: demand that stopped will finish nothing, so it holds nothing back
    // T-1803: an update that only quiet edits started (game/editCap.ts) caps nothing
    // T-1851: the same rule as the status tag: lifted once the zone in view is final, other zones still solving
    // T-2409: but held anywhere while a new line has no riders yet ("constructing...": it costs and earns nothing)
    const missing = demandUpdating.value && [...linesPending.value.values()].includes("build");
    clock.setCapped(!demandStopped.value && ((demandUpdating.value && !quietEdits.value && viewRecomputing.value) || missing || !!loadProgress.value || !!refiningZone.value));
  });
  // edits that came to nothing (refused, or no solve followed) are forgotten after a while
  effect(() => {
    const stamp = editStamp.value;
    const t = setTimeout(() => {
      if (editStamp.peek() === stamp && !demandUpdating.peek()) resetEdits();
    }, FORGET_EDITS_MS);
    return () => clearTimeout(t);
  });

  /** each line's last signature and status, null before the first world */
  let prev: Map<string, { sig: string; running: boolean }> | null = null;
  world.subscribe((w) => {
    if (!w) { prev = null; return; }
    const next = new Map<string, { sig: string; running: boolean }>();
    const changed: [string, PendingKind][] = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      const running = st?.status === "running";
      const sig = JSON.stringify([l.stops, l.tph, st?.status, st?.cars, st?.fast]);
      next.set(l.id, { sig, running });
      const was = prev?.get(l.id);
      if (prev && was?.sig !== sig && running) changed.push([l.id, was?.running ? "change" : "build"]);
    }
    const loading = clock.held.peek();
    prev = next;
    if (loading || !changed.length) return;
    const set = new Map(linesPending.peek());
    // a line still constructing stays so until its riders land
    for (const [id, k] of changed) if (set.get(id) !== "build") set.set(id, k);
    linesPending.value = set;
    setTimeout(() => {
      if (!demandUpdating.peek() && linesPending.peek().size) linesPending.value = new Map();
    }, NO_SOLVE_MS);
  });

  let wasUpdating = demandUpdating.peek();
  demandUpdating.subscribe((on) => {
    if (wasUpdating && !on) resetEdits();
    if (wasUpdating && !on && linesPending.peek().size) linesPending.value = new Map();
    wasUpdating = on;
  });
}
