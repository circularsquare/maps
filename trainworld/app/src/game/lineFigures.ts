// T-1665: what holds back while demand is worked out. The clock runs no faster than DEMAND_CAP while a
// solve or a load is going, and a line the player just rescheduled or built is "constructing..." where
// its riders show (Lines tab, line panel) until the result that counts it lands (T-1733: its
// cluster's local result, or the solve's end).

import { effect, signal } from "@preact/signals";
import { clock } from "./clock";
import { demandUpdating, refiningZone } from "./demand";
import { demandStopped, loadProgress } from "./loading";
import { world } from "./world";

/** Lines whose riders are being worked out again after an edit. */
export const linesPending = signal<ReadonlySet<string>>(new Set());

/** T-1733: these lines' riders have landed (a cluster's result, local or whole) */
export function linesSettled(ids: Iterable<string>) {
  const set = new Set(linesPending.peek());
  let changed = false;
  for (const id of ids) changed = set.delete(id) || changed;
  if (changed) linesPending.value = set;
}

/** Wall ms to wait for a solve to start before deciding the edit needs none (a line with no riders, or
 * demand idle), so the label never sticks. */
const NO_SOLVE_MS = 8000;

export function startLineFigures() {
  effect(() => {
    // T-1790: demand that stopped will finish nothing, so it holds nothing back
    clock.setCapped(!demandStopped.value && (demandUpdating.value || !!loadProgress.value || !!refiningZone.value));
  });

  /** each line's last signature, null before the first world */
  let prev: Map<string, string> | null = null;
  world.subscribe((w) => {
    if (!w) { prev = null; return; }
    const next = new Map<string, string>();
    const changed: string[] = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      const sig = JSON.stringify([l.stops, l.tph, st?.status, st?.cars, st?.fast]);
      next.set(l.id, sig);
      if (prev && prev.get(l.id) !== sig && st?.status === "running") changed.push(l.id);
    }
    const loading = clock.held.peek();
    prev = next;
    if (loading || !changed.length) return;
    const set = new Set(linesPending.peek());
    for (const id of changed) set.add(id);
    linesPending.value = set;
    setTimeout(() => {
      if (!demandUpdating.peek() && linesPending.peek().size) linesPending.value = new Set();
    }, NO_SOLVE_MS);
  });

  let wasUpdating = demandUpdating.peek();
  demandUpdating.subscribe((on) => {
    if (wasUpdating && !on && linesPending.peek().size) linesPending.value = new Set();
    wasUpdating = on;
  });
}
