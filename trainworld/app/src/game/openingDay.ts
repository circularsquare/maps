// Opening day (T-1434): a line that starts running for the first time in this page's session has
// its first game day marked: its row in the Lines tab counts the riders up from 0 once its first
// figure is shown. Held on the main thread only and not saved: a loaded game shows no opening days.

import { signal } from "@preact/signals";
import { clock } from "./clock";
import { world } from "./world";
import { today } from "./today";

/** One game day, s. */
const DAY_S = 86400;
/** Wall ms the riders count takes. */
export const COUNT_MS = 2400;

export interface Opening {
  /** game seconds when it started running */
  at: number;
  /** performance.now() when its riders first showed above 0, 0 before */
  landed: number;
}

/** The lines on their opening day, by id. */
export const openings = signal<ReadonlyMap<string, Opening>>(new Map());

/** The riders figure to show for a line on its opening day: the figure eased up from 0 (fast at first,
 * slow to finish) once it has landed. Null when the row is plain. */
export function countingRiders(o: Opening | undefined, riders: number, now = performance.now()): number | null {
  if (!o) return null;
  if (!o.landed) return 0;
  const t = (now - o.landed) / COUNT_MS;
  if (t >= 1) return null;
  return Math.round(riders * (1 - (1 - t) ** 3));
}

/** Start watching for lines that open. Call once, after the clock exists. */
export function startOpeningDay() {
  /** each line in the last world, with whether it runs; null before the first */
  let prev: Map<string, string> | null = null;
  /** lines that have run in this session: a repaired line does not open again */
  const ran = new Set<string>();
  world.subscribe((w) => {
    if (!w) { prev = null; return; }
    const next = new Map<string, string>(), opened: string[] = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      // it opens when it first runs with trains, which a line over built track can do at once
      const s = st?.status === "running" && st.trainsNeeded.some((n) => n > 0) ? "running" : "";
      next.set(l.id, s);
      if (s !== "running" || ran.has(l.id)) continue;
      ran.add(l.id);
      if (prev && prev.has(l.id)) opened.push(l.id);
    }
    prev = next;
    if (!opened.length) return;
    const m = new Map(openings.peek()), at = clock.now();
    for (const id of opened) m.set(id, { at, landed: 0 });
    openings.value = m;
  });
  // the first figure above 0 starts the count
  today.subscribe((t) => {
    const open = openings.peek();
    let m: Map<string, Opening> | null = null;
    for (const [id, o] of open) {
      if (!o.landed && (t.get(id)?.riders ?? 0) > 0) (m ??= new Map(open)).set(id, { ...o, landed: performance.now() });
    }
    if (m) openings.value = m;
  });
  // the day ends: a game minute is the clock's finest step
  clock.minute.subscribe(() => {
    const open = openings.peek(), now = clock.now();
    let m: Map<string, Opening> | null = null;
    for (const [id, o] of open) if (now - o.at >= DAY_S) (m ??= new Map(open)).delete(id);
    if (m) openings.value = m;
  });
}
