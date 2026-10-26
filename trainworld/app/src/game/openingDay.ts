// Opening day (T-1434): a line that starts running for the first time in this page's session has
// its first game day marked. Its stations get a small bunting under their names (map/stationLabels.ts),
// its row in the Lines tab counts the riders up from 0 once its first figure is shown, and the
// "bung" plays (map/buildSound.ts). Held on the main thread only and not saved: a loaded game shows
// no opening days.

import { signal } from "@preact/signals";
import { clock } from "./clock";
import { linesById, world } from "./world";
import { today } from "./today";
import type { WorldView } from "./types";

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

const listeners: ((ids: string[]) => void)[] = [];
/** Told of the lines that just opened (a construct that opens several tells once). */
export function onLinesOpened(f: (ids: string[]) => void) {
  listeners.push(f);
}

/** The riders figure to show for a line on its opening day: the figure eased up from 0 (fast at first,
 * slow to finish) once it has landed. Null when the row is plain. */
export function countingRiders(o: Opening | undefined, riders: number, now = performance.now()): number | null {
  if (!o) return null;
  if (!o.landed) return 0;
  const t = (now - o.landed) / COUNT_MS;
  if (t >= 1) return null;
  return Math.round(riders * (1 - (1 - t) ** 3));
}

const BUNTING = 7;
/** For each station of a line on its opening day, the CSS background its bunting is painted with: the
 * lines' colours, one per pennant in turn. */
export function ribbonColours(w: WorldView): ReadonlyMap<string, string> | null {
  const open = openings.peek();
  if (!open.size) return null;
  const by = new Map<string, string[]>();
  const at = linesById(w);
  for (const id of open.keys()) {
    const l = at.get(id);
    if (!l) continue;
    for (const s of l.stops) {
      const c = by.get(s);
      if (!c) by.set(s, [l.colour]);
      else if (!c.includes(l.colour) && c.length < 3) c.push(l.colour);
    }
  }
  const out = new Map<string, string>();
  for (const [s, c] of by) {
    out.set(s, c.length === 1 ? `linear-gradient(${c[0]},${c[0]})` : `repeating-linear-gradient(90deg,${c.map((x, i) => `${x} ${i * BUNTING}px ${(i + 1) * BUNTING}px`).join(",")})`);
  }
  return out;
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
    for (const f of listeners) f(opened);
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
