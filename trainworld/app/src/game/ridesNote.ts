// "+N daily rides" (T-1478): after a line's stops change and demand answers with the new figure, the
// map says how many riders a day the line gained (or, quietly, lost) for a few seconds
// (ui/RidesNotes.tsx). The figure is the one the line already has, `LineDemand.ridersPerDay`; nothing new is
// summed. Held on the main thread only and not saved.

import { effect, signal } from "@preact/signals";
import { demandView, lineFiguresIn, type DemandView } from "./demand";
import { world } from "./world";

/** Wall ms the note stays. */
export const NOTE_MS = 5000;
/** Wall ms the number takes to ease up to its figure. */
export const NOTE_COUNT_MS = 900;

export interface RidesNote {
  /** riders a day gained (negative: lost), whole */
  delta: number;
  /** the line's riders a day before and after, whole, for the hover */
  from: number;
  to: number;
  /** performance.now() when it landed */
  at: number;
}

/** The lines with a note showing, by id. */
export const ridesNotes = signal<ReadonlyMap<string, RidesNote>>(new Map());

/** The figure to show now: eased out from 0 (fast at first, slow to finish). */
export function shownDelta(n: RidesNote, now = performance.now()): number {
  const t = Math.min(1, (now - n.at) / NOTE_COUNT_MS);
  return Math.round(n.delta * (1 - (1 - t) ** 3));
}

/** The riders a day of a line whose figures are final and for these very stops, else null. */
function figureFor(dv: DemandView | null, id: string, stops: readonly string[]): number | null {
  const d = dv?.lines.get(id);
  if (!dv?.final || !d || d.estimate) return null;
  if (d.stops ? d.stops.length !== stops.length || d.stops.some((s, k) => s !== stops[k]) : !lineFiguresIn(dv, id, stops.length)) return null;
  return d.ridersPerDay;
}

/** Start watching for stop edits. Call once. */
export function startRidesNote() {
  /** each line's last known figure and the stops it was for */
  const base = new Map<string, { riders: number; stops: string }>();
  /** lines whose stops changed since, with the figure from before the first change */
  const pending = new Map<string, number>();
  let day = -1;
  const timers = new Map<string, number>();
  effect(() => {
    const w = world.value, dv = demandView.value;
    if (!w) { base.clear(); pending.clear(); return; }
    // another day's figures are another figure
    if (dv && dv.day !== day) { day = dv.day; base.clear(); pending.clear(); }
    const ids = new Set<string>();
    for (const l of w.save.lines) {
      ids.add(l.id);
      const key = l.stops.join(","), b = base.get(l.id);
      if (b && b.stops !== key && !pending.has(l.id)) pending.set(l.id, b.riders);
      const r = figureFor(dv, l.id, l.stops);
      if (r === null) continue;
      const from = pending.get(l.id);
      base.set(l.id, { riders: r, stops: key });
      if (from === undefined) continue;
      pending.delete(l.id);
      const delta = Math.round(r) - Math.round(from);
      if (!delta) continue;
      const m = new Map(ridesNotes.peek());
      m.set(l.id, { delta, from: Math.round(from), to: Math.round(r), at: performance.now() });
      ridesNotes.value = m;
      clearTimeout(timers.get(l.id));
      timers.set(l.id, setTimeout(() => {
        const left = new Map(ridesNotes.peek());
        left.delete(l.id);
        ridesNotes.value = left;
      }, NOTE_MS) as unknown as number);
    }
    for (const id of [...base.keys()]) if (!ids.has(id)) { base.delete(id); pending.delete(id); }
  });
}
