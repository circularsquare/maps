// The game clock on the main thread: game seconds (float64), anchored to wall time, frozen while
// paused or hidden. The renderer reads `now()` every frame; the UI reads `minute`, a signal that
// changes once per game minute, so the top bar re-renders a text node and nothing else.
//
// With T-025 the clock worker keeps the authoritative time; this copy stays for drawing.

import { computed, signal, type Signal } from "@preact/signals";
import { PERIODS, periodAt } from "./periods";
import type { Demand } from "./types";

/** Game seconds per wall second for the three speed buttons. Provisional. */
export const SPEEDS = [60, 240, 960] as const;

/** The demand level scheduled at game time `t`: the period table shared with the track model and
 * demand (game/periods.ts, sim/src/track/params.rs). */
export function demandAt(t: number): Demand {
  return PERIODS[periodAt(t)].demand;
}

class GameClock {
  private anchorSim: number;
  private anchorWall = performance.now();
  private hidden = document.visibilityState === "hidden";
  /** speed button index into SPEEDS */
  readonly speed = signal(0);
  readonly running = signal(true);
  /** absolute game minute, for display */
  readonly minute: Signal<number>;
  readonly demand = computed(() => demandAt(this.minute.value * 60));
  private listeners: (() => void)[] = [];

  constructor(start: number) {
    this.anchorSim = start;
    this.minute = signal(Math.floor(start / 60));
  }

  /** game seconds now */
  now(): number {
    if (!this.running.peek() || this.hidden) return this.anchorSim;
    return this.anchorSim + ((performance.now() - this.anchorWall) / 1000) * SPEEDS[this.speed.peek()];
  }

  /** Is time moving (so frames are needed)? */
  get ticking() {
    return this.running.peek() && !this.hidden;
  }

  private reanchor() {
    this.anchorSim = this.now();
    this.anchorWall = performance.now();
  }

  private changed() {
    this.tick();
    for (const f of this.listeners) f();
  }

  /** Called once per frame by the renderer: moves the display minute along. */
  tick() {
    const m = Math.floor(this.now() / 60);
    if (m !== this.minute.peek()) this.minute.value = m;
  }

  onChange(f: () => void) {
    this.listeners.push(f);
  }

  setRunning(on: boolean) {
    this.reanchor();
    this.running.value = on;
    this.changed();
  }

  setSpeed(i: number) {
    this.reanchor();
    this.speed.value = i;
    this.running.value = true;
    this.changed();
  }

  set(t: number) {
    this.anchorSim = t;
    this.anchorWall = performance.now();
    this.changed();
  }

  setHidden(h: boolean) {
    this.reanchor(); // freeze at the moment it was hidden
    this.hidden = h;
    this.reanchor();
    this.changed();
  }
}

export let clock: GameClock;
export function startClock(t: number) {
  clock = new GameClock(t);
  document.addEventListener("visibilitychange", () => clock.setHidden(document.visibilityState === "hidden"));
  return clock;
}
