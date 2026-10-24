// The game clock on the main thread: game seconds (float64), anchored to wall time, frozen while
// paused, or hidden unless "Run in background" is on (T-826). The renderer reads `now()` every frame; the UI reads `minute`, a signal that
// changes once per game minute, so the top bar re-renders a text node and nothing else. The clock
// keeps world time; `local` is the minute in the local time in view (T-309, game/localTime.ts).
//
// With T-025 the clock worker keeps the authoritative time; this copy stays for drawing.

import { computed, effect, signal, type Signal } from "@preact/signals";
import { PERIODS, periodAt } from "./periods";
import { currentDay } from "./days";
import { viewShift } from "./localTime";
import { viewCentre } from "./listScope";
import { countryAt, kindIn } from "./holidays";
import type { Demand } from "./types";

/** Game seconds per wall second for the three speed buttons. Provisional. */
export const SPEEDS = [60, 240, 960] as const;

/** A line's local time of day, seconds: the game hour moved by the line's time shift (T-309). */
export const lineSeconds = (st: { shift?: number } | undefined): number => clock.hour.value * 3600 + (st?.shift ?? 0);

/** The demand level scheduled at game time `t`: the period table shared with the track model and
 * demand (game/periods.ts, sim/src/track/params.rs). */
export function demandAt(t: number): Demand {
  return PERIODS[periodAt(t)].demand;
}

class GameClock {
  private anchorSim: number;
  private anchorWall = performance.now();
  private hidden = document.visibilityState === "hidden";
  /** T-826: time moves while hidden. The clock is wall-anchored and the worker settles money on its
   * own timers, so nothing here needs a tick while hidden; the minute catches up on the next frame. */
  private background = false;
  /** Frozen while a loaded game's demand is still being calculated (T-149, game/loading.ts):
   * separate from `running`, so the player's pause and speed survive it. */
  readonly held = signal(false);
  /** speed button index into SPEEDS */
  readonly speed = signal(0);
  readonly running = signal(true);
  /** absolute game minute of world time (game/localTime.ts) */
  readonly minute: Signal<number>;
  /** T-309: the game minute in local time where the player is looking, for the top bar, the
   * demand level and every "now" a view shows */
  readonly local = computed(() => this.minute.value + viewShift.value / 60);
  readonly demand = computed(() => demandAt(this.local.value * 60));
  /** the game hour of world time: periods and offsets are whole hours, so a line's level in its
   * own local time changes only with it (T-309) */
  readonly hour = computed(() => Math.floor(this.minute.value / 60));
  private listeners: (() => void)[] = [];

  constructor(start: number) {
    this.anchorSim = start;
    this.minute = signal(Math.floor(start / 60));
    // T-266, T-309: the demand views on "Today" follow the local day in view into and out of a
    // weekend, as the minutes pass or the view pans into another time zone; T-872: a holiday in
    // the view's country is a weekend day
    effect(() => {
      const c = viewCentre.value;
      const k = kindIn(c ? countryAt(c[0], c[1]) : null, Math.floor(this.local.value / 1440));
      if (k !== currentDay.peek()) currentDay.value = k;
    });
  }

  /** game seconds now */
  now(): number {
    if (!this.ticking) return this.anchorSim;
    return this.anchorSim + ((performance.now() - this.anchorWall) / 1000) * SPEEDS[this.speed.peek()];
  }

  /** Is time moving (so frames are needed)? */
  get ticking() {
    return this.running.peek() && (!this.hidden || this.background) && !this.held.peek();
  }

  /** Is the page hidden (nothing is drawn, time moves only with "Run in background")? */
  get isHidden() {
    return this.hidden;
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

  setBackground(on: boolean) {
    if (on === this.background) return;
    this.reanchor();
    this.background = on;
    this.reanchor();
    this.changed();
  }

  /** Hold or release the clock for loading (T-149). */
  setHeld(h: boolean) {
    if (h === this.held.peek()) return;
    this.reanchor();
    this.held.value = h;
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
