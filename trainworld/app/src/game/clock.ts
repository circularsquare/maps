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
/** T-1664: the slider's stops, in times normal speed (log-spaced to the eye) */
export const SLIDER_STOPS = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 64, 80, 100] as const;
/** T-1665: the fastest the clock runs while demand is still being worked out, times normal speed */
export const DEMAND_CAP = 4;

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
  /** speed button index into SPEEDS; a slider speed (below) keeps the index of the nearest button below it */
  readonly speed = signal(0);
  /** T-1664: a speed from the slider, in times normal speed, until a speed button is pressed; null otherwise */
  readonly custom = signal<number | null>(null);
  /** T-1665: demand is still being worked out, so the clock runs no faster than DEMAND_CAP */
  readonly capped = signal(false);
  /** the speed the player chose, game seconds per wall second */
  readonly chosenRate = computed(() => (this.custom.value ?? SPEEDS[this.speed.value] / SPEEDS[0]) * SPEEDS[0]);
  /** the speed the clock runs at: the chosen one, or the cap while it applies */
  readonly rateNow = computed(() => (this.capped.value ? Math.min(this.chosenRate.value, DEMAND_CAP * SPEEDS[0]) : this.chosenRate.value));
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
    // weekend, as the minutes pass or the view pans into another time zone; T-872, T-262: the view's
    // country's holidays and weekend decide the kind
    effect(() => {
      const c = viewCentre.value;
      const k = kindIn(c ? countryAt(c[0], c[1]) : null, Math.floor(this.local.value / 1440));
      if (k !== currentDay.peek()) currentDay.value = k;
    });
  }

  /** game seconds now */
  now(): number {
    if (!this.ticking) return this.anchorSim;
    return this.anchorSim + ((performance.now() - this.anchorWall) / 1000) * this.rateNow.peek();
  }

  /** game seconds per wall second now */
  rate(): number {
    return this.rateNow.peek();
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

  /** returns the unsubscribe */
  onChange(f: () => void) {
    this.listeners.push(f);
    return () => { this.listeners = this.listeners.filter((g) => g !== f); };
  }

  setRunning(on: boolean) {
    this.reanchor();
    this.running.value = on;
    this.changed();
  }

  setSpeed(i: number) {
    this.reanchor();
    this.custom.value = null;
    this.speed.value = i;
    this.running.value = true;
    this.changed();
  }

  /** T-1664: run at `x` times normal speed (the slider's); a button's own speed picks that button */
  setMultiple(x: number) {
    const i = SPEEDS.findIndex((s) => s === x * SPEEDS[0]);
    if (i >= 0) return this.setSpeed(i);
    this.reanchor();
    this.custom.value = x;
    this.speed.value = x < SPEEDS[1] / SPEEDS[0] ? 0 : x < SPEEDS[2] / SPEEDS[0] ? 1 : 2;
    this.running.value = true;
    this.changed();
  }

  /** T-1665: apply or lift the demand cap */
  setCapped(on: boolean) {
    if (on === this.capped.peek()) return;
    this.reanchor();
    this.capped.value = on;
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
