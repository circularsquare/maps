// T-3314: the page can tell that its figures have stopped updating: the clock worker has an edit or read the
// page sent that it has not answered, or the demand pool has a network it has been waiting to solve, with
// nothing arriving. Either shows the notice at the top centre (ui/FiguresStopped.tsx) with a Reload button.
// Thresholds are in notes/T-3314.md with the measurements behind them.

import { signal } from "@preact/signals";
import { demandStopped, startup } from "./loading";
import { staleBuild } from "./staleBuild";

/** An edit or read unanswered this long, ms. The longest seen in normal play was 17 s (importing the real save on
 * one core, notes/T-3314.md); 3 minutes is about ten times that. */
export const CLOCK_QUIET_MS = 180_000;
/** A network waiting to be solved with no download or worker step for this long, ms. The longest seen in
 * normal play was 8 s; every pack's bytes count as a step, so a slow download does not reach it. */
export const DEMAND_QUIET_MS = 180_000;
/** (`?figuresMs=N` sets both, for checks) */
const OVERRIDE = Number(typeof location === "undefined" ? 0 : new URLSearchParams(location.search).get("figuresMs")) || 0;
const TICK_MS = 2_000;

export const figuresStopped = signal(false);
/** The longest each quiet has been, ms, while the page was visible (`?debug=1` checks read it) */
export const figuresPeaks = { clock: 0, demand: 0 };

/** `clockSince` and `demandSince`: performance.now() of the oldest thing waiting without an answer, or null.
 * Time while the page was hidden does not count: a browser may hold a hidden page's messages and timers. */
export function watchFigures(clockSince: () => number | null, demandSince: () => number | null) {
  let visibleSince = performance.now();
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") visibleSince = performance.now(); });
  const quiet = (since: number | null, now: number) => (since === null ? 0 : Math.max(0, now - Math.max(since, visibleSince)));
  setInterval(() => {
    if (document.visibilityState === "hidden") return;
    const now = performance.now();
    const c = quiet(clockSince(), now), d = quiet(demandSince(), now);
    figuresPeaks.clock = Math.max(figuresPeaks.clock, c);
    figuresPeaks.demand = Math.max(figuresPeaks.demand, d);
    // before the game is up the loading card has its own watchdog; the other notices say their own thing
    const on = startup.peek() === "ready" && !demandStopped.peek() && !staleBuild.peek()
      && (c >= (OVERRIDE || CLOCK_QUIET_MS) || d >= (OVERRIDE || DEMAND_QUIET_MS));
    if (on !== figuresStopped.peek()) figuresStopped.value = on;
  }, TICK_MS);
}
