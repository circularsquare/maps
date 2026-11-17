// T-3520: time played in this browser, for the trainsfan achievement. Counted the way the plays ping counts
// it (game/playtime.ts): a second only while a game is open (not the start menu or the new game sheet) and
// the tab is visible. Kept in localStorage, so it spans saves and visits from this browser and starts at
// zero with this release; it counts even where the ping sends nothing (localhost, automated browsers).
// Each save earns the achievement itself, as for every other one. Two tabs open at once each add their own.

import { computed, signal } from "@preact/signals";
import { choosing } from "./loading";
import { moneyState } from "./money";

export const PLAYED_KEY = "trainworld.played";
export const TRAINSFAN_HOURS = 100;

function stored(): number {
  try {
    const v = Number(localStorage.getItem(PLAYED_KEY));
    return Number.isFinite(v) && v > 0 ? v : 0;
  } catch {
    return 0;
  }
}

/** Seconds played, saved and not yet saved. */
export const playedSeconds = signal(stored());
export const trainsfanReached = computed(() => playedSeconds.value >= TRAINSFAN_HOURS * 3600);

let unsaved = 0;
function save() {
  if (!unsaved) return;
  try {
    // adds to what is there, so a second tab's seconds are not written over
    localStorage.setItem(PLAYED_KEY, String(Math.round((stored() + unsaved) * 10) / 10));
    unsaved = 0;
  } catch { /* storage blocked: counts for this page only */ }
}

if (typeof document !== "undefined") {
  let last = performance.now(), sinceSave = 0;
  setInterval(() => {
    const now = performance.now(), dt = Math.min((now - last) / 1000, 5); // a sleeping laptop adds no hours
    last = now;
    if (document.visibilityState !== "visible" || choosing.peek() || !moneyState.peek()) return;
    unsaved += dt;
    playedSeconds.value += dt;
    sinceSave += dt;
    if (sinceSave >= 10) { sinceSave = 0; save(); }
  }, 1000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState !== "visible") save(); });
  addEventListener("pagehide", save);
}
