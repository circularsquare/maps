// T-1832 (notes/T-1823.md fix 9; Anita, DECISIONS 2026-10-29): after an hour of play, a game kept only
// in this browser (not on the account, never downloaded) gets a yellow dot beside the search glass
// (Anita's follow-up: a dot, not a card). This counts the hour and says when the dot is due;
// ui/KeepDot.tsx draws it. Only real play counts: the game running, the tab visible, a saved slot
// this tab writes (a `?new=1` page has none). The count and downloads are kept per slot in this
// browser (game/keepState.ts).
// `?keepMs=<ms>` shortens the hour for a check (and the two minutes of `cloudBehind`).

import { computed, signal } from "@preact/signals";
import { tiedTo } from "../workers/cloud";
import { saveSlotsState } from "../workers/clockClient";
import { cloudContext } from "./account";
import { clock } from "./clock";
import { keepEntry, readKeep, writeKeep } from "./keepState";

const HOUR_MS = 3_600_000, TICK_MS = 5000;
const goalMs = () => Number(new URLSearchParams(location.search).get("keepMs")) || HOUR_MS;

/** Is the playing game kept only in this browser: no account, or not yet tied to it? */
export function onlyHere(): boolean {
  const { slots, active } = saveSlotsState.value, slot = slots.find((s) => s.key === active);
  if (!slot || active.startsWith("account:")) return false;
  if (slot.cloud?.acct && slot.cloud.rev > 0) return false; // uploaded to an account at some point, even if it cannot be reached now
  const ctx = cloudContext();
  return !(ctx?.on && tiedTo(slot.cloud, ctx));
}

/** The dot is due: an hour played, only here, and never downloaded. It stays until the game is on the
 * account or downloaded; there is no way to close it. */
export const keepDue = computed(() => {
  const k = keepEntry.value;
  return !!k && k.e.ms >= goalMs() && !k.e.downloaded && onlyHere();
});

/** T-2614: why the playing game's latest change has not reached the account, once that has lasted
 * BEHIND_MS (a slip the next try mends shows nothing); null while it is on the account. */
export const cloudBehind = signal<string | null>(null);
const BEHIND_MS = 120_000;
let behindSince = 0;
const behindMs = () => Math.min(BEHIND_MS, goalMs());

function noteBehind() {
  const { slots, active, writable } = saveSlotsState.peek();
  const slot = writable && active ? slots.find((s) => s.key === active) : undefined;
  const ctx = cloudContext(), c = slot?.cloud;
  const why = slot && ctx?.on && tiedTo(c, ctx) && !c.gaveWay && c.error ? c.error : null;
  if (!why) behindSince = 0;
  else if (!behindSince) behindSince = Date.now();
  const due = why && Date.now() - behindSince >= behindMs() ? why : null;
  if (cloudBehind.peek() !== due) cloudBehind.value = due;
}

/** Count play time, five seconds at a time. */
export function startKeep() {
  let last = performance.now();
  const tick = () => {
    noteBehind();
    const now = performance.now(), dt = Math.min(now - last, TICK_MS * 2);
    last = now;
    const { slots, active, writable } = saveSlotsState.peek();
    const slot = writable && active ? slots.find((s) => s.key === active) : undefined;
    if (!slot) { if (keepEntry.peek()) keepEntry.value = null; return; }
    const all = readKeep();
    // forget games that were deleted
    const gone = Object.keys(all).filter((k) => !slots.some((s) => s.key === k));
    for (const k of gone) delete all[k];
    const e = all[slot.key] ?? { ms: 0 };
    const playing = !!clock && clock.running.peek() && !clock.held.peek() && document.visibilityState === "visible";
    // once the dot has nothing left to say, the count stops
    if (playing && !e.downloaded && e.ms < goalMs()) e.ms += dt;
    all[slot.key] = e;
    if (playing || gone.length) writeKeep(all);
    const k = keepEntry.peek();
    if (!k || k.slot !== slot.key || k.e.ms !== e.ms || k.e.downloaded !== e.downloaded) keepEntry.value = { slot: slot.key, e: { ...e } };
  };
  setInterval(tick, TICK_MS);
  tick();
}
