// T-1832 (notes/T-1823.md fix 9; Anita, DECISIONS 2026-10-29): after an hour of play, a game kept only
// in this browser (not on the account, never downloaded) gets a card saying so, with Sign in and
// Download. This counts the hour and says when the card is due; ui/KeepCard.tsx draws it. Only real
// play counts: the game running, the tab visible, a saved slot this tab writes (a `?new=1` page has
// none). The count, the x and downloads are kept per slot in this browser (game/keepState.ts).
// `?keepMs=<ms>` shortens the hour for a check.

import { computed } from "@preact/signals";
import { tiedTo } from "../workers/cloud";
import { saveSlotsState } from "../workers/clockClient";
import { cloudContext } from "./account";
import { clock } from "./clock";
import { keepEntry, readKeep, updateKeep, writeKeep } from "./keepState";

const HOUR_MS = 3_600_000, TICK_MS = 5000;
const goalMs = () => Number(new URLSearchParams(location.search).get("keepMs")) || HOUR_MS;

/** Is the playing game kept only in this browser: no account, or not yet tied to it? */
export function onlyHere(): boolean {
  const { slots, active } = saveSlotsState.value, slot = slots.find((s) => s.key === active);
  if (!slot || active.startsWith("account:")) return false;
  const ctx = cloudContext();
  return !(ctx?.on && tiedTo(slot.cloud, ctx));
}

/** The card is due: an hour played, only here, never downloaded, and not closed. */
export const keepDue = computed(() => {
  const k = keepEntry.value;
  return !!k && k.e.ms >= goalMs() && !k.e.closed && !k.e.downloaded && onlyHere();
});

export function closeKeep() {
  const k = keepEntry.peek();
  if (!k) return;
  updateKeep(k.slot, (e) => { e.closed = true; });
  keepEntry.value = { slot: k.slot, e: { ...k.e, closed: true } };
}

/** Count play time, five seconds at a time. */
export function startKeep() {
  let last = performance.now();
  const tick = () => {
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
    // once the card has nothing left to show, the count stops
    if (playing && !e.closed && !e.downloaded && e.ms < goalMs()) e.ms += dt;
    all[slot.key] = e;
    if (playing || gone.length) writeKeep(all);
    const k = keepEntry.peek();
    if (!k || k.slot !== slot.key || k.e.ms !== e.ms || k.e.closed !== e.closed || k.e.downloaded !== e.downloaded) keepEntry.value = { slot: slot.key, e: { ...e } };
  };
  setInterval(tick, TICK_MS);
  tick();
}
