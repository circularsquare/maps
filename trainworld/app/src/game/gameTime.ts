// T-3230: how long each stored game has been open, kept in this browser only (localStorage, by slot key).
// It counts the way the playtime ping does (game/playtime.ts): a game is open, not the start menu or the new
// game sheet, and the tab is visible, so a game's figure and the account's total mean the same minutes. Paused
// time counts: a player thinking over a network with the clock stopped is still playing. A game saved before
// this has no figure and shows none; the count starts from its next open.

import { moneyState } from "./money";
import { choosing } from "./loading";
import { shareableSlot } from "./account";
import { t } from "../i18n";

const KEY = "trainworld.gameSeconds";
const FLUSH_S = 15;

function read(): Record<string, number> {
  try {
    const o = JSON.parse(localStorage.getItem(KEY) || "{}");
    return o && typeof o === "object" ? o : {};
  } catch { return {}; }
}

/** Seconds a stored game has been open here; undefined when none was counted. */
export function gameSeconds(slot: string): number | undefined {
  const v = read()[slot];
  return typeof v === "number" && v > 0 ? v : undefined;
}

let timer = 0;
/** Begin counting. Idempotent. Adds to what is stored, so two tabs on one game do not overwrite each other. */
export function startGameTime() {
  if (timer) return;
  let slot = "", pending = 0;
  const flush = () => {
    if (!slot || pending <= 0) { pending = 0; return; }
    try {
      const all = read();
      all[slot] = (all[slot] ?? 0) + pending;
      localStorage.setItem(KEY, JSON.stringify(all));
    } catch { /* storage blocked: nothing is kept */ }
    pending = 0;
  };
  timer = window.setInterval(() => {
    const now = shareableSlot();
    if (now !== slot) { flush(); slot = now ?? ""; }
    if (!slot || document.visibilityState !== "visible" || choosing.peek() || !moneyState.peek()) return;
    if (++pending >= FLUSH_S) flush();
  }, 1000);
  addEventListener("pagehide", flush);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flush(); });
}

/** Drop the figures of games that are gone. */
export function pruneGameTime(keep: string[]) {
  try {
    const all = read(), set = new Set(keep);
    let changed = false;
    for (const k of Object.keys(all)) if (!set.has(k)) { delete all[k]; changed = true; }
    if (changed) localStorage.setItem(KEY, JSON.stringify(all));
  } catch { /* ignore */ }
}

/** Seconds as "25 min" or "3 h 20 min", at least a minute. */
export function playedText(seconds: number): string {
  const m = Math.max(1, Math.round(seconds / 60));
  if (m < 60) return t("playtime.min", { n: m });
  return m % 60 ? t("playtime.hoursMin", { h: Math.floor(m / 60), m: m % 60 }) : t("playtime.hours", { h: m / 60 });
}
