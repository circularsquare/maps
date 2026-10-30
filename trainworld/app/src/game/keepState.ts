// T-1832 (notes/T-1823.md fix 9): what this browser remembers about each game that is kept only here,
// for the "saved in this browser only" card (game/keepNudge.ts): real play time so far, whether the
// player closed the card, whether the game was ever downloaded. Per slot, in localStorage; nothing
// goes in the save file. A leaf module, so workers/clockClient.ts can mark a download without a cycle.

import { signal } from "@preact/signals";

export interface KeepEntry { ms: number; closed?: true; downloaded?: true }
const KEY = "anitabuilder-keep";

/** what this page last wrote, for when storage is unavailable (the card then works for this page only) */
let memory: Record<string, KeepEntry> = {};

export function readKeep(): Record<string, KeepEntry> {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    return v && typeof v === "object" ? v : {};
  } catch { return memory; }
}

export function writeKeep(all: Record<string, KeepEntry>) {
  memory = all;
  try { localStorage.setItem(KEY, JSON.stringify(all)); } catch { /* kept in memory */ }
}

/** Change one slot's entry (read, change, write, so another tab's other slots are kept). */
export function updateKeep(slot: string, f: (e: KeepEntry) => void) {
  const all = readKeep(), e = all[slot] ?? { ms: 0 };
  f(e);
  all[slot] = e;
  writeKeep(all);
}

/** the playing game's entry, null when it has no stored slot */
export const keepEntry = signal<{ slot: string; e: KeepEntry } | null>(null);

/** A game was downloaded as a file: the card has nothing more to say about it. */
export function markDownloaded(slot: string | undefined) {
  if (!slot) return;
  updateKeep(slot, (e) => { e.downloaded = true; });
  const k = keepEntry.peek();
  if (k?.slot === slot) keepEntry.value = { slot, e: { ...k.e, downloaded: true } };
}
