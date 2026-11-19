import { signal } from "@preact/signals";
import { t, type MessageKey } from "../i18n";

// T-3773: games that did not open, kept in localStorage so the start menu the page goes to can say why (a notice, and
// a line on the saved game's card) and a feedback report can carry the last one (T-3774 reads `readOpenFailures`).

declare const __TW_VERSION__: string | undefined;

export interface OpenFailure {
  /** wall ms */
  at: number;
  from: "slot" | "file" | "restore" | "view" | "new";
  /** the saved game's slot (a backup's: the slot it backs up); none for a file or a viewed network */
  slot?: string;
  name: string;
  /** a message key, or text as it is */
  error: string;
  /** a panic's message */
  why?: string;
  version: string;
}
export interface OpenFailures {
  /** every failed open in this browser since this record began */
  count: number;
  /** newest first, at most `KEEP` */
  recent: OpenFailure[];
  /** the last one's notice has not been shown on the start menu yet */
  unseen?: boolean;
}

const KEY = "tw-open-failures";
const KEEP = 10;

export function readOpenFailures(): OpenFailures {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "null") as OpenFailures | null;
    if (v && typeof v.count === "number" && Array.isArray(v.recent)) return v;
  } catch { /* storage unavailable or unreadable */ }
  return { count: 0, recent: [] };
}
function write(v: OpenFailures) {
  try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* storage unavailable */ }
  openFailures.value = v;
}
export const openFailures = signal<OpenFailures>(readOpenFailures());

export function noteOpenFailure(f: Omit<OpenFailure, "at" | "version">) {
  const v = readOpenFailures();
  write({ count: v.count + 1, recent: [{ ...f, at: Date.now(), version: typeof __TW_VERSION__ === "string" ? __TW_VERSION__ : "dev" }, ...v.recent].slice(0, KEEP), unseen: true });
}

/** A game in slot `key` opened: its card says nothing more about an earlier failure (the count stays). */
export function openedFine(key: string | undefined) {
  if (!key) return;
  const v = readOpenFailures();
  if (v.recent.some((f) => f.from === "slot" && f.slot === key)) write({ ...v, recent: v.recent.filter((f) => !(f.from === "slot" && f.slot === key)) });
}

const textOf = (f: OpenFailure) => f.why !== undefined ? t(f.error as MessageKey, { why: f.why }) : /^(clockWorker|issues|loadText)\.[A-Za-z]+$/.test(f.error) ? t(f.error as MessageKey) : f.error;

/** Why the saved game in slot `key` last failed to open, for its card; null if it has opened since. */
export function openFailText(key: string): string | null {
  const f = openFailures.value.recent.find((x) => x.from === "slot" && x.slot === key);
  return f ? textOf(f) : null;
}

/** The start menu's notice for the last failed open, once; null when it was shown or there is none. */
export function takeOpenFailNotice(): string | null {
  const v = readOpenFailures(), f = v.recent[0];
  if (!v.unseen || !f) return null;
  write({ ...v, unseen: false });
  return f.from === "slot" || f.from === "new" ?t("startMenu.openFailed", { name: f.name }) : textOf(f);
}
