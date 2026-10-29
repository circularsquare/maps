// T-1214: the browser is asked to keep this site's storage at the first save (a full disk emptied every
// slot on 2026-10-24); whether it agreed shows in the saves panel's head (ui/SavesPanel.tsx).

import { signal } from "@preact/signals";
import { forgetCopyFolder } from "../workers/saveFile";

/** whether the browser keeps this site's storage when the disk fills: null until asked, or where it
 * cannot say */
export const storageKept = signal<boolean | null>(null);

/** At game start (T-1395): a copy folder picked under T-1214 is forgotten; its files stay. */
export function startSaveSafety() {
  void forgetCopyFolder();
}

/** T-1826: the browser's "leave page?" question is on while the game has a change nothing could
 * write (autosave off, a failed write), and only then. */
export const askingBeforeLeaving = signal(false);
const holdPage = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
export function askBeforeLeaving(on: boolean) {
  if (on === askingBeforeLeaving.peek()) return;
  askingBeforeLeaving.value = on;
  if (on) addEventListener("beforeunload", holdPage);
  else removeEventListener("beforeunload", holdPage);
}

let asked = false;
/** Once a slot has been written: ask the browser to keep the storage (it decides without a prompt in
 * Chrome, by how the site is used). */
export async function keepStorage() {
  if (asked) return;
  asked = true;
  const s = navigator.storage;
  if (!s?.persist) return;
  try { storageKept.value = (await s.persisted()) || (await s.persist()); } catch { /* storage unavailable */ }
}
