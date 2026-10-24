// T-1214: keeping saved games when the browser's storage goes (a full disk emptied every slot on
// 2026-10-24). The browser is asked to keep this site's storage at the first save, and the player may
// pick a folder outside it (File System Access, Chrome and Edge) where the clock worker copies each
// slot at every save. Read by ui/SavesPanel.tsx; the folder's handle is kept beside the saves.

import { signal } from "@preact/signals";
import { readCopyFolder, writeCopyFolder } from "../workers/saveFile";

type Handle = FileSystemDirectoryHandle & {
  queryPermission?(d: { mode: "readwrite" }): Promise<PermissionState>;
  requestPermission?(d: { mode: "readwrite" }): Promise<PermissionState>;
};
type Picker = (o: { id?: string; mode?: "readwrite" }) => Promise<Handle>;
const picker = (globalThis as { showDirectoryPicker?: Picker }).showDirectoryPicker;

/** whether the browser keeps this site's storage when the disk fills: null until asked, or where it
 * cannot say */
export const storageKept = signal<boolean | null>(null);
/** the copies' folder: copies going there ("on"), waiting for the player to allow it again after a
 * restart ("allow"), or the last copy failed */
export const copyFolder = signal<{ name: string; state: "on" | "allow" | "failed"; at?: number; error?: string } | null>(null);
export const canCopyToFolder = typeof picker === "function";

let send: (dir: FileSystemDirectoryHandle | null) => void = () => {};
let held: Handle | null = null;

/** At game start: a folder picked on an earlier visit goes to the clock worker if still allowed. */
export async function startSaveSafety(post: (dir: FileSystemDirectoryHandle | null) => void) {
  send = post;
  const dir = (await readCopyFolder()) as Handle | null;
  if (!dir) return;
  held = dir;
  const p = dir.queryPermission ? await dir.queryPermission({ mode: "readwrite" }).catch(() => "prompt" as const) : "granted";
  copyFolder.value = { name: dir.name, state: p === "granted" ? "on" : "allow" };
  if (p === "granted") send(dir);
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

/** The player picks the folder (a click). */
export async function pickCopyFolder() {
  if (!picker) return;
  let dir: Handle;
  try { dir = await picker({ id: "anitabuilder-saves", mode: "readwrite" }); } catch { return; } // cancelled
  held = dir;
  await writeCopyFolder(dir);
  copyFolder.value = { name: dir.name, state: "on" };
  send(dir);
}
/** The player allows the folder again (a click; browsers ask once per visit). */
export async function allowCopyFolder() {
  const dir = held;
  if (!dir?.requestPermission) return;
  if ((await dir.requestPermission({ mode: "readwrite" }).catch(() => "denied")) !== "granted") return;
  copyFolder.value = { name: dir.name, state: "on" };
  send(dir);
}
export async function stopCopies() {
  held = null;
  await writeCopyFolder(null);
  copyFolder.value = null;
  send(null);
}
/** the clock worker's answer for a copy */
export function copied(at?: number, error?: string) {
  const c = copyFolder.peek();
  if (c) copyFolder.value = error ? { ...c, state: "failed", error } : { name: c.name, state: "on", at };
}
