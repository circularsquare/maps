// T-1214: keeping saved games when the browser's storage goes (a full disk emptied every slot on
// 2026-10-24). The browser is asked to keep this site's storage at the first save, and the player may
// pick a folder outside it (File System Access, Chrome and Edge) where the clock worker copies each
// slot at every save, into a `trainworld` folder inside the one picked (T-1235). Read by
// ui/SavesPanel.tsx; the picked folder's handle is kept beside the saves.

import { signal } from "@preact/signals";
import { readCopyFolder, writeCopyFolder } from "../workers/saveFile";

type Handle = FileSystemDirectoryHandle & {
  queryPermission?(d: { mode: "readwrite" }): Promise<PermissionState>;
  requestPermission?(d: { mode: "readwrite" }): Promise<PermissionState>;
};
type Picker = (o: { id?: string; mode?: "readwrite"; startIn?: "documents" }) => Promise<Handle>;
const picker = (globalThis as { showDirectoryPicker?: Picker }).showDirectoryPicker;

/** the folder the copies go in, inside the one picked (picking it itself works too) */
export const SUBFOLDER = "trainworld";

/** whether the browser keeps this site's storage when the disk fills: null until asked, or where it
 * cannot say */
export const storageKept = signal<boolean | null>(null);
/** the copies' folder: `name` the one picked, `path` where the copies go; copies going there ("on"),
 * waiting for the player to allow it again after a restart ("allow"), or the last copy failed */
export const copyFolder = signal<{ name: string; path: string; state: "on" | "allow" | "failed"; at?: number; error?: string } | null>(null);
export const canCopyToFolder = typeof picker === "function";

let send: (dir: FileSystemDirectoryHandle | null) => void = () => {};
let held: Handle | null = null;

const pathIn = (dir: Handle) => (dir.name === SUBFOLDER ? dir.name : `${dir.name}/${SUBFOLDER}`);

/** The copies go to the subfolder, made if missing; a failure is shown on the panel. */
async function sendTo(dir: Handle) {
  const c = { name: dir.name, path: pathIn(dir) };
  try {
    send(dir.name === SUBFOLDER ? dir : await dir.getDirectoryHandle(SUBFOLDER, { create: true }));
    copyFolder.value = { ...c, state: "on" };
  } catch (err) {
    copyFolder.value = { ...c, state: "failed", error: (err as Error).message };
  }
}

/** At game start: a folder picked on an earlier visit goes to the clock worker if still allowed. */
export async function startSaveSafety(post: (dir: FileSystemDirectoryHandle | null) => void) {
  send = post;
  const dir = (await readCopyFolder()) as Handle | null;
  if (!dir) return;
  held = dir;
  const p = dir.queryPermission ? await dir.queryPermission({ mode: "readwrite" }).catch(() => "prompt" as const) : "granted";
  if (p === "granted") await sendTo(dir);
  else copyFolder.value = { name: dir.name, path: pathIn(dir), state: "allow" };
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

/** The player picks the folder (a click). The picker opens in Documents the first time, then where
 * the last pick was. */
export async function pickCopyFolder() {
  if (!picker) return;
  let dir: Handle;
  try { dir = await picker({ id: "anitabuilder-saves", mode: "readwrite", startIn: "documents" }); } catch { return; } // cancelled
  held = dir;
  await writeCopyFolder(dir);
  await sendTo(dir);
}
/** The player allows the folder again (a click; browsers ask once per visit). */
export async function allowCopyFolder() {
  const dir = held;
  if (!dir?.requestPermission) return;
  if ((await dir.requestPermission({ mode: "readwrite" }).catch(() => "denied")) !== "granted") return;
  await sendTo(dir);
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
  if (c) copyFolder.value = error ? { ...c, state: "failed", error } : { name: c.name, path: c.path, state: "on", at };
}
