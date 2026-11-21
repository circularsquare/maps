// The save slot of the game in hand (T-3734): which slot it is in and whether it may be written there, autosave and
// the save by hand, the saves panel's actions, the slot's lease against other tabs, and the game's copy on the
// account (the cloud queue). The clock worker plays the game; it registers what this needs of it (`bindSlots`) and
// calls in. The storage itself is saveFile.ts, the account cloud.ts.

import type { MessageKey } from "../i18n"; // type only: this worker sends keys, the page words them (docs/localization.md "Workers")
import type { TrackApi } from "../wasm/track/sim";
import { HOME_CITY } from "../game/cities";
import { listSlots, writeSlot, SlotTakenError, renameSlot, leaseSlot, ORIGINAL_SLOT, OLD_FRAME_PREFIX, readStored, writeStored, listBackups, writeBackup, backupSlot, BACKUP_STEPS, slotName, writePending, readSync, writeSync, makeRoom, lastSync, type PendingLoad, type SlotBackup, type SlotLease, type SlotSummary, type StoredSave } from "./saveFile";
import type { FromClock, ToClock } from "./protocol";
import { claim, editedHere, editedThere, fetchGame, GAP_MS, lastListFailure, listGames, newGameId, onTokenGone, removeSlot, sendDeletes, settle as settleCloud, splitOff, tieNew, tiedTo, upload, writeFetched, type CloudContext } from "./cloud";
import { editOf, syncedEditOf } from "./versions";
import { modelDown } from "./clockPanic";

/** What this module needs of the clock worker: the game in hand, its model and its message queue. */
export interface SlotHost {
  api(): TrackApi | null;
  /** the worker's load count (`game`), which only its `restart` moves */
  game(): number;
  send(msg: FromClock): void;
  gameNow(): number;
  saveBytes(): Promise<Uint8Array>;
  slotSummary(): Promise<SlotSummary>;
  /** T-3869: `make` tried again while it fails for want of memory, the page's demand workers replaced first
   * (`always`: whenever it comes to that, else at most once in ten minutes) */
  roomFor<T>(make: () => Promise<T>, always: boolean): Promise<T>;
  /** T-3869: a failure for want of memory */
  outOfRoom(err: unknown): boolean;
  loadGame(bytes: Uint8Array, from: "autosave" | "file" | "new"): Promise<boolean>;
  /** `f` run in the game's own message queue, after the messages queued before it */
  inQueue(f: () => Promise<void>): Promise<void>;
  isPanic(err: unknown): boolean;
  panicLater(err: unknown, what: string): void;
  /** T-3788: the game in hand failed to load in place: no slot is held, the page goes to the start menu */
  openFailed(err: unknown, name: string): void;
}
let host: SlotHost;
export function bindSlots(h: SlotHost) { host = h; }
const send = (msg: FromClock) => host.send(msg);
const gameNow = () => host.gameNow();
const saveBytes = () => host.saveBytes();
const slotSummary = () => host.slotSummary();
const loadGame = (bytes: Uint8Array, from: "autosave" | "file" | "new") => host.loadGame(bytes, from);
const isPanic = (err: unknown) => host.isPanic(err);
const panicLater = (err: unknown, what: string) => host.panicLater(err, what);
const city = HOME_CITY;

/** The slot's state, which the clock worker reads and writes too (a player's change marks it `dirty`, `edited`). */
export const saves = {
  /** T-744, T-1826: the backup tiers' steps (`BACKUP_STEPS`), scaled down by `?backupMs=` for checks */
  backupSteps: BACKUP_STEPS,
  /** T-3734: the `game` the active slot holds, sent with the slot list. It moves when a load into the active slot itself
   * starts a game (`restart`), and for a load into another slot or a view only once that one is made active, so a
   * slot list sent during the load names the game being left. */
  slotGame: 0,
  /** something to autosave; when the last autosave was written */
  dirty: false,
  savedAt: Date.now(),
  saveInFlight: null as Promise<boolean> | null,
  /** the next autosave is a save by hand: it goes to the account at once, with `force` (T-988) */
  byHandNext: false,
  activeSlot: ORIGINAL_SLOT,
  activeLease: null as SlotLease | null,
  /** another tab saved this slot by hand and took its lease (T-1251): autosave stays off here */
  leaseTaken: false,
  /** the start menu's new game (T-677): its slot's name, given at its first write */
  newSlotName: null as { key: string; name: string } | null,
  /** the active slot while nothing is stored under it (a new game's, T-677) */
  unwritten: null as string | null,
  storageReady: true,
  /** `?new=1`, `?synth=1`, `?demandTest=1` (T-320): no slot is written until "Save as a copy" gives
   * the game a slot of its own; a `?new=1` game has no active slot ("") and reads none */
  testPage: false,
  /** T-3257: the game in hand is a network viewed from Community: it has no slot and nothing writes it until Save
   * gives it one under this name, after which it is a game like any other; null otherwise */
  viewing: null as string | null,
  /** T-3257: the viewed game's card for the saves panel (its picture and facts), drawn when it opens and again each
   * time the panel asks */
  viewCard: null as SlotSummary | null,
  saveFailure: undefined as string | undefined,
  /** while the game is being swapped for another slot's (T-743): nothing autosaves */
  switching: false,
  /** T-2552: the slot a game is being loaded into, which becomes the active one only after; for the "loaded" message */
  loadingInto: null as string | null,
  /** T-1826: a player's change (an edit, a list order, sandbox) not yet written, for the page's
   * "leave page?" prompt; the clock's own settling is not one */
  edited: false,
  unsavedSent: false,
  /** T-2037: the game's origin (versions.ts), the wall time of its latest player change (T-2030: the
   * clock's own saves leave it), and the slot's latest change as this page last read or wrote it, so a
   * save by hand can tell another tab's change it would write over */
  gameOrigin: undefined as string | undefined,
  editAt: 0,
  seenEdit: 0,
  /** T-1829: the playing slot's name as last listed, kept should the database be cleared under it */
  activeName: null as { key: string; name: string } | null,
  cloudCtx: null as CloudContext | null,
  /** the slot whose game this page has claimed (or uploaded) on the account */
  cloudClaimed: "",
  /** the slot another device opened: autosave is off here until a save by hand takes it back */
  cloudTaken: "",
  cloudQueue: Promise.resolve() as Promise<unknown>,
  /** the game's first load is done: the claim can follow */
  initDone: false,
  /** T-2614: when the game in hand, whose change did not reach the account, is sent again (0: nothing
   * owed). Autosave sends it too while the game runs; this covers a paused game. */
  cloudRetryAt: 0,
};

export function hold(lease: SlotLease | null) {
  saves.activeLease = lease;
  saves.leaseTaken = false;
  void lease?.lost.then(() => {
    if (saves.activeLease !== lease) return;
    saves.activeLease = null;
    saves.leaseTaken = true;
    void publishSlots();
  });
}

/** T-2703, T-3773: a start or an open that failed holds no slot (`writeBlock` "noLease"), so nothing it does
 * afterwards can write one */
export function letGoSlot() {
  saves.activeLease?.release();
  hold(null);
  saves.activeSlot = "";
  saves.unwritten = null;
  saves.newSlotName = null;
}

const slotChanges = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel("anitabuilder-save-slots");
slotChanges?.addEventListener("message", () => { if (host.api()) void publishSlots(); });
/** T-3869: a save or download that found no memory even after the demand workers were replaced */
export const NO_MEMORY_TO_SAVE: MessageKey = "clockWorker.noMemoryToSave";

/** T-2037: the game in hand is now `s`'s (a slot opened, a copy taken in) */
export function heldFrom(s: { origin?: string; edit?: number; at: number }) {
  saves.gameOrigin = s.origin;
  saves.editAt = saves.seenEdit = editOf(s);
}
/** T-2037: a game that starts here (a new game, a file opened): an origin of its own */
export function heldNew() {
  saves.gameOrigin = newGameId();
  saves.editAt = Date.now();
  saves.seenEdit = 0;
}

/** T-3734: why the game in hand may not be written to its slot now, first match in the order the saves panel names
 * it; null if it may. `attempting`: for a write about to be tried, which is itself the retry of a failed write or
 * of missing storage, so those two do not block it. */
type WriteBlock = "failed" | "storage" | "otherDevice" | "viewing" | "testPage" | "otherTab" | "noLease" | "switching";
function writeBlock(attempting = false): WriteBlock | null {
  if (!attempting && saves.saveFailure) return "failed";
  if (!attempting && !saves.storageReady) return "storage";
  if (saves.activeSlot && saves.cloudTaken === saves.activeSlot) return "otherDevice";
  if (!saves.activeLease) return saves.viewing !== null ? "viewing" : saves.testPage ? "testPage" : saves.leaseTaken ? "otherTab" : "noLease";
  if (saves.switching) return "switching";
  return null;
}
/** the slot is someone else's to write, or nobody's: leaving loses only a player's change */
const NOT_MINE: ReadonlySet<WriteBlock | null> = new Set(["otherDevice", "viewing", "testPage", "otherTab", "noLease"]);
const BLOCK_REASON: Record<Exclude<WriteBlock, "failed" | "switching">, MessageKey> = {
  storage: "clockWorker.noStorage", otherDevice: "clockWorker.otherDevice", viewing: "clockWorker.viewing",
  testPage: "clockWorker.testPage", otherTab: "clockWorker.otherTabSaved", noLease: "clockWorker.otherTabOpened",
};
/** a swap lasts a moment and the slot stays this page's, so the page is told it is writable meanwhile */
function writableNow(): boolean {
  const b = writeBlock();
  return b === null || b === "switching";
}
/** T-1826: tell the page whether closing it would lose a change: one made while nothing can write it. */
export function noteUnsaved() {
  const on = saves.edited && !writableNow();
  if (on !== saves.unsavedSent) send({ kind: "unsaved", on: (saves.unsavedSent = on) });
}

export async function publishSlots(error?: string) {
  let slots: Awaited<ReturnType<typeof listSlots>> = [];
  let backups: SlotBackup[] = [];
  try {
    slots = await listSlots(); saves.storageReady = true;
    if (saves.activeSlot) backups = await listBackups(saves.activeSlot);
  } catch { saves.storageReady = false; }
  const mine = slots.find((s) => s.key === saves.activeSlot);
  if (mine) saves.activeName = { key: mine.key, name: mine.name };
  const block = writeBlock();
  const writable = writableNow();
  noteUnsaved();
  const reason: MessageKey | string | undefined = block === null || block === "switching" ? undefined
    : block === "failed" ? saves.saveFailure
    : block === "noLease" && !navigator.locks ? "clockWorker.noLocks"
    : BLOCK_REASON[block];
  const view = saves.viewing !== null ? { viewing: { ...saves.viewCard, key: "", name: saves.viewing, at: 0, day: Math.floor(gameNow() / 86400) } } : {};
  send({ kind: "saveSlots", slots, active: saves.activeSlot, game: saves.slotGame, writable, at: slots.find(s => s.key === saves.activeSlot)?.at ?? 0, reason, error, backups, ...view });
}

async function flushBeforeSwitch() {
  // T-3257: a viewed game is let go unsaved (the page asked the player first)
  if (saves.viewing !== null) return;
  // T-1826, T-2843 finding 4: a slot this page may not write is left as it was; only a change the player made here
  // would be lost (the clock's own settling is not one)
  if (NOT_MINE.has(writeBlock(true))) {
    if (saves.edited) throw new Error("clockWorker.copyBeforeLeaving" satisfies MessageKey);
    return;
  }
  // the stopped model cannot be saved; the last autosave is what stays
  if (modelDown) return;
  if (!await autosave()) throw new Error("clockWorker.couldNotSaveLeaving" satisfies MessageKey);
}

/** T-743: New game, Open a file and a backup's restore each put the game in a slot of its own,
 * named `name`; the game being played is saved first and keeps its slot. `load` replaces the game
 * (throwing changes nothing). A new game's slot is written once something changes, a loaded one's
 * at once. A test page has no slot to keep: the game is replaced in the page and owns none. */
export async function intoNewSlot(name: string, load: () => Promise<unknown> | void, writeNow: boolean, version?: { origin?: string; edit: number; at: number }) {
  // T-2037: a backup opened is a version of its game (`version`); a file or a new game starts one
  const held = () => { if (version) { heldFrom(version); saves.gameOrigin ??= newGameId(); saves.seenEdit = 0; } else heldNew(); };
  if (saves.testPage) {
    await load();
    held();
    saves.activeLease?.release();
    saves.activeSlot = "";
    saves.activeLease = null;
    saves.dirty = writeNow;
    await publishSlots();
    return;
  }
  await flushBeforeSwitch();
  const key = "slot-" + crypto.randomUUID();
  const lease = await leaseSlot(key);
  saves.switching = true;
  saves.loadingInto = key;
  try {
    await load();
  } catch (err) { lease?.release(); throw err; } finally { saves.switching = false; saves.loadingInto = null; }
  held();
  saves.activeLease?.release();
  saves.activeSlot = key;
  saves.slotGame = host.game();
  hold(lease);
  saves.viewing = saves.viewCard = null;
  saves.unwritten = key;
  saves.newSlotName = { key, name };
  saves.saveFailure = undefined;
  saves.dirty = writeNow;
  saves.savedAt = Date.now();
  // T-3687: an empty game is not written (autosave), and the page must still hear which slot it is in
  if (!writeNow || !await autosave() || saves.unwritten === key) await publishSlots();
  slotChanges?.postMessage({ key });
}

/** T-3257: a network viewed from Community replaces the game in hand (saved first) and is held in no slot: it plays
 * as any game, and nothing writes it until Save (`saveView`) gives it a slot named `name`. */
export async function intoView(name: string, load: () => Promise<unknown>) {
  await flushBeforeSwitch();
  saves.switching = true;
  saves.loadingInto = ""; // the "loaded" it sends names no slot
  try { await load(); } finally { saves.switching = false; saves.loadingInto = null; }
  heldNew();
  saves.activeLease?.release();
  hold(null);
  saves.activeSlot = "";
  saves.slotGame = host.game();
  saves.activeName = null;
  saves.unwritten = null;
  saves.newSlotName = null;
  saves.saveFailure = undefined;
  saves.dirty = saves.edited = false;
  saves.viewing = name;
  saves.viewCard = await slotSummary().catch(() => null);
  await publishSlots();
}

/** T-3257: Save on a viewed game: a slot of its own under the view's name, a game like any other from then on. */
async function saveView() {
  await manageSlot({ kind: "saveSlot", action: "create", name: saves.viewing ?? undefined });
  if (saves.viewing === null) send({ kind: "savedByHand", at: saves.savedAt });
}

// T-988: games on the account (workers/cloud.ts, notes/T-988.md). The main thread posts the account;
// each slot write goes up after it (a save by hand at once, autosaves at most every GAP_MS), and the
// game opened is claimed on the account. All of it runs in one queue of its own, so the game's
// messages never wait on the network; only the swap of the game in hand goes through theirs.
const cloudTimers = new Map<string, ReturnType<typeof setTimeout>>();
function cloudRun(f: () => Promise<unknown>): Promise<unknown> {
  saves.cloudQueue = saves.cloudQueue.then(f).catch(() => {});
  return saves.cloudQueue;
}
onTokenGone(() => send({ kind: "accountGone" }));

/** After slot `key` was written: up now (`force`, a save by hand), or once the gap since its last
 * upload has passed. `fresh`: the slot's first write, which ties it to the account signed in. */
async function cloudAfterWrite(key: string, force: boolean, fresh: boolean) {
  const ctx = saves.cloudCtx;
  if (!ctx || saves.testPage) return;
  let rec = await readSync(key);
  if (!rec && fresh) rec = await tieNew(ctx, key);
  if (!tiedTo(rec, ctx) || !ctx.on) return;
  const wait = force ? 0 : (rec.upAt ?? 0) + GAP_MS - Date.now();
  if (wait > 0) {
    if (!cloudTimers.has(key)) cloudTimers.set(key, setTimeout(() => { cloudTimers.delete(key); void cloudUp(key); }, wait));
    return;
  }
  clearTimeout(cloudTimers.get(key));
  cloudTimers.delete(key);
  await cloudUp(key, force);
}

/** Slot `key` up as stored. The game in hand is claimed first; a 409 "elsewhere" on it means another
 * device opened it, and this page gives way (a save by hand sends `force` and takes it back). */
export function cloudUp(key: string, force = false): Promise<unknown> {
  return cloudRun(() => upNow(key, force));
}
async function upNow(key: string, force: boolean) {
  const ctx = saves.cloudCtx;
  if (!ctx?.on || saves.testPage) return;
  if (key === saves.activeSlot && !force && saves.cloudClaimed !== key) { await openOnAccount(ctx); await publishSlots(); return; }
  const [s, rec] = await Promise.all([readStored(key), readSync(key)]);
  // T-2030: the clock's own saves go up with the next player change, a save by hand or a rename
  if (!force && (!s || !rec || !editedHere(s, rec))) { if (key === saves.activeSlot) upFailed(false); return; }
  const there = async () => s && tiedTo(rec, ctx) ? (await listGames(ctx))?.games.find((g) => g.id === rec.id) : undefined;
  let r: Awaited<ReturnType<typeof upload>>;
  const g = force ? await there() : undefined;
  // T-2037: a save by hand never writes over a player change made elsewhere: this version splits off
  if (g && editedThere(g, rec!)) r = await splitOff(ctx, key);
  else r = await upload(ctx, key, force);
  if (r === "elsewhere") {
    // T-2030: a device that opened the game but changed nothing this one lacks does not take it
    const w = await there();
    if (w && !editedThere(w, rec!) && editedHere(s!, rec!)) r = await upload(ctx, key, true);
  }
  if (key === saves.activeSlot && r === "ok") { saves.cloudClaimed = key; if (saves.cloudTaken === key) saves.cloudTaken = ""; }
  if (key === saves.activeSlot && r === "elsewhere") await giveWay(key);
  if (key === saves.activeSlot) upFailed(r === "failed");
  await publishSlots();
}


export const CLOUD_RETRY_MS = 60_000;
function upFailed(failed: boolean) {
  saves.cloudRetryAt = failed ? Date.now() + CLOUD_RETRY_MS : 0;
}

async function giveWay(key: string) {
  saves.cloudTaken = key;
  saves.cloudClaimed = "";
  const rec = await readSync(key);
  if (rec) await writeSync(key, { ...rec, gaveWay: true });
}

/** The page hides: anything waiting for the gap goes now (best effort, a closing page can cut it). */
export function cloudFlush() {
  for (const [key, t] of cloudTimers) { clearTimeout(t); cloudTimers.delete(key); void cloudUp(key); }
  if (saves.activeSlot && saves.cloudTaken !== saves.activeSlot) void cloudUp(saves.activeSlot);
}

/** Settle every other slot this page can lock, then claim the game in hand. */
export function cloudKick(): Promise<unknown> {
  return cloudRun(async () => {
    const ctx = saves.cloudCtx;
    if (!ctx?.on || saves.testPage || !saves.initDone) return;
    await settleCloud(ctx, saves.activeSlot);
    await openOnAccount(ctx);
    await publishSlots();
  });
}

/** The game in hand claimed on the account. A newer account copy replaces it if nothing changed here
 * (the page loads it in place), else this browser's version carries on as a fork in a new slot and
 * the old slot takes the account's. Offline it plays on and claims when back. */
async function openOnAccount(ctx: CloudContext) {
  const key = saves.activeSlot;
  if (!key || !saves.activeLease || saves.cloudClaimed === key || saves.cloudTaken === key) return;
  let [stored, rec] = await Promise.all([readStored(key), readSync(key)]);
  if (!stored || !tiedTo(rec, ctx)) return; // a new game not written yet, or not on this account
  const list = await listGames(ctx);
  // T-2030: a player change here, written or not; the clock's own saves are not one
  const changed = () => editedHere(stored!, rec!) || (saves.dirty && saves.editAt > syncedEditOf(rec!));
  if (!list) {
    // unreachable: the game plays on, "Not on your account yet" while it has changed here
    if (saves.activeSlot === key && editedHere(stored, rec)) { upFailed(true); await writeSync(key, { ...rec, error: lastListFailure() }); await publishSlots(); }
    return;
  }
  if (saves.activeSlot !== key) return;
  const g = list.games.find((x) => x.id === rec!.id);
  if (!g) {
    if (list.tombstones.some((t) => t.id === rec!.id)) {
      // deleted on another device: a game changed here is kept as a new account game; one not
      // changed stays in this browser only, so playing it brings back nothing the player deleted
      if (!changed()) { await writeSync(key, { ...rec, acct: undefined, local: true }); return; }
      rec = { ...rec, id: newGameId(), rev: 0, syncedAt: 0, edit: undefined };
      await writeSync(key, rec);
    }
    const r = await upload(ctx, key);
    if (r === "ok") saves.cloudClaimed = key;
    upFailed(r === "failed");
    return;
  }
  if (g.rev < rec.rev) { rec = { ...rec, rev: g.rev }; await writeSync(key, rec); }
  if (g.rev === rec.rev) {
    const c = await claim(ctx, rec);
    if (c !== "ok") { if (c === "failed" && editedHere(stored, rec)) upFailed(true); return; } // tried again at the next save
    saves.cloudClaimed = key;
    rec = (await readSync(key)) ?? rec;
    if (rec.gaveWay) await writeSync(key, rec = { ...rec, gaveWay: undefined });
    if (editedHere((await readStored(key)) ?? stored, rec)) upFailed(await upload(ctx, key) === "failed");
    return;
  }
  if (changed()) {
    // T-2037: changed on both sides, this game in hand stays as it is: saved over a copy the clock alone
    // moved on (T-2030), else split off as a version of its own beside the account's
    if (saves.dirty) await autosave();
    const r = editedThere(g, rec) ? await splitOff(ctx, key) : await upload(ctx, key, true);
    if (r === "ok" && saves.activeSlot === key) saves.cloudClaimed = key;
    if (saves.activeSlot === key) upFailed(r === "failed");
    return;
  }
  if (rec.badRev === g.rev) return; // T-3788: this copy did not load here
  const got = await fetchGame(ctx, g);
  if (!got) return;
  let took = false;
  // through the game's own queue: nothing edits the game while it is swapped
  await host.inQueue(async () => {
    if (saves.activeSlot !== key || !saves.activeLease) return;
    stored = (await readStored(key)) ?? stored;
    if (changed()) return; // a change came meanwhile: settled at the next save
    // T-3788: the slot is replaced only once the copy has loaded; a copy that fails leaves it as it was
    saves.switching = true;
    try { await loadGame(got.stored.bytes, "autosave"); }
    catch (err) {
      await writeSync(key, { ...rec!, badRev: g.rev }).catch(() => {});
      host.openFailed(err, slotName(key, stored));
      return;
    } finally { saves.switching = false; }
    try { await writeFetched(ctx, key, got); } catch { saves.dirty = true; } // the game in hand is the account's: autosave writes it
    heldFrom(got.stored);
    saves.dirty = false;
    saves.savedAt = got.stored.at;
    took = true;
  });
  if (took && saves.activeSlot === key) {
    const now = await readSync(key);
    if (now && await claim(ctx, now) === "ok") saves.cloudClaimed = key;
  }
}

/** T-1196: a pending file or backup's bytes and the name of the slot it goes into; null if the backup
 * has gone */
export async function pendingGame(p: Exclude<PendingLoad, { kind: "new" }>): Promise<{ bytes: Uint8Array; name: string; version?: { origin?: string; edit: number; at: number }; view?: boolean } | null> {
  // i18n-ignore: a stored name or a developer diagnostic
  if (p.kind === "import") return { bytes: p.bytes, name: p.name || "Loaded game", version: p.origin ? { origin: p.origin, edit: p.edit ?? Date.now(), at: p.edit ?? Date.now() } : undefined, ...(p.view ? { view: true } : {}) };
  const from = backupSlot(p.key);
  if (!from) return null;
  const backup = await readStored(p.key), slot = await readStored(from);
  if (!backup) return null;
  // T-2037: a version of the game it backs up, under its name
  return { bytes: backup.bytes, name: slotName(from, slot), version: { origin: backup.origin ?? slot?.origin, edit: editOf(backup), at: backup.at } };
}

/** T-1196: the page reloads into another game, which frees all of this one's memory: the game in
 * hand is saved and its slot let go (the reloaded page's worker takes the next) */
export async function leave(msg: Extract<ToClock, { kind: "leave" }>, opening = false) {
  try {
    // T-3478: a game still opening has nothing changed to save (its edits wait behind the load), and
    // its model may hold an empty network between steps, so it is let go as it is in storage
    if (opening) saves.switching = true;
    else if (!saves.testPage) await flushBeforeSwitch();
    if (msg.pending && !await writePending(msg.pending)) throw new Error("Browser storage is unavailable. Download this game as a file to keep it.");
    saves.activeLease?.release();
    saves.activeLease = null;
    send({ kind: "left" });
  } catch (err) { send({ kind: "left", error: (err as Error).message }); }
}

/** T-1852: the game being played is deleted (its backups and account copy with it); nothing may write
 * the slot again, and the page then reloads to the start menu on "left". */
async function deletePlaying() {
  const key = saves.activeSlot;
  try {
    if (!saves.activeLease) throw new Error("That slot is open in another tab. Close it before changing this save.");
    saves.switching = true; // autosave stands off from here
    if (saves.saveInFlight) await saves.saveInFlight;
    saves.dirty = saves.edited = false;
    saves.byHandNext = false;
    saves.unwritten = null;
    saves.newSlotName = null;
    await removeSlot(key);
    saves.activeLease.release();
    hold(null);
    saves.activeSlot = "";
    saves.activeName = null;
    const ctx = saves.cloudCtx;
    // the account's copy goes before the page reloads, or the next menu lists it as a game only online
    if (ctx) await Promise.race([cloudRun(() => sendDeletes(ctx)), new Promise((go) => setTimeout(go, 5000))]);
    await publishSlots();
    slotChanges?.postMessage({ key });
    send({ kind: "left" });
  } catch (err) {
    saves.switching = false;
    await publishSlots((err as Error).message);
    send({ kind: "left", error: (err as Error).message });
  }
}

export function autosave(): Promise<boolean> {
  if (saves.saveInFlight) return saves.saveInFlight.then(ok => saves.dirty && ok ? autosave() : ok);
  const api = host.api();
  if (!api || !saves.dirty) return Promise.resolve(true);
  if (writeBlock(true)) return Promise.resolve(false);
  // T-677: a new game whose slot was never written is not kept while its network is empty
  if (saves.unwritten === saves.activeSlot && !api.nodes_info().length && !api.lines_info().length) {
    saves.dirty = saves.edited = false;
    return Promise.resolve(true);
  }
  const key = saves.activeSlot; // an in-flight write must never move to the next slot
  const lease = saves.activeLease;
  const force = saves.byHandNext;
  const wasEdited = saves.edited;
  saves.byHandNext = false;
  saves.dirty = saves.edited = false;
  saves.saveInFlight = (async () => {
    try {
      // T-3869: made and written again while it fails for want of memory, the page's demand workers replaced first
      const { bytes, at, day, summary, edit, name, fresh } = await host.roomFor(async () => {
        const day = Math.floor(gameNow() / 86400);
        const [bytes, summary] = await Promise.all([saveBytes(), slotSummary()]);
        const at = Date.now();
        const name = saves.newSlotName?.key === key ? saves.newSlotName.name : undefined;
        // T-2037: a game from before T-2037 is given its origin by `stampAll`, kept by writeSlot meanwhile
        const edit = saves.editAt || undefined;
        const record: StoredSave = { bytes, at, day, city, ...summary, name, origin: saves.gameOrigin, edit };
        const lastName = saves.activeName?.key === key ? saves.activeName.name : undefined;
        // T-1829: a failed write tries once more, on a connection opened again and, when the browser
        // is out of room, with the demand caches dropped
        // T-3611: the lease may have been taken while the save was built, and the message saying so may not have
        // arrived: the write itself refuses a slot another tab saved since this one read it
        const write = () => { if (saves.activeLease !== lease) throw new SlotTakenError(); return writeSlot(key, record, lastName, saves.seenEdit); };
        const fresh = await write().catch(async (err) => { if (err instanceof SlotTakenError || !await makeRoom(err)) throw err; return write(); });
        return { bytes, at, day, summary, edit, name, fresh };
      }, force);
      // T-1829: a slot this game had written found gone (the browser's storage cleared under it):
      // the game is written again above, and its tie to the account put back
      const first = fresh && saves.unwritten === key;
      const rec = fresh && !first ? lastSync(key) : null;
      if (rec) await writeSync(key, rec);
      if (name) saves.newSlotName = null;
      if (saves.unwritten === key) saves.unwritten = null;
      await writeBackup(key, { bytes, at, day, city, ...summary, origin: saves.gameOrigin, edit }, saves.backupSteps);
      void cloudAfterWrite(key, force, first); // T-988
      saves.savedAt = at;
      saves.seenEdit = saves.editAt;
      saves.saveFailure = undefined;
      send({ kind: "saved", at });
      await publishSlots();
      slotChanges?.postMessage({ key });
      return true;
    } catch (err) {
      saves.dirty = true;
      saves.edited ||= wasEdited;
      if (err instanceof SlotTakenError) {
        // another tab holds the slot now: autosave stays off here until a save by hand takes it back
        if (saves.activeLease === lease) { saves.activeLease = null; saves.leaseTaken = true; }
        await publishSlots();
        return false;
      }
      if (isPanic(err)) { panicLater(err, "autosave"); return false; } // not the browser's fault, and rebuilt then saved
      console.error("the game could not be saved:", err); // i18n-ignore
      saves.savedAt = Date.now(); // retry on the next autosave interval, rather than every tick
      // T-3869: short of memory even after the demand workers were replaced: Download needs memory too, so the player is
      // told to free some rather than sent to it
      saves.saveFailure = host.outOfRoom(err) ? NO_MEMORY_TO_SAVE : "clockWorker.browserCouldNotSave";
      await publishSlots(saves.saveFailure);
      return false;
    } finally { saves.saveInFlight = null; }
  })();
  return saves.saveInFlight;
}

/** T-2037: before this tab writes over slot `key` (whose lease it just took), a player change another
 * tab wrote there that this one never read moves to a slot of its own, a version of the same game
 * still tied to the account game it went up as; this slot is tied to a new one. */
async function keepOtherTabs(key: string) {
  const [stored, rec] = await Promise.all([readStored(key), readSync(key)]);
  if (!stored || editOf(stored) <= saves.seenEdit) return;
  saves.seenEdit = editOf(stored); // read here now: this tab's next save is meant to replace it
  const to = "slot-" + crypto.randomUUID();
  const lease = await leaseSlot(to);
  try {
    await writeSlot(to, { ...stored, origin: stored.origin ?? saves.gameOrigin });
    if (rec) await writeSync(to, rec);
  } finally { lease?.release(); }
  if (rec?.acct) await writeSync(key, { id: newGameId(), acct: rec.acct, rev: 0, syncedAt: 0 });
  else if (rec) await writeSync(key, { ...rec, id: newGameId() });
  slotChanges?.postMessage({ key: to });
}

/** T-1251: the slot written now, autosave on or off. A tab without the slot's lease takes it from
 * the tab holding it, whose autosave then stops; an empty new game is written too. */
async function saveByHand() {
  if (saves.viewing !== null) return void await saveView();
  try {
    if (saves.testPage || !saves.activeSlot) throw new Error("This test page keeps no saved game. Make a copy to keep it.");
    if (!saves.activeLease) {
      const lease = await leaseSlot(saves.activeSlot, true);
      if (!lease) throw new Error("This browser cannot protect save slots. Download the game as a file instead.");
      hold(lease);
      await keepOtherTabs(saves.activeSlot);
    }
    // T-988: a game another device opened is taken back, the account's copy kept there as a backup
    if (saves.cloudTaken === saves.activeSlot) saves.cloudTaken = "";
    if (saves.unwritten === saves.activeSlot) saves.unwritten = null;
    saves.saveFailure = undefined;
    saves.dirty = true;
    saves.byHandNext = true;
    // T-3698: a write refused because another tab saved the slot meanwhile; the next save by hand takes it back
    if (!await autosave()) throw new Error(writeBlock(true) === "otherTab" ? "clockWorker.savedInOtherTab" : saves.saveFailure ?? "clockWorker.couldNotSave");
    send({ kind: "savedByHand", at: saves.savedAt });
  } catch (err) {
    send({ kind: "savedByHand", error: (err as Error).message });
    await publishSlots((err as Error).message);
  }
}

export async function manageSlot(msg: Extract<ToClock, { kind: "saveSlot" }>) {
  try {
    if (msg.action === "refresh") {
      if (saves.viewing !== null) saves.viewCard = await slotSummary().catch(() => saves.viewCard);
      await publishSlots();
      return;
    }
    if (msg.action === "save") { await saveByHand(); return; }
    if (msg.action === "create") {
      const key = "slot-" + crypto.randomUUID();
      const lease = await leaseSlot(key);
      if (!lease) throw new Error("This browser cannot protect save slots. Download the game as a file instead.");
      try {
        // T-2037: a copy is another version of the same game
        saves.gameOrigin ??= newGameId();
        // T-3869: made and written again while it fails for want of memory, as a save by hand is
        await host.roomFor(async () => {
          const [bytes, summary] = await Promise.all([saveBytes(), slotSummary()]);
          // i18n-ignore: a stored name or a developer diagnostic
          await writeSlot(key, { bytes, at: Date.now(), day: Math.floor(gameNow() / 86400), city, name: msg.name?.trim() || "Saved game", ...summary, origin: saves.gameOrigin, edit: saves.editAt || undefined });
        }, true).catch((err) => { throw host.outOfRoom(err) ? new Error(NO_MEMORY_TO_SAVE) : err; });
        saves.seenEdit = saves.editAt;
        void cloudAfterWrite(key, false, true); // T-988: a new game on the account
        // Finish earlier writes before releasing the old slot's guard. The new copy is already safe.
        if (saves.activeLease) await autosave();
        saves.activeLease?.release();
        saves.activeSlot = key;
        hold(lease);
        saves.viewing = saves.viewCard = null;
        saves.saveFailure = undefined;
        saves.dirty = true;
        saves.savedAt = Date.now();
      } catch (err) { lease.release(); throw err; }
    } else if (msg.action === "open") {
      const key = msg.key;
      if (!key || key === saves.activeSlot) { await publishSlots(); return; }
      await flushBeforeSwitch();
      const lease = await leaseSlot(key);
      try {
        // Read after acquiring the guard: the previous owner may have saved just before release.
        const stored = await readStored(key);
        if (!stored) throw new Error("That save slot no longer exists.");
        saves.switching = true;
        saves.loadingInto = key;
        const converted = await loadGame(stored.bytes, "file").finally(() => { saves.switching = false; saves.loadingInto = null; });
        saves.activeLease?.release();
        saves.activeSlot = key;
        saves.slotGame = host.game();
        hold(lease);
        saves.viewing = saves.viewCard = null;
        heldFrom(stored);
        saves.saveFailure = undefined;
        saves.dirty = false;
        saves.savedAt = stored.at;
        if (converted) await saveConverted(key, stored);
      } catch (err) { lease?.release(); throw err; }
    } else if (msg.action === "restore") {
      const key = msg.key, from = backupSlot(key ?? "");
      if (!key || !from) throw new Error("That backup no longer exists.");
      const backup = await readStored(key), slot = await readStored(from);
      if (!backup) throw new Error("That backup no longer exists.");
      // T-2037: a version of the game it backs up, under its name
      await intoNewSlot(slotName(from, slot), () => loadGame(backup.bytes, "file"), true, { origin: backup.origin ?? slot?.origin, edit: editOf(backup), at: backup.at });
      return;
    } else {
      const key = msg.key;
      if (!key) throw new Error("Choose a save slot first.");
      if (msg.action === "delete" && key === saves.activeSlot) { await deletePlaying(); return; }
      const temporary = key === saves.activeSlot ? null : await leaseSlot(key);
      if (key === saves.activeSlot ? !saves.activeLease : !temporary) throw new Error("That slot is open in another tab. Close it before changing this save.");
      try {
        if (msg.action === "rename") await renameSlot(key, msg.name ?? "");
        else await removeSlot(key); // T-988: from the account too
      } finally { temporary?.release(); }
      // T-988: a rename goes up at once; a deletion now, or at the next chance
      const ctx = saves.cloudCtx;
      if (ctx && msg.action === "rename") void cloudRun(async () => { if (await upload(ctx, key) === "ok" && key === saves.activeSlot) saves.cloudClaimed = key; await publishSlots(); });
      else if (ctx) void cloudRun(() => sendDeletes(ctx));
    }
    await publishSlots();
    slotChanges?.postMessage({ key: saves.activeSlot });
  } catch (err) { await publishSlots((err as Error).message); }
}

/** After a slot's save was converted on load: keep the old-format copy under a key the saves list
 * does not show, then write the slot in the new format. */
export async function saveConverted(key: string, old: StoredSave) {
  await writeStored(OLD_FRAME_PREFIX + key, old);
  saves.dirty = true;
  await autosave();
}
