// T-988: the account's games on the main thread (notes/T-988.md): the list the start menu and the
// saves panel draw the account's other games from, their pictures, the start menu's settling before
// the clock worker starts, downloading and deleting an account game, the sign-in question, adding a
// game kept here only, the account's backups, and removing the account's games here at sign-out.

import { effect, signal } from "@preact/signals";
import { accountOpen, accountState, cloudContext, deviceId, forgetScoreSlots, session, signOut } from "./account";
import { answerMerge, changedHere, deleteGame, downloadGame, ELSEWHERE_MS, fetchBackup, forkName, gamePicture, listGames, settle, tiedTo, untied, type AccountBackup, type AccountGame, type AccountList } from "../workers/cloud";
import { deleteSlot, leaseSlot, listSlots, readStored, readSync, type SaveSlot } from "../workers/saveFile";
import { clockRunning, importGame, leaveToMenu, postAccount, saveSlotsState } from "../workers/clockClient";

/** The account's games and tombstones, as last listed; null signed out or unreachable. */
export const accountList = signal<AccountList | null>(null);
/** Pictures by game id and picture rev ("id:rev"), as data URLs. */
export const accountPictures = signal<Record<string, string>>({});
/** The slots the sign-in question is about; null when there is nothing to ask. */
export const mergeAsk = signal<SaveSlot[] | null>(null);
/** Moves on whenever this browser's slots changed under the start menu (it lists them again). */
export const slotsChanged = signal(0);

/** Whether the account's games are in use on this page (signed in, a server that keeps them). */
export const gamesOn = () => !!cloudContext()?.on;

/** The account's games not in this browser (no slot here holds their id), newest first. */
export function accountOnly(slots: SaveSlot[]): AccountGame[] {
  const here = new Set(slots.map((s) => s.cloud?.id).filter(Boolean));
  return (accountList.value?.games ?? []).filter((g) => !here.has(g.id));
}

/** When another device opened or saved slot's game in the last ten minutes: that time, else 0. */
export function elsewhereAt(slot: SaveSlot): number {
  const ctx = cloudContext(), g = ctx && tiedTo(slot.cloud, ctx) ? accountList.value?.games.find((x) => x.id === slot.cloud!.id) : undefined;
  if (!g || g.writer === deviceId()) return 0;
  const at = g.writerAt * 1000;
  return Date.now() - at < ELSEWHERE_MS ? at : 0;
}

/** An account game as a card's slot: its picture, facts and name. */
export function cardOf(g: AccountGame): SaveSlot {
  const f = g.facts ?? {}, thumb = g.pictureRev ? accountPictures.value[`${g.id}:${g.pictureRev}`] : undefined;
  return {
    key: "account:" + g.id, name: g.name, at: typeof f.at === "number" ? f.at : g.savedAt * 1000, day: f.day ?? 0, thumb, panels: thumb ? f.panels : undefined, shade: f.shade,
    cities: f.cities ?? [], lines: f.lines, stations: f.stations, cash: f.cash, mode: f.mode,
  };
}

async function pictures(list: AccountList, slots: SaveSlot[]) {
  const ctx = cloudContext();
  if (!ctx) return;
  const have = { ...accountPictures.peek() };
  const here = new Set(slots.map((s) => s.cloud?.id).filter(Boolean));
  for (const g of list.games) {
    const k = `${g.id}:${g.pictureRev}`;
    if (!g.pictureRev || here.has(g.id) || have[k]) continue;
    const p = await gamePicture(ctx, g.id);
    if (p) have[k] = p;
  }
  accountPictures.value = have;
}

let listing: Promise<void> | null = null;
/** List the account's games again (the saves panel opening, after a change), with their pictures. */
export function refreshAccountGames(): Promise<void> {
  const ctx = cloudContext();
  if (!ctx?.on) { accountList.value = null; return Promise.resolve(); }
  listing ??= (async () => {
    try {
      const list = await listGames(ctx);
      if (list) accountList.value = list;
      if (list) await pictures(list, saveSlotsState.peek().slots);
    } finally { listing = null; }
  })();
  return listing;
}

/** Wait until the page knows whether accounts are on (the config answered, or not). */
export function accountKnown(): Promise<void> {
  return new Promise((done) => {
    const stop = effect(() => { if (accountState.value !== "checking") { queueMicrotask(() => stop()); done(); } });
  });
}

let picked = false;
/** The start menu's pick is made: its settling stops before the next slot. */
export function stopStartSettle() { picked = true; }

/** The start menu's settling (T-988 "On load"): every slot this page can lock settled against the
 * account before the clock worker starts, then the account's list and pictures. Resolves (never
 * rejects) once done or found unreachable. */
export function settleOnStart(again = false): Promise<void> {
  const acct = session.peek()?.account.id ?? "";
  if (!again && settling && settlingFor === acct) return settling;
  settlingFor = acct;
  return settling = settleNow();
}
let settling: Promise<void> | null = null, settlingFor = "";
async function settleNow(): Promise<void> {
  await accountKnown();
  if (removing()) return removeOnStart();
  const ctx = cloudContext();
  if (!ctx?.on || picked) return;
  // once a game is picked no other slot is locked here, so the clock worker can take the picked one
  const list = await settle(ctx, "", () => picked).catch(() => null);
  if (!list) return;
  accountList.value = list;
  slotsChanged.value++;
  await pictures(list, await listSlots().catch(() => []));
}

/** An account game downloaded into a new slot; its key, or null with nothing done. */
export async function downloadAccountGame(g: AccountGame): Promise<string | null> {
  const ctx = cloudContext();
  return ctx?.on ? downloadGame(ctx, g) : null;
}

/** An account game not in this browser deleted from the account. */
export async function deleteAccountGame(id: string): Promise<boolean> {
  const ctx = cloudContext();
  if (!ctx?.on || !await deleteGame(ctx, id)) return false;
  await refreshAccountGames();
  return true;
}

// ---------------------------------------------------------------- the sign-in question

/** Ask about this browser's untied games, if there are any (signing in, or a load signed in). */
export async function checkMerge() {
  const ctx = cloudContext();
  if (!ctx?.on) { mergeAsk.value = null; return; }
  const list = untied(await listSlots().catch(() => []));
  mergeAsk.value = list.length ? list : null;
  if (list.length) accountOpen.value = true;
}

/** The sign-in question answered: the games added to the account (they go up now), or kept here only. */
export async function answerMergeAsk(add: boolean) {
  const ctx = cloudContext(), slots = mergeAsk.peek();
  mergeAsk.value = null;
  if (!ctx || !slots) return;
  if (add) return addToAccount(slots);
  await answerMerge(ctx, slots, false);
  slotsChanged.value++;
}

/** Games kept only in this browser (or never asked about) added to the account; they go up now. */
export async function addToAccount(slots: SaveSlot[]) {
  const ctx = cloudContext();
  if (!ctx?.on) return;
  const keys = new Set(slots.map((s) => s.key)), ask = mergeAsk.peek()?.filter((s) => !keys.has(s.key));
  if (ask) mergeAsk.value = ask.length ? ask : null;
  await answerMerge(ctx, slots, true);
  if (clockRunning()) postAccount(ctx, true);
  else await settleOnStart(true);
}

// ---------------------------------------------------------------- account backups

/** The account's backups of slot's game, newest first, when the account listed it (T-1395: then the
 * saves panel shows these and not this browser's); null when it is not on this account or the
 * account did not answer. */
export function accountBackups(slot: SaveSlot | undefined): AccountBackup[] | null {
  const ctx = cloudContext();
  if (!slot || !ctx?.on || !tiedTo(slot.cloud, ctx)) return null;
  const g = accountList.value?.games.find((x) => x.id === slot.cloud!.id);
  return g ? g.backups ?? [] : null;
}

/** An account backup opened as a new saved game, named as a restored backup is ("Tokyo at 9:41 AM"),
 * the game in hand saved first; the copy goes on the account as a game of its own. False if the backup
 * could not be downloaded. */
export async function restoreAccountBackup(slot: SaveSlot, b: AccountBackup): Promise<boolean> {
  const ctx = cloudContext();
  if (!ctx?.on || !slot.cloud) return false;
  const bytes = await fetchBackup(ctx, slot.cloud.id, b.rev);
  if (!bytes) return false;
  await importGame(new File([bytes as BlobPart], `${forkName(slot.name, b.savedAt * 1000)}.save`));
  return true;
}

// ---------------------------------------------------------------- signing out

/** sessionStorage: the account whose games leave this browser at the start menu, then signs out */
const REMOVE_KEY = "trainworld.signout-remove";
const removing = (): string => { try { return sessionStorage.getItem(REMOVE_KEY) ?? ""; } catch { return ""; } };

/** How many slots in this browser are tied to the account signed in (the sign-out question's). */
export async function tiedHere(): Promise<number> {
  const acct = session.peek()?.account.id;
  if (!acct) return 0;
  return (await listSlots().catch(() => [] as SaveSlot[])).filter((s) => s.cloud?.acct === acct && !s.cloud.local).length;
}

/** Sign out, removing this account's games from this browser: from a game the page goes to the start
 * menu first (the game saved), where `removeOnStart` finishes. */
export function signOutRemoving() {
  const acct = session.peek()?.account.id;
  if (!acct) return;
  const mark = () => { try { sessionStorage.setItem(REMOVE_KEY, acct); } catch { /* removed here, now */ } };
  if (clockRunning()) { leaveToMenu(mark); return; }
  mark();
  void settleOnStart(true);
}

/** The removal at sign-out: unsent changes go up first, then every slot of the account whose latest
 * version is on it leaves this browser (one with changes the server could not take stays, tied, for the
 * next sign-in), its figures kept on the account; then the page signs out. */
async function removeOnStart() {
  const acct = removing(), ctx = cloudContext();
  if (ctx?.acct === acct) {
    if (ctx.on) await settle(ctx).catch(() => null);
    const gone: string[] = [];
    for (const s of await listSlots().catch(() => [] as SaveSlot[])) {
      if (s.cloud?.acct !== acct || s.cloud.local) continue;
      const lease = await leaseSlot(s.key);
      if (!lease) continue; // open in another tab
      try {
        const [stored, rec] = await Promise.all([readStored(s.key), readSync(s.key)]);
        if (!stored || !rec || rec.acct !== acct || (changedHere(stored, rec) && !rec.gaveWay)) continue;
        await deleteSlot(s.key);
        gone.push(s.key);
      } catch { /* stays */ } finally { lease.release(); }
    }
    forgetScoreSlots(gone);
    mergeAsk.value = null;
    signOut();
  }
  try { sessionStorage.removeItem(REMOVE_KEY); } catch { /* blocked */ }
  accountList.value = null;
  slotsChanged.value++;
}

let watching = false;
/** Ask the sign-in question whenever someone signs in here (and once on a load already signed in). */
export function watchMerge() {
  if (watching) return;
  watching = true;
  let asked = "";
  effect(() => {
    const acct = session.value?.account.id ?? "", on = accountState.value === "on";
    if (!acct) asked = "";
    if (!acct || !on || acct === asked || removing()) return;
    asked = acct;
    void checkMerge();
    // signed in on the start menu: the account's games come in now
    if (!clockRunning()) void settleOnStart();
  });
}
