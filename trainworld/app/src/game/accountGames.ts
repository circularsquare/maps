// T-988: the account's games on the main thread (notes/T-988.md): the list the start menu and the
// saves panel draw the account's other games from, their pictures, the start menu's settling before
// the clock worker starts, downloading and deleting an account game, and the sign-in question.

import { effect, signal } from "@preact/signals";
import { accountOpen, accountState, cloudContext, deviceId, session } from "./account";
import { answerMerge, deleteGame, downloadGame, ELSEWHERE_MS, gamePicture, listGames, settle, tiedTo, untied, type AccountGame, type AccountList } from "../workers/cloud";
import { listSlots, type SaveSlot } from "../workers/saveFile";
import { clockRunning, postAccount, saveSlotsState } from "../workers/clockClient";

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
  await answerMerge(ctx, slots, add);
  if (!add) { slotsChanged.value++; return; }
  if (clockRunning()) postAccount(ctx, true);
  else await settleOnStart(true);
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
    if (!acct || !on || acct === asked) return;
    asked = acct;
    void checkMerge();
    // signed in on the start menu: the account's games come in now
    if (!clockRunning()) void settleOnStart();
  });
}
