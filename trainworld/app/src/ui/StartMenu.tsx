// T-677: the start menu. Every page load opens on it, unless a URL switch already says which game
// (`choiceFromUrl`): New game (the new game sheet, T-728), then the saved games in this browser, the
// most recent first, each a card as in the saves panel. Nothing heavy has started under it: main.ts
// starts the clock and demand workers with the pick. It covers the page and takes every key until then.
// With no saved game it opens on the new game sheet (T-1441).

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { HOME_CITY, isCity } from "../game/cities";
import { choosing } from "../game/loading";
import { leaseSlot, listSlots, newGameName, type SaveSlot } from "../workers/saveFile";
import { backupChoice, CHOICE_PARAMS, fileChoice, type GameChoice } from "../workers/clockClient";
import { groupHeads, groupVersions, type GameGroup } from "../workers/versions";
import { saveRefusal } from "../game/memoryCheck";
import { blockKeysOutside } from "./keys";
import { NewGameSheet } from "./NewGame";
import { StartLogo } from "./StartLogo";
import { AboutLink } from "./Credits";
import { AccountButton, startAccountUi } from "./AccountPanel";
import { deleteGroupQuestion, pickSaveFile, SlotFacts, SlotThumb, SlotTitle, useBackups, VersionsLine, type VersionActions } from "./SavesPanel";
import { Icon } from "./widgets";
import { cloudContext } from "../game/account";
import { accountBackupFile, accountList, accountOnly, cardOf, deleteAccountGame, downloadAccountGame, settleOnStart, slotsChanged, stopStartSettle } from "../game/accountGames";
import { removeSlot, sendDeletes } from "../workers/cloud";
import { t } from "../i18n";

declare const __TW_VERSION__: string | undefined;
/** The build's commit count (vite.config.ts), "dev" on a dev server. */
const VERSION = typeof __TW_VERSION__ === "string" ? __TW_VERSION__ : "dev";

/** The game the URL chooses, or null for the menu: `?new=1` a new game that owns no slot (T-320),
 * normal in New York unless `&mode=sandbox` or `&city=<zone id>` say otherwise (T-728);
 * `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` the game opened last; `?slot=<key>` and
 * `?load=file|restore|new` the game the saves panel picked (T-1196). */
export function choiceFromUrl(): GameChoice | null {
  const q = new URLSearchParams(location.search);
  const on = (k: string) => q.get(k) === "1";
  if (on("new")) {
    const city = q.get("city");
    return { fresh: true, newGame: true, game: { mode: q.get("mode") === "sandbox" ? "sandbox" : "normal", start: isCity(city) ? city! : HOME_CITY } };
  }
  // T-1196: a page reloaded into the game picked in the saves panel; the switch leaves the address
  // bar, so a reload from there goes to the menu (or, `?paused=1`, the game opened last)
  const slot = q.get("slot"), load = q.get("load");
  if (slot || load) {
    const u = new URL(location.href);
    for (const k of CHOICE_PARAMS) u.searchParams.delete(k);
    history.replaceState(history.state, "", u.href);
  }
  if (slot) return { slot };
  if (load === "new") {
    const city = q.get("city");
    return { pending: true, newGame: !on("real"), game: { mode: q.get("mode") === "sandbox" ? "sandbox" : "normal", ...(isCity(city) ? { start: city! } : {}) } };
  }
  if (load) return { pending: true };
  if (on("synth") || on("demandTest") || on("paused") || on("continue")) return {};
  return null;
}

/** T-927: the view behind the menu, centred around 10° E, 20° N: one zoom closer (T-1593, Anita:
 * "twice as zoomed in") than the zoom at which the band of the world from 60° S to 75° N covers the
 * page. Mercator y of 75° N is 2.028, of 60° S -1.317, so the band is 3.345 / 2π of the world's
 * 512 px tile height at zoom 0. */
export function worldCover(el: HTMLElement): { center: [number, number]; zoom: number } {
  const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
  return { center: [10, 20], zoom: 1 + Math.max(0, Math.log2(w / 512), Math.log2(h / (512 * 3.345 / (2 * Math.PI)))) };
}

/** T-1567: the logo over the menu, in place of the welcome heading; `?logo=0` leaves it out (the
 * menu as before, heading and all). */
const LOGO = new URLSearchParams(location.search).get("logo") !== "0";
/** the logo is shown: main.ts hides the basemap's names behind it */
export const START_LOGO = LOGO;

/** T-988: the start menu waits this long at most for the account before opening a game anyway. */
const SETTLE_WAIT_MS = 10_000;

/** Whether a game was played in this browser before (the remembered slot, clockClient.ts). */
function playedBefore(): boolean {
  try { return !!localStorage.getItem("anitabuilder-save-slot"); } catch { return false; }
}

function Menu({ slots: first0, pick, settling }: { slots: SaveSlot[]; pick: (c: GameChoice) => void; settling: Promise<void> }) {
  const first = useRef<HTMLButtonElement>(null);
  // T-1612: Load from file is for someone who has played here: a saved game, or a game played before
  // (the slot opened last is remembered even once its save is deleted)
  const [returning] = useState(() => first0.length + accountOnly(first0).length > 0 || playedBefore());
  // T-1441: a first visit, with no saved game to pick, opens on the new game sheet
  const [sheet, setSheet] = useState(!returning);
  useEffect(() => { if (!sheet) first.current?.focus(); }, [sheet]);
  // T-988: the slots again once the account has settled them (downloads, deletions, forks)
  const [slots, setSlots] = useState(first0);
  const changed = slotsChanged.value;
  useEffect(() => { if (changed) void listSlots().then(setSlots, () => {}); }, [changed]);
  // T-1197: a saved game too big for this device is refused with the reason
  const [refused, setRefused] = useState<string | null>(null);
  const open = async (key: string) => {
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    // T-988: an account game is downloaded into a new slot first
    const g = key.startsWith("account:") ? accountList.peek()?.games.find((x) => "account:" + x.id === key) : null;
    if (g) {
      const k = await downloadAccountGame(g);
      if (!k) { setRefused(t("startMenu.downloadFailed")); return; }
      key = k;
    }
    const why = await saveRefusal({ key });
    setRefused(why);
    if (!why) pick({ slot: key });
  };
  // T-1612: the card whose delete is being asked about
  const [removing, setRemoving] = useState<string | null>(null);
  const remove = async (s: SaveSlot) => {
    setRemoving(null);
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    if (s.key.startsWith("account:")) {
      if (!await deleteAccountGame(s.key.slice("account:".length))) setRefused(t("startMenu.deleteFailed"));
      return;
    }
    const lease = await leaseSlot(s.key);
    if (!lease) { setRefused(t("startMenu.openElsewhere")); return; }
    try { await removeSlot(s.key); } catch { setRefused(t("startMenu.deleteFailed")); } finally { lease.release(); }
    const ctx = cloudContext();
    if (ctx?.on) void sendDeletes(ctx).catch(() => {});
    setRefused(null);
    setSlots(await listSlots().catch(() => []));
  };
  // T-1612: a save file picked here opens as it does from the saves panel (its game waits in storage)
  const load = () => pickSaveFile(async (file) => {
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    const c = await fileChoice(file);
    if (typeof c === "string") setRefused(c);
    else pick(c);
  });
  const now = Date.now();
  // T-2037: one card per game, its version with the latest player change on top
  const backups = useBackups(slots.map((s) => s.key), slots.length);
  const groups = groupVersions(slots, accountOnly(slots), backups, "", accountList.value?.games ?? []);
  const wait = () => Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
  const chose = (c: GameChoice | string) => { if (typeof c === "string") return c; pick(c); return ""; };
  const actions = (g: GameGroup): VersionActions => ({
    open: async (v) => {
      if (v.kind === "slot") { await open(v.slot.key); return ""; }
      if (v.kind === "account") { await open("account:" + v.game.id); return ""; }
      await wait();
      if (v.kind === "backup") return chose(await backupChoice(v.backup.key, v.slot.key));
      const file = await accountBackupFile(v.game.id, g.top.kind === "slot" ? g.top.slot.name : g.top.game.name, v.backup);
      return file ? chose(await fileChoice(file, { origin: g.origin, edit: v.backup.savedAt * 1000 })) : t("startMenu.backupDownloadFailed");
    },
    remove: async (v) => {
      if (v.kind === "slot" || v.kind === "account") await remove(v.kind === "slot" ? v.slot : cardOf(v.game));
      return "";
    },
  });
  const removeGroup = async (g: GameGroup) => {
    setRemoving(null);
    for (const v of groupHeads(g)) await remove(v.kind === "slot" ? v.slot : cardOf(v.game));
  };
  // T-728: the new game sheet; its game gets a slot of its own named after its start, so no saved game is replaced
  // T-1567: the logo stays mounted over both, so its trains run on when the sheet opens
  // with the logo the welcome heading goes (Anita), the logo naming the menu
  return <div class={LOGO ? "start-menu with-logo" : "start-menu"} role="dialog" aria-modal="true" aria-labelledby={LOGO && !sheet ? undefined : "start-title"} aria-label={LOGO && !sheet ? "trainsworld" : undefined}>
    {LOGO && <StartLogo />}
    <p class="start-version muted small">{t("startMenu.build", { version: VERSION })}<AboutLink class="start-about" /></p>
    {sheet ? <div class="start-box new-game-box">
      <h1 id="start-title" class="big">{t("startMenu.newGame")}</h1>
      <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => pick({ newGame: true, newSlot: newGameName(slots, game), game })} />
    </div> : <div class="start-box">
      <AccountButton kind="start" />
      {!LOGO && <h1 id="start-title" class="big">{t("startMenu.welcome")}</h1>}
      <div class="save-files">
        <button ref={first} class="btn start-new" onClick={() => setSheet(true)}>{t("startMenu.newGame")}</button>
        {returning && <button class="btn start-file" onClick={load}>{t("startMenu.loadFile")}</button>}
      </div>
      {refused && <p class="muted small start-refused" role="alert">{refused}</p>}
      {groups.length > 0 && <>
        <h2 class="start-head">{t("startMenu.savedGames")}</h2>
        <ul class="start-saves">
          {groups.map((g) => {
            const s = g.top.kind === "slot" ? g.top.slot : cardOf(g.top.game);
            return <li key={g.origin} data-origin={g.origin} class={g.others.length ? "has-versions" : undefined}>
              <div class="start-slot">
                <button class="start-save" data-slot={s.key} onClick={() => void open(s.key)}>
                  <SlotThumb slot={s} />
                  <span class="save-info"><SlotTitle slot={s} /><SlotFacts slot={s} now={now} /></span>
                </button>
                <button class="btn icon start-delete" data-tip={t("startMenu.delete")} aria-label={t("startMenu.deleteGame", { name: s.name })} onClick={() => setRemoving(g.origin)}><Icon.bin /></button>
              </div>
              <VersionsLine group={g} now={now} act={actions(g)} />
              {removing === g.origin && <div class="callout bad">
                <p>{deleteGroupQuestion(g)}</p>
                <div class="pane-foot"><button class="btn text start-delete-yes" onClick={() => void removeGroup(g)}>{t("startMenu.deleteSave")}</button><button class="btn text" onClick={() => setRemoving(null)}>{t("startMenu.keepIt")}</button></div>
              </div>}
            </li>;
          })}
        </ul>
      </>}
    </div>}
  </div>;
}

/** Put up the menu; `onPick` gets the player's choice once, after the menu has gone. */
export function showStartMenu(onPick: (c: GameChoice) => void) {
  choosing.value = true;
  document.documentElement.classList.add("choosing");
  startAccountUi();
  const root = document.createElement("div");
  document.body.appendChild(root);
  // the account panel (T-972) opens over the menu and takes the keys while it is open
  const unblock = blockKeysOutside(root, () => !document.querySelector("dialog.acct-panel[open]"));
  let done = false;
  const pick = (c: GameChoice) => {
    if (done) return;
    done = true;
    stopStartSettle();
    unblock();
    render(null, root);
    root.remove();
    document.documentElement.classList.remove("choosing");
    choosing.value = false;
    onPick(c);
  };
  // T-988: the slots settled against the account (signed in) while the menu is up
  const settling = settleOnStart().catch(() => {});
  // storage blocked (a private window): no saved games, New game alone
  void listSlots().catch(() => [] as SaveSlot[]).then((slots) => { if (!done) render(h(Menu, { slots, pick, settling }), root); });
}
