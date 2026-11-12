// T-677: the start menu. Every page load opens on it, unless a URL switch already says which game
// (`choiceFromUrl`): New game (the new game sheet, T-728), then the saved games in this browser, the
// most recent first, each a card as in the saves panel. Nothing heavy has started under it: main.ts
// starts the clock and demand workers with the pick. It covers the page and takes every key until then.
// With no saved game it opens on the new game sheet (T-1441).

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { HOME_CITY, isCity } from "../game/cities";
import { choosing } from "../game/loading";
import { dropMarket, keepMarket } from "../game/ui";
import { failLists, listSlots, newGameName, StorageReadError, type SaveSlot } from "../workers/saveFile";
import { backupChoice, CHOICE_PARAMS, fileChoice, type GameChoice } from "../workers/clockClient";
import { groupHeads, groupVersions, type GameGroup, type Version } from "../workers/versions";
import { saveRefusal } from "../game/memoryCheck";
import { blockKeysOutside } from "./keys";
import { NewGameSheet } from "./NewGame";
import { StartLogo } from "./StartLogo";
import { AboutLink } from "./Credits";
import { AccountButton, startAccountUi } from "./AccountPanel";
import { LanguageSelect } from "./LanguageSelect";
import { DeleteAsk, pickSaveFile, SlotFacts, SlotThumb, SlotTitle, useBackups, VersionsLine, type VersionActions } from "./SavesPanel";
import { Icon } from "./widgets";
import { Changelog } from "./Changelog";
import { CommunityButton } from "./Community";
import { dismissUpdate, updateToShow } from "./updates";
import { accountBackupFile, accountList, accountOnly, cardOf, deleteOwed, deleteVersions, downloadAccountGame, settleOnStart, slotsChanged, stopStartSettle } from "../game/accountGames";
import { t } from "../i18n";

declare const __TW_VERSION__: string | undefined;
/** The build's commit count (vite.config.ts), "dev" on a dev server. */
const VERSION = typeof __TW_VERSION__ === "string" ? __TW_VERSION__ : "dev";

/** The game the URL chooses, or null for the menu: `?new=1` a new game that owns no slot (T-320),
 * normal in New York unless `&mode=sandbox` or `&city=<zone id>` say otherwise (T-728), `&side=left|right` its
 * running side (T-3297);
 * `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` the game opened last; `?slot=<key>` and
 * `?load=file|restore|new` the game the saves panel picked (T-1196). */
export function choiceFromUrl(): GameChoice | null {
  const q = new URLSearchParams(location.search);
  const on = (k: string) => q.get(k) === "1";
  if (on("new")) {
    const city = q.get("city");
    const side = q.get("side");
    return { fresh: true, newGame: true, game: { mode: q.get("mode") === "sandbox" ? "sandbox" : "normal", start: isCity(city) ? city! : HOME_CITY, ...(side === "left" || side === "right" ? { rightHand: side === "right" } : {}) } };
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

/** T-2810: the newest update, with its changelog and a way to hide it (ui/updates.ts). */
function UpdateRow({ onChangelog }: { onChangelog: () => void }) {
  const [update, setUpdate] = useState(updateToShow);
  if (!update) return null;
  const hide = () => { dismissUpdate(update.version); setUpdate(null); };
  return <div class="start-update">
    <span class="start-update-line">{t(update.line)}</span>
    <button class="btn" onClick={onChangelog}>{t("startMenu.changelog")}</button>
    <button class="btn icon mini" aria-label={t("startMenu.hideUpdate")} data-tip={t("startMenu.hideUpdate")} onClick={hide}><Icon.cross /></button>
  </div>;
}

function Menu({ slots: first0, unread: unread0, pick: pickGame, settling }: { slots: SaveSlot[]; unread: boolean; pick: (c: GameChoice) => void; settling: Promise<void> }) {
  // T-3073: every game opened here has inter-city trips unless the warning's lite button asked otherwise
  const pick = (c: GameChoice, lite = false) => { if (lite) dropMarket(); else keepMarket(); pickGame(c); };
  const first = useRef<HTMLButtonElement>(null);
  // T-2810: out in the menu's own element, away from the card's zoom
  const [changelog, setChangelog] = useState(false);
  // T-1612: Load from file is for someone who has played here: a saved game, or a game played before
  // (the slot opened last is remembered even once its save is deleted)
  const [returning] = useState(() => first0.length + accountOnly(first0).length > 0 || playedBefore() || unread0);
  // T-1441: a first visit, with no saved game to pick, opens on the new game sheet
  const [sheet, setSheet] = useState(!returning);
  useEffect(() => { if (!sheet) first.current?.focus(); }, [sheet]);
  // T-988: the slots again once the account has settled them (downloads, deletions, forks)
  const [slots, setSlotsOnly] = useState(first0);
  // T-2705: the saved games could not be read (storage kept failing), as against there being none
  const [unread, setUnread] = useState(unread0);
  const setSlots = (s: SaveSlot[]) => { setSlotsOnly(s); setUnread(false); };
  const changed = slotsChanged.value;
  useEffect(() => { if (changed) void listSlots().then(setSlots, () => {}); }, [changed]);
  // T-1197: a saved game too big for this device is refused with the reason
  const [refused, setRefused] = useState<string | null>(null);
  // T-3061: the device check is a guess (browsers round a device's memory down), so it warns and offers to load anyway
  // T-3073: `go(true)` loads it without inter-city trips; any other pick here has them again
  const [warned, setWarned] = useState<{ text: string; go: (lite?: boolean) => void } | null>(null);
  const open = async (key: string) => {
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    // T-988: an account game is downloaded into a new slot first
    const g = key.startsWith("account:") ? accountList.peek()?.games.find((x) => "account:" + x.id === key) : null;
    if (g) {
      const k = await downloadAccountGame(g);
      if (!k) { setRefused(t("startMenu.downloadFailed")); return; }
      key = k;
    }
    const why = await saveRefusal({ key }, true);
    setRefused(why && !why.soft ? why.text : null);
    setWarned(why?.soft ? { text: why.text, go: (lite) => pick({ slot: key }, lite) } : null);
    if (!why) pick({ slot: key });
  };
  // T-1612: the card whose delete is being asked about
  const [removing, setRemoving] = useState<string | null>(null);
  // T-2738: a card's every version, or one, here and on the account (`deleteVersions`)
  const remove = async (versions: Version[]) => {
    setRemoving(null);
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    setRefused(await deleteVersions(versions) || null);
    await listSlots().then(setSlots, (err) => { if (err instanceof StorageReadError) setUnread(true); else setSlots([]); });
  };
  // T-1612: a save file picked here opens as it does from the saves panel (its game waits in storage)
  const load = () => pickSaveFile(async (file) => {
    await Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
    const c = await fileChoice(file, undefined, true);
    setWarned(null);
    if (typeof c === "string") setRefused(c);
    else if ("warn" in c) setWarned({ text: c.warn, go: async (lite) => { const d = await fileChoice(file); if (typeof d === "string") setRefused(d); else if (!("warn" in d)) pick(d, lite); } });
    else pick(c);
  });
  const now = Date.now();
  // T-2037: one card per game, its version with the latest player change on top
  const backups = useBackups(slots.map((s) => s.key), slots.length);
  const groups = groupVersions(slots, accountOnly(slots), backups, "", accountList.value?.games ?? []);
  const wait = () => Promise.race([settling, new Promise((r) => setTimeout(r, SETTLE_WAIT_MS))]);
  const chose = (c: GameChoice | string | { warn: string }) => { if (typeof c === "string") return c; if ("warn" in c) return c.warn; pick(c); return ""; };
  const actions = (g: GameGroup): VersionActions => ({
    open: async (v) => {
      if (v.kind === "slot") { await open(v.slot.key); return ""; }
      if (v.kind === "account") { await open("account:" + v.game.id); return ""; }
      await wait();
      if (v.kind === "backup") return chose(await backupChoice(v.backup.key, v.slot.key));
      const file = await accountBackupFile(v.game.id, g.top.kind === "slot" ? g.top.slot.name : g.top.game.name, v.backup);
      return file ? chose(await fileChoice(file, { origin: g.origin, edit: v.backup.savedAt * 1000 })) : t("startMenu.backupDownloadFailed");
    },
    remove: async (v) => { await remove([v]); return ""; },
  });
  // T-728: the new game sheet; its game gets a slot of its own named after its start, so no saved game is replaced
  // T-1567: the logo stays mounted over both, so its trains run on when the sheet opens
  // with the logo the welcome heading goes (Anita), the logo naming the menu
  return <div class={LOGO ? "start-menu with-logo" : "start-menu"} role="dialog" aria-modal="true" aria-labelledby={LOGO && !sheet ? undefined : "start-title"} aria-label={LOGO && !sheet ? "trainsworld" : undefined}>
    {LOGO && <StartLogo />}
    <p class="start-version muted small">{t("startMenu.build", { version: VERSION })}<AboutLink class="start-about" /></p>
    {/* T-2753: the language on the card's first row at the right, so a new player (who lands on the sheet) sets it first */}
    {sheet ? <div class="start-box new-game-box">
      <div class="start-titlebar"><h1 id="start-title" class="big">{t("startMenu.newGame")}</h1><LanguageSelect class="start-top" /></div>
      <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => pick({ newGame: true, newSlot: newGameName(slots, game), game })} />
    </div> : <div class="start-box">
      {!LOGO && <h1 id="start-title" class="big">{t("startMenu.welcome")}</h1>}
      <div class="save-files">
        <button ref={first} class="btn start-new" onClick={() => setSheet(true)}>{t("startMenu.newGame")}</button>
        {returning && <button class="btn start-file" onClick={load}>{t("startMenu.loadFile")}</button>}
        <div class="start-top"><LanguageSelect /><AccountButton kind="start" /><CommunityButton start /></div>
      </div>
      {(refused || deleteOwed.value) && <p class="muted small start-refused" role="alert">{refused || t("cloud.deleteOwed")}</p>}
      {warned && !refused && <p class="muted small start-refused" role="alert">{warned.text} <button class="btn" onClick={() => { setWarned(null); warned.go(); }}>{t("startMenu.loadAnyway")}</button> <button class="btn" onClick={() => { setWarned(null); warned.go(true); }}>{t("startMenu.loadWithoutIntercity")}</button></p>}
      {unread && <p class="muted small start-refused start-unread" role="alert">{t("startMenu.savesUnreadable")}</p>}
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
                <DeleteAsk group={g} yesLabel={t("startMenu.deleteSave")} keepLabel={t("startMenu.keepIt")} yesClass="start-delete-yes" onYes={() => void remove(groupHeads(g))} onKeep={() => setRemoving(null)} />
              </div>}
            </li>;
          })}
        </ul>
      </>}
    </div>}
    {!sheet && <UpdateRow onChangelog={() => setChangelog(true)} />}
    <Changelog open={changelog} onClose={() => setChangelog(false)} />
  </div>;
}

/** Put up the menu; `onPick` gets the player's choice once, after the menu has gone. */
export function showStartMenu(onPick: (c: GameChoice) => void) {
  choosing.value = true;
  document.documentElement.classList.add("choosing");
  startAccountUi();
  const root = document.createElement("div");
  document.body.appendChild(root);
  // the account panel (T-972) and Community (a share link, T-3115) open over the menu and take the keys while open
  const unblock = blockKeysOutside(root, () => !document.querySelector("dialog.acct-panel[open], dialog.community-panel[open]"));
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
  // storage blocked (a private window): no saved games, New game alone; storage that kept failing to
  // read says so (T-2705)
  const q = new URLSearchParams(location.search);
  if (q.get("debug") === "1") failLists(Number(q.get("listFails")) || 0);
  void listSlots().then((slots) => ({ slots, unread: false }), (err) => ({ slots: [] as SaveSlot[], unread: err instanceof StorageReadError }))
    .then(({ slots, unread }) => { if (!done) render(h(Menu, { slots, unread, pick, settling }), root); });
}
