// The saved games panel (T-251; the cards from T-241; laid out by T-1395): a centred modal over the
// page. The head's icon says where the games are kept; New game and Open a file are above the list
// (each gets a new slot, T-743). One card per browser slot: a picture of its network, name, when it
// was saved, day, cities, lines, stations and cash. The game being played is marked and has Save, a
// copy, a download and its earlier versions (T-744, T-988); every other card has Open.

import { signal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { cityById, isCity } from "../game/cities";
import { money } from "../game/format";
import { SHADES } from "../game/shades";
import { shade } from "../game/ui";
import { useThumbIn } from "./thumbShade";
import { lastSaved } from "../game/money";
import { storageKept } from "../game/saveSafety";
import { exportGame, exportSlot, importGame, newGame, manageSaveSlot, savedByHand, saveSlotsState } from "../workers/clockClient";
import { listBackups, WHOLE_PANEL, type SaveSlot, type SlotBackup } from "../workers/saveFile";
import { accountConfig, accountOpen, accountState, cloudContext, session } from "../game/account";
import { accountList, accountOnly, addToAccount, cardOf, deleteAccountGame, downloadAccountGame, elsewhereAt, refreshAccountGames, restoreAccountBackup } from "../game/accountGames";
import { tiedTo, type CloudContext } from "../workers/cloud";
import { groupHeads, groupVersions, type GameGroup, type Version } from "../workers/versions";
import { Icon } from "./widgets";
import { viewStart } from "./locate";
import { Modal } from "./Modal";
import { Money } from "./MoneyText";
import { NewGameSheet } from "./NewGame";
import { t, tDate, tList } from "../i18n";

/** The saves panel is open. */
export const savesOpen = signal(false);

/** "Saved just now", "Saved 4 min ago", "Saved today at 14:05", "Saved Oct 12 at 14:05"; "Autosaved" for the game being played. */
function savedWhen(at: number, now: number, auto = false): string {
  const min = Math.floor((now - at) / 60000);
  if (min < 1) return t("savesPanel.savedNow", { auto });
  if (min < 60) return t("savesPanel.savedMinutes", { auto, count: min });
  const d = new Date(at), time = tDate(d, { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === new Date(now).toDateString()) return t("savesPanel.savedToday", { auto, time });
  return t("savesPanel.savedOn", { auto, date: tDate(d, { month: "short", day: "numeric" }), time });
}

/** "New York", "New York and Newark", "New York, Philadelphia and 2 more". */
function cityList(ids: string[]): string {
  const names = ids.filter(isCity).map((id) => cityById(id).name);
  if (names.length <= 3) return tList(names);
  return t("savesPanel.citiesMore", { first: names[0], second: names[1], count: names.length - 2 });
}

const stationsText = (n: number) => t("savesPanel.stations", { count: n });
const fullDate = (at: number) => tDate(at, { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const clockTime = (at: number) => tDate(at, { hour: "numeric", minute: "2-digit" });
const sameDay = (a: number, b: number) => new Date(a).toDateString() === new Date(b).toDateString();
/** "3:31 PM" today, "Oct 24 at 3:31 PM" before. */
const atTime = (at: number, now: number) => sameDay(at, now) ? clockTime(at) : t("savesPanel.dateAt", { date: tDate(at, { month: "short", day: "numeric" }), time: clockTime(at) });

/** A saved game's picture of its network (T-241): one panel per city it has built in, the most
 * stations first, each named, with its stations when there is more than one (T-894); built in more
 * than 3 cities, the last panel is the whole network (T-2394). The names are text over the picture,
 * in the colours of the theme it was drawn in. */
export function SlotThumb({ slot }: { slot: SaveSlot }) {
  // T-1852: drawn in the theme now set, whichever it was saved in
  const now = shade.value;
  const src = useThumbIn(slot.thumb ?? "", slot.shade ?? "light", now);
  if (!slot.thumb) return <div class="save-thumb empty muted small">{slot.loading ? t("savesPanel.loading") : slot.lines !== undefined ? t("savesPanel.nothingBuilt") : t("savesPanel.noPicture")}</div>;
  const panels = (slot.panels ?? []).filter((p) => p.city === WHOLE_PANEL || isCity(p.city));
  const sh = SHADES[now];
  return <div class="save-thumb" style={{ "--thumb-ink": sh.mapInk, "--thumb-land": sh.land }}>
    {src && <img src={src} alt="" data-tip={t("savesPanel.thumbTip", { name: slot.name })} />}
    {panels.map((p) => {
      const name = p.city === WHOLE_PANEL ? t("savesPanel.wholeNetwork") : cityById(p.city).name, stations = stationsText(p.stations);
      return <div class="thumb-label" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%`, width: `${p.w * 100}%` }} data-tip={t("savesPanel.panelTip", { name, stations })}>
        <div class="thumb-city">{name}</div>
        {panels.length > 1 && <div class="thumb-figure num">{stations}</div>}
      </div>;
    })}
  </div>;
}

/** When a saved game was saved (with `mark` after it), its day and cities, its lines, stations and
 * cash; also the start menu's (T-677). */
export function SlotFacts({ slot, now, autosaved, mark }: { slot: SaveSlot; now: number; autosaved?: boolean; mark?: ComponentChildren }) {
  const when = savedWhen(slot.at, now, !!autosaved);
  const cities = cityList(slot.cities ?? []);
  return <>
    <div class="small muted save-when"><span data-tip={t("savesPanel.lastSaved", { date: fullDate(slot.at) })}>{when}</span>{mark}</div>
    <div class="small muted num">
      <span data-tip={t("savesPanel.gameDay", { day: slot.day })}>{t("savesPanel.day", { day: slot.day })}</span>
      {cities ? <> · {cities}</> : null}
    </div>
    {slot.lines !== undefined && <div class="small muted num">
      <span data-tip={t("savesPanel.lines", { count: slot.lines })}>{t("savesPanel.lines", { count: slot.lines })}</span>
      {" · "}
      <span data-tip={stationsText(slot.stations!)}>{stationsText(slot.stations!)}</span>
      {slot.mode !== "sandbox" && <>
        {" · "}
        <span data-tip={t("savesPanel.cash", { amount: money(slot.cash! * 1e6) })}><Money dollars={slot.cash! * 1e6} /></span>
      </>}
    </div>}
  </>;
}

/** A saved game's name, with a mark when it is a sandbox (T-728); also the start menu's. While the
 * account keeps games (T-988), a cloud marks a game only on the account and a crossed cloud one only
 * in this browser. `panel`: the saves panel's, where that crossed cloud adds the game to the account. */
export function SlotTitle({ slot, panel }: { slot: SaveSlot; panel?: boolean }) {
  return <strong>{slot.name}{slot.mode === "sandbox" && <span class="save-sandbox small" data-tip={t("savesPanel.sandboxTip")}>{t("savesPanel.sandbox")}</span>}<CloudMark slot={slot} panel={panel} /></strong>;
}

/** T-988: the account's games are in use here (signed in, a server that keeps them); read as signals. */
function useGamesOn(): CloudContext | null {
  void session.value; void accountState.value; void accountConfig.value;
  const ctx = cloudContext();
  return ctx?.on ? ctx : null;
}

function CloudMark({ slot, panel }: { slot: SaveSlot; panel?: boolean }) {
  const ctx = useGamesOn();
  if (!ctx) return null;
  if (slot.key.startsWith("account:")) return <span class="save-cloud" data-tip={panel ? t("savesPanel.onlineOnly") : t("savesPanel.onAccount")}><Icon.cloud /></span>;
  const noAccount = !slot.cloud?.acct;
  if (tiedTo(slot.cloud, ctx)) return null;
  if (panel && noAccount) return <button class="save-cloud save-add" data-tip={t("savesPanel.addTip")} aria-label={t("savesPanel.addLabel", { name: slot.name })}
    onClick={(e) => { e.stopPropagation(); void addToAccount([slot]); }}><Icon.cloudOff /></button>;
  return <span class="save-cloud" data-tip={t("savesPanel.browserOnly")}><Icon.cloudOff /></span>;
}

/** T-988: after the playing game's "Autosaved", whether its last save reached the account. */
function AccountMark({ slot, now }: { slot: SaveSlot; now: number }) {
  const ctx = useGamesOn(), c = slot.cloud;
  if (!ctx || !tiedTo(c, ctx) || c.gaveWay) return null;
  if (c.error || !c.okAt) return <span class="save-cloud save-account warn" data-tip={c.error ? t("savesPanel.notSyncedWhy", { why: c.error.replace(/([^.])$/, "$1.") }) : t("savesPanel.notSynced")}><Icon.cloudOff /></span>;
  return <span class="save-cloud save-account" data-tip={t("savesPanel.syncedAt", { time: atTime(c.okAt, now) })}><Icon.cloud /></span>;
}

/** T-988: another device opened or saved this game in the last ten minutes. */
function Elsewhere({ slot }: { slot: SaveSlot }) {
  const ctx = useGamesOn();
  void accountList.value;
  const at = ctx ? elsewhereAt(slot) : 0;
  return at ? <div class="small muted save-elsewhere" data-tip={t("savesPanel.elsewhereTip", { date: fullDate(at) })}>{t("savesPanel.elsewhere")}</div> : null;
}

/** T-1247: a save's name typed over in place; Enter or leaving the box renames, Escape cancels. */
function NameField({ slot, onDone }: { slot: SaveSlot; onDone: () => void }) {
  const field = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  // the name is put in once, not as `value`, so a render while typing keeps what is typed
  useLayoutEffect(() => {
    const f = field.current;
    if (!f) return;
    f.value = slot.name;
    f.focus();
    f.select();
  }, []);
  const finish = (keep: boolean) => {
    if (done.current) return;
    done.current = true;
    const name = field.current?.value.trim() ?? "";
    if (keep && name && name !== slot.name) manageSaveSlot("rename", slot.key, name);
    onDone();
  };
  return <input ref={field} type="text" class="save-name-field" aria-label={t("savesPanel.nameLabel")} maxLength={80}
    onKeyDown={(e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    }}
    onBlur={() => finish(true)} />;
}

const openTip = () => t("savesPanel.openTip");

/** A saved game's buttons, the foot of its card at the right (in line with its other versions). */
function SlotActions({ slot, playing, writable, copying, onCopy, onDelete }: { slot: SaveSlot; playing: boolean; writable: boolean; copying: boolean; onCopy: () => void; onDelete: () => void }) {
  const [msg, setMsg] = useState("");
  const newer = saveSlotsState.value.newer;
  const saveTip = writable ? t("savesPanel.saveTip") : newer ? t("savesPanel.saveNewer") : /another device/.test(saveSlotsState.value.reason ?? "") ? t("savesPanel.saveOtherDevice") : t("savesPanel.saveOtherTab");
  return (
    <div class="save-actions">
      {msg && <span class="small bad">{msg}</span>}
      {playing ? <>
        <button class="btn icon save-copy" data-tip={t("savesPanel.copyTip")} aria-label={t("savesPanel.copy")} disabled={copying} onClick={onCopy}><Icon.copy /></button>
        <button class="btn icon save-download" data-tip={newer ? t("savesPanel.downloadNewer") : t("savesPanel.download")} aria-label={t("savesPanel.download")} onClick={exportGame}><Icon.download /></button>
        <button class="btn icon save-delete" data-tip={t("savesPanel.delete")} aria-label={t("savesPanel.delete")} onClick={onDelete}><Icon.bin /></button>
        <button class="btn text save-now" data-tip={saveTip} onClick={() => manageSaveSlot("save")}>{t("savesPanel.save")}</button>
      </> : <>
        <button class="btn icon save-download" data-tip={t("savesPanel.download")} aria-label={t("savesPanel.download")}
          onClick={async () => { setMsg(""); if (!await exportSlot(slot)) setMsg(t("savesPanel.readFailed")); }}><Icon.download /></button>
        <button class="btn icon save-delete" data-tip={t("savesPanel.delete")} aria-label={t("savesPanel.delete")} onClick={onDelete}><Icon.bin /></button>
        <button class="btn text save-open" data-tip={openTip()} onClick={() => { manageSaveSlot("open", slot.key); savesOpen.value = false; }}>{t("savesPanel.open")}</button>
      </>}
    </div>
  );
}

/** One saved game in this browser. */
function SlotCard({ slot, playing, now, writable, renaming, onRename, onRenamed }: { slot: SaveSlot; playing: boolean; now: number; writable: boolean; renaming: boolean; onRename: () => void; onRenamed: () => void }) {
  const byHand = savedByHand.value;
  const autosaved = playing && writable && !(byHand && Math.abs(byHand - slot.at) < 2000);
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title">
          {renaming
            ? <NameField slot={slot} onDone={onRenamed} />
            : <span class="save-name" onDblClick={onRename}><SlotTitle slot={slot} panel />{playing && <button class="save-pen" data-tip={t("savesPanel.rename")} aria-label={t("savesPanel.rename")} onClick={onRename}><Icon.pencil /></button>}</span>}
          {playing && <span class="save-playing small">{t("savesPanel.playing")}</span>}
        </div>
        <SlotFacts slot={slot} now={now} autosaved={autosaved} mark={playing ? <AccountMark slot={slot} now={now} /> : null} />
        {!playing && <Elsewhere slot={slot} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- T-2037: a game's other versions

/** The cards whose other versions are listed, by origin (kept for the visit, in the menu and panel). */
const versionsShown = signal<ReadonlySet<string>>(new Set());

/** Each slot's backups (main thread, from storage), read again when `stamp` moves. */
export function useBackups(keys: string[], stamp: number): Record<string, SlotBackup[]> {
  const [got, setGot] = useState<Record<string, SlotBackup[]>>({});
  const id = keys.join(",");
  useEffect(() => {
    let live = true;
    void Promise.all(keys.map(async (k) => [k, await listBackups(k).catch(() => [])] as const)).then((l) => { if (live) setGot(Object.fromEntries(l)); });
    return () => { live = false; };
  }, [id, stamp]);
  return got;
}

/** What a version's row says and its place in storage. */
function versionFacts(v: Version): { day?: number; stations?: number; tip?: string; cloud?: string } {
  switch (v.kind) {
    case "slot": return { day: v.slot.day, stations: v.slot.stations };
    case "account": return { day: v.game.facts?.day, stations: v.game.facts?.stations, cloud: t("savesPanel.accountCopy") };
    case "backup": return { day: v.backup.day, stations: v.backup.stations, tip: v.backup.pull ? t("savesPanel.backupPulled") : t("savesPanel.backupHere") };
    case "accountBackup": return { tip: t("savesPanel.backupAccount"), cloud: t("savesPanel.backupAccount") };
  }
}
const versionId = (v: Version) => v.kind === "slot" ? v.slot.key : v.kind === "account" ? "account:" + v.game.id : v.kind === "backup" ? v.backup.key : `account-backup:${v.game.id}:${v.backup.rev}`; // i18n-ignore (id)

/** "1 other version", "11 other versions". */
export const otherVersions = (n: number) => t("savesPanel.otherVersions", { count: n });

/** What opening and deleting a version does where the card is (the saves panel, the start menu);
 * each answers a reason to show, or "" when done. */
export interface VersionActions { open: (v: Version) => Promise<string>; remove: (v: Version) => Promise<string> }

/** The line at the bottom of a game's card, "11 other versions", and under it, opened, each other
 * version: when it was saved, its day and stations, Open, and for a slot or account game a delete. */
export function VersionsLine({ group, now, act, children }: { group: GameGroup; now: number; act: VersionActions; children?: ComponentChildren }) {
  const [asking, setAsking] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const n = group.others.length;
  if (!n && !children) return null;
  const open = versionsShown.value.has(group.origin);
  const toggle = () => {
    const s = new Set(versionsShown.peek());
    if (!s.delete(group.origin)) s.add(group.origin);
    versionsShown.value = s;
  };
  const run = async (p: Promise<string>) => { setMsg(""); const why = await p; if (why) setMsg(why); };
  return <div class="save-versions">
    <div class="save-foot">
      {n > 0 && <button class={"save-versions-line small" + (open ? " open" : "")} aria-expanded={open} onClick={toggle}><Icon.chevron />{otherVersions(n)}</button>}
      {children}
    </div>
    {open && <ul class="save-version-list">
      {group.others.map((v) => {
        const id = versionId(v), f = versionFacts(v), deletable = v.kind === "slot" || v.kind === "account";
        const where = v.kind === "account" ? "account" : v.kind === "slot" && onAccount(v.slot) ? "both" : "browser";
        return <li key={id} data-version={id} data-kind={v.kind}>
          <div class="save-version">
            <span class="small muted num" data-tip={f.tip}>
              {savedWhen(v.at, now)}{f.day !== undefined && <> · {t("savesPanel.day", { day: f.day })}</>}{f.stations !== undefined && <> · {stationsText(f.stations)}</>}
              {f.cloud && <span class="save-cloud" data-tip={f.cloud}><Icon.cloud /></span>}
            </span>
            {deletable && <button class="btn icon save-delete" data-tip={t("savesPanel.delete")} aria-label={t("savesPanel.deleteVersionLabel")} onClick={() => setAsking(id)}><Icon.bin /></button>}
            <button class="btn text save-version-open" onClick={() => void run(act.open(v))}>{t("savesPanel.open")}</button>
          </div>
          {asking === id && <div class="callout bad">
            <p>{t("savesPanel.deleteVersionQuestion", { where })}</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { setAsking(null); void run(act.remove(v)); }}>{t("savesPanel.deleteVersion")}</button><button class="btn text" onClick={() => setAsking(null)}>{t("savesPanel.keepIt")}</button></div>
          </div>}
        </li>;
      })}
    </ul>}
    {msg && <div class="small bad">{msg}</div>}
  </div>;
}

/** The question before a card's game is deleted with every version (T-2037); also the start menu's. */
export function deleteGroupQuestion(group: GameGroup, playing = false): string {
  const name = group.top.kind === "slot" ? group.top.slot.name : group.top.game.name;
  const heads = groupHeads(group);
  const browser = heads.some((v) => v.kind === "slot"), account = heads.some((v) => v.kind === "account" || onAccount(v.slot));
  const where = browser && account ? "both" : account ? "account" : "browser";
  const what = group.others.length ? t("savesPanel.deleteWhatMore", { name, versions: otherVersions(group.others.length) }) : t("savesPanel.deleteWhat", { name });
  return t("savesPanel.deleteGameQuestion", { playing, what, where });
}

/** The head's icon: signed out a crossed cloud that opens Sign in, signed in a cloud that opens the
 * account; where there are no accounts, a crossed cloud only while the browser may drop the saves. */
function HeadMark() {
  const s = session.value;
  if (accountState.value === "on") {
    const a = s?.account;
    const tip = !a ? t("savesPanel.signedOutTip") : a.username ? t("savesPanel.signedInAs", { name: a.username }) : a.logins.includes("google") ? t("savesPanel.signedInGoogle") : t("savesPanel.signedIn");
    return <button class="btn icon save-head-mark" data-tip={tip} aria-label={a ? t("savesPanel.account") : t("savesPanel.signIn")} onClick={() => (accountOpen.value = true)}>{a ? <Icon.cloud /> : <Icon.cloudOff />}</button>;
  }
  if (storageKept.value === false) return <span class="save-head-mark" data-tip={t("savesPanel.mayBeLost")}><Icon.cloudOff /></span>;
  return null;
}

/** The browser's file picker for a .save file; also the start menu's (T-1612). */
export function pickSaveFile(onFile: (file: File) => void) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".save";
  input.onchange = () => { if (input.files?.[0]) onFile(input.files[0]); };
  input.click();
}

/** "New York copy" for the game being played; "" when it has no slot (a test page). */
const copyName = (name: string | undefined) => (name ? t("savesPanel.copyName", { name }).slice(0, 80) : "");

function Saves({ onNewGame }: { onNewGame: () => void }) {
  const [rename, setRename] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  // the slot a copy was made from, until the copy is the slot played (its name then opens for typing)
  const [copyFrom, setCopyFrom] = useState<string | null>(null);
  // "2 min ago" moves on while the panel is open
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 20_000);
    return () => clearInterval(t);
  }, []);
  const saves = saveSlotsState.value;
  useEffect(() => {
    if (copyFrom === null) return;
    if (saves.active !== copyFrom) { setRename(saves.active); setCopyFrom(null); }
    else if (saves.error) setCopyFrom(null);
  }, [saves]);
  const at = lastSaved.value ?? 0;
  const ctx = useGamesOn();
  void accountList.value;
  const pick = () => pickSaveFile((file) => { void importGame(file); savesOpen.value = false; });
  const copy = () => {
    setCopyFrom(saves.active);
    manageSaveSlot("create", undefined, copyName(saves.slots.find((s) => s.key === saves.active)?.name));
  };
  const shownNow = Math.max(now, at);
  // T-2037: one card per game
  const backups = useBackups(saves.slots.map((s) => s.key), Math.floor(Math.max(0, ...saves.slots.map((s) => s.at)) / 60_000));
  const groups = groupVersions(saves.slots, ctx ? accountOnly(saves.slots) : [], backups, saves.active, ctx ? accountList.value?.games ?? [] : []);
  const shut = () => { savesOpen.value = false; return ""; };
  const actions = (g: GameGroup): VersionActions => ({
    open: async (v) => {
      if (v.kind === "slot") { manageSaveSlot("open", v.slot.key); return shut(); }
      if (v.kind === "backup") { manageSaveSlot("restore", v.backup.key); return shut(); }
      if (v.kind === "account") {
        const key = await downloadAccountGame(v.game);
        if (!key) return t("savesPanel.downloadFailed");
        manageSaveSlot("open", key);
        return shut();
      }
      const name = g.top.kind === "slot" ? g.top.slot.name : g.top.game.name;
      return await restoreAccountBackup(v.game.id, name, g.origin, v.backup) ? shut() : t("savesPanel.backupDownloadFailed");
    },
    remove: async (v) => {
      if (v.kind === "slot") manageSaveSlot("delete", v.slot.key);
      else if (v.kind === "account" && !await deleteAccountGame(v.game.id)) return t("savesPanel.deleteFailed");
      return "";
    },
  });
  // the card's delete: every slot and account game of the game, the one being played last
  const removeGroup = (g: GameGroup) => {
    for (const v of groupHeads(g)) {
      if (v.kind === "account") void deleteAccountGame(v.game.id);
      else if (v.slot.key !== saves.active) manageSaveSlot("delete", v.slot.key);
    }
    if (groupHeads(g).some((v) => v.kind === "slot" && v.slot.key === saves.active)) manageSaveSlot("delete", saves.active);
  };
  return (
    <>
      {!saves.writable && saves.reason && <p class="callout bad">{saves.reason}</p>}
      {saves.error && <p class="callout bad">{saves.error}</p>}
      <div class="save-files">
        <button class="btn text save-new" onClick={onNewGame}>{t("savesPanel.newGame")}</button>
        <button class="btn text" data-tip={t("savesPanel.openFileTip")} onClick={pick}>{t("savesPanel.openFile")}</button>
        {/* a test page's game has no card to copy it from */}
        {!saves.slots.some((s) => s.key === saves.active) && <button class="btn icon save-copy" data-tip={t("savesPanel.copy")} aria-label={t("savesPanel.copy")} disabled={copyFrom !== null} onClick={copy}><Icon.copy /></button>}
      </div>
      <ul class="save-slots">
        {groups.map((g) => {
          const top = g.top, slot = top.kind === "slot" ? top.slot : cardOf(top.game), playing = slot.key === saves.active;
          return <li key={g.origin} data-origin={g.origin} {...(top.kind === "slot" ? { "data-slot": slot.key } : { "data-account": top.game.id })} class={playing ? "on" : ""}>
            {top.kind === "slot"
              ? <SlotCard slot={slot} playing={playing} now={shownNow} writable={saves.writable} renaming={rename === slot.key}
                onRename={() => setRename(slot.key)} onRenamed={() => setRename(null)} />
              : <AccountCard slot={slot} now={shownNow} />}
            <VersionsLine group={g} now={shownNow} act={actions(g)}>
              {top.kind === "slot"
                ? <SlotActions slot={slot} playing={playing} writable={saves.writable} copying={copyFrom !== null} onCopy={copy} onDelete={() => setRemoving(g.origin)} />
                : <AccountActions slot={slot} onDelete={() => setRemoving(g.origin)} />}
            </VersionsLine>
            {removing === g.origin && <div class="callout bad">
              <p>{deleteGroupQuestion(g, playing)}</p>
              <div class="pane-foot"><button class="btn text" onClick={() => { removeGroup(g); setRemoving(null); }}>{t("savesPanel.deleteGame")}</button><button class="btn text" onClick={() => setRemoving(null)}>{t("savesPanel.keepIt")}</button></div>
            </div>}
          </li>;
        })}
      </ul>
    </>
  );
}

/** T-988: a game on the account and not in this browser; Open downloads it into a new slot. */
function AccountCard({ slot, now }: { slot: SaveSlot; now: number }) {
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title"><span class="save-name"><SlotTitle slot={slot} panel /></span></div>
        <SlotFacts slot={slot} now={now} />
      </div>
    </div>
  );
}

function AccountActions({ slot, onDelete }: { slot: SaveSlot; onDelete: () => void }) {
  const [msg, setMsg] = useState("");
  const g = accountList.value?.games.find((x) => "account:" + x.id === slot.key);
  return (
    <div class="save-actions">
      {msg && <span class="small bad">{msg}</span>}
      <button class="btn icon save-delete" data-tip={t("savesPanel.delete")} aria-label={t("savesPanel.delete")} onClick={onDelete}><Icon.bin /></button>
      <button class="btn text save-open" data-tip={openTip()} onClick={async () => {
        if (!g) return;
        setMsg("");
        const key = await downloadAccountGame(g);
        if (!key) { setMsg(t("savesPanel.downloadFailed")); return; }
        manageSaveSlot("open", key);
        savesOpen.value = false;
      }}>{t("savesPanel.open")}</button>
    </div>
  );
}

/** T-988: a slot whose game is also on the account signed in. */
function onAccount(slot: SaveSlot): boolean {
  const ctx = cloudContext();
  return !!ctx?.on && tiedTo(slot.cloud, ctx);
}

/** The modal. */
export function SavesPanel() {
  const open = savesOpen.value;
  // T-728: New game shows the new game sheet in the panel's place
  const [sheet, setSheet] = useState(false);
  useEffect(() => { if (!open) setSheet(false); }, [open]);
  useEffect(() => { if (open) { manageSaveSlot("refresh"); void refreshAccountGames(); } }, [open]);
  return (
    <Modal open={open} onClose={() => (savesOpen.value = false)} title={sheet ? t("savesPanel.newGame") : t("savesPanel.title")} head={sheet ? null : <HeadMark />}>
      {sheet
        ? <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => { newGame(game); viewStart(game.start); savesOpen.value = false; }} />
        : <Saves onNewGame={() => setSheet(true)} />}
    </Modal>
  );
}
