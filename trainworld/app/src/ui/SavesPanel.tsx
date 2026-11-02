// The saved games panel (T-251; the cards from T-241; laid out by T-1395): a centred modal over the
// page. The head's icon says where the games are kept; New game and Open a file are above the list
// (each gets a new slot, T-743). One card per browser slot: a picture of its network, name, when it
// was saved, day, cities, lines, stations and cash. The game being played is marked and has Save, a
// copy, a download and its earlier versions (T-744, T-988); every other card has Open.

import { signal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { cityById, isCity } from "../game/cities";
import { money, plural } from "../game/format";
import { SHADES } from "../game/shades";
import { shade } from "../game/ui";
import { useThumbIn } from "./thumbShade";
import { lastSaved } from "../game/money";
import { storageKept } from "../game/saveSafety";
import { exportGame, exportSlot, importGame, newGame, manageSaveSlot, savedByHand, saveSlotsState } from "../workers/clockClient";
import { listBackups, type SaveSlot, type SlotBackup } from "../workers/saveFile";
import { accountConfig, accountOpen, accountState, cloudContext, session } from "../game/account";
import { accountList, accountOnly, addToAccount, cardOf, deleteAccountGame, downloadAccountGame, elsewhereAt, refreshAccountGames, restoreAccountBackup } from "../game/accountGames";
import { tiedTo, type CloudContext } from "../workers/cloud";
import { groupHeads, groupVersions, type GameGroup, type Version } from "../workers/versions";
import { Icon } from "./widgets";
import { viewStart } from "./locate";
import { Modal } from "./Modal";
import { Money } from "./MoneyText";
import { NewGameSheet } from "./NewGame";

/** The saves panel is open. */
export const savesOpen = signal(false);

/** "Saved just now", "Saved 4 min ago", "Saved today at 14:05", "Saved Oct 12 at 14:05". */
function savedWhen(at: number, now: number): string {
  const min = Math.floor((now - at) / 60000);
  if (min < 1) return "Saved just now";
  if (min < 60) return `Saved ${min} min ago`;
  const d = new Date(at), time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === new Date(now).toDateString()) return `Saved today at ${time}`;
  return `Saved ${d.toLocaleDateString([], { month: "short", day: "numeric" })} at ${time}`;
}

/** "New York", "New York and Newark", "New York, Philadelphia and 2 more". */
function cityList(ids: string[]): string {
  const names = ids.filter(isCity).map((id) => cityById(id).name);
  if (names.length <= 2) return names.join(" and ");
  if (names.length === 3) return `${names[0]}, ${names[1]} and ${names[2]}`;
  return `${names[0]}, ${names[1]} and ${names.length - 2} more`;
}

const fullDate = (at: number) => new Date(at).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const clockTime = (at: number) => new Date(at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const sameDay = (a: number, b: number) => new Date(a).toDateString() === new Date(b).toDateString();
/** "3:31 PM" today, "Oct 24 at 3:31 PM" before. */
const atTime = (at: number, now: number) => sameDay(at, now) ? clockTime(at) : `${new Date(at).toLocaleDateString([], { month: "short", day: "numeric" })} at ${clockTime(at)}`;

/** A saved game's picture of its network (T-241): one panel per city it has built in, the most
 * stations first, each named, with its stations when there is more than one (T-894). The names are
 * text over the picture, in the colours of the theme it was drawn in. */
export function SlotThumb({ slot }: { slot: SaveSlot }) {
  // T-1852: drawn in the theme now set, whichever it was saved in
  const now = shade.value;
  const src = useThumbIn(slot.thumb ?? "", slot.shade ?? "light", now);
  if (!slot.thumb) return <div class="save-thumb empty muted small">{slot.lines !== undefined ? "Nothing built" : "No picture yet"}</div>;
  const panels = (slot.panels ?? []).filter((p) => isCity(p.city));
  const sh = SHADES[now];
  return <div class="save-thumb" style={{ "--thumb-ink": sh.mapInk, "--thumb-land": sh.land }}>
    {src && <img src={src} alt="" data-tip={`The network in "${slot.name}" when it was last saved`} />}
    {panels.map((p) => {
      const name = cityById(p.city).name, stations = plural(p.stations, "station");
      return <div class="thumb-label" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%`, width: `${p.w * 100}%` }} data-tip={`${name}: ${stations}`}>
        <div class="thumb-city">{name}</div>
        {panels.length > 1 && <div class="thumb-figure num">{stations}</div>}
      </div>;
    })}
  </div>;
}

/** When a saved game was saved (with `mark` after it), its day and cities, its lines, stations and
 * cash; also the start menu's (T-677). */
export function SlotFacts({ slot, now, autosaved, mark }: { slot: SaveSlot; now: number; autosaved?: boolean; mark?: ComponentChildren }) {
  const when = savedWhen(slot.at, now);
  const cities = cityList(slot.cities ?? []);
  return <>
    <div class="small muted save-when"><span data-tip={`Last saved ${fullDate(slot.at)}`}>{autosaved ? when.replace("Saved", "Autosaved") : when}</span>{mark}</div>
    <div class="small muted num">
      <span data-tip={`Game day ${slot.day}`}>Day {slot.day}</span>
      {cities ? <> · {cities}</> : null}
    </div>
    {slot.lines !== undefined && <div class="small muted num">
      <span data-tip={`${slot.lines} lines`}>{plural(slot.lines!, "line")}</span>
      {" · "}
      <span data-tip={`${slot.stations} stations`}>{plural(slot.stations!, "station")}</span>
      {slot.mode !== "sandbox" && <>
        {" · "}
        <span data-tip={`Cash when saved: ${money(slot.cash! * 1e6)}`}><Money dollars={slot.cash! * 1e6} /></span>
      </>}
    </div>}
  </>;
}

/** A saved game's name, with a mark when it is a sandbox (T-728); also the start menu's. While the
 * account keeps games (T-988), a cloud marks a game only on the account and a crossed cloud one only
 * in this browser. `panel`: the saves panel's, where that crossed cloud adds the game to the account. */
export function SlotTitle({ slot, panel }: { slot: SaveSlot; panel?: boolean }) {
  return <strong>{slot.name}{slot.mode === "sandbox" && <span class="save-sandbox small" data-tip="Infinite money">Sandbox</span>}<CloudMark slot={slot} panel={panel} /></strong>;
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
  if (slot.key.startsWith("account:")) return <span class="save-cloud" data-tip={panel ? "Only saved online. Opening will download locally." : "On your account"}><Icon.cloud /></span>;
  const noAccount = !slot.cloud?.acct;
  if (tiedTo(slot.cloud, ctx)) return null;
  if (panel && noAccount) return <button class="save-cloud save-add" data-tip="Only in this browser. Click to add it to your account." aria-label={`Add ${slot.name} to your account`}
    onClick={(e) => { e.stopPropagation(); void addToAccount([slot]); }}><Icon.cloudOff /></button>;
  return <span class="save-cloud" data-tip="Only in this browser"><Icon.cloudOff /></span>;
}

/** T-988: after the playing game's "Autosaved", whether its last save reached the account. */
function AccountMark({ slot, now }: { slot: SaveSlot; now: number }) {
  const ctx = useGamesOn(), c = slot.cloud;
  if (!ctx || !tiedTo(c, ctx) || c.gaveWay) return null;
  if (c.error || !c.okAt) return <span class="save-cloud save-account warn" data-tip={c.error ? `Not on your account yet. ${c.error.replace(/([^.])$/, "$1.")}` : "Not on your account yet"}><Icon.cloudOff /></span>;
  return <span class="save-cloud save-account" data-tip={`On your account, saved ${atTime(c.okAt, now)}`}><Icon.cloud /></span>;
}

/** T-988: another device opened or saved this game in the last ten minutes. */
function Elsewhere({ slot }: { slot: SaveSlot }) {
  const ctx = useGamesOn();
  void accountList.value;
  const at = ctx ? elsewhereAt(slot) : 0;
  return at ? <div class="small muted save-elsewhere" data-tip={`Opened or saved on another device ${fullDate(at)}`}>Open on another device</div> : null;
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
  return <input ref={field} type="text" class="save-name-field" aria-label="Save slot name" maxLength={80}
    onKeyDown={(e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    }}
    onBlur={() => finish(true)} />;
}

const OPEN_TIP = "Play this game. The game you are playing is saved first.";

/** One saved game in this browser. */
function SlotCard({ slot, playing, now, writable, renaming, copying, onRename, onRenamed, onCopy, onDelete }: { slot: SaveSlot; playing: boolean; now: number; writable: boolean; renaming: boolean; copying: boolean; onRename: () => void; onRenamed: () => void; onCopy: () => void; onDelete: () => void }) {
  const [msg, setMsg] = useState("");
  const byHand = savedByHand.value, newer = saveSlotsState.value.newer;
  const autosaved = playing && writable && !(byHand && Math.abs(byHand - slot.at) < 2000);
  const saveTip = writable ? "Save this game now (Ctrl+S)" : newer ? "A newer version saved this game. Reload the page to update." : /another device/.test(saveSlotsState.value.reason ?? "") ? "Save this game now (Ctrl+S). Autosave stops on the other device." : "Save this game now (Ctrl+S). Autosave stops in the other tab that has it open.";
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title">
          {renaming
            ? <NameField slot={slot} onDone={onRenamed} />
            : <span class="save-name" onDblClick={onRename}><SlotTitle slot={slot} panel />{playing && <button class="save-pen" data-tip="Rename" aria-label="Rename" onClick={onRename}><Icon.pencil /></button>}</span>}
          {playing && <span class="save-playing small">Playing</span>}
        </div>
        <SlotFacts slot={slot} now={now} autosaved={autosaved} mark={playing ? <AccountMark slot={slot} now={now} /> : null} />
        {!playing && <Elsewhere slot={slot} />}
        {msg && <div class="small bad">{msg}</div>}
        <div class="save-actions">
          {playing ? <>
            <button class="btn icon save-copy" data-tip="Make a copy of this save" aria-label="Make a copy" disabled={copying} onClick={onCopy}><Icon.copy /></button>
            <button class="btn icon save-download" data-tip={newer ? "Download as a file, as the newer version saved it" : "Download as a file"} aria-label="Download as a file" onClick={exportGame}><Icon.download /></button>
            <button class="btn icon save-delete" data-tip="Delete" aria-label="Delete" onClick={onDelete}><Icon.bin /></button>
            <button class="btn text save-now" data-tip={saveTip} onClick={() => manageSaveSlot("save")}>Save</button>
          </> : <>
            <button class="btn icon save-download" data-tip="Download as a file" aria-label="Download as a file"
              onClick={async () => { setMsg(""); if (!await exportSlot(slot)) setMsg("This game could not be read."); }}><Icon.download /></button>
            <button class="btn icon save-delete" data-tip="Delete" aria-label="Delete" onClick={onDelete}><Icon.bin /></button>
            <button class="btn text save-open" data-tip={OPEN_TIP} onClick={() => { manageSaveSlot("open", slot.key); savesOpen.value = false; }}>Open</button>
          </>}
        </div>
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
    case "account": return { day: v.game.facts?.day, stations: v.game.facts?.stations, cloud: "On your account. Open downloads it here." };
    case "backup": return { day: v.backup.day, stations: v.backup.stations, tip: v.backup.pull ? "This browser's copy before your account's replaced it" : "A backup in this browser" };
    case "accountBackup": return { tip: "A backup on your account", cloud: "A backup on your account" };
  }
}
const versionId = (v: Version) => v.kind === "slot" ? v.slot.key : v.kind === "account" ? "account:" + v.game.id : v.kind === "backup" ? v.backup.key : `account-backup:${v.game.id}:${v.backup.rev}`;

/** "1 other version", "11 other versions". */
export const otherVersions = (n: number) => n === 1 ? "1 other version" : `${n} other versions`;

/** What opening and deleting a version does where the card is (the saves panel, the start menu);
 * each answers a reason to show, or "" when done. */
export interface VersionActions { open: (v: Version) => Promise<string>; remove: (v: Version) => Promise<string> }

/** The line at the bottom of a game's card, "11 other versions", and under it, opened, each other
 * version: when it was saved, its day and stations, Open, and for a slot or account game a delete. */
export function VersionsLine({ group, now, act }: { group: GameGroup; now: number; act: VersionActions }) {
  const [asking, setAsking] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const n = group.others.length;
  if (!n) return null;
  const open = versionsShown.value.has(group.origin);
  const toggle = () => {
    const s = new Set(versionsShown.peek());
    if (!s.delete(group.origin)) s.add(group.origin);
    versionsShown.value = s;
  };
  const run = async (p: Promise<string>) => { setMsg(""); const why = await p; if (why) setMsg(why); };
  return <div class="save-versions">
    <button class={"save-versions-line small" + (open ? " open" : "")} aria-expanded={open} onClick={toggle}><Icon.chevron />{otherVersions(n)}</button>
    {open && <ul class="save-version-list">
      {group.others.map((v) => {
        const id = versionId(v), f = versionFacts(v), deletable = v.kind === "slot" || v.kind === "account";
        const where = v.kind === "account" ? "your account" : v.kind === "slot" && onAccount(v.slot) ? "this browser and your account" : "this browser";
        return <li key={id} data-version={id} data-kind={v.kind}>
          <div class="save-version">
            <span class="small muted num" data-tip={f.tip}>
              {savedWhen(v.at, now)}{f.day !== undefined && <> · Day {f.day}</>}{f.stations !== undefined && <> · {plural(f.stations, "station")}</>}
              {f.cloud && <span class="save-cloud" data-tip={f.cloud}><Icon.cloud /></span>}
            </span>
            {deletable && <button class="btn icon save-delete" data-tip="Delete" aria-label="Delete this version" onClick={() => setAsking(id)}><Icon.bin /></button>}
            <button class="btn text save-version-open" onClick={() => void run(act.open(v))}>Open</button>
          </div>
          {asking === id && <div class="callout bad">
            <p>Delete this version from {where}?</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { setAsking(null); void run(act.remove(v)); }}>Delete version</button><button class="btn text" onClick={() => setAsking(null)}>Keep it</button></div>
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
  const where = browser && account ? "this browser and your account" : account ? "your account" : "this browser";
  const what = group.others.length ? `"${name}" and its ${otherVersions(group.others.length)}` : `"${name}"`;
  return playing ? `Delete ${what} from ${where}? The game ends and you go back to the start menu.` : `Delete ${what} from ${where}?`;
}

/** The head's icon: signed out a crossed cloud that opens Sign in, signed in a cloud that opens the
 * account; where there are no accounts, a crossed cloud only while the browser may drop the saves. */
function HeadMark() {
  const s = session.value;
  if (accountState.value === "on") {
    const a = s?.account;
    const tip = !a ? "Kept in this browser only. Sign in to keep your games on your account." : a.username ? `Signed in as ${a.username}` : a.logins.includes("google") ? "Signed in with Google" : "Signed in";
    return <button class="btn icon save-head-mark" data-tip={tip} aria-label={a ? "Account" : "Sign in"} onClick={() => (accountOpen.value = true)}>{a ? <Icon.cloud /> : <Icon.cloudOff />}</button>;
  }
  if (storageKept.value === false) return <span class="save-head-mark" data-tip="These saves can be lost if the disk fills up."><Icon.cloudOff /></span>;
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
const copyName = (name: string | undefined) => (name ? `${name} copy`.slice(0, 80) : "");

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
  const t = Math.max(now, at);
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
        if (!key) return "This game could not be downloaded.";
        manageSaveSlot("open", key);
        return shut();
      }
      const name = g.top.kind === "slot" ? g.top.slot.name : g.top.game.name;
      return await restoreAccountBackup(v.game.id, name, g.origin, v.backup) ? shut() : "This backup could not be downloaded.";
    },
    remove: async (v) => {
      if (v.kind === "slot") manageSaveSlot("delete", v.slot.key);
      else if (v.kind === "account" && !await deleteAccountGame(v.game.id)) return "This game could not be deleted.";
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
        <button class="btn text save-new" onClick={onNewGame}>New game</button>
        <button class="btn text" data-tip="Open a .save file as a new saved game" onClick={pick}>Open a file</button>
        {/* a test page's game has no card to copy it from */}
        {!saves.slots.some((s) => s.key === saves.active) && <button class="btn icon save-copy" data-tip="Make a copy" aria-label="Make a copy" disabled={copyFrom !== null} onClick={copy}><Icon.copy /></button>}
      </div>
      <ul class="save-slots">
        {groups.map((g) => {
          const top = g.top, slot = top.kind === "slot" ? top.slot : cardOf(top.game), playing = slot.key === saves.active;
          return <li key={g.origin} data-origin={g.origin} {...(top.kind === "slot" ? { "data-slot": slot.key } : { "data-account": top.game.id })} class={playing ? "on" : ""}>
            {top.kind === "slot"
              ? <SlotCard slot={slot} playing={playing} now={t} writable={saves.writable} renaming={rename === slot.key} copying={copyFrom !== null}
                onRename={() => setRename(slot.key)} onRenamed={() => setRename(null)} onCopy={copy} onDelete={() => setRemoving(g.origin)} />
              : <AccountCard slot={slot} now={t} onDelete={() => setRemoving(g.origin)} />}
            <VersionsLine group={g} now={t} act={actions(g)} />
            {removing === g.origin && <div class="callout bad">
              <p>{deleteGroupQuestion(g, playing)}</p>
              <div class="pane-foot"><button class="btn text" onClick={() => { removeGroup(g); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
            </div>}
          </li>;
        })}
      </ul>
    </>
  );
}

/** T-988: a game on the account and not in this browser; Open downloads it into a new slot. */
function AccountCard({ slot, now, onDelete }: { slot: SaveSlot; now: number; onDelete: () => void }) {
  const [msg, setMsg] = useState("");
  const g = accountList.value?.games.find((x) => "account:" + x.id === slot.key);
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title"><span class="save-name"><SlotTitle slot={slot} panel /></span></div>
        <SlotFacts slot={slot} now={now} />
        {msg && <div class="small bad">{msg}</div>}
        <div class="save-actions">
          <button class="btn icon save-delete" data-tip="Delete" aria-label="Delete" onClick={onDelete}><Icon.bin /></button>
          <button class="btn text save-open" data-tip={OPEN_TIP} onClick={async () => {
            if (!g) return;
            setMsg("");
            const key = await downloadAccountGame(g);
            if (!key) { setMsg("This game could not be downloaded."); return; }
            manageSaveSlot("open", key);
            savesOpen.value = false;
          }}>Open</button>
        </div>
      </div>
    </div>
  );
}

/** T-988: a slot whose game is also on the account signed in. */
function onAccount(slot: SaveSlot): boolean {
  const ctx = cloudContext();
  return !!ctx?.on && tiedTo(slot.cloud, ctx);
}

/** The confirmation before a slot is deleted; also the start menu's (T-1612). */
export function deleteQuestion(slot: SaveSlot, playing = false): string {
  const where = onAccount(slot) ? "this browser and your account" : "this browser";
  return playing ? `Delete "${slot.name}" from ${where}? The game ends and you go back to the start menu.` : `Delete "${slot.name}" from ${where}?`;
}

/** The modal. */
export function SavesPanel() {
  const open = savesOpen.value;
  // T-728: New game shows the new game sheet in the panel's place
  const [sheet, setSheet] = useState(false);
  useEffect(() => { if (!open) setSheet(false); }, [open]);
  useEffect(() => { if (open) { manageSaveSlot("refresh"); void refreshAccountGames(); } }, [open]);
  return (
    <Modal open={open} onClose={() => (savesOpen.value = false)} title={sheet ? "New game" : "Saved games"} head={sheet ? null : <HeadMark />}>
      {sheet
        ? <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => { newGame(game); viewStart(game.start); savesOpen.value = false; }} />
        : <Saves onNewGame={() => setSheet(true)} />}
    </Modal>
  );
}
