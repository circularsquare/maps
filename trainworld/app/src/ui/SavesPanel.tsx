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
import { lastSaved } from "../game/money";
import { storageKept } from "../game/saveSafety";
import { exportGame, exportSlot, importGame, newGame, manageSaveSlot, savedByHand, saveSlotsState } from "../workers/clockClient";
import type { SaveSlot, SlotBackup } from "../workers/saveFile";
import { accountConfig, accountOpen, accountState, cloudContext, session } from "../game/account";
import { accountBackups, accountList, accountOnly, addToAccount, cardOf, deleteAccountGame, downloadAccountGame, elsewhereAt, refreshAccountGames, restoreAccountBackup } from "../game/accountGames";
import { tiedTo, type AccountBackup, type CloudContext } from "../workers/cloud";
import { Icon } from "./widgets";
import { viewStart } from "./locate";
import { Modal } from "./Modal";
import { Money } from "./MoneyText";
import { NewGameSheet } from "./NewGame";

/** The saves panel is open. */
export const savesOpen = signal(false);
/** The playing game's earlier versions are listed (kept for the visit). */
const versionsOpen = signal(false);

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
  if (!slot.thumb) return <div class="save-thumb empty muted small">{slot.lines !== undefined ? "Nothing built" : "No picture yet"}</div>;
  const panels = (slot.panels ?? []).filter((p) => isCity(p.city));
  const sh = SHADES[slot.shade ?? "light"];
  return <div class="save-thumb" style={{ "--thumb-ink": sh.mapInk, "--thumb-land": sh.land }}>
    <img src={slot.thumb} alt="" data-tip={`The network in "${slot.name}" when it was last saved`} />
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
  return <>
    <div class="small muted save-when"><span data-tip={`Last saved ${fullDate(slot.at)}`}>{autosaved ? when.replace("Saved", "Autosaved") : when}</span>{mark}</div>
    <div class="small muted num">
      <span data-tip={`Game day ${slot.day}`}>Day {slot.day}</span>
      {slot.cities?.length ? <> · {cityList(slot.cities)}</> : null}
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
  if (slot.key.startsWith("account:")) return <span class="save-cloud" data-tip={panel ? "On your account. Open downloads it here." : "On your account"}><Icon.cloud /></span>;
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
  const byHand = savedByHand.value;
  const autosaved = playing && writable && !(byHand && Math.abs(byHand - slot.at) < 2000);
  const saveTip = writable ? "Save this game now (Ctrl+S)" : /another device/.test(saveSlotsState.value.reason ?? "") ? "Save this game now (Ctrl+S). Autosave stops on the other device." : "Save this game now (Ctrl+S). Autosave stops in the other tab that has it open.";
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
            <button class="btn icon save-copy" data-tip="Make a copy" aria-label="Make a copy" disabled={copying} onClick={onCopy}><Icon.copy /></button>
            <button class="btn icon save-download" data-tip="Download as a file" aria-label="Download as a file" onClick={exportGame}><Icon.download /></button>
            <button class={"btn icon save-versions" + (versionsOpen.value ? " on" : "")} data-tip="Earlier versions" aria-label="Earlier versions" aria-pressed={versionsOpen.value}
              onClick={() => (versionsOpen.value = !versionsOpen.peek())}><Icon.history /></button>
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

const RESTORE_TIP = "Open this backup as a new saved game. The game you are playing is saved first.";

/** "Today at 2:58 PM", "Oct 24 at 2:58 PM". */
const versionWhen = (at: number, now: number) => sameDay(at, now) ? `Today at ${clockTime(at)}` : atTime(at, now);

/** The playing game's earlier versions, newest first, each opened as a new slot: the account's when
 * the account keeps the game (T-988), else this browser's (T-744). */
function Versions({ slot, backups, account, now }: { slot: SaveSlot; backups: SlotBackup[]; account: AccountBackup[] | null; now: number }) {
  const [msg, setMsg] = useState("");
  const rows = account
    ? account.map((b) => <div class="save-backup small" data-account-backup={b.rev}>
      <span class="muted num" data-tip={`A backup of this game on your account from ${fullDate(b.savedAt * 1000)}`}>{versionWhen(b.savedAt * 1000, now)}</span>
      <button class="btn text" data-tip={RESTORE_TIP} onClick={async () => {
        setMsg("");
        if (await restoreAccountBackup(slot, b)) savesOpen.value = false;
        else setMsg("This backup could not be downloaded.");
      }}>Open as a copy</button>
    </div>)
    : backups.map((b) => <div class="save-backup small" data-backup={b.key}>
      <span class="muted num" data-tip={`A backup of this game from ${fullDate(b.at)}, game day ${b.day}`}>{versionWhen(b.at, now)} · Day {b.day}</span>
      <button class="btn text" data-tip={RESTORE_TIP} onClick={() => { manageSaveSlot("restore", b.key); savesOpen.value = false; }}>Open as a copy</button>
    </div>);
  return <div class="save-backups">
    {rows.length ? rows : <div class="small muted">No earlier versions yet</div>}
    {msg && <div class="small bad">{msg}</div>}
  </div>;
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
  const pick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".save";
    input.onchange = () => { if (input.files?.[0]) { void importGame(input.files[0]); savesOpen.value = false; } };
    input.click();
  };
  const copy = () => {
    setCopyFrom(saves.active);
    manageSaveSlot("create", undefined, copyName(saves.slots.find((s) => s.key === saves.active)?.name));
  };
  const t = Math.max(now, at);
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
        {saves.slots.map((slot) => {
          const playing = slot.key === saves.active;
          return <li data-slot={slot.key} class={playing ? "on" : ""}>
            <SlotCard slot={slot} playing={playing} now={t} writable={saves.writable} renaming={rename === slot.key} copying={copyFrom !== null}
              onRename={() => setRename(slot.key)} onRenamed={() => setRename(null)} onCopy={copy} onDelete={() => setRemoving(slot.key)} />
            {playing && versionsOpen.value && <Versions slot={slot} backups={saves.backups} account={accountBackups(slot)} now={t} />}
            {removing === slot.key && <div class="callout bad">
              <p>{onAccount(slot) ? `Delete "${slot.name}" from this browser and your account?` : `Delete "${slot.name}" from this browser?`}</p>
              <div class="pane-foot"><button class="btn text" onClick={() => { manageSaveSlot("delete", slot.key); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
            </div>}
          </li>;
        })}
        {ctx && accountOnly(saves.slots).map((g) => {
          const card = cardOf(g);
          return <li data-account={g.id}>
            <AccountCard slot={card} now={t} onDelete={() => setRemoving(card.key)} />
            {removing === card.key && <div class="callout bad">
              <p>Delete "{g.name}" from your account?</p>
              <div class="pane-foot"><button class="btn text" onClick={() => { void deleteAccountGame(g.id); setRemoving(null); }}>Delete from your account</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
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
