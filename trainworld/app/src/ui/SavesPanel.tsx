// The saved games panel (T-251; the cards from T-241): a centred modal over the page. One card per
// browser slot: a picture of its network, name, when it was saved, day, cities, lines, stations and
// cash; the game being played is marked, every other has an Open button; under the game being
// played, its backups (T-744). The files, a new game and save a copy are above the list (T-1131); a
// copy, a loaded file and a new game each get a new slot (T-743).

import { signal } from "@preact/signals";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { cityById, isCity } from "../game/cities";
import { money, plural } from "../game/format";
import { SHADES } from "../game/shades";
import { lastSaved } from "../game/money";
import { allowCopyFolder, canCopyToFolder, copyFolder, pickCopyFolder, stopCopies, storageKept, SUBFOLDER } from "../game/saveSafety";
import { exportGame, importGame, newGame, manageSaveSlot, savedByHand, saveSlotsState } from "../workers/clockClient";
import type { SaveSlot, SlotBackup } from "../workers/saveFile";
import { accountConfig, accountState, cloudContext, session } from "../game/account";
import { accountList, accountOnly, cardOf, deleteAccountGame, downloadAccountGame, elsewhereAt, refreshAccountGames } from "../game/accountGames";
import { tiedTo, type CloudContext } from "../workers/cloud";
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

/** When a saved game was saved, its day and cities, its lines, stations and cash; also the start
 * menu's (T-677). */
export function SlotFacts({ slot, now, autosaved }: { slot: SaveSlot; now: number; autosaved?: boolean }) {
  const when = savedWhen(slot.at, now);
  return <>
    <div class="small muted" data-tip={`Last saved ${fullDate(slot.at)}`}>{autosaved ? when.replace("Saved", "Autosaved") : when}</div>
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
 * in this browser. */
export function SlotTitle({ slot }: { slot: SaveSlot }) {
  return <strong>{slot.name}{slot.mode === "sandbox" && <span class="save-sandbox small" data-tip="Infinite money">Sandbox</span>}<CloudMark slot={slot} /></strong>;
}

/** T-988: the account's games are in use here (signed in, a server that keeps them); read as signals. */
function useGamesOn(): CloudContext | null {
  void session.value; void accountState.value; void accountConfig.value;
  const ctx = cloudContext();
  return ctx?.on ? ctx : null;
}

function CloudMark({ slot }: { slot: SaveSlot }) {
  const ctx = useGamesOn();
  if (!ctx) return null;
  if (slot.key.startsWith("account:")) return <span class="save-cloud" data-tip="On your account"><Icon.cloud /></span>;
  if (tiedTo(slot.cloud, ctx)) return null;
  return <span class="save-cloud" data-tip="Only in this browser"><Icon.cloudOff /></span>;
}

/** "9:41 PM" today, "Oct 24" before. */
function shortWhen(at: number, now: number): string {
  const d = new Date(at);
  return d.toDateString() === new Date(now).toDateString() ? clockTime(at) : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** T-988: under the playing game's facts, whether its last save reached the account. */
function CloudLine({ slot, now }: { slot: SaveSlot; now: number }) {
  const ctx = useGamesOn(), c = slot.cloud;
  if (!ctx || !tiedTo(c, ctx) || c.gaveWay) return null;
  if (c.error || !c.okAt) return <div class="small muted save-account" data-tip={c.error ?? "Not sent yet"}>Not on your account yet</div>;
  return <div class="small muted save-account" data-tip={`Saved to your account ${fullDate(c.okAt)}`}>Saved to your account {shortWhen(c.okAt, now)}</div>;
}

/** T-988: another device opened or saved this game in the last ten minutes. */
function Elsewhere({ slot }: { slot: SaveSlot }) {
  const ctx = useGamesOn();
  void accountList.value;
  const at = ctx ? elsewhereAt(slot) : 0;
  return at ? <div class="small muted save-account" data-tip={`Opened or saved on another device ${fullDate(at)}`}>Open on another device</div> : null;
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

const clockTime = (at: number) => new Date(at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

/** One saved game. */
function SlotCard({ slot, playing, now, writable, renaming, onRename, onRenamed, onDelete }: { slot: SaveSlot; playing: boolean; now: number; writable: boolean; renaming: boolean; onRename: () => void; onRenamed: () => void; onDelete: () => void }) {
  const byHand = savedByHand.value;
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title">
          {renaming ? <NameField slot={slot} onDone={onRenamed} /> : <span class="save-name" onDblClick={onRename}><SlotTitle slot={slot} /></span>}
          {playing && <span class="save-playing small">Playing</span>}
        </div>
        <SlotFacts slot={slot} now={now} autosaved={playing && writable} />
        {playing ? <CloudLine slot={slot} now={now} /> : <Elsewhere slot={slot} />}
        <div class="save-actions">
          {playing && byHand && <span class="small muted save-by-hand num" data-tip={`Saved by hand ${fullDate(byHand)}`}>Saved {clockTime(byHand)}</span>}
          {playing && <button class="btn text save-now" data-tip={writable ? "Save this game now (Ctrl+S)" : /another device/.test(saveSlotsState.value.reason ?? "") ? "Save this game now (Ctrl+S). Autosave stops on the other device." : "Save this game now (Ctrl+S). Autosave stops in the other tab that has it open."} onClick={() => manageSaveSlot("save")}>Save</button>}
          {!playing && <button class="btn text save-open" data-tip="Play this game. The game you are playing is saved first." onClick={() => { manageSaveSlot("open", slot.key); savesOpen.value = false; }}>Open</button>}
          <button class="btn text" data-tip="Double click the name to rename it too"onClick={onRename}>Rename</button>
          {!playing && <button class="btn text" onClick={onDelete}>Delete</button>}
        </div>
      </div>
    </div>
  );
}

/** The playing slot's backups (T-744), each restored into a new slot. */
function Backups({ backups, now }: { backups: SlotBackup[]; now: number }) {
  return <div class="save-backups">
    {backups.map((b) => {
      return <div class="save-backup small" data-backup={b.key}>
        <span class="muted num" data-tip={`A backup of this game from ${fullDate(b.at)}, game day ${b.day}`}>{savedWhen(b.at, now).replace("Saved", "Backup")} · Day {b.day}</span>
        <button class="btn text" data-tip="Open this backup as a new saved game. The game you are playing is saved first." onClick={() => { manageSaveSlot("restore", b.key); savesOpen.value = false; }}>Restore as a copy</button>
      </div>;
    })}
  </div>;
}

function Saves({ onNewGame }: { onNewGame: () => void }) {
  // T-1295: the name a copy is saved under, asked only once Save a copy is clicked (null: not asked)
  const [newName, setNewName] = useState<string | null>(null);
  const [rename, setRename] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  // "2 min ago" moves on while the panel is open
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 20_000);
    return () => clearInterval(t);
  }, []);
  const saves = saveSlotsState.value;
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
  return (
    <>
      {!saves.writable && saves.reason && <p class="callout bad">{saves.reason}</p>}
      {saves.error && <p class="callout bad">{saves.error}</p>}
      <div class="pane-foot save-files">
        <button class="btn text" data-tip="Download this game as a .save file" onClick={exportGame}>Save to a file</button>
        <button class="btn text" data-tip="Open a .save file as a new saved game. The game you are playing is saved first." onClick={pick}>Load a file</button>
        <button class="btn text" data-tip="Start a new game as a new saved game. The game you are playing is saved first." onClick={onNewGame}>New game</button>
        <button class={"btn text" + (newName !== null ? " on" : "")} data-tip="Save a copy of this game under a new name and carry on playing the copy"
          onClick={() => setNewName(newName !== null ? null : copyName(saves.slots.find((s) => s.key === saves.active)?.name))}>Save a copy</button>
      </div>
      {newName !== null && <CopyName name={newName} setName={setNewName} />}
      <SaveSafety />
      <ul class="save-slots">
        {saves.slots.map((slot) => <li data-slot={slot.key} class={slot.key === saves.active ? "on" : ""}>
          <SlotCard slot={slot} playing={slot.key === saves.active} now={Math.max(now, at)} writable={saves.writable} renaming={rename === slot.key}
            onRename={() => setRename(slot.key)} onRenamed={() => setRename(null)} onDelete={() => setRemoving(slot.key)} />
          {slot.key === saves.active && saves.backups.length > 0 && <Backups backups={saves.backups} now={Math.max(now, at)} />}
          {removing === slot.key && <div class="callout bad">
            <p>{onAccount(slot) ? `Delete "${slot.name}" from this browser and your account?` : `Delete "${slot.name}" from this browser?`}</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { manageSaveSlot("delete", slot.key); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
          </div>}
        </li>)}
        {ctx && accountOnly(saves.slots).map((g) => {
          const card = cardOf(g);
          return <li data-account={g.id}>
            <AccountCard slot={card} now={Math.max(now, at)} onDelete={() => setRemoving(card.key)} />
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

/** "New York copy" for the game being played; "" when it has no slot (a test page). */
const copyName = (name: string | undefined) => (name ? `${name} copy`.slice(0, 80) : "");

/** T-1295: the name for Save a copy, prefilled and selected; Save or Enter saves, Cancel or Escape
 * closes it. */
function CopyName({ name, setName }: { name: string; setName: (n: string | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { input.current?.focus(); input.current?.select(); }, []);
  return (
    <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("create", undefined, name.trim()); setName(null); }}>
      <input ref={input} type="text" aria-label="Name of the copy" placeholder="Name" maxLength={80} value={name}
        onInput={(e) => setName((e.target as HTMLInputElement).value)}
        onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setName(null); } }} />
      <button class="btn text" type="submit">Save</button>
      <button class="btn text" type="button" onClick={() => setName(null)}>Cancel</button>
    </form>
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
        <div class="save-slot-title"><span class="save-name"><SlotTitle slot={slot} /></span></div>
        <SlotFacts slot={slot} now={now} />
        {msg && <div class="small bad">{msg}</div>}
        <div class="save-actions">
          <button class="btn text save-open" data-tip="Play this game. The game you are playing is saved first." onClick={async () => {
            if (!g) return;
            setMsg("");
            const key = await downloadAccountGame(g);
            if (!key) { setMsg("This game could not be downloaded."); return; }
            manageSaveSlot("open", key);
            savesOpen.value = false;
          }}>Open</button>
          <button class="btn text" onClick={onDelete}>Delete</button>
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

/** T-1214: whether the browser keeps these saves if the disk fills, and the folder they are copied to. */
function SaveSafety() {
  const kept = storageKept.value, c = copyFolder.value;
  const time = c?.at ? new Date(c.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  return <div class="save-safety">
    {kept === false && <p class="muted small" data-tip="The browser has not agreed to keep this site's storage, so a full disk can empty it. Games saved on this computer are kept.">These saves can be lost if the disk fills up.</p>}
    {canCopyToFolder && (!c
      ? <button class="btn text" data-tip={`Pick a folder on this computer. Each game is written as a .save file to a ${SUBFOLDER} folder inside it every time it is saved.`} onClick={() => void pickCopyFolder()}>Save on this computer</button>
      : c.state === "allow"
        ? <button class="btn text" data-tip={`The browser asks again for the folder on each visit. Games are saved in ${c.path}.`} onClick={() => void allowCopyFolder()}>Save to {c.name} again</button>
        : <p class="small">
          <span class={c.state === "failed" ? "bad" : "muted"} data-tip={c.state === "failed" ? c.error : "Each game is written there as a .save file every time it is saved"}>
            {c.state === "failed" ? `Saving to ${c.path} failed` : `Saved to ${c.path}${time ? ` at ${time}` : ""}`}
          </span>{" "}
          <button class="btn text" data-tip="Stop saving to this folder. The files there stay." onClick={() => void stopCopies()}>Stop</button>
        </p>)}
  </div>;
}

/** The modal. */
export function SavesPanel() {
  const open = savesOpen.value;
  // T-728: New game shows the new game sheet in the panel's place
  const [sheet, setSheet] = useState(false);
  useEffect(() => { if (!open) setSheet(false); }, [open]);
  useEffect(() => { if (open) { manageSaveSlot("refresh"); void refreshAccountGames(); } }, [open]);
  return (
    <Modal open={open} onClose={() => (savesOpen.value = false)} title={sheet ? "New game" : "Saved games"}>
      {sheet
        ? <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => { newGame(game); viewStart(game.start); savesOpen.value = false; }} />
        : <Saves onNewGame={() => setSheet(true)} />}
    </Modal>
  );
}
