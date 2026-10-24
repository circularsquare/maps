// The saved games panel (T-251; the cards from T-241): a centred modal over the page. One card per
// browser slot: a picture of its network, name, when it was saved, day, cities, lines, stations and
// cash; the game being played is marked, every other has an Open button; under the game being
// played, its backups (T-744). The files, a new game and save a copy are above the list (T-1131); a
// copy, a loaded file and a new game each get a new slot (T-743).

import { signal } from "@preact/signals";
import { useEffect, useState } from "preact/hooks";
import { cityById, isCity } from "../game/cities";
import { money, plural } from "../game/format";
import { SHADES } from "../game/shades";
import { lastSaved } from "../game/money";
import { allowCopyFolder, canCopyToFolder, copyFolder, pickCopyFolder, stopCopies, storageKept } from "../game/saveSafety";
import { exportGame, importGame, newGame, manageSaveSlot, saveSlotsState } from "../workers/clockClient";
import type { SaveSlot, SlotBackup } from "../workers/saveFile";
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

/** A saved game's name, with a mark when it is a sandbox (T-728); also the start menu's. */
export function SlotTitle({ slot }: { slot: SaveSlot }) {
  return <strong>{slot.name}{slot.mode === "sandbox" && <span class="save-sandbox small" data-tip="A sandbox game: nothing is refused for money">Sandbox</span>}</strong>;
}

/** One saved game. */
function SlotCard({ slot, playing, now, writable, onRename, onDelete }: { slot: SaveSlot; playing: boolean; now: number; writable: boolean; onRename: () => void; onDelete: () => void }) {
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title"><SlotTitle slot={slot} />{playing && <span class="save-playing small">Playing</span>}</div>
        <SlotFacts slot={slot} now={now} autosaved={playing && writable} />
        <div class="save-actions">
          {!playing && <button class="btn text save-open" data-tip="Play this game. The game you are playing is saved first." onClick={() => { manageSaveSlot("open", slot.key); savesOpen.value = false; }}>Open</button>}
          <button class="btn text" onClick={onRename}>Rename</button>
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
  const [newName, setNewName] = useState("");
  const [rename, setRename] = useState<{ key: string; name: string } | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  // "2 min ago" moves on while the panel is open
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 20_000);
    return () => clearInterval(t);
  }, []);
  const saves = saveSlotsState.value;
  const at = lastSaved.value ?? 0;
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
      </div>
      <SaveSafety />
      <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("create", undefined, newName); setNewName(""); }}>
        <input type="text" aria-label="New save slot name" placeholder="Name for a copy of this game" maxLength={80} value={newName} onInput={(e) => setNewName((e.target as HTMLInputElement).value)} />
        <button class="btn text" type="submit" data-tip="Save a copy of this game under a new name and carry on playing the copy">Save as a copy</button>
      </form>
      <ul class="save-slots">
        {saves.slots.map((slot) => <li data-slot={slot.key} class={slot.key === saves.active ? "on" : ""}>
          {rename?.key === slot.key ? <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("rename", slot.key, rename.name); setRename(null); }}>
            <input type="text" aria-label="Save slot name" maxLength={80} value={rename.name} onInput={(e) => setRename({ key: slot.key, name: (e.target as HTMLInputElement).value })} />
            <button class="btn text" type="submit">Rename</button>
            <button class="btn text" type="button" onClick={() => setRename(null)}>Cancel</button>
          </form> : <SlotCard slot={slot} playing={slot.key === saves.active} now={Math.max(now, at)} writable={saves.writable}
            onRename={() => setRename({ key: slot.key, name: slot.name })} onDelete={() => setRemoving(slot.key)} />}
          {slot.key === saves.active && saves.backups.length > 0 && <Backups backups={saves.backups} now={Math.max(now, at)} />}
          {removing === slot.key && <div class="callout bad">
            <p>Delete "{slot.name}" from this browser?</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { manageSaveSlot("delete", slot.key); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
          </div>}
        </li>)}
      </ul>
    </>
  );
}

/** T-1214: whether the browser keeps these saves if the disk fills, and the folder they are copied to. */
function SaveSafety() {
  const kept = storageKept.value, c = copyFolder.value;
  const time = c?.at ? new Date(c.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  return <div class="save-safety">
    {kept === false && <p class="muted small" data-tip="The browser has not agreed to keep this site's storage, so a full disk can empty it. Copies in a folder are kept.">These saves can be lost if the disk fills up.</p>}
    {canCopyToFolder && (!c
      ? <button class="btn text" data-tip="Pick a folder outside the browser. Each saved game is copied there as a .save file every time it is saved." onClick={() => void pickCopyFolder()}>Keep copies in a folder</button>
      : c.state === "allow"
        ? <button class="btn text" data-tip="The browser asks again for the folder on each visit" onClick={() => void allowCopyFolder()}>Allow copies to {c.name}</button>
        : <p class="small">
          <span class={c.state === "failed" ? "bad" : "muted"} data-tip={c.state === "failed" ? c.error : "Each saved game is copied there as a .save file every time it is saved"}>
            {c.state === "failed" ? `Copies to ${c.name} failed` : `Copied to ${c.name}${time ? ` at ${time}` : ""}`}
          </span>{" "}
          <button class="btn text" data-tip="Stop copying to this folder. The copies there stay." onClick={() => void stopCopies()}>Stop</button>
        </p>)}
  </div>;
}

/** The modal. */
export function SavesPanel() {
  const open = savesOpen.value;
  // T-728: New game shows the new game sheet in the panel's place
  const [sheet, setSheet] = useState(false);
  useEffect(() => { if (!open) setSheet(false); }, [open]);
  useEffect(() => { if (open) manageSaveSlot("refresh"); }, [open]);
  return (
    <Modal open={open} onClose={() => (savesOpen.value = false)} title={sheet ? "New game" : "Saved games"}>
      {sheet
        ? <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => { newGame(game); viewStart(game.start); savesOpen.value = false; }} />
        : <Saves onNewGame={() => setSheet(true)} />}
    </Modal>
  );
}
