// The saved games panel (T-251; the cards from T-241): a centred modal over the page, opened by the
// save button at the end of the dock's tab row (or the top bar's Autosave off notice), closed with
// Esc, its close button or a click outside it. One card per browser slot: a picture of its network,
// name, when it was saved, day, cities, lines, stations and cash; the game being played is marked,
// every other has an Open button. Then save a copy, the files and a new game. T-253: no notes.

import { signal } from "@preact/signals";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { CITIES, cityById } from "../game/cities";
import { money } from "../game/format";
import { lastSaved } from "../game/money";
import { exportGame, importGame, newGame, manageSaveSlot, saveSlotsState } from "../workers/clockClient";
import type { SaveSlot } from "../workers/saveFile";
import { Icon } from "./widgets";
import { Money } from "./MoneyText";

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
  const names = ids.filter((id) => CITIES.some((c) => c.id === id)).map((id) => cityById(id).name);
  if (names.length <= 2) return names.join(" and ");
  if (names.length === 3) return `${names[0]}, ${names[1]} and ${names[2]}`;
  return `${names[0]}, ${names[1]} and ${names.length - 2} more`;
}

const plural = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

/** A saved game's picture of its network (T-241). */
export function SlotThumb({ slot }: { slot: SaveSlot }) {
  return slot.thumb
    ? <img class="save-thumb" src={slot.thumb} alt="" data-tip={`The network in "${slot.name}" when it was last saved`} />
    : <div class="save-thumb empty muted small">{slot.lines !== undefined ? "Nothing built" : "No picture yet"}</div>;
}

/** When a saved game was saved, its day and cities, its lines, stations and cash; also the start
 * menu's (T-677). */
export function SlotFacts({ slot, now, autosaved }: { slot: SaveSlot; now: number; autosaved?: boolean }) {
  const when = savedWhen(slot.at, now);
  const exact = new Date(slot.at).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  return <>
    <div class="small muted" data-tip={`Last saved ${exact}`}>{autosaved ? when.replace("Saved", "Autosaved") : when}</div>
    <div class="small muted num">
      <span data-tip={`Game day ${slot.day}`}>Day {slot.day}</span>
      {slot.cities?.length ? <> · {cityList(slot.cities)}</> : null}
    </div>
    {slot.lines !== undefined && <div class="small muted num">
      <span data-tip={`${slot.lines} lines`}>{plural(slot.lines!, "line", "lines")}</span>
      {" · "}
      <span data-tip={`${slot.stations} stations`}>{plural(slot.stations!, "station", "stations")}</span>
      {" · "}
      <span data-tip={`Cash when saved: ${money(slot.cash! * 1e6)}`}><Money dollars={slot.cash! * 1e6} /></span>
    </div>}
  </>;
}

/** One saved game. */
function SlotCard({ slot, playing, now, writable, onRename, onDelete }: { slot: SaveSlot; playing: boolean; now: number; writable: boolean; onRename: () => void; onDelete: () => void }) {
  return (
    <div class="save-card">
      <SlotThumb slot={slot} />
      <div class="save-info">
        <div class="save-slot-title"><strong>{slot.name}</strong>{playing && <span class="save-playing small">Playing</span>}</div>
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

function Saves() {
  const [confirm, setConfirm] = useState(false);
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
      <ul class="save-slots">
        {saves.slots.map((slot) => <li data-slot={slot.key} class={slot.key === saves.active ? "on" : ""}>
          {rename?.key === slot.key ? <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("rename", slot.key, rename.name); setRename(null); }}>
            <input type="text" aria-label="Save slot name" maxLength={80} value={rename.name} onInput={(e) => setRename({ key: slot.key, name: (e.target as HTMLInputElement).value })} />
            <button class="btn text" type="submit">Rename</button>
            <button class="btn text" type="button" onClick={() => setRename(null)}>Cancel</button>
          </form> : <SlotCard slot={slot} playing={slot.key === saves.active} now={Math.max(now, at)} writable={saves.writable}
            onRename={() => setRename({ key: slot.key, name: slot.name })} onDelete={() => setRemoving(slot.key)} />}
          {removing === slot.key && <div class="callout bad">
            <p>Delete "{slot.name}" from this browser?</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { manageSaveSlot("delete", slot.key); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
          </div>}
        </li>)}
      </ul>
      <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("create", undefined, newName); setNewName(""); }}>
        <input type="text" aria-label="New save slot name" placeholder="Name for a copy of this game" maxLength={80} value={newName} onInput={(e) => setNewName((e.target as HTMLInputElement).value)} />
        <button class="btn text" type="submit" data-tip="Save a copy of this game under a new name and carry on playing the copy">Save as a copy</button>
      </form>
      <div class="pane-foot save-files">
        <button class="btn text" data-tip="Download this game as a .save file" onClick={exportGame}>Save to a file</button>
        <button class="btn text" onClick={pick}>Load a file</button>
        {!confirm && <button class="btn text" onClick={() => setConfirm(true)}>New game</button>}
      </div>
      {confirm && (
        <div class="callout bad">
          <p>Start again in an empty city? The game in this slot is replaced.</p>
          <div class="pane-foot">
            <button class="btn text" onClick={() => { setConfirm(false); newGame(); savesOpen.value = false; }}>Start a new game</button>
            <button class="btn text" onClick={() => setConfirm(false)}>Keep playing</button>
          </div>
        </div>
      )}
    </>
  );
}

/** The modal. A native dialog: focus is trapped and restored, Esc closes it; a click on the
 * backdrop (the dialog element itself, outside its box) closes it too. */
export function SavesPanel() {
  const open = savesOpen.value;
  const dialog = useRef<HTMLDialogElement>(null);
  useLayoutEffect(() => {
    const el = dialog.current;
    if (!open || !el) return;
    const before = document.activeElement;
    if (!el.open) el.showModal();
    manageSaveSlot("refresh");
    return () => {
      if (el.open) el.close();
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [open]);
  if (!open) return null;
  return (
    <dialog class="saves-panel" ref={dialog} aria-labelledby="saves-title"
      onCancel={(e) => { e.preventDefault(); savesOpen.value = false; }}
      onClick={(e) => { if (e.target === e.currentTarget) savesOpen.value = false; }}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Escape") { e.preventDefault(); savesOpen.value = false; } }}>
      <div class="saves-inner">
        <div class="saves-head">
          <h2 id="saves-title">Saved games</h2>
          <button class="btn icon" aria-label="Close" data-tip="Close (Esc)" onClick={() => (savesOpen.value = false)}><Icon.cross /></button>
        </div>
        <Saves />
      </div>
    </dialog>
  );
}
