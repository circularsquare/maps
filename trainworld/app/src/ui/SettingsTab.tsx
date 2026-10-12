// Settings (T-037): display settings (SPEC 8) and the theme, remembered per browser: the commuter
// bubbles (T-097: off, where commuters live, where they work), track colour,
// trains, station names, capacity markers (T-031), map labels, the riders drawn on the network
// (T-084: line width, station size, train fill). Then the game's saves (T-029).

import { useState } from "preact/hooks";
import { levelLabel } from "../game/format";
import { lastSaved } from "../game/money";
import { exportGame, importGame, newGame, manageSaveSlot, saveSlotsState } from "../workers/clockClient";
import { LEVEL_COLOURS } from "../game/palette";
import { LEVELS } from "../game/types";
import { display, setDisplay, theme, THEMES, type Theme } from "../game/ui";
import { Check, Segmented } from "./widgets";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };

export function SettingsTab() {
  const d = display.value;
  return (
    <>
      <div class="h">Local trips on the map</div>
      {d.commuters && (
        <label class="slider">
          <span>Bubble size</span>
          <input type="range" min="0.5" max="2" step="0.05" value={d.bubbleSize} onInput={(e) => setDisplay("bubbleSize", Number((e.target as HTMLInputElement).value))} />
        </label>
      )}
      <div class="h">Longer trips</div>
      <div class="checks">
        <Check checked={d.intercity} onChange={(v) => setDisplay("intercity", v)}>
          Bubbles outside the selected city
        </Check>
      </div>
      <div class="h">Track display</div>
      <div class="track-modes"><Segmented<"line" | "traffic" | "height" | "speed">
        options={[
          ["line", "Line"],
          ["traffic", "Line + traffic thickness"],
          ["height", "Height"],
          ["speed", "Max speed"],
        ]}
        value={d.trackColour}
        onChange={(v) => setDisplay("trackColour", v)}
      /></div>
      {d.trackColour === "speed" && <ul class="speed-key">
        {[[40, "#d85b4b"], [80, "#d9a437"], [120, "#55a886"], [160, "#397bc1"]].map(([v, colour]) => <li><i style={{background: String(colour)}} />{v} km/h</li>)}
      </ul>}
      {d.trackColour === "height" && (
        <ul class="levels-key">
          {LEVELS.map((l) => (
            <li>
              <i style={{ background: LEVEL_COLOURS[l] }} />
              {levelLabel(l)}
            </li>
          ))}
        </ul>
      )}
      <div class="h">Show</div>
      <div class="checks">
        <Check checked={d.trains} onChange={(v) => setDisplay("trains", v)}>
          Trains
        </Check>
        <Check checked={d.stationNames} onChange={(v) => setDisplay("stationNames", v)}>
          Station names
        </Check>
        <Check checked={d.capacity} onChange={(v) => setDisplay("capacity", v)}>
          Capacity markers
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
      </div>
      <div class="h">Demand forecast on the network</div>
      <div class="checks">
        <Check checked={d.stationRiders} onChange={(v) => setDisplay("stationRiders", v)}>
          Station demand circles
        </Check>
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          Train load forecast
        </Check>
      </div>
      {d.stationRiders && <p class="note muted small">Circle area shows daily entries and exits, up to 150,000. Click the platform box to inspect a station.</p>}
      {d.trainLoad && <>
        <p class="note muted small">Steady-state loads by stretch, not live passenger counts.</p>
        <div class="h">Train load</div>
        <Segmented<"average" | "busiest">
          options={[["average", "Average"], ["busiest", "Busiest hour"]]}
          value={d.trainLoadBasis}
          onChange={(v) => setDisplay("trainLoadBasis", v)}
        />
      </>}
      <div class="h">Theme</div>
      <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      <GameSaves />
    </>
  );
}

/** Saves (T-029): the game saves itself in this browser; a file can be exported and loaded. */
function GameSaves() {
  const [confirm, setConfirm] = useState(false);
  const [newName, setNewName] = useState("");
  const [rename, setRename] = useState<{ key: string; name: string } | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const saves = saveSlotsState.value;
  const at = lastSaved.value;
  const activeName = saves.slots.find(s => s.key === saves.active)?.name ?? "Original game";
  const pick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".save";
    input.onchange = () => input.files?.[0] && void importGame(input.files[0]);
    input.click();
  };
  return (
    <>
      <div class="h">Saved games in this browser</div>
      <p class="note muted">Playing: <strong>{activeName}</strong>. {saves.writable ? "This slot saves as you play." : saves.reason}</p>
      {saves.error && <p class="callout bad">{saves.error}</p>}
      <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("create", undefined, newName); setNewName(""); }}>
        <input type="text" aria-label="New save slot name" placeholder="Name a new slot" maxLength={80} value={newName} onInput={(e) => setNewName((e.target as HTMLInputElement).value)} />
        <button class="btn text" type="submit">Save to new slot</button>
      </form>
      <ul class="save-slots">
        {saves.slots.map((slot) => <li data-slot={slot.key}>
          {rename?.key === slot.key ? <form class="save-slot-new" onSubmit={(e) => { e.preventDefault(); manageSaveSlot("rename", slot.key, rename.name); setRename(null); }}>
            <input type="text" aria-label="Save slot name" maxLength={80} value={rename.name} onInput={(e) => setRename({ key: slot.key, name: (e.target as HTMLInputElement).value })} />
            <button class="btn text" type="submit">Rename</button>
            <button class="btn text" type="button" onClick={() => setRename(null)}>Cancel</button>
          </form> : <>
            <div class="save-slot-title"><strong>{slot.name}</strong>{slot.key === saves.active && <span class="muted small">Playing</span>}</div>
            <div class="muted small">Day {slot.day} - {new Date(slot.at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</div>
            <div class="pane-foot">
              {slot.key !== saves.active && <button class="btn text" onClick={() => manageSaveSlot("open", slot.key)}>Open</button>}
              <button class="btn text" onClick={() => setRename({ key: slot.key, name: slot.name })}>Rename</button>
              {slot.key !== saves.active && <button class="btn text" onClick={() => setRemoving(slot.key)}>Delete</button>}
            </div>
          </>}
          {removing === slot.key && <div class="callout bad">
            <p>Delete "{slot.name}" from this browser?</p>
            <div class="pane-foot"><button class="btn text" onClick={() => { manageSaveSlot("delete", slot.key); setRemoving(null); }}>Delete saved game</button><button class="btn text" onClick={() => setRemoving(null)}>Keep it</button></div>
          </div>}
        </li>)}
      </ul>
      <div class="h">Save files</div>
      <p class="note muted">{saves.writable && at ? `Saved in this browser at ${new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.` : "Download a copy to keep outside this browser."}</p>
      <div class="pane-foot">
        <button class="btn text" onClick={exportGame}>
          Save to a file
        </button>
        <button class="btn text" onClick={pick}>
          Load a file
        </button>
      </div>
      {confirm ? (
        <div class="callout bad">
          <p>Start again in an empty city? The game in this slot is replaced. Save to a new slot or a file first to keep it.</p>
          <div class="pane-foot">
            <button
              class="btn text"
              onClick={() => {
                setConfirm(false);
                newGame();
              }}
            >
              Start a new game
            </button>
            <button class="btn text" onClick={() => setConfirm(false)}>
              Keep playing
            </button>
          </div>
        </div>
      ) : (
        <div class="pane-foot">
          <button class="btn text" onClick={() => setConfirm(true)}>
            New game
          </button>
        </div>
      )}
    </>
  );
}
