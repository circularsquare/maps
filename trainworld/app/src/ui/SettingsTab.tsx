// Settings (T-037): display settings (SPEC 8) and the theme, remembered per browser: the commuter
// bubbles (T-097: off, where commuters live, where they work), track colour,
// trains, station names, capacity markers (T-031), map labels, the riders drawn on the network
// (T-084: line width, station size, train fill). Then the game's saves (T-029).

import { useState } from "preact/hooks";
import { levelLabel } from "../game/format";
import { lastSaved } from "../game/money";
import { exportGame, importGame, newGame } from "../workers/clockClient";
import { LEVEL_COLOURS } from "../game/palette";
import { LEVELS } from "../game/types";
import { display, setDisplay, theme, THEMES, type Theme } from "../game/ui";
import { Check, Segmented } from "./widgets";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };

export function SettingsTab() {
  const d = display.value;
  return (
    <>
      <div class="h">Commuters on the map</div>
      <Segmented<"off" | "home" | "work">
        options={[
          ["off", "Off"],
          ["home", "Homes"],
          ["work", "Jobs"],
        ]}
        value={d.commuters ? d.commuterEnd : "off"}
        onChange={(v) => (display.value = v === "off" ? { ...d, commuters: false } : { ...d, commuters: true, commuterEnd: v })}
      />
      {d.commuters && (
        <label class="slider">
          <span>Bubble size</span>
          <input type="range" min="0.5" max="2" step="0.05" value={d.bubbleSize} onInput={(e) => setDisplay("bubbleSize", Number((e.target as HTMLInputElement).value))} />
        </label>
      )}
      <div class="h">Track colour</div>
      <Segmented<"line" | "height">
        options={[
          ["line", "Line"],
          ["height", "Height"],
        ]}
        value={d.trackColour}
        onChange={(v) => setDisplay("trackColour", v)}
      />
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
      <div class="h">Riders on the network</div>
      <div class="checks">
        <Check checked={d.lineLoad} onChange={(v) => setDisplay("lineLoad", v)}>
          Line width
        </Check>
        <Check checked={d.stationRiders} onChange={(v) => setDisplay("stationRiders", v)}>
          Station size
        </Check>
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          How full trains are
        </Check>
      </div>
      <div class="h">Theme</div>
      <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      <GameSaves />
    </>
  );
}

/** Saves (T-029): the game saves itself in this browser; a file can be exported and loaded. */
function GameSaves() {
  const [confirm, setConfirm] = useState(false);
  const at = lastSaved.value;
  const pick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".save";
    input.onchange = () => input.files?.[0] && void importGame(input.files[0]);
    input.click();
  };
  return (
    <>
      <div class="h">Game</div>
      <p class="note muted">{at ? `Saved in this browser at ${new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.` : "Saved in this browser as you play."}</p>
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
          <p>Start again in an empty city? This game is lost unless you save it to a file first.</p>
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
