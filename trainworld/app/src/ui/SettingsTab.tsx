// Settings (T-037): display settings (SPEC 8) and the theme, remembered per browser: the commuter
// bubbles (T-097: off, where commuters live, where they work), trains, station names,
// capacity markers (T-031), map labels, the riders drawn on the network (T-084: line width by the
// track display, train fill; the station circles are in the Stations tab since T-213), the UI
// scale (T-247), and last the trip explorer's button (T-220). The saves are in their own panel
// (T-251, ui/SavesPanel.tsx). T-253: no explanatory notes, only controls and their labels.

import { FONT_KEYS, FONTS, type FontKey } from "../game/fonts";
import { ask, display, setDisplay, shade, SHADE_NAMES, theme, THEMES, tool, UI_SCALES, uiScale, type Shade, type Theme } from "../game/ui";
import { moneyState } from "../game/money";
import type { BubbleScheme } from "../game/palette";
import { DETAIL_RANGE, finestFromZoom } from "../map/demandOverlay";
import { RES_AREA_M2 } from "../workers/commutersProtocol";
import { makeSandbox } from "../workers/clockClient";
import { pickTool, TRIP_KEY } from "./tabs";
import { Check, Segmented, Stepper } from "./widgets";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };
const SHADE_NAME: Record<Shade, string> = { light: "Light", dark: "Dark" };
type DayShown = "average" | "weekday" | "weekend";
const SCHEME_OPTIONS: [BubbleScheme, string][] = [["mode", "Mode mix"], ["ramp", "Train share"]];
const DAY_OPTIONS: [DayShown, string][] = [["average", "Average"], ["weekday", "Weekday"], ["weekend", "Weekend"]];

export function SettingsTab() {
  const d = display.value;
  return (
    <>
      {/* T-169: the slider is the section's only control, so the heading goes with it */}
      <div class="h">Demand bubbles</div>
      {/* T-470: the opacity applies to every bubble, so the section is always there */}
      <label class="slider">
        <span>Size</span>
        <input type="range" min="0.5" max="2" step="0.05" value={d.bubbleSize} data-tip={`Bubbles at ${Math.round(d.bubbleSize * 100)}% of their usual size`}
          onInput={(e) => setDisplay("bubbleSize", Number((e.target as HTMLInputElement).value))} />
      </label>
      <label class="slider">
        <span>Opacity</span>
        <input type="range" min="0.2" max="1" step="0.05" value={d.bubbleOpacity} data-tip={`Bubbles at ${Math.round(d.bubbleOpacity * 100)}% of their full opacity`}
          onInput={(e) => setDisplay("bubbleOpacity", Number((e.target as HTMLInputElement).value))} />
      </label>
      {/* T-795: finer bubbles are more and smaller ones, at the same area per trip */}
      <label class="slider">
        <span>Detail</span>
        <input type="range" min={DETAIL_RANGE.min} max={DETAIL_RANGE.max} step="1" value={d.bubbleDetail}
          data-tip={`Smallest bubbles group trips by areas of ${(RES_AREA_M2[0] / 1e6).toFixed(1)} km², from zoom ${finestFromZoom(d.bubbleDetail).toFixed(1)}`}
          onInput={(e) => setDisplay("bubbleDetail", Number((e.target as HTMLInputElement).value))} />
      </label>
      {/* T-804: the bubbles' colours; bars and swatches by mode keep the mode colours */}
      <div class="slider" data-tip="Mode mix blends red for driving, blue for train and green for walking by share. Train share runs from one colour where nobody rides a train to another where everyone does.">
        <span>Colours</span>
        <Segmented<BubbleScheme> options={SCHEME_OPTIONS} value={d.bubbleColours === "ramp" ? "ramp" : "mode"} onChange={(v) => setDisplay("bubbleColours", v)} />
      </div>
      {d.bubbleColours === "ramp" && (
        <div class="pickers">
          <label data-tip="Bubbles where nobody goes by train">
            <input type="color" value={d.rampFrom} onInput={(e) => setDisplay("rampFrom", (e.target as HTMLInputElement).value)} />
            0% train
          </label>
          <label data-tip="Bubbles where everyone goes by train">
            <input type="color" value={d.rampTo} onInput={(e) => setDisplay("rampTo", (e.target as HTMLInputElement).value)} />
            100% train
          </label>
        </div>
      )}
      {/* T-222: longer-trip bubbles are always drawn; their setting is gone */}
      {/* T-703: the day the demand views are for (T-464), here since it left the map; the track display is on the map (ui/DemandSelector.tsx) */}
      <div class="slider" data-tip="The day the demand figures are for. The average counts 5 weekdays and 2 weekend days over 7. Trains, platforms and riders over the past 24 hours follow the clock.">
        <span>Day</span>
        <Segmented<DayShown> options={DAY_OPTIONS} value={d.dayShown === "weekday" || d.dayShown === "weekend" ? d.dayShown : "average"} onChange={(v) => setDisplay("dayShown", v)} />
      </div>
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
        <Check checked={d.flatMarks} onChange={(v) => setDisplay("flatMarks", v)}>
          <span data-tip="The round marks where track meets or crosses at one level">Flat junctions and crossings</span>
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
        <Check checked={d.realNetwork} onChange={(v) => setDisplay("realNetwork", v)}>
          <span data-tip="Real railways drawn faintly under yours, for reference: main lines, other passenger lines and metros. Appears once you zoom in to a region, brighter while track snaps to it.">Real network</span>
        </Check>
      </div>
      {/* T-213: station demand circles and one circle per complex are in the Stations tab */}
      {/* T-164: each train's own load and each platform's crowd, from the boarding account;
          T-253: the note on how the counts are made is now the platform option's hover */}
      <div class="h">Riders on trains and platforms</div>
      <div class="checks">
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          How full each train is
        </Check>
        {/* T-219: the platform fills up as riders wait; the bands and dots options are gone */}
        <Check checked={d.platformCrowds !== "off"} onChange={(v) => setDisplay("platformCrowds", v ? "patch" : "off")}>
          <span data-tip="Grey fills a platform as riders wait, red when a full train leaves some behind. From street zooms.">How full each platform is</span>
        </Check>
      </div>
      {/* T-686: both from the zoom where platforms are drawn */}
      <div class="h">Stations up close</div>
      <div class="checks">
        <Check checked={d.faces} onChange={(v) => setDisplay("faces", v)}>
          <span data-tip="A small face rises from the platform for every 20 or so riders who get off. Tired where riders have waited 10 minutes on average, sad where the last train left 20 or more behind. From street zooms.">Faces</span>
        </Check>
        <Check checked={d.sound} onChange={(v) => setDisplay("sound", v)}>
          <span data-tip="A chime as a train pulls in and soft pops as riders get on and off. From street zooms, louder the closer you are. Starts after your first click.">Sound</span>
        </Check>
      </div>
      {d.sound && (
        <label class="slider">
          <span>Volume</span>
          <input type="range" min="0.1" max="1" step="0.05" value={d.soundVolume} data-tip={`Station sounds at ${Math.round(d.soundVolume * 100)}% volume`}
            onInput={(e) => setDisplay("soundVolume", Number((e.target as HTMLInputElement).value))} />
        </label>
      )}
      <div class="h">Theme</div>
      <div class="stack">
        <Segmented options={SHADE_NAMES.map((s) => [s, SHADE_NAME[s]] as [Shade, string])} value={shade.value} onChange={(v) => (shade.value = v)} />
        <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      </div>
      <UiScale />
      {/* T-698: a trial of other rounded fonts for the panels, each loaded when first chosen */}
      <div class="h">Font</div>
      <div class="track-modes"><Segmented<FontKey>
        options={FONT_KEYS.map((k) => [k, FONTS[k].label] as [FontKey, string])}
        value={d.font}
        onChange={(v) => setDisplay("font", v)}
      /></div>
      <div class="h">Units and performance</div>
      <div class="checks">
        <Check checked={d.usUnits} onChange={(v) => setDisplay("usUnits", v)}>
          <span data-tip="Miles, mph and feet. Off shows kilometres, km/h and metres.">US units</span>
        </Check>
        <Check checked={d.moreCores} onChange={(v) => setDisplay("moreCores", v)}>
          <span data-tip="Up to 5 demand workers, using more of the processor and about 400 MB of memory each. Counts from the next time the page loads.">Use more cores</span>
        </Check>
        <Check checked={d.runInBackground} onChange={(v) => setDisplay("runInBackground", v)}>
          <span data-tip="The clock, fares and running costs carry on at the chosen speed while this tab is hidden. Off, the game pauses until the tab is shown again.">Run in background</span>
        </Check>
      </div>
      <GameMode />
      <TripStart />
    </>
  );
}

/** T-728: the game's mode; a normal game can be made a sandbox once, after a confirmation, never back. */
function GameMode() {
  const m = moneyState.value;
  if (!m) return null;
  const sandbox = m.mode === "sandbox";
  return (
    <>
      <div class="h">Game</div>
      <div class="game-mode">
        <span data-tip={sandbox ? "Nothing is refused for money and the cash is hidden" : "Building costs money, and an edit is refused when the cash runs short"}>{sandbox ? "Sandbox" : "Normal"}</span>
        {!sandbox && <button class="btn text" data-mode-switch onClick={() => (ask.value = {
          text: "Entering sandbox mode will disable money permanently and irreversibly.",
          yes: "Switch to sandbox mode", no: "Stay in normal mode", run: makeSandbox,
        })}>Switch to sandbox mode</button>}
      </div>
    </>
  );
}

/** T-247: the size of the panels and their text (and of the map's name tags), remembered per browser. */
function UiScale() {
  const s = uiScale.value;
  const i = UI_SCALES.findIndex((v) => Math.abs(v - s) < 0.001);
  const step = (d: number) => setDisplay("uiScale", UI_SCALES[Math.max(0, Math.min(UI_SCALES.length - 1, (i < 0 ? UI_SCALES.indexOf(1) : i) + d))]);
  const pct = Math.round(s * 100);
  return (
    <>
      <div class="h">Panel and text size</div>
      <Stepper wide label="panel and text size" value={<span data-tip={`Panels and text at ${pct}% of their usual size`}>{pct}%</span>} onStep={step} />
    </>
  );
}

/** The trip explorer's button (T-220: here since the City tab went; the T key from anywhere). */
function TripStart() {
  const on = tool.value === "trip";
  return (
    <div class="pane-foot trip-start">
      <button class={"btn text" + (on ? " on" : "")} data-tip={`Click two places or stations on the map to see how people would travel between them (${TRIP_KEY.toUpperCase()})`} onClick={() => pickTool("trip")}>
        Explore a trip<span class="key muted">{TRIP_KEY.toUpperCase()}</span>
      </button>
    </div>
  );
}
