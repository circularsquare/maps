// Settings (SPEC 8): the display settings and the theme, remembered per browser. The saves are in
// their own panel (ui/SavesPanel.tsx).

import { FONT_KEYS, FONTS, type FontKey } from "../game/fonts";
import { ask, display, setDisplay, shade, SHADE_NAMES, theme, THEMES, tool, UI_SCALES, uiScale, type Shade, type Theme } from "../game/ui";
import { moneyState } from "../game/money";
import type { BubbleScheme } from "../game/palette";
import { DETAIL_RANGE, finestFromZoom } from "../map/demandOverlay";
import { liteDetected } from "../map/lite";
import { RES_AREA_M2 } from "../workers/commutersProtocol";
import { makeSandbox } from "../workers/clockClient";
import { pickTool, TRIP_KEY } from "./toolKeys";
import { Check, Icon, Segmented, Stepper } from "./widgets";
import { Credits } from "./Credits";
import { helpOpen } from "./HelpPanel";
import { AccountButton, AccountSection } from "./AccountPanel";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };
const SHADE_NAME: Record<Shade, string> = { light: "Light", dark: "Dark" };
type DayShown ="average" | "weekday" | "saturday" | "sunday";
const SCHEME_OPTIONS: [BubbleScheme, string][] = [["mode", "Mode mix"], ["ramp", "Train share"]];
const DAY_OPTIONS: [DayShown, string][] = [["average", "Average"], ["weekday", "Weekday"], ["saturday", "Sat"], ["sunday", "Sun"]];
type Projection = "flat" | "globe";
const PROJECTION_OPTIONS: [Projection, string][] = [["flat", "Flat"], ["globe", "Globe"]];
type LighterMap = "auto" | "on" | "off";
const LIGHTER_OPTIONS: [LighterMap, string][] = [["auto", "Auto"], ["on", "On"], ["off", "Off"]];

export function SettingsTab() {
  const d = display.value;
  return (
    <>
      <div class="settings-top">
        <button class={"btn icon help-open" + (helpOpen.value ? " on" : "")} data-tip="Controls" aria-label="Controls" aria-pressed={helpOpen.value} onClick={() => (helpOpen.value = !helpOpen.peek())}>
          <Icon.help />
        </button>
        <AccountButton kind="settings" />
      </div>
      <div class="h">Demand bubbles</div>
      <Slider label="Size" min={0.5} max={2} step={0.05} value={d.bubbleSize} tip={`Bubbles at ${Math.round(d.bubbleSize * 100)}% of their usual size`} set={(v) => setDisplay("bubbleSize", v)} />
      <Slider label="Opacity" min={0.2} max={1} step={0.05} value={d.bubbleOpacity} tip={`Bubbles at ${Math.round(d.bubbleOpacity * 100)}% of their full opacity`} set={(v) => setDisplay("bubbleOpacity", v)} />
      {/* finer bubbles are more and smaller ones, at the same area per trip */}
      <Slider label="Detail" min={DETAIL_RANGE.min} max={DETAIL_RANGE.max} step={1} value={d.bubbleDetail}
        tip={`Smallest bubbles group trips by areas of ${(RES_AREA_M2[0] / 1e6).toFixed(1)} km², from zoom ${finestFromZoom(d.bubbleDetail).toFixed(1)}`} set={(v) => setDisplay("bubbleDetail", v)} />
      {/* the bubbles' colours; bars and swatches by mode keep the mode colours */}
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
      {/* the day the demand views are for; the track display is on the map (ui/DemandSelector.tsx) */}
      <div class="slider" data-tip="The day the demand figures are for. The average counts 5 weekdays, a Saturday and a Sunday over 7. Trains, platforms and riders over the past 24 hours follow the clock.">
        <span>Day</span>
        <Segmented<DayShown> options={DAY_OPTIONS} value={DAY_OPTIONS.some(([k]) => k === d.dayShown) ? d.dayShown : "average"} onChange={(v) => setDisplay("dayShown", v)} />
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
        <Check checked={d.fullMarks} onChange={(v) => setDisplay("fullMarks", v)}>
          <span data-tip="Red edges in the Traffic view where track is at capacity or riders wait for a train with room. The longer the wait, the stronger the red.">Full stretches</span>
        </Check>
        <Check checked={d.flatMarks} onChange={(v) => setDisplay("flatMarks", v)}>
          <span data-tip="The round marks where track meets or crosses at one level">Flat junctions and crossings</span>
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
        <Check checked={d.realNetwork} onChange={(v) => setDisplay("realNetwork", v)}>
          <span data-tip="Draw real-world railways on the base map">Real network</span>
        </Check>
        <Check checked={d.hillshade} onChange={(v) => setDisplay("hillshade", v)}>
          <span data-tip="Shade hills and mountains on the base map. Track on rolling ground costs 1.25 times as much, on hilly ground 1.6 and on mountains 2.5, except in bored tunnels (elevation -2 and -3).">Terrain</span>
        </Check>
      </div>
      <div class="h">Riders on trains and platforms</div>
      <div class="checks">
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          How full each train is
        </Check>
        <Check checked={d.platformCrowds !== "off"} onChange={(v) => setDisplay("platformCrowds", v ? "patch" : "off")}>
          <span data-tip="Grey fills a platform as riders wait, red when a full train leaves some behind. From street zooms.">How full each platform is</span>
        </Check>
      </div>
      <div class="h">Stations up close</div>
      <div class="checks">
        <Check checked={d.faces} onChange={(v) => setDisplay("faces", v)}>
          <span data-tip="A small face rises from the platform for every 20 or so riders who get off. Tired where riders have waited 10 minutes on average, sad where the last train left 20 or more behind. From street zooms.">Faces</span>
        </Check>
        <Check checked={d.sound} onChange={(v) => setDisplay("sound", v)}>
          <span data-tip="A chime as a train pulls in and soft pops as riders get on and off. From street zooms, louder the closer you are. Small sounds as you place and construct blueprint. Starts after your first click.">Sound</span>
        </Check>
      </div>
      {d.sound && <Slider label="Volume" min={0.1} max={1} step={0.05} value={d.soundVolume} tip={`Sounds at ${Math.round(d.soundVolume * 100)}% volume`} set={(v) => setDisplay("soundVolume", v)} />}
      <div class="h">Map</div>
      <div class="stack" data-tip="Zoomed in to a city, the globe becomes the flat map">
        <Segmented<Projection> options={PROJECTION_OPTIONS} value={d.projection === "globe" ? "globe" : "flat"} onChange={(v) => setDisplay("projection", v)} />
      </div>
      <Slider label="Line width" min={0.5} max={2} step={0.05} value={d.lineWidth} tip={`Lines, track and trains at ${Math.round(d.lineWidth * 100)}% of their usual width`} set={(v) => setDisplay("lineWidth", v)} />
      <Slider label="Base map" min={0} max={1} step={0.05} value={d.basemapContrast}
        tip={`Roads, place names, parks and the real network at ${Math.round(d.basemapContrast * 100)}% contrast against the land. Water keeps its colour.`} set={(v) => setDisplay("basemapContrast", v)} />
      <div class="h">Theme</div>
      <div class="stack">
        <Segmented options={SHADE_NAMES.map((s) => [s, SHADE_NAME[s]] as [Shade, string])} value={shade.value} onChange={(v) => (shade.value = v)} />
        <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      </div>
      <UiScale />
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
          <span data-tip="Up to 5 demand workers, using more of the processor and about 400 MB of memory each. Takes effect after a reload.">Use more cores</span>
        </Check>
        <Check checked={d.runInBackground} onChange={(v) => setDisplay("runInBackground", v)}>
          <span data-tip="The clock, fares and running costs carry on at the chosen speed while this tab is hidden. Off, the game pauses until the tab is shown again.">Run in background</span>
        </Check>
      </div>
      <div class="slider" data-tip={"Fewer roads, buildings and place names, drawn at a lower resolution. Auto turns it on if the map moves slowly." + (d.lighterMap === "auto" && liteDetected.value ? "\nOn now" : "")}>
        <span>Lighter map</span>
        <Segmented<LighterMap> options={LIGHTER_OPTIONS} value={d.lighterMap ?? "auto"} onChange={(v) => setDisplay("lighterMap", v)} />
      </div>
      <GameMode />
      <AccountSection />
      <TripStart />
      <Credits />
    </>
  );
}

/** A labelled range input. */
function Slider({ label, min, max, step, value, tip, set }: { label: string; min: number; max: number; step: number; value: number; tip: string; set: (v: number) => void }) {
  return (
    <label class="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} data-tip={tip} onInput={(e) => set(Number((e.target as HTMLInputElement).value))} />
    </label>
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
        <span data-tip={sandbox ? "Infinite money, and the cash is hidden" : "Construction costs money, and an edit is refused when the cash runs short"}>{sandbox ? "Sandbox" : "Normal"}</span>
        {!sandbox && <button class="btn text" data-mode-switch onClick={() => (ask.value = {
          text: "Money cannot be turned back on in this game.",
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

/** The trip explorer's button (the T key from anywhere). */
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
