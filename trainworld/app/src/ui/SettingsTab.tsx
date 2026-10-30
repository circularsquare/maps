// Settings (SPEC 8): the display settings and the theme, remembered per browser. The saves are in
// their own panel (ui/SavesPanel.tsx).

import type { ComponentChildren } from "preact";
import { awayNear, rushRun, whenAway, whenBack, whenOut } from "../game/days";
import { ask, display, setDisplay, shade, SHADE_NAMES, theme, THEMES, UI_SCALES, uiScale, type Shade, type Theme } from "../game/ui";
import { moneyState } from "../game/money";
import type { BubbleScheme } from "../game/palette";
import { DETAIL_RANGE } from "../map/demandOverlay";
import { makeSandbox, saveGuide } from "../workers/clockClient";
import { Check, Icon, Segmented, Stepper } from "./widgets";
import { Credits } from "./Credits";
import { helpOpen } from "./HelpPanel";
import { accountState } from "../game/account";
import { AccountButton, AccountSection, loginText } from "./AccountPanel";
import { FeedbackButton, FeedbackPanel } from "./FeedbackPanel";

const THEME_NAME: Record<Theme, string> = { pink: "Pink", blue: "Blue", green: "Green" };
const SHADE_NAME: Record<Shade, string> = { light: "Light", dark: "Dark" };
type DayShown = "average" | "weekday" | "saturday" | "sunday" | "away" | "out" | "back";
const SCHEME_OPTIONS: [BubbleScheme, string][] = [["mode", "3-color"], ["ramp", "2-color"]];
const DAY_OPTIONS: [DayShown, string][] = [["average", "Average"], ["weekday", "Weekday"], ["saturday", "Sat"], ["sunday", "Sun"]];
type Projection = "flat" | "globe";
const PROJECTION_OPTIONS: [Projection, string][] = [["flat", "Flat"], ["globe", "Globe"]];
// Map detail (the saved key is still `lighterMap`): Low is the lighter map that was "on", High is "off"
type LighterMap = "auto" | "on" | "off";
const LIGHTER_OPTIONS: [LighterMap, string][] = [["auto", "Auto"], ["on", "Low"], ["off", "High"]];

export function SettingsTab() {
  const d = display.value;
  // T-264, T-1373: a superholiday's days while one is within 21 days, each an icon named on hover
  const away = awayNear.value, rush = rushRun.value;
  // on a row of their own under the four, which leave no room beside them in the dock
  const days: [DayShown, ComponentChildren][] = [];
  const wOut = whenOut.value, wAway = whenAway.value, wBack = whenBack.value;
  if (rush && wOut) days.push(["out", <span class="day-away" data-tip={`A day of ${wOut}`} aria-label={wOut}><Icon.goOut /></span>]);
  if (away && wAway) days.push(["away", <span class="day-away" data-tip={`A day of ${wAway}`} aria-label={wAway}><Icon.calendar /></span>]);
  if (rush && wBack) days.push(["back", <span class="day-away" data-tip={`A day of ${wBack}`} aria-label={wBack}><Icon.goBack /></span>]);
  return (
    <>
      <div class="settings-top">
        <button class={"btn icon help-open" + (helpOpen.value ? " on" : "")} data-tip="Controls" aria-label="Controls" aria-pressed={helpOpen.value} onClick={() => (helpOpen.value = !helpOpen.peek())}>
          <Icon.help />
        </button>
        <AccountButton kind="settings" />
        {accountState.value === "on" && <span class="login-text muted">{loginText()}</span>}
      </div>
      <div class="h">Demand bubbles</div>
      <Slider label="Size" min={0.5} max={2} step={0.05} value={d.bubbleSize} tip={`Bubbles at ${Math.round(d.bubbleSize * 100)}% of their usual size`} set={(v) => setDisplay("bubbleSize", v)} />
      <Slider label="Opacity" min={0.2} max={1} step={0.05} value={d.bubbleOpacity} tip={`Bubbles at ${Math.round(d.bubbleOpacity * 100)}% of their full opacity`} set={(v) => setDisplay("bubbleOpacity", v)} />
      {/* finer bubbles are more and smaller ones, at the same area per trip */}
      <Slider label="Detail" min={DETAIL_RANGE.min} max={DETAIL_RANGE.max} step={1} value={d.bubbleDetail}
        tip="How many bubbles are aggregated together" set={(v) => setDisplay("bubbleDetail", v)} />
      {/* the bubbles' colours; bars and swatches by mode keep the mode colours */}
      <div class="slider">
        <span>Colors</span>
        <Segmented<BubbleScheme> options={SCHEME_OPTIONS} value={d.bubbleColours === "ramp" ? "ramp" : "mode"} onChange={(v) => setDisplay("bubbleColours", v)} />
      </div>
      {d.bubbleColours === "ramp" && (
        <div class="pickers">
          <label>
            <input type="color" value={d.rampFrom} onInput={(e) => setDisplay("rampFrom", (e.target as HTMLInputElement).value)} />
            0% train
          </label>
          <label>
            <input type="color" value={d.rampTo} onInput={(e) => setDisplay("rampTo", (e.target as HTMLInputElement).value)} />
            100% train
          </label>
        </div>
      )}
      <div class="h">Station circles</div>
      <Slider label="Size" min={0.5} max={2} step={0.05} value={d.stationCircleSize} set={(v) => setDisplay("stationCircleSize", v)} />
      {/* the day the demand views are for; the track display is on the map (ui/DemandSelector.tsx) */}
      <div class="slider" data-tip="Which day demand figures are shown for">
        <span>Day</span>
        <Segmented<DayShown> options={DAY_OPTIONS} value={days.some(([k]) => k === d.dayShown) ? d.dayShown : DAY_OPTIONS.some(([k]) => k === d.dayShown) ? d.dayShown : "average"} onChange={(v) => setDisplay("dayShown", v)} />
      </div>
      {days.length > 0 && (
        <div class="slider day-super">
          <span />
          <Segmented<DayShown> options={days} value={d.dayShown} onChange={(v) => setDisplay("dayShown", v)} />
        </div>
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
          <span data-tip="Marks where track cannot smoothly handle scheduled train frequencies">Track at capacity</span>
        </Check>
        <Check checked={d.fullMarks} onChange={(v) => setDisplay("fullMarks", v)}>
          <span data-tip="Marks where trains are full and passengers cannot board">Trains at capacity</span>
        </Check>
        <Check checked={d.flatMarks} onChange={(v) => setDisplay("flatMarks", v)}>
          Flat junctions and crossings
        </Check>
        <Check checked={d.drives} onChange={(v) => setDisplay("drives", v)}>
          <span data-tip="Show arcs for common driving trips">Drives</span>
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
        <Check checked={d.realNetwork} onChange={(v) => setDisplay("realNetwork", v)}>
          <span data-tip="Draw real-world railways on the base map">Real network</span>
        </Check>
        <Check checked={d.hillshade} onChange={(v) => setDisplay("hillshade", v)}>
          <span data-tip="Construction is more expensive in steep terrain">Terrain</span>
        </Check>
      </div>
      <div class="h">Riders on trains and platforms</div>
      <div class="checks">
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          How full each train is
        </Check>
        <Check checked={d.platformCrowds !== "off"} onChange={(v) => setDisplay("platformCrowds", v ? "patch" : "off")}>
          How full each platform is
        </Check>
      </div>
      <div class="h">Stations up close</div>
      <div class="checks">
        <Check checked={d.faces} onChange={(v) => setDisplay("faces", v)}>
          Faces
        </Check>
        <Check checked={d.sound} onChange={(v) => setDisplay("sound", v)}>
          Sound
        </Check>
      </div>
      {d.sound && <Slider label="Volume" min={0.1} max={1} step={0.05} value={d.soundVolume} tip={`All sounds at ${Math.round(d.soundVolume * 100)}% volume`} set={(v) => setDisplay("soundVolume", v)} />}
      <div class="h">Map</div>
      <div class="stack">
        <Segmented<Projection> options={PROJECTION_OPTIONS} value={d.projection === "globe" ? "globe" : "flat"} onChange={(v) => setDisplay("projection", v)} />
      </div>
      <Slider label="Line width" min={0.5} max={2} step={0.05} value={d.lineWidth} set={(v) => setDisplay("lineWidth", v)} />
      <Slider label="Base map" min={0} max={1} step={0.05} value={d.basemapContrast}
        tip="Base map contrast" set={(v) => setDisplay("basemapContrast", v)} />
      <div class="h">Theme</div>
      <div class="stack">
        <Segmented options={SHADE_NAMES.map((s) => [s, SHADE_NAME[s]] as [Shade, string])} value={shade.value} onChange={(v) => (shade.value = v)} />
        <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      </div>
      <UiScale />
      <div class="h">Units and performance</div>
      <div class="checks">
        <Check checked={d.usUnits} onChange={(v) => setDisplay("usUnits", v)}>
          <span data-tip="Miles, mph and feet. Off shows kilometers, km/h and meters.">US units</span>
        </Check>
        <Check checked={d.moreCores} onChange={(v) => setDisplay("moreCores", v)}>
          <span data-tip="Up to 5 demand workers, using more of the processor and about 400 MB of memory each. Takes effect after a reload.">Use more cores</span>
        </Check>
        <Check checked={d.runInBackground} onChange={(v) => setDisplay("runInBackground", v)}>
          <span data-tip="The clock, fares and running costs carry on at the chosen speed while this tab is hidden. Off, the game pauses until the tab is shown again.">Run in background</span>
        </Check>
      </div>
      <div class="slider" data-tip="Show fewer roads and other details">
        <span>Map detail</span>
        <Segmented<LighterMap> options={LIGHTER_OPTIONS} value={d.lighterMap ?? "auto"} onChange={(v) => setDisplay("lighterMap", v)} />
      </div>
      <GameMode />
      <AccountSection />
      <FeedbackButton />
      <FeedbackPanel />
      <Credits />
    </>
  );
}

/** A labelled range input. */
function Slider({ label, min, max, step, value, tip, set }: { label: string; min: number; max: number; step: number; value: number; tip?: string; set: (v: number) => void }) {
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
      {!sandbox && (
        <div class="game-mode">
          <span>Tutorial</span>
          <button class="btn text" data-guide-restart onClick={() => saveGuide({ done: 0 })}>Start again</button>
        </div>
      )}
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
