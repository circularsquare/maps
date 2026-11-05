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
import { PurposeSelect } from "./DemandSelector";
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
      {/* the day every demand figure is for, not only the bubbles' (T-2345) */}
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
      <div class="slider">
        <span>Purpose</span>
        <PurposeSelect />
      </div>
      <Slider label="Size" min={0.25} max={2} step={0.05} value={d.bubbleSize} percent set={(v) => setDisplay("bubbleSize", v)} />
      <Slider label="Opacity" min={0.2} max={1} step={0.05} value={d.bubbleOpacity} percent set={(v) => setDisplay("bubbleOpacity", v)} />
      {/* finer bubbles are more and smaller ones, at the same area per trip */}
      <Slider label="Detail" min={DETAIL_RANGE.min} max={DETAIL_RANGE.max} step={1} value={d.bubbleDetail}
        tip="How many nearby bubbles merge into one" set={(v) => setDisplay("bubbleDetail", v)} />
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
      <Slider label="Size" min={0.5} max={2} step={0.05} value={d.stationCircleSize} percent set={(v) => setDisplay("stationCircleSize", v)} />
      <div class="h">Show</div>
      <div class="checks">
        <Check checked={d.trains} onChange={(v) => setDisplay("trains", v)}>
          Trains
        </Check>
        <Check checked={d.stationNames} onChange={(v) => setDisplay("stationNames", v)}>
          Station names
        </Check>
        <Check checked={d.capacity} onChange={(v) => setDisplay("capacity", v)}>
          <span data-tip="Places where track or stations can't fit the trains scheduled">Track at capacity</span>
        </Check>
        <Check checked={d.fullMarks} onChange={(v) => setDisplay("fullMarks", v)}>
          <span data-tip="Stretches where trains carry more than their rated load at the busiest hour">Crowded trains</span>
        </Check>
        <Check checked={d.flatMarks} onChange={(v) => setDisplay("flatMarks", v)}>
          Flat junctions and crossings
        </Check>
        <Check checked={d.lineBands} onChange={(v) => setDisplay("lineBands", v)}>
          Lines side by side
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          Map labels
        </Check>
        <Check checked={d.realNetwork} onChange={(v) => setDisplay("realNetwork", v)}>
          Real world network
        </Check>
        <Check checked={d.hillshade} onChange={(v) => setDisplay("hillshade", v)}>
          <span data-tip="Show terrain. Steep terrain costs more to build on.">Terrain</span>
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
      {d.sound && <Slider label="Volume" min={0.1} max={1} step={0.05} value={d.soundVolume} percent set={(v) => setDisplay("soundVolume", v)} />}
      <div class="h">Map</div>
      <div class="stack">
        <Segmented<Projection> options={PROJECTION_OPTIONS} value={d.projection === "globe" ? "globe" : "flat"} onChange={(v) => setDisplay("projection", v)} />
      </div>
      <Slider label="Line width" min={0.5} max={2} step={0.05} value={d.lineWidth} percent set={(v) => setDisplay("lineWidth", v)} />
      <Slider label="Base map" min={0} max={1} step={0.05} value={d.basemapContrast} percent
        tip="Base map contrast" set={(v) => setDisplay("basemapContrast", v)} />
      <div class="checks">
        <Check checked={d.rotateMap} onChange={(v) => setDisplay("rotateMap", v)}>
          Rotate map with right drag
        </Check>
      </div>
      <div class="h">Theme</div>
      <div class="stack">
        <Segmented options={SHADE_NAMES.map((s) => [s, SHADE_NAME[s]] as [Shade, string])} value={shade.value} onChange={(v) => (shade.value = v)} />
        <Segmented options={THEMES.map((t) => [t, THEME_NAME[t]] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      </div>
      <UiScale />
      <div class="h">Units and performance</div>
      <div class="checks">
        <Check checked={d.usUnits} onChange={(v) => setDisplay("usUnits", v)}>
          US units
        </Check>
        <Check checked={d.moreCores} onChange={(v) => setDisplay("moreCores", v)}>
          <span data-tip="Runs up to 5 demand workers, each using about 400 MB of memory. Reload to apply.">Use more cores</span>
        </Check>
        <Check checked={d.runInBackground} onChange={(v) => setDisplay("runInBackground", v)}>
          Run in background
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
function Slider({ label, min, max, step, value, tip, percent, set }: { label: string; min: number; max: number; step: number; value: number; tip?: string; percent?: boolean; set: (v: number) => void }) {
  return (
    <label class="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} data-tip={tip} onInput={(e) => set(Number((e.target as HTMLInputElement).value))} />
      {/* T-1851: the value in a fixed-width tabular figure, so the bar never changes length while dragging */}
      <span class="slider-pct">{percent ? `${Math.round(value * 100)}%` : ""}</span>
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
        <span data-tip={sandbox ? "Unlimited money" : undefined}>{sandbox ? "Sandbox" : "Normal"}</span>
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
  const step = (d: number) => {
    const to = UI_SCALES[Math.max(0, Math.min(UI_SCALES.length - 1, (i < 0 ? UI_SCALES.indexOf(1) : i) + d))];
    if (to === s) return false;
    setDisplay("uiScale", to);
  };
  const pct = Math.round(s * 100);
  return (
    <>
      <div class="h">Panel and text size</div>
      <Stepper wide label="panel and text size" value={<span>{pct}%</span>} onStep={step} />
    </>
  );
}
