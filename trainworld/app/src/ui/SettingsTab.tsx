// Settings (SPEC 8): the display settings and the theme, remembered per browser. The saves are in
// their own panel (ui/SavesPanel.tsx).

import type { ComponentChildren } from "preact";
import { useState } from "preact/hooks";
import { CURRENCIES, CURRENCY, isCurrency } from "../game/currency";
import { exact } from "../game/format";
import { awayNear, rushRun, whenAway, whenBack, whenOut } from "../game/days";
import { ask, display, setDisplay, shade, SHADE_NAMES, theme, THEMES, UI_SCALES, uiScale, type Shade, type Theme } from "../game/ui";
import { moneyState } from "../game/money";
import type { BubbleScheme } from "../game/palette";
import { DETAIL_RANGE } from "../map/demandOverlay";
import { makeSandbox, saveGuide } from "../workers/clockClient";
import { mobileDevice } from "../workers/deviceMemory";
import { Check, Icon, Segmented, Stepper } from "./widgets";
import { Credits } from "./Credits";
import { helpOpen } from "./HelpPanel";
import { accountState } from "../game/account";
import { AccountButton, AccountSection, loginText } from "./AccountPanel";
import { CommunityButton } from "./Community";
import { PurposeSelect } from "./DemandSelector";
import { FeedbackButton, FeedbackPanel } from "./FeedbackPanel";
import { t, tNumber, type MessageKey } from "../i18n";
import { LanguageSelect } from "./LanguageSelect";

const THEME_NAME: Record<Theme, MessageKey> = { pink: "settingsTab.pink", blue: "settingsTab.blue", green: "settingsTab.green" };
const SHADE_NAME: Record<Shade, MessageKey> = { light: "settingsTab.light", dark: "settingsTab.dark" };
type DayShown = "average" | "weekday" | "saturday" | "sunday" | "away" | "out" | "back";
const SCHEME_OPTIONS: [BubbleScheme, MessageKey][] = [["mode", "settingsTab.scheme3"], ["ramp", "settingsTab.scheme2"], ["purpose", "settingsTab.schemePurpose"], ["colorblind", "settingsTab.schemeColorblind"]];
const DAY_OPTIONS: [DayShown, MessageKey][] = [["average", "settingsTab.average"], ["weekday", "settingsTab.weekday"], ["saturday", "settingsTab.sat"], ["sunday", "settingsTab.sun"]];
type Projection = "flat" | "globe";
type Ground = "map" | "satellite";
const GROUND_OPTIONS: [Ground, MessageKey][] = [["map", "settingsTab.mapDrawn"], ["satellite", "settingsTab.satellite"]];
const PROJECTION_OPTIONS: [Projection, MessageKey][] = [["flat", "settingsTab.flat"], ["globe", "settingsTab.globe"]];
// Map detail (the saved key is still `lighterMap`): Low is the lighter map that was "on", High is "off"
type LighterMap = "on" | "off";
const LIGHTER_OPTIONS: [LighterMap, MessageKey][] = [["on", "settingsTab.low"], ["off", "settingsTab.high"]];
/** Options worded in the current language, at render. */
const worded = <T,>(opts: [T, MessageKey][]): [T, string][] => opts.map(([v, k]) => [v, t(k)]);
const percent = (v: number) => tNumber(v, { style: "percent", maximumFractionDigits: 0 });

export function SettingsTab() {
  const d = display.value;
  // T-264, T-1373: a superholiday's days while one is within 21 days, each an icon named on hover
  const away = awayNear.value, rush = rushRun.value;
  // on a row of their own under the four, which leave no room beside them in the dock
  const days: [DayShown, ComponentChildren][] = [];
  const wOut = whenOut.value, wAway = whenAway.value, wBack = whenBack.value;
  if (rush && wOut) days.push(["out", <span class="day-away" data-tip={t("settingsTab.dayOf", { holiday: wOut })} aria-label={wOut}><Icon.goOut /></span>]);
  if (away && wAway) days.push(["away", <span class="day-away" data-tip={t("settingsTab.dayOf", { holiday: wAway })} aria-label={wAway}><Icon.calendar /></span>]);
  if (rush && wBack) days.push(["back", <span class="day-away" data-tip={t("settingsTab.dayOf", { holiday: wBack })} aria-label={wBack}><Icon.goBack /></span>]);
  return (
    <>
      <div class="settings-top">
        <button class={"btn icon help-open" + (helpOpen.value ? " on" : "")} data-tip={t("settingsTab.controls")} aria-label={t("settingsTab.controls")} aria-pressed={helpOpen.value} onClick={() => (helpOpen.value = !helpOpen.peek())}>
          <Icon.help />
        </button>
        <CommunityButton settings />
        <AccountButton kind="settings" />
        {accountState.value === "on" && <span class="login-text muted">{loginText()}</span>}
      </div>
      <LanguageRow />
      <div class="h">{t("settingsTab.bubbles")}</div>
      {/* the day every demand figure is for, not only the bubbles' (T-2345) */}
      <div class="slider" data-tip={t("settingsTab.dayTip")}>
        <span>{t("settingsTab.day")}</span>
        <Segmented<DayShown> options={worded(DAY_OPTIONS)} value={days.some(([k]) => k === d.dayShown) ? d.dayShown : DAY_OPTIONS.some(([k]) => k === d.dayShown) ? d.dayShown : "average"} onChange={(v) => setDisplay("dayShown", v)} />
      </div>
      {days.length > 0 && (
        <div class="slider day-super">
          <span />
          <Segmented<DayShown> options={days} value={d.dayShown} onChange={(v) => setDisplay("dayShown", v)} />
        </div>
      )}
      <div class="slider">
        <span>{t("settingsTab.purpose")}</span>
        <PurposeSelect />
      </div>
      <Slider label={t("settingsTab.bubbleSize")} min={0.25} max={2} step={0.05} value={d.bubbleSize} percent set={(v) => setDisplay("bubbleSize", v)} />
      <Slider label={t("settingsTab.opacity")} min={0.2} max={1} step={0.05} value={d.bubbleOpacity} percent set={(v) => setDisplay("bubbleOpacity", v)} />
      {/* finer bubbles are more and smaller ones, at the same area per trip */}
      <Slider label={t("settingsTab.detail")} min={DETAIL_RANGE.min} max={DETAIL_RANGE.max} step={1} value={d.bubbleDetail}
        tip={t("settingsTab.detailTip")} set={(v) => setDisplay("bubbleDetail", v)} />
      {/* the bubbles' colours; bars and swatches by mode keep the mode colours */}
      <div class="slider">
        <span>{t("settingsTab.colors")}</span>
        <Segmented<BubbleScheme> options={worded(SCHEME_OPTIONS)} value={d.bubbleColours} onChange={(v) => setDisplay("bubbleColours", v)} />
      </div>
      {d.bubbleColours === "ramp" && (
        <div class="pickers">
          <label>
            <input type="color" value={d.rampFrom} onInput={(e) => setDisplay("rampFrom", (e.target as HTMLInputElement).value)} />
            {t("settingsTab.ramp0")}
          </label>
          <label>
            <input type="color" value={d.rampTo} onInput={(e) => setDisplay("rampTo", (e.target as HTMLInputElement).value)} />
            {t("settingsTab.ramp100")}
          </label>
        </div>
      )}
      <div class="h">{t("settingsTab.stationCircles")}</div>
      <Slider label={t("settingsTab.circleSize")} min={0.5} max={2} step={0.05} value={d.stationCircleSize} percent set={(v) => setDisplay("stationCircleSize", v)} />
      <div class="h">{t("settingsTab.show")}</div>
      <div class="checks">
        <Check checked={d.trains} onChange={(v) => setDisplay("trains", v)}>
          {t("settingsTab.trains")}
        </Check>
        <Check checked={d.stationNames} onChange={(v) => setDisplay("stationNames", v)}>
          {t("settingsTab.stationNames")}
        </Check>
        <Check checked={d.capacity} onChange={(v) => setDisplay("capacity", v)}>
          <span data-tip={t("settingsTab.capacityTip")}>{t("settingsTab.capacity")}</span>
        </Check>
        <Check checked={d.fullMarks} onChange={(v) => setDisplay("fullMarks", v)}>
          <span data-tip={t("settingsTab.crowdedTip")}>{t("settingsTab.crowded")}</span>
        </Check>
        <Check checked={d.flatMarks} onChange={(v) => setDisplay("flatMarks", v)}>
          {t("settingsTab.flatMarks")}
        </Check>
        <Check checked={d.rampMarks} onChange={(v) => setDisplay("rampMarks", v)}>
          {t("settingsTab.rampMarks")}
        </Check>
        <Check checked={d.lineBands} onChange={(v) => setDisplay("lineBands", v)}>
          {t("settingsTab.lineBands")}
        </Check>
        <Check checked={d.basemapLabels} onChange={(v) => setDisplay("basemapLabels", v)}>
          {t("settingsTab.mapLabels")}
        </Check>
        <Check checked={d.realNetwork} onChange={(v) => setDisplay("realNetwork", v)}>
          {t("settingsTab.realNetwork")}
        </Check>
        <Check checked={d.hillshade} onChange={(v) => setDisplay("hillshade", v)}>
          <span data-tip={t("settingsTab.terrainTip")}>{t("settingsTab.terrain")}</span>
        </Check>
      </div>
      <div class="h">{t("settingsTab.riders")}</div>
      <div class="checks">
        <Check checked={d.trainLoad} onChange={(v) => setDisplay("trainLoad", v)}>
          {t("settingsTab.trainLoad")}
        </Check>
        <Check checked={d.platformCrowds !== "off"} onChange={(v) => setDisplay("platformCrowds", v ? "patch" : "off")}>
          {t("settingsTab.platformLoad")}
        </Check>
      </div>
      <div class="h">{t("settingsTab.upClose")}</div>
      <div class="checks">
        <Check checked={d.faces} onChange={(v) => setDisplay("faces", v)}>
          {t("settingsTab.faces")}
        </Check>
        <Check checked={d.sound} onChange={(v) => setDisplay("sound", v)}>
          {t("settingsTab.sound")}
        </Check>
      </div>
      {d.sound && <Slider label={t("settingsTab.volume")} min={0.1} max={1} step={0.05} value={d.soundVolume} percent set={(v) => setDisplay("soundVolume", v)} />}
      <div class="h">{t("settingsTab.map")}</div>
      <div class="stack">
        <Segmented<Projection> options={worded(PROJECTION_OPTIONS)} value={d.projection === "globe" ? "globe" : "flat"} onChange={(v) => setDisplay("projection", v)} />
        <Segmented<Ground> options={worded(GROUND_OPTIONS)} value={d.satellite ? "satellite" : "map"} onChange={(v) => setDisplay("satellite", v === "satellite")}
          tips={(v) => (v === "satellite" ? t("settingsTab.satelliteTip") : undefined)} />
      </div>
      <Slider label={t("settingsTab.lineWidth")} min={0.5} max={2} step={0.05} value={d.lineWidth} percent set={(v) => setDisplay("lineWidth", v)} />
      <Slider label={t("settingsTab.basemap")} min={0} max={1} step={0.05} value={d.basemapContrast} percent
        tip={t("settingsTab.basemapTip")} set={(v) => setDisplay("basemapContrast", v)} />
      <div class="checks">
        <Check checked={d.rotateMap} onChange={(v) => setDisplay("rotateMap", v)}>
          {t("settingsTab.rotate")}
        </Check>
      </div>
      <div class="h">{t("settingsTab.theme")}</div>
      <div class="stack">
        <Segmented options={SHADE_NAMES.map((s) => [s, t(SHADE_NAME[s])] as [Shade, string])} value={shade.value} onChange={(v) => (shade.value = v)} />
        <Segmented options={THEMES.map((th) => [th, t(THEME_NAME[th])] as [Theme, string])} value={theme.value} onChange={(v) => (theme.value = v)} />
      </div>
      <UiScale />
      <div class="h">{t("settingsTab.unitsPerf")}</div>
      <div class="checks">
        <Check checked={d.usUnits} onChange={(v) => setDisplay("usUnits", v)}>
          {t("settingsTab.usUnits")}
        </Check>
        <Check checked={d.moreCores} onChange={(v) => setDisplay("moreCores", v)}>
          <span data-tip={t("settingsTab.moreCoresTip")}>{t("settingsTab.moreCores")}</span>
        </Check>
        <Check checked={d.runInBackground} onChange={(v) => setDisplay("runInBackground", v)}>
          {t("settingsTab.runInBackground")}
        </Check>
        {mobileDevice() && (
          <Check checked={d.mobileIntercity} onChange={(v) => setDisplay("mobileIntercity", v)}>
            <span data-tip={t("settingsTab.mobileIntercityTip")}>{t("settingsTab.mobileIntercity")}</span>
          </Check>
        )}
      </div>
      <CurrencyRow />
      <div class="slider" data-tip={t("settingsTab.mapDetailTip")}>
        <span>{t("settingsTab.mapDetail")}</span>
        <Segmented<LighterMap> options={worded(LIGHTER_OPTIONS)} value={d.lighterMap} onChange={(v) => setDisplay("lighterMap", v)} />
      </div>
      <GameMode />
      <AccountSection />
      <FeedbackButton />
      <FeedbackPanel />
      <Credits />
    </>
  );
}

/** T-2798: the currency money is shown in, a select as the purpose's; the game counts US$ whatever it says. */
function CurrencyRow() {
  const [mouse, setMouse] = useState(false);
  const cur = isCurrency(display.value.currency) ? display.value.currency : "USD";
  const spec = CURRENCY[cur];
  return (
    <div class="slider" data-tip={cur === "USD" ? undefined : t("settingsTab.currencyTip", { rate: spec.rate >= 100 ? exact(spec.rate) : String(spec.rate), code: spec.label.split(" ")[0] })}>
      <span>{t("settingsTab.currency")}</span>
      <label class={"purpose-filter" + (mouse ? " mouse" : "")}>
        <select aria-label={t("settingsTab.currency")} value={cur}
          onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
          onChange={(e) => { const v = (e.currentTarget as HTMLSelectElement).value; if (isCurrency(v)) setDisplay("currency", v); }}>
          {CURRENCIES.map((c) => <option key={c} value={c}>{CURRENCY[c].label}</option>)}
        </select>
      </label>
    </div>
  );
}

/** T-2636: the language, right under the first row; first from the browser. */
const LanguageRow = () => <LanguageSelect class="slider" />;

/** A labelled range input. */
function Slider({ label, min, max, step, value, tip, percent: pct, set }: { label: string; min: number; max: number; step: number; value: number; tip?: string; percent?: boolean; set: (v: number) => void }) {
  return (
    <label class="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} data-tip={tip} onInput={(e) => set(Number((e.target as HTMLInputElement).value))} />
      {/* T-1851: the value in a fixed-width tabular figure, so the bar never changes length while dragging */}
      <span class="slider-pct">{pct ? percent(value) : ""}</span>
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
      <div class="h">{t("settingsTab.game")}</div>
      <div class="game-mode">
        <span data-tip={sandbox ? t("settingsTab.sandboxTip") : undefined}>{sandbox ? t("settingsTab.sandbox") : t("settingsTab.normal")}</span>
        {!sandbox && <button class="btn text" data-mode-switch onClick={() => (ask.value = {
          text: t("settingsTab.sandboxQuestion"),
          yes: t("settingsTab.sandboxSwitch"), no: t("settingsTab.sandboxStay"), run: makeSandbox,
        })}>{t("settingsTab.sandboxSwitch")}</button>}
      </div>
      {!sandbox && (
        <div class="game-mode">
          <span>{t("settingsTab.tutorial")}</span>
          <button class="btn text" data-guide-restart onClick={() => saveGuide({ done: 0 })}>{t("settingsTab.startAgain")}</button>
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
  return (
    <>
      <div class="h">{t("settingsTab.uiScale")}</div>
      <Stepper wide label={t("settingsTab.uiScaleLabel")} value={<span>{percent(s)}</span>} onStep={step} />
    </>
  );
}
