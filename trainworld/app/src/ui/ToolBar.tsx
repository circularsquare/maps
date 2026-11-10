// T-2282: the track and station tools' options on the map, in the selection bar's place (top centre,
// the map's bottom edge on a phone), shown only while that tool is on; T-2284: the click mode's
// colour as an aura round the map.

import { computed } from "@preact/signals";
import { useState } from "preact/hooks";
import { distUnit, length, levelLabel, levelName, millions, speed } from "../game/format";
import { linking } from "../game/links";
import { moneyState } from "../game/money";
import { CLICK_MODE_COLOURS, inkOn, LEVEL_COLOURS, legibleOn } from "../game/palette";
import { linesByNum, world } from "../game/world";
import { CAR_M, stepPlatform } from "../game/boarding";
import { carsText } from "./inspector/schedule";
import { autoRamp, buildLevel, buildPlatform, buildTramPlatform, DESIGN_SPEEDS, designSpeed, keepPlacing, lineDraft, quadTrack, setQuadTrack, setTrackType, shade, snapBeside, snapReal, snapStreets, stationOnTram, TRAM_PLATFORM, tool, trackType, type TrackType } from "../game/ui";
import { costBackground } from "../map/trackTagRows";
import { levelTip, multText, trackMult, unitPrice, upkeepAt } from "./trackPrice";
import { Icon, LevelButtons, Segmented, Stepper } from "./widgets";
import { t, tRich } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";

export type ClickMode = keyof typeof CLICK_MODE_COLOURS | "select";
/** What a click on the map does now: the tool, or picking a passage's other station (T-197). */
export const clickMode = computed<ClickMode>(() => (linking.value !== null ? "passage" : tool.value));

/** T-2687: the colour the click mode shows (the aura, the descriptor's square): the line's own while its
 * stops are edited, kept legible on the shade (`legibleOn`); the mode's fixed colour otherwise. */
export function clickColour(mode: ClickMode): string {
  if (mode === "select") return "transparent";
  const draft = lineDraft.value, w = world.value;
  if (mode === "line" && draft.line !== null && w) {
    const l = linesByNum(w).get(draft.line);
    if (l) return legibleOn(l.colour, shade.value === "dark");
  }
  return CLICK_MODE_COLOURS[mode];
}

const trackTypes = (): [TrackType, string][] => [[2, t("toolBar.regular")], ["hsr", t("toolBar.highSpeed")], ["tram", t("toolBar.tram")]];
const typeLabel = (tt: TrackType) => (tt === "tram" ? t("toolBar.tram") : tt === "hsr" ? t("toolBar.highSpeed") : t("toolBar.regular"));
const speedTip = () => t("toolBar.speedTip");

function stepSpeed(s: number) {
  const i = DESIGN_SPEEDS.indexOf(designSpeed.value) + s;
  const to = DESIGN_SPEEDS[Math.max(0, Math.min(DESIGN_SPEEDS.length - 1, i))];
  if (to === designSpeed.value) return false;
  designSpeed.value = to;
}

/** Two rows: the level, the ramp switch and the price; the track type, the three placing switches and
 * the target max speed. On a phone one row, the level, type and speed as one button each that opens
 * its choices above the bar. */
function TrackBar() {
  const lv = buildLevel.value;
  const tt = trackType.value;
  const tram = tt === "tram";
  const q = quadTrack.value;
  const m = moneyState.value;
  const mult = trackMult(lv, tt, q);
  const upkeep = upkeepAt(lv, tt, m, q);
  const [open, setOpen] = useState<"level" | "type" | "speed" | null>(null);
  const toggle = (k: "level" | "type" | "speed") => setOpen(open === k ? null : k);
  return (
    <>
      <div class="tool-bar track-bar" role="group" aria-label={t("toolBar.track")} style={{ "--mode": CLICK_MODE_COLOURS.track }}>
        <div class="tb-row">
          <span class="muted tb-wide">{t("toolBar.level")}</span>
          <div class="tb-wide"><LevelButtons value={lv} tip={(l) => levelTip(l, tt, m, q)} onPick={(l) => (buildLevel.value = l)} /></div>
          <button class="btn lv on tb-narrow tb-level" style={{ "--lc": LEVEL_COLOURS[lv], "--lk": inkOn(LEVEL_COLOURS[lv]) }}
            data-tip={levelTip(lv, tt, m, q)} aria-label={t("toolBar.level")} aria-expanded={open === "level"} onClick={() => toggle("level")}>{levelLabel(lv)}</button>
          {!tram && <button class={"btn icon tb-ramp" + (autoRamp.value ? " on" : "")} aria-label={t("toolBar.ramp")} aria-pressed={autoRamp.value}
            data-tip={t("toolBar.rampTip")}
            onClick={() => (autoRamp.value = !autoRamp.peek())}><Icon.autoRamp /></button>}
          <p class="muted small price tb-wide" data-tip={t(tram ? "toolBar.priceTipTram" : "toolBar.priceTip", { level: levelName(lv), hsr: tt === "hsr", street: tram && snapStreets.value }) + (upkeep ? "\n" + upkeep : "")}>
            {tRich("toolBar.price", { price: millions(unitPrice(lv, tt, q)), unit: distUnit(), cost: <span class={costBackground(mult) ? "cost-mult tinted" : "cost-mult"} style={{ background: costBackground(mult) }}>{t("toolBar.cost", { mult: multText(lv, tt, q) })}</span> })}
          </p>
        </div>
        <div class="tb-row">
          <div class="tb-wide"><Segmented<TrackType> options={trackTypes()} value={tt} onChange={setTrackType} /></div>
          <button class="btn tb-narrow tb-type" aria-label={t("toolBar.trackType")} aria-expanded={open === "type"} onClick={() => toggle("type")}>{typeLabel(tt)}</button>
          {!tram && <div class="group tb-count" role="group" aria-label={t("toolBar.trackCount")}>
            <button class={"btn icon" + (!q ? " on" : "")} aria-label={t("toolBar.doubleTrack")} aria-pressed={!q} data-tip={t("toolBar.doubleTrack")}
              onClick={() => setQuadTrack(false)}><Icon.tracks2 /></button>
            <button class={"btn icon" + (q ? " on" : "")} aria-label={t("toolBar.quadTrack")} aria-pressed={q} data-tip={t("toolBar.quadTrack")}
              onClick={() => setQuadTrack(true)}><Icon.tracks4 /></button>
          </div>}
          <span class="build-toggles">
            <button class={"btn icon" + (keepPlacing.value ? " on" : "")} aria-label={t("toolBar.keepPlacing")} aria-pressed={keepPlacing.value}
              data-tip={t(mobileDevice() ? "toolBar.keepPlacingTipTouch" : "toolBar.keepPlacingTip")}
              onClick={() => (keepPlacing.value = !keepPlacing.peek())}><Icon.keepPlacing /></button>
            {tram ? (
              <button class={"btn icon tb-streets" + (snapStreets.value ? " on" : "")} aria-label={t("toolBar.snapStreets")} aria-pressed={snapStreets.value}
                data-tip={t("toolBar.snapStreets") + (mobileDevice() ? "" : "\n" + t("toolBar.snapStreetsCtrl", { on: snapStreets.value }))}
                onClick={() => (snapStreets.value = !snapStreets.peek())}><Icon.snapReal /></button>
            ) : (
              <button class={"btn icon" + (snapReal.value ? " on" : "")} aria-label={t("toolBar.snapReal")} aria-pressed={snapReal.value}
                data-tip={t("toolBar.snapReal")}
                onClick={() => (snapReal.value = !snapReal.peek())}><Icon.snapReal /></button>
            )}
            <button class={"btn icon" + (snapBeside.value ? " on" : "")} aria-label={t("toolBar.snapBeside")} aria-pressed={snapBeside.value}
              data-tip={t("toolBar.snapBesideTip")}
              onClick={() => (snapBeside.value = !snapBeside.peek())}><Icon.snapBeside /></button>
          </span>
          {!tram && <span class="tb-wide tb-speed">
            <span data-tip={speedTip()}>{t("toolBar.targetSpeed")}</span>
            <Stepper value={speed(designSpeed.value, 5)} wide label={t("toolBar.targetSpeedLabel")} onStep={stepSpeed} />
          </span>}
          {!tram && <button class="btn tb-narrow tb-speed-btn" data-tip={`${t("toolBar.targetSpeed")}\n${speedTip()}`} aria-label={t("toolBar.targetSpeed")} aria-expanded={open === "speed"}
            onClick={() => toggle("speed")}>{speed(designSpeed.value, 5).split(" ")[0]}</button>}
        </div>
      </div>
      {open === "level" && (
        <div class="tool-bar-pop touch-levels">
          <LevelButtons value={lv} tip={(l) => levelTip(l, tt, m, q)} onPick={(l) => { buildLevel.value = l; setOpen(null); }} />
        </div>
      )}
      {open === "type" && (
        <div class="tool-bar-pop">
          <Segmented<TrackType> options={trackTypes()} value={tt} onChange={(ty) => { setTrackType(ty); setOpen(null); }} />
        </div>
      )}
      {open === "speed" && (
        <div class="tool-bar-pop">
          <div class="group">
            {DESIGN_SPEEDS.map((s) => (
              <button class={"btn" + (s === designSpeed.value ? " on" : "")} onClick={() => { designSpeed.value = s; setOpen(null); }}>{speed(s, 5)}</button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function StationBar() {
  const tram = stationOnTram.value;
  const len = tram ? buildTramPlatform : buildPlatform;
  return (
    <div class="tool-bar station-bar" role="group" aria-label={t("toolBar.station")} style={{ "--mode": CLICK_MODE_COLOURS.station }}>
      <div class="tb-row">
        <span>{t("toolBar.platformLength")}</span>
        <Stepper value={carsText(Math.round(len.value / CAR_M))} wide label={t("toolBar.platformLengthLabel")} onStep={(s) => {
          const to = tram ? Math.max(TRAM_PLATFORM.min, Math.min(TRAM_PLATFORM.max, len.value + s * TRAM_PLATFORM.step)) : stepPlatform(len.value, s);
          if (to === len.value) return false;
          len.value = to;
        }} />
      </div>
    </div>
  );
}

/** The map's tool bar for the track and station tools, and the aura while a click does anything
 * but select (the selection bar is `BlueprintBar`). */
export function MapToolBar() {
  const mode = clickMode.value;
  const tl = tool.value;
  return (
    <>
      {mode !== "select" && <div class="mode-aura" style={{ "--mode": clickColour(mode) }} />}
      {tl === "track" ? <TrackBar /> : tl === "station" ? <StationBar /> : null}
    </>
  );
}
