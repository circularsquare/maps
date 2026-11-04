// T-2282: the track and station tools' options on the map, in the selection bar's place (top centre,
// the map's bottom edge on a phone), shown only while that tool is on; T-2284: the click mode's
// colour as an aura round the map.

import { computed } from "@preact/signals";
import { useState } from "preact/hooks";
import { distUnit, length, levelLabel, levelName, speed } from "../game/format";
import { linking } from "../game/links";
import { moneyState } from "../game/money";
import { CLICK_MODE_COLOURS, inkOn, LEVEL_COLOURS } from "../game/palette";
import { stepPlatform } from "../game/boarding";
import { autoRamp, buildLevel, buildPlatform, DESIGN_SPEEDS, designSpeed, keepPlacing, setTrackType, snapBeside, snapReal, tool, trackType, type TrackType } from "../game/ui";
import { costBackground } from "../map/trackTagRows";
import { levelTip, multText, trackMult, unitPrice, upkeepAt } from "./trackPrice";
import { Icon, LevelButtons, Segmented, Stepper } from "./widgets";

export type ClickMode = keyof typeof CLICK_MODE_COLOURS | "select";
/** What a click on the map does now: the tool, or picking a passage's other station (T-197). */
export const clickMode = computed<ClickMode>(() => (linking.value !== null ? "passage" : tool.value));

const TRACK_TYPES: [TrackType, string][] = [[2, "Regular"], ["hsr", "High speed"]];
const SPEED_TIP = "Attempt to draw track straight enough to reach this speed.";

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
  const m = moneyState.value;
  const mult = trackMult(lv, tt);
  const upkeep = upkeepAt(lv, tt, m);
  const [open, setOpen] = useState<"level" | "type" | "speed" | null>(null);
  const toggle = (k: "level" | "type" | "speed") => setOpen(open === k ? null : k);
  return (
    <>
      <div class="tool-bar track-bar" role="group" aria-label="Track" style={{ "--mode": CLICK_MODE_COLOURS.track }}>
        <div class="tb-row">
          <span class="muted tb-wide">Level</span>
          <div class="tb-wide"><LevelButtons value={lv} tip={(l) => levelTip(l, tt, m)} onPick={(l) => (buildLevel.value = l)} /></div>
          <button class="btn lv on tb-narrow tb-level" style={{ "--lc": LEVEL_COLOURS[lv], "--lk": inkOn(LEVEL_COLOURS[lv]) }}
            data-tip={levelTip(lv, tt, m)} aria-label="Level" aria-expanded={open === "level"} onClick={() => toggle("level")}>{levelLabel(lv)}</button>
          <button class={"btn icon tb-ramp" + (autoRamp.value ? " on" : "")} aria-label="Ramp over track you cross" aria-pressed={autoRamp.value}
            data-tip="Automatically try to ramp over crossed tracks at the same level."
            onClick={() => (autoRamp.value = !autoRamp.peek())}><Icon.autoRamp /></button>
          <p class="muted small price tb-wide" data-tip={`${levelName(lv)}${tt === "hsr" ? ", high-speed track" : ""}${upkeep ? "\n" + upkeep : ""}`}>
            ${unitPrice(lv, tt)}M per {distUnit()} (<span class={costBackground(mult) ? "cost-mult tinted" : "cost-mult"} style={{ background: costBackground(mult) }}>{multText(lv, tt)} cost</span>)
          </p>
        </div>
        <div class="tb-row">
          <div class="tb-wide"><Segmented<TrackType> options={TRACK_TYPES} value={tt} onChange={setTrackType} /></div>
          <button class="btn tb-narrow tb-type" aria-label="Track type" aria-expanded={open === "type"} onClick={() => toggle("type")}>{tt === "hsr" ? "High speed" : "Regular"}</button>
          <span class="build-toggles">
            <button class={"btn icon" + (keepPlacing.value ? " on" : "")} aria-label="Keep placing track" aria-pressed={keepPlacing.value}
              data-tip="Keep placing track after a click (Caps Lock). Double click or press Esc to stop."
              onClick={() => (keepPlacing.value = !keepPlacing.peek())}><Icon.keepPlacing /></button>
            <button class={"btn icon" + (snapReal.value ? " on" : "")} aria-label="Snap to real world network" aria-pressed={snapReal.value}
              data-tip="Snap to real world network"
              onClick={() => (snapReal.value = !snapReal.peek())}><Icon.snapReal /></button>
            <button class={"btn icon" + (snapBeside.value ? " on" : "")} aria-label="Snap beside track" aria-pressed={snapBeside.value}
              data-tip={"Snap beside track\nHover over the blue region to draw track parallel to another track"}
              onClick={() => (snapBeside.value = !snapBeside.peek())}><Icon.snapBeside /></button>
          </span>
          <span class="tb-wide tb-speed">
            <span data-tip={SPEED_TIP}>Target max speed</span>
            <Stepper value={speed(designSpeed.value, 5)} wide label="target max speed" onStep={stepSpeed} />
          </span>
          <button class="btn tb-narrow tb-speed-btn" data-tip={`Target max speed\n${SPEED_TIP}`} aria-label="Target max speed" aria-expanded={open === "speed"}
            onClick={() => toggle("speed")}>{speed(designSpeed.value, 5).split(" ")[0]}</button>
        </div>
      </div>
      {open === "level" && (
        <div class="tool-bar-pop touch-levels">
          <LevelButtons value={lv} tip={(l) => levelTip(l, tt, m)} onPick={(l) => { buildLevel.value = l; setOpen(null); }} />
        </div>
      )}
      {open === "type" && (
        <div class="tool-bar-pop">
          <Segmented<TrackType> options={TRACK_TYPES} value={tt} onChange={(t) => { setTrackType(t); setOpen(null); }} />
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
  return (
    <div class="tool-bar station-bar" role="group" aria-label="Station" style={{ "--mode": CLICK_MODE_COLOURS.station }}>
      <div class="tb-row">
        <span>Platform length</span>
        <Stepper value={length(buildPlatform.value)} wide label="platform length" onStep={(s) => {
          const to = stepPlatform(buildPlatform.value, s);
          if (to === buildPlatform.value) return false;
          buildPlatform.value = to;
        }} />
      </div>
    </div>
  );
}

/** The map's tool bar for the track and station tools, and the aura while a click does anything
 * but select (the selection bar is `BlueprintBar`). */
export function MapToolBar() {
  const mode = clickMode.value;
  const t = tool.value;
  return (
    <>
      {mode !== "select" && <div class="mode-aura" style={{ "--mode": CLICK_MODE_COLOURS[mode] }} />}
      {t === "track" ? <TrackBar /> : t === "station" ? <StationBar /> : null}
    </>
  );
}
