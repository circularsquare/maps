// The Build tab: tools, elevation, track type, target speed and platforms, and the blueprint quote.

import { distUnit, length, levelName, levelUpkeepLine, perDistance, speed } from "../game/format";
import { edgeUpkeep } from "../game/regionMoney";
import { moneyState } from "../game/money";
import type { Level } from "../game/types";
import { batch } from "@preact/signals";
import { autoRamp, buildLevel, buildPlatform, DESIGN_SPEEDS, designSpeed, display, keepPlacing, setDisplay, setTrackType, snapBeside, snapReal, tool, trackType, type TrackType } from "../game/ui";
import { stepPlatform } from "../game/boarding";
import { Icon, LevelButtons, Segmented, Stepper } from "./widgets";
import { BlueprintBreakdown, BlueprintControls, BlueprintPick } from "./Blueprint";
import { BUS_KEY, levelKeyName, pickTool, TOOLS } from "./toolKeys";
import { newBusLine } from "../game/lineActions";
import { live } from "../map/live";
import { costBackground } from "../map/trackTagRows";

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
const BASE_M_PER_KM = 75;
const MULT: Record<Level, number> = { 3: 2.0, 2: 1.5, 1: 1.0, 0: 0.5, [-1]: 1.0, [-2]: 1.6, [-3]: 2.0 };
/** high-speed track's price over double track (params.rs `HSR_COST`, T-817) */
const HSR_COST = 1.5;

/** All building controls: tools, blueprint quote and construction, level, track and platforms. */
export function BuildTab() {
  const t = tool.value;
  const lv = buildLevel.value;
  const hsrF = trackType.value === "hsr" ? HSR_COST : 1;
  const mult = MULT[lv] * hsrF;
  const m = moneyState.value;
  const upkeepTip = (l: Level) => (m ? "\n" + levelUpkeepLine(edgeUpkeep({ built: true, tracks: 2, hsr: hsrF !== 1, lengthM: 1000, levelMin: l, levelMax: l }, m)) : "");
  return (
    <>
      <div class="build-options">
        <div class="group tool-pick">
          {TOOLS.map(([id, label, I, key, title]) => (
            <button class={"btn text" + (t === id ? " on" : "")} data-guide={id === "track" ? "1" : id === "station" ? "2" : undefined} data-tip={title.replace(/\n|$/, ` (${key})$&`)} onClick={() => pickTool(id)}>
              <I />
              <span class="tool-label">{label}</span>
              <span class="key muted">{key}</span>
            </button>
          ))}
          {/* T-1514: the bus tool, also on the Lines tab; it starts a new line, so it is no TOOLS entry */}
          <button class={"btn text tool-bus" + (t === "bus" ? " on" : "")} data-tip={`Draw a bus line (${BUS_KEY.toUpperCase()})`}
            onClick={() => (t === "bus" ? live.tools?.exit() : newBusLine())}>
            <Icon.bus />
            <span class="tool-label">Bus</span>
            <span class="key muted">{BUS_KEY.toUpperCase()}</span>
          </button>
        </div>
        <div class="h">Elevation</div>
        <div class="build-track-row">
          <LevelButtons value={lv} tip={(l) => `${levelName(l)} (${levelKeyName(l)})${upkeepTip(l)}`} onPick={(l) => (buildLevel.value = l)} />
        </div>
        {/* T-1394: the Terrain switch beside the price it changes; T-1802: the ramp switch left of it */}
        <div class="build-track-row price-row">
          <p class="muted small price" data-tip={`${levelName(lv)}${hsrF === 1 ? "" : ", high-speed track"}${upkeepTip(lv)}`}>
            ${Math.round(perDistance(BASE_M_PER_KM * mult) * 10) / 10}M per {distUnit()} (<span class={costBackground(mult) ? "cost-mult tinted" : "cost-mult"} style={{ background: costBackground(mult) }}>{mult.toFixed(hsrF === 1 ? 1 : 2)}x cost</span>)
          </p>
          <button class={"btn icon" + (autoRamp.value ? " on" : "")} aria-label="Ramp over track you cross" aria-pressed={autoRamp.value}
            data-tip="Automatically try to ramp over crossed tracks at the same elevation."
            onClick={() => (autoRamp.value = !autoRamp.peek())}><Icon.autoRamp /></button>
          <button class={"btn icon" + (display.value.hillshade ? " on" : "")} aria-label="Show terrain" aria-pressed={display.value.hillshade}
            data-tip="Steep ground costs more to build on."
            onClick={() => setDisplay("hillshade", !display.peek().hillshade)}><Icon.terrain /></button>
        </div>
        <div class="h">Track</div>
        <div class="build-track-row">
          <Segmented<TrackType> options={[[2, "Double"], [4, "Quad"], ["hsr", "High speed"]]} value={trackType.value} onChange={setTrackType} />
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
        </div>
        <ul class="rows build-rows">
          <li>
            <span class="grow" data-tip="Attempt to draw track straight enough to reach this speed.">Target max speed</span>
            <Stepper value={speed(designSpeed.value, 5)} wide label="target max speed" onStep={(s) => {
              const i = DESIGN_SPEEDS.indexOf(designSpeed.value) + s;
              const to = DESIGN_SPEEDS[Math.max(0, Math.min(DESIGN_SPEEDS.length - 1, i))];
              if (to === designSpeed.value) return false;
              designSpeed.value = to;
            }} />
          </li>
          <li>
            <span class="grow">Platform length</span>
            <Stepper value={length(buildPlatform.value)} wide label="platform length" onStep={(s) => {
              const to = stepPlatform(buildPlatform.value, s);
              if (to === buildPlatform.value) return false;
              buildPlatform.value = to;
            }} />
          </li>
        </ul>
        <BlueprintPick />
        <BlueprintBreakdown />
      </div>
      <BlueprintControls />
    </>
  );
}
