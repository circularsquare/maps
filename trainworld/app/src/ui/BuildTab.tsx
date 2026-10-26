// The Build tab: tools, elevation, track type, target speed and platforms, and the blueprint quote.

import { distUnit, length, levelName, perDistance, speed } from "../game/format";
import type { Level } from "../game/types";
import { batch } from "@preact/signals";
import { autoLevel, buildLevel, buildPlatform, DESIGN_SPEEDS, designSpeed, display, keepPlacing, setDisplay, setTrackType, snapReal, tool, trackType, type TrackType } from "../game/ui";
import { stepPlatform } from "../game/boarding";
import { Icon, LevelButtons, Segmented, Stepper } from "./widgets";
import { BlueprintBreakdown, BlueprintControls, BlueprintPick } from "./Blueprint";
import { pickTool, TOOLS } from "./toolKeys";
import { costBackground } from "../map/trackTagRows";

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
const BASE_M_PER_KM = 90;
const MULT: Record<Level, number> = { 3: 1.8, 2: 1.3, 1: 1.0, 0: 0.5, [-1]: 1.0, [-2]: 1.1, [-3]: 1.2 };
/** high-speed track's price over double track (params.rs `HSR_COST`, T-817) */
const HSR_COST = 1.5;

/** All building controls: tools, blueprint quote and construction, level, track and platforms. */
export function BuildTab() {
  const t = tool.value;
  const lv = buildLevel.value;
  const hsrF = trackType.value === "hsr" ? HSR_COST : 1;
  const mult = MULT[lv] * hsrF;
  return (
    <>
      <div class="build-options">
        <div class="group tool-pick">
          {TOOLS.map(([id, label, I, key, title]) => (
            <button class={"btn text" + (t === id ? " on" : "")} data-tip={`${title} (${key})`} onClick={() => pickTool(id)}>
              <I />
              <span class="tool-label">{label}</span>
              <span class="key muted">{key}</span>
            </button>
          ))}
        </div>
        <div class="h">Elevation</div>
        <div class="build-track-row">
          <LevelButtons value={lv} tip={levelName} onPick={(l) => batch(() => { buildLevel.value = l; autoLevel.value = false; })} />
          <button class={"btn icon" + (autoLevel.value ? " on" : "")} aria-label="Elevation of the track you extend" aria-pressed={autoLevel.value}
            data-tip="Draw at the elevation of the track you extend. On open ground, at the elevation picked."
            onClick={() => (autoLevel.value = !autoLevel.peek())}><Icon.autoLevel /></button>
        </div>
        {/* T-1394: the Terrain switch beside the price it changes */}
        <div class="build-track-row price-row">
          <p class="muted small price" data-tip={`${levelName(lv)}${hsrF === 1 ? "" : ", high-speed track"}`}>
            ${Math.round(perDistance(BASE_M_PER_KM * mult) * 10) / 10}M per {distUnit()} (<span class={costBackground(mult) ? "cost-mult tinted" : "cost-mult"} style={{ background: costBackground(mult) }}>{mult.toFixed(hsrF === 1 ? 1 : 2)}x cost</span>)
          </p>
          <button class={"btn icon" + (display.value.hillshade ? " on" : "")} aria-label="Show terrain" aria-pressed={display.value.hillshade}
            data-tip="Show terrain. Track costs more on slopes, hills and mountains."
            onClick={() => setDisplay("hillshade", !display.peek().hillshade)}><Icon.terrain /></button>
        </div>
        <div class="h">Track</div>
        <div class="build-track-row">
          <Segmented<TrackType> options={[[2, "Double"], [4, "Quad"], ["hsr", "High speed"]]} value={trackType.value} onChange={setTrackType} />
          <button class={"btn icon" + (keepPlacing.value ? " on" : "")} aria-label="Keep placing track" aria-pressed={keepPlacing.value}
            data-tip="Keep placing track (Caps Lock). Double click or press Esc to stop."
            onClick={() => (keepPlacing.value = !keepPlacing.peek())}><Icon.keepPlacing /></button>
          <button class={"btn icon" + (snapReal.value ? " on" : "")} aria-label="Snap to the real network" aria-pressed={snapReal.value}
            data-tip="Snap track to the real network"
            onClick={() => (snapReal.value = !snapReal.peek())}><Icon.snapReal /></button>
        </div>
        <ul class="rows build-rows">
          <li>
            <span class="grow" data-tip="Long track pieces are laid out straight enough for this speed">Target max speed</span>
            <Stepper value={speed(designSpeed.value, 5)} wide label="target max speed" onStep={(s) => {
              const i = DESIGN_SPEEDS.indexOf(designSpeed.value) + s;
              designSpeed.value = DESIGN_SPEEDS[Math.max(0, Math.min(DESIGN_SPEEDS.length - 1, i))];
            }} />
          </li>
          <li>
            <span class="grow" data-tip="Platform length for new stations">Platform length</span>
            <Stepper value={length(buildPlatform.value)} wide label="platform length" onStep={(s) => (buildPlatform.value = stepPlatform(buildPlatform.value, s))} />
          </li>
        </ul>
        <BlueprintPick />
        <BlueprintBreakdown />
      </div>
      <BlueprintControls />
    </>
  );
}
