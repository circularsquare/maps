import { activeCity, cityById, viewedZone } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { levelLabel, speed } from "../game/format";
import { loadProgress } from "../game/loading";
import { LEVEL_COLOURS, SPEED_STOPS } from "../game/palette";
import { LEVELS } from "../game/types";
import { asFilter, FILTER_PURPOSES, purposeColour } from "../game/purposes";
import { useState } from "preact/hooks";
import { display, purposeFilter, setDisplay } from "../game/ui";
import { StationsFull } from "./StationsFull";
import { StatusTag } from "./StatusTag";

/** The map's upper left: the demand buttons with the status tag to their right (T-349), and under
 * them the stations full alert (T-231). */
export function MapTopLeft() {
  return <div class="map-top-left">
    <div class="map-top-row">
      <DemandSelector />
      <StatusTag />
    </div>
    <TrackDisplay />
    <StationsFull />
  </div>;
}

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  const ready = !!viewedZone.value && cityDemandViews.value.has(activeCity.value);
  // T-149: the viewed city may still be loading after a pan
  const load = loadProgress.value;
  const pending = !viewedZone.value ? "Nothing modelled here" // T-318: dead space
    : load?.cities.includes(activeCity.value) ? `Loading ${cityById(activeCity.value).name}, ${Math.floor(load.fraction * 100)}%` : "Calculating demand";
  const selected = d.commuters ? d.commuterEnd : "none";
  return <div class="demand-selector" role="group" aria-label="Demand on the map">
    {([ ["none", "None"], ["home", "Homes"], ["work", "Destinations"] ] as const).map(([value, label]) =>
      <button type="button" class={`btn${selected === value ? " on" : ""}${!ready && value !== "none" ? " demand-pending" : ""}`} aria-pressed={selected === value}
        aria-disabled={!ready && value !== "none"} data-tip={!ready && value !== "none" ? pending : undefined}
        onClick={() => {
          if (!ready && value !== "none") return;
          const current = display.peek();
          display.value = value === "none" ? { ...current, commuters: false } : { ...current, commuters: true, commuterEnd: value };
        }}>{label}</button>)}
    <PurposeSelect />
  </div>;
}

const TRACK_OPTIONS = [
  ["line", "Line", "Lines in their own colours"],
  ["traffic", "Traffic", "Line colours, thickness by traffic"],
  ["height", "Elevation", "Track coloured by its height"],
  ["speed", "Max speed", "Track coloured by its speed limit"],
] as const;

/** The track display (T-703): the four ways the network is coloured, switched often, so on the map
 * under the demand buttons; the key of the height and speed colours sits under it. */
function TrackDisplay() {
  const mode = display.value.trackColour;
  return <div class="track-row">
    <div class="demand-selector" role="group" aria-label="Track display">
      {TRACK_OPTIONS.map(([value, label, tip]) =>
        <button type="button" class={`btn${mode === value ? " on" : ""}`} aria-pressed={mode === value} data-tip={tip}
          onClick={() => setDisplay("trackColour", value)}>{label}</button>)}
    </div>
    {mode === "speed" && <ul class="track-key speed-key">
      {SPEED_STOPS.map(([v, colour]) => <li><i style={{ background: colour }} />{speed(v, 5)}</li>)}
    </ul>}
    {mode === "height" && <ul class="track-key levels-key">
      {LEVELS.map((l) => <li><i style={{ background: LEVEL_COLOURS[l] }} />{levelLabel(l)}</li>)}
    </ul>}
  </div>;
}

/** The purpose filter (T-158): every demand drawing on the map narrowed to one trip purpose. */
function PurposeSelect() {
  const purpose = asFilter(purposeFilter.value);
  // T-169: Chrome counts a mouse-focused select as :focus-visible, so the ring would stay after a
  // pick. Like the bar's buttons, it shows for keyboard focus only (style.css).
  const [mouse, setMouse] = useState(false);
  return <label class={`purpose-filter${purpose !== "all" ? " on" : ""}${mouse ? " mouse" : ""}`}>
    {purpose !== "all" && <i class="sw" style={{ background: purposeColour(purpose) }} />}
    <select aria-label="Trip purpose on the map" value={purpose}
      onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
      onChange={(e) => { purposeFilter.value = asFilter((e.currentTarget as HTMLSelectElement).value); }}>
      <option value="all">All purposes</option>
      {FILTER_PURPOSES.map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}
    </select>
  </label>;
}
