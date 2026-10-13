import { activeCity, cityById } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { loadProgress } from "../game/loading";
import { PURPOSES, purposeColour, type PurposeFilter } from "../game/purposes";
import { useState } from "preact/hooks";
import { display, purposeFilter } from "../game/ui";
import { StationsFull } from "./StationsFull";

/** The map's upper left: the demand buttons, and under them the stations full alert (T-231). */
export function MapTopLeft() {
  return <div class="map-top-left">
    <DemandSelector />
    <StationsFull />
  </div>;
}

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  const ready = cityDemandViews.value.has(activeCity.value);
  // T-149: the viewed city may still be loading after a pan
  const load = loadProgress.value;
  const pending = load?.cities.includes(activeCity.value) ? `Loading ${cityById(activeCity.value).name}, ${Math.floor(load.fraction * 100)}%` : "Calculating demand";
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

/** The purpose filter (T-158): every demand drawing on the map narrowed to one trip purpose. */
function PurposeSelect() {
  const purpose = purposeFilter.value;
  // T-169: Chrome counts a mouse-focused select as :focus-visible, so the ring would stay after a
  // pick. Like the bar's buttons, it shows for keyboard focus only (style.css).
  const [mouse, setMouse] = useState(false);
  return <label class={`purpose-filter${purpose !== "all" ? " on" : ""}${mouse ? " mouse" : ""}`}>
    {purpose !== "all" && <i class="sw" style={{ background: purposeColour(purpose) }} />}
    <select aria-label="Trip purpose on the map" value={purpose}
      onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
      onChange={(e) => { purposeFilter.value = (e.currentTarget as HTMLSelectElement).value as PurposeFilter; }}>
      <option value="all">All purposes</option>
      {PURPOSES.map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}
    </select>
  </label>;
}
