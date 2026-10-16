import { activeCity, cityById, viewedZone } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { loadProgress } from "../game/loading";
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
    <DaySelect />
  </div>;
}

const DAY_OPTIONS = [["average", "Average day"], ["weekday", "Weekday"], ["weekend", "Weekend"]] as const;

/** The day the demand views are for (T-464): the average day, a weekday or a weekend day. */
function DaySelect() {
  const day = display.value.dayShown === "weekday" || display.value.dayShown === "weekend" ? display.value.dayShown : "average";
  const [mouse, setMouse] = useState(false);
  return <label class={`purpose-filter${day !== "average" ? " on" : ""}${mouse ? " mouse" : ""}`}
    data-tip="The day the demand figures are for. The average counts 5 weekdays and 2 weekend days over 7. Trains, platforms and riders over the past 24 hours follow the clock.">
    <select aria-label="Day the demand is for" value={day}
      onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
      onChange={(e) => setDisplay("dayShown", (e.currentTarget as HTMLSelectElement).value as typeof day)}>
      {DAY_OPTIONS.map(([v, name]) => <option key={v} value={v}>{name}</option>)}
    </select>
  </label>;
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
