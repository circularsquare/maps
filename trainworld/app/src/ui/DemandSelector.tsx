import { activeCity } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { display } from "../game/ui";

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  const ready = cityDemandViews.value.has(activeCity.value);
  const selected = d.commuters ? d.commuterEnd : "none";
  return <div class="demand-selector" role="group" aria-label="Demand on the map">
    {([ ["none", "None"], ["home", "Homes"], ["work", "Destinations"] ] as const).map(([value, label]) =>
      <button type="button" class={`btn${selected === value ? " on" : ""}${!ready && value !== "none" ? " demand-pending" : ""}`} aria-pressed={selected === value}
        aria-disabled={!ready && value !== "none"} title={!ready && value !== "none" ? "Calculating demand" : undefined}
        onClick={() => {
          if (!ready && value !== "none") return;
          const current = display.peek();
          display.value = value === "none" ? { ...current, commuters: false } : { ...current, commuters: true, commuterEnd: value };
        }}>{label}</button>)}
  </div>;
}
