import { activeCity, cityById, viewedZone } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { zoomedOut } from "../map/intercityView";
import { levelLabel, speed } from "../game/format";
import { loadProgress } from "../game/loading";
import { LEVEL_COLOURS, SPEED_STOPS } from "../game/palette";
import { LEVELS } from "../game/types";
import { asFilter, FILTER_PURPOSES, purposeColour } from "../game/purposes";
import { useState } from "preact/hooks";
import { display, purposeFilter, setDisplay, type MapRow } from "../game/ui";
import type { JSX } from "preact";
import { DemandStopped } from "./DemandStopped";
import { StatusTag } from "./StatusTag";
import { BusyTag } from "./BusyTag";
import { Icon } from "./widgets";

/** The map's upper left: the demand buttons with the status tag to their right (T-349), and under
 * them the station ridership (T-1448), the track display and the demand alert. */
export function MapTopLeft() {
  return <div class="map-top-left">
    <div class="map-top-row">
      <DemandSelector />
      <StatusTag />
      <BusyTag />
    </div>
    <StationRidership />
    <TrackDisplay />
    <DemandStopped />
  </div>;
}

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  // T-801: zoomed out the zones' bubbles need no zone in view
  const ready = zoomedOut.value || (!!viewedZone.value && cityDemandViews.value.has(activeCity.value));
  // T-149: the viewed city may still be loading after a pan
  const load = loadProgress.value;
  const pending = !viewedZone.value ? "Nothing modelled here" // T-318: dead space
    : load?.cities.includes(activeCity.value) ? `Loading ${cityById(activeCity.value).name}, ${Math.floor(load.fraction * 100)}%` : "Calculating demand";
  const selected = d.commuters ? d.commuterEnd : "none";
  if (folded("bubbles")) return <div class="demand-selector" role="group" aria-label="Demand on the map"><FoldCell row="bubbles" /></div>;
  return <div class="demand-selector" role="group" aria-label="Demand on the map">
    <FoldCell row="bubbles" />
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
  ["speed", "Speed", "Track coloured by the speed its trains reach, or by its speed limit where none run"],
] as const;

/** The row's icon, a button folding the row down to it and opening it again (T-1448). */
const FOLDS: Record<MapRow, { icon: () => JSX.Element; what: string }> = {
  bubbles: { icon: Icon.bubble, what: "demand bubble" },
  stations: { icon: Icon.interchange, what: "station ridership" },
  track: { icon: Icon.track, what: "track colour" },
};
const folded = (row: MapRow) => display.value.foldedRows.includes(row);
function FoldCell({ row }: { row: MapRow }) {
  const shut = folded(row), f = FOLDS[row];
  return <button type="button" class="hint-cell" aria-expanded={!shut} data-tip={`${shut ? "Show" : "Hide"} ${f.what} choices`}
    onClick={() => {
      const rows = display.peek().foldedRows;
      setDisplay("foldedRows", shut ? rows.filter((r) => r !== row) : [...rows, row]);
    }}><f.icon /></button>;
}

const STATION_OPTIONS = [
  ["none", "None", undefined],
  ["origins", "Origins", "Riders starting a trip at each station"],
  ["transfers", "Transfers", "Riders changing lines at each station"],
  ["total", "Total", "Riders starting a trip (blue) and changing lines (green) at each station"],
] as const;

/** Station ridership (T-482, on the map since T-1448): circles by the riders a day boarding at each
 * station complex, starting a trip there, changing lines there, or both. */
function StationRidership() {
  const mode = display.value.stationUse;
  return <div class="demand-selector" role="group" aria-label="Station ridership">
    <FoldCell row="stations" />
    {!folded("stations") && STATION_OPTIONS.map(([value, label, tip]) =>
      <button type="button" class={`btn${mode === value ? " on" : ""}`} aria-pressed={mode === value} data-tip={tip}
        onClick={() => setDisplay("stationUse", value)}>{label}</button>)}
  </div>;
}

/** The track display (T-703): the four ways the network is coloured, switched often, so on the map
 * under the demand buttons; the key of the height and speed colours sits under it. */
function TrackDisplay() {
  const mode = display.value.trackColour;
  if (folded("track")) return <div class="demand-selector" role="group" aria-label="Track display"><FoldCell row="track" /></div>;
  return <div class="track-row">
    <div class="demand-selector" role="group" aria-label="Track display">
      <FoldCell row="track" />
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
