import { activeCity, cityById, viewedZone } from "../game/cities";
import { cityDemandViews } from "../game/demand";
import { zoomedOut } from "../map/intercityView";
import { loadProgress } from "../game/loading";
import { asFilter, FILTER_PURPOSES, purposeColour } from "../game/purposes";
import { useState } from "preact/hooks";
import { display, purposeFilter, setDisplay, type MapRow } from "../game/ui";
import { guideFolding, guideOpened } from "../game/guide";
import type { JSX } from "preact";
import { DemandStopped } from "./DemandStopped";
import { StatusTag } from "./StatusTag";
import { BusyTag } from "./BusyTag";
import { Icon } from "./widgets";
import { Search } from "./Search";
import { KeepDot } from "./KeepDot";
import { EditNote } from "./EditNote";
import { PassingNote } from "./PassingNote";
import { live, mapBearing } from "../map/live";

/** The map's upper left: the demand buttons with the status tag to their right (T-349), and under
 * them the track display, the station ridership (T-1448, T-1477), the find button (T-1868; with the
 * "Saved only in this browser" dot beside it), the demand alert, what the last edit did (T-2125)
 * and a passing note (T-2166). */
export function MapTopLeft() {
  return <div class="map-top-left">
    <div class="map-top-row">
      <DemandSelector />
      <StatusTag />
      <BusyTag />
    </div>
    <TrackDisplay />
    <StationRidership />
    <div class="map-find">
      <Search />
      <NorthButton />
      <KeepDot />
    </div>
    <DemandStopped />
    <EditNote />
    <PassingNote />
  </div>;
}

/** T-2031: while the map is turned, a needle pointing north; a click turns the map back. */
function NorthButton() {
  const b = mapBearing.value;
  if (Math.abs(b) < 0.5) return null;
  return <button type="button" class="btn icon map-north" aria-label="Turn map back to north" data-tip="Turn map back to north"
    onClick={() => live.map?.easeTo({ bearing: 0, duration: 300 })}>
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" style={{ transform: `rotate(${-b}deg)` }}>
      <path d="M7 0.5 L9.3 7 L4.7 7 Z" fill="var(--accent)" />
      <path d="M7 13.5 L9.3 7 L4.7 7 Z" fill="currentColor" opacity="0.4" />
    </svg>
  </button>;
}

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  // T-801: zoomed out the zones' bubbles need no zone in view
  const ready = zoomedOut.value || (!!viewedZone.value && cityDemandViews.value.has(activeCity.value));
  // T-149: the viewed city may still be loading after a pan
  const load = loadProgress.value;
  const pending = !viewedZone.value ? "Nothing modeled here" // T-318: dead space
    : load?.cities.includes(activeCity.value) ? `Loading ${cityById(activeCity.value).name}, ${Math.floor(load.fraction * 100)}%` : "Calculating demand";
  const selected = d.commuters ? d.commuterEnd : "none";
  if (folded("bubbles")) return <div class="demand-selector" role="group" aria-label="Demand on the map"><FoldCell row="bubbles" /></div>;
  return <div class="demand-selector" role="group" aria-label="Demand on the map">
    <FoldCell row="bubbles" />
    {([ ["none", "None"], ["home", "Homes"], ["work", "Destinations"], ["both", "Total"] ] as const).map(([value, label]) =>
      <button type="button" class={`btn${selected === value ? " on" : ""}${!ready && value !== "none" ? " demand-pending" : ""}`} aria-pressed={selected === value} data-guide={value === "home" ? "homes" : undefined}
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
  ["line", "Line", "Show line colors"],
  ["traffic", "Traffic", "Show line colors, thickness by traffic"],
  ["height", "Elevation", "Show track elevation colors"],
  ["speed", "Speed", "Show track speed"],
] as const;

/** The row's icon, a button folding the row down to it and opening it again (T-1448). */
const FOLDS: Record<MapRow, { icon: () => JSX.Element; what: string }> = {
  bubbles: { icon: Icon.bubble, what: "demand bubble" },
  stations: { icon: Icon.stationDisplay, what: "station ridership" },
  track: { icon: Icon.trackDisplay, what: "track color" },
};
/** T-2123: the tutorial's first card folds every row the player has not opened since */
const folded = (row: MapRow) => display.value.foldedRows.includes(row) || (guideFolding.value && !guideOpened.value.includes(row));
function FoldCell({ row }: { row: MapRow }) {
  const shut = folded(row), f = FOLDS[row];
  return <button type="button" class="hint-cell" data-fold={row} aria-expanded={!shut} data-tip={`${shut ? "Show" : "Hide"} ${f.what} choices`}
    onClick={() => {
      const rows = display.peek().foldedRows;
      if (shut && guideFolding.peek() && !guideOpened.peek().includes(row)) guideOpened.value = [...guideOpened.peek(), row];
      setDisplay("foldedRows", shut ? rows.filter((r) => r !== row) : [...rows, row]);
    }}><f.icon /></button>;
}

const STATION_OPTIONS = [
  ["none", "None", undefined],
  ["origins", "Origins", "Show riders starting a trip at each station"],
  ["transfers", "Transfers", "Show transfers at each station"],
  ["total", "Total", "Show riders at each station"],
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
 * under the demand buttons. No key for the height and speed colours (T-1499): the track's hover says. */
function TrackDisplay() {
  const mode = display.value.trackColour;
  return <div class="demand-selector" role="group" aria-label="Track display">
    <FoldCell row="track" />
    {!folded("track") && TRACK_OPTIONS.map(([value, label, tip]) =>
      <button type="button" class={`btn${mode === value ? " on" : ""}`} aria-pressed={mode === value} data-tip={tip}
        onClick={() => setDisplay("trackColour", value)}>{label}</button>)}
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
