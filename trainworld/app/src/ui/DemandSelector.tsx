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
import { diagramShown } from "../diagram/view";
import { DiagramTopLeft } from "./DiagramButton";
import { live, mapBearing } from "../map/live";
import { DEMAND_KEY } from "./toolKeys";
import { t, tNumber, type MessageKey } from "../i18n";

/** The map's upper left: the demand buttons with the status tag to their right (T-349), and under
 * them the track display, the station ridership (T-1448, T-1477), the find button (T-1868; with the
 * "Saved only in this browser" dot beside it), the demand alert, what the last edit did (T-2125)
 * and a passing note (T-2166). */
export function MapTopLeft() {
  // T-2161: the map's rows say nothing about the network diagram; its own take their place
  if (diagramShown.value) return <DiagramTopLeft />;
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
  return <button type="button" class="btn icon map-north" aria-label={t("demandSelector.north")} data-tip={t("demandSelector.north")}
    onClick={() => live.map?.easeTo({ bearing: 0, duration: 300 })}>
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" style={{ transform: `rotate(${-b}deg)` }}>
      <path d="M7 0.5 L9.3 7 L4.7 7 Z" fill="var(--accent)" />
      <path d="M7 13.5 L9.3 7 L4.7 7 Z" fill="currentColor" opacity="0.4" />
    </svg>
  </button>;
}

const KEY = DEMAND_KEY.toUpperCase();
const KEYED: Record<string, MessageKey> = { none: "demandSelector.tipNone", home: "demandSelector.tipHomes", work: "demandSelector.tipDestinations", both: "demandSelector.tipTotal" };

/** T-801: zoomed out the zones' bubbles need no zone in view */
const demandReady = () => zoomedOut.value || (!!viewedZone.value && cityDemandViews.value.has(activeCity.value));

const showEnd = (value: "none" | "home" | "work" | "both") => {
  const current = display.peek();
  display.value = value === "none" ? { ...current, commuters: false } : { ...current, commuters: true, commuterEnd: value };
};

const CYCLE = ["none", "home", "work", "both"] as const;

/** T-2338: the D key, None to Homes to Destinations to Total and round again. */
export function cycleDemandEnd() {
  if (!demandReady()) return;
  const d = display.peek();
  showEnd(CYCLE[(CYCLE.indexOf(d.commuters ? d.commuterEnd : "none") + 1) % CYCLE.length]);
}

/** The map's demand view stays within reach whichever dock tab is open. */
export function DemandSelector() {
  const d = display.value;
  const ready = demandReady();
  // T-149: the viewed city may still be loading after a pan
  const load = loadProgress.value;
  const pending = !viewedZone.value ? t("demandSelector.nothing") // T-318: dead space
    : load?.cities.includes(activeCity.value) ? t("demandSelector.loading", { city: cityById(activeCity.value).name, percent: tNumber(Math.floor(load.fraction * 100)) }) : t("demandSelector.calculating");
  const selected = d.commuters ? d.commuterEnd : "none";
  if (folded("bubbles")) return <div class="demand-selector" role="group" aria-label={t("demandSelector.group")}><FoldCell row="bubbles" /></div>;
  return <div class="demand-selector" role="group" aria-label={t("demandSelector.group")}>
    <FoldCell row="bubbles" />
    {([ ["none", "demandSelector.none"], ["home", "demandSelector.homes"], ["work", "demandSelector.destinations"], ["both", "demandSelector.total"] ] as const).map(([value, label]) =>
      <button type="button" class={`btn${selected === value ? " on" : ""}${!ready && value !== "none" ? " demand-pending" : ""}`} aria-pressed={selected === value} data-guide={value === "home" ? "homes" : undefined}
        aria-disabled={!ready && value !== "none"} data-tip={!ready && value !== "none" ? pending : t(KEYED[value], { key: KEY })}
        onClick={() => {
          if (!ready && value !== "none") return;
          showEnd(value);
        }}>{t(label)}</button>)}
    <PurposeSelect />
  </div>;
}

const TRACK_OPTIONS = [
  ["line", "demandSelector.trackLine", "demandSelector.trackLineTip"],
  ["traffic", "demandSelector.trackTraffic", "demandSelector.trackTrafficTip"],
  ["height", "demandSelector.trackLevel", "demandSelector.trackLevelTip"],
  ["speed", "demandSelector.trackSpeed", "demandSelector.trackSpeedTip"],
  ["profit", "demandSelector.trackProfit", "demandSelector.trackProfitTip"],
] as const;

/** The row's icon, a button folding the row down to it and opening it again (T-1448). */
const FOLDS: Record<MapRow, { icon: () => JSX.Element; tip: MessageKey }> = {
  bubbles: { icon: Icon.bubble, tip: "demandSelector.foldBubbles" },
  stations: { icon: Icon.stationDisplay, tip: "demandSelector.foldStations" },
  track: { icon: Icon.trackDisplay, tip: "demandSelector.foldTrack" },
};
/** T-2123: the tutorial's first card folds its rows the player has not opened since */
const folded = (row: MapRow) => display.value.foldedRows.includes(row) || (guideFolding.value.includes(row) && !guideOpened.value.includes(row));
function FoldCell({ row }: { row: MapRow }) {
  const shut = folded(row), f = FOLDS[row];
  return <button type="button" class="hint-cell" data-fold={row} aria-expanded={!shut} data-tip={t(f.tip, { shut })}
    onClick={() => {
      const rows = display.peek().foldedRows;
      if (shut && guideFolding.peek().includes(row) && !guideOpened.peek().includes(row)) guideOpened.value = [...guideOpened.peek(), row];
      setDisplay("foldedRows", shut ? rows.filter((r) => r !== row) : [...rows, row]);
    }}><f.icon /></button>;
}

const STATION_OPTIONS = [
  ["none", "demandSelector.none", undefined],
  ["origins", "demandSelector.origins", "demandSelector.originsTip"],
  ["transfers", "demandSelector.transfers", "demandSelector.transfersTip"],
  ["total", "demandSelector.total", "demandSelector.totalTip"],
] as const;

/** Station ridership (T-482, on the map since T-1448): circles by the riders a day boarding at each
 * station complex, starting a trip there, changing lines there, or both. */
function StationRidership() {
  const mode = display.value.stationUse;
  return <div class="demand-selector" role="group" aria-label={t("demandSelector.stationGroup")}>
    <FoldCell row="stations" />
    {!folded("stations") && STATION_OPTIONS.map(([value, label, tip]) =>
      <button type="button" class={`btn${mode === value ? " on" : ""}`} aria-pressed={mode === value} data-tip={tip && t(tip)}
        onClick={() => setDisplay("stationUse", value)}>{t(label)}</button>)}
  </div>;
}

/** The track display (T-703): the five ways the network is coloured, switched often, so on the map
 * under the demand buttons. No key for the height, speed and profit colours (T-1499, T-2420): the track's hover says. */
function TrackDisplay() {
  const mode = display.value.trackColour;
  return <div class="demand-selector" role="group" aria-label={t("demandSelector.trackGroup")}>
    <FoldCell row="track" />
    {!folded("track") && TRACK_OPTIONS.map(([value, label, tip]) =>
      <button type="button" class={`btn${mode === value ? " on" : ""}`} aria-pressed={mode === value} data-tip={t(tip)} data-guide={value === "traffic" ? "traffic" : undefined}
        onClick={() => setDisplay("trackColour", value)}>{t(label)}</button>)}
  </div>;
}

/** The purpose filter (T-158): every demand drawing on the map narrowed to one trip purpose. */
export function PurposeSelect() {
  const purpose = asFilter(purposeFilter.value);
  // T-169: Chrome counts a mouse-focused select as :focus-visible, so the ring would stay after a
  // pick. Like the bar's buttons, it shows for keyboard focus only (style.css).
  const [mouse, setMouse] = useState(false);
  return <label class={`purpose-filter${purpose !== "all" ? " on" : ""}${mouse ? " mouse" : ""}`}>
    {purpose !== "all" && <i class="sw" style={{ background: purposeColour(purpose) }} />}
    <select aria-label={t("demandSelector.purpose")} value={purpose}
      onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
      onChange={(e) => { purposeFilter.value = asFilter((e.currentTarget as HTMLSelectElement).value); }}>
      <option value="all">{t("demandSelector.allPurposes")}</option>
      {FILTER_PURPOSES.map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}
    </select>
  </label>;
}
