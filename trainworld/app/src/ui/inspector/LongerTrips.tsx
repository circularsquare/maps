// Longer trips (T-118, SPEC 8): the market's non-commute trips of a bubble selection or of a
// selected station, by place, all modes and by train. The same numbers are drawn on the map as
// bubbles (map/intercityView.ts), one per far zone; the swatch is the map's train blue.

import { demandView, intercityPlaceName, type PlaceFlows } from "../../game/demand";
import { count } from "../../game/format";
import { MODE_COLOURS } from "../../game/palette";
import { display } from "../../game/ui";
import { intercityPick, intercityStationView } from "../../map/intercityView";
import { Row } from "../widgets";

/** The train colour of the map's bubbles. */
const dot = () => <i class="mk" style={{ background: MODE_COLOURS.train }} />;

function Direction({ title, verb, flows, own, railOnly }: { title: string; verb: string; flows: PlaceFlows; own: number; railOnly: boolean }) {
  return (
    <>
      <ul class="rows">
        <li class="b">
          <span class="grow">{title}</span>
          {!railOnly && <span class="val w-num">{count(flows.all)}</span>}
          <span class="val w-num">{count(flows.rail)}</span>
        </li>
      </ul>
      <ul class="rows sub-rows">
        {flows.places.map((p) => (
          <li>
            <span class="grow">{verb} {intercityPlaceName(p.place, own)}</span>
            {!railOnly && <span class="val w-num">{count(p.all)}</span>}
            <span class="val w-num">{count(p.rail)}</span>
          </li>
        ))}
      </ul>
    </>
  );
}

function Head() {
  return (
    <div class="h">
      Longer trips
      <span class="right muted small">a day</span>
    </div>
  );
}

/** For the commuters inspector: the selected areas' longer trips. */
export function SelectionLongerTrips() {
  const p = intercityPick.value;
  if (!demandView.value) return null;
  if (!p) {
    return (
      <>
        <Head />
        <p class="muted">Working it out.</p>
      </>
    );
  }
  const home = p.sel?.kind === "commuters" && p.sel.end === "home";
  const out = <Direction title="Leave from here" verb="To" flows={p.data.starts} own={p.own} railOnly={false} />;
  const back = <Direction title="Arrive here" verb="From" flows={p.data.ends} own={p.own} railOnly={false} />;
  return (
    <>
      <Head />
      <p class="muted small">Visits, shopping and business, not commutes. Counted for the areas the selection lies in.</p>
      <ul class="rows">
        <li class="muted small">
          <span class="grow" />
          <span class="val w-num">All</span>
          <span class="val w-num">{dot()} Train</span>
        </li>
      </ul>
      {home ? <>{out}{back}</> : <>{back}{out}</>}
      {display.value.intercity && <p class="muted small">Bubbles on the map: {home ? "destinations" : "origins"} outside this metropolitan area. Longer trips within it are listed here. Bubble sizes use a separate scale from local trips.</p>}
    </>
  );
}

/** For the station inspector: the longer-trip riders using this station. */
export function StationLongerTrips({ id }: { id: string }) {
  const v = intercityStationView.value;
  if (!demandView.value || !v || v.stationId !== id) return null;
  if (v.starts.rail <= 0 && v.ends.rail <= 0) {
    return (
      <>
        <Head />
        <ul class="rows">
          <Row label="Riders a day">0</Row>
        </ul>
      </>
    );
  }
  return (
    <>
      <Head />
      <Direction title="Start here" verb="To" flows={v.starts} own={-1} railOnly />
      <Direction title="End here" verb="From" flows={v.ends} own={-1} railOnly />
    </>
  );
}

