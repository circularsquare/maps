// Longer trips (T-118, SPEC 8): the market's non-commute trips of a bubble selection or of a
// selected station, by place, all modes and by train, car and plane (T-378). The same numbers are
// drawn on the map as bubbles (map/intercityView.ts), one per far zone.

import type { ComponentChildren } from "preact";
import { demandView, intercityPlaceName, type PlaceFlows } from "../../game/demand";
import { count, exact } from "../../game/format";
import { shownDayPhrase } from "../../game/ui";
import { modeColours } from "../../game/palette";
import { intercityPick, intercityStationView } from "../../map/intercityView";
import { Spinner } from "../Spinner";
import { Icon, Row } from "../widgets";
import { pctOf } from "./tripText";

/** Read while rendering: the colours follow the bubble scheme (T-833). */
const modes = () => {
  const c = modeColours();
  return [
    { key: "rail", icon: Icon.train, colour: c.train, by: "by train" },
    { key: "car", icon: Icon.ride, colour: c.drive, by: "by car" },
    { key: "air", icon: Icon.plane, colour: c.other, by: "by plane" },
  ] as const;
};
type Split = { all: number; rail: number; car: number; air: number };

/** The column heads: all modes, then each mode's icon with its hover. */
function ModeHead({ what }: { what: string }) {
  return (
    <li class="muted small">
      <span class="grow" />
      <span class="val w-num" data-tip={`${what} ${shownDayPhrase.value}, every mode`}>All</span>
      {modes().map((m) => (
        <span class="val w-mode" data-tip={`${what} ${m.by}`}>
          <span class="mode-icon" style={{ color: m.colour }}>{m.icon()}</span>
        </span>
      ))}
    </li>
  );
}

/** A row's numbers: all modes, then by train, car and plane, each with its hover. */
function ModeCells({ s, what }: { s: Split; what: string }) {
  return (
    <>
      <span class="val w-num" data-tip={`${exact(s.all)} ${what} ${shownDayPhrase.value}`}>{count(s.all)}</span>
      {modes().map((m) => (
        <span class="val w-mode" data-tip={`${exact(s[m.key])} ${m.by}, ${pctOf(s[m.key], s.all)} of ${exact(s.all)}`}>{count(s[m.key])}</span>
      ))}
    </>
  );
}

/** Trips with places that have no zone: all modes alone (their modes are not modelled). */
function ElsewhereRow({ n, label = "Elsewhere" }: { n: number; label?: ComponentChildren }) {
  if (!(n >= 0.5)) return null;
  return (
    <li>
      <span class="grow" data-tip="Places outside every zone of the game; how these trips travel is not modelled">{label}</span>
      <span class="val w-num" data-tip={`${exact(n)} trips ${shownDayPhrase.value}`}>{count(n)}</span>
      {modes().map(() => <span class="val w-mode" />)}
    </li>
  );
}

/** Rows biggest first (`rows` already are), with the Elsewhere row at its place among them. */
function bySize(rows: [number, ComponentChildren][], elsewhere: number, row: ComponentChildren): ComponentChildren[] {
  const at = rows.findIndex(([n]) => n < elsewhere);
  const out = rows.map(([, r]) => r);
  out.splice(at < 0 ? out.length : at, 0, row);
  return out;
}

function Direction({ title, verb, flows, own, railOnly }: { title: string; verb: string; flows: PlaceFlows; own: number; railOnly: boolean }) {
  if (railOnly) {
    return (
      <>
        <ul class="rows">
          <li class="b">
            <span class="grow">{title}</span>
            <span class="val w-num">{count(flows.rail)}</span>
          </li>
        </ul>
        <ul class="rows sub-rows">
          {flows.places.map((p) => (
            <li>
              <span class="grow">{verb} {intercityPlaceName(p.place, own)}</span>
              <span class="val w-num">{count(p.rail)}</span>
            </li>
          ))}
        </ul>
      </>
    );
  }
  const total = { ...flows, all: flows.all + flows.elsewhere };
  return (
    <>
      <ul class="rows">
        <li class="b">
          <span class="grow">{title}</span>
          <ModeCells s={total} what="trips" />
        </li>
      </ul>
      <ul class="rows sub-rows">
        {bySize(flows.places.map((p) => [p.all, (
          <li>
            <span class="grow" data-tip={`${verb} ${intercityPlaceName(p.place, own)}`}>{verb} {intercityPlaceName(p.place, own)}</span>
            <ModeCells s={p} what="trips" />
          </li>
        )]), flows.elsewhere, <ElsewhereRow n={flows.elsewhere} />)}
      </ul>
    </>
  );
}

/** T-253: what longer trips are, and what the map draws of them, is the heading's hover */
function Head({ tip }: { tip?: string }) {
  return (
    <div class="h">
      <span data-tip={tip ?? "Visits, shopping and business, not commutes"}>Inter-city trips</span>
      <span class="right muted small">{shownDayPhrase.value}</span>
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
        <p class="muted"><Spinner tip="Working out longer trips" /></p>
      </>
    );
  }
  const home = p.sel?.kind === "commuters" && p.sel.end === "home";
  const out = <Direction title="Leave from here" verb="To" flows={p.data.starts} own={p.own} railOnly={false} />;
  const back = <Direction title="Arrive here" verb="From" flows={p.data.ends} own={p.own} railOnly={false} />;
  return (
    <>
      <Head tip={`Visits, shopping and business, not commutes, for the areas the selection lies in. The map's bubbles are their ${home ? "destinations" : "origins"} outside this metropolitan area, on a separate scale from local trips.`} />
      <ul class="rows">
        <ModeHead what="Trips" />
      </ul>
      {home ? <>{out}{back}</> : <>{back}{out}</>}
    </>
  );
}

/** For the station inspector: the longer-trip riders using this station. */
export function StationLongerTrips({ id, scope = id }: { id: string; scope?: string }) {
  const v = intercityStationView.value;
  if (!demandView.value || !v || v.stationId !== id || v.scope !== scope) return null;
  if (v.starts.rail <= 0 && v.ends.rail <= 0) {
    return (
      <>
        <Head />
        <ul class="rows">
          <Row label={`Riders ${shownDayPhrase.value}`}>0</Row>
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
