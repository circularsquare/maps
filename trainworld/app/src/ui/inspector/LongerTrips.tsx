// Longer trips (T-118, SPEC 8): the market's non-commute trips of a bubble selection, by place, all
// modes and by train, car and plane (T-378). The same numbers are drawn on the map as bubbles
// (map/intercityView.ts), one per far zone.

import type { ComponentChildren } from "preact";
import { demandView, intercityPlaceName, type PlaceFlows } from "../../game/demand";
import { count, exact } from "../../game/format";
import { shownDayPhrase } from "../../game/ui";
import { modeColours } from "../../game/palette";
import { intercityPick } from "../../map/intercityView";
import { Spinner } from "../Spinner";
import { Icon } from "../widgets";
import { pctOf } from "./tripText";

/** Read while rendering: the colours follow the bubble scheme (T-833). */
const modes = (walk: boolean) => {
  const c = modeColours();
  return [
    { key: "rail", icon: Icon.train, colour: c.train, by: "by train" },
    // T-1229: walking only where the everyday outings across a zone line add some
    ...(walk ? [{ key: "walk", icon: Icon.walk, colour: c.walk, by: "on foot" } as const] : []),
    { key: "car", icon: Icon.ride, colour: c.drive, by: "by car" },
    { key: "air", icon: Icon.plane, colour: c.other, by: "by plane" },
  ] as const;
};
type Split = { all: number; rail: number; car: number; air: number; walk?: number };

/** The column heads: all modes, then each mode's icon with its hover. */
function ModeHead({ what, walk }: { what: string; walk: boolean }) {
  return (
    <li class="muted small">
      <span class="grow" />
      <span class="val w-num" data-tip={`${what} ${shownDayPhrase.value}, every mode`}>All</span>
      {modes(walk).map((m) => (
        <span class="val w-mode" data-tip={`${what} ${m.by}`}>
          <span class="mode-icon" style={{ color: m.colour }}>{m.icon()}</span>
        </span>
      ))}
    </li>
  );
}

/** A row's numbers: all modes, then by train, car and plane, each with its hover. */
function ModeCells({ s, what, walk }: { s: Split; what: string; walk: boolean }) {
  return (
    <>
      <span class="val w-num" data-tip={`${exact(s.all)} ${what} ${shownDayPhrase.value}`}>{count(s.all)}</span>
      {modes(walk).map((m) => {
        const n = s[m.key] ?? 0;
        return <span class="val w-mode" data-tip={`${exact(n)} ${m.by}, ${pctOf(n, s.all)} of ${exact(s.all)}`}>{count(n)}</span>;
      })}
    </>
  );
}

/** Trips with places that have no zone: all modes alone (their modes are not modelled). */
function ElsewhereRow({ n, walk, label = "Elsewhere" }: { n: number; walk: boolean; label?: ComponentChildren }) {
  if (!(n >= 0.5)) return null;
  return (
    <li>
      <span class="grow" data-tip="Places outside every zone of the game; how these trips travel is not modelled">{label}</span>
      <span class="val w-num" data-tip={`${exact(n)} trips ${shownDayPhrase.value}`}>{count(n)}</span>
      {modes(walk).map(() => <span class="val w-mode" />)}
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

/** Places a direction lists; the rest share one row. A selection in China has trips with about
 * 1,700 places, and their rows took the panel half a second to draw (T-1165). */
const PLACE_ROWS = 15;

/** The first `PLACE_ROWS` places, and the rest summed (null when there is no rest). */
function topPlaces<P extends Split>(places: P[]): { top: P[]; rest: (Split & { n: number }) | null } {
  if (places.length <= PLACE_ROWS + 1) return { top: places, rest: null };
  const rest = { n: places.length - PLACE_ROWS, all: 0, rail: 0, car: 0, air: 0, walk: 0 };
  for (let i = PLACE_ROWS; i < places.length; i++) {
    const p = places[i];
    rest.all += p.all; rest.rail += p.rail; rest.car += p.car; rest.air += p.air; rest.walk += p.walk ?? 0;
  }
  return { top: places.slice(0, PLACE_ROWS), rest };
}

function Direction({ title, verb, flows, own, walk = false }: { title: string; verb: string; flows: PlaceFlows; own: number; walk?: boolean }) {
  const total = { ...flows, all: flows.all + flows.elsewhere };
  const { top, rest } = topPlaces(flows.places);
  const rows: [number, ComponentChildren][] = top.map((p) => [p.all, (
    <li>
      <span class="grow" data-tip={`${verb} ${intercityPlaceName(p.place, own)}`}>{verb} {intercityPlaceName(p.place, own)}</span>
      <ModeCells s={p} what="trips" walk={walk} />
    </li>
  )]);
  if (rest) rows.push([0, (
    <li>
      <span class="grow" data-tip={`${exact(rest.n)} more places`}>{verb} other places</span>
      <ModeCells s={rest} what="trips" walk={walk} />
    </li>
  )]);
  return (
    <>
      <ul class="rows">
        <li class="b">
          <span class="grow">{title}</span>
          <ModeCells s={total} what="trips" walk={walk} />
        </li>
      </ul>
      <ul class="rows sub-rows">
        {bySize(rows, flows.elsewhere, <ElsewhereRow n={flows.elsewhere} walk={walk} />)}
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
  const walk = (p.data.starts.walk ?? 0) + (p.data.ends.walk ?? 0) > 0;
  const out = <Direction title="Leave from here" verb="To" flows={p.data.starts} own={p.own} walk={walk} />;
  const back = <Direction title="Arrive here" verb="From" flows={p.data.ends} own={p.own} walk={walk} />;
  return (
    <>
      <Head tip={`Visits, shopping and business, for the areas the selection lies in, with the errands, visits, school and medical trips that cross a zone line nearby. Commutes are left out. The map's bubbles are their ${home ? "destinations" : "origins"} outside this metropolitan area, on a separate scale from local trips.`} />
      <ul class="rows">
        <ModeHead what="Trips" walk={walk} />
      </ul>
      {home ? <>{out}{back}</> : <>{back}{out}</>}
    </>
  );
}
