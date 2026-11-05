// A city picked on the map zoomed far out (T-186, SPEC 8): its trips within the zone and its longer
// trips to and from other places, each a figure with a mode bar, and the three biggest places
// (T-600, Anita: brief statistics of what the hover has, no table of every place). The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). A click on the city in the
// Cities tab opens it too, the city view (T-1438), with CityDetails.tsx's rows around these.

import { cityById, type CityId } from "../../game/cities";
import { cityDemandViews, eachCorridor, intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { count, exact } from "../../game/format";
import { zoneLoads } from "../../game/loading";
import { BUBBLE_ALPHA, bubbleColour, setBubbleColours, setBubbleShade } from "../../game/palette";
import { display, purposeFilter, shade, shownDayPhrase } from "../../game/ui";
import { localModes } from "../../map/modeBar";
import { goTo } from "../Search";
import { Icon, SegBar } from "../widgets";
import { pctOf } from "./tripText";
import type { City } from "../../game/cities";
import { builtCities } from "../../game/worldTotals";
import { GoalTicks } from "../GoalTicks";
import { CityFigures, CityNetwork } from "./CityDetails";
import { HomeButton } from "./HomeButton";
import { TripGraph } from "./TripGraph";

/** `note`: a second line for the part's hover */
type Part = { name: string; n: number; cls: string; note?: string };
/** T-2215: why a trip has no route (the trips between two land parts no road joins, not by rail) */
const NO_ROAD = "No road joins their two ends";

/** A figure with its mode bar: train blue, walking green, car red, No route grey, as the map's bubbles. */
function TripsBar({ label, tip, what, parts, goals }: { label: string; tip?: string; what: string; parts: Part[]; goals?: City }) {
  const all = parts.reduce((s, p) => s + p.n, 0);
  const bar = <SegBar kind="split-bar" label={parts.map((p) => `${p.name} ${pctOf(p.n, all)}`).join(", ")}
    parts={parts.map((p) => ({ share: p.n / all, cls: p.cls, tip: `${p.name}: ${exact(p.n)} ${what} ${shownDayPhrase.value}, ${pctOf(p.n, all)}${p.note ? `\n${p.note}` : ""}` }))} />;
  return (
    <li class={"zone-trips" + (goals ? " with-goals" : "")}>
      <div class="line">
        <span class="grow" data-tip={tip}>{label}</span>
        <span class="val" data-tip={`${exact(all)} ${what} ${shownDayPhrase.value}`}>{count(all)}</span>
      </div>
      {/* T-1437: the goals over the city's own split (T-1551: only here, not on the top bar) */}
      {goals ? <div class="goal-bar">{bar}<GoalTicks city={goals} /></div> : bar}
    </li>
  );
}

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  // T-421: a zone still loading (its bar on the map) has no figures to state yet
  const market = zoneLoads.value.has(id as CityId) ? null : intercityDemand.value;
  if (!city) return null;
  const byPlace = new Map<number, { all: number; rail: number; air: number }>();
  const c = market?.corridors;
  if (c) eachCorridor(c, (from, to) => from !== to && (from === city.place || to === city.place), (j, v) => {
    const other = c.table.from[j] === city.place ? c.table.to[j] : c.table.from[j];
    const p = byPlace.get(other) ?? { all: 0, rail: 0, air: 0 };
    p.all += v.trips;
    p.rail += v.rail;
    p.air += v.air;
    byPlace.set(other, p);
  });
  const places = [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
  const interAll = places.reduce((s, [, p]) => s + p.all, 0);
  const interRail = places.reduce((s, [, p]) => s + p.rail, 0);
  const interAir = places.reduce((s, [, p]) => s + p.air, 0);
  // T-378: both ways, as the corridors count them
  const elsewhere = 2 * (market?.elsewhere?.[city.place] ?? 0);
  // within the zone: the open zone's modelled split, or only a count for a zone not opened
  const view = cityDemandViews.value.get(id as CityId);
  const lm = localModes(view);
  const localAll = view ? view.tripsPerDay : city.localTrips ?? 0;
  // the title mark is the zone's bubble colour (map/intercityView.ts): local and inter-region together
  const walk = lm?.walk ?? 0, none = lm?.none ?? 0;
  const rail = interRail + (lm?.train ?? 0);
  const d = display.value;
  setBubbleShade(shade.value);
  setBubbleColours(d.bubbleColours, d.rampFrom, d.rampTo);
  const mix = bubbleColour(rail, walk, Math.max(0, interAll + localAll - rail - walk - none), none);
  const filter = purposeFilter.value;
  const drawn = filter === "all";
  const top = places.slice(0, 3);
  const built = builtCities().has(city.id);
  return (
    <>
      <div class="title">
        <i class="mk" style={{ background: `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${BUBBLE_ALPHA})` }} />
        <span class="name">{city.name}</span>
        <HomeButton place={city.id} />
        <button class="btn icon" data-tip={`Go to ${city.name}`} aria-label={`Go to ${city.name}`} onClick={() => goTo({ kind: "city", name: city.name, city })}>
          <Icon.locate />
        </button>
      </div>
      <CityFigures city={city} built={built} />
      {!market && !lm ? (
        <p class="muted">Calculating demand</p>
      ) : (
        <>
          <ul class="rows">
            {lm ? (
              <TripsBar
                label={`Trips within ${city.name}`}
                what={`trips within ${city.name}`}
                parts={[
                  { name: "Train", n: lm.train, cls: "seg-train" },
                  { name: "Walk", n: lm.walk, cls: "seg-walk" },
                  { name: "Car", n: lm.drive, cls: "seg-drive" },
                  ...(lm.none > 0 ? [{ name: "No route", n: lm.none, cls: "seg-none", note: NO_ROAD }] : []),
                ]}
                goals={city}
              />
            ) : localAll > 0 ? (
              <li class="zone-trips">
                <div class="line">
                  <span class="grow">Trips within {city.name}</span>
                  <span class="val" data-tip={`${exact(localAll)} trips ${shownDayPhrase.value} within ${city.name}`}>{count(localAll)}</span>
                </div>
              </li>
            ) : null}
            {market && (
              <TripsBar
                label="To other places"
                tip="Visits, shopping and business, not commutes. Both directions are counted."
                what={`inter-city trips to and from ${city.name}`}
                parts={[
                  { name: "Train", n: interRail, cls: "seg-train" },
                  { name: "Car", n: Math.max(0, interAll - interRail - interAir), cls: "seg-drive" },
                  ...(interAir > 0 ? [{ name: "Plane", n: interAir, cls: "seg-other" }] : []),
                ]}
              />
            )}
          </ul>
          {/* T-2500: a zone still loading whose own figures are in (a load's round of its cluster) shows them
              and its graph; its trips to other places wait */}
          {!market && <p class="muted">Calculating demand</p>}
          {lm && <TripGraph city={city} />}
          {market && top.length > 0 && (
            <ul class="rows">
              <li class="muted small">Top destinations</li>
              {top.map(([place, p]) => (
                <li key={place}>
                  <span class="grow">{INTERCITY_PLACES[place]}</span>
                  <span class="val" data-tip={`${exact(p.all)} trips ${shownDayPhrase.value} between ${city.name} and ${INTERCITY_PLACES[place]}`}>{count(p.all)}</span>
                </li>
              ))}
            </ul>
          )}
          {market && !places.length && elsewhere <= 0 && <p class="muted small">No inter-city trips are modeled here yet.</p>}
          {!drawn && <p class="muted small">Not drawn under this purpose filter.</p>}
        </>
      )}
      {built && <CityNetwork city={city} />}
    </>
  );
}
