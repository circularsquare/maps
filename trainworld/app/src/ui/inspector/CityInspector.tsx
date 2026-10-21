// A city picked on the map zoomed far out (T-186, SPEC 8): its trips within the zone and its longer
// trips to and from other places, each a figure with a mode bar, and the three biggest places
// (T-600, Anita: brief statistics of what the hover has, no table of every place). The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). These numbers are the
// current forecast; the map's bubbles catch up to them over a game day after a change (T-156).

import { cityById, type CityId } from "../../game/cities";
import { cityDemandViews, intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { count } from "../../game/format";
import { zoneLoads } from "../../game/loading";
import { BUBBLE_ALPHA, bubbleColour, setBubbleColours } from "../../game/palette";
import { display, purposeFilter, shownDayPhrase } from "../../game/ui";
import { localModes } from "../../map/modeBar";
import { goTo } from "../Search";

const exact = (n: number) => Math.round(n).toLocaleString("en");
const pct = (n: number, of: number) => {
  const p = of > 0 ? (100 * n) / of : 0;
  return n >= 0.5 && p < 0.5 ? "<1%" : Math.round(p) + "%";
};

type Part = { name: string; n: number; cls: string };

/** A figure with its mode bar: train blue, walking green, car red, as the map's bubbles. */
function TripsBar({ label, tip, what, parts }: { label: string; tip?: string; what: string; parts: Part[] }) {
  const all = parts.reduce((s, p) => s + p.n, 0);
  return (
    <li class="zone-trips">
      <div class="line">
        <span class="grow" data-tip={tip}>{label}</span>
        <span class="val" data-tip={`${exact(all)} ${what} ${shownDayPhrase.value}`}>{count(all)}</span>
      </div>
      <div class="bar split-bar" role="img" aria-label={parts.map((p) => `${p.name} ${pct(p.n, all)}`).join(", ")}>
        {parts.map((p) => p.n > 0 && (
          <i class={"seg " + p.cls} style={{ width: (100 * p.n) / all + "%" }} data-tip={`${p.name}: ${exact(p.n)} ${what} ${shownDayPhrase.value}, ${pct(p.n, all)}`} />
        ))}
      </div>
    </li>
  );
}

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  // T-421: a zone still loading (its bar on the map) has no figures to state yet
  const market = zoneLoads.value.has(id as CityId) ? null : intercityDemand.value;
  if (!city) return null;
  const byPlace = new Map<number, { all: number; rail: number }>();
  const c = market?.corridors;
  for (let i = 0; c && i < c.n; i++) {
    const from = c.from[i], to = c.to[i];
    if (from === to || (from !== city.place && to !== city.place)) continue;
    const other = from === city.place ? to : from;
    const p = byPlace.get(other) ?? { all: 0, rail: 0 };
    p.all += c.trips[i];
    p.rail += c.rail[i];
    byPlace.set(other, p);
  }
  const places = [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
  const interAll = places.reduce((s, [, p]) => s + p.all, 0);
  const interRail = places.reduce((s, [, p]) => s + p.rail, 0);
  // T-378: both ways, as the corridors count them
  const elsewhere = 2 * (market?.elsewhere?.[city.place] ?? 0);
  // within the zone: the open zone's modelled split, or only a count for a zone not opened
  const view = cityDemandViews.value.get(id as CityId);
  const lm = localModes(view);
  const localAll = view ? view.tripsPerDay : city.localTrips ?? 0;
  // the title mark is the zone's bubble colour (map/intercityView.ts): local and inter-region together
  const walk = lm?.walk ?? 0;
  const rail = interRail + (lm?.train ?? 0);
  const d = display.value;
  setBubbleColours(d.bubbleColours, d.rampFrom, d.rampTo);
  const mix = bubbleColour(rail, walk, Math.max(0, interAll + localAll - rail - walk));
  const filter = purposeFilter.value;
  const drawn = filter === "all";
  const top = places.slice(0, 3);
  return (
    <>
      <div class="title">
        <i class="mk" style={{ background: `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${BUBBLE_ALPHA})` }} />
        <span class="name">{city.name}</span>
      </div>
      {!market ? (
        <p class="muted">Calculating demand.</p>
      ) : (
        <>
          <ul class="rows">
            {lm ? (
              <TripsBar
                label={`Within ${city.name}`}
                what={`trips within ${city.name}`}
                parts={[
                  { name: "Train", n: lm.train, cls: "seg-train" },
                  { name: "Walk", n: lm.walk, cls: "seg-walk" },
                  { name: "Car", n: lm.drive, cls: "seg-drive" },
                ]}
              />
            ) : localAll > 0 ? (
              <li class="zone-trips">
                <div class="line">
                  <span class="grow">Within {city.name}</span>
                  <span class="val" data-tip={`${exact(localAll)} trips ${shownDayPhrase.value} within ${city.name}`}>{count(localAll)}</span>
                </div>
              </li>
            ) : null}
            <TripsBar
              label="To other places"
              tip="Visits, shopping and business, not commutes. Both directions are counted."
              what={`inter-region trips to and from ${city.name}`}
              parts={[
                { name: "Train", n: interRail, cls: "seg-train" },
                { name: "Car or plane", n: Math.max(0, interAll - interRail), cls: "seg-drive" },
              ]}
            />
          </ul>
          {top.length > 0 && (
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
          {!places.length && elsewhere <= 0 && <p class="muted small">No inter-region trips are modelled here yet.</p>}
          {!drawn && <p class="muted small">Not drawn under this purpose filter.</p>}
        </>
      )}
      {/* T-220: the City tab's City centre button, here now that the tab is gone */}
      <div class="pane-foot">
        <button class="btn text" onClick={() => goTo({ kind: "city", name: city.name, city })}>Go to {city.name}</button>
      </div>
    </>
  );
}
