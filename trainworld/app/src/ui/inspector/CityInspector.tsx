// A city picked on the map zoomed far out (T-186, SPEC 8): its longer trips to and from every
// other place a day, all modes and by train, car and plane (T-378), from the regional market's
// corridors, and those with places that have no zone. The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). These numbers are the
// current forecast; the map's bubbles catch up to them over a game day after a change (T-156).

import { cityById, type CityId } from "../../game/cities";
import { intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { zoneLoads } from "../../game/loading";
import { BUBBLE_ALPHA, modeMix } from "../../game/palette";
import { purposeFilter, shownDayPhrase } from "../../game/ui";
import { goTo } from "../Search";
import { bySize, ElsewhereRow, ModeCells, ModeHead } from "./LongerTrips";

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  // T-421: a zone still loading (its bar on the map) has no figures to state yet
  const market = zoneLoads.value.has(id as CityId) ? null : intercityDemand.value;
  if (!city) return null;
  const byPlace = new Map<number, { all: number; rail: number; car: number; air: number }>();
  for (const c of market?.corridors ?? []) {
    if (c.from === c.to || (c.from !== city.place && c.to !== city.place)) continue;
    const other = c.from === city.place ? c.to : c.from;
    const p = byPlace.get(other) ?? { all: 0, rail: 0, car: 0, air: 0 };
    p.all += c.trips;
    p.rail += c.rail;
    p.car += c.car ?? c.trips - c.rail;
    p.air += c.air ?? 0;
    byPlace.set(other, p);
  }
  const places = [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
  const sum = (k: "all" | "rail" | "car" | "air") => places.reduce((s, [, p]) => s + p[k], 0);
  // T-378: both ways, as the corridors count them
  const elsewhere = 2 * (market?.elsewhere?.[city.place] ?? 0);
  const total = { all: sum("all") + elsewhere, rail: sum("rail"), car: sum("car"), air: sum("air") };
  const mix = modeMix(total.rail, 0, Math.max(0, sum("all") - total.rail));
  const filter = purposeFilter.value;
  const drawn = filter === "all";
  return (
    <>
      <div class="title">
        <i class="mk" style={{ background: `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${BUBBLE_ALPHA})` }} />
        <span class="name">{city.name}</span>
      </div>
      <div class="sub muted" data-tip="Visits, shopping and business, not commutes. Both directions are counted.">Inter-region trips to and from other places</div>
      {!market ? (
        <p class="muted">Calculating demand.</p>
      ) : (
        <>
          <ul class="rows">
            <ModeHead what="Trips" />
            <li class="b">
              <span class="grow">Trips {shownDayPhrase.value}</span>
              <ModeCells s={total} what="trips" />
            </li>
          </ul>
          <ul class="rows sub-rows">
            {bySize(places.map(([place, p]) => [p.all, (
              <li key={place}>
                <span class="grow">{INTERCITY_PLACES[place]}</span>
                <ModeCells s={p} what="trips" />
              </li>
            )]), elsewhere, <ElsewhereRow n={elsewhere} />)}
          </ul>
          {!places.length && <p class="muted small">No inter-region trips are modelled here yet.</p>}
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
