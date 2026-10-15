// A city picked on the map zoomed far out (T-186, SPEC 8): its longer trips to and from every
// other place a day, all modes and by train, from the regional market's corridors. The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). These numbers are the
// current forecast; the map's bubbles catch up to them over a game day after a change (T-156).

import { cityById } from "../../game/cities";
import { intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { count } from "../../game/format";
import { BUBBLE_ALPHA, MODE_COLOURS, modeMix } from "../../game/palette";
import { purposeFilter } from "../../game/ui";
import { goTo } from "../Search";

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  const market = intercityDemand.value;
  if (!city) return null;
  const byPlace = new Map<number, { all: number; rail: number }>();
  for (const c of market?.corridors ?? []) {
    if (c.from === c.to || (c.from !== city.place && c.to !== city.place)) continue;
    const other = c.from === city.place ? c.to : c.from;
    const p = byPlace.get(other) ?? { all: 0, rail: 0 };
    p.all += c.trips;
    p.rail += c.rail;
    byPlace.set(other, p);
  }
  const places = [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
  const all = places.reduce((s, [, p]) => s + p.all, 0), rail = places.reduce((s, [, p]) => s + p.rail, 0);
  const mix = modeMix(rail, 0, Math.max(0, all - rail));
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
            <li class="muted small">
              <span class="grow" />
              <span class="val w-num">All</span>
              <span class="val w-num"><i class="mk" style={{ background: MODE_COLOURS.train }} /> Train</span>
            </li>
            <li class="b">
              <span class="grow">Trips a day</span>
              <span class="val w-num">{count(all)}</span>
              <span class="val w-num">{count(rail)}</span>
            </li>
          </ul>
          <ul class="rows sub-rows">
            {places.map(([place, p]) => (
              <li key={place}>
                <span class="grow">{INTERCITY_PLACES[place]}</span>
                <span class="val w-num">{count(p.all)}</span>
                <span class="val w-num">{count(p.rail)}</span>
              </li>
            ))}
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
