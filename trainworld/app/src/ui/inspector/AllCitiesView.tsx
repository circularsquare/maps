// T-2395: the all cities view, opened from the Cities tab's first row: that row's sums laid out as a
// city view's figures (CityDetails.tsx), with its home and go-to buttons. Totals only (Anita).

import { count, exact, money } from "../../game/format";
import { pointsShort } from "../../game/points";
import { splitTip } from "../../game/splitTip";
import { stationZone } from "../../game/listScope";
import { builtCities } from "../../game/worldTotals";
import { stationsById, world } from "../../game/world";
import { WORLD_HOME } from "../../workers/saveFile";
import { ALL_CITIES, allCitiesTotals, moneyTip, net } from "../Cities";
import { locateAllCities } from "../locate";
import { Icon } from "../widgets";
import { HomeButton } from "./HomeButton";

export function AllCitiesView() {
  const w = world.value;
  if (!w) return null;
  const t = allCitiesTotals();
  const split = t.modes?.split ?? null, f = t.money;
  // the city view's Lines, over every city: lines with a stop at a station in one of them
  const built = builtCities(), byId = stationsById(w);
  const lines = w.save.lines.filter((l) => l.stops.some((s) => { const st = byId.get(s); return !!st && st.built && built.has(stationZone(st)!); })).length;
  return (
    <>
      <div class="title">
        <span class="name">{ALL_CITIES}</span>
        <HomeButton place={WORLD_HOME} />
        <button class="btn icon" data-tip="Go to all cities" aria-label="Go to all cities" onClick={locateAllCities}>
          <Icon.locate />
        </button>
      </div>
      <ul class="rows">
        <li>
          <span class="grow">Train share</span>
          <span class="val" data-tip={splitTip(t.where, split, t.modes?.rides ?? null, "")}>{split ? split.train.toFixed(1) + "%" : "–"}</span>
        </li>
        <li>
          <span class="grow">Cities</span>
          <span class="val">{exact(t.cities)}</span>
        </li>
        <li>
          <span class="grow">Stations</span>
          <span class="val">{exact(t.stations)}</span>
        </li>
        <li>
          <span class="grow">Lines</span>
          <span class="val">{exact(lines)}</span>
        </li>
        <li>
          <span class="grow">Train trips</span>
          <span class="val" data-tip={t.modes ? `${exact(t.modes.rides.train)} train trips per day${t.where}` : undefined}>{t.modes ? `${count(t.modes.rides.train)}/day` : "–"}</span>
        </li>
        <li>
          <span class="grow">Points</span>
          <span class="val" data-tip={t.pointsTip}>{pointsShort(t.pointsPast)}</span>
        </li>
        <li>
          <span class="grow">Profit</span>
          <span class={"val" + (f && net(f.d) * 1e6 <= -500 ? " money-neg" : "")} data-tip={f ? moneyTip("all cities", f) : undefined}>{f ? `${money(net(f.d) * 1e6)}/day` : "–"}</span>
        </li>
      </ul>
    </>
  );
}
