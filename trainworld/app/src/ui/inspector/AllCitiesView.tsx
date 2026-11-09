// T-2395: the all cities view, opened from the Cities tab's first row: that row's sums laid out as a
// city view's figures (CityDetails.tsx), with its home and go-to buttons. Totals only (Anita).

import { count, exact, money, moneyNegative } from "../../game/format";
import { t as tr } from "../../i18n";
import { pointsShort } from "../../game/points";
import { splitTip } from "../../game/splitTip";
import { stationZone } from "../../game/listScope";
import { builtCities } from "../../game/worldTotals";
import { stationsById, world } from "../../game/world";
import { WORLD_HOME } from "../../workers/saveFile";
import { allCitiesName, allCitiesTotals, moneyTip, net } from "../Cities";
import { locateAllCities } from "../locate";
import { Icon } from "../widgets";
import { HomeButton } from "./HomeButton";
import { pct1 } from "./tripText";

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
        <span class="name">{allCitiesName()}</span>
        <HomeButton place={WORLD_HOME} />
        <button class="btn icon" data-tip={tr("allCitiesView.goTo")} aria-label={tr("allCitiesView.goTo")} onClick={locateAllCities}>
          <Icon.locate />
        </button>
      </div>
      <ul class="rows">
        <li>
          <span class="grow">{tr("cityDetails.trainShare")}</span>
          <span class="val" data-tip={splitTip(t.where, split, t.modes?.rides ?? null, "")}>{split ? pct1(split.train / 100) : "–"}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.cities")}</span>
          <span class="val">{exact(t.cities)}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.stations")}</span>
          <span class="val">{exact(t.stations)}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.lines")}</span>
          <span class="val">{exact(lines)}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.trainTrips")}</span>
          <span class="val" data-tip={t.modes ? tr("allCitiesView.tripsTip", { count: exact(t.modes.rides.train), where: t.where }) : undefined}>{t.modes ? tr("inspector.perDay", { value: count(t.modes.rides.train) }) : "–"}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.points")}</span>
          <span class="val" data-tip={t.pointsTip}>{pointsShort(t.pointsPast)}</span>
        </li>
        <li>
          <span class="grow">{tr("cityDetails.profit")}</span>
          <span class={"val" + (f && moneyNegative(net(f.d) * 1e6) ? " money-neg" : "")} data-tip={f ? moneyTip(tr("allCitiesView.allCities"), f) : undefined}>{f ? tr("inspector.perDay", { value: money(net(f.d) * 1e6) }) : "–"}</span>
        </li>
      </ul>
    </>
  );
}
