// The Cities tab (T-550, its own tab since T-749): an All cities row, then one row for every region (a city pack or a zone, the
// regions of the Lines tab's place toggle) where the player has a station, with its mode split bar
// as the top bar draws it, its train rides a day, its profit a day (fares less running trains and upkeep), its
// points over the past 24 game hours (T-684, T-694) and its star once earned (T-685). A click opens
// the city view (T-1438), a double click moves the camera to the region.

import { activeCity, cityById, viewedZone, type City, type CityId } from "../game/cities";
import { cityDemandViews, ridesOf, type DemandView } from "../game/demand";
import { count, exact, money } from "../game/format";
import { AVERAGE_DAY } from "../game/days";
import { moneyState } from "../game/money";
import { POINTS_PER_KM, POINTS_PER_TRIP, pointsPast24h, pointsPerDay, pointsShort, STAR_SHARE } from "../game/points";
import { t, tSpaced } from "../i18n";
import { tPlace } from "../i18n/places";
import { regionLedger } from "../game/regionMoney";
import { splitTip } from "../game/splitTip";
import { citySort, selection, shownDay, shownDayOn, tool } from "../game/ui";
import { world } from "../game/world";
import { zonesLoaded } from "../game/zones";
import { builtCities, sumModes } from "../game/worldTotals";
import { live } from "../map/live";
import type { DayLedger, MoneyState } from "../workers/protocol";
import type { JSX } from "preact";
import { goalWhen } from "./GoalTicks";
import { locateAllCities, playerMovedCamera } from "./locate";
import { Icon, SortHead } from "./widgets";

/** Move the camera to a region's registry centre and zoom, at once; what search does for a city. */
export function goToCity(city: City) {
  playerMovedCamera();
  live.map?.jumpTo({ center: city.center, zoom: city.zoom });
  activeCity.value = city.id;
  // a station or bubble selection back in the old city would hold the focus there (main.ts)
  if (tool.peek() === "select") selection.value = null;
}

/** Fares less running trains and upkeep (T-867): the figure of a row, so a build day does not swamp it. */
export const net = (d: DayLedger) => d.fares - d.running - (d.upkeep ?? 0);
const collate = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

/** A region's profit for the figure: yesterday's ledger once there is one, before that today so far. */
export function figures(zone: CityId, m: MoneyState): { d: DayLedger; yesterday: boolean } | null {
  const todayDay = m.ledger.at(-1)?.day;
  if (todayDay === undefined) return null;
  if (m.ledger.some((d) => d.day === todayDay - 1)) {
    const d = m.regions?.[zone]?.ledger.find((x) => x.day === todayDay - 1) ?? { day: todayDay - 1, fares: 0, running: 0, trains: 0, build: 0 };
    return { d, yesterday: true };
  }
  const today = regionLedger(zone).today;
  return today ? { d: today, yesterday: false } : null;
}

/** The hover of a row's profit: fares and running trains, then new trains and construction (not in the figure) when there were any. */
export function moneyTip(name: string, f: { d: DayLedger; yesterday: boolean }): string {
  const { d } = f;
  const rows = [
    t("cities.profitTip", { yesterday: f.yesterday, name, money: money(net(d) * 1e6) }),
    t("cities.fares", { money: money(d.fares * 1e6) }),
    t("cities.running", { money: money(-d.running * 1e6) }),
    t("cities.upkeep", { money: money(-(d.upkeep ?? 0) * 1e6) }),
  ];
  if (d.trains) rows.push(t("cities.newTrains", { money: money(-d.trains * 1e6) }));
  if (d.build) rows.push(t("cities.construction", { money: money(-d.build * 1e6) }));
  return rows.join("\n");
}

/** T-694: a region's points, what it earned in the past 24 game hours (moves once a game hour). */
export function pointsTip(id: CityId, name: string): string {
  const past = pointsPast24h.value.get(id) ?? 0;
  const rate = pointsPerDay.value.get(id)?.[shownDay.value] ?? 0;
  return [
    t("cities.pointsPast", { points: exact(past), city: name }),
    t("cities.pointsRate", { points: exact(rate), day: shownDayOn.value }),
    t("cities.pointsRule", { trip: POINTS_PER_TRIP, km: POINTS_PER_KM }),
  ].join("\n");
}
function PointsCell({ id, name }: { id: CityId; name: string }) {
  return <span class="val w-points c-cities" data-tip={pointsTip(id, name)}>{pointsShort(pointsPast24h.value.get(id) ?? 0)}</span>;
}

/** T-685: the star beside a region that has reached the mark, with when (its local time, as the
 * top bar's clock hover names a day). */
function Star({ city, at }: { city: City; at: number }) {
  return <span class="city-star" data-tip={t("cities.starTip", { when: goalWhen(city, at), city: city.name, share: STAR_SHARE })}><Icon.star /></span>;
}

interface CityRow {
  id: CityId;
  name: string;
  stations: number;
  view: DemandView | undefined;
  trainShare: number | null;
  rides: number | null;
  money: { d: DayLedger; yesterday: boolean } | null;
}

/** The name of the sums over every region with a built station (T-2389: Anita, they are only the player's cities). */
export const ALL_CITIES = "All cities"; // i18n-ignore: the English fallback; read the name through allCitiesName()
export const allCitiesName = () => t("cities.all");

/** The phrase after "Trips per day" in a split hover (`splitTip`'s `where`). */
const whereIn = (city: string) => tSpaced(t("cities.where", { dayed: shownDay.value !== AVERAGE_DAY, city, day: shownDayOn.value }));

/** T-857, T-2395: the sums over every region with a built station (the Cities tab's first row and the
 * all cities view): their stations, the mode split of all their trips, profit (`money` null before
 * any region has a figure) and points. */
export function allCitiesTotals() {
  const built = builtCities(), views = cityDemandViews.value, m = moneyState.value;
  let stations = 0;
  for (const n of built.values()) stations += n;
  const modes = sumModes([...built.keys()].map((id) => views.get(id)));
  const sum: DayLedger = { day: 0, fares: 0, running: 0, trains: 0, build: 0 };
  let yesterday = true, counted = 0;
  for (const id of built.keys()) {
    const f = m ? figures(id, m) : null;
    if (!f) continue;
    const d = f.d;
    sum.fares += d.fares; sum.running += d.running; sum.upkeep = (sum.upkeep ?? 0) + (d.upkeep ?? 0); sum.trains += d.trains; sum.build += d.build;
    yesterday &&= f.yesterday;
    counted++;
  }
  const past = pointsPast24h.value, rate = pointsPerDay.value;
  let pointsPast = 0, pointsRate = 0;
  for (const id of built.keys()) {
    pointsPast += past.get(id) ?? 0;
    pointsRate += rate.get(id)?.[shownDay.value] ?? 0;
  }
  const where = tSpaced(t("cities.whereAll", { dayed: shownDay.value !== AVERAGE_DAY, day: shownDayOn.value }));
  return {
    cities: built.size, stations, modes, where,
    money: counted ? { d: sum, yesterday } : null,
    pointsPast,
    pointsTip: [t("cities.pointsPastAll", { points: exact(pointsPast) }), t("cities.pointsRate", { points: exact(pointsRate), day: shownDayOn.value })].join("\n"),
  };
}

/** The hover of the all cities name: how many cities and stations it sums. */
function allCitiesTip(sums: { cities: number; stations: number }): string {
  return t("cities.allTip", { name: allCitiesName(), cities: sums.cities, stations: sums.stations });
}

/** T-857: the first row, the sums over every row below: rides, profit and points, and the mode split
 * of all their trips. A click opens the all cities view (T-2395), a double click zooms out to them. */
function TotalRow({ on }: { on: boolean }) {
  const tot = allCitiesTotals();
  const split = tot.modes?.split ?? null, seen = !!tot.modes, rides = tot.modes?.rides.train ?? 0;
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const f = tot.money;
  return (
    <li class={"total" + (on ? " on" : "")} onClick={() => (selection.value = { kind: "world" })}
      onDblClick={() => { locateAllCities(); selection.value = { kind: "world" }; }}>
      <span class="grow" data-tip={allCitiesTip(tot)}>{allCitiesName()}</span>
      <span class="city-bar" data-tip={splitTip(tot.where, split, tot.modes?.rides ?? null, "")}>
        <span class="bar">
          <i class="seg seg-train" style={{ width: s.train + "%" }} />
          <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
          <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
          <i class="seg seg-none" style={{ width: (s.none ?? 0) + "%" }} />
        </span>
      </span>
      <span class="val w-riders c-cities" data-tip={seen ? t("cities.tripsTip", { count: count(rides), where: tot.where.trim() }) : undefined}>{seen ? count(rides) : "–"}</span>
      <span class={"val w-money c-cities c-money" + (f && net(f.d) * 1e6 <= -500 ? " neg" : "")} data-tip={f ? moneyTip(t("cities.allLower"), f) : undefined}>{f ? money(net(f.d) * 1e6) : "–"}</span>
      <span class="val w-points c-cities" data-tip={tot.pointsTip}>{pointsShort(tot.pointsPast)}</span>
    </li>
  );
}

export function CitiesList({ empty = null }: { empty?: JSX.Element | null }) {
  const w = world.value, m = moneyState.value;
  void zonesLoaded.value; // boundaries arriving re-place the stations
  if (!w) return null;
  const views = cityDemandViews.value;
  const built = builtCities();
  if (!built.size) return empty;
  const rows: CityRow[] = [...built].map(([id, stations]) => {
    const view = views.get(id);
    return { id, name: tPlace(id, cityById(id).name), stations, view, trainShare: view ? view.split.train : null, rides: view ? view.railTripsPerDay : null, money: m ? figures(id, m) : null };
  });
  const { key, dir } = citySort.value;
  const past = pointsPast24h.value;
  const val = (r: CityRow) => (key === "train" ? r.trainShare : key === "rides" ? r.rides : key === "money" ? (r.money ? net(r.money.d) : null) : key === "points" ? (past.get(r.id) ?? 0) : null);
  rows.sort((a, b) => {
    if (key !== "name") {
      const x = val(a), y = val(b);
      // a region with no figures yet sorts last whichever way the column runs
      if (x === null || y === null) { if (x !== y) return x === null ? 1 : -1; }
      else if (x !== y) return (x - y) * dir;
    }
    return collate(a.name, b.name) * (key === "name" ? dir : 1);
  });
  const here = viewedZone.value;
  // T-1438: a click opens the city view (as a station row opens the station's), and the selected
  // city's row is the one tinted; with none selected, the viewed region's
  const sel = selection.value, all = sel?.kind === "world", picked = sel?.kind === "city" ? sel.city : all ? "" : null;
  // T-693: a container, so a narrow dock hides the star and then the money before the names go
  return (
    <div class="cities-box">
      <SortHead sort={citySort} cols={[
        { key: "name", label: t("cities.colName"), tip: t("cities.sortName"), first: 1 },
        { key: "train", label: t("cities.colTrain"), tip: t("cities.sortTrain"), first: -1, cls: "w-citybar c-cities" },
        { key: "rides", label: t("cities.colTrips"), tip: t("cities.sortTrips"), first: -1, cls: "w-riders c-cities" },
        { key: "money", label: t("cities.colProfit"), tip: t("cities.sortProfit"), first: -1, cls: "w-money c-cities c-money" },
        { key: "points", label: t("cities.colPoints"), tip: t("cities.sortPoints"), first: -1, cls: "w-points c-cities" },
      ]} />
      <ul class="rows pick cities">
        <TotalRow on={all} />
        {rows.map((r) => {
          const v = r.view, where = whereIn(r.name);
          const rides = v ? ridesOf(v) : null;
          const s = v?.split ?? { train: 0, walk: 0, drive: 0 };
          const f = r.money;
          return (
            <li class={(picked ?? here) === r.id ? "on" : ""} onClick={() => (selection.value = { kind: "city", city: r.id })}
              onDblClick={() => { goToCity(cityById(r.id)); selection.value = { kind: "city", city: r.id }; }}>
              <span class="grow" data-tip={t("cities.rowTip", { name: r.name, stations: r.stations })}>{r.name}</span>
              {m?.stars?.[r.id] !== undefined && <Star city={cityById(r.id)} at={m.stars[r.id]} />}
              <span class="city-bar" data-tip={splitTip(where, v?.split ?? null, rides, "")}>
                <span class="bar">
                  <i class="seg seg-train" style={{ width: s.train + "%" }} />
                  <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
                  <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
                  <i class="seg seg-none" style={{ width: (s.none ?? 0) + "%" }} />
                </span>
              </span>
              <span class="val w-riders c-cities" data-tip={v ? t("cities.tripsTip", { count: count(v.railTripsPerDay), where: where.trim() }) : undefined}>{v ? count(v.railTripsPerDay) : "–"}</span>
              <span class={"val w-money c-cities c-money" + (f && net(f.d) * 1e6 <= -500 ? " neg" : "")} data-tip={f ? moneyTip(r.name, f) : undefined}>{f ? money(net(f.d) * 1e6) : "–"}</span>
              <PointsCell id={r.id} name={r.name} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}
