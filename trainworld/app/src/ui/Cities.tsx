// The Cities tab (T-550, its own tab since T-749): a World row, then one row for every region (a city pack or a zone, the
// regions of the Lines tab's place toggle) where the player has a station, with its mode split bar
// as the top bar draws it, its train rides a day, its profit a day (fares less running trains and upkeep), its
// points over the past 24 game hours (T-684, T-694) and its star once earned (T-685). A click moves the camera to the region.

import { activeCity, cityById, viewedZone, type City, type CityId } from "../game/cities";
import { cityDemandViews, type DemandView } from "../game/demand";
import { clockTime, count, dayOf, exact, money } from "../game/format";
import { AVERAGE_DAY, weekdayOf } from "../game/days";
import { shiftAt } from "../game/localTime";
import { moneyState } from "../game/money";
import { POINTS_PER_KM, POINTS_PER_TRIP, pointsPast24h, pointsPerDay, pointsShort, STAR_SHARE } from "../game/points";
import { regionLedger } from "../game/regionMoney";
import { splitTip } from "../game/splitTip";
import { citySort, selection, shownDay, shownDayOn, tool } from "../game/ui";
import { world } from "../game/world";
import { zonesLoaded } from "../game/zones";
import { builtCities, sumModes } from "../game/worldTotals";
import { live } from "../map/live";
import type { DayLedger, MoneyState } from "../workers/protocol";
import type { JSX } from "preact";
import { Icon, SortHead } from "./widgets";
import { LeaderboardButton } from "./Leaderboard";

/** Move the camera to a region's registry centre and zoom, at once; what search does for a city. */
export function goToCity(city: City) {
  live.map?.jumpTo({ center: city.center, zoom: city.zoom });
  activeCity.value = city.id;
  // a station or bubble selection back in the old city would hold the focus there (main.ts)
  if (tool.peek() === "select") selection.value = null;
}

/** Fares less running trains and upkeep (T-867): the figure of a row, so a build day does not swamp it. */
const net = (d: DayLedger) => d.fares - d.running - (d.upkeep ?? 0);
const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const collate = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

/** A region's profit for the figure: yesterday's ledger once there is one, before that today so far. */
function figures(zone: CityId, m: MoneyState): { d: DayLedger; yesterday: boolean } | null {
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
function moneyTip(name: string, f: { d: DayLedger; yesterday: boolean }): string {
  const { d } = f;
  const rows = [
    `Profit ${f.yesterday ? "yesterday" : "today so far"} in ${name}: ${money(net(d) * 1e6)} a day, fares minus running trains and upkeep`,
    `Fares ${money(d.fares * 1e6)}`,
    `Running trains ${money(-d.running * 1e6)}`,
    `Upkeep ${money(-(d.upkeep ?? 0) * 1e6)}`,
  ];
  if (d.trains) rows.push(`New trains ${money(-d.trains * 1e6)}`);
  if (d.build) rows.push(`Construction ${money(-d.build * 1e6)}`);
  return rows.join("\n");
}

/** T-694: a region's points, what it earned in the past 24 game hours (moves once a game hour). */
function PointsCell({ id, name }: { id: CityId; name: string }) {
  const past = pointsPast24h.value.get(id) ?? 0;
  const rate = pointsPerDay.value.get(id)?.[shownDay.value] ?? 0;
  const tip = [
    `${exact(past)} points in ${name} in the past 24 hours`,
    `${exact(rate)} points ${shownDayOn.value} with the riders now`,
    `${POINTS_PER_TRIP} for each train trip that starts here, changes included, and ${POINTS_PER_KM} for each km ridden`,
  ].join("\n");
  return <span class="val w-points c-cities" data-tip={tip}>{pointsShort(past)}</span>;
}

/** T-685: the star beside a region that has reached the mark, with when (its local time, as the
 * top bar's clock hover names a day). */
function Star({ city, at }: { city: City; at: number }) {
  const t = Math.floor((at + shiftAt(city.center[0], city.center[1])) / 60), d = dayOf(t);
  return <span class="city-star" data-tip={`Star earned on ${WEEKDAY_LONG[weekdayOf(d)]}, day ${d}, at ${clockTime(t)}, once the train share in ${city.name} reached ${STAR_SHARE}%`}><Icon.star /></span>;
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

/** T-857: the first row, the sums over every row below: rides, profit and points, and the mode split
 * of all their trips. Not a region, so a click does nothing. */
function TotalRow({ rows, day }: { rows: CityRow[]; day: string }) {
  const stations = rows.reduce((n, r) => n + r.stations, 0);
  const sum0 = sumModes(rows.map((r) => r.view));
  const split = sum0?.split ?? null, seen = !!sum0, rides = sum0?.rides.train ?? 0;
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const where = ` in the world${day}`;
  const sum: DayLedger = { day: 0, fares: 0, running: 0, trains: 0, build: 0 };
  let yesterday = true, counted = 0;
  for (const r of rows) {
    if (!r.money) continue;
    const d = r.money.d;
    sum.fares += d.fares; sum.running += d.running; sum.upkeep = (sum.upkeep ?? 0) + (d.upkeep ?? 0); sum.trains += d.trains; sum.build += d.build;
    yesterday &&= r.money.yesterday;
    counted++;
  }
  const past = pointsPast24h.value, rate = pointsPerDay.value;
  const pastAll = rows.reduce((n, r) => n + (past.get(r.id) ?? 0), 0);
  const rateAll = rows.reduce((n, r) => n + (rate.get(r.id)?.[shownDay.value] ?? 0), 0);
  const pointsTip = [
    `${exact(pastAll)} points in the world in the past 24 hours`,
    `${exact(rateAll)} points ${shownDayOn.value} with the riders now`,
  ].join("\n");
  return (
    <li class="total">
      <span class="grow" data-tip={`World\n${rows.length} ${rows.length === 1 ? "city" : "cities"}, ${stations} ${stations === 1 ? "station" : "stations"}`}>World</span>
      <LeaderboardButton />
      <span class="city-bar" data-tip={splitTip(where, split, sum0?.rides ?? null, "")}>
        <span class="bar">
          <i class="seg seg-train" style={{ width: s.train + "%" }} />
          <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
          <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
        </span>
      </span>
      <span class="val w-riders c-cities" data-tip={seen ? `${count(rides)} train rides per day${where}` : undefined}>{seen ? count(rides) : "–"}</span>
      <span class={"val w-money c-cities c-money" + (counted && net(sum) * 1e6 <= -500 ? " neg" : "")} data-tip={counted ? moneyTip("the world", { d: sum, yesterday }) : undefined}>{counted ? money(net(sum) * 1e6) : "–"}</span>
      <span class="val w-points c-cities" data-tip={pointsTip}>{pointsShort(pastAll)}</span>
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
    return { id, name: cityById(id).name, stations, view, trainShare: view ? view.split.train : null, rides: view ? view.railTripsPerDay : null, money: m ? figures(id, m) : null };
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
  const day = (shownDay.value === AVERAGE_DAY ? "" : " " + shownDayOn.value);
  const here = viewedZone.value;
  // T-693: a container, so a narrow dock hides the star and then the money before the names go
  return (
    <div class="cities-box">
      <SortHead sort={citySort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        { key: "train", label: "Train", tip: "Sort by the train share of rides", first: -1, cls: "w-citybar c-cities" },
        { key: "rides", label: "Rides", tip: "Sort by train rides per day", first: -1, cls: "w-riders c-cities" },
        { key: "money", label: "Profit", tip: "Sort by profit a day, fares minus running trains and upkeep", first: -1, cls: "w-money c-cities c-money" },
        { key: "points", label: "Points", tip: "Sort by points in the past 24 hours", first: -1, cls: "w-points c-cities" },
      ]} />
      <ul class="rows pick cities">
        <TotalRow rows={rows} day={day} />
        {rows.map((r) => {
          const v = r.view, where = ` in ${r.name}${day}`;
          const all = v?.tripsPerDay ?? 0;
          const rides = v ? { all, train: v.railTripsPerDay, walk: all * v.split.walk / 100, drive: all * v.split.drive / 100 } : null;
          const s = v?.split ?? { train: 0, walk: 0, drive: 0 };
          const f = r.money;
          return (
            <li class={here === r.id ? "on" : ""} onClick={() => goToCity(cityById(r.id))}>
              <span class="grow" data-tip={`${r.name}\n${r.stations} ${r.stations === 1 ? "station" : "stations"}`}>{r.name}</span>
              {m?.stars?.[r.id] !== undefined && <Star city={cityById(r.id)} at={m.stars[r.id]} />}
              <span class="city-bar" data-tip={splitTip(where, v?.split ?? null, rides, "")}>
                <span class="bar">
                  <i class="seg seg-train" style={{ width: s.train + "%" }} />
                  <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
                  <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
                </span>
              </span>
              <span class="val w-riders c-cities" data-tip={v ? `${count(v.railTripsPerDay)} train rides per day${where}` : undefined}>{v ? count(v.railTripsPerDay) : "–"}</span>
              <span class={"val w-money c-cities c-money" + (f && net(f.d) * 1e6 <= -500 ? " neg" : "")} data-tip={f ? moneyTip(r.name, f) : undefined}>{f ? money(net(f.d) * 1e6) : "–"}</span>
              <PointsCell id={r.id} name={r.name} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}
