// The Money tab's Cities section (T-550): one row for every region (a city pack or a zone, the
// regions of the Lines tab's place toggle) where the player has a station, with its mode split bar
// as the top bar draws it, its train rides a day, its money a day (fares less running trains), its
// points (T-684) and its star once earned (T-685). A click moves the camera to the region.

import { activeCity, cityById, viewedZone, type City, type CityId } from "../game/cities";
import { cityDemandViews, type DemandView } from "../game/demand";
import { clockTime, count, dayOf, money } from "../game/format";
import { weekdayName } from "../game/days";
import { stationZone } from "../game/listScope";
import { shiftAt } from "../game/localTime";
import { liveMoney, moneyState } from "../game/money";
import { POINTS_PER_KM, POINTS_PER_TRIP, pointsExact, pointsPerDay, pointsShort, STAR_SHARE } from "../game/points";
import { regionLedger } from "../game/regionMoney";
import { splitTip } from "../game/splitTip";
import { citySort, selection, shownDay, tool } from "../game/ui";
import { world } from "../game/world";
import { zonesLoaded } from "../game/zones";
import { live } from "../map/live";
import type { DayLedger, MoneyState } from "../workers/protocol";
import { Icon, SortHead } from "./widgets";

/** Move the camera to a region's registry centre and zoom, at once; what search does for a city. */
export function goToCity(city: City) {
  live.map?.jumpTo({ center: city.center, zoom: city.zoom });
  activeCity.value = city.id;
  // a station or bubble selection back in the old city would hold the focus there (main.ts)
  if (tool.peek() === "select") selection.value = null;
}

/** Fares less running trains: the figure of a row, so a build day does not swamp it. */
const net = (d: DayLedger) => d.fares - d.running;
const collate = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

/** A region's money for the figure: yesterday's ledger once there is one, before that today so far. */
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

/** The hover of a row's money: fares and running trains, then new trains and construction (not in the figure) when there were any. */
function moneyTip(name: string, f: { d: DayLedger; yesterday: boolean }): string {
  const { d } = f;
  const rows = [
    `${f.yesterday ? "Yesterday" : "Today so far"} in ${name}: ${money(net(d) * 1e6)} a day, fares less running trains`,
    `Fares ${money(d.fares * 1e6)}`,
    `Running trains ${money(-d.running * 1e6)}`,
  ];
  if (d.trains) rows.push(`New trains ${money(-d.trains * 1e6)}`);
  if (d.build) rows.push(`Construction ${money(-d.build * 1e6)}`);
  return rows.join("\n");
}

/** T-684: a region's points since the game began, its own component since they move as fares land. */
function PointsCell({ id, name }: { id: CityId; name: string }) {
  const total = liveMoney.value.points?.[id] ?? 0;
  const rate = pointsPerDay.value.get(id)?.[shownDay.value] ?? 0;
  const tip = [
    `${pointsExact(total)} points in ${name} since the game began`,
    `${pointsExact(rate)} points ${["on a weekday", "on a weekend day", "a day"][shownDay.value]} with the riders now`,
    `${POINTS_PER_TRIP} for each train trip that starts here, changes included, and ${POINTS_PER_KM} for each km ridden`,
  ].join("\n");
  return <span class="val w-points c-cities" data-tip={tip}>{pointsShort(total)}</span>;
}

/** T-685: the star beside a region that has reached the mark, with when (its local time). */
function Star({ city, at }: { city: City; at: number }) {
  const t = (at + shiftAt(city.center[0], city.center[1])) / 60;
  return <span class="city-star" data-tip={`Star earned on ${weekdayName(dayOf(t))} ${dayOf(t)} at ${clockTime(t)}, once the train share in ${city.name} reached ${STAR_SHARE}%`}><Icon.star /></span>;
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

export function CitiesList() {
  const w = world.value, m = moneyState.value;
  void zonesLoaded.value; // boundaries arriving re-place the stations
  if (!w) return null;
  const views = cityDemandViews.value;
  const built = new Map<CityId, number>();
  for (const s of w.save.stations) {
    if (!s.built) continue;
    const z = stationZone(s);
    if (z) built.set(z, (built.get(z) ?? 0) + 1);
  }
  if (!built.size) return null;
  const rows: CityRow[] = [...built].map(([id, stations]) => {
    const view = views.get(id);
    return { id, name: cityById(id).name, stations, view, trainShare: view ? view.split.train : null, rides: view ? view.railTripsPerDay : null, money: m ? figures(id, m) : null };
  });
  const { key, dir } = citySort.value;
  // T-684: by points as of the last whole hour, so the order does not shuffle as points land
  const val = (r: CityRow) => (key === "train" ? r.trainShare : key === "rides" ? r.rides : key === "money" ? (r.money ? net(r.money.d) : null) : key === "points" ? (m?.points?.[r.id] ?? 0) : null);
  rows.sort((a, b) => {
    if (key !== "name") {
      const x = val(a), y = val(b);
      // a region with no figures yet sorts last whichever way the column runs
      if (x === null || y === null) { if (x !== y) return x === null ? 1 : -1; }
      else if (x !== y) return (x - y) * dir;
    }
    return collate(a.name, b.name) * (key === "name" ? dir : 1);
  });
  const day = [" on a weekday", " on a weekend day", ""][shownDay.value];
  const here = viewedZone.value;
  return (
    <>
      <div class="h">Cities</div>
      <SortHead sort={citySort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        { key: "train", label: "Train", tip: "Sort by the train share of rides", first: -1, cls: "w-citybar c-cities" },
        { key: "rides", label: "Rides", tip: "Sort by train rides per day", first: -1, cls: "w-riders c-cities" },
        { key: "money", label: "Money", tip: "Sort by money a day, fares less running trains", first: -1, cls: "w-money c-cities" },
        { key: "points", label: "Points", tip: "Sort by points since the game began", first: -1, cls: "w-points c-cities" },
      ]} />
      <ul class="rows pick cities">
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
              <span class={"val w-money c-cities" + (f && net(f.d) * 1e6 <= -500 ? " neg" : "")} data-tip={f ? moneyTip(r.name, f) : undefined}>{f ? money(net(f.d) * 1e6) : "–"}</span>
              <PointsCell id={r.id} name={r.name} />
            </li>
          );
        })}
      </ul>
    </>
  );
}
