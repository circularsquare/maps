// The Money tab's Cities section (T-550): one row for every region (a city pack or a zone, the
// regions of the Lines tab's place toggle) where the player has a station, with its mode split bar
// as the top bar draws it, its train rides a day and its money a day (fares less running trains). A click moves the camera
// to the region.

import { activeCity, cityById, viewedZone, type City, type CityId } from "../game/cities";
import { cityDemandViews, type DemandView } from "../game/demand";
import { count, money } from "../game/format";
import { stationZone } from "../game/listScope";
import { moneyState } from "../game/money";
import { regionLedger } from "../game/regionMoney";
import { splitTip } from "../game/splitTip";
import { citySort, selection, shownDay, tool } from "../game/ui";
import { world } from "../game/world";
import { zonesLoaded } from "../game/zones";
import { live } from "../map/live";
import type { DayLedger, MoneyState } from "../workers/protocol";
import { SortHead } from "./widgets";

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
  const val = (r: CityRow) => (key === "train" ? r.trainShare : key === "rides" ? r.rides : key === "money" ? (r.money ? net(r.money.d) : null) : null);
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
        { key: "train", label: "Train", tip: "Sort by the train share of rides", first: -1, cls: "w-citybar" },
        { key: "rides", label: "Rides", tip: "Sort by train rides a day", first: -1, cls: "w-riders" },
        { key: "money", label: "Money", tip: "Sort by money a day, fares less running trains", first: -1, cls: "w-money" },
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
              <span class="city-bar" data-tip={splitTip(where, v?.split ?? null, rides, "")}>
                <span class="bar">
                  <i class="seg seg-train" style={{ width: s.train + "%" }} />
                  <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
                  <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
                </span>
              </span>
              <span class="val w-riders" data-tip={v ? `${count(v.railTripsPerDay)} train rides a day${where}` : undefined}>{v ? count(v.railTripsPerDay) : "–"}</span>
              <span class={"val w-money" + (f && net(f.d) * 1e6 <= -500 ? " neg" : "")} data-tip={f ? moneyTip(r.name, f) : undefined}>{f ? money(net(f.d) * 1e6) : "–"}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}
