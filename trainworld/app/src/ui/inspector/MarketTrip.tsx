// T-866: the trip explorer between zones. The inter-city trips between a bubble selection (or the
// trip tool's start) and a far place or 6 km area, as the regional market chooses them
// (game/trip.ts `MarketPeriod`): every area pair between the two ends added up by its trips. The
// train's routes are the market's own through the period's strategies; the walks or drives to the
// stations are known only as felt minutes, averaged over the area.

import { activeCity, cityById } from "../../game/cities";
import { count, distance, distUnit, exact, perDistance, roughDistance, speed } from "../../game/format";
import {
  AIR_FARE_USD, AIR_FIXED_MIN, AIR_KMH, AIR_NO_DIRECT_MIN, AIR_TERMINAL_MIN, AIR_USD_PER_KM, CAR_KMH, CAR_SHARED_BY, CAR_USD_PER_KM, carPartsOf, corridorOf,
  crossingsOf, DRIVE_FIXED_MIN, loadMarketRows, marketRow, marketRowsReady, MIN_PER_USD, ROAD_DETOUR, type CarCost, type Crossing, type FlightCost,
} from "../../game/marketRule";
import { modeColours } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { eachCorridor, intercityDemand, intercityPlaceName } from "../../game/demand";
import { peopleOf, tripPeriod, wholePeople, type MarketEnd, type MarketPeriod, type MarketShown, type TripRoute } from "../../game/trip";
import { shownDayPhrase } from "../../game/ui";
import { linesById, stationNames, world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { SegBar } from "../widgets";
import { legRows, PeriodRows, routeRow } from "./RouteParts";
import { equation, minText, num1, pctShare, roundTo, type Term } from "./tripText";

const usd = (v: number) => "$" + v.toFixed(2);
const perKm = (usdPerKm: number) => `${Math.round(100 * perDistance(usdPerKm))} cents a ${distUnit()}`;

/** T-1148: the crossings' rows. */
const CROSSING: Record<Crossing["kind"], { label: string; tip: (c: Crossing) => string }> = {
  channel: { label: "Channel crossing", tip: (c) => `Le Shuttle or a ferry across the Channel: $${c.usd} a car, split between the ${CAR_SHARED_BY} people in it, and ${c.min} minutes on top of the drive` },
  tsugaru: { label: "Ferry to Hokkaido", tip: (c) => `The ferry between Aomori and Hakodate: $${c.usd} a car, split between the ${CAR_SHARED_BY} people in it, and ${c.min / 60} hours on top of the drive. Cars cannot use the Seikan Tunnel.` },
};

/** The car's effective time as an equation, its parts in the order of the rows. */
function carEquation(c: CarCost): string {
  const terms: Term[] = [
    { label: "drive", real: num1(c.driveMin), weight: 1, perceived: c.driveMin },
    { label: "running", real: usd(c.runUsd), weight: MIN_PER_USD, perceived: c.runUsd * MIN_PER_USD },
    ...(c.parkMin >= 0.05 ? [{ label: "parking", real: "", weight: 1, perceived: c.parkMin }] : []),
    ...c.crossings.flatMap((x) => [
      { label: "crossing fare", real: usd(x.usd / CAR_SHARED_BY), weight: MIN_PER_USD, perceived: (x.usd / CAR_SHARED_BY) * MIN_PER_USD },
      { label: "crossing time", real: num1(x.min), weight: 1, perceived: x.min },
    ]),
  ];
  return `${equation(terms, c.total)}\nEach dollar counts as ${MIN_PER_USD} minutes.`;
}

/** By car: the drive, its running costs, parking, any crossing, and the effective time. `parts`
 * null: the effective time alone (a group of far places with different crossings). */
function CarRows({ total, parts, where }: { total: number; parts: CarCost | null; where: string }) {
  const c = parts;
  return (
    <ul class="rows">
      {c && (
        <>
          <li data-tip={`About ${distance(1000 * c.roadKm)} of road at ${speed(CAR_KMH)}, taken as ${ROAD_DETOUR} times the straight line, and ${DRIVE_FIXED_MIN} minutes more for setting off and arriving`}>
            <span class="grow">Drive</span>
            <span class="val">{minText(c.driveMin)}</span>
          </li>
          <li data-tip={`Fuel and wear at ${perKm(CAR_USD_PER_KM)} of road, split between the ${CAR_SHARED_BY} people in the car`}>
            <span class="grow">Running costs</span>
            <span class="val">{usd(c.runUsd)}</span>
          </li>
          {c.parkMin >= 0.5 && (
            <li data-tip={`Finding a space where the trip ends and paying for it, in minutes, split between the ${CAR_SHARED_BY} people in the car`}>
              <span class="grow">Parking</span>
              <span class="val">{minText(c.parkMin)}</span>
            </li>
          )}
          {c.crossings.map((x) => (
            <li data-tip={CROSSING[x.kind].tip(x)}>
              <span class="grow">{CROSSING[x.kind].label}</span>
              <span class="val">{usd(x.usd / CAR_SHARED_BY)}, {minText(x.min)}</span>
            </li>
          ))}
        </>
      )}
      <li data-tip={c ? `${carEquation(c)}\n${where}` : where}>
        <span class="grow">Effective time</span>
        <span class="val">{minText(total)}</span>
      </li>
    </ul>
  );
}

/** By plane: the drives to and from the airports, the time at them, the flight, the fare and the
 * effective time. */
function PlaneRows({ f, km }: { f: FlightCost; km: number }) {
  const terms: Term[] = [
    { label: "airport drives", real: num1(f.accessFrom + f.accessTo), weight: 1, perceived: f.accessFrom + f.accessTo },
    { label: "at the airports", real: String(AIR_TERMINAL_MIN), weight: 1, perceived: AIR_TERMINAL_MIN },
    { label: "flight", real: num1(f.flightMin), weight: 1, perceived: f.flightMin },
    { label: "fare", real: usd(f.fareUsd), weight: MIN_PER_USD, perceived: f.fareUsd * MIN_PER_USD },
  ];
  return (
    <ul class="rows">
      <li data-tip={`${minText(f.accessFrom)} driving to the airport where the trip starts and ${minText(f.accessTo)} from the one where it ends, on average over the people of each place`}>
        <span class="grow">To and from airports</span>
        <span class="val">{minText(f.accessFrom + f.accessTo)}</span>
      </li>
      <li data-tip="Check-in, security and boarding, then getting off and out at the other end">
        <span class="grow">At the airports</span>
        <span class="val">{minText(AIR_TERMINAL_MIN)}</span>
      </li>
      <li data-tip={`${distance(1000 * km)} at ${speed(AIR_KMH)}, and ${AIR_FIXED_MIN} minutes for taking off and landing` + (f.direct ? "" : `. No direct flight joins the two places, so a change adds ${AIR_NO_DIRECT_MIN / 60 === 1 ? "an hour" : `${AIR_NO_DIRECT_MIN} minutes`}.`)}>
        <span class="grow">Flight</span>
        <span class="val">{minText(f.flightMin)}</span>
      </li>
      <li data-tip={`$${AIR_FARE_USD} and ${perKm(AIR_USD_PER_KM)} flown`}>
        <span class="grow">Fare</span>
        <span class="val">{usd(f.fareUsd)}</span>
      </li>
      <li data-tip={`${equation(terms, f.total)}\nEach dollar counts as ${MIN_PER_USD} minutes.`}>
        <span class="grow">Effective time</span>
        <span class="val">{minText(f.total)}</span>
      </li>
    </ul>
  );
}

/** Two places' longer trips a day both ways as the solve split them (the corridor's, commutes between
 * the zones left out); null with none. */
function pairFigures(a: number, b: number): { trips: number; rail: number; air: number; car: number } | null {
  const c = intercityDemand.value?.corridors;
  if (!c || a === b) return null;
  let out: { trips: number; rail: number; air: number; car: number } | null = null;
  eachCorridor(c, (x, y) => (x === a && y === b) || (x === b && y === a), (_, f) => {
    const trips = f.trips - f.commutes, rail = f.rail - f.commuteRail;
    if (trips > 0) out = { trips, rail, air: f.air, car: Math.max(0, f.car - (f.commutes - f.commuteRail)) };
  });
  return out;
}

/** T-1148: the two places' trips by mode, under the selection's own. */
function PairModes({ pair, near, far, day }: { pair: { trips: number; rail: number; air: number; car: number }; near: number; far: number; day: string }) {
  const n = peopleOf(pair.trips);
  const model = [pair.rail / pair.trips, pair.air / pair.trips, pair.car / pair.trips];
  const whole = wholePeople(model, n).map((p) => p / n);
  const of = `the ${count(pair.trips)} inter-city trips ${day} between ${intercityPlaceName(near)} and ${intercityPlaceName(far)}, both ways`;
  const parts = ([["Train", whole[0], modeColours().train], ["Plane", whole[1], modeColours().other], ["Car", whole[2], modeColours().drive]] as [string, number, string][]).filter(([nm, v]) => nm === "Train" || v > 0);
  return (
    <>
      <div class="h">Between the two places<span class="right muted small" data-tip={`${exact(pair.trips)} inter-city trips ${day} between ${intercityPlaceName(near)} and ${intercityPlaceName(far)}, both ways`}>{count(pair.trips)} trips</span></div>
      <div class="trip-modes">
        <SegBar kind="trip-bar" label={parts.map(([nm, v]) => `${nm} ${pctShare(v)}`).join(", ")} parts={parts.map(([nm, v, c]) => ({ share: v, colour: c, tip: `${nm}: ${pctShare(v)} of ${of}` }))} />
        <ul class="rows">
          {parts.map(([nm, v, c]) => (
            <li data-tip={`${nm}: ${pctShare(v)} of ${of}`}>
              <i class="mk" style={{ background: c }} />
              <span class="grow">{nm}</span>
              <span class="val w-num">{pctShare(v)}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

/** T-1229: the everyday outings across a zone line between the two ends, by train, on foot and by
 * car, over the day as the solve split them. */
function EverydayModes({ legs, rail, walk, day }: { legs: number; rail: number; walk: number; day: string }) {
  const model = [rail / legs, walk / legs, Math.max(0, legs - rail - walk) / legs];
  const n = peopleOf(legs);
  const whole = wholePeople(model, n).map((p) => p / n);
  const of = `the ${count(legs)} trip legs ${day} of errands, visits, school and medical trips between the two ends`;
  const c = modeColours();
  const parts = ([["Train", whole[0], c.train], ["Walking", whole[1], c.walk], ["Car", whole[2], c.drive]] as [string, number, string][]).filter(([, v]) => v > 0);
  return (
    <>
      <div class="h">Everyday trips<span class="right muted small" data-tip={`${exact(legs)} trip legs ${day}, to the other end and back: errands, visits, school and medical trips between neighbouring areas on either side of a zone line. Inter-city trips are above.`}>{count(legs)} legs</span></div>
      <div class="trip-modes">
        <SegBar kind="trip-bar" label={parts.map(([nm, v]) => `${nm} ${pctShare(v)}`).join(", ")} parts={parts.map(([nm, v, col]) => ({ share: v, colour: col, tip: `${nm}: ${pctShare(v)} of ${of}` }))} />
        <ul class="rows">
          {parts.map(([nm, v, col]) => (
            <li data-tip={`${nm}: ${pctShare(v)} of ${of}`}>
              <i class="mk" style={{ background: col }} />
              <span class="grow">{nm}</span>
              <span class="val w-num">{pctShare(v)}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

/** The crossings between the near place and every far one, or null when they differ (or a row is
 * missing). */
function crossingSet({ near, fars }: { near: number; fars: number[] }): Crossing[] | null {
  const a = marketRow(near);
  if (!a) return null;
  let out: Crossing[] | null = null;
  for (const f of fars) {
    const b = marketRow(f);
    if (!b) return null;
    const c = crossingsOf(a, b);
    if (out && out.map((x) => x.kind).join() !== c.map((x) => x.kind).join()) return null;
    out = c;
  }
  return out;
}

/** The near end's place, and the far ends' places. */
function placesOf(m: MarketShown): { near: number; fars: number[] } {
  return { near: m.near?.place ?? cityById(activeCity.value).place, fars: [...new Set((m.fars ?? [m.far]).map((f) => f.place))] };
}

const NO_RAIL: Record<string, string> = {
  farClosed: "An end of this trip is not open in this game, so trains between the two are not worked out.",
  noRail: "No train serves both ends in this period.",
  unmatched: "No homes of the inter-city model are here.",
};

/** The far end as the panel names it. */
export function marketFarName(m: MarketShown): string {
  const fars = m.fars ?? [m.far];
  // T-885: a group: "3 parts of Philadelphia", "Philadelphia and 2 other places"
  if (fars.length > 1) {
    const places = [...new Set(fars.map((f) => f.place))];
    if (places.length === 1) return `${fars.length} parts of ${intercityPlaceName(places[0])}`;
    const whole = fars.every((f) => f.gid < 0);
    return `${intercityPlaceName(fars[0].place)} and ${fars.length - 1} other ${whole ? (fars.length === 2 ? "place" : "places") : (fars.length === 2 ? "area" : "areas")}`;
  }
  return marketEndName(m.far);
}

/** A market place or one of its 6 km zones, as the panel names it. */
export function marketEndName(e: MarketEnd): string {
  const place = intercityPlaceName(e.place);
  return e.gid >= 0 ? `part of ${place}` : place;
}

/** A market route's real minutes station to station: waits, rides and walks between stations. */
function stationTimes(r: TripRoute) {
  let wait = 0, ride = 0, xfer = 0;
  for (const l of r.legs) {
    if (l.kind === "ride") { wait += l.waitMin; ride += l.min; }
    else xfer += l.min;
  }
  return { wait, ride, xfer, total: wait + ride + xfer };
}

function Legs({ t, r }: { t: MarketPeriod; r: TripRoute }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  const pair = t.pairs[r.pair];
  const first = r.legs[0], last = r.legs[r.legs.length - 1];
  const name = (id: string) => st.get(id) ?? "a station";
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={`On foot or by car, felt as ${num1(pair.accessPerceived)} min on average over the homes at this end`}>
          <span class="grow">To {name(first.from)}</span>
          <span class="val muted">{minText(pair.accessPerceived)}</span>
        </li>
      )}
      {legRows(r, name, ln, (l) => `Wait ${l.waitMin.toFixed(1)} min (half the time between the useful trains), then ${l.min.toFixed(1)} min on the train over ${l.stops} ${l.stops === 1 ? "stop" : "stops"}`)}
      {pair && last && (
        <li data-tip={`On foot or dropped off, felt as ${num1(pair.egressPerceived)} min on average over the places at this end`}>
          <span class="grow">From {name(last.to)}</span>
          <span class="val muted">{minText(pair.egressPerceived)}</span>
        </li>
      )}
    </ul>
  );
}

/** The other routes the market's train riders take, by share. */
function OtherRoutes({ t, shown, riders }: { t: MarketPeriod; shown: number[]; riders: number | null }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  // T-911: `shown` is each route's share in whole train riders, the rest last; a route no one takes
  // at this count is not listed
  const others = t.routes.slice(1, 5).map((r, i) => [r, shown[i + 1]] as const).filter(([, v]) => !riders || v > 0);
  const rest = shown[shown.length - 1];
  if (!others.length && rest < 0.005) return null;
  return (
    <>
      <div class="h">Other routes<span class="right muted small">train riders</span></div>
      <ul class="rows">
        {others.map(([r, v]) => {
          const { names, from, to, label } = routeRow(r, st, ln);
          return (
            <li data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of these trips' train riders${riders ? `, ${pctShare(v)} of the ${count(riders)} counted here` : ""}, ${minText(stationTimes(r).total)} station to station`}>
              {label}
              <span class="val w-num">{pctShare(v)}</span>
            </li>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted" data-tip={`The train riders spread over ${t.pairs.length} pairs of stations near the two ends. Routes under 0.5% of them are not listed.`}>
            <span class="grow">Smaller routes</span>
            <span class="val w-num">{pctShare(rest)}</span>
          </li>
        )}
      </ul>
    </>
  );
}

/** The train's share of a period's trips, in whole trips. */
function trainShare(x: MarketPeriod, ways: number): number {
  const r0 = x.trips > 0 ? x.rail / x.trips : 0;
  return roundTo([r0, 1 - r0], Math.round(ways * x.trips) || null)[0];
}

export function MarketTrip({ m }: { m: MarketShown }) {
  const q = shownPeriod.value;
  const t = m.periods[q];
  const day = shownDayPhrase.value;
  const period = PERIODS[q].name;
  // T-1148: between two places both ways, as the city view's corridors count them; from a selection
  // one way, as its bubbles count them (the model's figures are each way)
  const ways = m.near ? 2 : 1;
  const sum = (k: "trips" | "rail" | "air" | "car") => ways * m.periods.reduce((s, p) => s + (p?.[k] ?? 0), 0);
  const all = sum("trips");
  // T-1229: the everyday outings across a zone line are legs a day both ways already
  const sumE = (k: "legs" | "rail" | "walk") => m.periods.reduce((s, p) => s + (p?.everyday[k] ?? 0), 0);
  const [eLegs, eRail, eWalk] = [sumE("legs"), sumE("rail"), sumE("walk")];
  // T-1148: the rows give the car's parts and the plane
  marketRowsReady.value;
  void loadMarketRows();
  if (!t) return <p class="muted">Calculating demand.</p>;
  // T-1148: a train in play in some period; with none the bar and rows leave the train out
  const inPlay = !m.unbuilt && m.periods.some((p) => p && !p.why);
  // T-911: in whole trips; the hovers keep the model's shares
  const model = all > 0 ? [sum("rail") / all, sum("air") / all, sum("car") / all] : [];
  const whole = wholePeople(model, peopleOf(all)).map((p) => p / peopleOf(all));
  const shares: [string, number, string, number][] = all > 0
    ? ([["Train", whole[0], modeColours().train, model[0]], ["Plane", whole[1], modeColours().other, model[1]], ["Car", whole[2], modeColours().drive, model[2]]] as [string, number, string, number][]).filter(([n]) => n !== "Train" || inPlay)
    : [];
  const ends = placesOf(m);
  const one = ends.fars.length === 1 ? ends.fars[0] : -1;
  // from where the trips start: the far end when the near cells are their destinations
  const corridor = one >= 0 && marketRowsReady.value ? (m.end === "work" ? corridorOf(one, ends.near) : corridorOf(ends.near, one)) : null;
  // the crossings when every far place shares them; the car's parts then follow from the sim's minutes
  const cross = marketRowsReady.value ? crossingSet(ends) : null;
  const carParts = cross && Number.isFinite(t.carPerceived) ? carPartsOf(t.km, t.carPerceived, cross) : null;
  // T-1148: from a selection or a 6 km area, the whole places' trips as the solve split them, beside it
  const pair = one >= 0 && !(m.near && m.near.gid < 0) ? pairFigures(ends.near, one) : null;
  const bare = m.unbuilt ?? [];
  const noStation = bare.length === 0 ? "" : bare.length === 1 ? `There is no station near ${intercityPlaceName(bare[0])}.`
    : bare.length === 2 && one >= 0 ? "There are no stations near either place." : `There are no stations near ${intercityPlaceName(bare[0])} and ${bare.length - 1} other places.`;
  // the period's train riders in whole trips, for the route shares (none: the model's)
  const riders = Math.round(ways * t.rail) || null;
  const top = t.routes.slice(0, 5).map((r) => r.share);
  // the routes too small to list take a leftover rider last
  const routeShares = roundTo([...top, Math.max(0, 1 - top.reduce((s, x) => s + x, 0))], riders, top.length);
  // a single home's share of its area's trips is a fraction of one
  const trips = all >= 0.5 ? `${count(all)} inter-city trips` : "Under one inter-city trip";
  const of = all >= 0.5 ? `the ${count(all)} inter-city trips ${day} between the two ends` : `the inter-city trips ${day} between the two ends`;
  const main = t.routes[0];
  const tm = main ? stationTimes(main) : null;
  const fare = main ? (t.baseFarePerceived + main.legs.reduce((s, l) => s + l.farePerceived, 0)) / t.minPerUsd : 0;
  const repTip = `Inter-city trips in the ${period.toLowerCase()}: every pair of 6 km areas between the two ends, added up by its trips${tripPeriod.value === null ? "; the period with most of them" : ""}`;
  // T-991: an end that is not open has no trips worked out, but the distance and the car are known
  // from the places' centres; with no distance at all, neither is shown
  const known = t.km > 0;
  const closed = t.why === "farClosed";
  return (
    <>
      {closed ? known && (
        <p class="muted small" data-tip="Straight line between the two places' centres">{roughDistance(1000 * t.km)} apart</p>
      ) : (
        <p class="muted small" data-tip={`${all >= 0.5 ? exact(all) : num1(all)} inter-city trips ${day} ${m.near ? "between the two ends, both ways" : m.end === "work" ? `from ${marketFarName(m)} to these places` : `from these homes to ${marketFarName(m)}`}: visits, business and leisure trips of 25 km and more. Commutes between the zones are left out.`}>
          {trips} {day}{known && `, ${roughDistance(1000 * t.km)} apart`}
        </p>
      )}
      {shares.length > 0 && (
        <div class="trip-modes">
          <SegBar kind="trip-bar" label={shares.map(([n, v]) => `${n} ${pctShare(v)}`).join(", ")}
            parts={shares.map(([n, v, c]) => ({ share: v, colour: c, tip: `${n}: ${pctShare(v)} of ${of}` }))} />
          <ul class="rows">
            {shares.filter(([n, v]) => n !== "Plane" || v > 0).map(([n, v, c, e]) => (
              <li data-tip={`${n}: ${pctShare(v)} of ${of}\nModel: ${(100 * e).toFixed(1)}%, rounded here to whole trips`}>
                <i class="mk" style={{ background: c }} />
                <span class="grow">{n}</span>
                <span class="val w-num">{pctShare(v)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {pair && <PairModes pair={pair} near={ends.near} far={one} day={day} />}
      {!closed && eLegs >= 0.5 && <EverydayModes legs={eLegs} rail={eRail} walk={eWalk} day={day} />}
      <div class="h">By train<span class="right muted small" data-tip={repTip}>Inter-city, {period.toLowerCase()}</span></div>
      {t.why || !main || !tm ? <p class="muted">{noStation || NO_RAIL[t.why || "noRail"]}</p> : (
        <>
          <Legs t={t} r={main} />
          <ul class="rows">
            <li data-tip={`Base fare $${(t.baseFarePerceived / t.minPerUsd).toFixed(2)} once a trip, plus the distance fare`}>
              <span class="grow">Fare</span>
              <span class="val">${fare.toFixed(2)}</span>
            </li>
            <li data-tip={`Station to station: ${[["wait", tm.wait], ["ride", tm.ride], ["walk between stations", tm.xfer]].filter(([, v]) => (v as number) > 0).map(([l, v]) => `${l} ${num1(v as number)}`).join(" + ")} = ${Math.round(tm.total)} min`}>
              <span class="grow">Time</span>
              <span class="val">{minText(tm.total)}</span>
            </li>
            <li data-tip={`The minutes the model weighs against the car and the plane, door to door, averaged over the area pairs the train serves (${pctShare(t.served)} of the trips)`}>
              <span class="grow">Effective time</span>
              <span class="val">{minText(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small" data-tip={`The train riders spread over ${t.pairs.length} pairs of stations near the two ends`}>
            {routeShares[0] >= 0.995 ? "Every train user on these trips takes this route." : `${pctShare(routeShares[0])} of the train users on these trips take this route.`}
          </p>
          <OtherRoutes t={t} shown={routeShares} riders={riders} />
        </>
      )}
      {known && (
        <>
          <div class="h">By car</div>
          {Number.isFinite(t.carPerceived)
            ? <CarRows total={t.carPerceived} parts={carParts} where={closed || m.unbuilt ? "Between the two places' centres" : "Averaged over the 6 km areas at the two ends, by their trips"} />
            : <ul class="rows"><li><span class="grow muted">No road joins the two ends.</span></li></ul>}
        </>
      )}
      {corridor?.flight && (
        <>
          <div class="h">By plane</div>
          <PlaneRows f={corridor.flight} km={corridor.km} />
        </>
      )}
      {!closed && inPlay && (
        <PeriodRows
          train={(i) => (m.periods[i] ? trainShare(m.periods[i]!, ways) : null)}
          tip={(i) => { const x = m.periods[i]; return x ? `${PERIODS[i].name}: ${exact(ways * x.trips)} inter-city trips, train ${pctShare(trainShare(x, ways))}` : PERIODS[i].name; }} />
      )}
    </>
  );
}
