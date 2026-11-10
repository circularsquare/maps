// T-866: the trip explorer between zones. The inter-city trips between a bubble selection (or the
// trip tool's start) and a far place or 6 km area, as the regional market chooses them
// (game/trip.ts `MarketPeriod`): every area pair between the two ends added up by its trips. The
// train's routes are the market's own through the period's strategies; the walks or drives to the
// stations are known only as felt minutes, averaged over the area.

import { activeCity, cityById } from "../../game/cities";
import { count, distance, distUnit, exact, moneyWhole, perDistance, roughDistance, smallMoney, speed, usingUsd } from "../../game/format";
import {
  AIR_FARE_USD, AIR_FIXED_MIN, AIR_KMH, AIR_NO_DIRECT_MIN, AIR_TERMINAL_MIN, AIR_USD_PER_KM, AIR_WAIT_MIN, CAR_KMH, CAR_SHARED_BY, carPartsOf, corridorOf,
  crossingsOf, DRIVE_FIXED_MIN, driveOf, loadMarketRows, marketRow, marketRowsReady, pairCarUsd, ROAD_DETOUR, travelWeight, type CarCost, type Crossing, type Drive, type FlightCost,
} from "../../game/marketRule";
import { fareScale } from "../../game/income";
import { moneyState } from "../../game/money";
import { modeColours } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { eachCorridor, intercityDemand, intercityPlaceName } from "../../game/demand";
import { peopleOf, tripPeriod, wholePeople, type MarketEnd, type MarketPeriod, type MarketShown, type TripRoute } from "../../game/trip";
import { shownDayPhrase } from "../../game/ui";
import { linesById, stationNames, world } from "../../game/world";
import { lowerWord, t as tr, tRich, tSeries } from "../../i18n";
import { shownPeriod } from "../../map/tripView";
import { SegBar } from "../widgets";
import { legRows, PeriodRows, routeRow, StationLink } from "./RouteParts";
import { equation, minNum1, minRound, minText, modeName, num1, pct1, pctShare, rideTip, roundTo, termLabel, usd, type ModeId, type Term } from "./tripText";

// T-2798: cents in dollars, as ever; another currency says the price
const perKm = (usdPerKm: number) => usingUsd()
  ? tr("marketTrip.centsPer", { cents: Math.round(100 * perDistance(usdPerKm)), unit: distUnit() })
  : tr("marketTrip.pricePer", { amount: smallMoney(perDistance(usdPerKm)), unit: distUnit() });

/** T-1148: the crossings' rows. */
const crossingLabel = (c: Crossing) => tr("marketTrip.crossingLabel", { kind: c.kind });
const crossingTip = (c: Crossing) => tr("marketTrip.crossingTip", {
  kind: c.kind, usd: moneyWhole(c.kind === "ferry" ? Math.round(c.usd) : c.usd), people: CAR_SHARED_BY, min: c.min, hours: c.min / 60,
  distance: distance(1000 * (c.km ?? 0)), wait: minText(c.min),
});

/** The car's effective time as an equation, its parts in the order of the rows. */
function carEquation(c: CarCost): string {
  const terms: Term[] = [
    { label: termLabel("drive"), real: num1(c.driveMin), weight: 1, perceived: c.driveMin },
    { label: termLabel("running"), real: usd(c.runUsd), weight: c.w, perceived: c.runUsd * c.w, usd: true },
    ...(c.parkMin >= 0.05 ? [{ label: termLabel("parking"), real: "", weight: 1, perceived: c.parkMin }] : []),
    // i18n-ignore: term ids
    ...c.crossings.flatMap((x) => [
      // i18n-ignore: term id
      { label: termLabel("crossing fare"), real: usd(x.usd / CAR_SHARED_BY), weight: c.w, perceived: (x.usd / CAR_SHARED_BY) * c.w, usd: true },
      // i18n-ignore: term id
      { label: termLabel("crossing time"), real: num1(x.min), weight: 1, perceived: x.min },
    ]),
  ];
  return `${equation(terms, c.total)}\n${tr("marketTrip.dollarWeight", { amount: smallMoney(1), w: num1(c.w) })}`;
}

/** By car: the drive, its running costs, parking, any crossing, and the effective time. `parts`
 * null: the effective time alone (a group of far places with different crossings). */
function CarRows({ total, parts, where }: { total: number; parts: CarCost | null; where: string }) {
  const c = parts;
  return (
    <ul class="rows">
      {c && (
        <>
          <li data-tip={c.routed
            ? tr("marketTrip.driveRouted", { road: distance(1000 * c.roadKm) })
            : tr("marketTrip.driveRule", { road: distance(1000 * c.roadKm), speed: speed(CAR_KMH), detour: ROAD_DETOUR, fixed: DRIVE_FIXED_MIN })}>
            <span class="grow">{tr("tripText.drive")}</span>
            <span class="val">{minText(c.driveMin)}</span>
          </li>
          <li data-tip={tr("marketTrip.runningTip", { rate: perKm(c.cu), people: CAR_SHARED_BY })}>
            <span class="grow">{tr("marketTrip.running")}</span>
            <span class="val">{usd(c.runUsd)}</span>
          </li>
          {c.parkMin >= 0.5 && (
            <li data-tip={tr("marketTrip.parkingTip", { people: CAR_SHARED_BY })}>
              <span class="grow">{tr("tripText.parking")}</span>
              <span class="val">{minText(c.parkMin)}</span>
            </li>
          )}
          {c.crossings.map((x) => (
            <li data-tip={crossingTip(x)}>
              <span class="grow">{crossingLabel(x)}</span>
              <span class="val">{usd(x.usd / CAR_SHARED_BY)}, {minText(x.min)}</span>
            </li>
          ))}
        </>
      )}
      <li data-tip={c ? `${carEquation(c)}\n${where}` : where}>
        <span class="grow">{tr("tripText.effectiveTime")}</span>
        <span class="val">{minText(total)}</span>
      </li>
    </ul>
  );
}

/** By plane: the drives to and from the airports, the time at them, the flight, the fare and the
 * effective time. */
function PlaneRows({ f: f0, km, w }: { f: FlightCost; km: number; w: number }) {
  // T-1902: the fare at the travellers' weight on money, as the choice weighs it
  const f = { ...f0, fw: w, total: f0.min + f0.fareUsd * w };
  // i18n-ignore: term ids
  const terms: Term[] = [
    // i18n-ignore: term id
    { label: termLabel("airport drives"), real: num1(f.accessFrom + f.accessTo), weight: 1, perceived: f.accessFrom + f.accessTo },
    // i18n-ignore: term id
    { label: termLabel("at the airports"), real: num1(AIR_TERMINAL_MIN), weight: 1, perceived: AIR_TERMINAL_MIN },
    // i18n-ignore: term id
    { label: termLabel("until the flight"), real: num1(AIR_WAIT_MIN), weight: 1, perceived: AIR_WAIT_MIN },
    { label: termLabel("flight"), real: num1(f.flightMin), weight: 1, perceived: f.flightMin },
    { label: termLabel("fare"), real: usd(f.fareUsd), weight: Math.round(f.fw * 10) / 10, perceived: f.fareUsd * f.fw, usd: true },
  ];
  return (
    <ul class="rows">
      <li data-tip={tr("marketTrip.airportDrivesTip", { from: minText(f.accessFrom), to: minText(f.accessTo) })}>
        <span class="grow">{tr("marketTrip.airportDrives")}</span>
        <span class="val">{minText(f.accessFrom + f.accessTo)}</span>
      </li>
      <li data-tip={tr("marketTrip.atAirportsTip")}>
        <span class="grow">{tr("marketTrip.atAirports")}</span>
        <span class="val">{minText(AIR_TERMINAL_MIN)}</span>
      </li>
      <li data-tip={tr("marketTrip.untilFlightTip")}>
        <span class="grow">{tr("marketTrip.untilFlight")}</span>
        <span class="val">{minText(AIR_WAIT_MIN)}</span>
      </li>
      <li data-tip={tr("marketTrip.flightTip", { distance: distance(1000 * km), speed: speed(AIR_KMH), fixed: AIR_FIXED_MIN, direct: f.direct, oneHour: AIR_NO_DIRECT_MIN === 60, minutes: AIR_NO_DIRECT_MIN })}>
        <span class="grow">{tr("marketTrip.flight")}</span>
        <span class="val">{minText(f.flightMin)}</span>
      </li>
      <li data-tip={tr("marketTrip.airFareTip", { fare: moneyWhole(AIR_FARE_USD), rate: perKm(AIR_USD_PER_KM) })}>
        <span class="grow">{tr("tripText.fare")}</span>
        <span class="val">{usd(f.fareUsd)}</span>
      </li>
      <li data-tip={`${equation(terms, f.total)}\n${tr("marketTrip.airDollarWeight", { amount: smallMoney(1), w: num1(f.fw) })}`}>
        <span class="grow">{tr("tripText.effectiveTime")}</span>
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

/** The bar and rows of a market panel's modes: `parts` are mode, share and colour. `of` finishes each hover. */
function ModeRows({ parts, of }: { parts: [ModeId, number, string][]; of: string }) {
  const modeOf = (m: ModeId, v: number) => tr("tripText.modeOf", { mode: modeName(m), share: pctShare(v), of });
  return (
    <div class="trip-modes">
      <SegBar kind="trip-bar" label={tSeries(parts.map(([m, v]) => tr("tripText.modeShare", { mode: modeName(m), share: pctShare(v) })))} parts={parts.map(([m, v, c]) => ({ share: v, colour: c, tip: modeOf(m, v) }))} />
      <ul class="rows">
        {parts.map(([m, v, c]) => (
          <li data-tip={modeOf(m, v)}>
            <i class="mk" style={{ background: c }} />
            <span class="grow">{modeName(m)}</span>
            <span class="val w-num">{pctShare(v)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** T-1148: the two places' trips by mode, under the selection's own. */
function PairModes({ pair, near, far, day }: { pair: { trips: number; rail: number; air: number; car: number }; near: number; far: number; day: string }) {
  const n = peopleOf(pair.trips);
  const model = [pair.rail / pair.trips, pair.air / pair.trips, pair.car / pair.trips];
  const whole = wholePeople(model, n).map((p) => p / n);
  const ends = { a: intercityPlaceName(near), b: intercityPlaceName(far), day };
  const c = modeColours();
  const parts = ([["train", whole[0], c.train], ["plane", whole[1], c.other], ["car", whole[2], c.drive]] as [ModeId, number, string][]).filter(([m, v]) => m === "train" || v > 0);
  return (
    <>
      <div class="h">{tr("marketTrip.betweenPlaces")}<span class="right muted small" data-tip={tr("marketTrip.pairTip", { amount: exact(pair.trips), ...ends })}>{tr("marketTrip.tripsCount", { amount: count(pair.trips) })}</span></div>
      <ModeRows parts={parts} of={tr("marketTrip.pairOf", { amount: count(pair.trips), ...ends })} />
    </>
  );
}

/** T-1229: the everyday outings across a zone line between the two ends, by train, on foot and by
 * car, over the day as the solve split them. */
function EverydayModes({ legs, rail, walk, day }: { legs: number; rail: number; walk: number; day: string }) {
  const model = [rail / legs, walk / legs, Math.max(0, legs - rail - walk) / legs];
  const n = peopleOf(legs);
  const whole = wholePeople(model, n).map((p) => p / n);
  const c = modeColours();
  const parts = ([["train", whole[0], c.train], ["walking", whole[1], c.walk], ["car", whole[2], c.drive]] as [ModeId, number, string][]).filter(([, v]) => v > 0);
  return (
    <>
      <div class="h">{tr("marketTrip.everyday")}<span class="right muted small" data-tip={tr("marketTrip.everydayTip", { amount: exact(legs), day })}>{tr("marketTrip.legsCount", { amount: count(legs) })}</span></div>
      <ModeRows parts={parts} of={tr("marketTrip.everydayOf", { amount: count(legs), day })} />
    </>
  );
}

/** T-880: the commutes between zones between the two ends, by train and by car, over the day. */
function CommuteModes({ legs, rail, day, who }: { legs: number; rail: number; day: string; who: string }) {
  const model = [rail / legs, Math.max(0, legs - rail) / legs];
  const n = peopleOf(legs);
  const whole = wholePeople(model, n).map((p) => p / n);
  const c = modeColours();
  const parts = ([["train", whole[0], c.train], ["car", whole[1], c.drive]] as [ModeId, number, string][]).filter(([m, v]) => m === "train" || v > 0);
  return (
    <>
      <div class="h">{tr("marketTrip.commutes")}<span class="right muted small" data-tip={tr("marketTrip.commuteTip", { amount: exact(legs), day, who })}>{tr("marketTrip.legsCount", { amount: count(legs) })}</span></div>
      <ModeRows parts={parts} of={tr("marketTrip.commuteOf", { amount: count(legs), day, who })} />
    </>
  );
}

/** The crossings and the drive between the near place and every far one, or null when they differ
 * (or a row is missing). */
function crossingSet({ near, fars }: { near: number; fars: number[] }): { crossings: Crossing[]; drive: Drive } | null {
  const a = marketRow(near);
  if (!a) return null;
  let out: { crossings: Crossing[]; drive: Drive } | null = null;
  const key = (c: Crossing[]) => c.map((x) => `${x.kind}${x.km ?? ""}`).join();
  for (const f of fars) {
    const b = marketRow(f);
    if (!b) return null;
    const c = crossingsOf(a, b), drive = driveOf(near, f);
    if (out && (key(out.crossings) !== key(c) || out.drive.rt !== drive.rt || out.drive.rk !== drive.rk)) return null;
    out = { crossings: c, drive };
  }
  return out;
}

/** The near end's place, and the far ends' places. */
function placesOf(m: MarketShown): { near: number; fars: number[] } {
  return { near: m.near?.place ?? cityById(activeCity.value).place, fars: [...new Set((m.fars ?? [m.far]).map((f) => f.place))] };
}

/** The far end as the panel names it. */
export function marketFarName(m: MarketShown): string {
  const fars = m.fars ?? [m.far];
  // T-885: a group: "3 parts of Philadelphia", "Philadelphia and 2 other places"
  if (fars.length > 1) {
    const places = [...new Set(fars.map((f) => f.place))];
    if (places.length === 1) return tr("marketTrip.partsOf", { count: fars.length, place: intercityPlaceName(places[0]) });
    return tr("marketTrip.andOthers", { place: intercityPlaceName(fars[0].place), count: fars.length - 1, whole: fars.every((f) => f.gid < 0) });
  }
  return marketEndName(m.far);
}

/** A market place or one of its 6 km zones, as the panel names it. */
export function marketEndName(e: MarketEnd): string {
  const place = intercityPlaceName(e.place);
  return e.gid >= 0 ? tr("marketTrip.partOf", { place }) : place;
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
  const name = (id: string) => st.get(id) ?? tr("tripText.aStation");
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={tr("marketTrip.accessTip", { min: minNum1(pair.accessPerceived) })}>
          <span class="grow">{tRich("marketTrip.to", { station: <StationLink id={first.from} name={name(first.from)} /> })}</span>
          <span class="val muted">{minText(pair.accessPerceived)}</span>
        </li>
      )}
      {legRows(r, name, ln, (l) => rideTip(l, w.lineStats[l.line]))}
      {pair && last && (
        <li data-tip={tr("marketTrip.egressTip", { min: minNum1(pair.egressPerceived) })}>
          <span class="grow">{tRich("marketTrip.from", { station: <StationLink id={last.to} name={name(last.to)} /> })}</span>
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
      <div class="h">{tr("tripText.otherRoutes")}<span class="right muted small">{tr("tripText.trainTrips")}</span></div>
      <ul class="rows">
        {others.map(([r, v]) => {
          const { names, from, to, label } = routeRow(r, st, ln);
          return (
            <li data-tip={tr("marketTrip.routeTip", { names, from, to, share: pct1(r.share), counted: riders ? tr("tripText.countedHere", { part: pctShare(v), riders: count(riders) }) : "", time: minText(stationTimes(r).total) })}>
              {label}
              <span class="val w-num">{pctShare(v)}</span>
            </li>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted">
            <span class="grow">{tr("tripText.smallerRoutes")}</span>
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
  // T-880: the commutes between the zones too, legs a day both ways, so the total is the bubble's
  const cLegs = m.periods.reduce((s, p) => s + (p?.commute.legs ?? 0), 0), cRail = m.periods.reduce((s, p) => s + (p?.commute.rail ?? 0), 0);
  const total = all + eLegs + cLegs;
  const others = eLegs >= 0.5 || cLegs >= 0.5;
  // T-1148: the rows give the car's parts and the plane
  marketRowsReady.value;
  void loadMarketRows();
  if (!t) return <p class="muted">{tr("cityInspector.calculating")}</p>;
  // T-1148: a train in play in some period; with none the bar and rows leave the train out
  const inPlay = !m.unbuilt && m.periods.some((p) => p && !p.why);
  // T-911: in whole trips; the hovers keep the model's shares
  const model = all > 0 ? [sum("rail") / all, sum("air") / all, sum("car") / all] : [];
  const whole = wholePeople(model, peopleOf(all)).map((p) => p / peopleOf(all));
  const shares: [ModeId, number, string, number][] = all > 0
    ? ([["train", whole[0], modeColours().train, model[0]], ["plane", whole[1], modeColours().other, model[1]], ["car", whole[2], modeColours().drive, model[2]]] as [ModeId, number, string, number][]).filter(([n]) => n !== "train" || inPlay)
    : [];
  const ends = placesOf(m);
  const one = ends.fars.length === 1 ? ends.fars[0] : -1;
  // from where the trips start: the far end when the near cells are their destinations
  const corridor = one >= 0 && marketRowsReady.value ? (m.end === "work" ? corridorOf(one, ends.near) : corridorOf(ends.near, one)) : null;
  // the crossings when every far place shares them; the car's parts then follow from the sim's minutes
  const cross = marketRowsReady.value ? crossingSet(ends) : null;
  const startCountry = marketRow(m.end === "work" && one >= 0 ? one : ends.near)?.country;
  const carParts = cross && Number.isFinite(t.carPerceived) ? carPartsOf(t.km, t.carPerceived, cross.crossings, cross.drive, travelWeight(startCountry), pairCarUsd(marketRow(ends.near), one >= 0 ? marketRow(one) : null)) : null;
  // T-1148: from a selection or a 6 km area, the whole places' trips as the solve split them, beside it
  const pair = one >= 0 && !(m.near && m.near.gid < 0) ? pairFigures(ends.near, one) : null;
  const bare = m.unbuilt ?? [];
  // i18n-ignore: select option names
  const noKind = bare.length === 1 ? "one" : bare.length === 2 && one >= 0 ? "either" : "many";
  const noStation = bare.length === 0 ? "" : tr("marketTrip.noStation", {
    kind: noKind, place: intercityPlaceName(bare[0]), count: bare.length - 1,
  });
  // the period's train riders in whole trips, for the route shares (none: the model's)
  const riders = Math.round(ways * t.rail) || null;
  const top = t.routes.slice(0, 5).map((r) => r.share);
  // the routes too small to list take a leftover rider last
  const routeShares = roundTo([...top, Math.max(0, 1 - top.reduce((s, x) => s + x, 0))], riders, top.length);
  const far = marketFarName(m);
  // a single home's share of its area's trips is a fraction of one
  const trips = others ? tr("marketTrip.count", { kind: "total", amount: count(total) }) : all >= 0.5 ? tr("marketTrip.count", { kind: "inter", amount: count(all) }) : tr("marketTrip.count", { kind: "under", amount: "" });
  const where = m.near ? tr("marketTrip.bothWays") : m.end === "work" ? tr("marketTrip.whereFrom", { far }) : tr("marketTrip.whereTo", { far });
  const longerTip = tr("marketTrip.longerTip", { amount: all >= 0.5 ? exact(all) : num1(all), day, where });
  const totalTip = tr("marketTrip.totalTip", {
    amount: exact(total), day, where,
    parts: tSeries(([["inter", all], ["commute", cLegs], ["everyday", eLegs]] as [string, number][]).filter(([, v]) => v >= 0.5).map(([k, v]) => tr("marketTrip.kindCount", { kind: k, amount: count(v) }))),
  });
  const commuters = m.near ? tr("marketTrip.bothWays") : m.end === "work" ? tr("marketTrip.whoWork", { far }) : tr("marketTrip.whoHome", { far });
  const of = tr("marketTrip.of", { counted: all >= 0.5, amount: count(all), day });
  const main = t.routes[0];
  const tm = main ? stationTimes(main) : null;
  // T-1277: the market prices the curve as set; a trip pays it times its starting country's scale
  const start = m.end === "work" ? one : ends.near;
  const sc = fareScale(start >= 0 ? marketRow(start)?.country : undefined, moneyState.value?.fares.scales);
  const fare = main ? (sc * (t.baseFarePerceived + main.legs.reduce((s, l) => s + l.farePerceived, 0))) / t.minPerUsd : 0;
  const periodLow = lowerWord(period);
  const repTip = tr("marketTrip.repTip", { period: periodLow, picked: tripPeriod.value !== null });
  // T-991: an end that is not open has no trips worked out, but the distance and the car are known
  // from the places' centres; with no distance at all, neither is shown
  const known = t.km > 0;
  const closed = t.why === "farClosed";
  // T-1902: what the train is weighed against, over the trips it can serve, and its share of those
  const hasCar = Number.isFinite(t.carPerceived), hasPlane = t.trips > 0 ? t.air / t.trips >= 0.005 : !!corridor?.flight;
  const rivals = hasCar && hasPlane ? "both" : hasCar ? "car" : "plane";
  const cities = m.near && m.near.gid < 0 && (m.fars ?? [m.far]).every((f) => f.gid < 0);
  const tripsDay = tr("tripInspector.tripsDay", { trips, day });
  // i18n-ignore: term ids
  const stationTerms = ([["wait", tm?.wait ?? 0], ["ride", tm?.ride ?? 0], ["walk between stations", tm?.xfer ?? 0]] as [string, number][]).filter(([, v]) => v > 0).map(([l, v]) => `${termLabel(l)} ${num1(v)}`).join(" + ");
  return (
    <>
      {closed ? known && (
        <p class="muted small" data-tip={tr("marketTrip.straightCenters")}>{tr("tripText.apart", { distance: roughDistance(1000 * t.km) })}</p>
      ) : (
        <p class="muted small" data-tip={others ? totalTip : longerTip}>
          {known ? tr("tripText.apartAfter", { head: tripsDay, distance: roughDistance(1000 * t.km) }) : tripsDay}
        </p>
      )}
      {!closed && others && all >= 0.5 && (
        <div class="h">{tr("marketTrip.interCity")}<span class="right muted small" data-tip={longerTip}>{tr("marketTrip.tripsCount", { amount: count(all) })}</span></div>
      )}
      {shares.length > 0 && (
        <div class="trip-modes">
          <SegBar kind="trip-bar" label={tSeries(shares.map(([n, v]) => tr("tripText.modeShare", { mode: modeName(n), share: pctShare(v) })))}
            parts={shares.map(([n, v, c]) => ({ share: v, colour: c, tip: tr("tripText.modeOf", { mode: modeName(n), share: pctShare(v), of }) }))} />
          <ul class="rows">
            {shares.filter(([n, v]) => n !== "plane" || v > 0).map(([n, v, c, e]) => (
              <li data-tip={`${tr("tripText.modeOf", { mode: modeName(n), share: pctShare(v), of })}\n${tr("tripText.modelLine", { share: pct1(e) })}`}>
                <i class="mk" style={{ background: c }} />
                <span class="grow">{modeName(n)}</span>
                <span class="val w-num">{pctShare(v)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {pair && <PairModes pair={pair} near={ends.near} far={one} day={day} />}
      {!closed && cLegs >= 0.5 && <CommuteModes legs={cLegs} rail={cRail} day={day} who={commuters} />}
      {!closed && eLegs >= 0.5 && <EverydayModes legs={eLegs} rail={eRail} walk={eWalk} day={day} />}
      <div class="h">{tr("tripText.byTrain")}<span class="right muted small" data-tip={others ? tr("marketTrip.trainOthersTip", { period: periodLow, picked: tripPeriod.value !== null }) : repTip}>{others ? period : tr("marketTrip.interCityPeriod", { period: periodLow })}</span></div>
      {t.why || !main || !tm ? <p class="muted">{noStation || tr("marketTrip.noRail", { why: t.why || "noRail" })}</p> : (
        <>
          <Legs t={t} r={main} />
          <ul class="rows">
            <li data-tip={tr("tripText.baseFare", { fare: usd((sc * t.baseFarePerceived) / t.minPerUsd), xfer: false, xferFare: "" })}>
              <span class="grow">{tr("tripText.fare")}</span>
              <span class="val">{usd(fare)}</span>
            </li>
            <li data-tip={tr("marketTrip.stationTip", { terms: stationTerms, total: minRound(tm.total) })}>
              <span class="grow">{tr("tripText.time")}</span>
              <span class="val">{minText(tm.total)}</span>
            </li>
            {t.served > 0 && (
              <li>
                <span class="grow">{tr("marketTrip.possible")}</span>
                <span class="val w-num">{pctShare(t.served)}</span>
              </li>
            )}
            {Number.isFinite(t.railPerceived) && (
              <li>
                <span class="grow">{tr("tripText.effectiveTime")}</span>
                <span class="val">{minText(t.railPerceived)}</span>
              </li>
            )}
            {Number.isFinite(t.railPerceived) && Number.isFinite(t.otherPerceived) && (
              <li data-tip={tr("marketTrip.rivalTip", { rival: rivals })}>
                <span class="grow">{tr("marketTrip.rivalRow", { rival: rivals })}</span>
                <span class="val">{minText(t.otherPerceived)}</span>
              </li>
            )}
          </ul>
          <p class="muted small" data-tip={tr("marketTrip.pairsTip", { n: t.pairs.length })}>
            {routeShares[0] >= 0.995 ? tr("marketTrip.everyUser", { cities }) : tr("marketTrip.shareUsers", { share: pctShare(routeShares[0]), cities })}
          </p>
          <OtherRoutes t={t} shown={routeShares} riders={riders} />
        </>
      )}
      {known && (
        <>
          <div class="h">{tr("tripText.byCar")}</div>
          {Number.isFinite(t.carPerceived)
            ? <CarRows total={t.carPerceived} parts={carParts} where={tr(closed || m.unbuilt ? "marketTrip.whereCenters" : "marketTrip.whereAveraged")} />
            : <ul class="rows"><li><span class="grow muted">{tr("marketTrip.noRoad")}</span></li></ul>}
        </>
      )}
      {corridor?.flight && (
        <>
          <div class="h">{tr("marketTrip.byPlane")}</div>
          <PlaneRows f={corridor.flight} km={corridor.km} w={travelWeight(startCountry)} />
        </>
      )}
      {!closed && inPlay && all >= 0.5 && (
        <PeriodRows
          train={(i) => (m.periods[i] ? trainShare(m.periods[i]!, ways) : null)}
          tip={(i) => { const x = m.periods[i]; return x ? tr("marketTrip.periodTip", { period: PERIODS[i].name, amount: exact(ways * x.trips), share: pctShare(trainShare(x, ways)) }) : PERIODS[i].name; }} />
      )}
    </>
  );
}
