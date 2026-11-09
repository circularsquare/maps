// The trip explorer's inspector (T-195, docs/ui-inspectors.md): the trips between two places by
// mode as the view counts them (T-586), then one representative trip as the demand model decides
// it, for one period: the train's main route with its walks, waits, rides and changes, the other
// routes the train riders spread over, and the drive. Every number is the model's (game/trip.ts);
// every drawn quantity has a hover with it.

import { Fragment } from "preact";
import { capitalise, count, distance, distUnit, exact, speed } from "../../game/format";
import { modeColours } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { purposeName, purposeWords, type PurposeFilter } from "../../game/purposes";
import { WALK_KMH } from "../../game/walks";
import { CAR_MIN_PER_USD, countsTrain, openRoute, peopleOf, routeCrowding, routeLabel, routeTerms, routeTotals, stationName, tripOpenedFrom, tripPeriod, tripView, wholePeople, type TripMix, type TripPeriod, type TripRoute } from "../../game/trip";
import { purposeFilter, selection, shownDayPhrase } from "../../game/ui";
import { linesById, stationNames, world } from "../../game/world";
import { lowerWord, t as tr, tRich, tSeries } from "../../i18n";
import { shownPeriod } from "../../map/tripView";
import { Actions, SegBar } from "../widgets";
import { MarketTrip } from "./MarketTrip";
import { legRows, PeriodRows, routeRow, StationLink } from "./RouteParts";
import { equation, minFix1, minNum1, minRound, minText, modeName, noRoad, num1, pct1, pctShare, rideTip, roundTo, sumLine, termLabel, tripsAmount, tripWords, usd, type ModeId, type Term } from "./tripText";

/** T-910: why the train is not offered. Too long a way to and from the trains says which end
 * holds it back: people drive to a station from home on trips of `rideMinKm` or more, and walk
 * from one at the far end. T-2265: the train takes longer than walking, in real minutes. */
function NoRail({ t }: { t: TripPeriod }) {
  if (t.why === "walkFurther") {
    const tip = tr("tripInspector.walkFurtherTip", { known: Number.isFinite(t.trainQuickestMin), train: minText(t.trainQuickestMin), walk: minText(t.walkMin) });
    return <p class="muted" data-tip={tip}>{tr("tripInspector.walkFurther")}</p>;
  }
  if (t.why !== "walksTooLong") return <p class="muted">{tr("tripInspector.noRail", { why: t.why })}</p>;
  const a = t.walkStartPerceived, b = t.walkEndPerceived;
  const text = tr("tripInspector.walksTooLong", { why: t.rideToStation === "short" ? "short" : t.rideToStation === "none" ? "none" : b >= a ? "found" : "byCar", far: Number.isFinite(a) });
  const how = !Number.isFinite(a) ? "none" : t.rideToStation === "tried" ? "footOrCar" : "foot";
  const tip = tr("tripInspector.noRailTip", { how, start: minText(a), end: minText(b), max: minRound(t.pairWalkMax) }) +
    (t.rideToStation === "short" ? "\n" + tr("tripInspector.driveOnlyFar", { km: num1(t.rideMinKm) }) : "");
  return <p class="muted" data-tip={tip}>{text}</p>;
}

/** "trip" or "trips" after a count. */
const trips = (n: number, f: PurposeFilter) => tripsAmount(count(n), n, f);

/** The effective time hover (T-699), the train's: the terms are the route's (the worker's perceived
 * minutes); the model's figure pools every station and route people use, so it is lower than the
 * main route's sum by a last term. `pooled` false: the route's own sum alone (T-870). */
function trainEquation(t: TripPeriod, r: TripRoute, pooled = true): string {
  const terms: Term[] = routeTerms(t, r).filter((x) => x.perceived !== 0 || x.unit === "usd").map((x) => ({
    label: termLabel(x.id),
    real: x.unit === "usd" ? usd(x.real) : x.real > 0 ? num1(x.real) : "",
    weight: x.weight,
    perceived: x.perceived,
    ...(x.unit === "usd" ? { usd: true } : {}),
  }));
  if (!pooled) return equation(terms);
  const rest = t.railPerceived - terms.reduce((s, x) => s + x.perceived, 0);
  // i18n-ignore: term id
  if (Math.abs(rest) >= 0.05) terms.push({ label: termLabel("choice of routes"), real: "", weight: 1, perceived: rest });
  return equation(terms, t.railPerceived) + (Math.abs(rest) >= 0.05 ? "\n" + tr("tripInspector.choiceOfRoutes") : "");
}

/** The real minutes of a route as an equation: "Walk 8 + wait 3 + ride 6 = 17 min". */
function timeEquation(tot: ReturnType<typeof routeTotals>): string {
  // i18n-ignore: term ids
  const parts: [string, number][] = [["drive", tot.driveMin], ["walk", tot.walkMin], ["wait", tot.wait], ["ride", tot.ride], ["walk between stations", tot.xfer]];
  return sumLine(parts.filter(([, m]) => m > 0).map(([l, m]) => `${termLabel(l)} ${num1(m)}`).join(" + "), tot.total);
}

/** Walks over this straight-line distance are not offered by the model. */
const WALK_REACH_M = 10000;

/** T-628, T-992: the headline gets a walk row when walking the whole way takes under this, real
 * minutes, or whatever it takes when walking has a share. */
const WALK_ROW_MIN = 30;

/** T-1148: what the model's walking and driving stand for (demand-plain.md). */
const STANDS_FOR: Partial<Record<ModeId, "tripInspector.standsWalk" | "tripInspector.standsDrive" | "tripInspector.standsNone">> = {
  walk: "tripInspector.standsWalk",
  drive: "tripInspector.standsDrive",
  // T-2215 (notes/T-2207.md 4)
  none: "tripInspector.standsNone",
};

/** The headline: the trips between the two places by mode as the view counts them (T-586), or,
 * when the view counts none, the representative trip's chances in the period shown. */
function ModeBar({ t, mix, of }: { t: TripPeriod; mix: TripMix | null; of: string }) {
  // T-2215: between two land parts no road joins, the counted walking and driving have no way (the
  // far end's cells still split them as walk and drive, T-2256)
  const m = mix ? { rail: mix.rail, walk: t.apart ? 0 : mix.walk, drive: t.apart ? 0 : mix.drive, none: t.apart ? mix.walk + mix.drive : 0 } : null;
  const n = m ? m.rail + m.walk + m.drive + m.none : 0;
  const model = n > 0 ? [m!.rail / n, m!.walk / n, m!.drive / n, m!.none / n] : [t.pTrain, t.pWalk, t.pDrive, t.pNone];
  // T-911: counted trips are whole people; the representative trip's chances stay as they are
  const [train, walk, drive, none] = roundTo(model, peopleOf(n));
  type Mode = "train" | "walk" | "drive" | "none";
  const parts: [Mode, number, string][] = [["train", train, "seg-train"], ["walk", walk, "seg-walk"], ["drive", drive, "seg-drive"],
    ...(t.apart || none > 0 ? [["none", none, "seg-none"] as [Mode, number, string]] : [])];
  const walkTip = tr("tripInspector.walkClause", { min: minText(t.walkMin) });
  const note = n > 0 ? "" : groupNote(t);
  const modelTip = (i: number) => (n > 0 ? "\n" + tr("tripText.modelLine", { share: pct1(model[i]) }) : "");
  const stands = (m: Mode) => (STANDS_FOR[m] ? "\n" + tr(STANDS_FOR[m]!) : "");
  const extra = (m: Mode) => (m === "walk" && !t.apart ? walkTip : "");
  // walking and driving between two land parts are 0: their rows go, By car and By walking say why
  const shown = (m: Mode, v: number) => (t.apart && (m === "walk" || m === "drive") ? v > 0 : m !== "walk" || v > 0 || t.walkMin < WALK_ROW_MIN);
  const modeOf = (m: Mode, share: string) => tr("tripText.modeOf", { mode: modeName(m), share, of });
  return (
    <div class="trip-modes">
      <SegBar kind="trip-bar" label={tSeries(parts.map(([m, v]) => tr("tripText.modeShare", { mode: modeName(m), share: pctShare(v) })))}
        parts={parts.map(([m, v, c]) => ({ share: v, cls: c, tip: modeOf(m, pctShare(v)) + extra(m) + stands(m) }))} />
      <ul class="rows">
        {parts.map((x, i) => [x, i] as const).filter(([[m, v]]) => shown(m, v)).map(([[m, v], i]) => (
          <li data-tip={modeOf(m, n > 0 ? pctShare(v) : pct1(v)) + extra(m) + modelTip(i) + note + stands(m)}>
            <i class="mk" style={{ background: modeColours()[m] }} />
            <span class="grow">{modeName(m)}</span>
            <span class="val w-num">{pctShare(v)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RouteLegs({ t, r }: { t: TripPeriod; r: TripRoute }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  const pair = t.pairs[r.pair];
  const first = r.legs[0], last = r.legs[r.legs.length - 1];
  const name = (id: string) => st.get(id) ?? tr("tripText.aStation");
  const walkTip = (m: number, min: number) => tr("tripInspector.walkTip", { distance: distance(m), min: minFix1(min), kmh: WALK_KMH });
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={pair.accessRide
          ? tr("tripInspector.rideAccessTip", { distance: distance(pair.accessM), min: minFix1(pair.accessMin) })
          : walkTip(pair.accessM, pair.accessMin)
            + (t.groupWalkStartMin >= 0 && Math.abs(t.groupWalkStartMin - pair.accessMin) >= 1 ? "\n" + tr("tripInspector.bubbleAccess", { min: minText(t.groupWalkStartMin) }) : "")}>
          <span class="grow">{tRich("tripInspector.accessTo", { ride: pair.accessRide, station: <StationLink id={first.from} name={name(first.from)} /> })}</span>
          <span class="val">{minText(pair.accessMin)}</span>
        </li>
      )}
      {legRows(r, name, ln, (l) => {
        // T-842: riders arriving when no train is due for hours weigh on the wait as felt
        return rideTip(l, w.lineStats[l.line]?.bus, l.fullWaitMin, l.waitPerceived - l.waitMin);
      })}
      {pair && last && (
        <li data-tip={walkTip(pair.egressM, pair.egressMin)}>
          <span class="grow">{tRich("tripInspector.walkFrom", { station: <StationLink id={last.to} name={name(last.to)} /> })}</span>
          <span class="val">{minText(pair.egressMin)}</span>
        </li>
      )}
    </ul>
  );
}

/** The fare, and on a route that starts with a drive to the station what the car costs in money
 * (T-915): its cost per trip and any paid parking at the station. */
function FareRows({ t, r }: { t: TripPeriod; r: TripRoute }) {
  const tot = routeTotals(t, r);
  const pair = t.pairs[r.pair];
  const car = pair?.accessRide ? pair.rideCar / CAR_MIN_PER_USD : 0;
  const parking = pair?.accessRide ? pair.rideStationMoney / CAR_MIN_PER_USD : 0;
  return (
    <>
      <li data-tip={tr("tripText.baseFare", { fare: usd(t.baseFarePerceived / t.minPerUsd), xfer: tot.xferUsd > 0, xferFare: tot.xferUsd > 0 ? usd(tot.xferUsd / tot.changes) : "" })}>
        <span class="grow">{tr("tripText.fare")}</span>
        <span class="val">{usd(tot.fareUsd)}</span>
      </li>
      {pair?.accessRide && (
        <li data-tip={tr("tripInspector.carCostTip", { cost: usd(car) }) + (parking >= 0.005 ? "\n" + tr("tripInspector.stationParkingTip", { cost: usd(parking) }) : "")}>
          <span class="grow">{tr("tripText.car")}</span>
          <span class="val">{usd(car + parking)}</span>
        </li>
      )}
    </>
  );
}

/** T-1070: what full trains add to a route's effective time, each row only where it adds anything. */
function CrowdRows({ r }: { r: TripRoute }) {
  const { crowding, fullWait } = routeCrowding(r);
  return (
    <>
      {crowding >= 0.05 && (
        <li data-tip={tr("tripInspector.crowdingTip", { min: minNum1(crowding) })}>
          <span class="grow">{tr("tripInspector.crowding")}</span>
          <span class="val">{minText(crowding)}</span>
        </li>
      )}
      {fullWait >= 0.05 && (
        <li data-tip={tr("tripInspector.fullWaitTip", { min: minNum1(fullWait) })}>
          <span class="grow">{tr("tripInspector.waitForRoom")}</span>
          <span class="val">{minText(fullWait)}</span>
        </li>
      )}
    </>
  );
}

/** A route's own perceived minutes, the sum of its terms (no choice of routes). */
const ownSum = (t: TripPeriod, r: TripRoute) => routeTerms(t, r).reduce((s, x) => s + x.perceived, 0);

/** T-870: an other route opened under its row: its legs, fare, time and effective time, and how
 * far it falls behind the main route's own sum, term by term in the hover. */
function RouteDetail({ t, r }: { t: TripPeriod; r: TripRoute }) {
  const tot = routeTotals(t, r);
  const own = ownSum(t, r), mainOwn = ownSum(t, t.routes[0]);
  // from the rounded figures, so the row agrees with the two shown
  const diff = Math.max(1, Math.round(own)) - Math.max(1, Math.round(mainOwn));
  // each term against the main route's, by label ("change" and "changes" are one)
  const by = (rt: TripRoute) => {
    const m = new Map<string, number>();
    for (const x of routeTerms(t, rt)) {
      const k = x.id === "changes" ? "change" : x.id;
      m.set(k, (m.get(k) ?? 0) + x.perceived);
    }
    return m;
  };
  const a = by(r), b = by(t.routes[0]);
  const parts = [...new Set([...b.keys(), ...a.keys()])]
    .map((k) => [k, (a.get(k) ?? 0) - (b.get(k) ?? 0)] as const)
    .filter(([, d]) => Math.abs(d) >= 0.05);
  const diffTip = tr("tripInspector.diffTip", { own: minText(own), main: minText(mainOwn) }) +
    (parts.length ? "\n" + parts.map(([k, d]) => tr("tripInspector.diffTerm", { term: termLabelCap(k), delta: tr("inspector.min", { n: (d > 0 ? "+" : "−") + num1(Math.abs(d)) }) })).join("\n") : "");
  return (
    <div class="trip-alt">
      <RouteLegs t={t} r={r} />
      <ul class="rows">
        <FareRows t={t} r={r} />
        <CrowdRows r={r} />
        <li data-tip={timeEquation(tot)}>
          <span class="grow">{tr("tripText.time")}</span>
          <span class="val">{minText(tot.total)}</span>
        </li>
        <li data-tip={trainEquation(t, r, false)}>
          <span class="grow">{tr("tripText.effectiveTime")}</span>
          <span class="val">{minText(own)}</span>
        </li>
        <li data-tip={diffTip}>
          <span class="grow">{tr("tripInspector.againstMain")}</span>
          <span class="val">{diff === 0 ? tr("tripInspector.same") : tr("inspector.min", { n: (diff > 0 ? "+" : "−") + Math.abs(diff) })}</span>
        </li>
      </ul>
    </div>
  );
}

const termLabelCap = (id: string) => capitalise(termLabel(id));

/** The main route's group, up to four other routes and the many small ones left (T-709), with
 * their shares of the train riders: in whole riders when `riders` (T-911; the small ones together
 * take a leftover rider last, since each of them is smaller than any listed), else the model's. */
function routeRows(t: TripPeriod, riders: number | null) {
  const st = stationNames(world.value!);
  // routes that read the same here (the same lines to a station of the same name, by different
  // platforms or walks) are one row
  const label = routeLabel(st);
  const main = t.routes[0] ? label(t.routes[0]) : "";
  const merged = new Map<string, TripRoute>();
  for (const r of t.routes.slice(1)) {
    const k = label(r);
    const m = merged.get(k);
    if (m) m.share += r.share;
    else merged.set(k, { ...r });
  }
  const others = [...merged].filter(([k]) => k !== main).sort(([, a], [, b]) => b.share - a.share).slice(0, 4);
  const all = mainShare(t);
  const rest = Math.max(0, 1 - all - others.reduce((s, [, r]) => s + r.share, 0));
  return { others, shown: roundTo([all, ...others.map(([, r]) => r.share), rest], riders, others.length + 1) };
}

function OtherRoutes({ t, riders }: { t: TripPeriod; riders: number | null }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  // T-911: a route no one takes at this count is not listed
  const { others, shown } = routeRows(t, riders);
  const rest = shown[shown.length - 1];
  const kept = others.map((o, i) => [o, shown[i + 1]] as const).filter(([, v]) => !riders || v > 0);
  if (!kept.length && rest < 0.005) return null;
  const open = openRoute.value;
  return (
    <>
      <div class="h">{tr("tripText.otherRoutes")}<span class="right muted small">{tr("tripText.trainTrips")}</span></div>
      <ul class="rows pick">
        {kept.map(([[k, r], v]) => {
          const { names, from, to, label } = routeRow(r, st, ln);
          const tot = routeTotals(t, r);
          const on = k === open;
          return (
            <Fragment key={k}>
              <li class={on ? "on" : ""} onClick={() => (openRoute.value = on ? null : k)}
                data-tip={tr("tripInspector.routeTip", { names, from, to, share: pct1(r.share), counted: riders ? tr("tripText.countedHere", { part: pctShare(v), riders: count(riders) }) : "", time: minText(tot.total) })
                  + "\n" + tr(on ? "tripInspector.clickClose" : "tripInspector.clickOpen")}>
                {label}
                <span class="val w-num">{pctShare(v)}</span>
              </li>
              {on && <li class="trip-alt-row"><RouteDetail t={t} r={r} /></li>}
            </Fragment>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted rest">
            <span class="grow">{tr("tripText.smallerRoutes")}</span>
            <span class="val w-num">{pctShare(rest)}</span>
          </li>
        )}
      </ul>
    </>
  );
}

/** The share of the trip's train users on the main route and the same trains from other platforms
 * or with other walks. */
function mainShare(t: TripPeriod): number {
  const main = t.routes[0];
  const label = routeLabel(stationNames(world.value!)), k = label(main);
  return main.share + t.routes.slice(1).filter((r) => label(r) === k).reduce((s, r) => s + r.share, 0);
}

/** How much of the trip's train riders the main route carries. */
function mainNote(t: TripPeriod, riders: number | null): string {
  const all = routeRows(t, riders).shown[0];
  if (all >= 0.995) return tr("tripInspector.everyUser");
  return tr("tripInspector.shareUsers", { share: pctShare(all) });
}

/** T-1902: the train is not offered from these two exact points, but the trips counted here
 * include train trips (the model works them out for the areas around the points): the route they
 * take, said why above it. */
function LiftedNote({ t }: { t: TripPeriod }) {
  const slower = t.why === "walkFurther";
  const tip = tr("tripInspector.liftedTip", { slower, walk: minText(t.walkMin), start: minText(t.walkStartPerceived), end: minText(t.walkEndPerceived), max: minRound(t.pairWalkMax) });
  return <p class="muted small" data-tip={tip}>{tr("tripInspector.liftedNote", { slower })}</p>;
}

/** T-720: the trip is the two places' own; the map's bubbles count it with the homes and places
 * near each end, which can differ. Empty when they agree to the shown percent. */
function groupNote(t: TripPeriod): string {
  if (Math.abs(t.pTrain - t.groupTrain) < 0.005 && Math.abs(t.pWalk - t.groupWalk) < 0.005) return "";
  return "\n" + tr("tripInspector.groupNote", { train: pctShare(t.groupTrain), apart: t.apart, none: pctShare(t.groupNone), drive: pctShare(t.groupDrive), walk: pctShare(t.groupWalk) });
}

/** T-865: an averaged trip's routes, the busiest first, by their share of the train riders over
 * every sampled pair. */
function BusiestRoutes({ t, riders }: { t: TripPeriod; riders: number | null }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  // T-911: whole train riders, as the headline counts them
  const shown = roundTo([...t.routes.map((r) => r.share), t.cut], riders, t.routes.length);
  const cut = shown[shown.length - 1];
  return (
    <>
      <div class="h">{tr("tripInspector.busiestRoutes")}<span class="right muted small">{tr("tripText.trainTrips")}</span></div>
      <ul class="rows">
        {t.routes.map((r, i) => {
          if (riders && shown[i] <= 0) return null;
          const { names, from, to, label } = routeRow(r, st, ln);
          return (
            <li data-tip={tr("tripInspector.busiestTip", { names, from, to, share: pct1(r.share), counted: riders ? tr("tripText.countedHere", { part: pctShare(shown[i]), riders: count(riders) }) : "" })}>
              {label}
              <span class="val w-num">{pctShare(shown[i])}</span>
            </li>
          );
        })}
        {cut >= 0.005 && (
          <li class="muted rest" data-tip={tr("tripInspector.outsideBusiest")}>
            <span class="grow">{tr("tripText.otherRoutes")}</span>
            <span class="val w-num">{pctShare(cut)}</span>
          </li>
        )}
      </ul>
    </>
  );
}

export function TripInspector() {
  const sel = selection.value;
  if (sel?.kind !== "trip") return null;
  const v = tripView.value;
  const q = shownPeriod.value;
  // opened by clicking a far-end bubble of a bubble selection (T-208)
  const back = tripOpenedFrom.value;
  // an end picked at a station is named (T-221)
  const fromName = stationName(sel.fromStation), toName = stationName(sel.toStation);
  const filter = purposeFilter.value;
  // T-866: between zones, from homes at one end to the far place (or the far place to them)
  const market = v?.market;
  const tripView_ = v?.trip;
  const about = tripView_ && !tripView_.outside ? tr("tripInspector.about", {
    crowd: tripView_.round > 0 ? tr("tripInspector.afterRounds", { amount: count(tripView_.round), rounds: tripView_.round }) : tr("tripInspector.beforeCrowding"),
    station: fromName || toName ? tr("tripInspector.stationEnd") : "",
  }) : undefined;
  const head = (
    <>
      <div class="title"><span class="name" data-tip={about}>{tr("tripInspector.title")}</span></div>
      {back && back.trip === sel && (
        <Actions>
          <button class="btn text" onClick={() => (selection.value = back.from)}>{tr("tripInspector.back")}</button>
        </Actions>
      )}
    </>
  );
  if (!sel.to) return head;
  if (market) return <>{head}<MarketTrip m={market} /></>;
  const trip = v?.trip;
  if (!trip) return <>{head}<p class="muted">{v?.failed ? tr("tripInspector.failed") : v?.asking ? tr("tripInspector.asking") : tr("cityInspector.calculating")}</p></>;
  if (trip.outside) return <>{head}<p class="muted">{tr("tripInspector.outside")}</p></>;
  const t = trip.periods[q];
  if (!t) return <>{head}<p class="muted">{tr("cityInspector.calculating")}</p></>;
  const main = t.routes[0];
  const tot = main ? routeTotals(t, main) : null;
  const pair = main ? t.pairs[main.pair] : null;  // T-586: the trips the view counts between the two places, and the trip the routes are read for
  const mix = v!.mix;
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const day = shownDayPhrase.value;
  const period = PERIODS[q].name;
  const periodLow = lowerWord(period);
  const of = n >= 0.5 ? tr("tripInspector.ofTrips", { trips: trips(n, filter), day }) : n > 0 ? tr("tripInspector.ofTrips", { trips: tripWords(filter), day }) : tr("tripInspector.ofPeople", { period: periodLow });
  // T-911: the train riders the headline counts, in whole trips, for the route shares (none: the model's)
  const riders = n > 0 ? wholePeople([mix!.rail, mix!.walk, mix!.drive], peopleOf(n))[0] || null : null;
  const trainCounted = countsTrain(mix);
  const countTip = n > 0 ? [tr("tripInspector.countTip", { amount: exact(n), trips: tripWords(filter), day }), ...(filter === "all" ? mix!.purposes.filter((p) => p.rail + p.walk + p.drive >= 0.5).map((p) => tr("tripText.labelValue", { label: purposeName(p.key), value: exact(p.rail + p.walk + p.drive) })) : [])].join("\n") : tr("tripInspector.straightLine", { distance: distance(t.distM) });
  const words = purposeWords(v!.purpose);
  const picked = tripPeriod.value !== null;
  const avg = (t.sampled ?? 0) > 1;
  // T-865: the busiest bubbles of each group, each to each, weighted by their trips
  const avgTip = avg ? tr("tripInspector.avgTip", { n: t.sampled ?? 0 }) : "";
  const repTip = tr("tripInspector.repTip", { avg, words, n: t.sampled ?? 0, period: periodLow, all: filter === "all", picked });
  const periodTip = (x: TripPeriod, i: number) => tr("tripInspector.periodTip", { period: PERIODS[i].name, words, train: pctShare(x.pTrain), apart: x.apart, none: pctShare(x.pNone), drive: pctShare(x.pDrive), walk: pctShare(x.pWalk) }) + groupNote(x);
  return (
    <>
      {head}
      <p class="muted small" data-tip={countTip}>{tr("tripText.apartAfter", { head: n >= 0.5 ? tr("tripInspector.tripsDay", { trips: trips(n, filter), day }) : n > 0 ? tr("tripInspector.underOne", { trips: trips(1, filter), day }) : period, distance: distance(t.distM) })}</p>
      <ModeBar t={t} mix={mix} of={of} />
      <div class="h">{tr("tripText.byTrain")}<span class="right muted small" data-tip={repTip}>{tr("tripInspector.purposePeriod", { purpose: purposeName(v!.purpose), period: periodLow })}</span></div>
      {!t.railOffered && !(t.lifted && trainCounted) ? <NoRail t={t} /> : avg ? (
        <>
          <ul class="rows">
            <li data-tip={avgTip + (t.railSampled !== t.sampled ? tr("tripInspector.railOfferedFor", { n: t.railSampled }) : "")}>
              <span class="grow">{tr("tripText.effectiveTime")}</span>
              <span class="val">{minText(t.railPerceived)}</span>
            </li>
          </ul>
          {t.routes.length > 0 && <BusiestRoutes t={t} riders={riders} />}
        </>
      ) : main && tot && pair ? (
        <>
          {!t.railOffered && <LiftedNote t={t} />}
          <RouteLegs t={t} r={main} />
          <ul class="rows">
            <FareRows t={t} r={main} />
            <CrowdRows r={main} />
            <li data-tip={timeEquation(tot)}>
              <span class="grow">{tr("tripText.time")}</span>
              <span class="val">{minText(tot.total)}</span>
            </li>
            <li data-tip={trainEquation(t, main)}>
              <span class="grow">{tr("tripText.effectiveTime")}</span>
              <span class="val">{minText(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small">{mainNote(t, riders)}</p>
          <OtherRoutes t={t} riders={riders} />
        </>
      ) : <p class="muted">{tr("tripText.noRoute")}</p>}
      <div class="h">{tr("tripText.byCar")}</div>
      {t.apart ? <p class="muted">{noRoad()}</p> : <ul class="rows">
        <li data-tip={tr("tripInspector.driveTip", { road: distance(1000 * t.driveRoadKm), line: distance(t.driveDistM), speed: speed(t.driveRoadKm / (Math.max(t.driveMin, 0.1) / 60)) }) + (avg ? `\n${avgTip}` : "")}>
          <span class="grow">{tr("tripText.drive")}</span>
          <span class="val">{minText(t.driveMin)}</span>
        </li>
        <li data-tip={tr("tripInspector.parkingTip")}>
          <span class="grow">{tr("tripText.parking")}</span>
          <span class="val">{minText(t.parkPerceived)}</span>
        </li>
        {t.carOwnPerceived > 0 && (
          <li data-tip={tr("tripInspector.ownershipTip", { unit: distUnit() })}>
            <span class="grow">{tr("tripInspector.carOwnership")}</span>
            <span class="val">{usd(t.carOwnPerceived / CAR_MIN_PER_USD)}</span>
          </li>
        )}
        <li data-tip={equation([
          { label: termLabel("drive"), real: num1(t.driveMin), weight: t.driveMin > 0 ? t.drivePerceived / t.driveMin : 1, perceived: t.drivePerceived },
          { label: termLabel("parking"), real: "", weight: 1, perceived: t.parkPerceived },
          ...(t.carOwnPerceived > 0 ? [{ label: termLabel("ownership"), real: usd(t.carOwnPerceived / CAR_MIN_PER_USD), weight: CAR_MIN_PER_USD, perceived: t.carOwnPerceived, usd: true }] : []),
        ], avg ? t.roadPerceived : undefined) + (avg ? `\n${avgTip}` : "")}>
          <span class="grow">{tr("tripText.effectiveTime")}</span>
          <span class="val">{minText(t.roadPerceived)}</span>
        </li>
      </ul>}
      {t.distM <= WALK_REACH_M && (
        <>
          <div class="h">{tr("tripInspector.byWalking")}</div>
          {t.apart ? <p class="muted">{noRoad()}</p> : <ul class="rows">
            <li data-tip={equation([{ label: termLabel("walk"), real: num1(t.walkMin), weight: t.walkWeight, perceived: t.walkMin * t.walkWeight }])}>
              <span class="grow">{tr("tripText.effectiveTime")}</span>
              <span class="val">{minText(t.walkMin * t.walkWeight)}</span>
            </li>
          </ul>}
        </>
      )}
      <PeriodRows
        train={(i) => trip.periods[i]?.pTrain ?? null}
        tip={(i) => { const x = trip.periods[i]; return x ? periodTip(x, i) : PERIODS[i].name; }} />
    </>
  );
}
