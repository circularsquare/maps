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
import { CAR_MIN_PER_USD, openRoute, peopleOf, routeCrowding, routeLabel, routeTerms, routeTotals, stationName, tripOpenedFrom, tripPeriod, tripView, wholePeople, type TripMix, type TripPeriod, type TripRoute } from "../../game/trip";
import { purposeFilter, selection, shownDayPhrase } from "../../game/ui";
import { linesById, stationNames, world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { Actions, SegBar } from "../widgets";
import { MarketTrip } from "./MarketTrip";
import { legRows, PeriodRows, routeRow } from "./RouteParts";
import { minText, num1, pctShare, roundTo } from "./tripText";

const usd = (v: number) => "$" + v.toFixed(2);

const NO_RAIL: Record<string, string> = {
  noStationStart: "No station is within a walk or a drive of the start.",
  noStationEnd: "No station is within a walk of the end.",
  noRoute: "No train runs between the stations near the two ends in this period.",
  walkFurther: "Walking there is shorter than the walks to, between and from the stations.",
};

/** T-910: why the train is not offered. Too long a way to and from the trains says which end
 * holds it back: people drive to a station from home on trips of `rideMinKm` or more, and walk
 * from one at the far end. */
function NoRail({ t }: { t: TripPeriod }) {
  if (t.why !== "walksTooLong") return <p class="muted">{NO_RAIL[t.why] ?? "No train."}</p>;
  const a = t.walkStartPerceived, b = t.walkEndPerceived;
  const near = Number.isFinite(a) ? "The nearest stations are too far to walk to" : "No station is within a walk of the start";
  const text = t.rideToStation === "short" ? `${near}, and the trip is too short to drive to one.`
    : t.rideToStation === "none" ? `${near}, and none is within a drive of the start.`
    : b >= a ? "The walk from the nearest station to the end is too long."
    : "The stations near the start are too far, even by car.";
  const to = Number.isFinite(a) ? `Getting to a station counts as ${minText(a)}${t.rideToStation === "tried" ? " on foot or by car" : " on foot"}` : "No station is within a walk of the start";
  const tip = `${to}, getting from one ${minText(b)} on foot. The train is offered when the two come to ${Math.round(t.pairWalkMax)} min or less.` +
    (t.rideToStation === "short" ? `\nPeople drive to a station only when the areas of the two ends are ${num1(t.rideMinKm)} km or more apart.` : "");
  return <p class="muted" data-tip={tip}>{text}</p>;
}

/** "trip" or "trips" after a count. */
const trips = (n: number, f: PurposeFilter) => (Math.round(n) === 1 ? tripWords(f).replace(/trips$/, "trip") : tripWords(f));

const weight = (v: number) => String(Math.round(v * 100) / 100);

/** One term of an effective time equation: "walk 8 × 1.5", "wait 3", "fare $2.90 × 3". */
interface Term {
  label: string;
  /** the real amount, already formatted ("" where the term is a count of its own minutes) */
  real: string;
  weight: number;
  perceived: number;
}

/** The effective time hover (T-699): only the equation, in the order of the rows, "Walk 8 × 1.5 +
 * wait 3 + ride 6 = 21 min". The total is the sum of the terms' unrounded minutes. */
function equation(terms: Term[], total = terms.reduce((s, x) => s + x.perceived, 0)): string {
  const text = terms.map((x, i) => {
    const minus = x.perceived < 0;
    const body = `${x.label}${x.real ? " " + x.real : " " + num1(Math.abs(x.perceived))}${x.real && Math.abs(x.weight - 1) >= 0.005 ? ` × ${weight(x.weight)}` : ""}`;
    return i === 0 ? body : `${minus ? "−" : "+"} ${body}`;
  }).join(" ");
  return `${capitalise(text)} = ${Math.round(total)} min`;
}

/** The train's effective time as an equation. The terms are the route's (the worker's perceived
 * minutes); the model's figure pools every station and route people use, so it is lower than the
 * main route's sum by a last term. `pooled` false: the route's own sum alone (T-870). */
function trainEquation(t: TripPeriod, r: TripRoute, pooled = true): string {
  const terms: Term[] = routeTerms(t, r).filter((x) => x.perceived !== 0 || x.unit === "usd").map((x) => ({
    label: x.label,
    real: x.unit === "usd" ? `$${x.real.toFixed(2)}` : x.real > 0 ? num1(x.real) : "",
    weight: x.weight,
    perceived: x.perceived,
  }));
  if (!pooled) return equation(terms);
  const rest = t.railPerceived - terms.reduce((s, x) => s + x.perceived, 0);
  if (Math.abs(rest) >= 0.05) terms.push({ label: "choice of routes", real: "", weight: 1, perceived: rest });
  return equation(terms, t.railPerceived) + (Math.abs(rest) >= 0.05 ? "\nChoice of routes is the time saved by having several ways to go." : "");
}

/** The real minutes of a route as an equation: "Walk 8 + wait 3 + ride 6 = 17 min". */
function timeEquation(tot: ReturnType<typeof routeTotals>): string {
  const parts: [string, number][] = [["drive", tot.driveMin], ["walk", tot.walkMin], ["wait", tot.wait], ["ride", tot.ride], ["walk between stations", tot.xfer]];
  const text = parts.filter(([, m]) => m > 0).map(([l, m]) => `${l} ${num1(m)}`).join(" + ");
  return `${capitalise(text)} = ${Math.round(tot.total)} min`;
}

/** Walks over this straight-line distance are not offered by the model. */
const WALK_REACH_M = 10000;

/** T-628, T-992: the headline gets a walk row when walking the whole way takes under this, real
 * minutes, or whatever it takes when walking has a share. */
const WALK_ROW_MIN = 30;

/** T-1148: what the model's walking and driving stand for (demand-plain.md). */
const STANDS_FOR: Record<string, string> = {
  Walk: "Walking here also covers cycling and short bus rides.",
  Drive: "Driving here also covers taxis and longer bus rides.",
};

/** The headline: the trips between the two places by mode as the view counts them (T-586), or,
 * when the view counts none, the representative trip's chances in the period shown. */
function ModeBar({ t, mix, of }: { t: TripPeriod; mix: TripMix | null; of: string }) {
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const model = n > 0 ? [mix!.rail / n, mix!.walk / n, mix!.drive / n] : [t.pTrain, t.pWalk, t.pDrive];
  // T-911: counted trips are whole people; the representative trip's chances stay as they are
  const [train, walk, drive] = roundTo(model, peopleOf(n));
  const parts: [string, number, string][] = [["Train", train, "seg-train"], ["Walk", walk, "seg-walk"], ["Drive", drive, "seg-drive"]];
  const walkTip = `; walking the whole way takes about ${minText(t.walkMin)}`;
  const note = n > 0 ? "" : groupNote(t);
  const modelTip = (i: number) => (n > 0 ? `\nModel: ${(100 * model[i]).toFixed(1)}%, rounded here to whole trips` : "");
  const stands = (m: string) => (STANDS_FOR[m] ? `\n${STANDS_FOR[m]}` : "");
  return (
    <div class="trip-modes">
      <SegBar kind="trip-bar" label={parts.map(([n, v]) => `${n} ${pctShare(v)}`).join(", ")}
        parts={parts.map(([m, v, c]) => ({ share: v, cls: c, tip: `${m}: ${pctShare(v)} of ${of}${m === "Walk" ? walkTip : ""}${stands(m)}` }))} />
      <ul class="rows">
        {parts.map((x, i) => [x, i] as const).filter(([[m, v]]) => m !== "Walk" || v > 0 || t.walkMin < WALK_ROW_MIN).map(([[m, v], i]) => (
          <li data-tip={`${m}: ${n > 0 ? pctShare(v) : `${(100 * v).toFixed(1)}%`} of ${of}${m === "Walk" ? walkTip : ""}${modelTip(i)}${note}${stands(m)}`}>
            <i class="mk" style={{ background: modeColours()[m === "Train" ? "train" : m === "Walk" ? "walk" : "drive"] }} />
            <span class="grow">{m}</span>
            <span class="val w-num">{pctShare(v)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** "errands trips", "trips" */
const tripWords = (f: PurposeFilter) => (f === "all" ? "trips" : `${purposeWords(f)} trips`);

function RouteLegs({ t, r }: { t: TripPeriod; r: TripRoute }) {
  const w = world.value!;
  const st = stationNames(w), ln = linesById(w);
  const pair = t.pairs[r.pair];
  const first = r.legs[0], last = r.legs[r.legs.length - 1];
  const name = (id: string) => st.get(id) ?? "a station";
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={pair.accessRide
          ? `Driving, a bus or a bike over ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min on the road`
          : `Walk ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min at ${WALK_KMH} km/h`
            + (t.groupWalkStartMin >= 0 && Math.abs(t.groupWalkStartMin - pair.accessMin) >= 1 ? `\nThe map's bubbles count the homes near here at ${minText(t.groupWalkStartMin)} to their nearest station` : "")}>
          <span class="grow">{pair.accessRide ? "Drive" : "Walk"} to {name(first.from)}</span>
          <span class="val">{minText(pair.accessMin)}</span>
        </li>
      )}
      {legRows(r, name, ln, (l) => {
        // T-842: riders arriving when no train is due for hours weigh on the wait as felt
        const span = l.waitPerceived - l.waitMin, full = l.fullWaitMin;
        const v = w.lineStats[l.line]?.bus ? "bus" : "train";
        return `Wait ${(l.waitMin - full).toFixed(1)} min (half the time between the useful ${v}s)${full >= 0.05 ? ` and ${full.toFixed(1)} min more for a ${v} with room` : ""}, then ${l.min.toFixed(1)} min on the ${v} over ${l.stops} ${l.stops === 1 ? "stop" : "stops"}` +
          (span >= 0.05 ? `. Some riders arrive when no ${v} is due for hours, so the wait counts as ${span.toFixed(1)} min more` : "");
      })}
      {pair && last && (
        <li data-tip={`Walk ${distance(pair.egressM)} in a straight line, ${pair.egressMin.toFixed(1)} min at ${WALK_KMH} km/h`}>
          <span class="grow">Walk from {name(last.to)}</span>
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
      <li data-tip={`Base fare ${usd(t.baseFarePerceived / t.minPerUsd)} once a trip, plus the distance fare`}>
        <span class="grow">Fare</span>
        <span class="val">{usd(tot.fareUsd)}</span>
      </li>
      {pair?.accessRide && (
        <li data-tip={`Owning the car costs ${usd(car)} for the drive to the station, charged per trip and per distance driven` +
          (parking >= 0.005 ? `\nParking at the station costs ${usd(parking)} on average` : "")}>
          <span class="grow">Car</span>
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
        <li data-tip={`Once a train is past 70% of its crush load, riders count each minute on board as more than one, 3 at crush and 5.4 at 120%. On this route that adds ${num1(crowding)} min`}>
          <span class="grow">Crowding</span>
          <span class="val">{minText(crowding)}</span>
        </li>
      )}
      {fullWait >= 0.05 && (
        <li data-tip={`Riders left at the platform by full trains wait ${num1(fullWait)} min more on average for a train with room. Each minute counts as one, up to an hour at a stop`}>
          <span class="grow">Waiting for a train with room</span>
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
      const k = x.label === "changes" ? "change" : x.label;
      m.set(k, (m.get(k) ?? 0) + x.perceived);
    }
    return m;
  };
  const a = by(r), b = by(t.routes[0]);
  const parts = [...new Set([...b.keys(), ...a.keys()])]
    .map((k) => [k, (a.get(k) ?? 0) - (b.get(k) ?? 0)] as const)
    .filter(([, d]) => Math.abs(d) >= 0.05);
  const diffTip = `This route's effective time is ${minText(own)}, the main route's ${minText(mainOwn)}, both before choice of routes` +
    (parts.length ? "\n" + parts.map(([k, d]) => `${capitalise(k)} ${d > 0 ? "+" : "−"}${num1(Math.abs(d))} min`).join("\n") : "");
  return (
    <div class="trip-alt">
      <RouteLegs t={t} r={r} />
      <ul class="rows">
        <FareRows t={t} r={r} />
        <CrowdRows r={r} />
        <li data-tip={timeEquation(tot)}>
          <span class="grow">Time</span>
          <span class="val">{minText(tot.total)}</span>
        </li>
        <li data-tip={trainEquation(t, r, false)}>
          <span class="grow">Effective time</span>
          <span class="val">{minText(own)}</span>
        </li>
        <li data-tip={diffTip}>
          <span class="grow">Against the main route</span>
          <span class="val">{diff === 0 ? "same" : `${diff > 0 ? "+" : "−"}${Math.abs(diff)} min`}</span>
        </li>
      </ul>
    </div>
  );
}

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
      <div class="h">Other routes<span class="right muted small">train riders</span></div>
      <ul class="rows pick">
        {kept.map(([[k, r], v]) => {
          const { names, from, to, label } = routeRow(r, st, ln);
          const tot = routeTotals(t, r);
          const on = k === open;
          return (
            <Fragment key={k}>
              <li class={on ? "on" : ""} onClick={() => (openRoute.value = on ? null : k)}
                data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of this trip's train riders${riders ? `, ${pctShare(v)} of the ${count(riders)} counted here` : ""}, ${minText(tot.total)} long\n${on ? "Click to close" : "Click to see its legs and effective time"}`}>
                {label}
                <span class="val w-num">{pctShare(v)}</span>
              </li>
              {on && <li class="trip-alt-row"><RouteDetail t={t} r={r} /></li>}
            </Fragment>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted rest" data-tip={spreadTip(t)}>
            <span class="grow">Smaller routes</span>
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

/** Why no route carries most of them (T-709): the model spreads the train users over the station
 * pairs near the two ends and the useful trains at each station; routes under 0.5% are not kept. */
function spreadTip(t: TripPeriod): string {
  const top = t.pairs.reduce((m, p) => Math.max(m, p.share), 0);
  return `The model spreads this trip's train users over ${t.pairs.length} ${t.pairs.length === 1 ? "pair" : "pairs"} of stations near the two ends, the busiest pair taking ${pctShare(top)}, and over the useful trains at each station. Routes under 0.5% of them are not listed.`;
}

/** How much of the trip's train riders the main route carries. */
function mainNote(t: TripPeriod, riders: number | null): string {
  const all = routeRows(t, riders).shown[0];
  if (all >= 0.995) return "Every train user on this trip takes this route.";
  return `${pctShare(all)} of the train users on this trip take this route` + (1 - all >= 0.005 ? ", the rest use different routes." : ".");
}

/** T-720: the trip is the two places' own; the map's bubbles count it with the homes and places
 * near each end, which can differ. Empty when they agree to the shown percent. */
function groupNote(t: TripPeriod): string {
  if (Math.abs(t.pTrain - t.groupTrain) < 0.005 && Math.abs(t.pWalk - t.groupWalk) < 0.005) return "";
  return `\nCounted with the homes and places near here, as the map's bubbles count it: train ${pctShare(t.groupTrain)}, drive ${pctShare(t.groupDrive)}, walk ${pctShare(t.groupWalk)}`;
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
      <div class="h">Busiest routes<span class="right muted small">train riders</span></div>
      <ul class="rows">
        {t.routes.map((r, i) => {
          if (riders && shown[i] <= 0) return null;
          const { names, from, to, label } = routeRow(r, st, ln);
          return (
            <li data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of the train riders between the two groups${riders ? `, ${pctShare(shown[i])} of the ${count(riders)} counted here` : ""}`}>
              {label}
              <span class="val w-num">{pctShare(shown[i])}</span>
            </li>
          );
        })}
        {cut >= 0.005 && (
          <li class="muted rest" data-tip="Routes outside the five busiest">
            <span class="grow">Other routes</span>
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
  const tr = v?.trip;
  const about = tr && !tr.outside ? `Figures from the demand model${tr.round > 0 ? ` after ${count(tr.round)} crowding ${tr.round === 1 ? "round" : "rounds"}` : ", before crowding"}. Trips are worked out area by area, so nearby places give the same answer.${fromName || toName ? " A station picked as an end stands for the homes or places right around it." : ""}` : undefined;
  const head = (
    <>
      <div class="title"><span class="name" data-tip={about}>Trip</span></div>
      {back && back.trip === sel && (
        <Actions>
          <button class="btn text" onClick={() => (selection.value = back.from)}>Back</button>
        </Actions>
      )}
    </>
  );
  if (!sel.to) return head;
  if (market) return <>{head}<MarketTrip m={market} /></>;
  const trip = v?.trip;
  if (!trip) return <>{head}<p class="muted">{v?.failed ? "This trip could not be worked out." : v?.asking ? "Working out this trip" : "Calculating demand"}</p></>;
  if (trip.outside) return <>{head}<p class="muted">One of the two places is outside the city's homes and jobs.</p></>;
  const t = trip.periods[q];
  if (!t) return <>{head}<p class="muted">Calculating demand</p></>;
  const main = t.routes[0];
  const tot = main ? routeTotals(t, main) : null;
  const pair = main ? t.pairs[main.pair] : null;  // T-586: the trips the view counts between the two places, and the trip the routes are read for
  const mix = v!.mix;
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const day = shownDayPhrase.value;
  const period = PERIODS[q].name;
  const of = n >= 0.5 ? `the ${count(n)} ${trips(n, filter)} ${day} between the two places` : n > 0 ? `the ${tripWords(filter)} ${day} between the two places` : `the people making this trip in the ${period.toLowerCase()}`;
  // T-911: the train riders the headline counts, in whole trips, for the route shares (none: the model's)
  const riders = n > 0 ? wholePeople([mix!.rail, mix!.walk, mix!.drive], peopleOf(n))[0] || null : null;
  const countTip = n > 0 ? [`${exact(n)} ${tripWords(filter)} ${day} between the two places, both ways`, ...(filter === "all" ? mix!.purposes.filter((p) => p.rail + p.walk + p.drive >= 0.5).map((p) => `${purposeName(p.key)}: ${exact(p.rail + p.walk + p.drive)}`) : [])].join("\n") : `${distance(t.distM)} in a straight line`;
  const words = purposeWords(v!.purpose);
  const picked = tripPeriod.value !== null;
  const avg = (t.sampled ?? 0) > 1;
  // T-865: the busiest bubbles of each group, each to each, weighted by their trips
  const avgTip = avg ? `Averaged over ${t.sampled} pairs of areas, the busiest few of each group, weighted by their trips` : "";
  const repTip = `The routes are ${avg ? `${words} trips averaged over ${t.sampled} pairs of areas` : `one ${words} trip`} in the ${period.toLowerCase()}: ${filter === "all" ? "the purpose with most of the trips between these places" : "the purpose shown"}, and ${picked ? "the period picked below" : "the period with most of its trips"}`;
  return (
    <>
      {head}
      <p class="muted small" data-tip={countTip}>{n >= 0.5 ? `${count(n)} ${trips(n, filter)} ${day}` : n > 0 ? `Under 1 ${trips(1, filter)} ${day}` : period}, {distance(t.distM)} apart</p>
      <ModeBar t={t} mix={mix} of={of} />
      <div class="h">By train<span class="right muted small" data-tip={repTip}>{purposeName(v!.purpose)}, {period.toLowerCase()}</span></div>
      {!t.railOffered ? <NoRail t={t} /> : avg ? (
        <>
          <ul class="rows">
            <li data-tip={`${avgTip}${t.railSampled !== t.sampled ? `; the train is offered for ${t.railSampled} of them` : ""}`}>
              <span class="grow">Effective time</span>
              <span class="val">{minText(t.railPerceived)}</span>
            </li>
          </ul>
          {t.routes.length > 0 && <BusiestRoutes t={t} riders={riders} />}
        </>
      ) : main && tot && pair ? (
        <>
          <RouteLegs t={t} r={main} />
          <ul class="rows">
            <FareRows t={t} r={main} />
            <CrowdRows r={main} />
            <li data-tip={timeEquation(tot)}>
              <span class="grow">Time</span>
              <span class="val">{minText(tot.total)}</span>
            </li>
            <li data-tip={trainEquation(t, main)}>
              <span class="grow">Effective time</span>
              <span class="val">{minText(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small" data-tip={spreadTip(t)}>{mainNote(t, riders)}</p>
          <OtherRoutes t={t} riders={riders} />
        </>
      ) : <p class="muted">No route.</p>}
      <div class="h">By car</div>
      <ul class="rows">
        <li data-tip={`${distance(1000 * t.driveRoadKm)} by road (${distance(t.driveDistM)} in a straight line) at an average of ${speed(t.driveRoadKm / (Math.max(t.driveMin, 0.1) / 60))}${avg ? `\n${avgTip}` : ""}`}>
          <span class="grow">Drive</span>
          <span class="val">{minText(t.driveMin)}</span>
        </li>
        <li data-tip="Finding a space, plus the cost of parking and any congestion charge counted in minutes">
          <span class="grow">Parking</span>
          <span class="val">{minText(t.parkPerceived)}</span>
        </li>
        {t.carOwnPerceived > 0 && (
          <li data-tip={`What owning a car costs on this trip: a set amount for each trip and more for each ${distUnit()}`}>
            <span class="grow">Car ownership</span>
            <span class="val">{usd(t.carOwnPerceived / CAR_MIN_PER_USD)}</span>
          </li>
        )}
        <li data-tip={equation([
          { label: "drive", real: num1(t.driveMin), weight: t.driveMin > 0 ? t.drivePerceived / t.driveMin : 1, perceived: t.drivePerceived },
          { label: "parking", real: "", weight: 1, perceived: t.parkPerceived },
          ...(t.carOwnPerceived > 0 ? [{ label: "ownership", real: usd(t.carOwnPerceived / CAR_MIN_PER_USD), weight: CAR_MIN_PER_USD, perceived: t.carOwnPerceived }] : []),
        ], avg ? t.roadPerceived : undefined) + (avg ? `\n${avgTip}` : "")}>
          <span class="grow">Effective time</span>
          <span class="val">{minText(t.roadPerceived)}</span>
        </li>
      </ul>
      {t.distM <= WALK_REACH_M && (
        <>
          <div class="h">By walking</div>
          <ul class="rows">
            <li data-tip={equation([{ label: "walk", real: num1(t.walkMin), weight: t.walkWeight, perceived: t.walkMin * t.walkWeight }])}>
              <span class="grow">Effective time</span>
              <span class="val">{minText(t.walkMin * t.walkWeight)}</span>
            </li>
          </ul>
        </>
      )}
      <PeriodRows
        train={(i) => trip.periods[i]?.pTrain ?? null}
        tip={(i) => { const x = trip.periods[i]; return x ? `${PERIODS[i].name}, one ${words} trip: train ${pctShare(x.pTrain)}, drive ${pctShare(x.pDrive)}, walk ${pctShare(x.pWalk)}${groupNote(x)}` : PERIODS[i].name; }} />
    </>
  );
}
