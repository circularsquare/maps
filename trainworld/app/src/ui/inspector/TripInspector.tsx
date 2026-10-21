// The trip explorer's inspector (T-195, docs/ui-inspectors.md): the trips between two places by
// mode as the view counts them (T-586), then one representative trip as the demand model decides
// it, for one period: the train's main route with its walks, waits, rides and changes, the other
// routes the train riders spread over, and the drive. Every number is the model's (game/trip.ts);
// every drawn quantity has a hover with it.

import { count, distance, roughDistance, speed } from "../../game/format";
import { MODE_COLOURS, modeColours } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { purposeName, purposeWords, type PurposeFilter } from "../../game/purposes";
import { WALK_KMH } from "../../game/walks";
import { routeTerms, routeTotals, stationName, tripOpenedFrom, tripPeriod, tripView, type TripMix, type TripPeriod, type TripRoute } from "../../game/trip";
import { purposeFilter, selection, shownDayPhrase, tool } from "../../game/ui";
import { world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { Chip } from "../widgets";

const min = (m: number) => `${Math.max(1, Math.round(m))} min`;
const pct = (v: number) => (v > 0 && v < 0.005 ? "<1%" : `${Math.round(100 * v)}%`);
const usd = (v: number) => "$" + v.toFixed(2);

const NO_RAIL: Record<string, string> = {
  noStationStart: "No station is within reach of the start.",
  noStationEnd: "No station is within reach of the end.",
  walksTooLong: "The walks to and from the nearest stations are too long for anyone to take the train.",
  noRoute: "No train runs between the stations near the two ends in this period.",
};

const num = (v: number) => String(Math.round(v * 10) / 10);
const exact = (n: number) => Math.round(n).toLocaleString("en");
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
    const body = `${x.label}${x.real ? " " + x.real : " " + num(Math.abs(x.perceived))}${x.real && Math.abs(x.weight - 1) >= 0.005 ? ` × ${weight(x.weight)}` : ""}`;
    return i === 0 ? body : `${minus ? "−" : "+"} ${body}`;
  }).join(" ");
  return `${text[0].toUpperCase()}${text.slice(1)} = ${Math.round(total)} min`;
}

/** The train's effective time as an equation. The terms are the first route's (the worker's
 * perceived minutes); the model's figure pools every station and route people use, so it is lower
 * than that route's sum by a last term. */
function trainEquation(t: TripPeriod, r: TripRoute): string {
  const terms: Term[] = routeTerms(t, r).filter((x) => x.perceived !== 0 || x.unit === "usd").map((x) => ({
    label: x.label === "station" ? "at the station" : x.label,
    real: x.unit === "usd" ? `$${x.real.toFixed(2)}` : x.real > 0 ? num(x.real) : "",
    weight: x.weight,
    perceived: x.perceived,
  }));
  const rest = t.railPerceived - terms.reduce((s, x) => s + x.perceived, 0);
  if (Math.abs(rest) >= 0.05) terms.push({ label: "choice of routes", real: "", weight: 1, perceived: rest });
  return equation(terms, t.railPerceived) + (Math.abs(rest) >= 0.05 ? "\nChoice of routes is the time saved by having several ways to go." : "");
}

/** The real minutes of a route as an equation: "Walk 8 + wait 3 + ride 6 = 17 min". */
function timeEquation(tot: ReturnType<typeof routeTotals>): string {
  const parts: [string, number][] = [["drive", tot.driveMin], ["walk", tot.walkMin], ["wait", tot.wait], ["ride", tot.ride], ["walk between stations", tot.xfer]];
  const text = parts.filter(([, m]) => m > 0).map(([l, m]) => `${l} ${num(m)}`).join(" + ");
  return `${text[0].toUpperCase()}${text.slice(1)} = ${Math.round(tot.total)} min`;
}

/** Walks over this straight-line distance are not offered by the model. */
const WALK_REACH_M = 10000;

/** T-628: the headline gets a walk row when walking the whole way takes under this, real minutes. */
const WALK_ROW_MIN = 30;

/** The headline: the trips between the two places by mode as the view counts them (T-586), or,
 * when the view counts none, the representative trip's chances in the period shown. */
function ModeBar({ t, mix, of }: { t: TripPeriod; mix: TripMix | null; of: string }) {
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const [train, walk, drive] = n > 0 ? [mix!.rail / n, mix!.walk / n, mix!.drive / n] : [t.pTrain, t.pWalk, t.pDrive];
  const parts: [string, number, string][] = [["Train", train, "seg-train"], ["Walk", walk, "seg-walk"], ["Drive", drive, "seg-drive"]];
  const walkTip = `; walking the whole way takes about ${min(t.walkMin)}`;
  const note = n > 0 ? "" : groupNote(t);
  return (
    <div class="trip-modes">
      <div class="bar trip-bar" role="img" aria-label={parts.map(([n, v]) => `${n} ${pct(v)}`).join(", ")}>
        {parts.map(([m, v, c]) => v > 0 && <i class={"seg " + c} style={{ width: `${100 * v}%` }} data-tip={`${m}: ${pct(v)} of ${of}${m === "Walk" ? walkTip : ""}`} />)}
      </div>
      <ul class="rows">
        {parts.filter(([m]) => m !== "Walk" || t.walkMin < WALK_ROW_MIN).map(([m, v]) => (
          <li data-tip={`${m}: ${(100 * v).toFixed(1)}% of ${of}${m === "Walk" ? walkTip : ""}${note}`}>
            <i class="mk" style={{ background: modeColours()[m === "Train" ? "train" : m === "Walk" ? "walk" : "drive"] }} />
            <span class="grow">{m}</span>
            <span class="val w-num">{pct(v)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Where the trip goes, in "to work by Grand Central" (T-586: the purpose filter's). */
const DEST: Record<PurposeFilter, string> = {
  all: "a destination", work: "work", school: "school or university", university: "school or university", shopping: "shops and errands",
  social: "social and leisure", medical: "medical care", airport: "the airport", longer: "a destination",
};
/** "shopping and errands trips", "trips" */
const tripWords = (f: PurposeFilter) => (f === "all" ? "trips" : `${purposeWords(f)} trips`);

function RouteLegs({ t, r }: { t: TripPeriod; r: TripRoute }) {
  const w = world.value!;
  const st = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const ln = new Map(w.save.lines.map((l) => [l.id, l]));
  const pair = t.pairs[r.pair];
  const first = r.legs[0], last = r.legs[r.legs.length - 1];
  const name = (id: string) => st.get(id) ?? "a station";
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={pair.accessRide
          ? `Driving, a bus or a bike over ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min on the road`
          : `Walk ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min at ${WALK_KMH} km/h`
            + (t.groupWalkStartMin >= 0 && Math.abs(t.groupWalkStartMin - pair.accessMin) >= 1 ? `\nThe map's bubbles count the homes near here at ${min(t.groupWalkStartMin)} to their nearest station` : "")}>
          <span class="grow">{pair.accessRide ? "Drive" : "Walk"} to {name(first.from)}</span>
          <span class="val">{min(pair.accessMin)}</span>
        </li>
      )}
      {r.legs.map((l, i) => {
        if (l.kind === "walk") {
          return (
            <li data-tip={`${l.min.toFixed(1)} min on foot`}>
              <span class="grow">Change: walk to {name(l.to)}</span>
              <span class="val">{min(l.min)}</span>
            </li>
          );
        }
        const line = ln.get(l.line);
        // T-842: riders arriving when no train is due for hours weigh on the wait as felt
        const span = l.waitPerceived - l.waitMin;
        const tip = `Wait ${l.waitMin.toFixed(1)} min (half the time between the useful trains), then ${l.min.toFixed(1)} min on the train over ${l.stops} ${l.stops === 1 ? "stop" : "stops"}` +
          (span >= 0.05 ? `. Some riders arrive when no train is due for hours, so the wait counts as ${span.toFixed(1)} min more` : "");
        return (
          <li data-tip={tip}>
            {line && <Chip line={line} size="sm" />}
            <span class="grow">{i > 0 && r.legs[i - 1].kind === "ride" ? "Change to " : ""}{line?.name ?? "A line"} to {name(l.to)}</span>
            <span class="val">{min(l.waitMin + l.min)}</span>
          </li>
        );
      })}
      {pair && last && (
        <li data-tip={`Walk ${distance(pair.egressM)} in a straight line, ${pair.egressMin.toFixed(1)} min at ${WALK_KMH} km/h`}>
          <span class="grow">Walk from {name(last.to)}</span>
          <span class="val">{min(pair.egressMin)}</span>
        </li>
      )}
    </ul>
  );
}

/** Routes that read the same in the panel: the same lines to a station of the same name. */
const routeLabel = (st: Map<string, string>) => (r: TripRoute) =>
  r.legs.filter((l) => l.kind === "ride").map((l) => l.line).join(">") + ">" + (st.get(r.legs[r.legs.length - 1]?.to ?? "") ?? "");

function OtherRoutes({ t }: { t: TripPeriod }) {
  const w = world.value!;
  const st = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const ln = new Map(w.save.lines.map((l) => [l.id, l]));
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
  const others = [...merged].filter(([k]) => k !== main).map(([, r]) => r).sort((a, b) => b.share - a.share).slice(0, 4);
  // what the main route's group and these rows leave: many small routes (T-709)
  const rest = 1 - mainShare(t) - others.reduce((s, r) => s + r.share, 0);
  if (!others.length && rest < 0.005) return null;
  return (
    <>
      <div class="h">Other routes<span class="right muted small">train riders</span></div>
      <ul class="rows">
        {others.map((r) => {
          const rides = r.legs.filter((l) => l.kind === "ride");
          const from = rides[0] ? st.get(rides[0].from) : "", to = r.legs.length ? st.get(r.legs[r.legs.length - 1].to) : "";
          const names = rides.map((l) => ln.get(l.line)?.name ?? "a line").join(" then ");
          const tot = routeTotals(t, r);
          return (
            <li data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of this trip's train riders, ${min(tot.total)} long`}>
              {rides.map((l) => { const line = ln.get(l.line); return line && <Chip line={line} size="sm" />; })}
              <span class="grow">{names} to {to}</span>
              <span class="val w-num">{pct(r.share)}</span>
            </li>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted" data-tip={spreadTip(t)}>
            <span class="grow">Smaller routes</span>
            <span class="val w-num">{pct(rest)}</span>
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
  const st = new Map(world.value!.save.stations.map((s) => [s.id, s.name]));
  const label = routeLabel(st), k = label(main);
  return main.share + t.routes.slice(1).filter((r) => label(r) === k).reduce((s, r) => s + r.share, 0);
}

/** Why no route carries most of them (T-709): the model spreads the train users over the station
 * pairs near the two ends and the useful trains at each station; routes under 0.5% are not kept. */
function spreadTip(t: TripPeriod): string {
  const top = t.pairs.reduce((m, p) => Math.max(m, p.share), 0);
  return `The model spreads this trip's train users over ${t.pairs.length} ${t.pairs.length === 1 ? "pair" : "pairs"} of stations near the two ends, the busiest pair taking ${pct(top)}, and over the useful trains at each station. Routes under 0.5% of them are not listed.`;
}

/** How much of the trip's train riders the main route carries. */
function mainNote(t: TripPeriod): string {
  if (t.routes[0].share >= 0.995) return "Every train user on this trip takes this route.";
  const all = mainShare(t);
  return `${pct(all)} of the train users on this trip take this route` + (1 - all >= 0.005 ? ", the rest use different routes." : ".");
}

/** T-720: the trip is the two places' own; the map's bubbles count it with the homes and places
 * near each end, which can differ. Empty when they agree to the shown percent. */
function groupNote(t: TripPeriod): string {
  if (Math.abs(t.pTrain - t.groupTrain) < 0.005 && Math.abs(t.pWalk - t.groupWalk) < 0.005) return "";
  return `\nCounted with the homes and places near here, as the map's bubbles count it: train ${pct(t.groupTrain)}, drive ${pct(t.groupDrive)}, walk ${pct(t.groupWalk)}`;
}

function ByPeriod({ periods, words }: { periods: (TripPeriod | null)[]; words: string }) {
  const q = shownPeriod.value;
  return (
    <>
      <div class="h">Train by period<span class="right muted small">click to show</span></div>
      <ul class="rows pick">
        {PERIODS.map((p, i) => {
          const t = periods[i];
          return (
            <li class={i === q ? "on" : ""} onClick={() => (tripPeriod.value = i)}
              data-tip={t ? `${p.name}, one ${words} trip: train ${pct(t.pTrain)}, drive ${pct(t.pDrive)}, walk ${pct(t.pWalk)}${groupNote(t)}` : p.name}>
              <span class="grow">{p.name}</span>
              <span class="trip-hbar"><i class="hbar" style={{ width: `${Math.round(60 * (t?.pTrain ?? 0))}px`, background: MODE_COLOURS.train }} /></span>
              <span class="val w-num">{t ? pct(t.pTrain) : ""}</span>
            </li>
          );
        })}
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
  const sub = sel.to
    ? `From a home ${fromName ? `by ${fromName}` : "at the open ring"} to ${DEST[filter]} ${toName ? `by ${toName}` : fromName ? "at the filled disc" : "at the filled one"}`
    // T-253: the next click is the trip tool's hint at the map's foot, not repeated here
    : fromName ? `From ${fromName}` : "Start picked";
  const tr = v?.trip;
  const about = tr && !tr.outside ? `The demand model's figures${tr.round > 0 ? ` after ${count(tr.round)} crowding ${tr.round === 1 ? "round" : "rounds"}` : ", before crowding"}. The model decides trips area by area, so nearby places give the same answer.${fromName || toName ? " A station picked as an end stands for the homes or places right around it." : ""}` : undefined;
  const head = (
    <>
      <div class="title"><span class="name" data-tip={about}>Trip</span></div>
      <div class="sub muted">{sub}</div>
      {back && back.trip === sel && (
        <div class="pane-foot">
          <button class="btn text" onClick={() => (selection.value = back.from)}>Back</button>
        </div>
      )}
    </>
  );
  if (!sel.to) return head;
  const trip = v?.trip;
  if (!trip) return <>{head}<p class="muted">{v?.asking ? "Asking the demand model." : "Calculating demand."}</p></>;
  if (trip.outside) return <>{head}<p class="muted">One of the two places is outside the city's homes and jobs.</p></>;
  const t = trip.periods[q];
  if (!t) return <>{head}<p class="muted">Calculating demand.</p></>;
  const main = t.routes[0];
  const tot = main ? routeTotals(t, main) : null;
  const pair = main ? t.pairs[main.pair] : null;  // T-586: the trips the view counts between the two places, and the trip the routes are read for
  const mix = v!.mix;
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const day = shownDayPhrase.value;
  const period = PERIODS[q].name;
  const of = n > 0 ? `the ${count(n)} ${tripWords(filter)} ${day} between the two places` : `the people making this trip in the ${period.toLowerCase()}`;
  const countTip = n > 0 ? [`${exact(n)} ${tripWords(filter)} ${day} between the two places, both ways`, ...(filter === "all" ? mix!.purposes.filter((p) => p.rail + p.walk + p.drive >= 0.5).map((p) => `${purposeName(p.key)}: ${exact(p.rail + p.walk + p.drive)}`) : [])].join("\n") : `${distance(t.distM)} in a straight line`;
  const words = purposeWords(v!.purpose);
  const picked = tripPeriod.value !== null;
  const repTip = `The routes are one ${words} trip in the ${period.toLowerCase()}: ${filter === "all" ? "the purpose with most of the trips between these places" : "the purpose shown"}, and ${picked ? "the period picked below" : "the period with most of its trips"}`;
  return (
    <>
      {head}
      <p class="muted small" data-tip={countTip}>{n > 0 ? `${count(n)} ${tripWords(filter)} ${day}` : period}, {distance(t.distM)} apart</p>
      <ModeBar t={t} mix={mix} of={of} />
      <div class="h">By train<span class="right muted small" data-tip={repTip}>{purposeName(v!.purpose)}, {period.toLowerCase()}</span></div>
      {!t.railOffered ? <p class="muted">{NO_RAIL[t.why] ?? "No train."}</p> : main && tot && pair ? (
        <>
          <RouteLegs t={t} r={main} />
          <ul class="rows">
            <li data-tip={`Base fare ${usd(t.baseFarePerceived / t.minPerUsd)} once a trip, plus the distance fare`}>
              <span class="grow">Fare</span>
              <span class="val">{usd(tot.fareUsd)}</span>
            </li>
            <li data-tip={timeEquation(tot)}>
              <span class="grow">Time</span>
              <span class="val">{min(tot.total)}</span>
            </li>
            <li data-tip={trainEquation(t, main)}>
              <span class="grow">Effective time</span>
              <span class="val">{min(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small" data-tip={spreadTip(t)}>{mainNote(t)}</p>
          <OtherRoutes t={t} />
        </>
      ) : <p class="muted">No route.</p>}
      <div class="h">By car</div>
      <ul class="rows">
        <li data-tip={`${distance(t.driveDistM)} in a straight line, ${distance(1000 * t.driveRoadKm)} by road, at an average ${speed(t.driveRoadKm / (Math.max(t.driveMin, 0.1) / 60))}`}>
          <span class="grow">Drive</span>
          <span class="val">{min(t.driveMin)}</span>
        </li>
        <li data-tip="Minutes to park, plus the price of parking and any congestion charge, counted as minutes">
          <span class="grow">Parking</span>
          <span class="val">{min(t.parkPerceived)}</span>
        </li>
        {t.carOwnPerceived > 0 && (
          <li data-tip="The cost of owning the car, charged per trip and per distance driven">
            <span class="grow">Car ownership</span>
            <span class="val">{usd(t.carOwnPerceived / t.minPerUsd)}</span>
          </li>
        )}
        <li data-tip={equation([
          { label: "drive", real: num(t.driveMin), weight: t.driveMin > 0 ? t.drivePerceived / t.driveMin : 1, perceived: t.drivePerceived },
          { label: "parking", real: "", weight: 1, perceived: t.parkPerceived },
          ...(t.carOwnPerceived > 0 ? [{ label: "ownership", real: usd(t.carOwnPerceived / t.minPerUsd), weight: t.minPerUsd, perceived: t.carOwnPerceived }] : []),
        ])}>
          <span class="grow">Effective time</span>
          <span class="val">{min(t.roadPerceived)}</span>
        </li>
      </ul>
      <div class="h">By walking</div>
      {t.distM > WALK_REACH_M ? <p class="muted">Walks over {roughDistance(WALK_REACH_M)} are not offered.</p> : (
        <ul class="rows">
          <li data-tip={equation([{ label: "walk", real: num(t.walkMin), weight: t.walkWeight, perceived: t.walkMin * t.walkWeight }])}>
            <span class="grow">Effective time</span>
            <span class="val">{min(t.walkMin * t.walkWeight)}</span>
          </li>
        </ul>
      )}
      <ByPeriod periods={trip.periods} words={words} />
    </>
  );
}
