// The trip explorer's inspector (T-195, docs/ui-inspectors.md): the trips between two places by
// mode as the view counts them (T-586), then one representative trip as the demand model decides
// it, for one period: the train's main route with its walks, waits, rides and changes, the other
// routes the train riders spread over, and the drive. Every number is the model's (game/trip.ts);
// every drawn quantity has a hover with it.

import { count, distance } from "../../game/format";
import { MODE_COLOURS } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { purposeName, purposeWords, type PurposeFilter } from "../../game/purposes";
import { WALK_KMH } from "../../game/walks";
import { routeTerms, routeTotals, stationName, tripOpenedFrom, tripPeriod, tripView, type TripMix, type TripPeriod, type TripRoute } from "../../game/trip";
import { purposeFilter, selection, shownDayPhrase, tool } from "../../game/ui";
import { world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { Chip, Row } from "../widgets";

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

/** The "Counts as" hover: the first route's perceived total term by term (the worker's own
 * perceived minutes), then what the weights are. The shown figure averages the routes. */
function countsAsTip(t: TripPeriod, r: TripRoute): string {
  const terms = routeTerms(t, r).filter((x) => x.perceived !== 0 || x.unit === "usd");
  const p = (v: number) => Math.round(v * 10) / 10;
  const total = terms.reduce((s, x) => s + p(x.perceived), 0);
  const lines = terms.map((x) => {
    const w = Math.abs(x.weight - 1) < 0.005 && x.unit === "min" ? "" : ` × ${weight(x.weight)}`;
    const real = x.unit === "usd" ? `$${x.real.toFixed(2)}` : x.real > 0 ? `${num(x.real)} min` : "";
    return `${x.label}${real ? " " + real : ""}${w} = ${p(x.perceived).toFixed(1)}`;
  });
  const avg = Math.abs(total - t.railPerceived) >= 0.5;
  return [
    "The route listed first, in the minutes the model weighs against driving:",
    ...lines,
    `Total ${total.toFixed(1)}`,
    ...(avg ? [`The ${Math.round(t.railPerceived)} shown pools the stations and routes people near here use, so it can be lower than any one route.`] : []),
    `Walks count for more than a minute on board, crowding stretches the ride, the wait is half the time between trains, and a dollar of fare counts as ${weight(t.minPerUsd)} minutes.`,
  ].join("\n");
}

/** T-628: walking gets its own row when walking the whole way takes under this, real minutes. */
const WALK_ROW_MIN = 30;

/** The headline: the trips between the two places by mode as the view counts them (T-586), or,
 * when the view counts none, the representative trip's chances in the period shown. */
function ModeBar({ t, mix, of }: { t: TripPeriod; mix: TripMix | null; of: string }) {
  const n = mix ? mix.rail + mix.walk + mix.drive : 0;
  const [train, walk, drive] = n > 0 ? [mix!.rail / n, mix!.walk / n, mix!.drive / n] : [t.pTrain, t.pWalk, t.pDrive];
  const parts: [string, number, string][] = [["Train", train, "seg-train"], ["Walk", walk, "seg-walk"], ["Drive", drive, "seg-drive"]];
  const walkTip = `; walking the whole way takes about ${min(t.walkMin)}`;
  return (
    <div class="trip-modes">
      <div class="bar trip-bar" role="img" aria-label={parts.map(([n, v]) => `${n} ${pct(v)}`).join(", ")}>
        {parts.map(([m, v, c]) => v > 0 && <i class={"seg " + c} style={{ width: `${100 * v}%` }} data-tip={`${m}: ${pct(v)} of ${of}${m === "Walk" ? walkTip : ""}`} />)}
      </div>
      <ul class="rows">
        {parts.filter(([m]) => m !== "Walk" || t.walkMin < WALK_ROW_MIN).map(([m, v]) => (
          <li data-tip={`${m}: ${(100 * v).toFixed(1)}% of ${of}${m === "Walk" ? walkTip : ""}`}>
            <i class="mk" style={{ background: MODE_COLOURS[m === "Train" ? "train" : m === "Walk" ? "walk" : "drive"] }} />
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
          ? `Driving, a bus or a bike over ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min on the road. The ride's fixed cost and the parking or drop-off at the station add ${pair.rideExtra.toFixed(1)} min, so it counts as ${pair.accessPerceived.toFixed(1)} min`
          : `Walk ${distance(pair.accessM)} in a straight line, ${pair.accessMin.toFixed(1)} min at ${WALK_KMH} km/h; it counts as ${pair.accessPerceived.toFixed(1)} min`}>
          <span class="grow">{pair.accessRide ? "Drive" : "Walk"} to {name(first.from)}</span>
          <span class="val">{min(pair.accessMin)}</span>
        </li>
      )}
      {r.legs.map((l, i) => {
        if (l.kind === "walk") {
          return (
            <li data-tip={`${l.min.toFixed(1)} min on foot; it counts as ${l.perceived.toFixed(1)} min`}>
              <span class="grow">Change: walk to {name(l.to)}</span>
              <span class="val">{min(l.min)}</span>
            </li>
          );
        }
        const line = ln.get(l.line);
        const crowd = l.perceived - l.min;
        const tip = `Wait ${l.waitMin.toFixed(1)} min (half the time between the useful trains), then ${l.min.toFixed(1)} min on the train over ${l.stops} ${l.stops === 1 ? "stop" : "stops"}`
          + (crowd >= 0.05 ? `. Crowding makes the ride count as ${l.perceived.toFixed(1)} min` : "")
          + (l.farePerceived > 0 ? `. Distance fare ${usd(l.farePerceived / t.minPerUsd)}` : "");
        return (
          <li data-tip={tip}>
            {line && <Chip line={line} size="sm" />}
            <span class="grow">{i > 0 && r.legs[i - 1].kind === "ride" ? "Change to " : ""}{line?.name ?? "A line"} to {name(l.to)}</span>
            <span class="val">{min(l.waitMin + l.min)}</span>
          </li>
        );
      })}
      {pair && last && (
        <li data-tip={`Walk ${distance(pair.egressM)} in a straight line, ${pair.egressMin.toFixed(1)} min at ${WALK_KMH} km/h; it counts as ${pair.egressPerceived.toFixed(1)} min`}>
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
  if (!others.length) return null;
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
            <li data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of this trip's train riders, ${min(tot.total)} door to door`}>
              {rides.map((l) => { const line = ln.get(l.line); return line && <Chip line={line} size="sm" />; })}
              <span class="grow">{names} to {to}</span>
              <span class="val w-num">{pct(r.share)}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}

/** How much of the trip's train riders the main route carries. The model spreads them over the
 * stations near each end and over the useful trains at each station. */
function mainNote(t: TripPeriod): string {
  const main = t.routes[0];
  if (main.share >= 0.995) return "Every train rider on this trip takes this route.";
  const st = new Map(world.value!.save.stations.map((s) => [s.id, s.name]));
  const label = routeLabel(st), k = label(main);
  const same = t.routes.slice(1).filter((r) => label(r) === k).reduce((s, r) => s + r.share, 0);
  return `${pct(main.share)} of the train riders on this trip take this route` +
    (same >= 0.005 ? `, and ${pct(same)} more the same trains from other platforms or with other walks.` : ".") +
    (1 - main.share - same >= 0.005 ? " The rest use other stations or trains nearby." : "");
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
              data-tip={t ? `${p.name}, one ${words} trip: train ${pct(t.pTrain)}, drive ${pct(t.pDrive)}, walk ${pct(t.pWalk)}` : p.name}>
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
          <span class="muted small">{back.what}</span>
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
  const pair = main ? t.pairs[main.pair] : null;
  // T-586: the trips the view counts between the two places, and the trip the routes are read for
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
            <Row label="Door to door" bold>{min(tot.total)}</Row>
            <li data-tip={`${tot.driveMin > 0 ? `Driving to the station ${tot.driveMin.toFixed(1)} min, walking` : "Walking"} ${tot.walkMin.toFixed(1)} min, waiting ${tot.wait.toFixed(1)} min, on trains ${tot.ride.toFixed(1)} min${tot.xfer > 0 ? `, walking between stations ${tot.xfer.toFixed(1)} min` : ""}`}>
              <span class="grow muted">{tot.driveMin > 0 ? `Drive ${min(tot.driveMin)}, walk ${min(tot.walkMin + tot.xfer)}` : `Walk ${min(tot.walkMin + tot.xfer)}`}, wait {min(tot.wait)}, ride {min(tot.ride)}</span>
            </li>
            <li data-tip={`Base fare ${usd(t.baseFarePerceived / t.minPerUsd)} once a trip, plus the distance fare`}>
              <span class="grow">Fare</span>
              <span class="val">{usd(tot.fareUsd)}</span>
            </li>
            <li data-tip={countsAsTip(t, main)}>
              <span class="grow">Counts as</span>
              <span class="val">{min(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small">{mainNote(t)}</p>
          <OtherRoutes t={t} />
        </>
      ) : <p class="muted">No route.</p>}
      <div class="h">By car</div>
      <ul class="rows">
        <li data-tip={`${distance(t.driveDistM)} in a straight line, 1.3 times that by road, at the speeds of the places at each end`}>
          <span class="grow">Drive</span>
          <span class="val">{min(t.driveMin)}</span>
        </li>
        <li data-tip={`Parking at the end counts as ${t.parkPerceived.toFixed(1)} min: 3 to park, more where jobs are dense`}>
          <span class="grow">Parking</span>
          <span class="val">{min(t.parkPerceived)}</span>
        </li>
        {t.carOwnPerceived > 0 && (
          <li data-tip={`Owning a car costs ${usd(t.carOwnPerceived / t.minPerUsd)} a trip, which counts as ${t.carOwnPerceived.toFixed(1)} min`}>
            <span class="grow">Car ownership</span>
            <span class="val">{min(t.carOwnPerceived)}</span>
          </li>
        )}
        <li data-tip={`The drive counts 1.33 times (${t.drivePerceived.toFixed(1)} min) plus parking${t.carOwnPerceived > 0 ? " and car ownership" : ""}`}>
          <span class="grow">Counts as</span>
          <span class="val">{min(t.roadPerceived)}</span>
        </li>
      </ul>
      <ByPeriod periods={trip.periods} words={words} />
    </>
  );
}
