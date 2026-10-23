// T-866: the trip explorer between zones. The inter-city trips between a bubble selection (or the
// trip tool's start) and a far place or 6 km area, as the regional market chooses them
// (game/trip.ts `MarketPeriod`): every area pair between the two ends added up by its trips. The
// train's routes are the market's own through the period's strategies; the walks or drives to the
// stations are known only as felt minutes, averaged over the area.

import { count, roughDistance } from "../../game/format";
import { MODE_COLOURS, modeColours } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { intercityPlaceName } from "../../game/demand";
import { peopleOf, tripPeriod, wholePeople, type MarketEnd, type MarketPeriod, type MarketShown, type TripRoute } from "../../game/trip";
import { shownDayPhrase } from "../../game/ui";
import { world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { Chip } from "../widgets";

const min = (m: number) => `${Math.max(1, Math.round(m))} min`;
const pct = (v: number) => (v > 0 && v < 0.005 ? "<1%" : `${Math.round(100 * v)}%`);
const num = (v: number) => String(Math.round(v * 10) / 10);
const exact = (n: number) => Math.round(n).toLocaleString("en");
/** T-911: shares as whole people out of `n` (null or 0: the model's own shares). */
const roundTo = (shares: number[], n: number | null, pooled = -1) => (n ? wholePeople(shares, n, pooled).map((p) => p / n) : shares);

const NO_RAIL: Record<string, string> = {
  farClosed: "The far end is not open in this game, so trains to it are not worked out.",
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
  const st = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const ln = new Map(w.save.lines.map((l) => [l.id, l]));
  const pair = t.pairs[r.pair];
  const first = r.legs[0], last = r.legs[r.legs.length - 1];
  const name = (id: string) => st.get(id) ?? "a station";
  return (
    <ul class="rows trip-legs">
      {pair && first && (
        <li data-tip={`On foot or by car, felt as ${num(pair.accessPerceived)} min on average over the homes at this end`}>
          <span class="grow">To {name(first.from)}</span>
          <span class="val muted">{min(pair.accessPerceived)}</span>
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
        return (
          <li data-tip={`Wait ${l.waitMin.toFixed(1)} min (half the time between the useful trains), then ${l.min.toFixed(1)} min on the train over ${l.stops} ${l.stops === 1 ? "stop" : "stops"}`}>
            {line && <Chip line={line} size="sm" />}
            <span class="grow">{i > 0 && r.legs[i - 1].kind === "ride" ? "Change to " : ""}{line?.name ?? "A line"} to {name(l.to)}</span>
            <span class="val">{min(l.waitMin + l.min)}</span>
          </li>
        );
      })}
      {pair && last && (
        <li data-tip={`On foot or dropped off, felt as ${num(pair.egressPerceived)} min on average over the places at this end`}>
          <span class="grow">From {name(last.to)}</span>
          <span class="val muted">{min(pair.egressPerceived)}</span>
        </li>
      )}
    </ul>
  );
}

/** The other routes the market's train riders take, by share. */
function OtherRoutes({ t, shown, riders }: { t: MarketPeriod; shown: number[]; riders: number | null }) {
  const w = world.value!;
  const st = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const ln = new Map(w.save.lines.map((l) => [l.id, l]));
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
          const rides = r.legs.filter((l) => l.kind === "ride");
          const from = rides[0] ? st.get(rides[0].from) : "", to = r.legs.length ? st.get(r.legs[r.legs.length - 1].to) : "";
          const names = rides.map((l) => ln.get(l.line)?.name ?? "a line").join(" then ");
          return (
            <li data-tip={`${names} from ${from} to ${to}: ${(100 * r.share).toFixed(1)}% of these trips' train riders${riders ? `, ${pct(v)} of the ${count(riders)} counted here` : ""}, ${min(stationTimes(r).total)} station to station`}>
              {rides.map((l) => { const line = ln.get(l.line); return line && <Chip line={line} size="sm" />; })}
              <span class="grow">{names} to {to}</span>
              <span class="val w-num">{pct(v)}</span>
            </li>
          );
        })}
        {rest >= 0.005 && (
          <li class="muted" data-tip={`The train riders spread over ${t.pairs.length} pairs of stations near the two ends. Routes under 0.5% of them are not listed.`}>
            <span class="grow">Smaller routes</span>
            <span class="val w-num">{pct(rest)}</span>
          </li>
        )}
      </ul>
    </>
  );
}

export function MarketTrip({ m }: { m: MarketShown }) {
  const q = shownPeriod.value;
  const t = m.periods[q];
  const day = shownDayPhrase.value;
  const period = PERIODS[q].name;
  const far = marketFarName(m);
  const sum = (k: "trips" | "rail" | "air" | "car") => m.periods.reduce((s, p) => s + (p?.[k] ?? 0), 0);
  const all = sum("trips");
  if (!t) return <p class="muted">Calculating demand.</p>;
  // T-911: in whole trips; the hovers keep the model's shares
  const model = all > 0 ? [sum("rail") / all, sum("air") / all, sum("car") / all] : [];
  const whole = wholePeople(model, peopleOf(all)).map((p) => p / peopleOf(all));
  const shares: [string, number, string, number][] = all > 0
    ? [["Train", whole[0], MODE_COLOURS.train, model[0]], ["Plane", whole[1], modeColours().other, model[1]], ["Car", whole[2], MODE_COLOURS.drive, model[2]]]
    : [];
  // the period's train riders in whole trips, for the route shares (none: the model's)
  const riders = Math.round(t.rail) || null;
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
  return (
    <>
      <p class="muted small" data-tip={`${all >= 0.5 ? exact(all) : num(all)} inter-city trips ${day} between the two ends, both ways: visits, business and leisure trips of 25 km and more. Commutes between the zones are left out.`}>
        {trips} {day}, {roughDistance(1000 * (t.km || 0))} apart
      </p>
      {shares.length > 0 && (
        <div class="trip-modes">
          <div class="bar trip-bar" role="img" aria-label={shares.map(([n, v]) => `${n} ${pct(v)}`).join(", ")}>
            {shares.map(([n, v, c]) => v > 0 && <i class="seg" style={{ width: `${100 * v}%`, background: c }} data-tip={`${n}: ${pct(v)} of ${of}`} />)}
          </div>
          <ul class="rows">
            {shares.filter(([n, v]) => n !== "Plane" || v > 0).map(([n, v, c, e]) => (
              <li data-tip={`${n}: ${pct(v)} of ${of}\nModel: ${(100 * e).toFixed(1)}%, rounded here to whole trips`}>
                <i class="mk" style={{ background: c }} />
                <span class="grow">{n}</span>
                <span class="val w-num">{pct(v)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div class="h">By train<span class="right muted small" data-tip={repTip}>Inter-city, {period.toLowerCase()}</span></div>
      {t.why || !main || !tm ? <p class="muted">{NO_RAIL[t.why || "noRail"]}</p> : (
        <>
          <Legs t={t} r={main} />
          <ul class="rows">
            <li data-tip={`Base fare $${(t.baseFarePerceived / t.minPerUsd).toFixed(2)} once a trip, plus the distance fare`}>
              <span class="grow">Fare</span>
              <span class="val">${fare.toFixed(2)}</span>
            </li>
            <li data-tip={`Station to station: ${[["wait", tm.wait], ["ride", tm.ride], ["walk between stations", tm.xfer]].filter(([, v]) => (v as number) > 0).map(([l, v]) => `${l} ${num(v as number)}`).join(" + ")} = ${Math.round(tm.total)} min`}>
              <span class="grow">Time</span>
              <span class="val">{min(tm.total)}</span>
            </li>
            <li data-tip={`The minutes the model weighs against the car and the plane, door to door, averaged over the area pairs the train serves (${pct(t.served)} of the trips)`}>
              <span class="grow">Effective time</span>
              <span class="val">{min(t.railPerceived)}</span>
            </li>
          </ul>
          <p class="muted small" data-tip={`The train riders spread over ${t.pairs.length} pairs of stations near the two ends`}>
            {routeShares[0] >= 0.995 ? "Every train user on these trips takes this route." : `${pct(routeShares[0])} of the train users on these trips take this route.`}
          </p>
          <OtherRoutes t={t} shown={routeShares} riders={riders} />
        </>
      )}
      <div class="h">By car</div>
      <ul class="rows">
        {Number.isFinite(t.carPerceived) ? (
          <li data-tip={`Felt minutes of the drive, its costs and parking at the far end, averaged over the area pairs`}>
            <span class="grow">Effective time</span>
            <span class="val">{min(t.carPerceived)}</span>
          </li>
        ) : <li><span class="grow muted">No road joins the two ends.</span></li>}
      </ul>
      <div class="h">Train by period<span class="right muted small">click to show</span></div>
      <ul class="rows pick">
        {PERIODS.map((p, i) => {
          const x = m.periods[i];
          const r0 = x && x.trips > 0 ? x.rail / x.trips : 0;
          const r = roundTo([r0, 1 - r0], x ? Math.round(x.trips) || null : null)[0];
          return (
            <li class={i === q ? "on" : ""} onClick={() => (tripPeriod.value = i)}
              data-tip={x ? `${p.name}: ${exact(x.trips)} inter-city trips, train ${pct(r)}` : p.name}>
              <span class="grow">{p.name}</span>
              <span class="trip-hbar"><i class="hbar" style={{ width: `${Math.round(60 * r)}px`, background: MODE_COLOURS.train }} /></span>
              <span class="val w-num">{x ? pct(r) : ""}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}
