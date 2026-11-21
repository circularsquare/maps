// T-3887: the fares a new game in `country` gives the demand model and the Money tab, worked out by the game's own
// functions, for tools/frontier.py (bundled with rolldown and run in node; prints one JSON line). The made-up city's
// line is charged these: what the demand worker sets (`set_fares`, `set_fare_curve`, `set_fare_weight`,
// `set_transfer_fare`) and the multiples a bus's and a high-speed line's rides pay above the base.
import "./frontier_shim.js"; // first: the browser globals the game's modules touch as they load
import { busFare, curveKnots, faresOf, hsrFare, scaledKnots, START_FARE, transferFare, type Fares } from "../app/src/game/fareCurve";
import { roundedDefault, startNodes } from "../app/src/game/fareDefault";
import { FARE_BASELINE, FARE_E, fareScale, fareWeight, incomeScale, START } from "../app/src/game/income";

const country = process.argv[2] ?? "US";
// a new game's fares where it starts (workers/clockFares.ts `startFares`)
const fare: Fares = { ...faresOf({ nodes: startNodes(country) }, START_FARE), scales: { [START]: incomeScale(country) }, baseline: FARE_BASELINE, basis: "usd", income: FARE_E } as Fares;
// as the demand worker sets them for a city in `country` (workers/demand.worker.ts `setUpPops`)
const scale = fareScale(country, fare.scales);
const rounded = roundedDefault(fare, country);
const base = rounded ? rounded.base : fare.base * scale;
const perKm = rounded ? rounded.perKm : fare.perKm * scale;
const knots = rounded ? curveKnots(rounded) : scaledKnots(fare, scale);
console.log(JSON.stringify({
  country, base, perKm, knots: Array.from(knots), weight: fareWeight(country),
  transfer: Math.min(transferFare(fare) * scale, base), bus: busFare(fare), hsr: hsrFare(fare),
}));
