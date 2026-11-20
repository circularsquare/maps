// T-3892: ferry boat sizes (sim `ferry::Boat`, notes/T-3892.md 2): what each holds and does, and its fleet,
// price and running cost from the money state. Every reader of a boat's money goes through here.

import type { MoneyState } from "../workers/protocol";
import { MEDIUM, type Boat } from "./fareCurve";

export { MEDIUM, type Boat };

/** Per boat size, Small, Medium, Large: riders at 100% load, top speed in open water (km/h), its riders as
 * train cars' (demand's cars per boat, the sim's `Boat::cars`). The sim holds the rules; these are for text. */
export const BOATS: readonly { riders: number; kmh: number; cars: number }[] = [
  { riders: 160, kmh: 65, cars: 1 },
  { riders: 320, kmh: 45, cars: 2 },
  { riders: 1280, kmh: 45, cars: 8 },
];

/** A line's boat size from its row (Medium for a line with none). */
export function boatOf(bus: { ferry?: { boat?: Boat } } | undefined): Boat {
  return bus?.ferry?.boat ?? MEDIUM;
}

type BoatRates = Pick<MoneyState, "runCostBoatKm" | "runCostSmallBoatKm" | "runCostLargeBoatKm">;
/** Running cost a boat-km of a boat size, real US$. */
export function boatRunKm(m: BoatRates, boat: Boat): number {
  return boat === 0 ? m.runCostSmallBoatKm : boat === 2 ? m.runCostLargeBoatKm : m.runCostBoatKm;
}

type BoatFleet = Pick<MoneyState, "fleetFerry" | "boatsNeeded" | "boatPrice" | "fleetFerrySmall" | "boatsNeededSmall" | "smallBoatPrice" | "fleetFerryLarge" | "boatsNeededLarge" | "largeBoatPrice">;
/** A boat size's fleet: owned, needed by the running lines, and a boat's price US$M. */
export function boatStock(m: BoatFleet, boat: Boat): { owned: number; needed: number; price: number } {
  if (boat === 0) return { owned: m.fleetFerrySmall, needed: m.boatsNeededSmall, price: m.smallBoatPrice };
  if (boat === 2) return { owned: m.fleetFerryLarge, needed: m.boatsNeededLarge, price: m.largeBoatPrice };
  return { owned: m.fleetFerry, needed: m.boatsNeeded, price: m.boatPrice };
}
/** `boatStock` of a ferry line's own boat size, as [owned, needed, price]. */
export function boatFleet(m: BoatFleet, st: { bus?: { ferry?: { boat?: Boat } } }): [number, number, number] {
  const s = boatStock(m, boatOf(st.bus));
  return [s.owned, s.needed, s.price];
}
