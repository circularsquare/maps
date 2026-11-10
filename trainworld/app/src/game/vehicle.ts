// What a line runs and the noun for it (T-3054): train, tram, bus or boat. Every panel and hover that
// names a line's vehicle takes the noun from here, so a tram is never called a train.

import { t } from "../i18n";
import { BOAT_CARS, BUS_CARS } from "./boarding";
import type { LineStats } from "./types";

export type Vehicle = "train" | "tram" | "bus" | "boat";

export const vehicleOf = (st: LineStats): Vehicle => (st.bus?.ferry ? "boat" : st.bus ? "bus" : st.tram ? "tram" : "train");

/** A tram runs on rails and is sized in cars like a train, so the two share every rule except their noun. */
export const onRails = (v: Vehicle): boolean => v === "train" || v === "tram";

/** A vehicle's size in train cars, for its seats, rated load and crush (`capacity`). */
export const vehicleCars = (st: LineStats): number => {
  const v = vehicleOf(st);
  return v === "boat" ? BOAT_CARS : v === "bus" ? BUS_CARS : st.cars;
};

/** Other words a line's name may start with for its own mode, besides the player's language's noun. */
const MODE_WORDS: Record<Vehicle, string[]> = { train: ["train"], tram: ["tram"], bus: ["bus"], boat: ["boat", "ferry"] };

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** A line's name for a title or hover that ends in the vehicle's noun: when the name starts with its own
 * mode's word, in English or in the player's language ("Bus 53" on a bus, any case), the word goes and
 * the noun follows ("53 bus"). A name that is only that word stays whole. */
export function nameBeforeNoun(name: string, v: Vehicle): string {
  const words = [...MODE_WORDS[v], t("vehicle.noun", { vehicle: v })].filter(Boolean).map(escape);
  const m = new RegExp(`^(?:${words.join("|")})(?!\\p{L})[\\s\\-\\u2013:]*`, "iu").exec(name);
  const rest = m ? name.slice(m[0].length).trim() : name;
  return m && rest ? rest : name;
}
