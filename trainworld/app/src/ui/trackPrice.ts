// The track tool's price at a level and its level buttons' hovers (T-2271), for the track bar (T-2282).

import { distUnit, levelName, levelUpkeepLine, perDistance } from "../game/format";
import type { MoneyState } from "../workers/protocol";
import { edgeUpkeep } from "../game/regionMoney";
import type { Level } from "../game/types";
import type { TrackType } from "../game/ui";
import { levelKeyName } from "./toolKeys";
import { t, tNumber } from "../i18n";

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
const BASE_M_PER_KM = 75;
const MULT: Record<Level, number> = { 3: 2.0, 2: 1.5, 1: 1.0, 0: 0.5, [-1]: 1.0, [-2]: 1.6, [-3]: 2.0 };
/** high-speed track's price over double track (params.rs `HSR_COST`, T-817) */
const HSR_COST = 1.5;

/** The cost multiplier at a level, high-speed track's included. */
export const trackMult = (l: Level, t: TrackType) => MULT[l] * (t === "hsr" ? HSR_COST : 1);
/** "1.0x", "1.50x" for high-speed track */
export const multText = (l: Level, t: TrackType) => tNumber(trackMult(l, t), { minimumFractionDigits: t === "hsr" ? 2 : 1, maximumFractionDigits: t === "hsr" ? 2 : 1 }) + "x";   // i18n-ignore
/** The price in US$M per shown distance unit, to a tenth. */
export const unitPrice = (l: Level, t: TrackType) => Math.round(perDistance(BASE_M_PER_KM * trackMult(l, t)) * 10) / 10;

/** "Upkeep $20k per km per day", or "" before the money has come. */
export function upkeepAt(l: Level, t: TrackType, m: MoneyState | null): string {
  return m ? levelUpkeepLine(edgeUpkeep({ built: true, tracks: 2, hsr: t === "hsr", lengthM: 1000, levelMin: l, levelMax: l }, m)) : "";
}

/** A level button's hover: "Viaduct (Shift+3)", "1.0x cost, $75M per km", "Upkeep $20k per km per day". */
export function levelTip(l: Level, tt: TrackType, m: MoneyState | null): string {
  return [`${levelName(l)} (${levelKeyName(l)})`, t("trackPrice.cost", { mult: multText(l, tt), price: tNumber(unitPrice(l, tt)), unit: distUnit() }), upkeepAt(l, tt, m)].filter(Boolean).join("\n");
}
