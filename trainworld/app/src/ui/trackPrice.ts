// The track tool's price at a level and its level buttons' hovers (T-2271), for the track bar (T-2282).

import { distUnit, levelName, levelUpkeepLine, millions, perDistance } from "../game/format";
import type { MoneyState } from "../workers/protocol";
import { edgeUpkeep } from "../game/regionMoney";
import type { Level } from "../game/types";
import { kindLevelMult, TRACK_SPECS } from "../game/trackKinds";
import { snapStreets, type TrackType } from "../game/ui";
import { levelKeyName } from "./toolKeys";
import { t, tNumber } from "../i18n";

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
export const BASE_M_PER_KM = 75;
/** Single track's cost multiple (T-3093); its upkeep is the money state's `upkeepSingle`. */
const SINGLE_MULT = 0.6;

/** The cost multiplier at a level, high-speed track's included; quad track (T-2931) is two double tracks'. */
export const trackMult = (l: Level, t: TrackType, quad = false, single = false) =>
  kindLevelMult(t === "tram" ? tramKind() : t === "hsr" ? "highSpeed" : "regular", l) * (quad ? 2 : single ? SINGLE_MULT : 1);
/** T-2927: the tram track the tool lays: on streets while the street snap is on */
export const tramKind = () => (snapStreets.value ? "streetTram" : "tram") as "streetTram" | "tram";
/** "1.0x", "1.50x" for high-speed track */
const multOf = (v: number, t: TrackType) => tNumber(v, { minimumFractionDigits: t === "hsr" ? 2 : 1, maximumFractionDigits: t === "hsr" ? 2 : 1 }) + "x";   // i18n-ignore
export const multText = (l: Level, t: TrackType, quad = false, single = false) => multOf(trackMult(l, t, quad, single), t);
/** The price in US$M per shown distance unit, to a tenth. */
export const unitPrice = (l: Level, t: TrackType, quad = false, single = false) => Math.round(perDistance(BASE_M_PER_KM * trackMult(l, t, quad, single)) * 10) / 10;

/** T-2676: a level's upkeep as a multiple of a viaduct's (ordinary double track at +1, the level
 * whose cost multiple is 1.0x too), high-speed track's included: 0.7x on the ground, 2.0x in cut and
 * cover, 2.6x in a bored tunnel at T-2619's rates. */
export function upkeepMult(l: Level, t: TrackType, m: Parameters<typeof edgeUpkeep>[1], quad = false, single = false): number {
  const at = (lev: Level, hsr: boolean, tracks = 2) => edgeUpkeep({ built: true, tracks, hsr, lengthM: 1000, levelMin: lev, levelMax: lev }, m);
  const base = at(1, false);
  return base > 0 ? (at(l, t === "hsr", single ? 1 : 2) / base) * (quad ? 2 : 1) * (t === "tram" ? TRACK_SPECS[tramKind()].upkeep : 1) : 1;
}
/** "0.7x upkeep" ("1.50x" for high-speed track, as the cost multiple), or "" before the money has come. */
export function upkeepAt(l: Level, t: TrackType, m: MoneyState | null, quad = false, single = false): string {
  return m ? levelUpkeepLine(multOf(upkeepMult(l, t, m, quad, single), t)) : "";
}

/** A level button's hover: "Viaduct (Shift+3)", "1.0x cost, $75M per km", "1.0x upkeep". */
export function levelTip(l: Level, tt: TrackType, m: MoneyState | null, quad = false, single = false): string {
  return [`${levelName(l)} (${levelKeyName(l)})`, t("trackPrice.cost", { mult: multText(l, tt, quad, single), price: millions(unitPrice(l, tt, quad, single)), unit: distUnit() }), upkeepAt(l, tt, m, quad, single)].filter(Boolean).join("\n");
}
