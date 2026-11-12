// T-3297: which side a new game's trains, trams and buses keep to, before the player picks: the side its start
// country's railways run on, or for a real network the side its metro runs on (the shipped saves are built so,
// pipeline/realauto.json's `left`, realnet.py's `--left`). Sources and the calls: notes/T-3297.md.

import { cityById, isCity } from "./cities";

/** Countries (ISO 3166 alpha-2) whose railways keep left: those whose roads do, but Indonesia (its railways keep
 * right); and those whose roads keep right but railways left (British-built networks, Wikipedia "Left- and
 * right-hand traffic", Rail traffic; Sweden kept left in 1967; mainland China's and South Korea's mainline). */
const LEFT_RAIL = new Set([
  // roads and railways on the left
  "AG", "AI", "AU", "BB", "BD", "BM", "BN", "BS", "BT", "BW", "CC", "CK", "CX", "CY", "DM", "FJ", "FK", "GB", "GD",
  "GG", "GY", "HK", "IE", "IM", "IN", "JE", "JM", "JP", "KE", "KI", "KN", "KY", "LC", "LK", "LS", "MO", "MS", "MT",
  "MU", "MV", "MW", "MY", "MZ", "NA", "NF", "NP", "NR", "NU", "NZ", "PG", "PK", "PN", "SB", "SC", "SG", "SH", "SR",
  "SZ", "TC", "TH", "TK", "TL", "TO", "TT", "TV", "TZ", "UG", "VC", "VG", "VI", "WS", "ZA", "ZM", "ZW",
  // roads on the right, railways on the left
  "AR", "BE", "BO", "BR", "CH", "CN", "EG", "FR", "IL", "IQ", "IT", "KH", "KR", "LA", "MA", "MC", "MM", "NG", "PE",
  "PT", "SE", "SI", "SM", "SN", "TN", "TW", "UY", "VA", "VE",
]);

/** Real networks (`game/realNetworks.json` ids) whose metro keeps left. Paris, Seoul, Taipei, São Paulo and
 * mainland China's metros keep right although their mainlines keep left; Madrid's metro keeps left although Spain's
 * railways keep right. */
const REAL_LEFT = new Set(["lon", "tyo", "osa", "sdj", "hon", "sin", "mad", "cai", "ned"]);

/** Whether railways in a country keep right (true for an unknown one). */
export function countryRightHand(country: string | null | undefined): boolean {
  return !country || !LEFT_RAIL.has(country);
}

/** A new game's running side before the player picks: the start's country's railways, a real network's metro;
 * right with no start (the world map). */
export function startRightHand(start: string | null | undefined, real: boolean): boolean {
  if (!start || !isCity(start)) return true;
  if (real) return !REAL_LEFT.has(start);
  return countryRightHand(cityById(start).country);
}
