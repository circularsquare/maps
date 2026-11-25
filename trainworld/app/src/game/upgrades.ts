import { t } from "../i18n";
// Station upgrades (T-1281, notes/T-1281.md): the bits are `sim/src/track/upgrades.rs`'s.

export const UPGRADE = { terminal: 1, parkRide: 2, waitingRoom: 4, passing: 8 } as const;

/** In the station panel's order; the clock worker's `upgradeCost` answer follows it too. */
export const UPGRADE_BITS: number[] = [UPGRADE.terminal, UPGRADE.parkRide, UPGRADE.waitingRoom, UPGRADE.passing];

/** Set on a station building's members in demand's snapshot only, never bought or saved (T-2220,
 * `upgrades::IN_BUILDING`): a change there counts 0.8. */
export const IN_BUILDING = 16;
/** Set on BRT stations in demand's snapshot only (T-3378, `upgrades::BRT_STATION`): a wait there counts as in a
 * waiting room, and it takes a tram stop's slot among a home's nearest stations. */
export const BRT_STATION = 32;

export interface UpgradeText {
  bit: number;
  name: string;
  /** what it does, one sentence for its hover */
  does: (tram?: boolean, port?: boolean) => string;
  /** the same, built for a whole complex (T-1528); absent: not offered there */
  doesAll?: (tram?: boolean) => string;
}

export const UPGRADE_TEXT: UpgradeText[] = [
  { bit: UPGRADE.terminal, get name() { return t("upgrades.terminal"); }, does: () => t("upgrades.terminalDoes") },
  { bit: UPGRADE.parkRide, get name() { return t("upgrades.parking"); }, does: () => t("upgrades.parkingDoes"),
    doesAll: () => t("upgrades.parkingDoesAll") },
  { bit: UPGRADE.waitingRoom, get name() { return t("upgrades.waiting"); }, does: (tram, port) => t("upgrades.waitingDoes", { vehicle: port ? "boat" : tram ? "tram" : "train" }),
    doesAll: (tram) => t("upgrades.waitingDoesAll", { vehicle: tram ? "tram" : "train" }) },
  // T-2232: Anita's wording
  { bit: UPGRADE.passing, get name() { return t("upgrades.passing"); }, does: () => t("upgrades.passingDoes") },
];

/** A passage's price (sim `upgrades::passage_price`), US$M: a base plus a rate a km of the straight line. */
export const PASSAGE_BASE = 10, PASSAGE_PER_KM = 90;
export const passagePrice = (m: number) => PASSAGE_BASE + PASSAGE_PER_KM * m / 1000;

/** T-3922: the upkeep a game day, US$, of what costs `priceM` US$M to buy on a station (`upgrades::upkeep`); 0
 * before the money has come. */
export const upgradeUpkeep = (priceM: number, m: { upkeepUpgradeShare: number } | null | undefined, bit?: number) =>
  m ? priceM * 1e6 * (m.upkeepUpgradeShare ?? 0) * (bit === UPGRADE.waitingRoom ? WAITING_ROOM_UPKEEP_FACTOR : 1) : 0;

/** T-4342: a waiting room's share of the upgrade upkeep rate (sim `upgrades::WAITING_ROOM_UPKEEP_FACTOR`). */
export const WAITING_ROOM_UPKEEP_FACTOR = 0.5;

export const hasUpgrade =(bits: number | undefined, bit: number) => ((bits ?? 0) & bit) !== 0;
