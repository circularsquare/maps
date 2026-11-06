import { t } from "../i18n";
// Station upgrades (T-1281, notes/T-1281.md): the bits are `sim/src/track/upgrades.rs`'s.

export const UPGRADE = { terminal: 1, parkRide: 2, waitingRoom: 4, passing: 8 } as const;

/** In the station panel's order; the clock worker's `upgradeCost` answer follows it too. */
export const UPGRADE_BITS: number[] = [UPGRADE.terminal, UPGRADE.parkRide, UPGRADE.waitingRoom, UPGRADE.passing];

/** Set on a station building's members in demand's snapshot only, never bought or saved (T-2220,
 * `upgrades::IN_BUILDING`): a change there counts 0.8. */
export const IN_BUILDING = 16;

export interface UpgradeText {
  bit: number;
  name: string;
  /** what it does, one sentence for its hover */
  does: string;
  /** the same, built for a whole complex (T-1528); absent: not offered there */
  doesAll?: string;
}

export const UPGRADE_TEXT: UpgradeText[] = [
  { bit: UPGRADE.terminal, get name() { return t("upgrades.terminal"); }, get does() { return t("upgrades.terminalDoes"); } },
  { bit: UPGRADE.parkRide, get name() { return t("upgrades.parking"); }, get does() { return t("upgrades.parkingDoes"); },
    get doesAll() { return t("upgrades.parkingDoesAll"); } },
  { bit: UPGRADE.waitingRoom, get name() { return t("upgrades.waiting"); }, get does() { return t("upgrades.waitingDoes"); },
    get doesAll() { return t("upgrades.waitingDoesAll"); } },
  // T-2232: Anita's wording
  { bit: UPGRADE.passing, get name() { return t("upgrades.passing"); }, get does() { return t("upgrades.passingDoes"); } },
];

/** A passage's price (sim `upgrades::passage_price`), US$M: a base plus a rate a km of the straight line. */
export const PASSAGE_BASE = 10, PASSAGE_PER_KM = 90;
export const passagePrice = (m: number) => PASSAGE_BASE + PASSAGE_PER_KM * m / 1000;

export const hasUpgrade = (bits: number | undefined, bit: number) => ((bits ?? 0) & bit) !== 0;
