// Station upgrades (T-1281, notes/T-1281.md): the bits are `sim/src/track/upgrades.rs`'s.

export const UPGRADE = { terminal: 1, parkRide: 2, waitingRoom: 4 } as const;

/** In the station panel's order; the clock worker's `upgradeCost` answer follows it too. */
export const UPGRADE_BITS: number[] = [UPGRADE.terminal, UPGRADE.parkRide, UPGRADE.waitingRoom];

export interface UpgradeText {
  bit: number;
  name: string;
  /** what it does, one sentence for its hover */
  does: string;
  /** the same, built for a whole complex (T-1528); absent: not offered there */
  doesAll?: string;
}

export const UPGRADE_TEXT: UpgradeText[] = [
  { bit: UPGRADE.terminal, name: "Terminal", does: "Upgrade turning capacity from 15 trains per hour to 30 trains per hour." },
  { bit: UPGRADE.parkRide, name: "Parking lot", does: "Traveling to this station from far away feels 10 minutes shorter.",
    doesAll: "Traveling to any station in this complex from far away feels 10 minutes shorter." },
  { bit: UPGRADE.waitingRoom, name: "Waiting room", does: "Waiting here for a train feels 20% shorter.",
    doesAll: "Waiting at any station in this complex for a train feels 20% shorter." },
];

/** A passage's price (sim `upgrades::passage_price`), US$M: a base plus a rate a km of the straight line. */
export const PASSAGE_BASE = 10, PASSAGE_PER_KM = 90;
export const passagePrice = (m: number) => PASSAGE_BASE + PASSAGE_PER_KM * m / 1000;

export const hasUpgrade = (bits: number | undefined, bit: number) => ((bits ?? 0) & bit) !== 0;
