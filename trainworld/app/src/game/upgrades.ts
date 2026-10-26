// Station upgrades (T-1281, notes/T-1281.md): the bits are `sim/src/track/upgrades.rs`'s.

export const UPGRADE = { terminal: 1, parkRide: 2, waitingRoom: 4 } as const;

/** In the station panel's order; the clock worker's `upgradeCost` answer follows it too. */
export const UPGRADE_BITS: number[] = [UPGRADE.terminal, UPGRADE.parkRide, UPGRADE.waitingRoom];

export interface UpgradeText {
  bit: number;
  name: string;
  /** what it does, one sentence for its hover */
  does: string;
}

export const UPGRADE_TEXT: UpgradeText[] = [
  { bit: UPGRADE.terminal, name: "Terminal", does: "Trains turn in two sidings past the platform, up to 30 an hour, and the platforms stay free for other trains." },
  { bit: UPGRADE.parkRide, name: "Park and ride", does: "A car park, bus bays and bike racks. Riding to this station costs 10 perceived minutes less." },
  { bit: UPGRADE.waitingRoom, name: "Waiting room", does: "Waits here count at 0.8 of their minutes when people choose how to travel." },
];

/** A passage's price (sim `upgrades::passage_price`), US$M: a base plus a rate a km of the straight line. */
export const PASSAGE_BASE = 10, PASSAGE_PER_KM = 90;
export const passagePrice = (m: number) => PASSAGE_BASE + PASSAGE_PER_KM * m / 1000;

export const hasUpgrade = (bits: number | undefined, bit: number) => ((bits ?? 0) & bit) !== 0;
