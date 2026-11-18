// T-3493: the corridor table's shares of the longer trips not by rail that fly and that drive (game/demand.ts
// `CorridorRows`), 16 bits each: a step of 1 / SHARE_ONE, about 0.0015% of a corridor's trips. As 32-bit floats
// they were 62 MB of the page on a one-city game (8.2M rows).
export const SHARE_ONE = 65534;
/** a share that is NaN (near places on two landmasses) */
export const SHARE_NAN = 65535;
export const shareOf = (u: number): number => (u === SHARE_NAN ? NaN : u / SHARE_ONE);
export const shareCode = (s: number): number => (s !== s ? SHARE_NAN : Math.round(Math.min(1, Math.max(0, s)) * SHARE_ONE));
