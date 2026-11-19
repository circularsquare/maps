// Train profile indices (T-3542). Each (line, run, slot) has a profile: profile
// `(line * 2 + run) * SLOTS + slot`, one entry of `TrackApi.profile_meta` (4 floats) each, laid out in
// that order. A slot is a demand period (game/periods.ts `PERIODS`, slot = period); periods every line
// runs alike share the first one's (the sim's `Network::period_slot`), and a trip names the slot it
// runs. A slot no period uses has no phases.

/** The sim's `params::SLOTS` (the clock worker checks it against `TrackApi.profile_slots`). */
export const SLOTS = 5;
/** Profiles a line has: both runs' slots. */
export const PROFILES_PER_LINE = 2 * SLOTS;
/** `profile_meta` floats per line. */
export const META_PER_LINE = PROFILES_PER_LINE * 4;
/** The slot each row (high, medium, low) is kept in on a line of three rows: the sim's `ROW_SLOT`. */
export const ROW_SLOT = [0, 3, 4] as const;

export const profileOf = (line: number, run: number, slot: number): number => (line * 2 + run) * SLOTS + slot;
export const profileLine = (p: number): number => Math.floor(p / PROFILES_PER_LINE);
export const profileRun = (p: number): number => Math.floor(p / SLOTS) % 2;
export const profileSlot = (p: number): number => p % SLOTS;
/** The row (high, medium, low) whose trains a trip's slot runs, on a line of three rows. */
export const profileRow = (p: number): number => ROW_SLOT.filter((k) => k <= profileSlot(p)).length - 1;
/** For checks, which decode a trip's profile in the page (`window.tw.profiles`). */
export const PROFILES = { SLOTS, PROFILES_PER_LINE, profileOf, profileLine, profileRun, profileSlot, profileRow };
