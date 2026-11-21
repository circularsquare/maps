// Train profile indices (T-3542, T-3346 stage 2). Each (line, run, slot) has its slot's profile and
// up to `PROFILE_VARS - 1` train-by-train variants (a local's own waits at passing stations, an express
// timed behind its local): profile `((line * 2 + run) * SLOTS + slot) * PROFILE_VARS + var`, one entry
// of `TrackApi.profile_meta` (4 floats) each, laid out in that order. A slot is a demand period
// (game/periods.ts `PERIODS`, slot = period); periods every line runs alike share the first one's (the
// sim's `Network::period_slot`), and a trip names the slot and variant it runs. A slot no period uses,
// and a variant a run has not, has no phases.

/** The sim's `params::SLOTS` (the clock worker checks it against `TrackApi.profile_slots`). */
export const SLOTS = 5;
/** The sim's `timing::PROFILE_VARS` (checked against `TrackApi.profile_vars`). */
export const PROFILE_VARS = 17;
/** Profiles a run has: its slots' and their variants. */
export const PROFILES_PER_RUN = SLOTS * PROFILE_VARS;
/** Profiles a line has: both runs'. */
export const PROFILES_PER_LINE = 2 * PROFILES_PER_RUN;
/** `profile_meta` floats per line. */
export const META_PER_LINE = PROFILES_PER_LINE * 4;
/** The slot each row (high, medium, low) is kept in on a line of three rows: the sim's `ROW_SLOT`. */
export const ROW_SLOT = [0, 3, 4] as const;

export const profileOf = (line: number, run: number, slot: number, v = 0): number => ((line * 2 + run) * SLOTS + slot) * PROFILE_VARS + v;
export const profileLine = (p: number): number => Math.floor(p / PROFILES_PER_LINE);
export const profileRun = (p: number): number => Math.floor(p / PROFILES_PER_RUN) % 2;
export const profileSlot = (p: number): number => Math.floor(p / PROFILE_VARS) % SLOTS;
export const profileVar = (p: number): number => p % PROFILE_VARS;
/** The row (high, medium, low) whose trains a trip's slot runs, on a line of three rows. */
export const profileRow = (p: number): number => ROW_SLOT.filter((k) => k <= profileSlot(p)).length - 1;
/** For checks, which decode a trip's profile in the page (`window.tw.profiles`). */
export const PROFILES = { SLOTS, PROFILE_VARS, PROFILES_PER_RUN, PROFILES_PER_LINE, profileOf, profileLine, profileRun, profileSlot, profileVar, profileRow };
