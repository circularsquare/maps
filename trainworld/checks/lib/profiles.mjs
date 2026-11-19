// T-3542, T-3346 stage 2: the app's train profile strides (app/src/map/profiles.ts `SLOTS`, `PROFILE_VARS`), for
// checks that decode a trip's profile outside the page: profile `((line * 2 + run) * SLOTS + slot) * PROFILE_VARS + var`.
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('../../app/src/map/profiles.ts', import.meta.url), 'utf8');
export const SLOTS = Number(/export const SLOTS = (\d+);/.exec(src)[1]);
export const PROFILE_VARS = Number(/export const PROFILE_VARS = (\d+);/.exec(src)[1]);
export const PROFILES_PER_RUN = SLOTS * PROFILE_VARS;
export const PROFILES_PER_LINE = 2 * PROFILES_PER_RUN;
