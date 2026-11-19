// T-3542: the app's train profile stride (app/src/map/profiles.ts `SLOTS`), for checks that decode a trip's
// profile inside the page: profile `(line * 2 + run) * SLOTS + slot`.
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('../../app/src/map/profiles.ts', import.meta.url), 'utf8');
export const SLOTS = Number(/export const SLOTS = (\d+);/.exec(src)[1]);
export const PROFILES_PER_LINE = 2 * SLOTS;
