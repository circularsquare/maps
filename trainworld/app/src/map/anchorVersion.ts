// The draw anchor's version on the main thread (T-317), as a signal: it changes when `anchor()`
// (map/geo.ts) switches, which the clock client does when the state built at a new anchor arrives
// (workers/clockClient.ts). The demand views rebuild what they hold in local units when it changes.
import { signal } from "@preact/signals";

export const anchorVersion = signal(0);
