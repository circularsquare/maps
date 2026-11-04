// T-2395: the game's home, where the camera starts when the game loads: a zone id, or `WORLD_HOME`
// (every built station in view, zoomed out). Kept in the save (`SaveHeader.home`); set from the
// city view's and the all cities view's home buttons (workers/clockClient.ts `setHome`).

import { signal } from "@preact/signals";

/** The game in hand's home; null for none (the load fits as `startOpenFit` picks). */
export const home = signal<string | null>(null);
