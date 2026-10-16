// Loading and readiness (T-149, SPEC 7). The demand pool loads only the city packs a game needs
// (the viewed city and every city with a built station in reach) and the regional market; this
// is what the UI reads about that. Written by workers/demandClient.ts; read by the loading screen
// (ui/LoadingScreen.tsx), the top bar and the demand selector.
//
// A loaded game (the page opening, a file, a browser slot, a new game) holds the clock until the
// zone in view has its final figures from its fresh solve (T-347, T-421: its cluster's final may
// land well before the rest's, whose zones then show their own loading bars). Input is never blocked:
// the player can look around and build while demand loads. Only a failed load puts up a screen
// (ui/LoadingScreen.tsx).

import { signal } from "@preact/signals";
import { clock } from "./clock";
import { cityById, type CityId } from "./cities";

export interface LoadProgress {
  /** the cities being loaded, the viewed one first */
  cities: CityId[];
  /** 0-1, the share of the load's expected time done (T-243: each step weighted by how long it is
   * expected to take, moving with time inside a step; workers/demandClient.ts) */
  fraction: number;
  /** what is running now, in plain words */
  step: string;
  /** the startup load (the first game's) */
  startup: boolean;
}

/** What the demand pool is loading now; null when nothing is. */
export const loadProgress = signal<LoadProgress | null>(null);
/** T-343: each zone being loaded whose figures have not arrived yet, with its share done (0-1).
 * A zone of the load above has that load's share; a zone opened for its bubbles only, its own. */
export const zoneLoads = signal<ReadonlyMap<CityId, number>>(new Map());
/** "loading" until the first game's demand has its first estimate, "failed" if it could not load. */
export const startup = signal<"loading" | "ready" | "failed">("loading");
/** T-318: nothing needs demand now: the view is over dead space and no built station reaches a
 * zone. Nothing loads, no card or status says "Loading data", and the clock is not held for it. */
export const demandIdle = signal(false);
/** Why demand could not load (the pool is stopped); null while it works. */
export const loadFailure = signal<string | null>(null);
/** T-484, T-492: zones whose pack could not be downloaded or opened, with why and whether it is tried
 * again. Demand carries on without them (workers/demandClient.ts); a zone leaves once its pack opens. */
export const zoneFailures = signal<ReadonlyMap<CityId, string>>(new Map());

/** The clock waits for the next game's demand: set at startup and by every whole-game load. */
let holdArmed = true;

/** A whole game was loaded or started (workers/clockClient.ts): hold the clock until its demand is done. */
export function holdForGameLoad() {
  if (loadFailure.peek()) return; // no demand to wait for
  holdArmed = true;
  clock?.setHeld(true);
}

/** The demand client takes the hold for the network it is about to solve (true once per hold). */
export function takeGameLoadHold(): boolean {
  const armed = holdArmed;
  holdArmed = false;
  return armed;
}

/** The held game's demand has its final result: the clock runs. */
export function loadComplete() {
  clock?.setHeld(false);
  if (startup.peek() === "loading") startup.value = "ready";
}

/** Demand could not load. At startup the loading screen says so and offers a way on; later the
 * clock is released at once, since there is no result to wait for. */
export function loadFailed(why: string) {
  loadFailure.value = why;
  loadProgress.value = null;
  holdArmed = false;
  if (startup.peek() === "loading") startup.value = "failed";
  // after startup; at a failed startup the player's choice releases it (every worker may report)
  else if (startup.peek() === "ready") clock?.setHeld(false);
}

/** Carry on without demand after a failed startup (the loading screen's second button). */
export function continueWithoutDemand() {
  clock?.setHeld(false);
  startup.value = "ready";
}

/** "New York", "New York and Philadelphia", "New York, Philadelphia and Boston". */
export function cityNames(ids: CityId[]): string {
  const names = ids.map((id) => cityById(id).name);
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
