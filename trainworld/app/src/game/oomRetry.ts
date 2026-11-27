// T-3933, T-4396: a load that ran out of memory opens again once with less. Two ways a load ends so
// (notes/T-3933.md): the tab dies (iOS kills it; Chrome's renderer exits with its out of memory code), and the next
// page load in the same tab finds the died-load flag (game/loading.ts); or the last solving worker fails a WASM grow
// the page sees, and the page saves and reloads itself into that same state. Either way the next page: when the load
// ran more than one solving worker, half of them (at least 1) become the tab's pool cap (`rememberPoolCap`, as a
// worker's own memory failure does); when it already ran one solving worker, intercity trips are turned off
// (`dropMarket`), or if they were already off, less detail (`coarseDemand`) is turned on, and a game in a slot opens
// again from it. Once: a retry that runs out too, or a load with nothing left to drop, gets the out of memory screen.
import type { GameChoice } from "../workers/clockClient";
import { poolSize, rememberPoolCap } from "../workers/demandClient";
import { lodMode } from "../workers/demandLod";
import { diedLoad, handleLoadFail, keepLoadForRetry, loadFailed, loadInHand, reloadForRetry, startRetry } from "./loading";
import { coarseDemand, dropMarket, marketOff, setDisplay } from "./ui";

type Load = NonNullable<ReturnType<typeof loadInHand>>;
type RetryPlan =
  | { kind: "cores"; fewer: number }
  | { kind: "intercity" }
  | { kind: "detail" };
const no = (why: string) => { console.warn(`a load ran out of memory; not loading it again: ${why}`); return null; }; // i18n-ignore

/** What a retry of `d` will lower (setting the pool cap for `cores`), or null (and why, in the console) when none
 * applies. The most workers the load ran counts, not the count it died with: a fresh page with half holds less than a
 * pool that a worker's failure cut to that size. Unknown (it died before the demand pool reported): this page's pool.
 * Settings (`intercity`, `detail`) are only changed in `retryDiedLoad` once the slot is known, never before a reload. */
function retryPlan(d: Load): RetryPlan | null {
  if (d.retry) return no("it was already the retry");
  const ran = d.workers ?? poolSize();
  if (ran > 1) {
    const fewer = Math.max(1, Math.floor(ran / 2));
    rememberPoolCap(fewer); // a Try again from the out of memory screen also loads with fewer
    // `?demandWorkers` fixes the pool whatever the cap, so the retry would end the same way
    if (poolSize() >= ran) return no("the address sets the workers");
    return { kind: "cores", fewer };
  }
  if (!marketOff()) return { kind: "intercity" };
  if (!coarseDemand() && lodMode(true) === "coarse") return { kind: "detail" };
  return no("it ran 1 solving worker, with intercity trips off and less detail on");
}

/** The game to open again after a died load, or null for the out of memory screen. Call once, at startup, after
 * `lastLoadDied()` said the last load died. */
export function retryDiedLoad(): GameChoice | null {
  const d = diedLoad();
  if (!d) return no("an older flag");
  const plan = retryPlan(d);
  if (!plan) return null;
  if (!d.slot) return no("its game was not in a slot yet");
  if (plan.kind === "cores") {
    console.warn(`a load ran out of memory with up to ${d.workers ?? "?"} solving workers; loading it again with ${plan.fewer}`); // i18n-ignore
  } else if (plan.kind === "intercity") {
    dropMarket();
    console.warn("a load ran out of memory on 1 solving worker; loading it again without intercity trips"); // i18n-ignore
  } else {
    setDisplay("coarseDemand", true);
    console.warn("a load ran out of memory on 1 solving worker; loading it again in less detail"); // i18n-ignore
  }
  startRetry(plan.kind);
  return { slot: d.slot };
}

/** The page's own sight of a load running out of memory (`loadFailed`): save the game, then reload into the retry
 * (the slot is known once the save lands), else the failure screen after all, also when the save fails.
 * `save`: main.ts's save before a reload, which rejects if the game could not be saved. */
export function retryLoadsInPage(save: () => Promise<void>) {
  let reloading = false;
  handleLoadFail((why, memory) => {
    if (reloading) return true; // (another worker's failure on the way out)
    const d = memory ? loadInHand() : null;
    if (!d || !retryPlan(d)) return false;
    reloading = true;
    keepLoadForRetry(true);
    const giveUp = () => { reloading = false; keepLoadForRetry(false); loadFailed(why, true); };
    void save().then(() => {
      if (loadInHand()?.slot) return reloadForRetry();
      no("its game was not in a slot yet");
      giveUp();
    }, () => { no("its game could not be saved"); giveUp(); });
    return true;
  });
}
