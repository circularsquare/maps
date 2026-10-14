// T-349: the demand status ("Updating demand 2/4…", "Loading data", "Estimate, refining 2/4…") as a
// small tag on the map's top row, to the right of the demand buttons. It appears and goes in one step.
// Its hover says more.

import { demandStep, demandUpdating, demandView } from "../game/demand";
import { cityNames, demandIdle, loadFailure, loadProgress } from "../game/loading";
import { CACHE_VERSION } from "../workers/demandCache";
import { LoadBar, loadingCardShown } from "./LoadingScreen";

export function StatusTag() {
  const step = demandStep.value;
  // T-149, T-172: a city loading (at startup, after a pan, or construction reaching it) says how
  // far it has got
  const load = loadProgress.value;
  const busy = loadFailure.value ? "Demand could not load"
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? "Loading data"
    : !demandView.value ? (demandIdle.value ? null : "Loading data…") // T-318: nothing to load over dead space
    // T-232: figures before a solve's final round are shown only where nothing final is, and say so
    : !demandView.value.final ? `Estimate, refining ${Math.min(step.done + 1, step.of)}/${step.of}…`
    : demandUpdating.value ? `Updating demand ${Math.min(step.done + 1, step.of)}/${step.of}…` : null;
  if (!busy) return null;
  const tip = loadFailure.value ?? (load && !loadingCardShown() ? `Loading ${cityNames(load.cities)}, ${Math.floor(load.fraction * 100)}%. ${load.step}`
    : !demandView.value ? busy
    : !demandView.value.final ? "These figures are an early estimate. They settle when the forecast is finished."
    // T-328: the figures kept from the last time this game was open, while it is worked out again
    : demandView.value.version === CACHE_VERSION ? "These figures are from the last time this game was open. They are being worked out again."
    : "These figures are for the network before your last change. They change once the new forecast is finished.");
  return <div class="status-tag" role="status" data-tip={tip}>
    <span class="demand-busy">{busy}</span>
    {load && !loadFailure.value && <LoadBar fraction={load.fraction} step={load.step} thin />}
  </div>;
}
