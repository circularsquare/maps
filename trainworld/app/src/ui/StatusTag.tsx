// T-349: the demand status ("Trips here being recomputed…", "Loading data") as a
// small tag on the map's top row, to the right of the demand buttons. It appears and goes in one step.
// Its hover says more.

import { computed } from "@preact/signals";
import { demandBehindFrom, demandScope, demandSolving, demandView, focusZones, refiningZone, viewRecomputing } from "../game/demand";
import { linesPending } from "../game/lineFigures";
import { count } from "../game/format";
import { clock, DEMAND_CAP } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { loadFailText } from "../game/loadText";
import { cityNames,demandIdle, demandStopped, loadFailure, loadProgress, zoneFailures } from "../game/loading";
import { LoadBar, loadingCardShown } from "./LoadingScreen";
import { BunnyTrain } from "./Spinner";
import { bubblesWaiting } from "../map/commuterView";

/** T-2409: whole game days since the oldest edit no final covers, while figures in play still miss it
 * (a line with no riders or old ones, or the zone in view recomputing); 0 otherwise. A computed, so
 * the tag renders when the count changes, not each game minute. */
const behindDays = computed(() => {
  const from = demandBehindFrom.value;
  if (from === null || !(linesPending.value.size || viewRecomputing.value)) return 0;
  return Math.max(0, Math.floor((clock.minute.value - from) / 1440));
});

function recomputing(): string {
  // T-2045: the zone in view being solved alone ahead of a load's whole solve
  if (focusZones.value.has(activeCity.value)) return "Trips here being recomputed…";
  const scope = demandScope.value;
  const n = scope && !refiningZone.value ? demandSolving.value?.size ?? scope : 0;
  return n > 1 ? `Trips in ${count(n)} cities being recomputed…` : "Trips here being recomputed…";
}

export function StatusTag() {
  // T-149, T-172: a city loading (at startup, after a pan, or construction reaching it) says how
  // far it has got
  const load = loadProgress.value;
  // T-484, T-492: the zone in view could not load (the others carry on); its text says whether it is tried again
  const zoneFailure = viewedZone.value ? zoneFailures.value.get(activeCity.value) : undefined;
  // T-757: after startup the alert under the demand buttons says it instead
  if (demandStopped.value) return null;
  const busy = loadFailure.value ? "Demand could not load"
    : zoneFailure ? `${cityById(activeCity.value).name} could not load`
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? "Loading data"
    : !demandView.value ? (demandIdle.value ? null : "Loading data…") // T-318: nothing to load over dead space
    // T-1493: the zone in view has figures but its bubbles are still on their way
    : bubblesWaiting.value ? "Loading data…"
    : behindDays.value >= 1 ? (behindDays.value === 1 ? "Calculation 1 day behind" : `Calculation ${count(behindDays.value)} days behind`)
    // T-1851: every case where the trips in view are rough, early, from before an edit or on their way
    // say one thing; once the zone in view is final, other zones still solving show nothing
    // T-1904: and whether that is the zone in view alone or a whole solve of many, which takes longer
    : viewRecomputing.value ? recomputing() : null;
  if (!busy) return null;
  const base = (loadFailure.value ? loadFailText(loadFailure.value) : undefined) ?? (zoneFailure ? zoneFailure : load && !loadingCardShown() ? `Loading ${cityNames(load.cities)}, ${Math.floor(load.fraction * 100)}%. ${load.step}`
    : !demandView.value ? busy
    : bubblesWaiting.value ? `Loading the bubbles for ${cityById(activeCity.value).name}`
    : "");
  const tip = [base, clock.capped.value ? `Time capped at ${DEMAND_CAP}x while updating demand` : ""].filter(Boolean).join("\n") || undefined;
  return <div class="status-tag" role="status" data-tip={tip}>
    {!loadFailure.value && !zoneFailure && <BunnyTrain small />}
    <span class="demand-busy">{busy}</span>
    {load && !loadFailure.value && !zoneFailure && <LoadBar fraction={load.fraction} step={load.step} thin />}
  </div>;
}
