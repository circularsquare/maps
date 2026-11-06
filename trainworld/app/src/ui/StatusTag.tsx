// T-349: the demand status ("Trips here being recomputed…", "Loading data") as a
// small tag on the map's top row, to the right of the demand buttons. It appears and goes in one step.
// Its hover says more.

import { computed, effect, signal } from "@preact/signals";
import { demandBehindFrom, demandScope, demandSolving, demandUpdating, demandView, focusZones, intercityDemand, refiningZone, viewRecomputing } from "../game/demand";
import { linesPending } from "../game/lineFigures";
import { count } from "../game/format";
import { clock, DEMAND_CAP } from "../game/clock";
import { activeCity, cityById, viewedZone, type CityId } from "../game/cities";
import { STAGE_TEXT } from "../game/loadBar";
import { loadFailText } from "../game/loadText";
import { cityNames,demandIdle, demandStopped, loadFailure, loadLines, loadProgress, solveLeft, zoneFailures, zoneLoads, zoneSteps } from "../game/loading";
import { selection } from "../game/ui";
import { tripView } from "../game/trip";
import { LoadBar, loadingCardShown } from "./LoadingScreen";
import { BunnyTrain } from "./Spinner";
import { bubblesWaiting, farWaiting } from "../map/commuterView";
import { groundNet, shadeDownloads } from "../game/terrain";

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

/** T-2523: the zone a clicked thing's panel is still waiting on (a bubble's far end, a trip, a city's trips), or null */
const clickWait = computed<CityId | null>(() => {
  const sel = selection.value;
  if (farWaiting.value) return activeCity.value;
  if (sel?.kind === "trip" && sel.to) {
    const v = tripView.value;
    return v?.trip || v?.market || v?.failed ? null : activeCity.value;
  }
  if (sel?.kind === "city" && !demandIdle.value && (zoneLoads.value.has(sel.city) || !intercityDemand.value)) return sel.city;
  return null;
});

/** T-2523: what that wait is on, in the words of the load's stages: the zone's packs, its own solve during a
 * load (the focus pool), an edit's solve or a whole solve */
function waitText(zone: CityId): string {
  const name = cityById(zone).name;
  if (zoneSteps.value.get(zone)?.step === STAGE_TEXT.data) return `Loading ${name}…`;
  if (focusZones.value.get(zone) === false) return `Calculating trips in ${name}…`;
  if (!demandUpdating.value) return "Loading data…";
  if (demandBehindFrom.value !== null) return "Calculating trips after your change…";
  const scope = demandScope.value;
  const n = scope && !refiningZone.value ? demandSolving.value?.size ?? scope : 0;
  return n > 1 ? `Calculating trips in ${count(n)} cities…` : `Calculating trips in ${name}…`;
}

/** its hover: the time left of a whole solve (or the zone's own load), or that nothing has come for a while */
function waitTip(zone: CityId): string {
  second.value;
  const z = zoneSteps.value.get(zone);
  const left = z ?? (focusZones.value.get(zone) === false ? undefined : solveLeft.value ?? undefined);
  return loadLines(left ? { leftMs: left.leftMs, at: left.at } : undefined).join("\n");
}
/** T-2613: terrain has been downloading for TERRAIN_SLOW_MS (grids and masks for the clock worker, or
 * the hillshade's tiles), so a quick download on a pan never flashes the tag */
const TERRAIN_SLOW_MS = 1500;
const terrainSlow = signal(false);
let slowTimer = 0;
effect(() => {
  const on = groundNet.value.pending > 0 || shadeDownloads.value > 0;
  if (!on) { clearTimeout(slowTimer); slowTimer = 0; terrainSlow.value = false; }
  else if (!slowTimer && !terrainSlow.peek()) slowTimer = window.setTimeout(() => { slowTimer = 0; terrainSlow.value = true; }, TERRAIN_SLOW_MS);
});
const TERRAIN_FAIL_TIP = "Check your internet connection. It is tried again within 30 s.";
function terrainText(): string | null {
  return terrainSlow.value ? "Downloading terrain…" : groundNet.value.failed ? "Terrain could not load" : null;
}

/** a second's tick while a click waits, so its hover counts down */
const second = signal(0);
let ticking = 0;
effect(() => {
  const on = clickWait.value !== null;
  if (on && !ticking) ticking = window.setInterval(() => second.value++, 1000);
  else if (!on && ticking) { clearInterval(ticking); ticking = 0; }
});

export function StatusTag() {
  // T-149, T-172: a city loading (at startup, after a pan, or construction reaching it) says how
  // far it has got
  const load = loadProgress.value;
  // T-484, T-492: the zone in view could not load (the others carry on); its text says whether it is tried again
  const zoneFailure = viewedZone.value ? zoneFailures.value.get(activeCity.value) : undefined;
  // T-2613: terrain on its way or failed, said after a load's own news
  const terrain = terrainText();
  // T-757: after startup the alert under the demand buttons says it instead (terrain still shows)
  const stopped = demandStopped.value;
  const busy = stopped ? (loadingCardShown() ? null : terrain)
    : loadFailure.value ? "Demand could not load"
    : zoneFailure ? `${cityById(activeCity.value).name} could not load`
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? "Loading data"
    : terrain ? terrain
    : !demandView.value ? (demandIdle.value ? null : "Loading data…") // T-318: nothing to load over dead space
    // T-1493: the zone in view has figures but its bubbles are still on their way
    : bubblesWaiting.value ? "Loading data…"
    // T-2523: a click whose answer waits says what the game is working on
    : clickWait.value ? waitText(clickWait.value)
    : behindDays.value >= 1 ? (behindDays.value === 1 ? "Calculation 1 day behind" : `Calculation ${count(behindDays.value)} days behind`)
    // T-1851: every case where the trips in view are rough, early, from before an edit or on their way
    // say one thing; once the zone in view is final, other zones still solving show nothing
    // T-1904: and whether that is the zone in view alone or a whole solve of many, which takes longer
    : viewRecomputing.value ? recomputing() : null;
  if (!busy) return null;
  const onTerrain = busy === terrain, terrainFailed = onTerrain && !terrainSlow.value;
  const base = onTerrain ? (terrainFailed ? TERRAIN_FAIL_TIP : "")
    : (loadFailure.value ? loadFailText(loadFailure.value) : undefined) ?? (zoneFailure ? zoneFailure : load && !loadingCardShown() ? `Loading ${cityNames(load.cities)}, ${Math.floor(load.fraction * 100)}%. ${load.step}`
    : !demandView.value ? busy
    : bubblesWaiting.value ? `Loading the bubbles for ${cityById(activeCity.value).name}`
    : clickWait.value ? waitTip(clickWait.value)
    : "");
  const tip = [base, !onTerrain && clock.capped.value ? `Time capped at ${DEMAND_CAP}x while updating demand` : ""].filter(Boolean).join("\n") || undefined;
  const failed = onTerrain ? terrainFailed : !!loadFailure.value || !!zoneFailure;
  return <div class="status-tag" role="status" data-tip={tip}>
    {!failed && <BunnyTrain small />}
    <span class="demand-busy">{busy}</span>
    {load && !onTerrain && !loadFailure.value && !zoneFailure && <LoadBar fraction={load.fraction} step={load.step} thin />}
  </div>;
}
