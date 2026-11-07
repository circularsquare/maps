// T-349: the demand status ("Trips here being recomputed…", "Loading data") as a
// small tag on the map's top row, to the right of the demand buttons. It appears and goes in one step.
// Its hover says more. T-2693: a pill ring and a thin bar in every case (see `share` below).

import { computed, effect, signal } from "@preact/signals";
import { demandBehindFrom, demandScope, demandStep, demandSolving, demandUpdating, demandView, focusZones, intercityDemand, refiningZone, viewRecomputing } from "../game/demand";
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
import { PillRing } from "./Spinner";
import { bubblesWaiting, farWaiting } from "../map/commuterView";
import { groundNet, shadeDownloads } from "../game/terrain";
import { t } from "../i18n";

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
  if (focusZones.value.has(activeCity.value)) return t("statusTag.recomputingHere");
  const scope = demandScope.value;
  const n = scope && !refiningZone.value ? demandSolving.value?.size ?? scope : 0;
  return n > 1 ? t("statusTag.recomputingCities", { count: count(n) }) : t("statusTag.recomputingHere");
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
  if (zoneSteps.value.get(zone)?.step === STAGE_TEXT.data) return t("statusTag.loadingPlace", { name });
  if (focusZones.value.get(zone) === false) return t("statusTag.calculatingPlace", { name });
  if (!demandUpdating.value) return t("statusTag.loadingDataDots");
  if (demandBehindFrom.value !== null) return t("statusTag.calculatingAfterChange");
  const scope = demandScope.value;
  const n = scope && !refiningZone.value ? demandSolving.value?.size ?? scope : 0;
  return n > 1 ? t("statusTag.calculatingCities", { count: count(n) }) : t("statusTag.calculatingPlace", { name });
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
/** T-2693: the most downloads waiting at once since the last time none were, so the bar is the share finished */
let terrainPeak = 0;
effect(() => {
  const waiting = groundNet.value.pending + shadeDownloads.value;
  const on = waiting > 0;
  terrainPeak = on ? Math.max(terrainPeak, waiting) : 0;
  if (!on) { clearTimeout(slowTimer); slowTimer = 0; terrainSlow.value = false; }
  else if (!slowTimer && !terrainSlow.peek()) slowTimer = window.setTimeout(() => { slowTimer = 0; terrainSlow.value = true; }, TERRAIN_SLOW_MS);
});
function terrainText(): string | null {
  return terrainSlow.value ? t("statusTag.terrainDownloading") : groundNet.value.failed ? t("statusTag.terrainFailed") : null;
}

/** T-2693: how long each kind of wait took last time, ms, for a bar where nothing measures the work; the
 * first time, DEFAULT_WAIT_MS. A bar from this stops at 95% until the wait ends. */
const lastWait = new Map<string, number>();
const DEFAULT_WAIT_MS = 3000;
let waitKind: string | null = null, waitSince = 0;
const waitTick = signal(0);
let waitTimer = 0;
function trackWait(kind: string | null) {
  if (kind === waitKind) return;
  const now = performance.now();
  if (waitKind && now - waitSince > 400) lastWait.set(waitKind, Math.min(120_000, now - waitSince));
  waitKind = kind;
  waitSince = now;
}
/** the share of the last such wait that has passed; ticks four times a second while it is on screen */
function timeShare(kind: string): number {
  waitTick.value;
  if (!waitTimer) waitTimer = window.setInterval(() => waitTick.value++, 250);
  return Math.min(0.95, (performance.now() - waitSince) / (lastWait.get(kind) ?? DEFAULT_WAIT_MS));
}
function stopWaitTimer() {
  if (waitTimer) { clearInterval(waitTimer); waitTimer = 0; }
}
/** results published so far of the running solve's, when one runs */
function roundShare(): number | null {
  const { done, of } = demandStep.value;
  return demandUpdating.value && of > 0 ? Math.min(0.95, done / of) : null;
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
    : loadFailure.value ? t("statusTag.demandFailed")
    : zoneFailure ? t("statusTag.placeFailed", { name: cityById(activeCity.value).name })
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? t("statusTag.loadingData")
    : terrain ? terrain
    : !demandView.value ? (demandIdle.value ? null : t("statusTag.loadingDataDots")) // T-318: nothing to load over dead space
    // T-1493: the zone in view has figures but its bubbles are still on their way
    : bubblesWaiting.value ? t("statusTag.loadingDataDots")
    // T-2523: a click whose answer waits says what the game is working on
    : clickWait.value ? waitText(clickWait.value)
    : behindDays.value >= 1 ? t("statusTag.behind", { count: behindDays.value })
    // T-1851: every case where the trips in view are rough, early, from before an edit or on their way
    // say one thing; once the zone in view is final, other zones still solving show nothing
    // T-1904: and whether that is the zone in view alone or a whole solve of many, which takes longer
    : viewRecomputing.value ? recomputing()
    // T-2622: the clock capped for an edit whose figures elsewhere are not in, the zone in view final
    : clock.capped.value ? t("statusTag.calculatingAfterChange") : null;
  if (!busy) { trackWait(null); stopWaitTimer(); return null; }
  const onTerrain = busy === terrain, terrainFailed = onTerrain && !terrainSlow.value;
  const base = onTerrain ? (terrainFailed ? t("statusTag.terrainFailTip") : "")
    : (loadFailure.value ? loadFailText(loadFailure.value) : undefined) ?? (zoneFailure ? zoneFailure : load && !loadingCardShown() ? t("statusTag.loadingTip", { cities: cityNames(load.cities), pct: Math.floor(load.fraction * 100), step: load.step })
    : !demandView.value ? busy
    : bubblesWaiting.value ? t("statusTag.loadingBubbles", { name: cityById(activeCity.value).name })
    : clickWait.value ? waitTip(clickWait.value)
    : "");
  const tip = [base, !onTerrain && clock.capped.value ? t("statusTag.capped", { cap: DEMAND_CAP }) : ""].filter(Boolean).join("\n") || undefined;
  const failed = onTerrain ? terrainFailed : !!loadFailure.value || !!zoneFailure;
  // T-2693: a bar in every case: the load's own share, a zone's, the solve's rounds, the downloads finished,
  // else the share of the last such wait that has passed
  let kind = "solve", share: number | null = null;
  if (!failed) {
    if (onTerrain) { kind = "terrain"; const left = groundNet.value.pending + shadeDownloads.value; share = terrainPeak > 1 ? 1 - left / terrainPeak : null; }
    else if (load) { kind = "load"; share = load.fraction; }
    else if (!demandView.value) kind = "data";
    else if (bubblesWaiting.value) { kind = "bubbles"; share = zoneLoads.value.get(activeCity.value) ?? null; }
    else if (clickWait.value) { kind = "click"; share = zoneLoads.value.get(clickWait.value) ?? roundShare(); }
    else { share = roundShare() ?? zoneLoads.value.get(activeCity.value) ?? null; }
  }
  trackWait(failed ? null : kind);
  const measured = share !== null;
  if (!failed && !measured) share = timeShare(kind); else stopWaitTimer();
  return <div class="status-tag" role="status" data-tip={tip}>
    {!failed && <PillRing />}
    <span class="demand-busy">{busy}</span>
    {!failed && <LoadBar fraction={share!} step={busy} thin quiet={kind !== "load"} />}
  </div>;
}
