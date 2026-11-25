// T-349: the demand status ("Trips here being recomputed…", "Loading data") as a
// small tag on the map's top row, to the right of the demand buttons. It appears and goes in one step.
// Its hover says more. T-2693: a thin bar in every case (see `share` below); no ring since T-2728.

import { computed, effect, signal } from "@preact/signals";
import { demandBehindFrom, demandHeld, demandScope, demandStep, demandSolving, demandWork, editHeld, solveRestarted, workShare, demandUpdating, demandView, focusZones, intercityDemand, refiningZone, viewRecomputing } from "../game/demand";
import { linesPending } from "../game/lineFigures";
import { count } from "../game/format";
import { clock, DEMAND_CAP } from "../game/clock";
import { activeCity, cityById, viewedZone, type CityId } from "../game/cities";
import { barPlace, nextLeft, STAGE_TEXT, type BarPlace } from "../game/loadBar";
import { editLeftOf, editPrior } from "../game/editLeft";
import { loadFailText } from "../game/loadText";
import { cityNames,demandIdle, demandStopped, loadFailure, loadLines, loadProgress, memoryResolve, oldestDownload, solveLeft, tripsOff, zoneFailures, zoneLoads, zoneSteps } from "../game/loading";
import { selection } from "../game/ui";
import { tripView } from "../game/trip";
import { LoadBar, loadingCardShown } from "./LoadingScreen";
import { bubblesWaiting, farWaiting } from "../map/commuterView";
import { groundNet, shadeDownloads } from "../game/terrain";
import { retryGround } from "../workers/clockClient";
import { basemapSlow } from "../map/basemap";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";

/** T-2409: whole game days since the oldest edit no final covers, while figures in play still miss it
 * (a line with no riders or old ones, or the zone in view recomputing); 0 otherwise. A computed, so
 * the tag renders when the count changes, not each game minute. */
const behindDays = computed(() => {
  const from = demandBehindFrom.value;
  if (from === null || !(linesPending.value.size || viewRecomputing.value)) return 0;
  return Math.max(0, Math.floor((clock.minute.value - from) / 1440));
});

/** T-3601: an edit no final covers yet is being solved (its local result may already show) */
const editUnsettled = computed(() => demandUpdating.value && demandBehindFrom.value !== null);

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
  if (sel?.kind === "city" && !demandIdle.value && !tripsOff.value && (zoneLoads.value.has(sel.city) || !intercityDemand.value)) return sel.city;
  return null;
});

/** T-2523: what that wait is on, in the words of the load's stages: the zone's packs, its own solve during a
 * load (the focus pool), an edit's solve or a whole solve */
function waitText(zone: CityId): string {
  const name = tPlace(zone, cityById(zone).name);
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
  // T-3599: an edit's solve: the time left of it
  if (waitText(zone) === t("statusTag.calculatingAfterChange")) return afterChangeLeft();
  const z = zoneSteps.value.get(zone);
  const left = z ?? (focusZones.value.get(zone) === false ? undefined : solveLeft.value ?? undefined);
  return loadLines(left ? { leftMs: left.leftMs, at: left.at } : undefined).join("\n");
}

/** T-3599: the time left of an edit's solve as the hover shows it, counted down between estimates and taking
 * a new one in as `nextLeft` does (so it never climbs by degrees); null until the solve has one */
const editShown: Record<"all" | "view", { leftMs: number; at: number } | null> = { all: null, view: null };
// (a solve that starts as a held edit's whole wait goes on keeps the count: `editHeld` clears just after)
effect(() => { if (!demandWork.value && !editHeld.peek()) editShown.all = editShown.view = null; });
/** (`view`: to the zone in view's figures, for the tag saying they are being recomputed) */
function afterChangeLeft(part: "all" | "view" = "all"): string {
  const ms = shownLeft(part);
  if (ms === null) return t("statusTag.leftUnknown");
  return ms < 60_000 ? t("statusTag.leftUnderMinute") : t("statusTag.leftMinutes", { n: Math.round(ms / 60_000) });
}
/** the time left as the hover shows it, ms (`afterChangeLeft`), or null while it can't be told */
function shownLeft(part: "all" | "view" = "all"): number | null {
  return leftNow(part)?.ms ?? null;
}
/** that, and whether the workers' units bound it (T-4264: `editLeftOf`; a held edit's part is a guess) */
function leftNow(part: "all" | "view" = "all"): { ms: number; bounded: boolean } | null {
  const now = performance.now();
  const w = demandWork.value;
  const units = w ? editLeftOf(w, now, editPrior, part) : null;
  // (a load's solve, or one after memory was freed, has the load's estimate)
  let est = w ? units?.ms ?? null : part === "all" && solveLeft.value ? Math.max(0, solveLeft.value.leftMs - (now - solveLeft.value.at)) : null;
  let bounded = w ? !!units?.bounded : part === "all" && !!solveLeft.value?.bounded;
  // a newer edit waits for this solve to end, then is solved as long again, roughly
  const held = editHeld.value;
  if (w && est !== null && held) { est += now - held.sentAt + est; bounded = false; }
  const was = editShown[part];
  if (est !== null) editShown[part] = { leftMs: nextLeft(was ? was.leftMs : null, est, was ? now - was.at : 0), at: now };
  const s = editShown[part];
  return s ? { ms: Math.max(0, s.leftMs - (now - s.at)), bounded: est !== null && bounded } : null;
}

/** T-4264: the tag's bar's place (`barPlace`) in the wait on show. Another kind of wait, or the tag gone over a second,
 * starts it from empty (a held edit's solve follows the one before it within that). */
let tagPlace: BarPlace = { best: 0 };
let placeKind: string | null = null, goneAt = 0;
function placeFor(kind: string): BarPlace {
  if (kind !== placeKind || (goneAt && performance.now() - goneAt > 1000)) tagPlace = { best: 0 };
  placeKind = kind;
  goneAt = 0;
  return tagPlace;
}
const tagGone = () => { goneAt ||= performance.now(); };
/** T-3868: since when trips have been calculated without a break (one edit's solve after another, a held edit
 * included). Units done measured the bar before: their times differ by ten times within a solve (a whole solve's
 * last 42 of 60 units took 5 s of 210), and each edit started it at 0. */
let solvingSince: number | null = null;
let solveEnd = 0;
effect(() => {
  clearTimeout(solveEnd);
  if (demandUpdating.value) solvingSince ??= performance.now();
  // (a held edit's solve is sent just after the final before it: the same wait goes on)
  else solveEnd = window.setTimeout(() => { if (!demandUpdating.peek()) solvingSince = null; }, 1000);
});
// T-3925: a newer edit stopped the solve and started a new one: its wait starts from 0
effect(() => {
  if (!solveRestarted.value) return;
  solvingSince = performance.now();
  tagPlace = { best: 0 };
  editShown.all = editShown.view = null;
});
/** a share of a wait, and whether it is bounded: past half on the bar only then (`barPlace`) */
type Work = { share: number; bounded: boolean };
/** the share of that wait passed, by the time left the hover shows: what has passed over that and what is left;
 * null while no time left is known */
function timedShare(): Work | null {
  const left = solvingSince === null ? null : leftNow();
  if (left === null || solvingSince === null) return null;
  const el = performance.now() - solvingSince;
  tickWork();
  return { share: el / Math.max(1, el + left.ms), bounded: left.bounded };
}
/** T-2613: terrain has been downloading for TERRAIN_SLOW_MS (grids and masks for the clock worker, or
 * the hillshade's tiles), so a quick download on a pan never flashes the tag */
const TERRAIN_SLOW_MS = 1500;
const terrainSlow = signal(false);
let slowTimer = 0;
/** T-3479: until this time a retry click keeps the tag on "Downloading terrain…" whatever the downloads do */
let retryHoldUntil = 0;
/** T-2693: the most downloads waiting at once since the last time none were, so the bar is the share finished */
let terrainPeak = 0;
effect(() => {
  const waiting = groundNet.value.pending + shadeDownloads.value;
  const on = waiting > 0;
  terrainPeak = on ? Math.max(terrainPeak, waiting) : 0;
  if (!on) { clearTimeout(slowTimer); slowTimer = 0; if (Date.now() >= retryHoldUntil) terrainSlow.value = false; }
  else if (!slowTimer && !terrainSlow.peek()) slowTimer = window.setTimeout(() => { slowTimer = 0; terrainSlow.value = true; }, TERRAIN_SLOW_MS);
});
/** T-3479: the failed tag clicked: the downloads go again now, and the tag says so at once, for a second at
 * least, so a try that fails as fast still shows it was made */
function retryTerrain() {
  retryGround();
  terrainSlow.value = true;
  retryHoldUntil = Date.now() + 1000;
  window.setTimeout(() => { if (!groundNet.peek().pending && !shadeDownloads.peek()) terrainSlow.value = false; }, 1000);
}
function terrainText(): string | null {
  return terrainSlow.value ? t("statusTag.terrainDownloading") : groundNet.value.failed ? t("statusTag.terrainFailed") : null;
}

/** T-2693: how long each kind of wait took last time, ms, for a bar where nothing measures the work; the
 * first time, DEFAULT_WAIT_MS. A guess: a bar from this stops at half until the wait ends (T-4264). */
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
  return (performance.now() - waitSince) / (lastWait.get(kind) ?? DEFAULT_WAIT_MS);
}
function stopWaitTimer() {
  if (waitTimer) { clearInterval(waitTimer); waitTimer = 0; }
}
/** T-2729: a quarter second's tick while the bar moves through the unit in hand of an edit's solve */
const workTick = signal(0);
let workTimer = 0;
function tickWork() {
  workTick.value;
  if (!workTimer) workTimer = window.setInterval(() => {
    if (demandUpdating.peek()) workTick.value++;
    else { clearInterval(workTimer); workTimer = 0; }
  }, 250);
}
/** the running solve's share, when one runs: bounded only by a time left from its workers' units (T-4264: the units
 * done and the rounds published are guesses, since units differ by ten times and on one worker every round lands in the
 * last seconds) */
function solveShare(): Work | null {
  // T-3868: by the time left, where it is known
  const timed = demandUpdating.value ? timedShare() : null;
  if (timed !== null) return timed;
  const { done, of } = demandStep.value;
  // T-2729: an edit's solve publishes only at each stage's end (its local solve while that is
  // awaited, then the whole one): the workers' units done instead, the unit in hand by its time
  const w = demandWork.value;
  if (demandUpdating.value && w) {
    const share = workShare(w, of === 2 && done === 0 ? 0 : 1, performance.now());
    if (share !== null) { tickWork(); return { share, bounded: false }; }
  }
  return demandUpdating.value && of > 0 ? { share: done / of, bounded: false } : null;
}

/** T-2756: a city-data download open this long means a slow connection (from a fast one each takes about a
 * second); terrain still downloading this long after its tag showed, the same */
const SLOW_DOWNLOAD_MS = 10_000;
/** the hover's extra line when the wait on show is slow for a reason the page can tell, else null */
function slowLine(kind: string): string | null {
  const now = performance.now(), dl = oldestDownload();
  if ((dl !== null && now - dl > SLOW_DOWNLOAD_MS) || (kind === "terrain" && terrainSlow.value && now - waitSince > SLOW_DOWNLOAD_MS)) return t("statusTag.slowConnection");
  // workers restarted (or cut) to give back memory solve the whole game again before they answer anything
  if (kind !== "terrain" && memoryResolve.value) return t("statusTag.freedMemory");
  return null;
}
/** a second's tick while the tag shows, so a slow wait's line comes when it is due */
const slowTick = signal(0);
let slowTimer2 = 0;
function tickSlow(on: boolean) {
  if (on && !slowTimer2) slowTimer2 = window.setInterval(() => slowTick.value++, 1000);
  else if (!on && slowTimer2) { clearInterval(slowTimer2); slowTimer2 = 0; }
}

/** a second's tick while a click waits, so its hover counts down */
const second = signal(0);
let ticking = 0;
effect(() => {
  const on = clickWait.value !== null;
  if (on && !ticking) ticking = window.setInterval(() => second.value++, 1000);
  else if (!on && ticking) { clearInterval(ticking); ticking = 0; }
});

/** T-3884: zoomed out, no zone in view, while a whole solve of several zones runs: the tag says so (T-1851 ties it to
 * the zone in view otherwise, and a solve nobody sees looked like lost work) */
const wholeSolveUnseen = computed(() => !viewedZone.value && demandUpdating.value && demandScope.value > 1 && !demandHeld.value);

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
  // T-2870: the basemap's tiles missing for 3 s
  const mapWait = basemapSlow.value ? t("statusTag.loadingMap") : null;
  const busy = stopped ? (loadingCardShown() ? null : terrain ?? mapWait)
    : loadFailure.value ? t("statusTag.demandFailed")
    : zoneFailure ? t("statusTag.placeFailed", { name: tPlace(activeCity.value, cityById(activeCity.value).name) })
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? t("statusTag.loadingData")
    : terrain ? terrain
    : mapWait ? mapWait
    : wholeSolveUnseen.value ? t("statusTag.recomputingCities", { count: count(demandSolving.value?.size ?? demandScope.value) })
    : !demandView.value ? (demandIdle.value || tripsOff.value ? null : t("statusTag.loadingDataDots")) // T-318: nothing to load over dead space; T-3851: nor with trips off
    // T-1493: the zone in view has figures but its bubbles are still on their way
    : bubblesWaiting.value ? t("statusTag.loadingDataDots")
    // T-3427: edits made while paused, held for the game to run ("Recalculate trips while paused" off)
    : demandHeld.value ? t("statusTag.heldPaused")
    // T-2523: a click whose answer waits says what the game is working on
    : clickWait.value ? waitText(clickWait.value)
    : behindDays.value >= 1 ? t("statusTag.behind", { count: behindDays.value })
    // T-1851: every case where the trips in view are rough, early, from before an edit or on their way
    // say one thing; once the zone in view is final, other zones still solving show nothing
    // T-1904: and whether that is the zone in view alone or a whole solve of many, which takes longer
    : viewRecomputing.value ? recomputing()
    // T-2622: the clock capped for an edit whose figures elsewhere are not in, the zone in view final;
    // T-3601: or an edit's whole solve still running after its local result showed (lines there hold the
    // longer trips of before the edit until its final)
    : clock.capped.value || editUnsettled.value ? t("statusTag.calculatingAfterChange") : null;
  if (!busy) { trackWait(null); stopWaitTimer(); tickSlow(false); tagGone(); return null; }
  // T-3427: edits held while paused: nothing is working, so a hover and no bar
  if (demandHeld.value && busy === t("statusTag.heldPaused")) {
    trackWait(null); stopWaitTimer(); tickSlow(false); tagGone();
    return <div class="status-tag" role="status" data-tip={t("statusTag.heldPausedTip")}><span class="demand-busy">{busy}</span></div>;
  }
  const onTerrain = busy === terrain, terrainFailed = onTerrain && !terrainSlow.value;
  const onMap = !onTerrain && busy === mapWait;
  const base = onMap ? "" : onTerrain ? (terrainFailed ? t("statusTag.terrainFailTip") : "")
    : (loadFailure.value ? loadFailText(loadFailure.value) : undefined) ?? (zoneFailure ? zoneFailure : load && !loadingCardShown() ? t("statusTag.loadingTip", { cities: cityNames(load.cities), pct: Math.floor(load.fraction * 100), step: load.step })
    : wholeSolveUnseen.value ? (demandWork.value && shownLeft() !== null ? afterChangeLeft() : "")
    // T-4264: a wait behind a running solve has that solve's time left
    : !demandView.value ? (demandUpdating.value && shownLeft() !== null ? afterChangeLeft() : busy)
    : bubblesWaiting.value ? (demandUpdating.value && shownLeft() !== null ? afterChangeLeft() : "")
    : clickWait.value ? waitTip(clickWait.value)
    : busy === t("statusTag.calculatingAfterChange") ? afterChangeLeft()
    // (a wait past a game day at the cap, 6 min, says how far behind instead: the same time left)
    : behindDays.value >= 1 && busy === t("statusTag.behind", { count: behindDays.value }) && demandUpdating.value ? afterChangeLeft()
    // T-3599: the zone in view's trips recomputed after an edit
    : viewRecomputing.value && busy === recomputing() && demandUpdating.value && demandBehindFrom.value !== null ? afterChangeLeft("view")
    // T-3876: a load's whole solve, timed by its workers' units as an edit's is
    : viewRecomputing.value && busy === recomputing() && demandUpdating.value && demandWork.value && shownLeft() !== null ? afterChangeLeft()
    : "");
  const tip = [base, !onTerrain && clock.capped.value ? t("statusTag.capped", { cap: DEMAND_CAP }) : ""].filter(Boolean).join("\n") || undefined;
  const failed = onTerrain ? terrainFailed : !!loadFailure.value || !!zoneFailure;
  // T-2693: a bar in every case: the load's own bar or a zone's (placed already), else the running solve's share, the
  // downloads finished, or the share of the last such wait that has passed; T-4264: past half only where bounded
  let kind = "solve", placedShare: number | null = null, work: Work | null = null;
  if (!failed) {
    const zone = (id: CityId) => zoneLoads.value.get(id) ?? null;
    if (onMap) kind = "map";
    else if (onTerrain) { kind = "terrain"; const left = groundNet.value.pending + shadeDownloads.value; work = terrainPeak > 1 ? { share: 1 - left / terrainPeak, bounded: true } : null; }
    else if (load) { kind = "load"; placedShare = load.fraction; }
    else if (wholeSolveUnseen.value) work = solveShare();
    else if (!demandView.value) { kind = "data"; placedShare = zone(activeCity.value); if (placedShare === null) work = solveShare(); }
    else if (bubblesWaiting.value) { kind = "bubbles"; placedShare = zone(activeCity.value); if (placedShare === null) work = solveShare(); }
    else if (clickWait.value) { kind = "click"; placedShare = zone(clickWait.value); if (placedShare === null) work = solveShare(); }
    else { work = solveShare(); if (!work) placedShare = zone(activeCity.value); }
  }
  trackWait(failed ? null : kind);
  let share = 0;
  if (failed) tagGone();
  else if (placedShare !== null) { share = placedShare; tagGone(); }
  else {
    const guess = work ? work.share : timeShare(kind);
    if (work) stopWaitTimer();
    share = barPlace(placeFor(kind), guess, work?.bounded ? work.share : null);
  }
  if (failed || placedShare !== null) stopWaitTimer();
  // T-2756: a slow wait says why, where the page can tell
  tickSlow(!failed);
  slowTick.value;
  const slow = failed ? null : slowLine(kind);
  const tipAll = [tip, slow].filter(Boolean).join("\n") || undefined;
  // T-3479: terrain that failed is tried again on a click
  if (terrainFailed) return <button type="button" class="status-tag" aria-live="polite" data-tip={tipAll} onClick={retryTerrain}>
    <span class="demand-busy">{busy}</span>
  </button>;
  return <div class="status-tag" role="status" data-tip={tipAll}>
    <span class="demand-busy">{busy}</span>
    {!failed && <LoadBar fraction={share} step={busy} thin quiet={kind !== "load"} />}
  </div>;
}
